"""One-time migration of legacy entity voice NPZ files into Catalog BLOBs.

Online code must never read these files.  This command is intentionally a
standalone, default-dry-run bridge for installations that created entity voice
samples before ``voice_samples.voice_embedding`` became the runtime contract.
Successful migration keeps the legacy file as an operator backup while
clearing its Catalog pointer so it cannot participate in normal execution.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.vector_store.milvus.milvus_schema import EMBEDDING_DIMS


def _validated_vector(value: Any, *, source: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = int(EMBEDDING_DIMS["speaker"])
    if vector.size != expected:
        raise ValueError(f"{source} 维度为 {vector.size}，预期 {expected}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{source} 包含 NaN 或 Inf")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{source} 是零向量")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _load_legacy_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"旧声音向量不存在: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if payload.files != ["embedding"]:
            raise ValueError(
                f"旧声音向量 {path} 必须且只能包含 embedding，实际为 {payload.files}"
            )
        return _validated_vector(payload["embedding"], source=str(path))


def _validated_blob(value: bytes | None, *, sample_id: str) -> np.ndarray:
    if not isinstance(value, bytes):
        raise ValueError(f"声纹样本 {sample_id} 缺少内联向量")
    return _validated_vector(
        np.frombuffer(value, dtype=np.float32),
        source=f"声纹样本 {sample_id}",
    )


def _selected_rows(
    catalog: Catalog,
    *,
    entity_ids: set[str] | None,
    sample_ids: set[str] | None,
) -> tuple[list[dict], list[dict]]:
    clauses: list[str] = []
    values: list[str] = []
    if entity_ids:
        clauses.append(f"entity_id IN ({','.join('?' for _ in entity_ids)})")
        values.extend(sorted(entity_ids))
    if sample_ids:
        clauses.append(f"id IN ({','.join('?' for _ in sample_ids)})")
        values.extend(sorted(sample_ids))
    query = "SELECT * FROM voice_samples"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY entity_id,created_at,id"
    with catalog.connect() as connection:
        known_entities = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT entity_id FROM voice_samples"
            ).fetchall()
        }
        known_samples = {
            str(row[0])
            for row in connection.execute("SELECT id FROM voice_samples").fetchall()
        }
        rows = [dict(row) for row in connection.execute(query, values).fetchall()]

    errors: list[dict] = []
    for entity_id in sorted((entity_ids or set()) - known_entities):
        errors.append({"entity_id": entity_id, "error": "未找到声音样本"})
    for sample_id in sorted((sample_ids or set()) - known_samples):
        errors.append({"sample_id": sample_id, "error": "未找到声音样本"})
    return rows, errors


def _write_verified_blob(
    catalog: Catalog,
    row: dict,
    vector: np.ndarray,
) -> None:
    sample_id = str(row["id"])
    legacy_path = str(row.get("embedding_path") or "")
    blob = vector.tobytes()
    with catalog.connect() as connection:
        current = connection.execute(
            "SELECT voice_embedding,embedding_path FROM voice_samples WHERE id=?",
            (sample_id,),
        ).fetchone()
        if current is None:
            raise RuntimeError(f"声纹样本 {sample_id} 在迁移期间被删除")
        if current["voice_embedding"] is not None or current["embedding_path"] != legacy_path:
            raise RuntimeError(f"声纹样本 {sample_id} 在迁移期间发生变化")
        cursor = connection.execute(
            """UPDATE voice_samples SET voice_embedding=?,embedding_path=''
               WHERE id=? AND voice_embedding IS NULL AND embedding_path=?""",
            (blob, sample_id, legacy_path),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"声纹样本 {sample_id} 并发写入保护失败")
        stored = connection.execute(
            "SELECT voice_embedding,embedding_path FROM voice_samples WHERE id=?",
            (sample_id,),
        ).fetchone()
        if stored is None or stored["embedding_path"] != "":
            raise RuntimeError(f"声纹样本 {sample_id} 写后路径校验失败")
        actual = _validated_blob(stored["voice_embedding"], sample_id=sample_id)
        if not np.allclose(actual, vector, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"声纹样本 {sample_id} 写后向量校验失败")


def _clear_verified_legacy_path(catalog: Catalog, row: dict, vector: np.ndarray) -> None:
    sample_id = str(row["id"])
    legacy_path = str(row.get("embedding_path") or "")
    with catalog.connect() as connection:
        current = connection.execute(
            "SELECT voice_embedding,embedding_path FROM voice_samples WHERE id=?",
            (sample_id,),
        ).fetchone()
        if current is None:
            raise RuntimeError(f"声纹样本 {sample_id} 在迁移期间被删除")
        actual = _validated_blob(current["voice_embedding"], sample_id=sample_id)
        if not np.allclose(actual, vector, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"声纹样本 {sample_id} 在迁移期间发生变化")
        cursor = connection.execute(
            """UPDATE voice_samples SET embedding_path=''
               WHERE id=? AND embedding_path=? AND voice_embedding=?""",
            (sample_id, legacy_path, current["voice_embedding"]),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"声纹样本 {sample_id} 清理旧路径失败")


def migrate(
    catalog: Catalog,
    *,
    apply: bool = False,
    entity_ids: set[str] | None = None,
    sample_ids: set[str] | None = None,
) -> dict:
    rows, selection_errors = _selected_rows(
        catalog,
        entity_ids=entity_ids,
        sample_ids=sample_ids,
    )
    report: dict[str, list[dict]] = {
        "migrated": [],
        "skipped": [],
        "errors": selection_errors,
    }
    for row in rows:
        sample_id = str(row["id"])
        legacy_path = str(row.get("embedding_path") or "")
        try:
            if row.get("voice_embedding") is not None:
                vector = _validated_blob(row["voice_embedding"], sample_id=sample_id)
                if not legacy_path:
                    report["skipped"].append(
                        {"sample_id": sample_id, "status": "already_migrated"}
                    )
                elif apply:
                    _clear_verified_legacy_path(catalog, row, vector)
                    report["migrated"].append(
                        {"sample_id": sample_id, "status": "cleared_legacy_path"}
                    )
                else:
                    report["migrated"].append(
                        {"sample_id": sample_id, "status": "would_clear_legacy_path"}
                    )
                continue

            if not legacy_path:
                raise ValueError(f"声纹样本 {sample_id} 既无内联向量也无旧路径")
            path = Path(legacy_path)
            vector = _load_legacy_vector(path)
            if apply:
                _write_verified_blob(catalog, row, vector)
            report["migrated"].append(
                {
                    "sample_id": sample_id,
                    "entity_id": str(row["entity_id"]),
                    "legacy_path": legacy_path,
                    "status": "migrated" if apply else "would_migrate",
                }
            )
        except Exception as exc:
            logging.exception("Voice sample migration failed for %s", sample_id)
            report["errors"].append({"sample_id": sample_id, "error": str(exc)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write verified BLOBs. Without this flag the command is read-only.",
    )
    parser.add_argument("--entity-id", action="append", dest="entity_ids")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = migrate(
        Catalog(get_settings().db_path),
        apply=bool(args.apply),
        entity_ids=set(args.entity_ids) if args.entity_ids else None,
        sample_ids=set(args.sample_ids) if args.sample_ids else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
