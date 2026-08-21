#!/usr/bin/env python3
"""Blue-green copy of Speaker rows into an immutable DiskANN collection.

The source collection is read-only and is never released, reindexed, renamed,
or deleted.  The target is accepted only after schema, index, row counts,
per-version counts, and an order-independent content fingerprint all match.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pymilvus import Collection, connections, utility  # noqa: E402
from pymilvus.exceptions import IndexNotExistException  # noqa: E402

from app.core.settings import get_settings  # noqa: E402
from app.vector_store.milvus.milvus_client import (  # noqa: E402
    get_collection_index_config,
)
from app.vector_store.milvus.milvus_schema import (  # noqa: E402
    EMBEDDING_DIMS,
    create_speaker_schema,
)


DEFAULT_SOURCE = "speaker_embeddings"
COPY_FIELDS = (
    "pk",
    "video_id",
    "asset_version",
    "model_version",
    "utterance_idx",
    "start_ms",
    "end_ms",
    "asr_chunk_idx",
    "track_id",
    "embedding",
)
REQUIRED_TARGET_FIELDS = frozenset(COPY_FIELDS)
EXPECTED_INDEX_TYPE = "DISKANN"
EXPECTED_METRIC_TYPE = "COSINE"
_BATCH_SIZE = 500
_DIGEST_MODULUS = 1 << 256


def _index_details(collection: Collection) -> dict[str, str | None]:
    try:
        index = collection.index()
    except IndexNotExistException:
        return {"index_type": None, "metric_type": None}
    params = getattr(index, "params", {}) or {}
    return {
        "index_type": params.get("index_type"),
        "metric_type": params.get("metric_type"),
    }


def _state(name: str, *, alias: str) -> dict[str, Any]:
    if not utility.has_collection(name, using=alias):
        return {"exists": False, "ready": False}
    collection = Collection(name, using=alias)
    fields = {field.name for field in collection.schema.fields}
    index = _index_details(collection)
    return {
        "exists": True,
        "ready": (
            REQUIRED_TARGET_FIELDS.issubset(fields)
            and index["index_type"] == EXPECTED_INDEX_TYPE
            and index["metric_type"] == EXPECTED_METRIC_TYPE
        ),
        "fields": sorted(fields),
        "row_count": int(collection.num_entities),
        **index,
    }


def _iter_pages(
    collection: Collection,
    *,
    fields: list[str],
    timeout: float,
) -> Iterator[list[dict[str, Any]]]:
    iterator = collection.query_iterator(
        batch_size=_BATCH_SIZE,
        expr="",
        output_fields=fields,
        timeout=timeout,
    )
    try:
        while True:
            try:
                page = iterator.next()
            except StopIteration:
                break
            if not page:
                break
            yield [dict(row) for row in page]
    finally:
        iterator.close()


def _row_digest(row: dict[str, Any]) -> int:
    missing = [field for field in COPY_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"Speaker row is missing fields: {missing}")
    embedding = np.asarray(row["embedding"], dtype="<f4")
    if embedding.shape != (EMBEDDING_DIMS["speaker"],):
        raise RuntimeError(f"invalid Speaker embedding shape: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise RuntimeError("Speaker embedding contains non-finite values")
    metadata = [row[field] for field in COPY_FIELDS if field != "embedding"]
    payload = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    digest.update(embedding.tobytes(order="C"))
    return int.from_bytes(digest.digest(), "big")


def _empty_summary() -> dict[str, Any]:
    return {
        "row_count": 0,
        "digest_xor": 0,
        "digest_sum": 0,
        "version_counts": defaultdict(int),
    }


def _add_rows(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        digest = _row_digest(row)
        summary["row_count"] += 1
        summary["digest_xor"] ^= digest
        summary["digest_sum"] = (summary["digest_sum"] + digest) % _DIGEST_MODULUS
        key = f'{row["video_id"]}\u0000{row["asset_version"]}'
        summary["version_counts"][key] += 1


def _serializable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": int(summary["row_count"]),
        "digest_xor": f'{int(summary["digest_xor"]):064x}',
        "digest_sum": f'{int(summary["digest_sum"]):064x}',
        "version_counts": dict(sorted(summary["version_counts"].items())),
    }


def _scan_summary(collection: Collection, *, timeout: float) -> dict[str, Any]:
    summary = _empty_summary()
    for page in _iter_pages(collection, fields=list(COPY_FIELDS), timeout=timeout):
        _add_rows(summary, page)
    return _serializable_summary(summary)


def copy_collection(
    *,
    source_name: str,
    target_name: str,
    execute: bool,
) -> dict[str, Any]:
    settings = get_settings()
    if source_name == target_name:
        raise ValueError("source and target Speaker collection names must differ")
    alias = "speaker_blue_green_copy"
    connections.connect(
        alias=alias,
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        timeout=settings.milvus_query_timeout_seconds,
    )
    try:
        operation_timeout = max(60.0, float(settings.milvus_query_timeout_seconds))
        source_state = _state(source_name, alias=alias)
        target_state = _state(target_name, alias=alias)
        if not source_state["exists"]:
            raise RuntimeError(f"source Speaker collection does not exist: {source_name}")
        missing_source = sorted(set(COPY_FIELDS) - set(source_state["fields"]))
        if missing_source:
            raise RuntimeError(
                f"source Speaker collection is missing copy fields: {missing_source}"
            )
        if target_state["exists"]:
            raise RuntimeError(
                f"target Speaker collection already exists: {target_name}; "
                "inspect or choose a new immutable target name"
            )
        plan = {
            "status": "dry_run" if not execute else "copying",
            "source": source_name,
            "target": target_name,
            "source_state": source_state,
            "target_state": target_state,
        }
        if not execute:
            return plan

        source = Collection(source_name, using=alias)
        target = Collection(
            name=target_name,
            schema=create_speaker_schema(),
            consistency_level="Strong",
            using=alias,
        )
        source_summary = _empty_summary()
        for page in _iter_pages(
            source,
            fields=list(COPY_FIELDS),
            timeout=float(settings.milvus_query_timeout_seconds),
        ):
            _add_rows(source_summary, page)
            target.insert(page, timeout=operation_timeout)
        target.flush(timeout=operation_timeout)
        target.create_index(
            field_name="embedding",
            index_params=get_collection_index_config("speaker_embeddings"),
            timeout=operation_timeout,
        )
        target.load(timeout=operation_timeout)

        source_summary_out = _serializable_summary(source_summary)
        target_summary = _scan_summary(
            target,
            timeout=float(settings.milvus_query_timeout_seconds),
        )
        verified_state = _state(target_name, alias=alias)
        if not verified_state["ready"]:
            raise RuntimeError(
                f"target Speaker schema/index verification failed: {verified_state}"
            )
        if int(source_state["row_count"]) != source_summary_out["row_count"]:
            raise RuntimeError(
                "source Speaker count changed during copy: "
                f'{source_state["row_count"]} != {source_summary_out["row_count"]}'
            )
        if source_summary_out != target_summary:
            raise RuntimeError(
                "target Speaker content verification failed: "
                f"source={source_summary_out}, target={target_summary}"
            )
        if int(verified_state["row_count"]) != target_summary["row_count"]:
            raise RuntimeError(
                "target Speaker entity count disagrees with scanned rows: "
                f'{verified_state["row_count"]} != {target_summary["row_count"]}'
            )
        return {
            **plan,
            "status": "copied_and_verified",
            "target_state": verified_state,
            "summary": target_summary,
        }
    finally:
        connections.disconnect(alias)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把旧 Speaker 行蓝绿复制到新的 DiskANN collection"
    )
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE)
    parser.add_argument("--target-collection")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-create-target", action="store_true")
    args = parser.parse_args()
    if args.execute != args.confirm_create_target:
        parser.error("执行复制必须同时传入 --execute --confirm-create-target")
    settings = get_settings()
    target = args.target_collection or settings.milvus_speaker_collection
    report = copy_collection(
        source_name=str(args.source_collection),
        target_name=str(target),
        execute=bool(args.execute),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
