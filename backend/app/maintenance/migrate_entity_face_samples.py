"""One-time migration of registered entity references into Milvus.

This command deliberately ignores the legacy ``entities.embedding_path``
payload.  A face sample is reconstructed from the retained reference image
with the same FaceEncoder configuration and Milvus row contract used by the
online entity registration flow.

The command is read-only unless ``--apply`` is supplied.  Existing Milvus
samples make the operation idempotent: they are skipped by default.  Explicit
``--replace`` writes and verifies the deterministic reference sample before it
removes the samples that existed at the start of the migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.catalog.db import Catalog
from app.core.settings import get_settings
from app.indexing.common import normalize
from app.vector_store.milvus.milvus_client import get_milvus_client
from app.vector_store.milvus.milvus_schema import (
    EMBEDDING_DIMS,
    entity_face_sample_pk,
)


_COLLECTION = "entity_face_samples"
_QUERY_BATCH_SIZE = 1_000


def _expr_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _reference_sample_id(entity_id: str) -> str:
    """Return a stable sample id so interrupted runs can be retried safely."""
    return hashlib.sha256(
        f"entity-reference\0{entity_id}".encode("utf-8")
    ).hexdigest()[:32]


def _query_samples(collection, entity_id: str, *, timeout: float) -> list[dict]:
    expr = f"entity_id == {_expr_string(entity_id)}"
    fields = [
        "pk",
        "entity_id",
        "sample_id",
        "source_video_id",
        "source_asset_version",
        "source_group_idx",
        "quality",
        "embedding",
    ]
    if not hasattr(collection, "query_iterator"):
        return list(
            collection.query(
                expr=expr,
                output_fields=fields,
                limit=16_384,
                timeout=timeout,
            )
        )

    rows: list[dict] = []
    iterator = collection.query_iterator(
        batch_size=_QUERY_BATCH_SIZE,
        expr=expr,
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
            rows.extend(page)
    finally:
        iterator.close()
    return rows


def _build_encoder(settings):
    # Import lazily so a dry-run neither loads InsightFace nor claims a device.
    from app.encoders.face import FaceEncoder

    return FaceEncoder(
        settings.face_model,
        settings.face_provider,
        settings.npu_device_id,
        str(settings.app_model_dir / "insightface"),
        settings.face_ort_intra_op_threads,
        settings.face_ort_inter_op_threads,
    )


def _validated_embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = int(EMBEDDING_DIMS["face"])
    if vector.size != expected:
        raise ValueError(
            f"FaceEncoder returned dimension {vector.size}; expected {expected}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("FaceEncoder returned a non-finite embedding")
    if float(np.linalg.norm(vector)) <= 1e-12:
        raise ValueError("FaceEncoder returned a zero-norm embedding")
    return normalize(vector)


def _validated_persisted_embedding(value: Any) -> np.ndarray:
    """Validate that a Milvus sample is directly usable for cosine matching."""
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    expected = int(EMBEDDING_DIMS["face"])
    if vector.size != expected:
        raise ValueError(
            f"persisted face embedding dimension is {vector.size}; expected {expected}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("persisted face embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("persisted face embedding has zero norm")
    if not np.isclose(norm, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError(
            f"persisted face embedding is not unit-normalized (norm={norm:.6f})"
        )
    return vector


def _usable_sample_count(rows: Iterable[dict]) -> tuple[int, list[str]]:
    usable = 0
    errors: list[str] = []
    for row in rows:
        try:
            _validated_persisted_embedding(row.get("embedding"))
            usable += 1
        except (TypeError, ValueError) as exc:
            errors.append(f"{row.get('pk') or '<missing-pk>'}: {exc}")
    return usable, errors


def _verify_target_sample(
    rows: Iterable[dict],
    *,
    entity_id: str,
    sample_id: str,
    target_pk: str,
    expected_embedding: np.ndarray,
) -> None:
    matches = [
        row for row in rows if str(row.get("pk") or "") == target_pk
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Milvus did not return exactly one newly written reference sample "
            f"(found={len(matches)})"
        )
    row = matches[0]
    persisted = _validated_persisted_embedding(row.get("embedding"))
    expected_contract = {
        "entity_id": entity_id,
        "sample_id": sample_id,
        "source_video_id": "",
        "source_asset_version": "",
        "source_group_idx": -1,
    }
    for field, expected in expected_contract.items():
        if row.get(field) != expected:
            raise RuntimeError(
                f"persisted reference sample has invalid {field}: "
                f"{row.get(field)!r} != {expected!r}"
            )
    if not np.allclose(
        persisted,
        expected_embedding,
        rtol=1e-5,
        atol=1e-6,
    ):
        cosine = float(np.dot(persisted, expected_embedding))
        raise RuntimeError(
            "persisted reference embedding does not match encoded value "
            f"(cosine={cosine:.8f})"
        )


def _delete_initial_samples(
    collection,
    initial_rows: Iterable[dict],
    *,
    keep_pk: str,
) -> tuple[int, set[str]]:
    old_pks = sorted(
        {
            str(row.get("pk") or "")
            for row in initial_rows
            if row.get("pk") and str(row["pk"]) != keep_pk
        }
    )
    if not old_pks:
        return 0, set()
    result = collection.delete(
        f"pk in {json.dumps(old_pks, ensure_ascii=False)}"
    )
    collection.flush()
    return int(getattr(result, "delete_count", 0)), set(old_pks)


def _cleanup_new_target(
    collection,
    *,
    entity_id: str,
    target_pk: str,
    timeout: float,
) -> None:
    """Best-effort rollback of a target created by a failed write/verify step."""
    collection.delete(f"pk == {_expr_string(target_pk)}")
    collection.flush()
    remaining = _query_samples(collection, entity_id, timeout=timeout)
    if any(str(row.get("pk") or "") == target_pk for row in remaining):
        raise RuntimeError("failed to remove the unverified target sample")


def _validate_replace_deletion(
    rows: Iterable[dict],
    *,
    deleted_pks: set[str],
    delete_count: int,
) -> None:
    if delete_count != len(deleted_pks):
        raise RuntimeError(
            "Milvus delete_count does not match the requested old samples "
            f"(deleted={delete_count}, expected={len(deleted_pks)})"
        )
    remaining = sorted(
        str(row.get("pk") or "")
        for row in rows
        if str(row.get("pk") or "") in deleted_pks
    )
    if remaining:
        raise RuntimeError(
            f"Milvus still contains replaced face sample PKs: {remaining[:10]}"
        )


def migrate(
    *,
    apply: bool,
    replace: bool = False,
    entity_ids: set[str] | None = None,
    catalog=None,
    client=None,
    encoder=None,
    settings=None,
) -> dict:
    """Audit or migrate registered entity reference images.

    Dependencies are injectable to keep this one-time maintenance operation
    testable without an InsightFace model or a live Milvus instance.
    """
    settings = settings or get_settings()
    catalog = catalog or Catalog(settings.db_path)
    client = client or get_milvus_client()
    collection = client.collection(_COLLECTION)
    timeout = float(settings.milvus_query_timeout_seconds)
    report: dict[str, Any] = {
        "apply": bool(apply),
        "replace": bool(replace),
        "migrated": [],
        "skipped": [],
        "errors": [],
    }

    shared_encoder = encoder
    all_entities = catalog.list_entities()
    found_ids = {str(entity.get("id") or "") for entity in all_entities}
    entities = all_entities
    if entity_ids is not None:
        entities = [
            entity for entity in entities if str(entity.get("id")) in entity_ids
        ]
        for missing in sorted(entity_ids - found_ids):
            report["errors"].append(
                {"entity_id": missing, "error": "entity does not exist in Catalog"}
            )

    for entity in entities:
        entity_id = str(entity.get("id") or "")
        legacy_path_present = bool(entity.get("embedding_path"))
        try:
            initial_rows = _query_samples(
                collection, entity_id, timeout=timeout
            )
            if initial_rows and not replace:
                usable_count, invalid_samples = _usable_sample_count(initial_rows)
                if usable_count < 1:
                    detail = "; ".join(invalid_samples[:3])
                    raise RuntimeError(
                        f"Milvus has {len(initial_rows)} face sample row(s), but none "
                        "contains a usable normalized 512-d embedding; rerun with "
                        f"--replace after checking the reference image. {detail}"
                    )
                sample_id = _reference_sample_id(entity_id)
                target_pk = entity_face_sample_pk(entity_id, sample_id)
                target_present = any(
                    str(row.get("pk") or "") == target_pk for row in initial_rows
                )
                target_verified = False
                if apply and target_present:
                    reference_value = str(entity.get("reference_path") or "").strip()
                    if not reference_value:
                        raise ValueError(
                            "deterministic reference sample exists, but reference_path is missing"
                        )
                    reference_path = Path(reference_value)
                    if not reference_path.is_file():
                        raise FileNotFoundError(
                            f"reference image does not exist: {reference_path}"
                        )
                    if shared_encoder is None:
                        shared_encoder = _build_encoder(settings)
                    expected = _validated_embedding(
                        shared_encoder.encode_reference(str(reference_path))
                    )
                    _verify_target_sample(
                        initial_rows,
                        entity_id=entity_id,
                        sample_id=sample_id,
                        target_pk=target_pk,
                        expected_embedding=expected,
                    )
                    target_verified = True
                path_cleared = False
                if apply and legacy_path_present:
                    # A deterministic target must match a fresh reference-image
                    # encoding.  Other pre-existing samples retain the legacy
                    # behaviour of requiring at least one usable vector.
                    catalog.update_entity_embedding(entity_id, "")
                    path_cleared = True
                report["skipped"].append(
                    {
                        "entity_id": entity_id,
                        "name": entity.get("name", ""),
                        "reason": "samples_already_present",
                        "sample_count": len(initial_rows),
                        "usable_sample_count": usable_count,
                        "invalid_sample_count": len(invalid_samples),
                        "deterministic_target_present": target_present,
                        "deterministic_target_verified": target_verified,
                        "would_verify_deterministic_target": bool(
                            not apply and target_present
                        ),
                        "embedding_path_cleared": path_cleared,
                        "would_clear_embedding_path": bool(
                            not apply and legacy_path_present
                        ),
                    }
                )
                continue

            reference_value = str(entity.get("reference_path") or "").strip()
            if not reference_value:
                report["skipped"].append(
                    {
                        "entity_id": entity_id,
                        "name": entity.get("name", ""),
                        "reason": "no_reference_path",
                        "sample_count": len(initial_rows),
                    }
                )
                continue
            reference_path = Path(reference_value)
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"reference image does not exist: {reference_path}"
                )

            sample_id = _reference_sample_id(entity_id)
            target_pk = entity_face_sample_pk(entity_id, sample_id)
            if not apply:
                report["migrated"].append(
                    {
                        "entity_id": entity_id,
                        "name": entity.get("name", ""),
                        "sample_id": sample_id,
                        "reference_path": str(reference_path),
                        "existing_sample_count": len(initial_rows),
                        "status": "would_replace" if replace else "would_migrate",
                        "would_clear_embedding_path": legacy_path_present,
                    }
                )
                continue

            if shared_encoder is None:
                shared_encoder = _build_encoder(settings)
            vector = _validated_embedding(
                shared_encoder.encode_reference(str(reference_path))
            )
            target_preexisted = any(
                str(row.get("pk") or "") == target_pk for row in initial_rows
            )
            try:
                collection.upsert(
                    [
                        {
                            "pk": target_pk,
                            "entity_id": entity_id,
                            "sample_id": sample_id,
                            "source_video_id": "",
                            "source_asset_version": "",
                            "source_group_idx": -1,
                            "quality": 1.0,
                            "embedding": vector.astype(np.float32).tolist(),
                        }
                    ]
                )
                collection.flush()

                after_write = _query_samples(
                    collection, entity_id, timeout=timeout
                )
                _verify_target_sample(
                    after_write,
                    entity_id=entity_id,
                    sample_id=sample_id,
                    target_pk=target_pk,
                    expected_embedding=vector,
                )
            except Exception as write_exc:
                if not target_preexisted:
                    try:
                        _cleanup_new_target(
                            collection,
                            entity_id=entity_id,
                            target_pk=target_pk,
                            timeout=timeout,
                        )
                    except Exception as cleanup_exc:
                        raise RuntimeError(
                            f"{write_exc}; cleanup of unverified target failed: "
                            f"{cleanup_exc}"
                        ) from write_exc
                raise

            removed = 0
            if replace:
                removed, deleted_pks = _delete_initial_samples(
                    collection, initial_rows, keep_pk=target_pk
                )
                after_replace = _query_samples(
                    collection, entity_id, timeout=timeout
                )
                _verify_target_sample(
                    after_replace,
                    entity_id=entity_id,
                    sample_id=sample_id,
                    target_pk=target_pk,
                    expected_embedding=vector,
                )
                _validate_replace_deletion(
                    after_replace,
                    deleted_pks=deleted_pks,
                    delete_count=removed,
                )

            # Do this last.  Every failure above deliberately leaves the old
            # Catalog path untouched so operators retain a recovery pointer.
            if legacy_path_present:
                catalog.update_entity_embedding(entity_id, "")
            report["migrated"].append(
                {
                    "entity_id": entity_id,
                    "name": entity.get("name", ""),
                    "sample_id": sample_id,
                    "reference_path": str(reference_path),
                    "existing_sample_count": len(initial_rows),
                    "removed_sample_count": removed,
                    "status": "replaced" if replace else "migrated",
                    "embedding_path_cleared": legacy_path_present,
                }
            )
        except Exception as exc:
            logging.exception("Entity face migration failed for %s", entity_id)
            report["errors"].append(
                {
                    "entity_id": entity_id,
                    "name": entity.get("name", ""),
                    "error": str(exc),
                }
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write verified samples. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace existing samples after the new reference sample has been "
            "written and verified."
        ),
    )
    parser.add_argument("--entity-id", action="append", dest="entity_ids")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = migrate(
        apply=bool(args.apply),
        replace=bool(args.replace),
        entity_ids=set(args.entity_ids) if args.entity_ids else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
