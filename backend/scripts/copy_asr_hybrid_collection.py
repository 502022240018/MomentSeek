#!/usr/bin/env python3
"""Blue-green copy of a legacy ASR collection into the hybrid ASR schema.

The source collection is never altered or deleted.  Rows are streamed into a
new collection, where Milvus generates the BM25 function output from ``text``.
The target is usable only after schema, indexes, counts, version counts, and an
order-independent content fingerprint all match the source payload.
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

from app.core.settings import get_settings  # noqa: E402
from app.vector_store.milvus.milvus_client import (  # noqa: E402
    _COLLECTION_CONFIGS,
)
from app.vector_store.milvus.milvus_schema import (  # noqa: E402
    EMBEDDING_DIMS,
    create_asr_schema,
)


DEFAULT_SOURCE = "asr_embeddings"
COPY_FIELDS = (
    "pk",
    "video_id",
    "asset_version",
    "model_version",
    "segment_idx",
    "start_ms",
    "end_ms",
    "text",
    "has_embedding",
    "embedding",
)
REQUIRED_TARGET_FIELDS = frozenset((*COPY_FIELDS, "sparse_embedding"))
REQUIRED_TARGET_FUNCTIONS = frozenset({"bm25_asr"})
REQUIRED_TARGET_INDEXES = frozenset({"embedding", "sparse_embedding"})
_BATCH_SIZE = 500
_DIGEST_MODULUS = 1 << 256


def _state(name: str, *, alias: str) -> dict[str, Any]:
    if not utility.has_collection(name, using=alias):
        return {"exists": False, "ready": False}
    collection = Collection(name, using=alias)
    fields = {field.name for field in collection.schema.fields}
    functions = {
        function.name
        for function in (getattr(collection.schema, "functions", None) or [])
    }
    indexes = {
        index.field_name
        for index in collection.indexes
        if getattr(index, "field_name", None)
    }
    return {
        "exists": True,
        "ready": (
            REQUIRED_TARGET_FIELDS.issubset(fields)
            and REQUIRED_TARGET_FUNCTIONS.issubset(functions)
            and REQUIRED_TARGET_INDEXES.issubset(indexes)
        ),
        "fields": sorted(fields),
        "functions": sorted(functions),
        "indexes": sorted(indexes),
        "row_count": int(collection.num_entities),
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
        raise RuntimeError(f"ASR row is missing fields: {missing}")
    embedding = np.asarray(row["embedding"], dtype="<f4")
    if embedding.shape != (EMBEDDING_DIMS["asr"],):
        raise RuntimeError(f"invalid ASR embedding shape: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise RuntimeError("ASR embedding contains non-finite values")
    metadata = [
        row[field]
        for field in COPY_FIELDS
        if field != "embedding"
    ]
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
        summary["digest_sum"] = (
            summary["digest_sum"] + digest
        ) % _DIGEST_MODULUS
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


def _create_target(name: str, *, alias: str) -> Collection:
    target = Collection(
        name=name,
        schema=create_asr_schema(),
        consistency_level="Strong",
        using=alias,
    )
    return target


def copy_collection(
    *,
    source_name: str,
    target_name: str,
    execute: bool,
) -> dict[str, Any]:
    settings = get_settings()
    if source_name == target_name:
        raise ValueError("source and target ASR collection names must differ")
    alias = "asr_blue_green_copy"
    connections.connect(
        alias=alias,
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        timeout=settings.milvus_query_timeout_seconds,
    )
    try:
        operation_timeout = max(
            60.0,
            float(settings.milvus_query_timeout_seconds),
        )
        source_state = _state(source_name, alias=alias)
        target_state = _state(target_name, alias=alias)
        if not source_state["exists"]:
            raise RuntimeError(f"source ASR collection does not exist: {source_name}")
        missing_source = sorted(set(COPY_FIELDS) - set(source_state["fields"]))
        if missing_source:
            raise RuntimeError(
                f"source ASR collection is missing copy fields: {missing_source}"
            )
        if target_state["exists"]:
            raise RuntimeError(
                f"target ASR collection already exists: {target_name}; "
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
        target = _create_target(target_name, alias=alias)
        source_summary = _empty_summary()
        for page in _iter_pages(source, fields=list(COPY_FIELDS), timeout=float(
            settings.milvus_query_timeout_seconds
        )):
            _add_rows(source_summary, page)
            target.insert(page, timeout=operation_timeout)
        target.flush(timeout=operation_timeout)
        for field_name, index_params in _COLLECTION_CONFIGS[
            "asr_embeddings"
        ]["indexes"].items():
            target.create_index(
                field_name=field_name,
                index_params=index_params,
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
            raise RuntimeError(f"target ASR schema/index verification failed: {verified_state}")
        if int(source_state["row_count"]) != source_summary_out["row_count"]:
            raise RuntimeError(
                "source ASR count changed during copy: "
                f'{source_state["row_count"]} != {source_summary_out["row_count"]}'
            )
        if source_summary_out != target_summary:
            raise RuntimeError(
                "target ASR content verification failed: "
                f"source={source_summary_out}, target={target_summary}"
            )
        if int(verified_state["row_count"]) != target_summary["row_count"]:
            raise RuntimeError(
                "target ASR entity count disagrees with scanned rows: "
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
        description="把旧 ASR 行蓝绿复制到新的 DiskANN + BM25 collection"
    )
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE)
    parser.add_argument("--target-collection")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-create-target", action="store_true")
    args = parser.parse_args()
    if args.execute != args.confirm_create_target:
        parser.error("执行复制必须同时传入 --execute --confirm-create-target")
    settings = get_settings()
    target = args.target_collection or settings.milvus_asr_collection
    report = copy_collection(
        source_name=str(args.source_collection),
        target_name=str(target),
        execute=bool(args.execute),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
