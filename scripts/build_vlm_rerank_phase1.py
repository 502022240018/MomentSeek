from __future__ import annotations

import argparse
import concurrent.futures
import json
import mimetypes
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONSTRAINT_KEYS = ("subject", "appearance", "scene", "action", "object", "temporal", "text_or_speech")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "momentseek-vlm-rerank-phase1"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def api_post_search(base_url: str, query: dict[str, Any], limit: int, timeout: float) -> dict[str, Any]:
    body, content_type = _multipart({
        "query_text": str(query["query"]),
        "modalities": ",".join(query["retrieval_modalities"]),
        "limit": str(limit),
    })
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", "api/search"),
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"search API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot connect to {base_url}: {exc.reason}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("search API returned an unexpected payload")
    return payload


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("video_id")) != str(right.get("video_id")):
        return False
    left_start, left_end = float(left.get("start_time") or 0), float(left.get("end_time") or 0)
    right_start, right_end = float(right.get("start_time") or 0), float(right.get("end_time") or 0)
    return min(left_end, right_end) > max(left_start, right_start)


def _merge_channel_results(channel_results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin channel Top-K and merge candidates that overlap in time."""
    pool: list[dict[str, Any]] = []
    max_length = max((len(rows) for rows in channel_results.values()), default=0)
    for rank_index in range(max_length):
        for channel, rows in channel_results.items():
            if rank_index >= len(rows):
                continue
            incoming = dict(rows[rank_index])
            source = {"channel": channel, "rank": rank_index + 1, "score": incoming.get("score")}
            existing = next((item for item in pool if _overlaps(item, incoming)), None)
            if existing is None:
                incoming["retrieval_sources"] = [source]
                pool.append(incoming)
                continue
            existing["start_time"] = min(float(existing.get("start_time") or 0), float(incoming.get("start_time") or 0))
            existing["end_time"] = max(float(existing.get("end_time") or 0), float(incoming.get("end_time") or 0))
            existing["modalities"] = sorted(set(existing.get("modalities", [])) | set(incoming.get("modalities", [])))
            existing["evidence"] = list(existing.get("evidence", [])) + list(incoming.get("evidence", []))
            existing["retrieval_sources"].append(source)
    return pool


def candidate_pool(base_url: str, query: dict[str, Any], limit: int, timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if query["mode"] == "visual_only":
        payload = api_post_search(base_url, query, limit, timeout)
        rows = []
        for rank, result in enumerate(payload["results"][:limit], 1):
            result = dict(result)
            result["retrieval_sources"] = [{"channel": "visual", "rank": rank, "score": result.get("score")}]
            rows.append(result)
        return rows, {"strategy": "single_channel", "channels": ["visual"], "elapsed_seconds": payload.get("elapsed_seconds")}

    channel_results: dict[str, list[dict[str, Any]]] = {}
    elapsed: dict[str, Any] = {}
    for channel in ("visual", "asr"):
        channel_query = dict(query)
        channel_query["retrieval_modalities"] = [channel]
        payload = api_post_search(base_url, channel_query, limit, timeout)
        channel_results[channel] = payload["results"][:limit]
        elapsed[channel] = payload.get("elapsed_seconds")
    return _merge_channel_results(channel_results), {
        "strategy": "channel_union",
        "channels": ["visual", "asr"],
        "top_k_per_channel": limit,
        "elapsed_seconds_by_channel": elapsed,
    }


def _frame_times(start: float, end: float, count: int) -> list[float]:
    if end <= start:
        return [max(0.0, start)]
    # Avoid exact clip boundaries, where decoders can return a neighbouring shot.
    return [round(start + (end - start) * (index + 1) / (count + 1), 3) for index in range(count)]


def _download(url: str, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "MomentSeek-phase1-builder/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            content_type = response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"frame API returned HTTP {exc.code}: {url}") from exc
    if not data:
        raise RuntimeError(f"frame API returned an empty image: {url}")
    if not content_type.startswith("image/") and mimetypes.guess_type(destination.name)[0] is None:
        raise RuntimeError(f"frame API returned {content_type}, not an image: {url}")
    destination.write_bytes(data)


def _model_evidence(items: Any) -> list[dict[str, Any]]:
    """Keep semantic evidence while stripping retrieval scores and decisions."""
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {
            key: item[key]
            for key in ("modality", "text", "best_time", "unit_type", "unit_id")
            if item.get(key) not in (None, "")
        }
        if row:
            output.append(row)
    return output


def validate_queries(rows: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id") or "").strip()
        if not query_id or query_id in ids:
            raise ValueError(f"missing or duplicate query_id: {query_id!r}")
        ids.add(query_id)
        if row.get("mode") not in {"visual_only", "evidence_fusion"}:
            raise ValueError(f"{query_id}: mode must be visual_only or evidence_fusion")
        if not str(row.get("query") or "").strip():
            raise ValueError(f"{query_id}: query is empty")
        constraints = row.get("constraints")
        if not isinstance(constraints, list) or len(constraints) < 3:
            raise ValueError(f"{query_id}: at least three constraints are required")
        modalities = row.get("retrieval_modalities")
        if not isinstance(modalities, list) or not modalities:
            raise ValueError(f"{query_id}: retrieval_modalities is required")


def collect(args: argparse.Namespace) -> None:
    queries = read_jsonl(args.queries)
    validate_queries(queries)
    output = args.output
    candidates_dir = output / "candidates"
    annotation_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_url": args.base_url,
        "query_source": str(args.queries),
        "top_k": args.limit,
        "frames_per_candidate": args.frames,
        "queries": len(queries),
    }
    for query in queries:
        query_id = query["query_id"]
        pool, retrieval_metadata = candidate_pool(args.base_url, query, args.limit, args.timeout)
        collected: list[dict[str, Any]] = []
        for rank, result in enumerate(pool, 1):
            video_id = str(result["video_id"])
            start = float(result.get("start_time") or 0.0)
            end = float(result.get("end_time") or start)
            candidate_id = f"{query_id}__r{rank:02d}"
            frame_paths: list[str] = []
            downloads: list[tuple[str, Path]] = []
            for frame_index, timestamp in enumerate(_frame_times(start, end, args.frames), 1):
                relative = Path("frames") / query_id / f"{candidate_id}__f{frame_index:02d}.jpg"
                url = urllib.parse.urljoin(
                    args.base_url.rstrip("/") + "/",
                    f"api/videos/{urllib.parse.quote(video_id)}/frame?time={timestamp:.3f}",
                )
                downloads.append((url, output / relative))
                frame_paths.append(relative.as_posix())
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, args.frames)) as executor:
                futures = [executor.submit(_download, url, destination, args.timeout) for url, destination in downloads]
                for future in futures:
                    future.result()
            candidate = {
                "candidate_id": candidate_id,
                "rank": rank,
                "video_id": video_id,
                "video_name": result.get("video_name", video_id),
                "start_time": start,
                "end_time": end,
                "baseline_score": result.get("score"),
                "baseline_modalities": result.get("modalities", []),
                "retrieval_sources": result.get("retrieval_sources", []),
                "above_threshold": result.get("above_threshold"),
                "frame_paths": frame_paths,
                "evidence": result.get("evidence", []),
                "model_input": {
                    "query": query["query"],
                    "mode": query["mode"],
                    "frame_paths": frame_paths,
                    "evidence": _model_evidence(result.get("evidence", []))
                    if query["mode"] == "evidence_fusion" else [],
                },
            }
            collected.append(candidate)
            annotation_rows.append({
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "candidate_id": candidate_id,
                "original_rank": rank,
                "video_id": video_id,
                "start_time": start,
                "end_time": end,
                "relevance": None,
                "constraint_labels": {key: None for key in CONSTRAINT_KEYS},
                "reason": "",
                "reviewer": "",
            })
        result_rows.append({
            "schema_version": SCHEMA_VERSION,
            "query_id": query_id,
            "query": query["query"],
            "mode": query["mode"],
            "constraints": query["constraints"],
            "expected_video": query.get("expected_video"),
            "retrieval": {
                **retrieval_metadata,
                "returned": len(collected),
            },
            "candidates": collected,
        })
        print(f"{query_id}: collected {len(collected)} candidates")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(candidates_dir / "candidate_sets.jsonl", result_rows)
    write_jsonl(output / "annotations.jsonl", annotation_rows)
    print(f"dataset ready: {output.resolve()}")


def validate_dataset(path: Path) -> None:
    queries_path = path / "candidates" / "candidate_sets.jsonl"
    annotations_path = path / "annotations.jsonl"
    query_rows = read_jsonl(queries_path)
    annotations = read_jsonl(annotations_path)
    expected = {
        (row["query_id"], candidate["candidate_id"])
        for row in query_rows for candidate in row.get("candidates", [])
    }
    observed: set[tuple[str, str]] = set()
    errors: list[str] = []
    for row in annotations:
        key = (str(row.get("query_id")), str(row.get("candidate_id")))
        if key in observed:
            errors.append(f"duplicate annotation {key}")
        observed.add(key)
        relevance = row.get("relevance")
        if relevance is not None and (not isinstance(relevance, int) or not 0 <= relevance <= 3):
            errors.append(f"{key}: relevance must be null or integer 0..3")
        labels = row.get("constraint_labels", {})
        if not isinstance(labels, dict) or any(value not in (None, True, False) for value in labels.values()):
            errors.append(f"{key}: constraint labels must be null/true/false")
    if expected != observed:
        errors.append(f"candidate/annotation mismatch: missing={len(expected-observed)}, unknown={len(observed-expected)}")
    missing_frames = [
        frame for row in query_rows for candidate in row.get("candidates", [])
        for frame in candidate.get("frame_paths", []) if not (path / frame).is_file()
    ]
    if missing_frames:
        errors.append(f"missing {len(missing_frames)} frame files")
    if errors:
        raise ValueError("dataset validation failed:\n- " + "\n- ".join(errors))
    completed = sum(row.get("relevance") is not None for row in annotations)
    print(json.dumps({"queries": len(query_rows), "candidates": len(expected), "annotated": completed}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MomentSeek phase-1 VLM reranking dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="search Top-K and download ordered candidate frames")
    collect_parser.add_argument("--queries", type=Path, default=Path("eval/vlm_rerank_phase1/queries.seed.jsonl"))
    collect_parser.add_argument("--output", type=Path, default=Path("runtime/eval/vlm_rerank_phase1"))
    collect_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    collect_parser.add_argument("--limit", type=int, default=20)
    collect_parser.add_argument("--frames", type=int, default=4)
    collect_parser.add_argument("--timeout", type=float, default=60.0)
    validate_parser = subparsers.add_parser("validate", help="validate frames and human annotation values")
    validate_parser.add_argument("--dataset", type=Path, default=Path("runtime/eval/vlm_rerank_phase1"))
    args = parser.parse_args()
    if args.command == "collect":
        if args.limit < 1 or args.frames < 1:
            parser.error("--limit and --frames must be positive")
        collect(args)
    else:
        validate_dataset(args.dataset)


if __name__ == "__main__":
    main()
