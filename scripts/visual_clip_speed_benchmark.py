from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from visual_clip_eval import (
    ModelSpec,
    OpenClipRunner,
    _sliding_square_crops,
    _split_sheet_cells,
    build_queries,
    load_image_items,
    load_sequence_items,
    parse_model_spec,
)


def _sync(runner: OpenClipRunner) -> None:
    if str(runner.device).startswith("cuda") and runner.torch.cuda.is_available():
        runner.torch.cuda.synchronize()
    if str(runner.device).startswith("npu") and hasattr(runner.torch, "npu"):
        runner.torch.npu.synchronize()


def _now() -> float:
    return time.perf_counter()


def _round(value: float) -> float:
    return round(float(value), 6)


def _image_tensor_from_pil(runner: OpenClipRunner, image: Image.Image, mode: str):
    if mode == "center_crop":
        return runner.preprocess(image.convert("RGB"))
    if mode == "letterbox":
        return runner._letterbox_tensor(image)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def encode_sources_timed(
    runner: OpenClipRunner,
    sources: list[str | Path | Image.Image],
    mode: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = runner.torch
    outputs: list[np.ndarray] = []
    preprocess_seconds = 0.0
    transfer_seconds = 0.0
    encode_seconds = 0.0
    to_cpu_seconds = 0.0

    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            batch_sources = sources[start:start + batch_size]
            t0 = _now()
            tensors = []
            for source in batch_sources:
                if isinstance(source, Image.Image):
                    tensors.append(_image_tensor_from_pil(runner, source, mode))
                else:
                    tensors.append(runner._image_tensor(source, mode))
            batch = torch.stack(tensors)
            preprocess_seconds += _now() - t0

            t0 = _now()
            batch = batch.to(runner.device)
            _sync(runner)
            transfer_seconds += _now() - t0

            t0 = _now()
            encoded = runner.model.encode_image(batch)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            _sync(runner)
            encode_seconds += _now() - t0

            t0 = _now()
            outputs.append(encoded.float().cpu().numpy())
            to_cpu_seconds += _now() - t0

    embeddings = np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.empty((0, 0), dtype=np.float32)
    timing = {
        "preprocess_cpu_seconds": _round(preprocess_seconds),
        "transfer_to_device_seconds": _round(transfer_seconds),
        "encode_device_seconds": _round(encode_seconds),
        "to_cpu_seconds": _round(to_cpu_seconds),
        "encode_pipeline_seconds": _round(preprocess_seconds + transfer_seconds + encode_seconds + to_cpu_seconds),
        "views": len(sources),
        "views_per_second_total": _round(len(sources) / max(1e-9, preprocess_seconds + transfer_seconds + encode_seconds + to_cpu_seconds)),
        "views_per_second_encoder_only": _round(len(sources) / max(1e-9, encode_seconds)),
    }
    return embeddings, timing


def encode_texts_timed(
    runner: OpenClipRunner,
    texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = runner.torch
    outputs: list[np.ndarray] = []
    tokenize_seconds = 0.0
    transfer_seconds = 0.0
    encode_seconds = 0.0
    to_cpu_seconds = 0.0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            t0 = _now()
            tokens = runner.tokenizer(batch_texts)
            tokenize_seconds += _now() - t0

            t0 = _now()
            tokens = tokens.to(runner.device)
            _sync(runner)
            transfer_seconds += _now() - t0

            t0 = _now()
            encoded = runner.model.encode_text(tokens)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            _sync(runner)
            encode_seconds += _now() - t0

            t0 = _now()
            outputs.append(encoded.float().cpu().numpy())
            to_cpu_seconds += _now() - t0
    embeddings = np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.empty((0, 0), dtype=np.float32)
    timing = {
        "text_tokenize_cpu_seconds": _round(tokenize_seconds),
        "text_transfer_to_device_seconds": _round(transfer_seconds),
        "text_encode_device_seconds": _round(encode_seconds),
        "text_to_cpu_seconds": _round(to_cpu_seconds),
        "text_pipeline_seconds": _round(tokenize_seconds + transfer_seconds + encode_seconds + to_cpu_seconds),
        "queries": len(texts),
        "queries_per_second_total": _round(len(texts) / max(1e-9, tokenize_seconds + transfer_seconds + encode_seconds + to_cpu_seconds)),
    }
    return embeddings, timing


def warmup_runner(
    runner: OpenClipRunner,
    image_paths: list[str | Path],
    texts: list[str],
    image_batch_size: int,
    text_batch_size: int,
) -> dict[str, Any]:
    torch = runner.torch
    start = _now()
    image_batch = min(max(1, image_batch_size), len(image_paths))
    text_batch = min(max(1, text_batch_size), len(texts))
    with torch.inference_mode():
        if image_paths:
            tensors = [runner._image_tensor(path, "center_crop") for path in image_paths[:image_batch]]
            batch = torch.stack(tensors).to(runner.device)
            _sync(runner)
            encoded = runner.model.encode_image(batch)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            _sync(runner)
            _ = encoded.float().cpu().numpy()
        if texts:
            tokens = runner.tokenizer(texts[:text_batch]).to(runner.device)
            _sync(runner)
            encoded = runner.model.encode_text(tokens)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            _sync(runner)
            _ = encoded.float().cpu().numpy()
    return {
        "warmup_seconds": _round(_now() - start),
        "warmup_image_batch": image_batch if image_paths else 0,
        "warmup_text_batch": text_batch if texts else 0,
    }


def _prepare_image_sliding(paths: list[str]) -> tuple[list[Image.Image], np.ndarray, np.ndarray, float]:
    t0 = _now()
    views: list[Image.Image] = []
    view_item_indices: list[int] = []
    offsets = [0]
    for item_index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        crops = _sliding_square_crops(image)
        views.extend(crops)
        view_item_indices.extend([item_index] * len(crops))
        offsets.append(len(views))
    return views, np.asarray(view_item_indices, dtype=np.int32), np.asarray(offsets, dtype=np.int32), _now() - t0


def _prepare_sequence_cells(sheets: list[dict[str, Any]], sliding: bool) -> tuple[list[Image.Image], np.ndarray, np.ndarray, float]:
    t0 = _now()
    views: list[Image.Image] = []
    view_item_indices: list[int] = []
    offsets = [0]
    for item_index, sheet in enumerate(sheets):
        cells = _split_sheet_cells(sheet)
        if sliding:
            expanded = []
            for cell in cells:
                expanded.extend(_sliding_square_crops(cell))
            cells = expanded
        views.extend(cells)
        view_item_indices.extend([item_index] * len(cells))
        offsets.append(len(views))
    return views, np.asarray(view_item_indices, dtype=np.int32), np.asarray(offsets, dtype=np.int32), _now() - t0


def score_direct_timed(query_embeddings: np.ndarray, item_embeddings: np.ndarray, repeat: int = 1) -> dict[str, Any]:
    t0 = _now()
    result = None
    for _ in range(repeat):
        result = query_embeddings @ item_embeddings.T
    elapsed = (_now() - t0) / repeat
    return {
        "score_cpu_seconds": _round(elapsed),
        "score_queries_per_second": _round(len(query_embeddings) / max(1e-9, elapsed)),
        "score_shape": list(result.shape) if result is not None else [0, 0],
    }


def score_views_timed(
    query_embeddings: np.ndarray,
    view_embeddings: np.ndarray,
    offsets: np.ndarray,
    item_count: int,
    aggregate: str,
    repeat: int = 1,
) -> dict[str, Any]:
    def aggregate_one(values: np.ndarray) -> float:
        if aggregate == "mean":
            return float(np.mean(values))
        ordered = np.sort(values)[::-1]
        if aggregate == "max":
            return float(ordered[0])
        if aggregate == "top3":
            return float(np.mean(ordered[:min(3, len(ordered))]))
        if aggregate == "mvp_mix":
            mean = float(np.mean(values))
            top1 = float(ordered[0])
            top3 = float(np.mean(ordered[:min(3, len(ordered))]))
            return float(0.65 * top1 + 0.25 * top3 + 0.10 * mean)
        raise ValueError(f"Unknown aggregate: {aggregate}")

    t0 = _now()
    scores = np.zeros((len(query_embeddings), item_count), dtype=np.float32)
    for _ in range(repeat):
        for query_index, query in enumerate(query_embeddings):
            view_scores = view_embeddings @ query
            for item_index in range(item_count):
                start, end = int(offsets[item_index]), int(offsets[item_index + 1])
                scores[query_index, item_index] = aggregate_one(view_scores[start:end])
    elapsed = (_now() - t0) / repeat
    return {
        "score_cpu_seconds": _round(elapsed),
        "score_queries_per_second": _round(len(query_embeddings) / max(1e-9, elapsed)),
        "score_shape": list(scores.shape),
        "score_aggregate": aggregate,
    }


def _row(
    model: ModelSpec,
    device: str,
    task: str,
    scenario: str,
    source_prepare_seconds: float,
    encode_timing: dict[str, Any],
    score_timing: dict[str, Any],
    item_count: int,
) -> dict[str, Any]:
    row = {
        "model": model.slug,
        "model_name": model.model_name,
        "pretrained": model.pretrained,
        "device": device,
        "task": task,
        "scenario": scenario,
        "items": item_count,
        "source_prepare_seconds": _round(source_prepare_seconds),
    }
    row.update(encode_timing)
    row.update(score_timing)
    row["total_without_text_seconds"] = _round(
        source_prepare_seconds
        + encode_timing.get("encode_pipeline_seconds", 0)
        + score_timing.get("score_cpu_seconds", 0)
    )
    return row


def benchmark_model(args: argparse.Namespace, spec: ModelSpec) -> dict[str, Any]:
    started = _now()
    load_start = _now()
    runner = OpenClipRunner(spec, args.device)
    load_seconds = _now() - load_start

    image_item_ids, image_paths, _image_items = load_image_items(args.image_manifest)
    sequence_item_ids, sheet_paths, sheets = load_sequence_items(args.sequence_manifest)
    image_queries = build_queries(args.image_annotations, args.max_queries_per_item, include_captions=False)
    sequence_queries = build_queries(args.sequence_annotations, args.max_queries_per_item, include_captions=False)
    warmup_info = warmup_runner(
        runner,
        image_paths,
        [query.query for query in image_queries],
        args.batch_size,
        args.text_batch_size,
    )

    image_text_embeddings, image_text_timing = encode_texts_timed(
        runner,
        [query.query for query in image_queries],
        args.text_batch_size,
    )
    sequence_text_embeddings, sequence_text_timing = encode_texts_timed(
        runner,
        [query.query for query in sequence_queries],
        args.text_batch_size,
    )

    rows: list[dict[str, Any]] = []

    # Image direct views.
    for mode in ("center_crop", "letterbox"):
        embeddings, timing = encode_sources_timed(runner, image_paths, mode, args.batch_size)
        score_timing = score_direct_timed(image_text_embeddings, embeddings)
        rows.append(_row(spec, runner.device, "image", mode, 0.0, timing, score_timing, len(image_item_ids)))

    # Image spatial sliding views.
    image_views, _indices, image_offsets, prepare_seconds = _prepare_image_sliding(image_paths)
    embeddings, timing = encode_sources_timed(runner, image_views, "center_crop", args.batch_size)
    for aggregate in ("max", "top3", "mvp_mix"):
        score_timing = score_views_timed(image_text_embeddings, embeddings, image_offsets, len(image_item_ids), aggregate)
        rows.append(_row(spec, runner.device, "image", f"sliding_{aggregate}", prepare_seconds, timing, score_timing, len(image_item_ids)))

    # Sequence whole contact-sheet views.
    for mode in ("center_crop", "letterbox"):
        embeddings, timing = encode_sources_timed(runner, sheet_paths, mode, args.batch_size)
        score_timing = score_direct_timed(sequence_text_embeddings, embeddings)
        rows.append(_row(spec, runner.device, "sequence", f"sheet_whole_{mode}", 0.0, timing, score_timing, len(sequence_item_ids)))

    # Sequence cell views.
    for mode in ("center_crop", "letterbox"):
        cells, _indices, cell_offsets, prepare_seconds = _prepare_sequence_cells(sheets, sliding=False)
        embeddings, timing = encode_sources_timed(runner, cells, mode, args.batch_size)
        for aggregate in ("mean", "max", "top3", "mvp_mix"):
            score_timing = score_views_timed(sequence_text_embeddings, embeddings, cell_offsets, len(sequence_item_ids), aggregate)
            rows.append(_row(spec, runner.device, "sequence", f"cells_{aggregate}_{mode}", prepare_seconds, timing, score_timing, len(sequence_item_ids)))

    # Sequence cell + spatial sliding views.
    sliding_cells, _indices, sliding_offsets, prepare_seconds = _prepare_sequence_cells(sheets, sliding=True)
    embeddings, timing = encode_sources_timed(runner, sliding_cells, "center_crop", args.batch_size)
    for aggregate in ("max", "top3", "mvp_mix"):
        score_timing = score_views_timed(sequence_text_embeddings, embeddings, sliding_offsets, len(sequence_item_ids), aggregate)
        rows.append(_row(spec, runner.device, "sequence", f"cells_sliding_{aggregate}", prepare_seconds, timing, score_timing, len(sequence_item_ids)))

    return {
        "model": {
            "model_name": spec.model_name,
            "pretrained": spec.pretrained,
            "slug": spec.slug,
        },
        "device": runner.device,
        "load_seconds": _round(load_seconds),
        **warmup_info,
        "elapsed_seconds": _round(_now() - started),
        "image_text_timing": image_text_timing,
        "sequence_text_timing": sequence_text_timing,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CLIP visual eval CPU/GPU timing by preprocessing strategy.")
    parser.add_argument("--model-spec", action="append", default=None, help="OpenCLIP spec, e.g. ViT-B-16::openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-queries-per-item", type=int, default=3)
    parser.add_argument("--image-manifest", type=Path, default=Path("eval/visual/image_retrieval/frames.balanced_v2_300.local.json"))
    parser.add_argument("--image-annotations", type=Path, default=Path("eval/visual/image_retrieval/auto_annotations.qwen_fallback.image_balanced_v2_300.local.jsonl"))
    parser.add_argument("--sequence-manifest", type=Path, default=Path("eval/visual/sequence_retrieval/contact_sheets.balanced_v2_200.2fps_hq.local.json"))
    parser.add_argument("--sequence-annotations", type=Path, default=Path("eval/visual/sequence_retrieval/auto_annotations.qwen_fallback.sequence_balanced_v2_200_2fps_hq.local.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("eval/visual/outputs/clip_speed_benchmark_balanced_v2.local.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("eval/visual/outputs/clip_speed_benchmark_balanced_v2.local.csv"))
    args = parser.parse_args()

    specs = [parse_model_spec(value) for value in (args.model_spec or ["ViT-B-32::openai"])]
    all_runs = []
    all_rows = []
    for spec in specs:
        print(json.dumps({"event": "benchmark_start", "model": spec.slug}, ensure_ascii=False))
        run = benchmark_model(args, spec)
        all_runs.append(run)
        for row in run["rows"]:
            row["model_load_seconds"] = run["load_seconds"]
            row["warmup_seconds"] = run["warmup_seconds"]
            row["warmup_image_batch"] = run["warmup_image_batch"]
            row["warmup_text_batch"] = run["warmup_text_batch"]
            row["run_elapsed_seconds"] = run["elapsed_seconds"]
            all_rows.append(row)
        print(json.dumps({"event": "benchmark_done", "model": spec.slug, "elapsed_seconds": run["elapsed_seconds"]}, ensure_ascii=False))

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_size": args.batch_size,
        "text_batch_size": args.text_batch_size,
        "runs": all_runs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, all_rows)
    print(json.dumps({
        "out_json": str(args.out_json),
        "out_csv": str(args.out_csv),
        "rows": len(all_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
