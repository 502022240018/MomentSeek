from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visual_clip_eval import (
    OpenClipRunner,
    _evaluate_scores,
    _normalize_rows,
    _sliding_square_crops,
    _view_score_strategies,
    build_queries,
    load_image_items,
    parse_model_spec,
)


def now() -> float:
    return time.perf_counter()


def rounded(value: float) -> float:
    return round(float(value), 6)


def sync_device(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    if device.startswith("npu") and hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


def move_to_device(batch: dict[str, Any], device: str, dtype=None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=dtype)
        out[key] = value
    return out


def model_label(model_arg: str) -> str:
    safe = model_arg.replace("/", "_").replace(":", "_").replace("\\", "_")
    safe = safe.replace("::", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)


@dataclass
class TimedEmbeddings:
    embeddings: np.ndarray
    timing: dict[str, Any]


class BaseRunner:
    model_id: str
    device: str
    torch: Any

    def encode_texts_timed(self, texts: list[str], batch_size: int) -> TimedEmbeddings:
        raise NotImplementedError

    def encode_sources_timed(self, sources: list[str | Path | Image.Image], batch_size: int) -> TimedEmbeddings:
        raise NotImplementedError


class TimedOpenClipRunner(BaseRunner):
    def __init__(self, spec: str, device: str):
        self.model_id = f"openclip:{spec}"
        self.runner = OpenClipRunner(parse_model_spec(spec), device)
        self.device = self.runner.device
        self.torch = self.runner.torch

    def _image_tensor(self, source: str | Path | Image.Image):
        if isinstance(source, Image.Image):
            return self.runner.preprocess(source.convert("RGB"))
        return self.runner._image_tensor(source, "center_crop")

    def encode_sources_timed(self, sources: list[str | Path | Image.Image], batch_size: int) -> TimedEmbeddings:
        torch = self.torch
        outputs: list[np.ndarray] = []
        preprocess_seconds = transfer_seconds = encode_seconds = to_cpu_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, len(sources), batch_size):
                batch_sources = sources[start:start + batch_size]
                t0 = now()
                tensors = [self._image_tensor(source) for source in batch_sources]
                batch = torch.stack(tensors)
                preprocess_seconds += now() - t0

                t0 = now()
                batch = batch.to(self.device)
                sync_device(torch, self.device)
                transfer_seconds += now() - t0

                t0 = now()
                encoded = self.runner.model.encode_image(batch)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                sync_device(torch, self.device)
                encode_seconds += now() - t0

                t0 = now()
                outputs.append(encoded.float().cpu().numpy())
                to_cpu_seconds += now() - t0
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        return TimedEmbeddings(embeddings, timing_dict(len(sources), preprocess_seconds, transfer_seconds, encode_seconds, to_cpu_seconds))

    def encode_texts_timed(self, texts: list[str], batch_size: int) -> TimedEmbeddings:
        torch = self.torch
        outputs: list[np.ndarray] = []
        tokenize_seconds = transfer_seconds = encode_seconds = to_cpu_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                t0 = now()
                tokens = self.runner.tokenizer(batch_texts)
                tokenize_seconds += now() - t0

                t0 = now()
                tokens = tokens.to(self.device)
                sync_device(torch, self.device)
                transfer_seconds += now() - t0

                t0 = now()
                encoded = self.runner.model.encode_text(tokens)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                sync_device(torch, self.device)
                encode_seconds += now() - t0

                t0 = now()
                outputs.append(encoded.float().cpu().numpy())
                to_cpu_seconds += now() - t0
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        timing = timing_dict(len(texts), tokenize_seconds, transfer_seconds, encode_seconds, to_cpu_seconds)
        timing["tokenize_cpu_seconds"] = timing.pop("preprocess_cpu_seconds")
        return TimedEmbeddings(embeddings, timing)


class TimedHfRunner(BaseRunner):
    def __init__(self, model_id: str, device: str, dtype: str = "bf16", trust_remote_code: bool = True):
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoModel, AutoProcessor

        if device.startswith("npu"):
            torch.npu.set_device(device)
        self.torch = torch
        self.device = device
        self.model_id = f"hf:{model_id}"
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(dtype, torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

    def _image_features(self, batch: dict[str, Any]):
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(**batch)
        if hasattr(self.model, "encode_image"):
            return self.model.encode_image(**batch)
        raise AttributeError(f"{type(self.model).__name__} has no get_image_features/encode_image")

    def _text_features(self, batch: dict[str, Any]):
        if type(self.model).__name__ == "ChineseCLIPModel" and hasattr(self.model, "text_model") and hasattr(self.model, "text_projection"):
            text_outputs = self.model.text_model(**batch, return_dict=True)
            pooled = getattr(text_outputs, "pooler_output", None)
            if pooled is None:
                pooled = text_outputs.last_hidden_state[:, 0]
            return self.model.text_projection(pooled)
        if hasattr(self.model, "get_text_features"):
            return self.model.get_text_features(**batch)
        if hasattr(self.model, "encode_text"):
            return self.model.encode_text(**batch)
        raise AttributeError(f"{type(self.model).__name__} has no get_text_features/encode_text")

    def encode_sources_timed(self, sources: list[str | Path | Image.Image], batch_size: int) -> TimedEmbeddings:
        torch = self.torch
        outputs: list[np.ndarray] = []
        preprocess_seconds = transfer_seconds = encode_seconds = to_cpu_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, len(sources), batch_size):
                batch_sources = sources[start:start + batch_size]
                t0 = now()
                images = [
                    source.convert("RGB") if isinstance(source, Image.Image) else Image.open(source).convert("RGB")
                    for source in batch_sources
                ]
                batch = self.processor(images=images, return_tensors="pt")
                preprocess_seconds += now() - t0

                t0 = now()
                batch = move_to_device(batch, self.device, dtype=self.dtype)
                sync_device(torch, self.device)
                transfer_seconds += now() - t0

                t0 = now()
                encoded = self._image_features(batch)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                sync_device(torch, self.device)
                encode_seconds += now() - t0

                t0 = now()
                outputs.append(encoded.float().cpu().numpy())
                to_cpu_seconds += now() - t0
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        return TimedEmbeddings(embeddings, timing_dict(len(sources), preprocess_seconds, transfer_seconds, encode_seconds, to_cpu_seconds))

    def encode_texts_timed(self, texts: list[str], batch_size: int) -> TimedEmbeddings:
        torch = self.torch
        outputs: list[np.ndarray] = []
        tokenize_seconds = transfer_seconds = encode_seconds = to_cpu_seconds = 0.0
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                t0 = now()
                model_class = type(self.model).__name__.lower()
                is_siglip = "siglip" in model_class or "siglip" in self.model_id.lower()
                # SigLIP/SigLIP2 text towers are trained with fixed-length padding.
                # Dynamic batch padding can make retrieval nearly random because the
                # pooled text representation depends on the expected max-length layout.
                padding: bool | str = "max_length" if is_siglip else True
                processor_kwargs: dict[str, Any] = {"text": batch_texts, "padding": padding, "truncation": True, "return_tensors": "pt"}
                text_config = getattr(getattr(self.model, "config", None), "text_config", None)
                max_length = getattr(text_config, "max_position_embeddings", None)
                if max_length is None and is_siglip:
                    tokenizer = getattr(self.processor, "tokenizer", None)
                    max_length = getattr(tokenizer, "model_max_length", None)
                    if max_length and max_length > 100_000:
                        max_length = None
                if max_length:
                    processor_kwargs["max_length"] = max_length
                batch = self.processor(**processor_kwargs)
                tokenize_seconds += now() - t0

                t0 = now()
                batch = move_to_device(batch, self.device)
                sync_device(torch, self.device)
                transfer_seconds += now() - t0

                t0 = now()
                encoded = self._text_features(batch)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                sync_device(torch, self.device)
                encode_seconds += now() - t0

                t0 = now()
                outputs.append(encoded.float().cpu().numpy())
                to_cpu_seconds += now() - t0
        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        timing = timing_dict(len(texts), tokenize_seconds, transfer_seconds, encode_seconds, to_cpu_seconds)
        timing["tokenize_cpu_seconds"] = timing.pop("preprocess_cpu_seconds")
        return TimedEmbeddings(embeddings, timing)


def timing_dict(count: int, preprocess: float, transfer: float, encode: float, to_cpu: float) -> dict[str, Any]:
    total = preprocess + transfer + encode + to_cpu
    return {
        "preprocess_cpu_seconds": rounded(preprocess),
        "transfer_to_device_seconds": rounded(transfer),
        "encode_device_seconds": rounded(encode),
        "to_cpu_seconds": rounded(to_cpu),
        "encode_pipeline_seconds": rounded(total),
        "views": int(count),
        "views_per_second_total": rounded(count / max(1e-9, total)),
        "views_per_second_encoder_only": rounded(count / max(1e-9, encode)),
    }


def make_runner(model_arg: str, device: str, dtype: str) -> BaseRunner:
    if model_arg.startswith("openclip:"):
        return TimedOpenClipRunner(model_arg[len("openclip:"):], device)
    if model_arg.startswith("hf:"):
        return TimedHfRunner(model_arg[len("hf:"):], device, dtype=dtype)
    raise ValueError("model must start with openclip: or hf:")


def prepare_sliding(paths: list[str]) -> tuple[list[Image.Image], np.ndarray, float]:
    t0 = now()
    views: list[Image.Image] = []
    offsets = [0]
    for path in paths:
        image = Image.open(path).convert("RGB")
        views.extend(_sliding_square_crops(image))
        offsets.append(len(views))
    return views, np.asarray(offsets, dtype=np.int32), now() - t0


def evaluate_model(args: argparse.Namespace, model_arg: str, annotation_path: Path, language: str) -> dict[str, Any]:
    started = now()
    load_start = now()
    runner = make_runner(model_arg, args.device, args.dtype)
    load_seconds = now() - load_start

    item_ids, image_paths, _items = load_image_items(args.image_manifest)
    queries = build_queries(annotation_path, args.max_queries_per_item, include_captions=False)
    query_texts = [query.query for query in queries]

    text = runner.encode_texts_timed(query_texts, args.text_batch_size)
    center = runner.encode_sources_timed(image_paths, args.batch_size)
    center_scores = text.embeddings @ center.embeddings.T
    counter = [0]
    center_result = _evaluate_scores(
        queries,
        text.embeddings,
        item_ids,
        lambda _query_embedding, matrix=center_scores, counter=counter: matrix.__getitem__(counter.__setitem__(0, counter[0] + 1) or counter[0] - 1),
    )

    sliding_views, offsets, prepare_seconds = prepare_sliding(image_paths)
    sliding = runner.encode_sources_timed(sliding_views, args.batch_size)
    sliding_results: dict[str, Any] = {}
    for aggregate in args.sliding_aggregates:
        strategy = f"sliding_{aggregate}"
        sliding_results[f"{strategy}_center_crop"] = _evaluate_scores(
            queries,
            text.embeddings,
            item_ids,
            lambda query_embedding, name=strategy: _view_score_strategies(
                query_embedding,
                sliding.embeddings,
                offsets,
                len(item_ids),
                prefix="sliding",
            )[name],
        )

    rows = [
        row_for_timing(runner.model_id, language, "center_crop", 0.0, center.timing, center_result["overall"], len(item_ids)),
    ]
    for aggregate in args.sliding_aggregates:
        result = sliding_results[f"sliding_{aggregate}_center_crop"]
        rows.append(row_for_timing(
            runner.model_id,
            language,
            f"sliding_{aggregate}_center_crop",
            prepare_seconds,
            sliding.timing,
            result["overall"],
            len(item_ids),
        ))

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_arg": model_arg,
        "runner_model_id": runner.model_id,
        "device": runner.device,
        "language": language,
        "image_manifest": str(args.image_manifest),
        "image_annotations": str(annotation_path),
        "items": len(item_ids),
        "queries": len(queries),
        "model_load_seconds": rounded(load_seconds),
        "text_timing": text.timing,
        "strategies": {
            "center_crop": center_result,
            **sliding_results,
        },
        "timing_rows": rows,
        "elapsed_seconds": rounded(now() - started),
    }


def precompute_image_embeddings(
    args: argparse.Namespace,
    runner: BaseRunner,
    item_ids: list[str],
    image_paths: list[str],
) -> dict[str, Any]:
    center = runner.encode_sources_timed(image_paths, args.batch_size)
    sliding_views, offsets, prepare_seconds = prepare_sliding(image_paths)
    sliding = runner.encode_sources_timed(sliding_views, args.batch_size)
    return {
        "center": center,
        "sliding": sliding,
        "sliding_offsets": offsets,
        "sliding_prepare_seconds": prepare_seconds,
        "items": len(item_ids),
        "sliding_views": len(sliding_views),
    }


def warmup_runner(args: argparse.Namespace, runner: BaseRunner, image_paths: list[str]) -> None:
    image_count = min(max(1, args.batch_size), len(image_paths))
    if image_count:
        _ = runner.encode_sources_timed(image_paths[:image_count], image_count)
    _ = runner.encode_texts_timed(["warmup text", "预热文本"], 2)


def evaluate_language_with_embeddings(
    args: argparse.Namespace,
    runner: BaseRunner,
    model_arg: str,
    language: str,
    annotation_path: Path,
    item_ids: list[str],
    image_bundle: dict[str, Any],
    model_load_seconds: float,
    image_precompute_seconds: float,
) -> dict[str, Any]:
    started = now()
    queries = build_queries(annotation_path, args.max_queries_per_item, include_captions=False)
    query_texts = [query.query for query in queries]
    text = runner.encode_texts_timed(query_texts, args.text_batch_size)

    center: TimedEmbeddings = image_bundle["center"]
    center_scores = text.embeddings @ center.embeddings.T
    counter = [0]
    center_result = _evaluate_scores(
        queries,
        text.embeddings,
        item_ids,
        lambda _query_embedding, matrix=center_scores, counter=counter: matrix.__getitem__(counter.__setitem__(0, counter[0] + 1) or counter[0] - 1),
    )

    sliding: TimedEmbeddings = image_bundle["sliding"]
    offsets: np.ndarray = image_bundle["sliding_offsets"]
    sliding_results: dict[str, Any] = {}
    for aggregate in args.sliding_aggregates:
        strategy = f"sliding_{aggregate}"
        sliding_results[f"{strategy}_center_crop"] = _evaluate_scores(
            queries,
            text.embeddings,
            item_ids,
            lambda query_embedding, name=strategy: _view_score_strategies(
                query_embedding,
                sliding.embeddings,
                offsets,
                len(item_ids),
                prefix="sliding",
            )[name],
        )

    rows = [
        row_for_timing(runner.model_id, language, "center_crop", 0.0, center.timing, center_result["overall"], len(item_ids)),
    ]
    for aggregate in args.sliding_aggregates:
        result = sliding_results[f"sliding_{aggregate}_center_crop"]
        rows.append(row_for_timing(
            runner.model_id,
            language,
            f"sliding_{aggregate}_center_crop",
            image_bundle["sliding_prepare_seconds"],
            sliding.timing,
            result["overall"],
            len(item_ids),
        ))

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_arg": model_arg,
        "runner_model_id": runner.model_id,
        "device": runner.device,
        "language": language,
        "image_manifest": str(args.image_manifest),
        "image_annotations": str(annotation_path),
        "items": len(item_ids),
        "queries": len(queries),
        "model_load_seconds": rounded(model_load_seconds),
        "image_precompute_seconds": rounded(image_precompute_seconds),
        "text_timing": text.timing,
        "strategies": {
            "center_crop": center_result,
            **sliding_results,
        },
        "timing_rows": rows,
        "elapsed_seconds": rounded(now() - started),
    }


def row_for_timing(
    model: str,
    language: str,
    strategy: str,
    source_prepare_seconds: float,
    timing: dict[str, Any],
    metrics: dict[str, Any],
    item_count: int,
) -> dict[str, Any]:
    row = {
        "model": model,
        "language": language,
        "task": "image",
        "strategy": strategy,
        "items": item_count,
        "source_prepare_seconds": rounded(source_prepare_seconds),
    }
    row.update(timing)
    row["total_index_seconds"] = rounded(source_prepare_seconds + timing.get("encode_pipeline_seconds", 0.0))
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            row[key] = value
    return row


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
    parser = argparse.ArgumentParser(description="Evaluate image/frame retrieval models with center crop and sliding window strategies.")
    parser.add_argument("--model", action="append", required=True, help="openclip:MODEL::PRETRAINED or hf:MODEL_ID")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-queries-per-item", type=int, default=3)
    parser.add_argument("--image-manifest", type=Path, default=Path("eval/visual/image_retrieval/frames.balanced_v2_300.local.json"))
    parser.add_argument("--image-annotations-en", type=Path, default=Path("eval/visual/image_retrieval/auto_annotations.qwen_fallback.image_balanced_v2_300.local.jsonl"))
    parser.add_argument("--image-annotations-zh", type=Path, default=Path("eval/visual/image_retrieval/auto_annotations.qwen_fallback.image_balanced_v2_300.zh.local.jsonl"))
    parser.add_argument("--languages", default="en,zh")
    parser.add_argument("--sliding-aggregates", default="mvp_mix,max", help="comma-separated: mvp_mix,max,top3,mean")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/visual/outputs/model_sweep"))
    parser.add_argument("--run-name", default="image_model_sweep")
    args = parser.parse_args()
    args.sliding_aggregates = [item.strip() for item in args.sliding_aggregates.split(",") if item.strip()]

    annotations = {
        "en": args.image_annotations_en,
        "zh": args.image_annotations_zh,
    }
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    all_summaries = []
    all_rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    item_ids, image_paths, _items = load_image_items(args.image_manifest)
    for model_arg in args.model:
        print(json.dumps({"event": "load_model", "model": model_arg}, ensure_ascii=False))
        load_start = now()
        runner = make_runner(model_arg, args.device, args.dtype)
        model_load_seconds = now() - load_start
        print(json.dumps({"event": "warmup", "model": model_arg}, ensure_ascii=False))
        warmup_runner(args, runner, image_paths)
        print(json.dumps({"event": "encode_images", "model": model_arg}, ensure_ascii=False))
        image_start = now()
        image_bundle = precompute_image_embeddings(args, runner, item_ids, image_paths)
        image_precompute_seconds = now() - image_start
        for language in languages:
            print(json.dumps({"event": "start", "model": model_arg, "language": language}, ensure_ascii=False))
            result = evaluate_language_with_embeddings(
                args,
                runner,
                model_arg,
                language,
                annotations[language],
                item_ids,
                image_bundle,
                model_load_seconds,
                image_precompute_seconds,
            )
            slug = model_label(model_arg)
            json_path = args.out_dir / f"{args.run_name}.{slug}.{language}.json"
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for row in result["timing_rows"]:
                row["json"] = json_path.name
                all_rows.append(row)
            summary = {
                "model": result["runner_model_id"],
                "language": language,
                "json": str(json_path),
                "elapsed_seconds": result["elapsed_seconds"],
            }
            all_summaries.append(summary)
            print(json.dumps({"event": "done", **summary}, ensure_ascii=False))
    csv_path = args.out_dir / f"{args.run_name}.summary.csv"
    write_csv(csv_path, all_rows)
    print(json.dumps({"event": "all_done", "csv": str(csv_path), "runs": all_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
