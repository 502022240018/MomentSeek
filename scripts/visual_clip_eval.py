from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Query:
    query_id: str
    item_id: str
    group_id: str
    query: str
    query_type: str
    source: str


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    pretrained: str

    @property
    def slug(self) -> str:
        value = f"{self.model_name}_{self.pretrained}"
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
        return value.replace("/", "_")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _best_annotation_rows(path: Path) -> list[dict[str, Any]]:
    """Keep one row per item, preferring parsed OK rows over historical failures."""
    best: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        current = best.get(item_id)
        if current is None:
            best[item_id] = row
        elif current.get("status") != "ok" and row.get("status") == "ok":
            best[item_id] = row
        elif current.get("status") == row.get("status"):
            best[item_id] = row
    return list(best.values())


def _query_values(annotation: dict[str, Any], max_queries: int, include_captions: bool) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in annotation.get("suggested_queries") or []:
        if isinstance(item, dict):
            text = str(item.get("query") or "").strip()
            query_type = str(item.get("query_type") or "suggested").strip() or "suggested"
        else:
            text = str(item).strip()
            query_type = "suggested"
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            values.append((text, query_type, "suggested_queries"))
        if len(values) >= max_queries:
            break
    if include_captions:
        for field, query_type in (("caption_en", "caption_en"), ("scene", "scene")):
            text = str(annotation.get(field) or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                values.append((text, query_type, field))
    return values


def build_queries(annotation_path: Path, max_queries_per_item: int, include_captions: bool) -> list[Query]:
    queries: list[Query] = []
    for row in _best_annotation_rows(annotation_path):
        if row.get("status") != "ok":
            continue
        item_id = str(row.get("item_id"))
        annotation = row.get("annotation") or {}
        for local_index, (text, query_type, source) in enumerate(
            _query_values(annotation, max_queries_per_item, include_captions),
            start=1,
        ):
            queries.append(Query(
                query_id=f"{item_id}__q{local_index:02d}",
                item_id=item_id,
                group_id=str(row.get("group_id") or ""),
                query=text,
                query_type=query_type,
                source=source,
            ))
    return queries


class OpenClipRunner:
    def __init__(self, spec: ModelSpec, device: str):
        import open_clip
        import torch

        self.spec = spec
        self.torch = torch
        self.device = self._resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            spec.model_name,
            pretrained=spec.pretrained,
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(spec.model_name)
        self.image_size, self.mean, self.std = self._extract_preprocess_config()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            if device.startswith("npu"):
                self._activate_npu(device)
            return device
        if self.torch.cuda.is_available():
            return "cuda"
        try:
            self._activate_npu("npu:0")
            if getattr(self.torch, "npu").is_available():
                return "npu:0"
        except Exception:
            pass
        return "cpu"

    def _activate_npu(self, device: str) -> None:
        import torch_npu  # noqa: F401

        if hasattr(self.torch, "npu"):
            self.torch.npu.set_device(device)

    def _extract_preprocess_config(self) -> tuple[int, np.ndarray, np.ndarray]:
        image_size = 224
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        for transform in getattr(self.preprocess, "transforms", []):
            name = type(transform).__name__
            if name == "CenterCrop":
                value = transform.size
                image_size = int(value if isinstance(value, int) else min(value))
            elif name == "Normalize":
                mean = np.asarray(transform.mean, dtype=np.float32)
                std = np.asarray(transform.std, dtype=np.float32)
        return image_size, mean, std

    def _letterbox_tensor(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        scale = self.image_size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        background = tuple(int(round(value * 255)) for value in self.mean)
        canvas = Image.new("RGB", (self.image_size, self.image_size), background)
        left = (self.image_size - resized.width) // 2
        top = (self.image_size - resized.height) // 2
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas).astype(np.float32) / 255.0
        array = (array - self.mean) / self.std
        return self.torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def _image_tensor(self, path: str | Path, mode: str):
        image = Image.open(path).convert("RGB")
        if mode == "center_crop":
            return self.preprocess(image)
        if mode == "letterbox":
            return self._letterbox_tensor(image)
        raise ValueError(f"Unknown image preprocess mode: {mode}")

    def encode_images(self, paths: list[str | Path], mode: str, batch_size: int) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start:start + batch_size]
                tensors = [self._image_tensor(path, mode) for path in batch_paths]
                batch = torch.stack(tensors).to(self.device)
                encoded = self.model.encode_image(batch)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def encode_pil_images(self, images: list[Image.Image], mode: str, batch_size: int) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                tensors = [
                    self.preprocess(image.convert("RGB")) if mode == "center_crop" else self._letterbox_tensor(image)
                    for image in batch_images
                ]
                batch = torch.stack(tensors).to(self.device)
                encoded = self.model.encode_image(batch)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                tokens = self.tokenizer(batch_texts).to(self.device)
                encoded = self.model.encode_text(tokens)
                if isinstance(encoded, (tuple, list)):
                    encoded = encoded[0]
                encoded = torch.nn.functional.normalize(encoded, dim=-1)
                outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _metric_summary(ranks: list[int]) -> dict[str, Any]:
    if not ranks:
        return {"queries": 0}
    values = np.asarray(ranks, dtype=np.float32)
    return {
        "queries": int(len(values)),
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "recall_at_20": float(np.mean(values <= 20)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def _rank_positive(scores: np.ndarray, positive_index: int) -> int:
    positive_score = float(scores[positive_index])
    # 1 + number of strictly higher scores. Ties share the best tied rank.
    return int(1 + np.sum(scores > positive_score))


def _evaluate_scores(
    queries: list[Query],
    query_embeddings: np.ndarray,
    item_ids: list[str],
    score_fn,
) -> dict[str, Any]:
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    ranks: list[int] = []
    by_type: dict[str, list[int]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    missing = 0
    for query_index, query in enumerate(queries):
        positive_index = item_to_index.get(query.item_id)
        if positive_index is None:
            missing += 1
            continue
        scores = score_fn(query_embeddings[query_index])
        rank = _rank_positive(scores, positive_index)
        ranks.append(rank)
        by_type[query.query_type].append(rank)
        top_indices = np.argsort(scores)[::-1][:10]
        details.append({
            "query_id": query.query_id,
            "query": query.query,
            "query_type": query.query_type,
            "positive_item_id": query.item_id,
            "rank": rank,
            "positive_score": float(scores[positive_index]),
            "top10": [
                {"item_id": item_ids[int(index)], "score": float(scores[int(index)])}
                for index in top_indices
            ],
        })
    return {
        "overall": _metric_summary(ranks),
        "by_query_type": {
            query_type: _metric_summary(type_ranks)
            for query_type, type_ranks in sorted(by_type.items())
        },
        "missing_positive_queries": missing,
        "details": details,
    }


def load_image_items(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    payload = _read_json(path)
    frames = payload["frames"]
    return (
        [str(item["image_id"]) for item in frames],
        [str(item["path"]) for item in frames],
        frames,
    )


def load_sequence_items(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    payload = _read_json(path)
    sheets = payload["sheets"]
    return (
        [str(item["sheet_id"]) for item in sheets],
        [str(item["path"]) for item in sheets],
        sheets,
    )


def _split_sheet_cells(sheet: dict[str, Any]) -> list[Image.Image]:
    path = Path(str(sheet["path"]))
    image = Image.open(path).convert("RGB")
    sample_count = len(sheet.get("sample_times") or [])
    if sample_count <= 0:
        sample_count = 1
    cell_width = int(sheet.get("cell_width") or 512)
    cell_height = int(sheet.get("cell_height") or 288)
    rows = 2 if sample_count > 3 else 1
    cols = int(math.ceil(sample_count / rows))
    cells: list[Image.Image] = []
    for index in range(sample_count):
        row = index // cols
        col = index % cols
        left = col * cell_width
        top = row * cell_height
        cells.append(image.crop((left, top, left + cell_width, top + cell_height)))
    return cells


def _crop_starts(length: int, window: int, max_positions: int = 5) -> list[int]:
    if length <= window:
        return [0]
    aspect = length / max(1, window)
    if aspect < 1.15:
        count = 1
    elif aspect < 2.0:
        count = 3
    else:
        count = max_positions
    count = min(max_positions, count)
    if count <= 1:
        return [(length - window) // 2]
    span = length - window
    starts = [round(index * span / (count - 1)) for index in range(count)]
    # Keep order stable while removing duplicates from very small spans.
    return list(dict.fromkeys(int(value) for value in starts))


def _sliding_square_crops(image: Image.Image, max_positions: int = 5) -> list[Image.Image]:
    """Return square spatial windows over the long image axis.

    CLIP's default preprocessing resizes the short side and center-crops to a
    square. On 16:9 or wider frames this discards left/right content, which is
    exactly where subtitles, logos, scoreboards, and small objects often live.
    This view generator keeps the CLIP-native square input but evaluates several
    square crops along the long axis:

    - near-square image: one center crop;
    - 16:9-ish image: left / center / right;
    - very wide image: up to five horizontal crops;
    - tall image: analogous vertical crops.
    """
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    if width >= height:
        return [
            image.crop((left, 0, left + side, side))
            for left in _crop_starts(width, side, max_positions=max_positions)
        ]
    return [
        image.crop((0, top, side, top + side))
        for top in _crop_starts(height, side, max_positions=max_positions)
    ]


def _cache_path(cache_dir: Path, prefix: str, spec: ModelSpec, mode: str) -> Path:
    return cache_dir / f"{prefix}.{spec.slug}.{mode}.npz"


def _load_embeddings_cache(path: Path, expected_ids: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    ids = [str(value) for value in data["item_ids"]]
    if ids != expected_ids:
        return None
    return data["embeddings"].astype(np.float32)


def _save_embeddings_cache(path: Path, item_ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        item_ids=np.asarray(item_ids, dtype="U256"),
        embeddings=embeddings.astype(np.float32),
    )


def evaluate_image_task(
    runner: OpenClipRunner,
    image_manifest: Path,
    image_annotations: Path,
    max_queries_per_item: int,
    include_captions: bool,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    item_ids, image_paths, items = load_image_items(image_manifest)
    queries = build_queries(image_annotations, max_queries_per_item, include_captions)
    query_embeddings = runner.encode_texts([query.query for query in queries], batch_size)
    strategies: dict[str, Any] = {}
    for mode in ("center_crop", "letterbox"):
        cache_path = _cache_path(cache_dir, "image_items", runner.spec, mode)
        embeddings = _load_embeddings_cache(cache_path, item_ids) if use_cache else None
        if embeddings is None:
            embeddings = runner.encode_images(image_paths, mode, batch_size)
            if use_cache:
                _save_embeddings_cache(cache_path, item_ids, embeddings)
        scores_matrix = query_embeddings @ embeddings.T
        strategies[mode] = _evaluate_scores(
            queries,
            query_embeddings,
            item_ids,
            lambda query_embedding, matrix=scores_matrix, counter=[0]: _score_from_matrix(matrix, counter),
        )
    view_embeddings, _view_item_indices, view_offsets = _spatial_view_embeddings_for_paths(
        runner,
        item_ids,
        image_paths,
        "image_spatial_views",
        batch_size,
        cache_dir,
        use_cache,
    )
    for strategy_name in ("sliding_mean", "sliding_max", "sliding_top3", "sliding_mvp_mix"):
        strategies[f"{strategy_name}_center_crop"] = _evaluate_scores(
            queries,
            query_embeddings,
            item_ids,
            lambda query_embedding, name=strategy_name, views=view_embeddings, offsets=view_offsets:
                _view_score_strategies(query_embedding, views, offsets, len(item_ids), prefix="sliding")[name],
        )
    return {
        "manifest": str(image_manifest),
        "annotations": str(image_annotations),
        "items": len(item_ids),
        "queries": len(queries),
        "query_type_counts": dict(Counter(query.query_type for query in queries)),
        "strategies": strategies,
        "item_sample": items[:3],
    }


def _score_from_matrix(matrix: np.ndarray, counter: list[int]) -> np.ndarray:
    index = counter[0]
    counter[0] += 1
    return matrix[index]


def _load_sequence_cell_cache(path: Path, expected_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    ids = [str(value) for value in data["item_ids"]]
    if ids != expected_ids:
        return None
    return (
        data["cell_embeddings"].astype(np.float32),
        data["cell_item_indices"].astype(np.int32),
        data["cell_offsets"].astype(np.int32),
    )


def _save_sequence_cell_cache(
    path: Path,
    item_ids: list[str],
    cell_embeddings: np.ndarray,
    cell_item_indices: np.ndarray,
    cell_offsets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        item_ids=np.asarray(item_ids, dtype="U256"),
        cell_embeddings=cell_embeddings.astype(np.float32),
        cell_item_indices=cell_item_indices.astype(np.int32),
        cell_offsets=cell_offsets.astype(np.int32),
    )


def _load_view_cache(path: Path, expected_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    ids = [str(value) for value in data["item_ids"]]
    if ids != expected_ids:
        return None
    return (
        data["view_embeddings"].astype(np.float32),
        data["view_item_indices"].astype(np.int32),
        data["view_offsets"].astype(np.int32),
    )


def _save_view_cache(
    path: Path,
    item_ids: list[str],
    view_embeddings: np.ndarray,
    view_item_indices: np.ndarray,
    view_offsets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        item_ids=np.asarray(item_ids, dtype="U256"),
        view_embeddings=view_embeddings.astype(np.float32),
        view_item_indices=view_item_indices.astype(np.int32),
        view_offsets=view_offsets.astype(np.int32),
    )


def _spatial_view_embeddings_for_paths(
    runner: OpenClipRunner,
    item_ids: list[str],
    paths: list[str],
    cache_prefix: str,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _cache_path(cache_dir, cache_prefix, runner.spec, "sliding_square_center_crop")
    cached = _load_view_cache(cache_path, item_ids) if use_cache else None
    if cached is not None:
        return cached
    views: list[Image.Image] = []
    view_item_indices: list[int] = []
    offsets = [0]
    for item_index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        crops = _sliding_square_crops(image)
        views.extend(crops)
        view_item_indices.extend([item_index] * len(crops))
        offsets.append(len(views))
    embeddings = runner.encode_pil_images(views, "center_crop", batch_size)
    result = (
        embeddings,
        np.asarray(view_item_indices, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    if use_cache:
        _save_view_cache(cache_path, item_ids, *result)
    return result


def _sequence_cell_embeddings(
    runner: OpenClipRunner,
    item_ids: list[str],
    sheets: list[dict[str, Any]],
    mode: str,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _cache_path(cache_dir, "sequence_cells", runner.spec, mode)
    cached = _load_sequence_cell_cache(cache_path, item_ids) if use_cache else None
    if cached is not None:
        return cached
    all_cells: list[Image.Image] = []
    cell_item_indices: list[int] = []
    offsets = [0]
    for item_index, sheet in enumerate(sheets):
        cells = _split_sheet_cells(sheet)
        all_cells.extend(cells)
        cell_item_indices.extend([item_index] * len(cells))
        offsets.append(len(all_cells))
    embeddings = runner.encode_pil_images(all_cells, mode, batch_size)
    result = (
        embeddings,
        np.asarray(cell_item_indices, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    if use_cache:
        _save_sequence_cell_cache(cache_path, item_ids, *result)
    return result


def _sequence_cell_sliding_embeddings(
    runner: OpenClipRunner,
    item_ids: list[str],
    sheets: list[dict[str, Any]],
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _cache_path(cache_dir, "sequence_cell_spatial_views", runner.spec, "sliding_square_center_crop")
    cached = _load_view_cache(cache_path, item_ids) if use_cache else None
    if cached is not None:
        return cached
    views: list[Image.Image] = []
    view_item_indices: list[int] = []
    offsets = [0]
    for item_index, sheet in enumerate(sheets):
        cells = _split_sheet_cells(sheet)
        for cell in cells:
            crops = _sliding_square_crops(cell)
            views.extend(crops)
            view_item_indices.extend([item_index] * len(crops))
        offsets.append(len(views))
    embeddings = runner.encode_pil_images(views, "center_crop", batch_size)
    result = (
        embeddings,
        np.asarray(view_item_indices, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
    )
    if use_cache:
        _save_view_cache(cache_path, item_ids, *result)
    return result


def _cell_score_strategies(
    query_embedding: np.ndarray,
    cell_embeddings: np.ndarray,
    cell_offsets: np.ndarray,
    item_count: int,
) -> dict[str, np.ndarray]:
    cell_scores = cell_embeddings @ query_embedding
    mean_scores = np.zeros(item_count, dtype=np.float32)
    max_scores = np.zeros(item_count, dtype=np.float32)
    top3_scores = np.zeros(item_count, dtype=np.float32)
    for item_index in range(item_count):
        start, end = int(cell_offsets[item_index]), int(cell_offsets[item_index + 1])
        values = cell_scores[start:end]
        if not len(values):
            continue
        mean_scores[item_index] = float(np.mean(values))
        ordered = np.sort(values)[::-1]
        max_scores[item_index] = float(ordered[0])
        top3_scores[item_index] = float(np.mean(ordered[:min(3, len(ordered))]))
    return {
        "cells_mean": mean_scores,
        "cells_max": max_scores,
        "cells_top3": top3_scores,
        "cells_mvp_mix": (0.65 * max_scores) + (0.25 * top3_scores) + (0.10 * mean_scores),
    }


def _view_score_strategies(
    query_embedding: np.ndarray,
    view_embeddings: np.ndarray,
    view_offsets: np.ndarray,
    item_count: int,
    prefix: str = "sliding",
) -> dict[str, np.ndarray]:
    view_scores = view_embeddings @ query_embedding
    mean_scores = np.zeros(item_count, dtype=np.float32)
    max_scores = np.zeros(item_count, dtype=np.float32)
    top3_scores = np.zeros(item_count, dtype=np.float32)
    for item_index in range(item_count):
        start, end = int(view_offsets[item_index]), int(view_offsets[item_index + 1])
        values = view_scores[start:end]
        if not len(values):
            continue
        mean_scores[item_index] = float(np.mean(values))
        ordered = np.sort(values)[::-1]
        max_scores[item_index] = float(ordered[0])
        top3_scores[item_index] = float(np.mean(ordered[:min(3, len(ordered))]))
    return {
        f"{prefix}_mean": mean_scores,
        f"{prefix}_max": max_scores,
        f"{prefix}_top3": top3_scores,
        f"{prefix}_mvp_mix": (0.65 * max_scores) + (0.25 * top3_scores) + (0.10 * mean_scores),
    }


def evaluate_sequence_task(
    runner: OpenClipRunner,
    sequence_manifest: Path,
    sequence_annotations: Path,
    max_queries_per_item: int,
    include_captions: bool,
    batch_size: int,
    cache_dir: Path,
    use_cache: bool,
) -> dict[str, Any]:
    item_ids, sheet_paths, sheets = load_sequence_items(sequence_manifest)
    queries = build_queries(sequence_annotations, max_queries_per_item, include_captions)
    query_embeddings = runner.encode_texts([query.query for query in queries], batch_size)
    strategies: dict[str, Any] = {}

    for mode in ("center_crop", "letterbox"):
        sheet_cache_path = _cache_path(cache_dir, "sequence_sheets", runner.spec, mode)
        sheet_embeddings = _load_embeddings_cache(sheet_cache_path, item_ids) if use_cache else None
        if sheet_embeddings is None:
            sheet_embeddings = runner.encode_images(sheet_paths, mode, batch_size)
            if use_cache:
                _save_embeddings_cache(sheet_cache_path, item_ids, sheet_embeddings)
        sheet_scores_matrix = query_embeddings @ sheet_embeddings.T
        strategies[f"sheet_whole_{mode}"] = _evaluate_scores(
            queries,
            query_embeddings,
            item_ids,
            lambda query_embedding, matrix=sheet_scores_matrix, counter=[0]: _score_from_matrix(matrix, counter),
        )

        cell_embeddings, _cell_item_indices, cell_offsets = _sequence_cell_embeddings(
            runner,
            item_ids,
            sheets,
            mode,
            batch_size,
            cache_dir,
            use_cache,
        )
        for strategy_name in ("cells_mean", "cells_max", "cells_top3", "cells_mvp_mix"):
            strategies[f"{strategy_name}_{mode}"] = _evaluate_scores(
                queries,
                query_embeddings,
                item_ids,
                lambda query_embedding, name=strategy_name, cells=cell_embeddings, offsets=cell_offsets:
                    _cell_score_strategies(query_embedding, cells, offsets, len(item_ids))[name],
            )

    sliding_view_embeddings, _view_item_indices, sliding_view_offsets = _sequence_cell_sliding_embeddings(
        runner,
        item_ids,
        sheets,
        batch_size,
        cache_dir,
        use_cache,
    )
    for strategy_name in ("cells_sliding_mean", "cells_sliding_max", "cells_sliding_top3", "cells_sliding_mvp_mix"):
        strategies[f"{strategy_name}_center_crop"] = _evaluate_scores(
            queries,
            query_embeddings,
            item_ids,
            lambda query_embedding, name=strategy_name, views=sliding_view_embeddings, offsets=sliding_view_offsets:
                _view_score_strategies(query_embedding, views, offsets, len(item_ids), prefix="cells_sliding")[name],
        )

    sample_counts = Counter(len(sheet.get("sample_times") or []) for sheet in sheets)
    return {
        "manifest": str(sequence_manifest),
        "annotations": str(sequence_annotations),
        "items": len(item_ids),
        "queries": len(queries),
        "query_type_counts": dict(Counter(query.query_type for query in queries)),
        "sample_time_count_summary": {str(key): value for key, value in sorted(sample_counts.items())},
        "strategies": strategies,
        "item_sample": sheets[:3],
    }


def _strategy_rows(task_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for strategy_name, result in task_result.get("strategies", {}).items():
        overall = result.get("overall", {})
        rows.append({
            "strategy": strategy_name,
            **overall,
        })
    return sorted(rows, key=lambda row: (row.get("recall_at_10", 0), row.get("mrr", 0)), reverse=True)


def _format_pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def write_markdown_report(result: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Visual CLIP evaluation report",
        "",
        f"- Created at: `{result['created_at']}`",
        f"- Device: `{result['device']}`",
        f"- Model: `{result['model']['model_name']}` / `{result['model']['pretrained']}`",
        f"- Max queries per item: `{result['max_queries_per_item']}`",
        f"- Include captions: `{result['include_captions']}`",
        "",
        "> 注意：这版 query 来自 VLM 自动标注的 suggested_queries，所以适合做模型/预处理/聚合策略的横向比较；绝对分数仍需要人工审核后的 query 集确认。",
        "",
    ]
    for task_name in ("image", "sequence"):
        task = result.get(task_name)
        if not task:
            continue
        lines += [
            f"## {task_name}",
            "",
            f"- Items: `{task['items']}`",
            f"- Queries: `{task['queries']}`",
            f"- Query types: `{task['query_type_counts']}`",
            "",
            "| strategy | R@1 | R@5 | R@10 | R@20 | MRR | median rank | mean rank |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in _strategy_rows(task):
            lines.append(
                "| {strategy} | {r1} | {r5} | {r10} | {r20} | {mrr:.3f} | {median:.1f} | {mean:.1f} |".format(
                    strategy=row["strategy"],
                    r1=_format_pct(row.get("recall_at_1")),
                    r5=_format_pct(row.get("recall_at_5")),
                    r10=_format_pct(row.get("recall_at_10")),
                    r20=_format_pct(row.get("recall_at_20")),
                    mrr=float(row.get("mrr", 0)),
                    median=float(row.get("median_rank", 0)),
                    mean=float(row.get("mean_rank", 0)),
                )
            )
        lines += ["", "### By query type", ""]
        for strategy_name, strategy_result in task.get("strategies", {}).items():
            lines += [
                f"#### {strategy_name}",
                "",
                "| query_type | queries | R@1 | R@5 | R@10 | MRR | median rank |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for query_type, metrics in sorted(strategy_result.get("by_query_type", {}).items()):
                lines.append(
                    "| {qt} | {queries} | {r1} | {r5} | {r10} | {mrr:.3f} | {median:.1f} |".format(
                        qt=query_type,
                        queries=metrics.get("queries", 0),
                        r1=_format_pct(metrics.get("recall_at_1")),
                        r5=_format_pct(metrics.get("recall_at_5")),
                        r10=_format_pct(metrics.get("recall_at_10")),
                        mrr=float(metrics.get("mrr", 0)),
                        median=float(metrics.get("median_rank", 0)),
                    )
                )
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_model_spec(value: str) -> ModelSpec:
    if "::" in value:
        model_name, pretrained = value.split("::", 1)
    else:
        model_name, pretrained = value, "openai"
    return ModelSpec(model_name.strip(), pretrained.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP variants on the MomentSeek visual eval set.")
    parser.add_argument("--task", choices=["all", "image", "sequence"], default="all")
    parser.add_argument("--model-spec", action="append", default=None, help="OpenCLIP model spec, e.g. ViT-B-32::openai")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-queries-per-item", type=int, default=3)
    parser.add_argument("--include-captions", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("runtime/eval/visual/clip_eval_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--image-manifest", type=Path, default=Path("eval/visual/image_retrieval/frames.balanced_v2_300.local.json"))
    parser.add_argument("--image-annotations", type=Path, default=Path("eval/visual/image_retrieval/auto_annotations.qwen_fallback.image_balanced_v2_300.local.jsonl"))
    parser.add_argument("--sequence-manifest", type=Path, default=Path("eval/visual/sequence_retrieval/contact_sheets.balanced_v2_200.2fps_hq.local.json"))
    parser.add_argument("--sequence-annotations", type=Path, default=Path("eval/visual/sequence_retrieval/auto_annotations.qwen_fallback.sequence_balanced_v2_200_2fps_hq.local.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/visual/outputs"))
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    model_specs = [parse_model_spec(value) for value in (args.model_spec or ["ViT-B-32::openai"])]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_cache = not args.no_cache
    summaries = []
    for spec in model_specs:
        started = time.perf_counter()
        print(json.dumps({"event": "load_model", "model": spec.model_name, "pretrained": spec.pretrained}, ensure_ascii=False))
        runner = OpenClipRunner(spec, args.device)
        result: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "task": args.task,
            "device": runner.device,
            "model": {"model_name": spec.model_name, "pretrained": spec.pretrained, "slug": spec.slug},
            "max_queries_per_item": args.max_queries_per_item,
            "include_captions": bool(args.include_captions),
            "batch_size": args.batch_size,
            "use_cache": use_cache,
        }
        if args.task in {"all", "image"}:
            print(json.dumps({"event": "evaluate_image", "model": spec.slug}, ensure_ascii=False))
            result["image"] = evaluate_image_task(
                runner,
                args.image_manifest,
                args.image_annotations,
                args.max_queries_per_item,
                args.include_captions,
                args.batch_size,
                args.cache_dir,
                use_cache,
            )
        if args.task in {"all", "sequence"}:
            print(json.dumps({"event": "evaluate_sequence", "model": spec.slug}, ensure_ascii=False))
            result["sequence"] = evaluate_sequence_task(
                runner,
                args.sequence_manifest,
                args.sequence_annotations,
                args.max_queries_per_item,
                args.include_captions,
                args.batch_size,
                args.cache_dir,
                use_cache,
            )
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        run_label = args.run_name or f"{args.task}_{spec.slug}_{timestamp}"
        safe_run_label = re.sub(r"[^A-Za-z0-9._-]+", "_", run_label).strip("_")
        json_path = args.out_dir / f"clip_eval_{safe_run_label}.local.json"
        md_path = args.out_dir / f"clip_eval_{safe_run_label}.local.md"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown_report(result, md_path)
        summaries.append({"model": spec.slug, "json": str(json_path), "markdown": str(md_path), "elapsed_seconds": result["elapsed_seconds"]})
        print(json.dumps({"event": "done", **summaries[-1]}, ensure_ascii=False, indent=2))
    print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
