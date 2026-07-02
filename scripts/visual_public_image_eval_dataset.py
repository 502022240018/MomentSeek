from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image


OPEN_IMAGES_BBOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
OPEN_IMAGES_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
OPEN_IMAGES_METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"

SMALL_ITEM_LABELS = {
    "Backpack",
    "Bag",
    "Ball",
    "Baseball bat",
    "Baseball glove",
    "Bicycle helmet",
    "Book",
    "Bottle",
    "Bowl",
    "Camera",
    "Cello",
    "Clock",
    "Coffee cup",
    "Computer keyboard",
    "Computer mouse",
    "Cup",
    "Dice",
    "Digital clock",
    "Doll",
    "Drinking straw",
    "Fedora",
    "Flashlight",
    "Flowerpot",
    "Football",
    "Football helmet",
    "Glasses",
    "Goggles",
    "Guitar",
    "Handbag",
    "Helmet",
    "Ipod",
    "Kitchen knife",
    "Laptop",
    "Mobile phone",
    "Mug",
    "Paddle",
    "Pen",
    "Pencil case",
    "Plate",
    "Remote control",
    "Rugby ball",
    "Sandal",
    "Skateboard",
    "Ski pole",
    "Spoon",
    "Suitcase",
    "Tablet computer",
    "Tennis racket",
    "Toothbrush",
    "Toy",
    "Traffic light",
    "Umbrella",
    "Vehicle registration plate",
    "Watch",
    "Wine glass",
}

EXCLUDED_QUERY_LABELS = {
    "Boy",
    "Building",
    "Clothing",
    "Dress",
    "Footwear",
    "Girl",
    "Human arm",
    "Human beard",
    "Human body",
    "Human ear",
    "Human eye",
    "Human face",
    "Human foot",
    "Human hair",
    "Human hand",
    "Human head",
    "Human leg",
    "Human mouth",
    "Human nose",
    "Man",
    "Mammal",
    "Person",
    "Skyscraper",
    "Tower",
    "Tree",
    "Window",
    "Woman",
}

TINY_OBJECT_THRESHOLD = 0.006


@dataclass(frozen=True)
class Box:
    image_id: str
    label_id: str
    label: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def area_ratio(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def edge_sides(self, margin: float) -> list[str]:
        sides: list[str] = []
        if self.xmin <= margin:
            sides.append("left")
        if self.xmax >= 1.0 - margin:
            sides.append("right")
        if self.ymin <= margin:
            sides.append("top")
        if self.ymax >= 1.0 - margin:
            sides.append("bottom")
        return sides


def _download_file(url: str, path: Path, retries: int = 4) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "MomentSeek-public-eval/1.0"}) as response:
                response.raise_for_status()
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                tmp_path.replace(path)
                return
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _read_classes(path: Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                classes[row[0]] = row[1]
    return classes


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("ImageID")
            if image_id:
                metadata[image_id] = row
    return metadata


def _read_boxes(path: Path, classes: dict[str, str]) -> dict[str, list[Box]]:
    boxes_by_image: dict[str, list[Box]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Confidence") != "1":
                continue
            if row.get("IsGroupOf") == "1" or row.get("IsDepiction") == "1" or row.get("IsInside") == "1":
                continue
            label_id = str(row.get("LabelName") or "")
            label = classes.get(label_id)
            if not label:
                continue
            try:
                box = Box(
                    image_id=str(row["ImageID"]),
                    label_id=label_id,
                    label=label,
                    xmin=float(row["XMin"]),
                    xmax=float(row["XMax"]),
                    ymin=float(row["YMin"]),
                    ymax=float(row["YMax"]),
                )
            except Exception:
                continue
            boxes_by_image[box.image_id].append(box)
    return boxes_by_image


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value or "item"


def _label_query(label: str) -> str:
    label = label.strip()
    if not label:
        return "an object"
    article = "an" if label[0].lower() in "aeiou" else "a"
    return f"{article} {label.lower()}"


def _is_eval_label(label: str) -> bool:
    return label in SMALL_ITEM_LABELS and label not in EXCLUDED_QUERY_LABELS


def _location_phrase(box: Box) -> str:
    x_center = (box.xmin + box.xmax) / 2
    y_center = (box.ymin + box.ymax) / 2
    horizontal = "left" if x_center < 0.33 else "right" if x_center > 0.67 else "center"
    vertical = "upper" if y_center < 0.33 else "lower" if y_center > 0.67 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "near the center"
    if horizontal == "center":
        return f"in the {vertical} area"
    if vertical == "middle":
        return f"in the {horizontal} area"
    return f"in the {vertical} {horizontal} area"


def _download_image(
    image_id: str,
    meta: dict[str, str],
    out_path: Path,
    image_source: str,
    max_bytes: int = 24_000_000,
    max_seconds: float = 14.0,
) -> tuple[int, int, str] | None:
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with Image.open(out_path) as image:
                return image.size[0], image.size[1], "cache"
        except Exception:
            out_path.unlink(missing_ok=True)
    s3_url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
    original_url = meta.get("OriginalURL") or ""
    thumbnail_url = meta.get("Thumbnail300KURL") or ""
    if image_source == "hd":
        sources = [("original_url", original_url), ("open_images_s3", s3_url)]
    elif image_source == "thumbnail":
        sources = [("thumbnail300k", thumbnail_url)]
    else:
        sources = [("open_images_s3", s3_url), ("original_url", original_url), ("thumbnail300k", thumbnail_url)]
    for source_name, url in sources:
        if not url:
            continue
        try:
            with requests.get(url, stream=True, timeout=(3, 5), headers={"User-Agent": "Mozilla/5.0 MomentSeek-public-eval/1.0"}) as response:
                if response.status_code >= 400:
                    continue
                content_type = response.headers.get("content-type", "")
                if "image" not in content_type:
                    continue
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                total = 0
                started = time.monotonic()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=512 * 1024):
                        if time.monotonic() - started > max_seconds:
                            raise RuntimeError("image download too slow")
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise RuntimeError("image too large")
                        handle.write(chunk)
                with Image.open(tmp_path) as image:
                    image.verify()
                tmp_path.replace(out_path)
                with Image.open(out_path) as image:
                    return image.size[0], image.size[1], source_name
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[name-defined]
            except Exception:
                pass
    return None


def _candidate_score(
    image_id: str,
    boxes: list[Box],
    small_threshold: float,
    tiny_threshold: float,
    edge_margin: float,
) -> tuple[int, int, int, int]:
    eval_boxes = [box for box in boxes if _is_eval_label(box.label)]
    small_count = sum(1 for box in eval_boxes if box.area_ratio <= small_threshold)
    tiny_count = sum(1 for box in eval_boxes if box.area_ratio <= tiny_threshold)
    edge_count = sum(1 for box in eval_boxes if box.area_ratio <= small_threshold and box.edge_sides(edge_margin))
    label_count = len({box.label for box in eval_boxes})
    return (tiny_count, small_count + edge_count, label_count, -len(image_id))


def _pick_images(
    boxes_by_image: dict[str, list[Box]],
    metadata: dict[str, dict[str, str]],
    max_images: int,
    seed: int,
    small_threshold: float,
    tiny_threshold: float,
    edge_margin: float,
    candidate_multiplier: int,
) -> list[str]:
    rng = random.Random(seed)
    candidates = [
        image_id for image_id, boxes in boxes_by_image.items()
        if image_id in metadata and boxes
        and (
            any(_is_eval_label(box.label) and box.area_ratio <= small_threshold for box in boxes)
        )
    ]
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda image_id: _candidate_score(
            image_id,
            boxes_by_image[image_id],
            small_threshold,
            tiny_threshold,
            edge_margin,
        ),
        reverse=True,
    )
    return candidates[: max_images * candidate_multiplier]


def _add_query(
    query_map: dict[str, dict[str, Any]],
    query_type: str,
    query: str,
    image_id: str,
    source: str,
    label: str,
    notes: str,
) -> None:
    query_id = f"{query_type}__{_slug(query).lower()}"
    row = query_map.setdefault(query_id, {
        "query_id": query_id,
        "query": query,
        "language": "en",
        "query_type": query_type,
        "source": source,
        "label": label,
        "positive_image_ids": [],
        "hard_negative_image_ids": [],
        "notes": notes,
    })
    positives = row["positive_image_ids"]
    if image_id not in positives:
        positives.append(image_id)


def build_open_images_subset(
    *,
    out_root: Path,
    frames_out: Path,
    queries_out: Path,
    examples_out: Path,
    max_images: int,
    max_queries_per_type: int,
    seed: int,
    small_threshold: float,
    tiny_threshold: float,
    edge_margin: float,
    min_side: int,
    min_width: int,
    min_height: int,
    image_source: str,
    candidate_multiplier: int,
    max_workers: int,
    batch_candidates: int,
) -> None:
    source_dir = out_root / "source"
    bbox_path = source_dir / "validation-annotations-bbox.csv"
    classes_path = source_dir / "class-descriptions-boxable.csv"
    metadata_path = source_dir / "validation-images-with-rotation.csv"
    _download_file(OPEN_IMAGES_BBOX_URL, bbox_path)
    _download_file(OPEN_IMAGES_CLASSES_URL, classes_path)
    _download_file(OPEN_IMAGES_METADATA_URL, metadata_path)

    classes = _read_classes(classes_path)
    metadata = _read_metadata(metadata_path)
    boxes_by_image = _read_boxes(bbox_path, classes)
    candidate_ids = _pick_images(
        boxes_by_image,
        metadata,
        max_images,
        seed,
        small_threshold,
        tiny_threshold,
        edge_margin,
        candidate_multiplier,
    )

    frames: list[dict[str, Any]] = []
    selected_boxes: dict[str, list[Box]] = {}
    image_dir = out_root / "images" / "open_images_validation"
    image_dir.mkdir(parents=True, exist_ok=True)

    def download_candidate(image_id: str) -> tuple[dict[str, Any], list[Box]] | None:
        meta = metadata.get(image_id) or {}
        image_path = image_dir / f"{image_id}.jpg"
        downloaded = _download_image(image_id, meta, image_path, image_source=image_source)
        if not downloaded:
            return None
        width, height, source_name = downloaded
        if min(width, height) < min_side or width < min_width or height < min_height:
            image_path.unlink(missing_ok=True)
            if source_name == "cache":
                downloaded = _download_image(image_id, meta, image_path, image_source=image_source)
                if not downloaded:
                    return None
                width, height, source_name = downloaded
            if min(width, height) < min_side or width < min_width or height < min_height:
                image_path.unlink(missing_ok=True)
                return None
        rel_path = image_path.as_posix()
        frame = {
            "image_id": f"open_images_val_{image_id}",
            "group_id": "open_images_validation",
            "variant_id": "open_images_validation",
            "path": rel_path,
            "width": width,
            "height": height,
            "resolution_label": f"{height}p",
            "view_type": "public_image",
            "source_dataset": "Open Images validation",
            "source_image_id": image_id,
            "license": meta.get("License"),
            "source_url": meta.get("OriginalURL"),
            "thumbnail_url": meta.get("Thumbnail300KURL"),
            "download_source": source_name,
        }
        return frame, boxes_by_image[image_id]

    max_candidates = min(len(candidate_ids), max_images * candidate_multiplier)
    attempted = 0
    for start in range(0, max_candidates, batch_candidates):
        if len(frames) >= max_images:
            break
        batch = candidate_ids[start:start + batch_candidates]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(download_candidate, image_id): image_id
                for image_id in batch
            }
            for future in as_completed(future_to_id):
                attempted += 1
                result = future.result()
                if result:
                    frame, boxes = result
                    frames.append(frame)
                    selected_boxes[str(frame["image_id"])] = boxes
                    if len(frames) % 10 == 0:
                        print(f"collected {len(frames)} HD public images after {attempted} attempts...", flush=True)
                if attempted % 100 == 0:
                    print(f"attempted {attempted} candidates; collected {len(frames)} HD images...", flush=True)
                if len(frames) >= max_images:
                    for pending in future_to_id:
                        pending.cancel()
                    break

    query_map: dict[str, dict[str, Any]] = {}
    for image_id, boxes in selected_boxes.items():
        for box in boxes:
            if not _is_eval_label(box.label):
                continue
            if box.area_ratio <= small_threshold:
                location = _location_phrase(box)
                _add_query(
                    query_map,
                    "small_object",
                    f"a small {box.label.lower()} {location}",
                    image_id,
                    "open_images_bbox",
                    box.label,
                    f"bbox area ratio <= {small_threshold:.3f}; actual={box.area_ratio:.4f}",
                )
                if box.area_ratio <= tiny_threshold:
                    _add_query(
                        query_map,
                        "tiny_object",
                        f"a tiny {box.label.lower()} {location}",
                        image_id,
                        "open_images_bbox",
                        box.label,
                        f"bbox area ratio <= {tiny_threshold:.3f}; actual={box.area_ratio:.4f}",
                    )
                for side in box.edge_sides(edge_margin):
                    _add_query(
                        query_map,
                        "edge_small_object",
                        f"a small {box.label.lower()} near the {side} edge of the image",
                        image_id,
                        "open_images_bbox",
                        box.label,
                        f"small bbox touches {side} edge within margin {edge_margin:.2f}; area={box.area_ratio:.4f}",
                    )

    queries_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_map.values():
        queries_by_type[str(row["query_type"])].append(row)
    rng = random.Random(seed)
    queries: list[dict[str, Any]] = []
    for query_type, rows in sorted(queries_by_type.items()):
        rows.sort(key=lambda row: (-len(row["positive_image_ids"]), row["query"]))
        head = rows[: max_queries_per_type * 2]
        rng.shuffle(head)
        queries.extend(sorted(head[:max_queries_per_type], key=lambda row: row["query_id"]))

    frames_payload = {
        "schema_version": 1,
        "created_for": "MomentSeek public HD small-object retrieval evaluation v1",
        "source_datasets": [
            {
                "name": "Open Images validation",
                "bbox_annotations": OPEN_IMAGES_BBOX_URL,
                "class_descriptions": OPEN_IMAGES_CLASSES_URL,
                "image_metadata": OPEN_IMAGES_METADATA_URL,
            }
        ],
        "sampling": {
            "seed": seed,
            "max_images": max_images,
            "small_threshold": small_threshold,
            "tiny_threshold": tiny_threshold,
            "edge_margin": edge_margin,
            "min_side": min_side,
            "min_width": min_width,
            "min_height": min_height,
            "image_source": image_source,
            "candidate_multiplier": candidate_multiplier,
            "download_attempts": attempted,
            "max_workers": max_workers,
            "batch_candidates": batch_candidates,
        },
        "frames": frames,
    }
    frames_out.parent.mkdir(parents=True, exist_ok=True)
    frames_out.write_text(json.dumps(frames_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    queries_out.parent.mkdir(parents=True, exist_ok=True)
    with queries_out.open("w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_examples_html(examples_out, frames, queries, selected_boxes)


def _img_src(out_path: Path, image_path: Path) -> str:
    return Path(__import__("os").path.relpath(image_path, out_path.parent)).as_posix()


def _write_examples_html(path: Path, frames: list[dict[str, Any]], queries: list[dict[str, Any]], boxes_by_frame: dict[str, list[Box]]) -> None:
    frame_by_id = {str(frame["image_id"]): frame for frame in frames}
    samples_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        samples_by_type[str(row["query_type"])].append(row)
    css = """
body { margin: 0; background: #f6f7f9; color: #18202a; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 28px 24px 56px; }
h1 { margin: 0 0 8px; font-size: 28px; }
.summary, details { background: #fff; border: 1px solid #dfe4ea; border-radius: 8px; padding: 16px; margin: 14px 0; }
summary { cursor: pointer; font-size: 18px; font-weight: 700; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
figure { margin: 0; border: 1px solid #dfe4ea; border-radius: 8px; overflow: hidden; background: #fbfcfe; }
.thumb { aspect-ratio: 16 / 10; background: #e9edf3; display: flex; align-items: center; justify-content: center; }
img { width: 100%; height: 100%; object-fit: contain; display: block; }
figcaption { display: grid; gap: 4px; padding: 8px; }
code { font-size: 12px; word-break: break-all; }
.query { font-weight: 700; font-size: 15px; }
.meta { color: #596575; font-size: 12px; }
"""
    lines = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>public_small_object_eval_v1 代表 query 和图片</title>",
        f"<style>{css}</style></head><body><main>",
        "<h1>public_small_object_eval_v1 代表 query 和图片</h1>",
        "<section class=\"summary\"><p>该页展示从 Open Images bbox 自动构造的公开高清小物体检索 query。只保留具体小物件标签和小面积 bbox；每个 query 的正样本可以有多张图，评测时按 multi-positive 口径计算。</p></section>",
    ]
    for query_type, title in [
        ("small_object", "小物体 small_object"),
        ("tiny_object", "极小物体 tiny_object"),
        ("edge_small_object", "边缘小物体 edge_small_object"),
    ]:
        rows = samples_by_type.get(query_type, [])[:10]
        lines.append(f"<details open><summary>{title} ({len(samples_by_type.get(query_type, []))} 条 query)</summary><div class=\"grid\">")
        for row in rows:
            positives = row.get("positive_image_ids") or []
            frame = frame_by_id.get(str(positives[0])) if positives else None
            if not frame:
                continue
            image_path = Path(str(frame["path"]))
            boxes = boxes_by_frame.get(str(frame["image_id"])) or []
            labels = ", ".join(sorted({box.label for box in boxes})[:8])
            lines.append(
                "<figure>"
                f"<div class=\"thumb\"><img src=\"{_img_src(path, image_path)}\" alt=\"{frame['image_id']}\"></div>"
                "<figcaption>"
                f"<span class=\"query\">{row['query']}</span>"
                f"<code>{row['query_id']}</code>"
                f"<span class=\"meta\">query_type={row['query_type']} | positives={len(positives)} | label={row.get('label')}</span>"
                f"<span class=\"meta\">image={frame['width']}x{frame['height']} | labels={labels}</span>"
                f"<span class=\"meta\">{row.get('notes', '')}</span>"
                "</figcaption></figure>"
            )
        lines.append("</div></details>")
    lines.append("</main></body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public image-level eval v1 from Open Images bbox annotations.")
    parser.add_argument("--out-root", type=Path, default=Path("runtime/eval/visual/public_image_eval_v1"))
    parser.add_argument("--frames-out", type=Path, default=Path("eval/visual/image_retrieval/frames.public_image_eval_v1.local.json"))
    parser.add_argument("--queries-out", type=Path, default=Path("eval/visual/image_retrieval/queries.public_image_eval_v1.local.jsonl"))
    parser.add_argument("--examples-out", type=Path, default=Path("eval/visual/outputs/public_image_eval_v1_examples.local.html"))
    parser.add_argument("--max-images", type=int, default=90)
    parser.add_argument("--max-queries-per-type", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--small-threshold", type=float, default=0.02)
    parser.add_argument("--tiny-threshold", type=float, default=TINY_OBJECT_THRESHOLD)
    parser.add_argument("--edge-margin", type=float, default=0.06)
    parser.add_argument("--min-side", type=int, default=0)
    parser.add_argument("--min-width", type=int, default=1280)
    parser.add_argument("--min-height", type=int, default=720)
    parser.add_argument("--image-source", choices=("hd", "thumbnail", "all"), default="hd")
    parser.add_argument("--candidate-multiplier", type=int, default=80)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=80)
    args = parser.parse_args()

    build_open_images_subset(
        out_root=args.out_root,
        frames_out=args.frames_out,
        queries_out=args.queries_out,
        examples_out=args.examples_out,
        max_images=args.max_images,
        max_queries_per_type=args.max_queries_per_type,
        seed=args.seed,
        small_threshold=args.small_threshold,
        tiny_threshold=args.tiny_threshold,
        edge_margin=args.edge_margin,
        min_side=args.min_side,
        min_width=args.min_width,
        min_height=args.min_height,
        image_source=args.image_source,
        candidate_multiplier=max(1, args.candidate_multiplier),
        max_workers=max(1, args.max_workers),
        batch_candidates=max(1, args.batch_candidates),
    )
    print(args.frames_out)
    print(args.queries_out)
    print(args.examples_out)


if __name__ == "__main__":
    main()
