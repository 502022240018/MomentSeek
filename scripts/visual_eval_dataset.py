from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_group_id(name: str, fallback: str) -> str:
    known = {
        "世界杯广告.mp4": "worldcup_ad",
        "世界杯广告": "worldcup_ad",
        "世界杯比赛.mp4": "worldcup_match",
        "世界杯比赛": "worldcup_match",
        "球星牛奶广告": "milk_ad",
        "球星广告.mp4": "football_star_ad",
        "球星广告": "football_star_ad",
        "给阿嬷的情书预告片": "grandma_letter_trailer",
        "2025-04-16 加更：五哈团美食速度挑战纯享.mp4": "wuha_food_speed_challenge",
        "2025-04-17 加更：五哈冰杯挑战+模仿秀纯享.mkv": "wuha_ice_cup_imitation_4k",
        "2025-04-20 第2期下：五哈版决战天山之巅 够癫！.mkv": "wuha_tianshan_challenge_4k",
        "书籍纪录片.mp4": "book_documentary",
        "书籍纪录片": "book_documentary",
        "电视剧昨夜降至04.mp4": "drama_zuoye_04",
        "电视剧昨夜降至04": "drama_zuoye_04",
        "04_1080p[云视网yuntv.net].mkv": "drama_yuntv_04_1080p",
    }
    if name in known:
        return known[name]
    stem = Path(name).stem
    value = "".join(ch.lower() if ch.isascii() and ch.isalnum() else "_" for ch in stem).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or fallback


def _resolution_label(height: int) -> str:
    standard = {
        2160: "2160p_4k",
        1440: "1440p",
        1080: "1080p",
        720: "720p",
        544: "544p",
        480: "480p",
        360: "360p",
    }
    return standard.get(int(height), f"{int(height)}p")


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _time_id(seconds: float) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    return f"{milliseconds // 1000:06d}_{milliseconds % 1000:03d}"


def _iter_variants(payload: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for group in payload.get("video_groups", []):
        for variant in group.get("variants", []):
            yield group, variant


def scan_catalog(db_path: Path, out_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("select * from videos order by created_at"))
    conn.close()

    used_group_ids: set[str] = set()
    groups = []
    for row in rows:
        name = str(row["name"])
        base_group_id = _safe_group_id(name, str(row["id"])[:8])
        group_id = base_group_id
        suffix = 2
        while group_id in used_group_ids:
            group_id = f"{base_group_id}_{suffix}"
            suffix += 1
        used_group_ids.add(group_id)

        width = int(row["width"] or 0)
        height = int(row["height"] or 0)
        fps = float(row["fps"] or 0)
        duration = float(row["duration"] or 0)
        variant_id = f"{group_id}_source_{width}x{height}" if width and height else f"{group_id}_source"
        groups.append({
            "group_id": group_id,
            "name": name,
            "catalog_video_id": row["id"],
            "content_tags": [],
            "variants": [{
                "variant_id": variant_id,
                "role": "source",
                "path": _as_posix(Path(row["file_path"])),
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "resolution_label": _resolution_label(height),
                "is_upscaled": False,
                "source_variant_id": None,
                "notes": "Generated from runtime/catalog.sqlite3.",
            }],
        })

    payload = {
        "schema_version": 1,
        "created_for": "MomentSeek visual retrieval evaluation",
        "video_groups": groups,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def scan_directory(root: Path, out_path: Path, recursive: bool = True) -> dict[str, Any]:
    extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    pattern = "**/*" if recursive else "*"
    files = sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in extensions)
    used_group_ids: set[str] = set()
    groups = []
    for index, path in enumerate(files, start=1):
        meta = _probe_video_cv2(path)
        base_group_id = _safe_group_id(path.name, f"video_{index:03d}")
        group_id = base_group_id
        suffix = 2
        while group_id in used_group_ids:
            group_id = f"{base_group_id}_{suffix}"
            suffix += 1
        used_group_ids.add(group_id)

        width = int(meta["width"])
        height = int(meta["height"])
        variant_id = f"{group_id}_source_{width}x{height}" if width and height else f"{group_id}_source"
        groups.append({
            "group_id": group_id,
            "name": path.stem,
            "source_kind": "directory_scan",
            "content_tags": [],
            "variants": [{
                "variant_id": variant_id,
                "role": "source",
                "path": _as_posix(path),
                "width": width,
                "height": height,
                "fps": float(meta["fps"]),
                "duration": float(meta["duration"]),
                "resolution_label": _resolution_label(height),
                "is_upscaled": False,
                "source_variant_id": None,
                "notes": f"Generated from directory scan: {root}",
            }],
        })

    payload = {
        "schema_version": 1,
        "created_for": "MomentSeek visual retrieval evaluation",
        "source_directory": _as_posix(root),
        "video_groups": groups,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _probe_video_cv2(path: Path) -> dict[str, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for probing when ffprobe is unavailable.") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": frames / fps if fps else 0,
    }


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_variant_ffmpeg(source: Path, target: Path, height: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:{height}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(target),
    ]
    subprocess.run(command, check=True)


def _make_variant_cv2(source: Path, target: Path, height: int) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for variant generation when ffmpeg is unavailable.") from exc

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video dimensions: {source}")
    width = int(round(source_width * height / source_height))
    if width % 2:
        width += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {target}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if height < source_height else cv2.INTER_CUBIC)
        writer.write(resized)
    writer.release()
    cap.release()


@dataclass
class VariantTask:
    group_id: str
    source_variant: dict[str, Any]
    height: int
    target: Path
    allow_upscale: bool

    @property
    def source_path(self) -> Path:
        return Path(str(self.source_variant["path"]))

    @property
    def source_height(self) -> int:
        return int(self.source_variant.get("height") or 0)

    @property
    def source_width(self) -> int:
        return int(self.source_variant.get("width") or 0)

    @property
    def should_make(self) -> bool:
        if self.height == self.source_height:
            return False
        return self.allow_upscale or self.height < self.source_height


def _best_source_variant(group: dict[str, Any]) -> dict[str, Any]:
    variants = group.get("variants", [])
    if not variants:
        raise RuntimeError(f"Group has no variants: {group.get('group_id')}")
    return max(variants, key=lambda item: int(item.get("height") or 0))


def make_variants(
    manifest_path: Path,
    out_root: Path,
    out_manifest: Path,
    heights: list[int],
    allow_upscale: bool,
) -> dict[str, Any]:
    payload = _load_manifest(manifest_path)
    use_ffmpeg = _ffmpeg_available()
    out_root.mkdir(parents=True, exist_ok=True)

    for group in payload.get("video_groups", []):
        group_id = str(group["group_id"])
        source = _best_source_variant(group)
        source_path = Path(str(source["path"]))
        if not source_path.exists():
            raise FileNotFoundError(f"Source video does not exist: {source_path}")

        existing_by_height = {int(v.get("height") or 0): v for v in group.get("variants", [])}
        for height in heights:
            if height in existing_by_height:
                continue
            target = out_root / group_id / f"{group_id}_{height}p.mp4"
            task = VariantTask(group_id, source, int(height), target, allow_upscale)
            if not task.should_make:
                continue
            if use_ffmpeg:
                _make_variant_ffmpeg(task.source_path, task.target, task.height)
            else:
                _make_variant_cv2(task.source_path, task.target, task.height)

            meta = _probe_video_cv2(task.target)
            group.setdefault("variants", []).append({
                "variant_id": f"{group_id}_{height}p",
                "role": "generated_resolution_variant",
                "path": _as_posix(task.target),
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "fps": float(meta["fps"]),
                "duration": float(meta["duration"]),
                "resolution_label": _resolution_label(int(meta["height"])),
                "is_upscaled": bool(task.height > task.source_height),
                "source_variant_id": source["variant_id"],
                "notes": "Generated by scripts/visual_eval_dataset.py. Audio is intentionally omitted; visual evaluation only.",
            })

        group["variants"] = sorted(group.get("variants", []), key=lambda item: int(item.get("height") or 0))

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def extract_frames(
    manifest_path: Path,
    out_root: Path,
    out_path: Path,
    interval_seconds: float,
    max_frames_per_variant: int | None,
    jpeg_quality: int,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame extraction.") from exc

    payload = _load_manifest(manifest_path)
    frames: list[dict[str, Any]] = []
    out_root.mkdir(parents=True, exist_ok=True)

    for group, variant in _iter_variants(payload):
        group_id = str(group["group_id"])
        variant_id = str(variant["variant_id"])
        video_path = Path(str(variant["path"]))
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or variant.get("fps") or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(variant.get("duration") or (frame_count / fps if fps else 0))
        target_dir = out_root / group_id / variant_id
        target_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        timestamp = 0.0
        while timestamp <= max(0.0, duration - 1e-3):
            if max_frames_per_variant is not None and count >= max_frames_per_variant:
                break
            image_time_id = _time_id(timestamp)
            image_id = f"{variant_id}_t{image_time_id}"
            image_path = target_dir / f"t{image_time_id}.jpg"
            frame = cv2.imread(str(image_path)) if image_path.exists() else None
            if frame is None:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    raise RuntimeError(f"Cannot encode frame: {image_path}")
                image_path.write_bytes(encoded.tobytes())
            height, width = frame.shape[:2]
            frames.append({
                "image_id": image_id,
                "group_id": group_id,
                "variant_id": variant_id,
                "path": _as_posix(image_path),
                "time": round(float(timestamp), 3),
                "width": int(width),
                "height": int(height),
                "resolution_label": variant.get("resolution_label") or _resolution_label(int(height)),
                "view_type": "original_frame",
                "source_video_path": _as_posix(video_path),
            })
            count += 1
            timestamp += float(interval_seconds)
        cap.release()

    result = {
        "schema_version": 1,
        "created_for": "MomentSeek image-level visual retrieval evaluation",
        "source_manifest": _as_posix(manifest_path),
        "interval_seconds": float(interval_seconds),
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def make_segments(
    manifest_path: Path,
    out_path: Path,
    segment_seconds: float,
    stride_seconds: float,
    min_segment_seconds: float,
) -> dict[str, Any]:
    payload = _load_manifest(manifest_path)
    segments: list[dict[str, Any]] = []

    for group, variant in _iter_variants(payload):
        group_id = str(group["group_id"])
        variant_id = str(variant["variant_id"])
        duration = float(variant.get("duration") or 0)
        start = 0.0
        while start < duration:
            end = min(start + float(segment_seconds), duration)
            if end - start < float(min_segment_seconds):
                break
            segment_id = f"{variant_id}_s{_time_id(start)}_{_time_id(end)}"
            segments.append({
                "segment_id": segment_id,
                "group_id": group_id,
                "variant_id": variant_id,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "resolution_label": variant.get("resolution_label") or _resolution_label(int(variant.get("height") or 0)),
                "source_video_path": variant.get("path"),
            })
            start += float(stride_seconds)

    result = {
        "schema_version": 1,
        "created_for": "MomentSeek sequence-level visual retrieval evaluation",
        "source_manifest": _as_posix(manifest_path),
        "segment_seconds": float(segment_seconds),
        "stride_seconds": float(stride_seconds),
        "segments": segments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _letterbox_resize(frame, width: int, height: int):
    import cv2
    import numpy as np

    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 32, dtype=frame.dtype)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def make_contact_sheets(
    segments_path: Path,
    out_root: Path,
    out_path: Path,
    frames_per_segment: int | None,
    sheet_sample_fps: float | None,
    cell_width: int,
    cell_height: int,
    max_per_variant: int | None,
    jpeg_quality: int,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for contact sheet generation.") from exc

    segments_payload = json.loads(segments_path.read_text(encoding="utf-8"))
    out_root.mkdir(parents=True, exist_ok=True)
    sheets: list[dict[str, Any]] = []
    counts_by_variant: dict[str, int] = {}
    caps: dict[str, Any] = {}

    try:
        for segment in segments_payload.get("segments", []):
            variant_id = str(segment["variant_id"])
            if max_per_variant is not None and counts_by_variant.get(variant_id, 0) >= max_per_variant:
                continue
            video_path = Path(str(segment["source_video_path"]))
            if not video_path.exists():
                raise FileNotFoundError(f"Video does not exist: {video_path}")

            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.0, end - start)
            if sheet_sample_fps is not None and sheet_sample_fps > 0:
                frame_count = max(1, int(np.ceil(duration * float(sheet_sample_fps))))
            else:
                frame_count = int(frames_per_segment or 6)
            if frame_count <= 1:
                timestamps = [(start + end) / 2.0]
            else:
                timestamps = [
                    min(end, start + index / float(sheet_sample_fps))
                    for index in range(frame_count)
                ] if sheet_sample_fps is not None and sheet_sample_fps > 0 else [
                    start + (end - start) * index / max(1, frame_count - 1)
                    for index in range(frame_count)
                ]
            rows = 2 if frame_count > 3 else 1
            cols = int(np.ceil(frame_count / rows))

            group_id = str(segment["group_id"])
            segment_id = str(segment["segment_id"])
            target_dir = out_root / group_id / variant_id
            target_dir.mkdir(parents=True, exist_ok=True)
            sheet_path = target_dir / f"s{_time_id(start)}_{_time_id(end)}.jpg"
            if sheet_path.exists():
                counts_by_variant[variant_id] = counts_by_variant.get(variant_id, 0) + 1
                sheets.append({
                    "sheet_id": segment_id,
                    "segment_id": segment_id,
                    "group_id": group_id,
                    "variant_id": variant_id,
                    "path": _as_posix(sheet_path),
                    "start": start,
                    "end": end,
                    "sample_times": [round(value, 3) for value in timestamps],
                    "width": int(cols * cell_width),
                    "height": int(rows * cell_height),
                    "resolution_label": segment.get("resolution_label"),
                    "source_video_path": _as_posix(video_path),
                })
                continue

            video_key = str(video_path)
            if video_key not in caps:
                cap = cv2.VideoCapture(video_key)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {video_path}")
                caps[video_key] = cap
            cap = caps[video_key]

            cells = []
            for timestamp in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    frame = np.full((cell_height, cell_width, 3), 32, dtype=np.uint8)
                cell = _letterbox_resize(frame, cell_width, cell_height)
                label = f"{timestamp:.1f}s"
                cv2.rectangle(cell, (0, 0), (92, 24), (0, 0, 0), thickness=-1)
                cv2.putText(cell, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(cell)

            while len(cells) < rows * cols:
                cells.append(np.full((cell_height, cell_width, 3), 32, dtype=np.uint8))
            sheet_rows = []
            for row in range(rows):
                sheet_rows.append(np.concatenate(cells[row * cols:(row + 1) * cols], axis=1))
            sheet = np.concatenate(sheet_rows, axis=0)

            ok, encoded = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok:
                raise RuntimeError(f"Failed to encode contact sheet: {sheet_path}")
            sheet_path.write_bytes(encoded.tobytes())
            counts_by_variant[variant_id] = counts_by_variant.get(variant_id, 0) + 1
            sheets.append({
                "sheet_id": segment_id,
                "segment_id": segment_id,
                "group_id": group_id,
                "variant_id": variant_id,
                "path": _as_posix(sheet_path),
                "start": start,
                "end": end,
                "sample_times": [round(value, 3) for value in timestamps],
                "width": int(sheet.shape[1]),
                "height": int(sheet.shape[0]),
                "resolution_label": segment.get("resolution_label"),
                "source_video_path": _as_posix(video_path),
            })
    finally:
        for cap in caps.values():
            cap.release()

    sample_time_count_summary: dict[str, int] = {}
    for sheet in sheets:
        sample_count = len(sheet.get("sample_times", []))
        sample_time_count_summary[str(sample_count)] = sample_time_count_summary.get(str(sample_count), 0) + 1

    result = {
        "schema_version": 1,
        "created_for": "MomentSeek sequence-level visual auto-annotation",
        "source_segments": _as_posix(segments_path),
        "frames_per_segment": int(max((len(sheet.get("sample_times", [])) for sheet in sheets), default=int(frames_per_segment or 0))),
        "sheet_sample_fps": float(sheet_sample_fps or 0),
        "cell_width": int(cell_width),
        "cell_height": int(cell_height),
        "sample_time_count_summary": sample_time_count_summary,
        "sheets": sheets,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _evenly_select(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    selected = []
    used = set()
    for index in range(count):
        source_index = round(index * (len(items) - 1) / (count - 1))
        while source_index in used and source_index + 1 < len(items):
            source_index += 1
        while source_index in used and source_index - 1 >= 0:
            source_index -= 1
        used.add(source_index)
        selected.append(items[source_index])
    return selected


def sample_manifest(
    source_path: Path,
    out_path: Path,
    item_key: str,
    caps: dict[str, int],
    default_cap: int,
    note: str,
) -> dict[str, Any]:
    payload = _load_manifest(source_path)
    items = list(payload.get(item_key, []))
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_group.setdefault(str(item.get("group_id", "unknown")), []).append(item)

    selected: list[dict[str, Any]] = []
    summary = []
    for group_id in sorted(by_group):
        group_items = sorted(
            by_group[group_id],
            key=lambda item: float(item.get("time", item.get("start", 0)) or 0),
        )
        cap = int(caps.get(group_id, default_cap))
        group_selected = _evenly_select(group_items, cap)
        selected.extend(group_selected)
        summary.append({
            "group_id": group_id,
            "available": len(group_items),
            "selected": len(group_selected),
            "cap": cap,
        })

    result = {
        "schema_version": payload.get("schema_version", 1),
        "created_for": "MomentSeek visual CLIP selection benchmark",
        "source_manifest": _as_posix(source_path),
        "sampling": {
            "strategy": "evenly_spaced_by_group",
            "default_cap": default_cap,
            "caps": caps,
            "note": note,
        },
        item_key: selected,
        "summary": summary,
    }
    # Preserve fields that downstream helpers use.
    for key in ("interval_seconds", "segment_seconds", "stride_seconds", "created_for"):
        if key in payload and key not in result:
            result[key] = payload[key]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _parse_caps(values: list[str] | None) -> dict[str, int]:
    caps: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Cap must be group_id=count, got: {value}")
        group_id, count = value.split("=", 1)
        caps[group_id.strip()] = int(count)
    return caps


def validate_queries(manifest_path: Path, queries_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    group_ids = {str(group["group_id"]) for group in manifest.get("video_groups", [])}
    total = 0
    labeled = 0
    missing_groups = []
    query_types: dict[str, int] = {}
    resolution_sensitivity: dict[str, int] = {}

    for line_number, line in enumerate(queries_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        item = json.loads(line)
        group_id = str(item.get("group_id", ""))
        if group_id not in group_ids:
            missing_groups.append({"line": line_number, "query_id": item.get("query_id"), "group_id": group_id})
        positives = item.get("positives") or item.get("positive_image_ids") or []
        if positives:
            labeled += 1
        query_type = str(item.get("query_type", "unknown"))
        query_types[query_type] = query_types.get(query_type, 0) + 1
        sensitivity = str(item.get("resolution_sensitivity", "unknown"))
        resolution_sensitivity[sensitivity] = resolution_sensitivity.get(sensitivity, 0) + 1

    return {
        "total_queries": total,
        "labeled_queries": labeled,
        "unlabeled_queries": total - labeled,
        "query_types": query_types,
        "resolution_sensitivity": resolution_sensitivity,
        "missing_groups": missing_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MomentSeek visual evaluation datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan-catalog", help="Create a local visual eval video manifest from catalog.sqlite3.")
    scan_parser.add_argument("--db", type=Path, default=Path("runtime/catalog.sqlite3"))
    scan_parser.add_argument("--out", type=Path, default=Path("eval/visual/videos.local.json"))

    scan_dir_parser = subparsers.add_parser("scan-directory", help="Create a local visual eval video manifest from a directory.")
    scan_dir_parser.add_argument("--root", type=Path, required=True)
    scan_dir_parser.add_argument("--out", type=Path, default=Path("eval/visual/videos.local.json"))
    scan_dir_parser.add_argument("--no-recursive", action="store_true")

    variant_parser = subparsers.add_parser("make-variants", help="Generate same-content resolution variants.")
    variant_parser.add_argument("--manifest", type=Path, default=Path("eval/visual/videos.local.json"))
    variant_parser.add_argument("--out-root", type=Path, default=Path("runtime/eval/visual/resolution_variants"))
    variant_parser.add_argument("--out-manifest", type=Path, default=Path("eval/visual/videos.variants.local.json"))
    variant_parser.add_argument("--heights", type=int, nargs="+", default=[360, 720, 1080, 2160])
    variant_parser.add_argument("--allow-upscale", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate query JSONL against a visual eval manifest.")
    validate_parser.add_argument("--manifest", type=Path, default=Path("eval/visual/videos.local.json"))
    validate_parser.add_argument("--queries", type=Path, default=Path("eval/visual/queries.seed.jsonl"))

    frames_parser = subparsers.add_parser("extract-frames", help="Extract fixed-interval frames for image-level evaluation.")
    frames_parser.add_argument("--manifest", type=Path, default=Path("eval/visual/videos.variants.local.json"))
    frames_parser.add_argument("--out-root", type=Path, default=Path("runtime/eval/visual/frames"))
    frames_parser.add_argument("--out", type=Path, default=Path("eval/visual/image_retrieval/frames.local.json"))
    frames_parser.add_argument("--interval-seconds", type=float, default=5.0)
    frames_parser.add_argument("--max-frames-per-variant", type=int, default=None)
    frames_parser.add_argument("--jpeg-quality", type=int, default=92)

    segments_parser = subparsers.add_parser("make-segments", help="Create fixed-window segments for sequence-level evaluation.")
    segments_parser.add_argument("--manifest", type=Path, default=Path("eval/visual/videos.variants.local.json"))
    segments_parser.add_argument("--out", type=Path, default=Path("eval/visual/sequence_retrieval/segments.local.json"))
    segments_parser.add_argument("--segment-seconds", type=float, default=5.0)
    segments_parser.add_argument("--stride-seconds", type=float, default=5.0)
    segments_parser.add_argument("--min-segment-seconds", type=float, default=1.0)

    sheets_parser = subparsers.add_parser("make-contact-sheets", help="Create contact sheets for sequence-level auto-annotation.")
    sheets_parser.add_argument("--segments", type=Path, default=Path("eval/visual/sequence_retrieval/segments.local.json"))
    sheets_parser.add_argument("--out-root", type=Path, default=Path("runtime/eval/visual/contact_sheets"))
    sheets_parser.add_argument("--out", type=Path, default=Path("eval/visual/sequence_retrieval/contact_sheets.local.json"))
    sheets_parser.add_argument("--frames-per-segment", type=int, default=6)
    sheets_parser.add_argument("--sheet-sample-fps", type=float, default=None, help="If set, sample this many frames per second inside each segment.")
    sheets_parser.add_argument("--cell-width", type=int, default=320)
    sheets_parser.add_argument("--cell-height", type=int, default=180)
    sheets_parser.add_argument("--max-per-variant", type=int, default=None)
    sheets_parser.add_argument("--jpeg-quality", type=int, default=90)

    sample_parser = subparsers.add_parser("sample-manifest", help="Create an evenly-spaced benchmark subset manifest.")
    sample_parser.add_argument("--source", type=Path, required=True)
    sample_parser.add_argument("--out", type=Path, required=True)
    sample_parser.add_argument("--item-key", choices=["frames", "segments", "sheets"], required=True)
    sample_parser.add_argument("--default-cap", type=int, default=20)
    sample_parser.add_argument("--cap", action="append", default=[], help="Per-group cap, e.g. worldcup_ad=20")
    sample_parser.add_argument("--note", default="")

    args = parser.parse_args()
    if args.command == "scan-catalog":
        payload = scan_catalog(args.db, args.out)
        print(json.dumps({
            "out": str(args.out),
            "video_groups": len(payload.get("video_groups", [])),
        }, ensure_ascii=False, indent=2))
    elif args.command == "scan-directory":
        payload = scan_directory(args.root, args.out, recursive=not args.no_recursive)
        print(json.dumps({
            "out": str(args.out),
            "video_groups": len(payload.get("video_groups", [])),
            "source_directory": str(args.root),
        }, ensure_ascii=False, indent=2))
    elif args.command == "make-variants":
        payload = make_variants(args.manifest, args.out_root, args.out_manifest, args.heights, args.allow_upscale)
        print(json.dumps({
            "out_manifest": str(args.out_manifest),
            "video_groups": len(payload.get("video_groups", [])),
            "heights": args.heights,
            "allow_upscale": args.allow_upscale,
        }, ensure_ascii=False, indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_queries(args.manifest, args.queries), ensure_ascii=False, indent=2))
    elif args.command == "extract-frames":
        payload = extract_frames(
            args.manifest,
            args.out_root,
            args.out,
            args.interval_seconds,
            args.max_frames_per_variant,
            args.jpeg_quality,
        )
        print(json.dumps({
            "out": str(args.out),
            "frames": len(payload.get("frames", [])),
            "interval_seconds": args.interval_seconds,
        }, ensure_ascii=False, indent=2))
    elif args.command == "make-segments":
        payload = make_segments(
            args.manifest,
            args.out,
            args.segment_seconds,
            args.stride_seconds,
            args.min_segment_seconds,
        )
        print(json.dumps({
            "out": str(args.out),
            "segments": len(payload.get("segments", [])),
            "segment_seconds": args.segment_seconds,
            "stride_seconds": args.stride_seconds,
        }, ensure_ascii=False, indent=2))
    elif args.command == "make-contact-sheets":
        payload = make_contact_sheets(
            args.segments,
            args.out_root,
            args.out,
            args.frames_per_segment,
            args.sheet_sample_fps,
            args.cell_width,
            args.cell_height,
            args.max_per_variant,
            args.jpeg_quality,
        )
        print(json.dumps({
            "out": str(args.out),
            "sheets": len(payload.get("sheets", [])),
            "frames_per_segment": payload.get("frames_per_segment"),
            "sheet_sample_fps": payload.get("sheet_sample_fps"),
            "sample_time_count_summary": payload.get("sample_time_count_summary"),
            "max_per_variant": args.max_per_variant,
        }, ensure_ascii=False, indent=2))
    elif args.command == "sample-manifest":
        payload = sample_manifest(
            args.source,
            args.out,
            args.item_key,
            _parse_caps(args.cap),
            args.default_cap,
            args.note,
        )
        print(json.dumps({
            "out": str(args.out),
            "item_key": args.item_key,
            "items": len(payload.get(args.item_key, [])),
            "summary": payload.get("summary", []),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
