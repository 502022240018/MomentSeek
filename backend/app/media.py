from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    frames: int
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"无法打开视频: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = VideoInfo(
        duration=frames / fps if fps > 0 else 0,
        fps=fps,
        frames=frames,
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    capture.release()
    return info


def iter_sampled_frames(path: str | Path, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps 必须大于 0")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"无法打开视频: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
    step = max(1, round(source_fps / sample_fps))
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % step == 0:
                yield frame_number / source_fps, frame
            frame_number += 1
    finally:
        capture.release()


def save_thumbnail(frame: np.ndarray, path: str | Path, max_width: int = 480) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise OSError(f"缩略图保存失败: {path}")


def extract_audio(video_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


_TIMECODE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})")


def parse_timecode(value: str) -> float:
    match = _TIMECODE.search(value.strip())
    if not match:
        raise ValueError(f"无法解析时间: {value}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000

