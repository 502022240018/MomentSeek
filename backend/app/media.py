from __future__ import annotations

import re
import shutil
import subprocess
import uuid
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


def _read_exact(stream, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            return None
        buffer += chunk
    return bytes(buffer)


def iter_ffmpeg_frames(path: str | Path, sample_fps: float, out_height: int = 0) -> Iterator[tuple[float, np.ndarray]]:
    """Decode + sample (+ optional downscale) via ffmpeg, yielding (timestamp, bgr_frame).

    ffmpeg decodes multithreaded in C and the fps/scale filters resample and shrink
    in one pass, so we skip cv2's single-threaded full-resolution decode and the
    per-frame resize. Frames come out as bgr24 (cv2 convention) so consumers are
    unchanged. Raises on setup/stream error so callers can fall back to cv2.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps 必须大于 0")
    info = probe_video(path)
    src_w, src_h = int(info.width), int(info.height)
    if src_w <= 0 or src_h <= 0:
        raise OSError(f"无法获取视频尺寸: {path}")
    if out_height and out_height < src_h:
        out_h = int(out_height)
        out_w = int(round(src_w * out_h / src_h / 2) * 2)  # even width for rawvideo
    else:
        out_h, out_w = src_h, src_w
    video_filter = f"fps={sample_fps}"
    if (out_w, out_h) != (src_w, src_h):
        video_filter += f",scale={out_w}:{out_h}"
    command = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", video_filter, "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
    frame_bytes = out_w * out_h * 3
    index = 0
    try:
        while True:
            raw = _read_exact(process.stdout, frame_bytes)
            if raw is None:
                break
            yield index / sample_fps, np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3)
            index += 1
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait()
    if process.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg 抽帧失败 (code {process.returncode})")


_FRAME_SENTINEL = object()


def read_frames(path: str | Path, sample_fps: float, out_height: int = 0, prefer_ffmpeg: bool = True) -> Iterator[tuple[float, np.ndarray]]:
    """Yield sampled frames, preferring ffmpeg; fall back to cv2 if ffmpeg can't start."""
    if prefer_ffmpeg:
        iterator = iter_ffmpeg_frames(path, sample_fps, out_height)
        first = _FRAME_SENTINEL
        try:
            first = next(iterator)
        except Exception:  # setup failure or zero frames -> fall back to cv2
            first = _FRAME_SENTINEL
        if first is not _FRAME_SENTINEL:
            yield first
            yield from iterator
            return
    yield from iter_sampled_frames(path, sample_fps)


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
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def export_preview_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    max_seconds: float = 45.0,
) -> Path:
    """Export a browser-friendly MP4 preview for a matched moment.

    Search results point to short clips instead of asking the browser to stream a
    full source video and seek across a tunnel. We transcode to H.264/AAC because
    uploaded sources may be HEVC/H.265 or have no filename extension, both of
    which frequently fail in browser `<video>` playback.
    """
    start = max(0.0, float(start_time))
    end = max(start + 0.25, float(end_time))
    duration = min(max_seconds, end - start)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", "scale='min(1280,iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-tag:v", "avc1",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        temp_path.replace(output_path)
    except subprocess.CalledProcessError as exc:
        temp_path.unlink(missing_ok=True)
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"片段导出失败: {details[-1200:]}") from exc
    return output_path


def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("未找到 ffmpeg；请安装 ffmpeg 或 imageio-ffmpeg") from exc


_TIMECODE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})")


def parse_timecode(value: str) -> float:
    match = _TIMECODE.search(value.strip())
    if not match:
        raise ValueError(f"无法解析时间: {value}")
    hours, minutes, seconds, millis = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000
