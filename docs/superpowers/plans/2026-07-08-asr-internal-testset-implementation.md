# ASR Internal Testset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first internal ASR testset tooling and dataset skeleton for MomentSeek, with Chinese-first manifest validation, platform subtitle truth import, and static-MP4 packaging for locally prepared open ASR sources.

**Architecture:** Add one focused script module under `scripts/` that owns ASR testset records, sidecar/SRT/VTT writing, static video packaging, manifest validation, and a small CLI. Keep source media out of git; repo assets live under `eval/asr/internal_testset/`. Tests import the script directly, following existing script-test patterns.

**Tech Stack:** Python stdlib, `Pillow`, `imageio-ffmpeg`, existing `backend/app/indexing/asr.py::load_sidecar`, pytest.

---

## Scope Check

The approved spec has two natural parts:

1. Build reusable local tooling and repo assets.
2. Acquire large/open-source media and generate the full 12h-15h testset.

This plan makes part 1 fully executable now and part 2 ready for local caches. WenetSpeech access may require a form/email and source media must stay internal; the implementation must support `data/asr_internal_testset/cache/` and `data/asr_internal_testset/generated_media/` without committing media files.

## File Structure

- Modify: `.gitignore`
  - Ignore ASR internal source caches and generated MP4s.
- Create: `scripts/asr_internal_testset.py`
  - Owns dataclasses, JSONL IO, sidecar/SRT/VTT writers, manifest validation, platform-truth import, static MP4 build, and CLI.
- Create: `backend/tests/test_asr_internal_testset.py`
  - Tests script functions using temporary files and direct module import.
- Create: `eval/asr/internal_testset/README.md`
  - Human instructions for the internal ASR testset.
- Create: `eval/asr/internal_testset/sources.md`
  - Source/license/use restriction notes.
- Create via CLI: `eval/asr/internal_testset/manifest.jsonl`
  - First manifest rows, starting with `昨夜降至04`.
- Create via CLI: `eval/asr/internal_testset/truth/*.sidecar.json`
  - Truth sidecars copied/generated for test samples.
- Create via CLI: `eval/asr/internal_testset/reports/validation_summary.json`
  - Build/validation summary.

## Task 1: Protect Local Media Caches

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add ASR internal cache ignores**

Patch `.gitignore`:

```gitignore
data/asr_internal_testset/cache/
data/asr_internal_testset/generated_media/
data/asr_internal_testset/tmp/
```

- [ ] **Step 2: Verify gitignore behavior**

Run:

```powershell
git check-ignore data/asr_internal_testset/cache/source.wav data/asr_internal_testset/generated_media/sample.mp4 data/asr_internal_testset/tmp/frame.png
```

Expected: all three paths print.

- [ ] **Step 3: Commit**

```powershell
git add -- .gitignore
git commit -m "chore: ignore ASR internal testset media"
```

## Task 2: Add Core Records, IO, And Subtitle Writers

**Files:**
- Create: `scripts/asr_internal_testset.py`
- Test: `backend/tests/test_asr_internal_testset.py`

- [ ] **Step 1: Write failing tests for sidecar/SRT/VTT writing**

Create `backend/tests/test_asr_internal_testset.py` with:

```python
import importlib.util
import json
from pathlib import Path


def _load_script():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "asr_internal_testset.py"
    spec = importlib.util.spec_from_file_location("asr_internal_testset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_writes_sidecar_srt_and_vtt(tmp_path):
    module = _load_script()
    segments = [
        module.TranscriptSegment(start_time=0.0, end_time=1.25, text="你好，世界"),
        module.TranscriptSegment(start_time=2.0, end_time=4.5, text="第二句话"),
    ]

    sidecar = tmp_path / "sample.sidecar.json"
    srt = tmp_path / "sample.srt"
    vtt = tmp_path / "sample.vtt"
    module.write_sidecar_json(segments, sidecar)
    module.write_srt(segments, srt)
    module.write_vtt(segments, vtt)

    assert json.loads(sidecar.read_text(encoding="utf-8")) == [
        {"start_time": 0.0, "end_time": 1.25, "text": "你好，世界"},
        {"start_time": 2.0, "end_time": 4.5, "text": "第二句话"},
    ]
    assert "00:00:01,250" in srt.read_text(encoding="utf-8")
    assert "00:00:04.500" in vtt.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_writes_sidecar_srt_and_vtt -q
```

Expected: FAIL because `scripts/asr_internal_testset.py` does not exist.

- [ ] **Step 3: Implement core records and writers**

Create `scripts/asr_internal_testset.py` with these initial contents:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TranscriptSegment:
    start_time: float
    end_time: float
    text: str
    raw_text: str = ""
    source_id: str = ""

    def to_sidecar(self) -> dict[str, Any]:
        return {
            "start_time": round(float(self.start_time), 3),
            "end_time": round(float(self.end_time), 3),
            "text": self.text.strip(),
        }


@dataclass
class TestsetSample:
    sample_id: str
    version: str
    source_dataset: str
    source_url: str
    source_item_id: str
    language: str
    text_script: str
    scenario_tags: list[str]
    duration_seconds: float
    media_kind: str
    generated_media_path: str
    truth_sidecar_path: str
    truth_srt_path: str
    truth_vtt_path: str = ""
    license: str = ""
    internal_use_only: bool = True
    redistribute_original_media: bool = False
    source_hash: str = ""
    generated_hash: str = ""
    sampling_seed: int = 20260708
    media_available: bool = True
    notes: str = ""
    source_segments: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _srt_time(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def validate_segments(segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    last_end = -1.0
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        if segment.end_time <= segment.start_time:
            raise ValueError(f"segment {index} has non-positive duration")
        if segment.start_time < last_end:
            raise ValueError(f"segment {index} is not monotonic")
        cleaned.append(TranscriptSegment(segment.start_time, segment.end_time, text, segment.raw_text, segment.source_id))
        last_end = segment.end_time
    if not cleaned:
        raise ValueError("no non-empty transcript segments")
    return cleaned


def write_sidecar_json(segments: Iterable[TranscriptSegment], path: str | Path) -> None:
    cleaned = validate_segments(segments)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([segment.to_sidecar() for segment in cleaned], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_srt(segments: Iterable[TranscriptSegment], path: str | Path) -> None:
    cleaned = validate_segments(segments)
    blocks = []
    for index, segment in enumerate(cleaned, start=1):
        blocks.append(f"{index}\n{_srt_time(segment.start_time)} --> {_srt_time(segment.end_time)}\n{segment.text}\n")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(blocks), encoding="utf-8")


def write_vtt(segments: Iterable[TranscriptSegment], path: str | Path) -> None:
    cleaned = validate_segments(segments)
    blocks = ["WEBVTT\n"]
    for segment in cleaned:
        blocks.append(f"{_vtt_time(segment.start_time)} --> {_vtt_time(segment.end_time)}\n{segment.text}\n")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(blocks), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_writes_sidecar_srt_and_vtt -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/asr_internal_testset.py backend/tests/test_asr_internal_testset.py
git commit -m "feat: add ASR testset subtitle writers"
```

## Task 3: Add Manifest Validation

**Files:**
- Modify: `scripts/asr_internal_testset.py`
- Modify: `backend/tests/test_asr_internal_testset.py`

- [ ] **Step 1: Write failing manifest validation test**

Append to `backend/tests/test_asr_internal_testset.py`:

```python
def test_validate_manifest_enforces_duration_language_and_sidecar(tmp_path):
    module = _load_script()
    truth = tmp_path / "truth" / "sample.sidecar.json"
    module.write_sidecar_json([module.TranscriptSegment(0, 3600, "长音频")], truth)
    manifest = tmp_path / "manifest.jsonl"
    module.write_jsonl([
        {
            "sample_id": "sample-1",
            "version": "v1",
            "source_dataset": "WenetSpeech",
            "source_url": "https://wenet-e2e.github.io/WenetSpeech/",
            "source_item_id": "local",
            "language": "zh",
            "text_script": "Hans",
            "scenario_tags": ["zh", "zh_real_video_podcast", "long"],
            "duration_seconds": 3600.0,
            "media_kind": "generated_static_mp4",
            "generated_media_path": "data/asr_internal_testset/generated_media/sample.mp4",
            "truth_sidecar_path": str(truth),
            "truth_srt_path": "",
            "license": "internal",
            "internal_use_only": True,
            "redistribute_original_media": False,
            "media_available": False,
        }
    ], manifest)

    summary = module.validate_manifest(manifest, repo_root=tmp_path)

    assert summary["sample_count"] == 1
    assert summary["total_hours"] == 1.0
    assert summary["zh_hours"] == 1.0
    assert summary["wenet_hours"] == 1.0
    assert summary["errors"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_validate_manifest_enforces_duration_language_and_sidecar -q
```

Expected: FAIL because `validate_manifest` is missing.

- [ ] **Step 3: Implement manifest validation**

Append to `scripts/asr_internal_testset.py`:

```python
def _resolve_manifest_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_sidecar_for_validation(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("segments", [])
    if not isinstance(payload, list):
        raise ValueError("sidecar must be a list or object with segments")
    return payload


def validate_manifest(manifest_path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    manifest = Path(manifest_path)
    root = Path(repo_root)
    records = read_jsonl(manifest)
    errors: list[str] = []
    sample_ids: set[str] = set()
    total_seconds = 0.0
    zh_seconds = 0.0
    long_seconds = 0.0
    wenet_seconds = 0.0
    sidecar_count = 0
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            errors.append(f"line {index}: missing sample_id")
        elif sample_id in sample_ids:
            errors.append(f"line {index}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        duration = float(record.get("duration_seconds") or 0.0)
        if duration <= 0:
            errors.append(f"line {index}: duration_seconds must be positive")
        total_seconds += max(0.0, duration)
        if str(record.get("language") or "").startswith("zh"):
            zh_seconds += max(0.0, duration)
        if "long" in [str(tag) for tag in record.get("scenario_tags") or []] or duration >= 900:
            long_seconds += max(0.0, duration)
        if str(record.get("source_dataset") or "") == "WenetSpeech":
            wenet_seconds += max(0.0, duration)
        if record.get("internal_use_only") is not True:
            errors.append(f"line {index}: internal_use_only must be true")
        if record.get("redistribute_original_media") is not False:
            errors.append(f"line {index}: redistribute_original_media must be false")
        sidecar_value = str(record.get("truth_sidecar_path") or "")
        if not sidecar_value:
            errors.append(f"line {index}: missing truth_sidecar_path")
            continue
        sidecar_path = _resolve_manifest_path(sidecar_value, root)
        if not sidecar_path.exists():
            errors.append(f"line {index}: missing sidecar {sidecar_value}")
            continue
        try:
            sidecar = _load_sidecar_for_validation(sidecar_path)
            previous_end = -1.0
            for segment_index, segment in enumerate(sidecar):
                start = float(segment.get("start_time", segment.get("start", 0)))
                end = float(segment.get("end_time", segment.get("end", 0)))
                text = str(segment.get("text", "")).strip()
                if not text:
                    errors.append(f"line {index}: empty sidecar text at segment {segment_index}")
                if end <= start:
                    errors.append(f"line {index}: invalid sidecar time at segment {segment_index}")
                if start < previous_end:
                    errors.append(f"line {index}: non-monotonic sidecar at segment {segment_index}")
                previous_end = end
            sidecar_count += 1
        except Exception as exc:
            errors.append(f"line {index}: sidecar parse failed: {exc}")
    return {
        "manifest_path": str(manifest),
        "sample_count": len(records),
        "sidecar_count": sidecar_count,
        "total_hours": round(total_seconds / 3600.0, 3),
        "zh_hours": round(zh_seconds / 3600.0, 3),
        "long_hours": round(long_seconds / 3600.0, 3),
        "wenet_hours": round(wenet_seconds / 3600.0, 3),
        "errors": errors,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_validate_manifest_enforces_duration_language_and_sidecar -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/asr_internal_testset.py backend/tests/test_asr_internal_testset.py
git commit -m "feat: validate ASR internal testset manifests"
```

## Task 4: Add Static MP4 Packaging

**Files:**
- Modify: `scripts/asr_internal_testset.py`
- Modify: `backend/tests/test_asr_internal_testset.py`

- [ ] **Step 1: Write failing test for ffmpeg packaging command**

Append to `backend/tests/test_asr_internal_testset.py`:

```python
def test_build_static_mp4_invokes_ffmpeg_with_static_frame(tmp_path, monkeypatch):
    module = _load_script()
    audio = tmp_path / "audio.wav"
    with module.wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        output = Path(args[-1])
        output.write_bytes(b"fake mp4")
        return object()

    monkeypatch.setattr(module, "_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = tmp_path / "out.mp4"
    module.build_static_mp4(audio, output, ["ASR v1", "WenetSpeech", "internal only"])

    assert output.read_bytes() == b"fake mp4"
    assert calls
    assert "-loop" in calls[0]
    assert "-shortest" in calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_build_static_mp4_invokes_ffmpeg_with_static_frame -q
```

Expected: FAIL because `build_static_mp4` is missing.

- [ ] **Step 3: Implement static MP4 builder**

Append imports near the top of `scripts/asr_internal_testset.py`:

```python
from PIL import Image, ImageDraw, ImageFont
```

Append functions:

```python
def _ffmpeg_exe() -> str:
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def _make_static_frame(path: Path, lines: list[str], size: tuple[int, int] = (1280, 720)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(28, 33, 40))
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype("arial.ttf", 46)
        font_body = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
    y = 96
    for index, line in enumerate(lines):
        font = font_title if index == 0 else font_body
        draw.text((80, y), line, fill=(245, 247, 250), font=font)
        y += 64 if index == 0 else 42
    draw.text((80, size[1] - 84), "MomentSeek ASR internal testset - do not redistribute media", fill=(180, 188, 199), font=font_body)
    image.save(path)


def build_static_mp4(audio_path: str | Path, output_path: str | Path, title_lines: list[str]) -> None:
    audio = Path(audio_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asr-testset-frame-") as temp_dir:
        frame = Path(temp_dir) / "frame.png"
        _make_static_frame(frame, title_lines)
        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                "1",
                "-i",
                str(frame),
                "-i",
                str(audio),
                "-shortest",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_build_static_mp4_invokes_ffmpeg_with_static_frame -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/asr_internal_testset.py backend/tests/test_asr_internal_testset.py
git commit -m "feat: package ASR samples as static MP4"
```

## Task 5: Add Platform Truth Import For 昨夜降至04

**Files:**
- Modify: `scripts/asr_internal_testset.py`
- Modify: `backend/tests/test_asr_internal_testset.py`

- [ ] **Step 1: Write failing import test**

Append to `backend/tests/test_asr_internal_testset.py`:

```python
def test_import_platform_truth_creates_manifest_and_copies_truth(tmp_path):
    module = _load_script()
    source_truth = tmp_path / "eval" / "asr" / "truth"
    source_truth.mkdir(parents=True)
    (source_truth / "yesterday.sidecar.json").write_text(
        json.dumps([
            {"start_time": 1.0, "end_time": 2.0, "text": "第一句"},
            {"start_time": 3.0, "end_time": 5.0, "text": "第二句"},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    (source_truth / "yesterday.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n第一句\n", encoding="utf-8")
    out_dir = tmp_path / "eval" / "asr" / "internal_testset"

    sample = module.import_platform_truth(
        sample_id="asr_v1_platform_yesterday_ep04",
        source_sidecar=source_truth / "yesterday.sidecar.json",
        source_srt=source_truth / "yesterday.srt",
        out_dir=out_dir,
        repo_root=tmp_path,
        source_video_path=tmp_path / "runtime-server" / "uploads" / "missing.mp4",
    )

    assert sample["sample_id"] == "asr_v1_platform_yesterday_ep04"
    assert sample["source_dataset"] == "platform_uploaded_subtitle"
    assert sample["duration_seconds"] == 4.0
    assert sample["media_available"] is False
    assert (out_dir / "truth" / "asr_v1_platform_yesterday_ep04.sidecar.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_import_platform_truth_creates_manifest_and_copies_truth -q
```

Expected: FAIL because `import_platform_truth` is missing.

- [ ] **Step 3: Implement platform truth import**

Append to `scripts/asr_internal_testset.py`:

```python
def _segments_from_sidecar(path: str | Path) -> list[TranscriptSegment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("segments", [])
    return validate_segments([
        TranscriptSegment(
            start_time=float(item.get("start_time", item.get("start", 0))),
            end_time=float(item.get("end_time", item.get("end", 0))),
            text=str(item.get("text", "")).strip(),
        )
        for item in payload
    ])


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def import_platform_truth(
    *,
    sample_id: str,
    source_sidecar: str | Path,
    source_srt: str | Path | None,
    out_dir: str | Path,
    repo_root: str | Path,
    source_video_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    output = Path(out_dir)
    truth_dir = output / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    segments = _segments_from_sidecar(source_sidecar)
    duration = max(segment.end_time for segment in segments)
    target_sidecar = truth_dir / f"{sample_id}.sidecar.json"
    target_srt = truth_dir / f"{sample_id}.srt"
    target_vtt = truth_dir / f"{sample_id}.vtt"
    shutil.copyfile(source_sidecar, target_sidecar)
    if source_srt and Path(source_srt).exists():
        shutil.copyfile(source_srt, target_srt)
    else:
        write_srt(segments, target_srt)
    write_vtt(segments, target_vtt)
    media_path = Path(source_video_path) if source_video_path else Path("runtime-server/uploads/a293b5981126444182208da7ba6274f5.mp4")
    sample = TestsetSample(
        sample_id=sample_id,
        version="v1",
        source_dataset="platform_uploaded_subtitle",
        source_url="internal://runtime-server/uploads/a293b5981126444182208da7ba6274f5",
        source_item_id="a293b5981126444182208da7ba6274f5",
        language="zh",
        text_script="Hans",
        scenario_tags=["zh", "zh_long_video", "platform_subtitle_truth", "long"],
        duration_seconds=round(duration, 3),
        media_kind="platform_uploaded_video",
        generated_media_path=_repo_relative(media_path, root),
        truth_sidecar_path=_repo_relative(target_sidecar, root),
        truth_srt_path=_repo_relative(target_srt, root),
        truth_vtt_path=_repo_relative(target_vtt, root),
        license="internal platform uploaded sample",
        internal_use_only=True,
        redistribute_original_media=False,
        source_hash=sha256_file(source_sidecar),
        generated_hash=sha256_file(media_path) if media_path.exists() else "",
        media_available=media_path.exists(),
        notes="Extracted embedded subtitle truth from 昨夜降至04; original video must not be redistributed.",
    )
    return sample.to_json()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_import_platform_truth_creates_manifest_and_copies_truth -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/asr_internal_testset.py backend/tests/test_asr_internal_testset.py
git commit -m "feat: import platform ASR subtitle truth"
```

## Task 6: Add CLI And Dataset Initialization

**Files:**
- Modify: `scripts/asr_internal_testset.py`
- Create: `eval/asr/internal_testset/README.md`
- Create: `eval/asr/internal_testset/sources.md`

- [ ] **Step 1: Implement CLI commands**

Append to `scripts/asr_internal_testset.py`:

```python
def write_validation_summary(manifest_path: str | Path, output_path: str | Path, repo_root: str | Path = ".") -> dict[str, Any]:
    summary = validate_manifest(manifest_path, repo_root=repo_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def cmd_import_yesterday(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    sample = import_platform_truth(
        sample_id=args.sample_id,
        source_sidecar=args.sidecar,
        source_srt=args.srt,
        out_dir=out_dir,
        repo_root=args.repo_root,
        source_video_path=args.video,
    )
    manifest = out_dir / "manifest.jsonl"
    records = read_jsonl(manifest) if manifest.exists() else []
    records = [record for record in records if record.get("sample_id") != sample["sample_id"]]
    records.append(sample)
    write_jsonl(records, manifest)
    print(json.dumps({"manifest": str(manifest), "sample_id": sample["sample_id"]}, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace) -> None:
    summary = write_validation_summary(args.manifest, args.out, repo_root=args.repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate the MomentSeek internal ASR testset.")
    sub = parser.add_subparsers(dest="command", required=True)
    yesterday = sub.add_parser("import-yesterday", help="Import the existing 昨夜降至04 subtitle truth into the internal testset.")
    yesterday.add_argument("--repo-root", default=".")
    yesterday.add_argument("--out", default="eval/asr/internal_testset")
    yesterday.add_argument("--sample-id", default="asr_v1_platform_yesterday_ep04")
    yesterday.add_argument("--sidecar", default="eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.sidecar.json")
    yesterday.add_argument("--srt", default="eval/asr/truth/yesterday_ep04_embedded_zh_simplified_20260708.srt")
    yesterday.add_argument("--video", default="runtime-server/uploads/a293b5981126444182208da7ba6274f5.mp4")
    yesterday.set_defaults(func=cmd_import_yesterday)
    validate = sub.add_parser("validate", help="Validate an ASR internal testset manifest.")
    validate.add_argument("--repo-root", default=".")
    validate.add_argument("--manifest", default="eval/asr/internal_testset/manifest.jsonl")
    validate.add_argument("--out", default="eval/asr/internal_testset/reports/validation_summary.json")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create README**

Create `eval/asr/internal_testset/README.md`:

```markdown
# ASR Internal Testset

This directory contains repo-safe metadata and transcript truth for the MomentSeek internal ASR testset.

Raw source audio/video and generated MP4 files are internal-only and must not be committed. Keep them under:

```text
data/asr_internal_testset/cache/
data/asr_internal_testset/generated_media/
```

First commands:

```powershell
python scripts/asr_internal_testset.py import-yesterday
python scripts/asr_internal_testset.py validate
```

The target v1 set is Chinese-first, about 12 hours, hard-capped at 15 hours. WenetSpeech should be the largest source once local access is available.
```

- [ ] **Step 3: Create sources notes**

Create `eval/asr/internal_testset/sources.md`:

```markdown
# ASR Internal Testset Sources

All media in this testset is for internal evaluation only.

| Source | Role | Use Boundary |
|---|---|---|
| WenetSpeech | Chinese video/podcast/multidomain main source | Internal, non-commercial; original audio copyright remains with original owners. Do not redistribute media. |
| AliMeeting | Chinese meetings, far/near field, overlap | Follow CC BY-SA 4.0 attribution/share-alike requirements. |
| AISHELL-1 | Clean Mandarin read speech baseline | Apache License v2.0. |
| Common Voice Chinese | Crowdsourced Chinese short clips/accent/device variation | Use validated clips and retain source metadata. |
| FLEURS | CJK and neighboring multilingual samples | CC BY 4.0, keep original text. |
| GigaSpeech/Earnings-22 | Small English real-speech/entity-number supplement | Internal supplement only; keep source-specific license notes. |
| 昨夜降至04 | Existing platform subtitle truth | Internal uploaded sample; do not redistribute original video. |
```

- [ ] **Step 4: Commit**

```powershell
git add scripts/asr_internal_testset.py eval/asr/internal_testset/README.md eval/asr/internal_testset/sources.md
git commit -m "feat: add ASR internal testset CLI"
```

## Task 7: Generate Initial Repo-Safe Testset Assets

**Files:**
- Create/modify via CLI: `eval/asr/internal_testset/manifest.jsonl`
- Create via CLI: `eval/asr/internal_testset/truth/asr_v1_platform_yesterday_ep04.sidecar.json`
- Create via CLI: `eval/asr/internal_testset/truth/asr_v1_platform_yesterday_ep04.srt`
- Create via CLI: `eval/asr/internal_testset/truth/asr_v1_platform_yesterday_ep04.vtt`
- Create via CLI: `eval/asr/internal_testset/reports/validation_summary.json`

- [ ] **Step 1: Import 昨夜降至04**

Run from repo root:

```powershell
python scripts/asr_internal_testset.py import-yesterday
```

Expected: JSON output with `sample_id` equal to `asr_v1_platform_yesterday_ep04`.

- [ ] **Step 2: Validate manifest**

Run:

```powershell
python scripts/asr_internal_testset.py validate
```

Expected: summary JSON with `errors: []`, `sample_count: 1`, and `sidecar_count: 1`.

- [ ] **Step 3: Verify sidecar is readable by the platform ASR loader**

Run:

```powershell
cd backend
python -c "from app.indexing.asr import load_sidecar; chunks=load_sidecar('../eval/asr/internal_testset/truth/asr_v1_platform_yesterday_ep04.sidecar.json'); print(len(chunks)); assert len(chunks)==653"
```

Expected: prints `653`.

- [ ] **Step 4: Commit repo-safe generated assets**

```powershell
git add eval/asr/internal_testset/manifest.jsonl eval/asr/internal_testset/truth/ eval/asr/internal_testset/reports/validation_summary.json
git commit -m "data: seed ASR internal testset with platform subtitle truth"
```

## Task 8: Add Local Source Plan Packaging For Open ASR Sources

**Files:**
- Modify: `scripts/asr_internal_testset.py`
- Modify: `backend/tests/test_asr_internal_testset.py`

- [ ] **Step 1: Write failing test for source plan packaging without real ffmpeg**

Append to `backend/tests/test_asr_internal_testset.py`:

```python
def test_build_sample_from_plan_writes_truth_manifest_and_media(tmp_path, monkeypatch):
    module = _load_script()
    audio = tmp_path / "clip.wav"
    with module.wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)

    def fake_static_mp4(_audio, output, _lines):
        Path(output).write_bytes(b"fake mp4")

    monkeypatch.setattr(module, "build_static_mp4", fake_static_mp4)
    plan = {
        "sample_id": "asr_v1_wenet_short_001",
        "source_dataset": "WenetSpeech",
        "source_url": "https://wenet-e2e.github.io/WenetSpeech/",
        "source_item_id": "local-wenet-1",
        "language": "zh",
        "text_script": "Hans",
        "scenario_tags": ["zh", "zh_real_video_podcast"],
        "license": "CC BY 4.0 with internal media restrictions",
        "segments": [
            {"audio_path": str(audio), "start_time": 0.0, "end_time": 2.0, "text": "测试文本"}
        ],
    }

    sample = module.build_sample_from_plan(plan, tmp_path / "eval/asr/internal_testset", tmp_path)

    assert sample["sample_id"] == "asr_v1_wenet_short_001"
    assert sample["duration_seconds"] == 2.0
    assert Path(tmp_path / sample["truth_sidecar_path"]).exists()
    assert Path(tmp_path / sample["generated_media_path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py::test_build_sample_from_plan_writes_truth_manifest_and_media -q
```

Expected: FAIL because `build_sample_from_plan` is missing.

- [ ] **Step 3: Implement generic local plan packaging**

Append to `scripts/asr_internal_testset.py`:

```python
def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def build_sample_from_plan(plan: dict[str, Any], out_dir: str | Path, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    output = Path(out_dir)
    sample_id = str(plan["sample_id"])
    media_dir = root / "data" / "asr_internal_testset" / "generated_media"
    truth_dir = output / "truth"
    media_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    source_segments = []
    cursor = 0.0
    source_hashes = []
    for index, item in enumerate(plan.get("segments") or []):
        audio_path = Path(str(item["audio_path"]))
        start = float(item.get("start_time", 0.0))
        end = float(item.get("end_time") or _wav_duration_seconds(audio_path))
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        duration = max(0.0, end - start)
        segments.append(TranscriptSegment(cursor, cursor + duration, text, raw_text=str(item.get("raw_text", "")), source_id=str(item.get("source_id", index))))
        source_segments.append({"audio_path": str(audio_path), "start_time": start, "end_time": end, "text": text, "source_id": str(item.get("source_id", index))})
        if audio_path.exists():
            source_hashes.append(sha256_file(audio_path))
        cursor += duration
    segments = validate_segments(segments)
    sidecar = truth_dir / f"{sample_id}.sidecar.json"
    srt = truth_dir / f"{sample_id}.srt"
    vtt = truth_dir / f"{sample_id}.vtt"
    write_sidecar_json(segments, sidecar)
    write_srt(segments, srt)
    write_vtt(segments, vtt)
    if len(source_segments) != 1 or float(source_segments[0]["start_time"]) != 0.0:
        raise ValueError("first implementation supports one full local WAV per sample; split/concat should be added only after tests")
    source_audio = Path(str(source_segments[0]["audio_path"]))
    media = media_dir / f"{sample_id}.mp4"
    build_static_mp4(
        source_audio,
        media,
        [
            f"ASR internal testset: {sample_id}",
            str(plan.get("source_dataset") or ""),
            str(plan.get("language") or ""),
        ],
    )
    sample = TestsetSample(
        sample_id=sample_id,
        version=str(plan.get("version") or "v1"),
        source_dataset=str(plan["source_dataset"]),
        source_url=str(plan.get("source_url") or ""),
        source_item_id=str(plan.get("source_item_id") or sample_id),
        language=str(plan.get("language") or "zh"),
        text_script=str(plan.get("text_script") or "Hans"),
        scenario_tags=[str(tag) for tag in plan.get("scenario_tags") or []],
        duration_seconds=round(max(segment.end_time for segment in segments), 3),
        media_kind="generated_static_mp4",
        generated_media_path=_repo_relative(media, root),
        truth_sidecar_path=_repo_relative(sidecar, root),
        truth_srt_path=_repo_relative(srt, root),
        truth_vtt_path=_repo_relative(vtt, root),
        license=str(plan.get("license") or "internal"),
        internal_use_only=True,
        redistribute_original_media=False,
        source_hash=";".join(source_hashes),
        generated_hash=sha256_file(media),
        media_available=True,
        notes=str(plan.get("notes") or "built from local source plan"),
        source_segments=source_segments,
    )
    return sample.to_json()
```

- [ ] **Step 4: Add CLI command for local plans**

Add parser command:

```python
def cmd_build_plan(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    sample = build_sample_from_plan(plan, args.out, args.repo_root)
    manifest = Path(args.out) / "manifest.jsonl"
    records = read_jsonl(manifest) if manifest.exists() else []
    records = [record for record in records if record.get("sample_id") != sample["sample_id"]]
    records.append(sample)
    write_jsonl(records, manifest)
    print(json.dumps({"sample_id": sample["sample_id"], "manifest": str(manifest)}, ensure_ascii=False))
```

Register it in `build_parser()`:

```python
    build_plan = sub.add_parser("build-plan", help="Build one sample from a local JSON source plan.")
    build_plan.add_argument("--repo-root", default=".")
    build_plan.add_argument("--out", default="eval/asr/internal_testset")
    build_plan.add_argument("--plan", required=True)
    build_plan.set_defaults(func=cmd_build_plan)
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 6: Commit**

```powershell
git add scripts/asr_internal_testset.py backend/tests/test_asr_internal_testset.py
git commit -m "feat: build ASR testset samples from local plans"
```

## Final Verification

- [ ] Run script tests:

```powershell
cd backend
python -m pytest tests/test_asr_internal_testset.py -q
```

Expected: PASS.

- [ ] Validate generated initial dataset:

```powershell
python scripts/asr_internal_testset.py validate
```

Expected: `errors` is an empty list.

- [ ] Check platform loader compatibility:

```powershell
cd backend
python -c "from app.indexing.asr import load_sidecar; chunks=load_sidecar('../eval/asr/internal_testset/truth/asr_v1_platform_yesterday_ep04.sidecar.json'); print(len(chunks)); assert len(chunks)==653"
```

Expected: prints `653`.

- [ ] Review git status:

```powershell
git status --short
```

Expected: only intentional files are modified/untracked. Do not stage unrelated existing ASR experiment files.

## Follow-Up After This Plan

Once WenetSpeech/AliMeeting/AISHELL/Common Voice/FLEURS local caches are available, create one JSON build plan per generated sample under `data/asr_internal_testset/tmp/plans/` and run:

```powershell
python scripts/asr_internal_testset.py build-plan --plan data/asr_internal_testset/tmp/plans/<sample>.json
python scripts/asr_internal_testset.py validate
```

The first full data build must target:

```text
total: 10-15h
recommended: about 12h
WenetSpeech: >= 4h
Chinese: >= 75%
long samples: 35%-45%
```
