from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.indexing.milvus_indexer import MilvusWriteContext


EMBEDDING_DIM = 192


def _meaningful(text: str) -> bool:
    return sum(character.isalnum() for character in text) >= 2


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.size == 0:
        return rows.reshape(0, EMBEDDING_DIM)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def _assign_tracks(times_ms: np.ndarray, turns: list[list[float]]) -> np.ndarray:
    assigned = np.full((len(times_ms),), -1, dtype=np.int32)
    for index, (start_ms, end_ms) in enumerate(times_ms):
        overlap: dict[int, float] = {}
        for start, end, track in turns:
            duration = max(0.0, min(float(end_ms), end * 1000) - max(float(start_ms), start * 1000))
            if duration:
                overlap[int(track)] = overlap.get(int(track), 0.0) + duration
        if overlap:
            assigned[index] = max(overlap, key=overlap.get)
    return assigned


def _asr_references(window_times_ms: np.ndarray, asr_times_ms: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Map a short voice window to the ASR chunk with the largest overlap."""
    references = np.full((len(window_times_ms),), -1, dtype=np.int32)
    for index, (start_ms, end_ms) in enumerate(window_times_ms):
        overlaps = np.maximum(
            0,
            np.minimum(end_ms, asr_times_ms[eligible, 1]) - np.maximum(start_ms, asr_times_ms[eligible, 0]),
        )
        if overlaps.size and overlaps.max() > 0:
            references[index] = int(eligible[int(np.argmax(overlaps))])
    return references


def _adaptive_turn_units(
    chunks: list[list[float]], labels: np.ndarray, asr_times_ms: np.ndarray, eligible: np.ndarray,
    *, minimum_voice_ms: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exclusive speaker turns and split them only at natural ASR boundaries."""
    turns: list[list[float]] = []
    for start, end, label in [
        [float(start), float(end), int(label)]
        for (start, end), label in zip(chunks, labels)
    ]:
        if not turns or start > turns[-1][1]:
            turns.append([start, end, label])
        elif int(label) == int(turns[-1][2]):
            turns[-1][1] = max(turns[-1][1], end)
        else:
            boundary = (turns[-1][1] + start) / 2
            turns[-1][1] = boundary
            turns.append([boundary, end, label])
    units: list[tuple[int, int, int, int]] = []
    for turn_start, turn_end, track_id in turns:
        turn_start_ms, turn_end_ms = round(turn_start * 1000), round(turn_end * 1000)
        for chunk_index in eligible:
            start_ms = max(turn_start_ms, int(asr_times_ms[chunk_index, 0]))
            end_ms = min(turn_end_ms, int(asr_times_ms[chunk_index, 1]))
            if end_ms - start_ms >= minimum_voice_ms:
                units.append((start_ms, end_ms, int(chunk_index), int(track_id)))
    if not units:
        return (
            np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.asarray([[start, end] for start, end, _, _ in units], dtype=np.int32),
        np.asarray([chunk_index for _, _, chunk_index, _ in units], dtype=np.int32),
        np.asarray([track_id for _, _, _, track_id in units], dtype=np.int32),
    )


def _density_fallback_labels(embeddings: np.ndarray) -> np.ndarray:
    """Use the repository's density backend when spectral eigengap collapses to one speaker."""
    from speakerlab.process.cluster import CommonClustering

    cluster = CommonClustering(
        cluster_type="umap_hdbscan", n_neighbors=20, n_components=60,
        min_samples=20, min_cluster_size=10, mer_cos=0.8,
    )
    labels = np.asarray(cluster(embeddings), dtype=np.int32)
    valid = np.unique(labels[labels >= 0])
    if not len(valid):
        return np.zeros((len(embeddings),), dtype=np.int32)
    centers = _normalize(np.stack([embeddings[labels == label].mean(axis=0) for label in valid]))
    noise = np.flatnonzero(labels < 0)
    if len(noise):
        labels[noise] = valid[np.argmax(_normalize(embeddings[noise]) @ centers.T, axis=1)]
    # Compact arbitrary density labels into the on-disk track id range.
    mapping = {int(label): index for index, label in enumerate(np.unique(labels))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def _track_caches(embeddings: np.ndarray, track_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_tracks = track_indices[track_indices >= 0]
    track_count = int(valid_tracks.max()) + 1 if valid_tracks.size else 0
    tracks = np.zeros((track_count, embeddings.shape[1] if embeddings.ndim == 2 else EMBEDDING_DIM), dtype=np.float32)
    representatives = np.full((track_count,), -1, dtype=np.int32)
    for track in range(track_count):
        indices = np.flatnonzero(track_indices == track)
        if not len(indices):
            continue
        center = _normalize(embeddings[indices].mean(axis=0, keepdims=True))[0]
        tracks[track] = center
        representatives[track] = int(indices[np.argmax(embeddings[indices] @ center)])
    return tracks, representatives


def save_speaker_index(
    output_path: str | Path,
    *,
    utterance_times_ms: np.ndarray,
    utterance_embeddings: np.ndarray,
    asr_chunk_indices: np.ndarray,
    auto_track_indices: np.ndarray,
) -> dict:
    times = np.asarray(utterance_times_ms, dtype=np.int32)
    embeddings = _normalize(utterance_embeddings)
    chunk_indices = np.asarray(asr_chunk_indices, dtype=np.int32)
    track_indices = np.asarray(auto_track_indices, dtype=np.int32)
    count = len(embeddings)
    if times.shape != (count, 2):
        raise ValueError("utterance_times_ms must have shape [N, 2]")
    if chunk_indices.shape != (count,) or track_indices.shape != (count,):
        raise ValueError("utterance references must have shape [N]")
    refs = np.column_stack((chunk_indices, track_indices)).astype(np.int32, copy=False)
    tracks, representatives = _track_caches(embeddings, track_indices)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        target,
        utterance_embeddings=embeddings.astype(np.float16),
        utterance_times_ms=times,
        utterance_refs=refs,
        track_embeddings=tracks.astype(np.float16),
        track_representative_indices=representatives,
    )

    return {"utterances": count, "tracks": len(tracks), "embedding_dim": embeddings.shape[1] if count else EMBEDDING_DIM}


def load_speaker_index(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "utterance_embeddings", "utterance_times_ms", "utterance_refs",
            "track_embeddings", "track_representative_indices",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"speaker.npz missing fields: {sorted(missing)}")
        result = {name: data[name] for name in required}
    count = len(result["utterance_embeddings"])
    if result["utterance_times_ms"].shape != (count, 2) or result["utterance_refs"].shape != (count, 2):
        raise ValueError("speaker.npz utterance arrays are not aligned")
    return result


def _load_3dspeaker(repo: Path):
    script = repo / "speakerlab" / "bin" / "infer_diarization.py"
    if not script.exists():
        raise RuntimeError(f"3D-Speaker not found: {repo}")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("momentseek_3dspeaker", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_wav(video_path: str, wav_path: Path) -> None:
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", video_path,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "ffmpeg audio extraction failed")


def _extract_embeddings(pipeline, chunks: list[list[float]], waveform, batch_size: int = 64) -> np.ndarray:
    """Avoid padding every utterance in a long video to one global maximum."""
    blocks = [
        pipeline.do_emb_extraction(chunks[start:start + batch_size], waveform)
        for start in range(0, len(chunks), batch_size)
    ]
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, EMBEDDING_DIM), np.float32)


def build_speaker_index(
    *, video_path: str, asr_path: str, output_path: str, working_dir: str,
    model_repo: str, model_cache_dir: str, device: str = "cuda",
    milvus_ctx: "MilvusWriteContext | None" = None,
) -> dict:
    started = time.perf_counter()
    _asr_file = Path(asr_path)
    if not _asr_file.exists() or _asr_file.stat().st_size == 0:
        # asr.npz was removed after Milvus ingestion (Milvus-only mode).
        # Read chunk times and texts from the ASR collection instead.
        if milvus_ctx is None:
            raise RuntimeError(
                "asr.npz is absent but no milvus_ctx provided; "
                "cannot read ASR data for speaker indexing"
            )
        from app.indexing.milvus_client import get_milvus_client
        _client = get_milvus_client()
        _rows = _client.collection_for("asr").query(
            expr=f'video_id == "{milvus_ctx.video_id}"',
            output_fields=["segment_idx", "start_ms", "end_ms", "text"],
        )
        # Deduplicate by segment_idx and rebuild arrays preserving original ordering.
        _chunks_by_idx: dict[int, tuple[int, int, str]] = {}
        for _r in _rows:
            _idx = int(_r.get("segment_idx") or 0)
            if _idx not in _chunks_by_idx:
                _chunks_by_idx[_idx] = (int(_r["start_ms"]), int(_r["end_ms"]), str(_r.get("text") or ""))
        if _chunks_by_idx:
            _max_idx = max(_chunks_by_idx)
            _times_list = [[0, 0]] * (_max_idx + 1)
            _texts_list: list[str] = [""] * (_max_idx + 1)
            for _idx, (_s, _e, _t) in _chunks_by_idx.items():
                _times_list[_idx] = [_s, _e]
                _texts_list[_idx] = _t
            times = np.asarray(_times_list, dtype=np.int32)
            texts = _texts_list
        else:
            times = np.empty((0, 2), dtype=np.int32)
            texts = []
    else:
        with np.load(asr_path, allow_pickle=True) as asr:
            times = asr["chunk_times_ms"].astype(np.int32)
            texts = [str(value) for value in asr["texts"]]
    eligible = np.asarray([
        index for index, (bounds, text) in enumerate(zip(times, texts))
        if bounds[1] > bounds[0] and _meaningful(text)
    ], dtype=np.int32)
    if not len(eligible):
        # ASR 中无有效人声片段（纯背景音乐/无音频），跳过 speaker 索引
        # 不写入任何文件（包括 NPZ），直接返回空结果
        return {
            "utterances": 0,
            "tracks": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    module = _load_3dspeaker(Path(model_repo))
    pipeline = module.Diarization3Dspeaker(device=device, model_cache_dir=model_cache_dir)
    work = Path(working_dir)
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "speaker_audio.wav"
    _extract_wav(video_path, wav_path)
    waveform = module.load_audio(str(wav_path), None, pipeline.fs)
    vad_regions = pipeline.do_vad(waveform)
    chunks = [chunk for start, end in vad_regions for chunk in pipeline.chunk(start, end)]
    if not chunks:
        raise RuntimeError("音频中没有可用于说话人索引的有效语音")
    embeddings = _extract_embeddings(pipeline, chunks, waveform)
    track_indices = np.asarray(pipeline.cluster(embeddings), dtype=np.int32)
    clustering_backend = "spectral"
    if len(track_indices) >= 40 and len(np.unique(track_indices)) == 1:
        fallback = _density_fallback_labels(embeddings)
        if len(np.unique(fallback)) > 1:
            track_indices = fallback
            clustering_backend = "umap_hdbscan_fallback"
    utterance_times, chunk_indices, track_indices = _adaptive_turn_units(
        chunks, track_indices, times, eligible,
    )
    if not len(utterance_times):
        raise RuntimeError("说话人 turn 无法与 ASR 时间轴对齐")
    adaptive_chunks = [[float(start) / 1000, float(end) / 1000] for start, end in utterance_times]
    embeddings = _extract_embeddings(pipeline, adaptive_chunks, waveform)
    result = save_speaker_index(
        output_path, utterance_times_ms=utterance_times,
        utterance_embeddings=embeddings, asr_chunk_indices=chunk_indices,
        auto_track_indices=track_indices,
    )
    final = {
        **result,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "embedding_model": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "diarization_model": "modelscope/3D-Speaker",
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
        "clustering_backend": clustering_backend,
        "segmentation": "adaptive_speaker_turn_asr_boundary",
    }
    if milvus_ctx is not None:
        # P2: write directly from in-memory arrays — no NPZ round-trip.
        # save_speaker_index already wrote the NPZ (needed for track caches);
        # we reuse it as the recovery artifact, so recovery_save_fn is a no-op.
        from app.indexing.milvus_indexer import write_modality_from_memory
        norm_emb = _normalize(np.asarray(embeddings, dtype=np.float32))
        refs_arr = np.column_stack((
            np.asarray(chunk_indices, dtype=np.int32),
            np.asarray(track_indices, dtype=np.int32),
        ))
        write_modality_from_memory(
            milvus_ctx, "speaker",
            {
                "utterance_embeddings": norm_emb,
                "utterance_times_ms":   np.asarray(utterance_times, dtype=np.int32),
                "utterance_refs":       refs_arr,
            },
            recovery_save_fn=None,  # NPZ already written by save_speaker_index above
        )
    return final


def encode_voice_query(
    audio_path: str, *, model_repo: str, model_cache_dir: str, device: str = "cuda"
) -> np.ndarray:
    """Extract multiple query embeddings without averaging potentially different speakers."""
    module = _load_3dspeaker(Path(model_repo))
    pipeline = module.Diarization3Dspeaker(device=device, model_cache_dir=model_cache_dir)
    waveform = module.load_audio(audio_path, None, pipeline.fs)
    regions = pipeline.do_vad(waveform)
    chunks = [chunk for start, end in regions for chunk in pipeline.chunk(start, end)]
    if not chunks and waveform.shape[-1] >= pipeline.fs // 2:
        chunks = [[0.0, waveform.shape[-1] / pipeline.fs]]
    if not chunks:
        raise ValueError("上传文件中没有足够的有效语音")
    return _normalize(_extract_embeddings(pipeline, chunks, waveform))
