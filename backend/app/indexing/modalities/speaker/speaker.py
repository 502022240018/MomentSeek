from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.vector_store.milvus.milvus_indexer import MilvusWriteContext


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
    """Use sklearn's native density backend when the eigengap collapses."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    values = _normalize(embeddings)
    components = min(32, len(values) - 1, values.shape[1])
    reduced = PCA(n_components=components, whiten=True, random_state=0).fit_transform(values)
    labels = np.asarray(
        HDBSCAN(min_samples=20, min_cluster_size=10).fit_predict(reduced),
        dtype=np.int32,
    )
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


def _track_count(track_indices: np.ndarray) -> int:
    valid_tracks = np.asarray(track_indices, dtype=np.int32)
    valid_tracks = valid_tracks[valid_tracks >= 0]
    return int(valid_tracks.max()) + 1 if valid_tracks.size else 0


def _asr_source_from_milvus(
    milvus_ctx: "MilvusWriteContext",
    asr_asset_version: str,
) -> tuple[np.ndarray, list[str]]:
    """Load one complete, version-pinned ASR timeline from Milvus."""
    if not str(asr_asset_version).strip():
        raise RuntimeError("Milvus speaker build requires the published ASR asset version")
    collection = milvus_ctx.client.collection_for("asr")
    expression = (
        f"video_id == {json.dumps(milvus_ctx.video_id)} and "
        f"asset_version == {json.dumps(str(asr_asset_version))}"
    )
    fields = ["segment_idx", "start_ms", "end_ms", "text"]
    rows: list[dict] = []
    from app.core.settings import get_settings

    timeout = get_settings().milvus_query_timeout_seconds
    iterator_factory = getattr(collection, "query_iterator", None)
    if callable(iterator_factory):
        iterator = iterator_factory(
            batch_size=2000,
            expr=expression,
            output_fields=fields,
            timeout=timeout,
        )
        try:
            while True:
                page = iterator.next()
                if not page:
                    break
                rows.extend(page)
        finally:
            iterator.close()
    else:
        rows = collection.query(
            expr=expression,
            output_fields=fields,
            limit=16_384,
            timeout=timeout,
        )

    by_index: dict[int, tuple[int, int, str]] = {}
    for row in rows:
        raw_index = row.get("segment_idx")
        if raw_index is None or int(raw_index) < 0:
            raise RuntimeError(
                f"Milvus ASR row has an invalid segment_idx for video {milvus_ctx.video_id}"
            )
        index = int(raw_index)
        value = (
            int(row.get("start_ms") or 0),
            int(row.get("end_ms") or 0),
            str(row.get("text") or ""),
        )
        previous = by_index.get(index)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"Milvus ASR contains conflicting duplicate segment_idx={index} "
                f"for video {milvus_ctx.video_id}"
            )
        by_index[index] = value

    indices = sorted(by_index)
    if indices != list(range(len(indices))):
        raise RuntimeError(
            f"Milvus ASR coverage is sparse for video {milvus_ctx.video_id}: got {indices}"
        )
    if not indices:
        return np.empty((0, 2), dtype=np.int32), []
    return (
        np.asarray([[by_index[index][0], by_index[index][1]] for index in indices], dtype=np.int32),
        [by_index[index][2] for index in indices],
    )


def _load_3dspeaker(repo: Path):
    script = repo / "speakerlab" / "bin" / "infer_diarization.py"
    if not script.exists():
        raise RuntimeError(f"3D-Speaker not found: {repo}")
    repo_text = str(repo.resolve())
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    return importlib.import_module("app.indexing.modalities.speaker.speaker_3dspeaker_runtime")


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
    *, video_path: str, working_dir: str,
    model_repo: str, model_cache_dir: str, device: str = "cuda",
    milvus_ctx: "MilvusWriteContext",
    asr_asset_version: str,
) -> dict:
    started = time.perf_counter()
    if milvus_ctx is None:
        raise RuntimeError("Speaker indexing requires a Milvus write context")
    times, texts = _asr_source_from_milvus(milvus_ctx, asr_asset_version)
    eligible = np.asarray([
        index for index, (bounds, text) in enumerate(zip(times, texts))
        if bounds[1] > bounds[0] and _meaningful(text)
    ], dtype=np.int32)
    if not len(eligible):
        # Publish an explicit empty version so the catalog and manifest never
        # advertise a stale speaker index from an earlier ASR run.
        result = {
            "utterances": 0,
            "tracks": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        from app.vector_store.milvus.milvus_indexer import write_modality_from_memory

        result["milvus_rows"] = write_modality_from_memory(
            milvus_ctx,
            "speaker",
            {
                "utterance_embeddings": np.empty((0, EMBEDDING_DIM), dtype=np.float32),
                "utterance_times_ms": np.empty((0, 2), dtype=np.int32),
                "utterance_refs": np.empty((0, 2), dtype=np.int32),
            },
        )
        return result

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
            clustering_backend = "sklearn_hdbscan_fallback"
    utterance_times, chunk_indices, track_indices = _adaptive_turn_units(
        chunks, track_indices, times, eligible,
    )
    if not len(utterance_times):
        raise RuntimeError("说话人 turn 无法与 ASR 时间轴对齐")
    adaptive_chunks = [[float(start) / 1000, float(end) / 1000] for start, end in utterance_times]
    embeddings = _extract_embeddings(pipeline, adaptive_chunks, waveform)
    embeddings = _normalize(embeddings)
    refs = np.column_stack((chunk_indices, track_indices)).astype(np.int32, copy=False)
    result = {
        "utterances": len(embeddings),
        "tracks": _track_count(track_indices),
        "embedding_dim": embeddings.shape[1] if len(embeddings) else EMBEDDING_DIM,
    }
    final = {
        **result,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "embedding_model": "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "diarization_model": "modelscope/3D-Speaker",
        "embedding_space": "3dspeaker-campplus-zh-en-192-v1",
        "clustering_backend": clustering_backend,
        "segmentation": "adaptive_speaker_turn_asr_boundary",
    }
    from app.vector_store.milvus.milvus_indexer import write_modality_from_memory

    final["milvus_rows"] = write_modality_from_memory(
        milvus_ctx,
        "speaker",
        {
            "utterance_embeddings": embeddings,
            "utterance_times_ms": np.asarray(utterance_times, dtype=np.int32),
            "utterance_refs": refs,
        },
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
