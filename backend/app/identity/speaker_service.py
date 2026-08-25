from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from app.catalog.db import Catalog
from app.core.settings import Settings, get_settings
from app.vector_store.milvus.milvus_schema import EMBEDDING_DIMS
from app.vector_store.milvus.row_contract import (
    required_nonnegative_int_field,
    required_time_window,
)


SPEAKER_PREVIEW_UTTERANCES = 5


class SpeakerMilvusCoverageError(RuntimeError):
    """Milvus speaker rows do not represent one complete, index-safe video."""


def _validated_voice_vectors(vectors: np.ndarray, *, source: str) -> np.ndarray:
    """Return normalized CAM++ vectors or reject the whole reference.

    Planner evidence must never be built from a partly corrupt reference set.
    Keeping the validation here also gives the standalone voice-search API and
    Planner Lab exactly the same input contract.
    """
    values = np.asarray(vectors, dtype=np.float32)
    expected_dim = int(EMBEDDING_DIMS["speaker"])
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or not len(values) or values.shape[1] != expected_dim:
        raise ValueError(
            f"{source} 声纹向量维度无效，期望 (*, {expected_dim})，实际 {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{source} 声纹向量包含非有限值")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{source} 声纹向量包含零向量")
    return values / norms


def entity_voice_embeddings(catalog: Catalog, entity_id: str) -> np.ndarray:
    """Load trusted entity voice samples stored in SQLite BLOBs."""
    entity = catalog.get_entity(entity_id)
    if not entity:
        raise ValueError("人物不存在")
    samples = catalog.list_voice_sample_embeddings(entity_id)
    if not samples:
        raise ValueError(f"人物“{entity.get('name') or entity_id}”还没有注册声纹")
    expected_dim = int(EMBEDDING_DIMS["speaker"])
    vectors: list[np.ndarray] = []
    for sample in samples:
        payload = sample.get("voice_embedding")
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError(f"声纹样本 {sample.get('id')} 缺少内联向量")
        vector = np.frombuffer(payload, dtype=np.float32)
        if vector.size != expected_dim:
            raise ValueError(
                f"声纹样本 {sample.get('id')} 维度无效，期望 {expected_dim}，实际 {vector.size}"
            )
        vectors.append(vector.copy())
    return _validated_voice_vectors(np.vstack(vectors), source="人物库")


def encode_voice_reference_file(settings: Settings, source_path: Path) -> np.ndarray:
    """Decode an uploaded media file and encode it once as a voice reference."""
    from app.indexing.modalities.speaker.speaker import encode_voice_query

    wav_path = source_path.with_suffix(".voice.wav")
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise ValueError(process.stderr.strip() or "无法读取上传声音")
        vectors = encode_voice_query(
            str(wav_path),
            model_repo=str(
                settings.resolve_path(settings.app_model_dir / settings.speaker_model_repo)
            ),
            model_cache_dir=str(
                settings.resolve_path(
                    settings.app_model_dir / settings.speaker_model_cache_dir
                )
            ),
            device=settings.speaker_device,
        )
        return _validated_voice_vectors(vectors, source="上传音频")
    finally:
        wav_path.unlink(missing_ok=True)


def resolve_voice_reference_vectors(
    catalog: Catalog,
    settings: Settings,
    reference: dict,
    *,
    upload_path: Path | None = None,
) -> np.ndarray:
    """Resolve one already schema-validated Planner voice reference."""
    kind = str(reference.get("kind") or "")
    if kind == "upload":
        if upload_path is None:
            raise ValueError("上传声音引用缺少音频文件")
        return encode_voice_reference_file(settings, upload_path)
    if kind == "entity":
        return entity_voice_embeddings(catalog, str(reference.get("entity_id") or ""))
    if kind == "utterance":
        vector = speaker_utterance_embedding(
            catalog,
            str(reference.get("video_id") or ""),
            int(reference.get("utterance_index", -1)),
        )
        return _validated_voice_vectors(vector, source="视频说话片段")
    raise ValueError("不支持的声音引用类型")


def get_milvus_client():
    from app.vector_store.milvus.milvus_client import get_milvus_client as factory

    return factory()


def ensure_milvus_reachable() -> None:
    from app.vector_store.milvus.milvus_client import ensure_milvus_reachable as ensure

    ensure()


def _published_modality(
    catalog: Catalog,
    video_id: str,
    modality: str,
    *,
    require_rows: bool = True,
) -> dict:
    publication = catalog.get_modality_publication(video_id, modality)
    if not publication or publication.get("status") != "ready":
        raise SpeakerMilvusCoverageError(
            f"Milvus {modality} version is not published for video {video_id}"
        )
    version = publication.get("asset_version")
    if version is None or not str(version).strip():
        raise SpeakerMilvusCoverageError(
            f"Milvus {modality} version is not published for video {video_id}"
        )
    try:
        row_count = int(publication.get("row_count"))
    except (TypeError, ValueError):
        raise SpeakerMilvusCoverageError(
            f"Milvus {modality} publication has an invalid row_count for video {video_id}"
        ) from None
    if row_count < 0:
        raise SpeakerMilvusCoverageError(
            f"Milvus {modality} publication has a negative row_count for video {video_id}"
        )
    if require_rows and row_count == 0:
        raise SpeakerMilvusCoverageError(
            f"Milvus {modality} index for video {video_id} has 0 rows"
        )
    return publication


def _expected_speaker_utterances(publication: dict, video_id: str) -> int:
    value = publication.get("utterances", publication.get("row_count"))
    if isinstance(value, bool):
        value = None
    try:
        expected = int(value)
        row_count = int(publication["row_count"])
    except (KeyError, TypeError, ValueError):
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker publication has invalid utterance metadata for video {video_id}"
        ) from None
    if expected < 0 or expected != row_count:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker publication count mismatch for video {video_id}: "
            f"utterances={expected}, row_count={row_count}"
        )
    return expected


def _published_speaker_for_current_asr(
    catalog: Catalog,
    video_id: str,
    *,
    require_rows: bool = True,
) -> dict:
    """Return Speaker only when it was built from the current ready ASR."""
    speaker = _published_modality(
        catalog, video_id, "speaker", require_rows=require_rows
    )
    asr = _published_modality(catalog, video_id, "asr", require_rows=False)
    source_version = str(
        speaker.get("source_asr_asset_version") or ""
    ).strip()
    current_version = str(asr["asset_version"])
    if not source_version:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker publication is missing source_asr_asset_version "
            f"for video {video_id}"
        )
    if source_version != current_version:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker/ASR publication mismatch for video {video_id}: "
            f"speaker source={source_version}, current ASR={current_version}"
        )
    return speaker


def _published_asset_version(catalog: Catalog, video_id: str, modality: str) -> str:
    publication = (
        _published_speaker_for_current_asr(catalog, video_id)
        if modality == "speaker"
        else _published_modality(catalog, video_id, modality)
    )
    return str(publication["asset_version"])


def _milvus_rows(
    catalog: Catalog,
    video_id: str,
    modality: str,
    fields: list[str],
    *,
    publication: dict | None = None,
) -> list[dict]:
    published = publication or _published_modality(catalog, video_id, modality)
    ensure_milvus_reachable()
    collection = get_milvus_client().collection_for(modality)
    version = str(published["asset_version"])
    expression = (
        f"video_id == {json.dumps(video_id)} and "
        f"asset_version == {json.dumps(version)}"
    )
    rows: list[dict] = []
    timeout = get_settings().milvus_query_timeout_seconds
    if hasattr(collection, "query_iterator"):
        iterator = collection.query_iterator(
            batch_size=2000,
            expr=expression,
            output_fields=fields,
            timeout=timeout,
        )
        try:
            while True:
                try:
                    page = iterator.next()
                except StopIteration:
                    break
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
    return rows


def _texts_from_milvus(catalog: Catalog, video_id: str) -> list[str]:
    try:
        publication = _published_modality(catalog, video_id, "asr")
        rows = _milvus_rows(
            catalog,
            video_id,
            "asr",
            ["segment_idx", "text"],
            publication=publication,
        )
    except SpeakerMilvusCoverageError:
        raise
    except Exception as exc:
        raise SpeakerMilvusCoverageError(
            f"Milvus ASR text is unavailable for video {video_id}: {exc}"
        ) from exc
    indexed_rows: list[tuple[int, str]] = []
    for row in rows:
        value = row.get("segment_idx")
        if value is None or int(value) < 0:
            raise SpeakerMilvusCoverageError(
                f"Milvus ASR row has an invalid segment_idx for video {video_id}"
            )
        indexed_rows.append((int(value), str(row.get("text") or "")))
    indexed_rows.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed_rows]
    expected_count = int(publication["row_count"])
    expected_indices = list(range(expected_count))
    if indices != expected_indices:
        raise SpeakerMilvusCoverageError(
            f"Milvus ASR coverage is sparse, duplicated, or incomplete for video {video_id}: "
            f"expected {expected_indices}, got {indices}"
        )
    return [text for _, text in indexed_rows]


def _speaker_data_from_milvus(
    catalog: Catalog,
    video_id: str,
    *,
    expected_utterances: int | None = None,
    publication: dict | None = None,
) -> dict[str, np.ndarray] | None:
    rows = _milvus_rows(
        catalog,
        video_id,
        "speaker",
        [
            "utterance_idx",
            "start_ms",
            "end_ms",
            "asr_chunk_idx",
            "track_id",
            "embedding",
        ],
        publication=publication,
    )
    if not rows:
        return None
    indexed_rows: list[tuple[int, dict]] = []
    for row in rows:
        try:
            index = required_nonnegative_int_field(row, "utterance_idx")
            required_time_window(row)
            required_nonnegative_int_field(row, "asr_chunk_idx")
            required_nonnegative_int_field(row, "track_id")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpeakerMilvusCoverageError(
                f"Milvus speaker row has invalid metadata for video {video_id}: {exc}"
            ) from exc
        indexed_rows.append((index, row))
    indexed_rows.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed_rows]
    expected_indices = list(range(len(indexed_rows)))
    if indices != expected_indices:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker coverage is sparse or duplicated for video {video_id}: "
            f"expected {expected_indices}, got {indices}"
        )
    if expected_utterances is not None and len(indexed_rows) != expected_utterances:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker coverage is incomplete for video {video_id}: "
            f"expected {expected_utterances}, got {len(indexed_rows)}"
        )
    rows = [row for _, row in indexed_rows]
    embeddings = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    expected_dim = int(EMBEDDING_DIMS["speaker"])
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != expected_dim
        or not np.all(np.isfinite(embeddings))
    ):
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker embeddings are invalid for video {video_id}; "
            f"expected finite vectors with dimension {expected_dim}"
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker embeddings contain zero vectors for video {video_id}"
        )
    # Embeddings are written as unit vectors (speaker.py _normalize), so this
    # re-normalisation is defensive: it guards against any floating-point
    # round-trip drift that could accumulate before reaching this point.
    embeddings /= norms
    times = np.asarray([
        list(required_time_window(row))
        for row in rows
    ], dtype=np.int32)
    refs = np.asarray([
        [
            required_nonnegative_int_field(row, "asr_chunk_idx"),
            required_nonnegative_int_field(row, "track_id"),
        ]
        for row in rows
    ], dtype=np.int32)
    track_ids = refs[:, 1]
    track_count = int(track_ids.max()) + 1 if len(track_ids) else 0
    # The per-track centroid is needed only to pick each track's representative
    # utterance; video_speakers never consumes the centroid vectors themselves
    # (it re-ranks over the overlay-corrected membership in _rank_speaker_utterances).
    # So compute the centroid as a local and keep only the representative indices —
    # materialising a track_embeddings matrix here was dead weight.
    representatives = np.full((track_count,), -1, dtype=np.int32)
    for track_id in range(track_count):
        members = np.flatnonzero(track_ids == track_id)
        if not len(members):
            continue
        center = embeddings[members].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        representatives[track_id] = int(
            members[int(np.argmax(embeddings[members] @ center))]
        )
    return {
        "utterance_embeddings": embeddings,
        "utterance_times_ms": times,
        "utterance_refs": refs,
        "track_representative_indices": representatives,
    }


def _load_speaker_data(catalog: Catalog, video_id: str) -> dict[str, np.ndarray]:
    """Load online speaker state exclusively from Milvus."""
    publication = _published_speaker_for_current_asr(catalog, video_id)
    data = _speaker_data_from_milvus(
        catalog,
        video_id,
        expected_utterances=_expected_speaker_utterances(publication, video_id),
        publication=publication,
    )
    if data is None:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker data is missing for video {video_id}"
        )
    return data


def speaker_utterance_embedding(
    catalog: Catalog,
    video_id: str,
    utterance_index: int,
) -> np.ndarray:
    """Read one utterance from the Milvus source of truth."""
    data = _load_speaker_data(catalog, video_id)
    try:
        return data["utterance_embeddings"][utterance_index].astype(np.float32)
    except IndexError as exc:
        raise IndexError("声音片段不存在") from exc


def _speaker_preview_bounds_ms(
    start_ms: int,
    end_ms: int,
    *,
    duration_seconds: float = 0.0,
) -> tuple[int, int]:
    """Expand an immutable evidence interval into a useful playback interval."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("声音证据时间边界无效")
    settings = get_settings()
    padding_ms = round(max(0.0, settings.speaker_preview_padding_seconds) * 1000)
    minimum_ms = round(max(0.0, settings.speaker_preview_min_seconds) * 1000)
    maximum_ms = round(max(0.0, settings.speaker_preview_max_seconds) * 1000)
    if maximum_ms > 0:
        maximum_ms = max(maximum_ms, minimum_ms)
    desired_ms = max(minimum_ms, end_ms - start_ms + 2 * padding_ms)
    if maximum_ms > 0:
        desired_ms = min(desired_ms, maximum_ms)
    desired_ms = max(end_ms - start_ms, desired_ms)
    center_ms = (start_ms + end_ms) // 2
    preview_start = max(0, center_ms - desired_ms // 2)
    preview_end = preview_start + desired_ms
    duration_ms = round(max(0.0, duration_seconds) * 1000)
    if duration_ms > 0 and preview_end > duration_ms:
        preview_end = duration_ms
        preview_start = max(0, preview_end - desired_ms)
    preview_start = min(preview_start, start_ms)
    preview_end = max(preview_end, end_ms)
    if duration_ms > 0:
        preview_end = min(preview_end, duration_ms)
    return int(preview_start), int(preview_end)


def _speaker_clip_url(video_id: str, start_ms: int, end_ms: int) -> str:
    return f"/api/videos/{video_id}/clip?{urlencode({'start': start_ms / 1000, 'end': end_ms / 1000})}"


def _speaker_utterances(
    data: dict,
    texts: list[str],
    overlays: dict,
    video_id: str,
    *,
    duration_seconds: float = 0.0,
) -> list[dict]:
    refs = data["utterance_refs"].astype(np.int32)
    times = data["utterance_times_ms"].astype(np.int32)
    utterances = []
    for index, ((start_ms, end_ms), (chunk_index, auto_track)) in enumerate(zip(times, refs)):
        if not 0 <= int(chunk_index) < len(texts):
            raise SpeakerMilvusCoverageError(
                f"Milvus speaker row {index} references missing ASR chunk "
                f"{int(chunk_index)} for video {video_id}"
            )
        override = overlays["utterances"].get(index, {})
        final_track = override.get("corrected_track_id")
        if final_track is None and index not in overlays["utterances"]:
            final_track = int(auto_track)
        preview_start_ms, preview_end_ms = _speaker_preview_bounds_ms(
            int(start_ms),
            int(end_ms),
            duration_seconds=duration_seconds,
        )
        utterances.append({
            "index": index,
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "preview_start_ms": preview_start_ms,
            "preview_end_ms": preview_end_ms,
            "asr_chunk_index": int(chunk_index),
            "text": texts[int(chunk_index)],
            "auto_track_id": int(auto_track),
            "track_id": final_track,
            "searchable": bool(override.get("searchable", 1)),
            "clip_url": _speaker_clip_url(video_id, preview_start_ms, preview_end_ms),
        })
    return utterances


def _speaker_track_ids(data: dict, utterances: list[dict]) -> set[int]:
    track_ids = set(range(len(data["track_representative_indices"])))
    track_ids.update(
        int(item["track_id"])
        for item in utterances
        if item["track_id"] is not None and int(item["track_id"]) >= 0
    )
    return track_ids


def _rank_speaker_utterances(indices: list[int], embeddings: np.ndarray, utterances: list[dict]) -> list[int]:
    member_vectors = embeddings[indices]
    centroid = member_vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    scores = {index: float(score) for index, score in zip(indices, member_vectors @ centroid)}
    return sorted(
        indices,
        key=lambda index: (
            -scores[index],
            -(utterances[index]["end_ms"] - utterances[index]["start_ms"]),
            index,
        ),
    )


def _speaker_preview_indices(
    candidates: list[int],
    utterances: list[dict],
) -> list[int]:
    preview = []
    for index in candidates:
        start_ms, end_ms = utterances[index]["start_ms"], utterances[index]["end_ms"]
        overlaps = any(
            min(end_ms, utterances[chosen]["end_ms"]) > max(start_ms, utterances[chosen]["start_ms"])
            for chosen in preview
        )
        if overlaps:
            continue
        preview.append(index)
        if len(preview) == SPEAKER_PREVIEW_UTTERANCES:
            break
    return preview


def _speaker_track_view(
    track_id: int,
    *,
    utterances: list[dict],
    embeddings: np.ndarray,
    auto_representatives: np.ndarray,
    overlays: dict,
) -> tuple[dict, list[int]]:
    overlay = overlays["speakers"].get(track_id, {})
    indices = [item["index"] for item in utterances if item["track_id"] == track_id]
    representative = overlay.get("representative_utterance_index")
    if representative is None and track_id < len(auto_representatives):
        representative = int(auto_representatives[track_id])
    if representative not in indices:
        representative = -1
    preview = []
    if indices:
        ranked = _rank_speaker_utterances(indices, embeddings, utterances)
        if representative < 0:
            representative = ranked[0]
        candidates = [representative, *(index for index in ranked if index != representative)]
        preview = _speaker_preview_indices(candidates, utterances)
    view = {
        "track_id": track_id,
        "label": overlay.get("display_name") or f"Speaker {track_id}",
        "display_name": overlay.get("display_name"),
        "representative_utterance_index": representative,
        "utterance_indices": preview,
        "utterance_count": len(indices),
        "duration_ms": sum(utterances[i]["end_ms"] - utterances[i]["start_ms"] for i in indices),
        "hidden": bool(overlay.get("hidden", 0)),
        "entity_id": overlays["bindings"].get(track_id, {}).get("entity_id"),
    }
    return view, preview


def video_speakers(catalog: Catalog, video_id: str) -> dict:
    data = _load_speaker_data(catalog, video_id)
    texts = _texts_from_milvus(catalog, video_id)
    overlays = catalog.speaker_overlays(video_id)
    video = catalog.get_video(video_id) or {}
    utterances = _speaker_utterances(
        data,
        texts,
        overlays,
        video_id,
        duration_seconds=float(video.get("duration") or 0.0),
    )
    track_ids = _speaker_track_ids(data, utterances)
    auto_representatives = data["track_representative_indices"].astype(np.int32)
    embeddings = data["utterance_embeddings"].astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    tracks = []
    preview_indices: set[int] = set()
    for track_id in sorted(track_ids):
        track, preview = _speaker_track_view(
            track_id,
            utterances=utterances,
            embeddings=embeddings,
            auto_representatives=auto_representatives,
            overlays=overlays,
        )
        preview_indices.update(preview)
        tracks.append(track)
    tracks.sort(key=lambda item: (-item["duration_ms"], item["track_id"]))
    preview_utterances = [item for item in utterances if item["index"] in preview_indices]
    return {"video_id": video_id, "tracks": tracks, "utterances": preview_utterances}


def voice_search(
    catalog: Catalog, *, query_video_id: str, query_utterance_index: int,
    video_ids: list[str] | None = None, limit: int = 50,
) -> list[dict]:
    query = _load_speaker_data(catalog, query_video_id)
    if not 0 <= query_utterance_index < len(query["utterance_embeddings"]):
        raise IndexError("查询声音不存在")
    query_vector = query["utterance_embeddings"][query_utterance_index].astype(np.float32)
    return voice_search_vectors(
        catalog, query_vectors=query_vector[None, :], video_ids=video_ids, limit=limit,
        exclude=(query_video_id, query_utterance_index),
    )


def voice_search_vectors(
    catalog: Catalog, *, query_vectors: np.ndarray,
    video_ids: list[str] | None = None, limit: int = 50,
    exclude: tuple[str, int] | None = None,
) -> list[dict]:
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim != 2 or not len(queries):
        raise ValueError("没有有效查询声纹")
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    hits = _voice_search_vectors_milvus(
        catalog,
        queries=queries,
        video_ids=video_ids,
        limit=limit,
        exclude=exclude,
    )
    return hits[:limit]


def _attach_voice_hit_texts(
    catalog: Catalog,
    hits: list[dict],
    *,
    limit: int,
) -> None:
    """Attach ASR text with exactly one Milvus ASR read per result video."""
    texts_by_video: dict[str, list[str]] = {}
    for hit in hits[:limit]:
        video_id = str(hit["video_id"])
        if video_id not in texts_by_video:
            texts_by_video[video_id] = _texts_from_milvus(catalog, video_id)
        texts = texts_by_video[video_id]
        chunk_index = int(hit["asr_chunk_index"])
        if not 0 <= chunk_index < len(texts):
            raise SpeakerMilvusCoverageError(
                f"Milvus speaker hit references missing ASR chunk {chunk_index} "
                f"for video {video_id}"
            )
        hit["text"] = texts[chunk_index]


def _voice_search_vectors_milvus(
    catalog: Catalog,
    *,
    queries: np.ndarray,
    video_ids: list[str] | None,
    limit: int,
    exclude: tuple[str, int] | None,
) -> list[dict]:
    # ``None`` is the all-video scope; an explicit empty selection is empty and
    # must not connect to Milvus or silently expand to the whole catalog.
    if video_ids is not None and not video_ids:
        return []
    from app.vector_store.milvus.milvus_client import (
        ensure_milvus_reachable,
        get_milvus_client,
    )
    from app.vector_store.milvus.milvus_search import milvus_speaker_candidates_scoped

    ensure_milvus_reachable()
    client = get_milvus_client()
    selected = None if video_ids is None else set(video_ids)
    videos_by_id: dict[str, dict] = {}
    asset_versions: dict[str, str] = {}
    for video in catalog.list_videos():
        video_id = video["id"]
        if selected is not None and video_id not in selected:
            continue
        # A no-speech video has a ready Catalog publication with row_count=0 so
        # stale Milvus rows can never become visible. It cannot contribute a
        # voice-search hit, so skip it instead of aborting the cross-video query.
        try:
            speaker_version = _published_asset_version(catalog, video_id, "speaker")
        except SpeakerMilvusCoverageError:
            continue
        videos_by_id[video_id] = video
        asset_versions[video_id] = speaker_version
    candidates = milvus_speaker_candidates_scoped(
        client,
        queries,
        asset_versions,
        limit,
        threshold=-1.0,
    )
    hits: list[dict] = []
    overlays_by_video: dict[str, dict] = {}
    for candidate in candidates:
        video_id = candidate.video_id
        utterance_index = int(candidate.unit_id if candidate.unit_id is not None else -1)
        if exclude == (video_id, utterance_index):
            continue
        if video_id not in overlays_by_video:
            overlays_by_video[video_id] = catalog.speaker_overlays(video_id)["utterances"]
        override = overlays_by_video[video_id].get(utterance_index, {})
        if not bool(override.get("searchable", 1)):
            continue
        video = videos_by_id[video_id]
        evidence_start_ms = int(round(candidate.start_time * 1000))
        evidence_end_ms = int(round(candidate.end_time * 1000))
        preview_start_ms, preview_end_ms = _speaker_preview_bounds_ms(
            evidence_start_ms,
            evidence_end_ms,
            duration_seconds=float(video.get("duration") or 0.0),
        )
        hits.append({
            "video_id": video_id,
            "video_name": video["name"],
            "utterance_index": utterance_index,
            "asr_chunk_index": int(candidate.features.get("asr_chunk_idx", -1)),
            "track_id": override.get(
                "corrected_track_id",
                int(candidate.features.get("track_id", -1)),
            ),
            "start_ms": evidence_start_ms,
            "end_ms": evidence_end_ms,
            "preview_start_ms": preview_start_ms,
            "preview_end_ms": preview_end_ms,
            "score": float(candidate.score),
            "clip_url": _speaker_clip_url(video_id, preview_start_ms, preview_end_ms),
        })
    hits.sort(key=lambda item: item["score"], reverse=True)
    _attach_voice_hit_texts(catalog, hits, limit=limit)
    # Text enrichment above already covers hits[:limit]; return the full sorted
    # list so the single authoritative truncation happens in voice_search_vectors.
    return hits
