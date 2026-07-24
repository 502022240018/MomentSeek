from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from app.db import Catalog
from app.indexing.speaker import load_speaker_index
from app.settings import get_settings


SPEAKER_PREVIEW_UTTERANCES = 5


class SpeakerMilvusCoverageError(RuntimeError):
    """Milvus speaker rows do not represent one complete, index-safe video."""


def get_milvus_client():
    from app.indexing.milvus_client import get_milvus_client as factory

    return factory()


def ensure_milvus_reachable() -> None:
    from app.indexing.milvus_client import ensure_milvus_reachable as ensure

    ensure()


def milvus_read_enabled() -> bool:
    from app.indexing.milvus_flags import milvus_read_enabled as enabled

    return enabled()


def milvus_fallback_enabled() -> bool:
    from app.indexing.milvus_flags import milvus_fallback_enabled as enabled

    return enabled()


def _expected_speaker_utterances(path: Path) -> int | None:
    from app.indexing.manifest import load_index_manifest

    manifest = load_index_manifest(path.parent)
    if manifest is None:
        return None
    channel = (manifest.get("channels") or {}).get("speaker")
    if not isinstance(channel, dict):
        return None
    value = channel.get("utterances")
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _texts(asr_path: Path, video_id: str | None = None) -> list[str]:
    if video_id is not None and milvus_read_enabled():
        texts = _texts_from_milvus(video_id)
        if texts:
            return texts
    with np.load(asr_path, allow_pickle=False) as data:
        return [str(value) for value in data["texts"]]


def _milvus_rows(video_id: str, modality: str, fields: list[str]) -> list[dict]:
    ensure_milvus_reachable()
    collection = get_milvus_client().collection_for(modality)
    expression = f'video_id == "{video_id}"'
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
    return rows


def _texts_for_video(index_dir: Path, video_id: str) -> list[str]:
    if milvus_read_enabled():
        try:
            texts = _texts_from_milvus(video_id)
            if texts:
                return texts
        except Exception:
            pass
    path = index_dir / video_id / "asr.npz"
    return _texts(path) if path.exists() else []


def _texts_from_milvus(video_id: str) -> list[str]:
    try:
        ensure_milvus_reachable()
        rows = get_milvus_client().collection_for("asr").query(
            expr=f'video_id == "{video_id}"',
            output_fields=["segment_idx", "text"],
            limit=16_384,
            timeout=get_settings().milvus_query_timeout_seconds,
        )
    except Exception:
        return []
    if not rows:
        return []
    values = {
        int(row.get("segment_idx") or 0): str(row.get("text") or "")
        for row in rows
    }
    return [values.get(index, "") for index in range(max(values) + 1)]


def _speaker_data_from_milvus(
    video_id: str,
    *,
    expected_utterances: int | None = None,
) -> dict[str, np.ndarray] | None:
    rows = _milvus_rows(
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
    )
    if not rows:
        return None
    indexed_rows: list[tuple[int, dict]] = []
    for row in rows:
        value = row.get("utterance_idx")
        if value is None:
            raise SpeakerMilvusCoverageError(
                f"Milvus speaker row is missing utterance_idx for video {video_id}"
            )
        index = int(value)
        if index < 0:
            raise SpeakerMilvusCoverageError(
                f"Milvus speaker row has a negative utterance_idx for video {video_id}"
            )
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
    if embeddings.ndim != 2 or not embeddings.shape[1]:
        raise SpeakerMilvusCoverageError(
            f"Milvus speaker embeddings are invalid for video {video_id}"
        )
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    times = np.asarray([
        [int(row.get("start_ms") or 0), int(row.get("end_ms") or 0)]
        for row in rows
    ], dtype=np.int32)
    refs = np.asarray([
        [int(row.get("asr_chunk_idx") or 0), int(row.get("track_id") or 0)]
        for row in rows
    ], dtype=np.int32)
    track_ids = refs[:, 1]
    track_count = int(track_ids.max()) + 1 if len(track_ids) else 0
    track_embeddings = np.zeros((track_count, embeddings.shape[1]), dtype=np.float32)
    representatives = np.full((track_count,), -1, dtype=np.int32)
    for track_id in range(track_count):
        members = np.flatnonzero(track_ids == track_id)
        if not len(members):
            continue
        center = embeddings[members].mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        track_embeddings[track_id] = center
        representatives[track_id] = int(
            members[int(np.argmax(embeddings[members] @ center))]
        )
    return {
        "utterance_embeddings": embeddings.astype(np.float16),
        "utterance_times_ms": times,
        "utterance_refs": refs,
        "track_embeddings": track_embeddings.astype(np.float16),
        "track_representative_indices": representatives,
    }


def _load_speaker_data(path: Path, video_id: str) -> dict[str, np.ndarray]:
    if milvus_read_enabled():
        try:
            data = _speaker_data_from_milvus(
                video_id,
                expected_utterances=_expected_speaker_utterances(path),
            )
            if data is not None:
                return data
        except Exception:
            if not milvus_fallback_enabled():
                raise
        else:
            if not milvus_fallback_enabled():
                raise SpeakerMilvusCoverageError(
                    f"Milvus speaker data is missing for video {video_id}"
                )
    return load_speaker_index(path)


def speaker_utterance_embedding(
    index_dir: Path,
    video_id: str,
    utterance_index: int,
) -> np.ndarray:
    """Read one utterance from Milvus primary with NPZ fallback."""
    data = _load_speaker_data(index_dir / video_id / "speaker.npz", video_id)
    try:
        return data["utterance_embeddings"][utterance_index].astype(np.float32)
    except IndexError as exc:
        raise IndexError("声音片段不存在") from exc


def _speaker_utterances(data: dict, texts: list[str], overlays: dict, video_id: str) -> list[dict]:
    refs = data["utterance_refs"].astype(np.int32)
    times = data["utterance_times_ms"].astype(np.int32)
    utterances = []
    for index, ((start_ms, end_ms), (chunk_index, auto_track)) in enumerate(zip(times, refs)):
        override = overlays["utterances"].get(index, {})
        final_track = override.get("corrected_track_id")
        if final_track is None and index not in overlays["utterances"]:
            final_track = int(auto_track)
        utterances.append({
            "index": index,
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "asr_chunk_index": int(chunk_index),
            "text": texts[int(chunk_index)] if 0 <= int(chunk_index) < len(texts) else "",
            "auto_track_id": int(auto_track),
            "track_id": final_track,
            "searchable": bool(override.get("searchable", 1)),
            "clip_url": f"/api/videos/{video_id}/clip?{urlencode({'start': start_ms / 1000, 'end': end_ms / 1000})}",
        })
    return utterances


def _speaker_track_ids(data: dict, utterances: list[dict]) -> set[int]:
    track_ids = set(range(len(data["track_embeddings"])))
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


def video_speakers(index_dir: Path, catalog: Catalog, video_id: str) -> dict:
    speaker_path = index_dir / video_id / "speaker.npz"
    data = _load_speaker_data(speaker_path, video_id)
    texts = _texts_for_video(index_dir, video_id)
    overlays = catalog.speaker_overlays(video_id)
    utterances = _speaker_utterances(data, texts, overlays, video_id)
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
    index_dir: Path, catalog: Catalog, *, query_video_id: str, query_utterance_index: int,
    video_ids: list[str] | None = None, limit: int = 50,
) -> list[dict]:
    query_path = index_dir / query_video_id / "speaker.npz"
    query = _load_speaker_data(query_path, query_video_id)
    if not 0 <= query_utterance_index < len(query["utterance_embeddings"]):
        raise IndexError("查询声音不存在")
    query_vector = query["utterance_embeddings"][query_utterance_index].astype(np.float32)
    return voice_search_vectors(
        index_dir, catalog, query_vectors=query_vector[None, :], video_ids=video_ids, limit=limit,
        exclude=(query_video_id, query_utterance_index),
    )


def voice_search_vectors(
    index_dir: Path, catalog: Catalog, *, query_vectors: np.ndarray,
    video_ids: list[str] | None = None, limit: int = 50,
    exclude: tuple[str, int] | None = None,
) -> list[dict]:
    queries = np.asarray(query_vectors, dtype=np.float32)
    if queries.ndim != 2 or not len(queries):
        raise ValueError("没有有效查询声纹")
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    selected = set(video_ids) if video_ids else None
    hits: list[dict] = []
    if milvus_read_enabled():
        try:
            milvus_hits, covered_video_ids = _voice_search_vectors_milvus(
                index_dir,
                catalog,
                queries=queries,
                video_ids=video_ids,
                limit=limit,
                exclude=exclude,
            )
        except Exception:
            if not milvus_fallback_enabled():
                raise
        else:
            expected_video_ids = {
                str(video["id"])
                for video in catalog.list_videos()
                if (selected is None or str(video["id"]) in selected)
                and (
                    (index_dir / str(video["id"]) / "speaker.npz").exists()
                    or (_expected_speaker_utterances(
                        index_dir / str(video["id"]) / "speaker.npz"
                    ) or 0) > 0
                )
            }
            coverage_gaps = expected_video_ids - covered_video_ids
            if not coverage_gaps:
                return milvus_hits[:limit]
            if not milvus_fallback_enabled():
                raise SpeakerMilvusCoverageError(
                    "Milvus speaker coverage is missing for video(s): "
                    + ", ".join(sorted(coverage_gaps))
                )
            hits.extend(milvus_hits)
            selected = coverage_gaps
    for video in catalog.list_videos():
        video_id = video["id"]
        if selected is not None and video_id not in selected:
            continue
        path = index_dir / video_id / "speaker.npz"
        try:
            data = _load_speaker_data(path, video_id)
        except FileNotFoundError:
            continue
        vectors = data["utterance_embeddings"].astype(np.float32)
        if not len(vectors):
            continue
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        # A concrete sentence is the primary match. Multiple uploaded query
        # samples support it through max similarity without averaging speakers.
        scores = np.max(vectors @ queries.T, axis=1)
        overlays = catalog.speaker_overlays(video_id)["utterances"]
        for utterance_index in np.argsort(-scores)[: min(limit, len(scores))]:
            if exclude == (video_id, int(utterance_index)):
                continue
            override = overlays.get(int(utterance_index), {})
            if not bool(override.get("searchable", 1)):
                continue
            start_ms, end_ms = data["utterance_times_ms"][utterance_index]
            chunk_index, auto_track = data["utterance_refs"][utterance_index]
            hits.append({
                "video_id": video_id, "video_name": video["name"],
                "utterance_index": int(utterance_index), "asr_chunk_index": int(chunk_index),
                "track_id": override.get("corrected_track_id", int(auto_track)),
                "start_ms": int(start_ms), "end_ms": int(end_ms),
                "score": float(scores[utterance_index]),
                "clip_url": f"/api/videos/{video_id}/clip?{urlencode({'start': start_ms / 1000, 'end': end_ms / 1000})}",
            })
    hits.sort(key=lambda item: item["score"], reverse=True)
    texts_by_video: dict[str, list[str]] = {}
    for hit in hits[:limit]:
        texts = texts_by_video.setdefault(
            hit["video_id"], _texts_for_video(index_dir, hit["video_id"])
        )
        index = hit["asr_chunk_index"]
        hit["text"] = texts[index] if 0 <= index < len(texts) else ""
    return hits[:limit]


def _voice_search_vectors_milvus(
    index_dir: Path,
    catalog: Catalog,
    *,
    queries: np.ndarray,
    video_ids: list[str] | None,
    limit: int,
    exclude: tuple[str, int] | None,
) -> tuple[list[dict], set[str]]:
    from app.indexing.milvus_client import (
        ensure_milvus_reachable,
        get_milvus_client,
    )
    from app.indexing.milvus_search import milvus_speaker_candidates

    ensure_milvus_reachable()
    client = get_milvus_client()
    selected = set(video_ids) if video_ids else None
    hits: list[dict] = []
    covered_video_ids: set[str] = set()
    for video in catalog.list_videos():
        video_id = video["id"]
        if selected is not None and video_id not in selected:
            continue
        best_by_utterance = {}
        for query in queries:
            for candidate in milvus_speaker_candidates(
                client, video_id, query, limit, threshold=-1.0
            ):
                previous = best_by_utterance.get(candidate.unit_id)
                if previous is None or candidate.score > previous.score:
                    best_by_utterance[candidate.unit_id] = candidate
        if best_by_utterance:
            covered_video_ids.add(video_id)
        overlays = catalog.speaker_overlays(video_id)["utterances"]
        for utterance_index, candidate in best_by_utterance.items():
            if exclude == (video_id, int(utterance_index)):
                continue
            override = overlays.get(int(utterance_index), {})
            if not bool(override.get("searchable", 1)):
                continue
            hits.append({
                "video_id": video_id,
                "video_name": video["name"],
                "utterance_index": int(utterance_index),
                "asr_chunk_index": int(candidate.features.get("asr_chunk_idx", -1)),
                "track_id": override.get(
                    "corrected_track_id",
                    int(candidate.features.get("track_id", -1)),
                ),
                "start_ms": int(round(candidate.start_time * 1000)),
                "end_ms": int(round(candidate.end_time * 1000)),
                "score": float(candidate.score),
                "clip_url": (
                    f"/api/videos/{video_id}/clip?"
                    f"{urlencode({'start': candidate.start_time, 'end': candidate.end_time})}"
                ),
            })
    hits.sort(key=lambda item: item["score"], reverse=True)
    texts_by_video: dict[str, list[str]] = {}
    for hit in hits[:limit]:
        texts = texts_by_video.setdefault(
            hit["video_id"], _texts_for_video(index_dir, hit["video_id"])
        )
        chunk_index = hit["asr_chunk_index"]
        hit["text"] = texts[chunk_index] if 0 <= chunk_index < len(texts) else ""
    return hits[:limit], covered_video_ids
