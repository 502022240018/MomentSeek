from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from app.db import Catalog
from app.indexing.speaker import load_speaker_index


SPEAKER_PREVIEW_UTTERANCES = 5


def _texts(asr_path: Path) -> list[str]:
    with np.load(asr_path, allow_pickle=False) as data:
        return [str(value) for value in data["texts"]]


def video_speakers(index_dir: Path, catalog: Catalog, video_id: str) -> dict:
    speaker_path = index_dir / video_id / "speaker.npz"
    asr_path = index_dir / video_id / "asr.npz"
    if not speaker_path.exists() or not asr_path.exists():
        raise FileNotFoundError("该视频尚未构建 Speaker 索引")
    data = load_speaker_index(speaker_path)
    texts = _texts(asr_path)
    overlays = catalog.speaker_overlays(video_id)
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
    track_ids = set(range(len(data["track_embeddings"])))
    track_ids.update(int(item["track_id"]) for item in utterances if item["track_id"] is not None and int(item["track_id"]) >= 0)
    auto_representatives = data["track_representative_indices"].astype(np.int32)
    embeddings = data["utterance_embeddings"].astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    tracks = []
    preview_indices: set[int] = set()
    for track_id in sorted(track_ids):
        overlay = overlays["speakers"].get(track_id, {})
        indices = [item["index"] for item in utterances if item["track_id"] == track_id]
        representative = overlay.get("representative_utterance_index")
        if representative is None and track_id < len(auto_representatives):
            representative = int(auto_representatives[track_id])
        if representative not in indices:
            representative = -1
        if indices:
            member_vectors = embeddings[indices]
            centroid = member_vectors.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centrality = member_vectors @ centroid
            scores = {index: float(score) for index, score in zip(indices, centrality)}
            ranked = sorted(
                indices,
                key=lambda index: (
                    -scores[index],
                    -(utterances[index]["end_ms"] - utterances[index]["start_ms"]),
                    index,
                ),
            )
            if representative < 0:
                representative = ranked[0]
            candidates = ([representative] if representative >= 0 else []) + [
                index for index in ranked if index != representative
            ]
            preview = []
            for index in candidates:
                start_ms, end_ms = utterances[index]["start_ms"], utterances[index]["end_ms"]
                if any(
                    min(end_ms, utterances[chosen]["end_ms"]) > max(start_ms, utterances[chosen]["start_ms"])
                    for chosen in preview
                ):
                    continue
                preview.append(index)
                if len(preview) == SPEAKER_PREVIEW_UTTERANCES:
                    break
        else:
            preview = []
        preview_indices.update(preview)
        tracks.append({
            "track_id": track_id,
            "label": overlay.get("display_name") or f"Speaker {track_id}",
            "display_name": overlay.get("display_name"),
            "representative_utterance_index": representative,
            "utterance_indices": preview,
            "utterance_count": len(indices),
            "duration_ms": sum(utterances[i]["end_ms"] - utterances[i]["start_ms"] for i in indices),
            "hidden": bool(overlay.get("hidden", 0)),
            "entity_id": overlays["bindings"].get(track_id, {}).get("entity_id"),
        })
    tracks.sort(key=lambda item: (-item["duration_ms"], item["track_id"]))
    preview_utterances = [item for item in utterances if item["index"] in preview_indices]
    return {"video_id": video_id, "tracks": tracks, "utterances": preview_utterances}


def voice_search(
    index_dir: Path, catalog: Catalog, *, query_video_id: str, query_utterance_index: int,
    video_ids: list[str] | None = None, limit: int = 50,
) -> list[dict]:
    query_path = index_dir / query_video_id / "speaker.npz"
    query = load_speaker_index(query_path)
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
    hits = []
    for video in catalog.list_videos():
        video_id = video["id"]
        if selected is not None and video_id not in selected:
            continue
        path = index_dir / video_id / "speaker.npz"
        if not path.exists():
            continue
        data = load_speaker_index(path)
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
        texts = texts_by_video.setdefault(hit["video_id"], _texts(index_dir / hit["video_id"] / "asr.npz"))
        index = hit["asr_chunk_index"]
        hit["text"] = texts[index] if 0 <= index < len(texts) else ""
    return hits[:limit]
