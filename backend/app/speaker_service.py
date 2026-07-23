from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from app.db import Catalog


SPEAKER_PREVIEW_UTTERANCES = 5


def _texts_from_milvus(video_id: str) -> list[str]:
    """从 Milvus ASR collection 读取文本。

    Returns:
        按 segment_idx 排序的 ASR chunk 文本列表。
        如果 Milvus 无数据或失败，返回空列表。
    """
    try:
        from app.indexing.milvus_client import get_milvus_client

        client = get_milvus_client()
        col = client.collection_for("asr")

        # Query all ASR records for this video
        rows = col.query(
            expr=f'video_id == "{video_id}"',
            output_fields=["segment_idx", "text"],
            limit=16384,  # Max reasonable ASR chunk count
        )

        if not rows:
            return []

        # Sort by segment_idx to match NPZ order
        rows.sort(key=lambda r: int(r.get("segment_idx") or 0))

        # Build sparse mapping: segment_idx -> text
        # Some segment_idx may be missing (semantic indexing is sparse)
        segment_texts: dict[int, str] = {}
        for row in rows:
            seg_idx = int(row.get("segment_idx") or 0)
            text = str(row.get("text") or "")
            segment_texts[seg_idx] = text

        # Return dense list: fill missing indices with empty strings
        if not segment_texts:
            return []

        max_idx = max(segment_texts.keys())
        return [segment_texts.get(i, "") for i in range(max_idx + 1)]

    except Exception:
        # Any Milvus error (connection, query failure, etc.) returns empty list.
        return []


def _texts(video_id: str) -> list[str]:
    """读取 ASR 文本，从 Milvus 读取（Milvus 是唯一存储后端）。"""
    return _texts_from_milvus(video_id)


def _speaker_data_from_milvus(video_id: str) -> "dict[str, np.ndarray] | None":
    """从 Milvus speaker collection 重建与 load_speaker_index() 相同结构的字典。

    Returns:
        包含 utterance_embeddings / utterance_times_ms / utterance_refs /
        track_embeddings / track_representative_indices 的字典，
        或 None（Milvus 无数据或连接失败）。
    """
    try:
        from app.indexing.milvus_client import get_milvus_client
        from app.indexing.speaker import EMBEDDING_DIM

        col = get_milvus_client().collection_for("speaker")
        # Milvus caps (offset + limit) at 16 384.  Use QueryIterator when
        # available (pymilvus ≥ 2.3) so videos with many utterances are
        # handled correctly; fall back to the hard limit otherwise.
        expr = f'video_id == "{video_id}"'
        _fields = ["utterance_idx", "start_ms", "end_ms", "asr_chunk_idx", "track_id", "embedding"]
        rows: list[dict] = []
        if hasattr(col, "query_iterator"):
            _iter = col.query_iterator(batch_size=2000, expr=expr, output_fields=_fields)
            try:
                while True:
                    page = _iter.next()
                    if not page:
                        break
                    rows.extend(page)
            finally:
                _iter.close()
        else:
            rows = col.query(expr=expr, output_fields=_fields, limit=16384)
        if not rows:
            return None

        # Reconstruct utterance arrays sorted by utterance_idx.
        rows.sort(key=lambda r: int(r.get("utterance_idx") or 0))
        embeddings = np.array([r["embedding"] for r in rows], dtype=np.float32)
        times = np.array(
            [[int(r.get("start_ms") or 0), int(r.get("end_ms") or 0)] for r in rows],
            dtype=np.int32,
        )
        refs = np.array(
            [[int(r.get("asr_chunk_idx") or 0), int(r.get("track_id") or 0)] for r in rows],
            dtype=np.int32,
        )

        # Normalise (Milvus stores raw float32; NPZ stores normalised float16).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-12)

        # Recompute per-track centroids and representative utterance indices.
        track_ids = refs[:, 1]
        track_count = int(track_ids.max()) + 1 if len(embeddings) else 0
        dim = embeddings.shape[1] if len(embeddings) else EMBEDDING_DIM
        track_embeddings = np.zeros((track_count, dim), dtype=np.float32)
        track_representatives = np.full((track_count,), -1, dtype=np.int32)
        for t in range(track_count):
            members = np.flatnonzero(track_ids == t)
            if not len(members):
                continue
            center = embeddings[members].mean(axis=0)
            center /= max(float(np.linalg.norm(center)), 1e-12)
            track_embeddings[t] = center
            track_representatives[t] = int(members[int(np.argmax(embeddings[members] @ center))])

        return {
            "utterance_embeddings":        embeddings.astype(np.float16),
            "utterance_times_ms":          times,
            "utterance_refs":              refs,
            "track_embeddings":            track_embeddings.astype(np.float16),
            "track_representative_indices": track_representatives,
        }
    except Exception:
        return None


def _load_speaker_data(speaker_path: Path, video_id: str) -> "dict[str, np.ndarray]":
    """加载 speaker index，从 Milvus 读取（Milvus 是唯一存储后端）。

    Raises FileNotFoundError if Milvus has no data for this video.
    """
    data = _speaker_data_from_milvus(video_id)
    if data is not None:
        return data
    raise FileNotFoundError("该视频尚未构建 Speaker 索引")


def video_speakers(index_dir: Path, catalog: Catalog, video_id: str) -> dict:
    speaker_path = index_dir / video_id / "speaker.npz"
    # _load_speaker_data reads from Milvus; raises FileNotFoundError when no data.
    data = _load_speaker_data(speaker_path, video_id)
    texts = _texts(video_id)
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
    hits = []
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
        vid = hit["video_id"]
        texts = texts_by_video.setdefault(vid, _texts(vid))
        index = hit["asr_chunk_index"]
        hit["text"] = texts[index] if 0 <= index < len(texts) else ""
    return hits[:limit]


def _load_voice_embeddings_for_entity(catalog: Catalog, entity_id: str) -> np.ndarray | None:
    """从数据库 BLOB 加载实体的语音 embeddings。

    Phase 3 起 voice_embedding 字段存储在数据库中，embedding_path 仅保留
    对迁移前（Pre-Phase 3）遗留数据的兼容读取，新数据不再写入文件。

    Returns:
        形状为 [N, 192] 的 embeddings 数组，或 None（如果没有样本）
    """
    samples = catalog.list_voice_samples(entity_id)
    if not samples:
        return None

    embeddings = []
    for sample in samples:
        if sample.get("voice_embedding"):
            # 主路径：从数据库 BLOB 读取（Phase 3+）
            vector = np.frombuffer(sample["voice_embedding"], dtype=np.float32)
            embeddings.append(vector)
        elif sample.get("embedding_path") and Path(sample["embedding_path"]).exists():
            # 兼容路径：读取 Pre-Phase 3 遗留 embedding 文件（迁移完成后可删除此分支）
            try:
                vector = np.load(sample["embedding_path"])["embedding"]
                embeddings.append(vector)
            except Exception:
                # 跳过损坏或格式不兼容的文件
                continue

    if not embeddings:
        return None

    return np.stack(embeddings, axis=0)
