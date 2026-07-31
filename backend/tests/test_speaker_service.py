from pathlib import Path
from unittest.mock import patch, Mock
import re

import numpy as np
import pytest
from app.db import Catalog
from app.indexing.speaker import load_speaker_index, save_speaker_index
from app.speaker_service import (
    SpeakerMilvusCoverageError,
    _load_speaker_data,
    _speaker_data_from_milvus,
    speaker_utterance_embedding,
    video_speakers,
    voice_search,
    voice_search_vectors,
)


def _make_speaker_data(vectors: np.ndarray) -> dict:
    """Build a speaker data dict identical to what _speaker_data_from_milvus() returns."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = Path(f.name)
    save_speaker_index(
        tmp,
        utterance_times_ms=np.asarray([[0, 1000], [2000, 3000]]),
        utterance_embeddings=vectors,
        asr_chunk_indices=np.asarray([0, 1]),
        auto_track_indices=np.asarray([0, 1]),
    )
    data = load_speaker_index(tmp)
    tmp.unlink(missing_ok=True)
    return data


_ASR_TEXTS = ["first sentence", "second sentence"]


def make_video(catalog: Catalog, video_id: str, vectors: np.ndarray):
    catalog.create_video({
        "id": video_id, "name": video_id, "file_path": f"/tmp/{video_id}.mp4",
        "duration": 10, "fps": 25, "width": 100, "height": 100, "status": "ready",
    })
    return _make_speaker_data(vectors)


def save_video_speaker_index(
    index_root: Path,
    video_id: str,
    vectors: np.ndarray,
) -> Path:
    path = index_root / video_id / "speaker.npz"
    save_speaker_index(
        path,
        utterance_times_ms=np.asarray([[0, 1000], [2000, 3000]]),
        utterance_embeddings=vectors,
        asr_chunk_indices=np.asarray([0, 1]),
        auto_track_indices=np.asarray([0, 1]),
    )
    return path


def _milvus_speaker_row(index: int, embedding: list[float]) -> dict:
    return {
        "utterance_idx": index,
        "start_ms": index * 1000,
        "end_ms": (index + 1) * 1000,
        "asr_chunk_idx": index,
        "track_id": index,
        "embedding": embedding,
    }


def test_video_speakers_applies_mutable_sqlite_overlay(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    speaker_data = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    catalog.upsert_video_speaker("a", 0, display_name="Host")
    catalog.upsert_utterance_override("a", 1, 0, False)

    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service._speaker_data_from_milvus", return_value=speaker_data),
        patch("app.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
    ):
        payload = video_speakers(tmp_path / "indexes", catalog, "a")

    assert payload["tracks"][0]["label"] == "Host"
    assert payload["tracks"][0]["utterance_indices"] == [0, 1]
    assert payload["utterances"][1]["searchable"] is False


def test_voice_search_matches_individual_utterances(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    data_a = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    data_b = make_video(catalog, "b", np.asarray([[.99, .01], [-1, 0]], dtype=np.float32))

    def _mock_speaker(video_id: str, **_kwargs):
        return {"a": data_a, "b": data_b}[video_id]

    # Mock Milvus connection and search results
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    # Mock search() to return per-video results
    # When querying video "a", return 2 hits (both utterances from video "a")
    # When querying video "b", return 1 hit (best match from video "b")
    def mock_search_side_effect(data, anns_field, param, limit, expr, output_fields, timeout=None):
        # Extract video_id from expr using regex for robustness
        match = re.search(r'video_id\s*==\s*["\']([^"\']+)["\']', expr)
        if not match:
            return [[]]  # Return empty if expr format unexpected

        video_id = match.group(1)

        if video_id == "a":
            # Video "a" has 2 utterances
            mock_hit_a0 = Mock()
            mock_hit_a0.distance = 1.0  # Perfect self-match
            mock_hit_a0.entity = Mock()
            mock_hit_a0.entity.get = lambda field, default=None: {
                "utterance_idx": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "track_id": 0,
                "asr_chunk_idx": 0,
                "embedding": [1.0, 0.0],
                "_distance": 1.0,
            }.get(field, default)

            mock_hit_a1 = Mock()
            mock_hit_a1.distance = 0.0  # Low similarity to second utterance
            mock_hit_a1.entity = Mock()
            mock_hit_a1.entity.get = lambda field, default=None: {
                "utterance_idx": 1,
                "start_ms": 2000,
                "end_ms": 3000,
                "track_id": 1,
                "asr_chunk_idx": 1,
                "embedding": [0.0, 1.0],
                "_distance": 0.0,
            }.get(field, default)

            return [[mock_hit_a0, mock_hit_a1]]

        elif video_id == "b":
            # Video "b" has 2 utterances, return best match
            mock_hit_b = Mock()
            mock_hit_b.distance = 0.99  # High match to video "a"'s query
            mock_hit_b.entity = Mock()
            mock_hit_b.entity.get = lambda field, default=None: {
                "utterance_idx": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "track_id": 0,
                "asr_chunk_idx": 0,
                "embedding": [0.99, 0.01],
                "_distance": 0.99,
            }.get(field, default)

            return [[mock_hit_b]]

        return [[]]

    mock_collection.search.side_effect = mock_search_side_effect

    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service._speaker_data_from_milvus", side_effect=_mock_speaker),
        patch("app.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
        patch("app.indexing.milvus_client.ensure_milvus_reachable", return_value=None),
        patch("app.indexing.milvus_client.get_milvus_client", return_value=mock_client),
    ):
        hits = voice_search(
            tmp_path / "indexes", catalog, query_video_id="a", query_utterance_index=0, limit=3
        )

    assert hits[0]["score"] > .99
    assert {hit["video_id"] for hit in hits[:2]} == {"a", "b"}
    assert all("text" in hit and "clip_url" in hit for hit in hits)


def test_speaker_utterance_embedding_uses_primary_loader(tmp_path):
    data = _make_speaker_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    )
    with patch("app.speaker_service._load_speaker_data", return_value=data) as load:
        vector = speaker_utterance_embedding(tmp_path, "video-1", 1)

    load.assert_called_once_with(
        tmp_path / "video-1" / "speaker.npz",
        "video-1",
    )
    np.testing.assert_allclose(vector, np.asarray([0, 1], dtype=np.float32))


def test_speaker_data_from_milvus_rejects_sparse_utterance_indices():
    rows = [
        _milvus_speaker_row(0, [1.0, 0.0]),
        _milvus_speaker_row(2, [0.0, 1.0]),
    ]
    with patch("app.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="sparse or duplicated"):
            _speaker_data_from_milvus("video-1")


def test_speaker_data_from_milvus_checks_expected_utterance_count():
    rows = [_milvus_speaker_row(0, [1.0, 0.0])]
    with patch("app.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="expected 2, got 1"):
            _speaker_data_from_milvus("video-1", expected_utterances=2)


def test_load_speaker_data_falls_back_after_invalid_milvus_rows(tmp_path):
    path = save_video_speaker_index(
        tmp_path,
        "video-1",
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service.milvus_fallback_enabled", return_value=True),
        patch(
            "app.speaker_service._speaker_data_from_milvus",
            side_effect=SpeakerMilvusCoverageError("incomplete"),
        ),
    ):
        data = _load_speaker_data(path, "video-1")

    assert len(data["utterance_embeddings"]) == 2


def test_load_speaker_data_honors_disabled_fallback(tmp_path):
    path = tmp_path / "video-1" / "speaker.npz"
    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service.milvus_fallback_enabled", return_value=False),
        patch(
            "app.speaker_service._speaker_data_from_milvus",
            side_effect=RuntimeError("Milvus unavailable"),
        ),
        patch("app.speaker_service.load_speaker_index") as load_npz,
    ):
        with pytest.raises(RuntimeError, match="Milvus unavailable"):
            _load_speaker_data(path, "video-1")

    load_npz.assert_not_called()


def test_voice_search_uses_npz_only_for_uncovered_milvus_videos(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    index_root = tmp_path / "indexes"
    save_video_speaker_index(
        index_root,
        "a",
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    save_video_speaker_index(
        index_root,
        "b",
        np.asarray([[0.95, 0.05], [0, 1]], dtype=np.float32),
    )
    milvus_hit = {
        "video_id": "a",
        "video_name": "a",
        "utterance_index": 0,
        "asr_chunk_index": 0,
        "track_id": 0,
        "start_ms": 0,
        "end_ms": 1000,
        "score": 1.0,
        "clip_url": "/a",
        "text": "a",
    }

    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service.milvus_fallback_enabled", return_value=True),
        patch(
            "app.speaker_service._voice_search_vectors_milvus",
            return_value=([milvus_hit], {"a"}),
        ),
        patch("app.speaker_service._speaker_data_from_milvus", return_value=None),
        patch("app.speaker_service._texts_for_video", return_value=["first", "second"]),
    ):
        hits = voice_search_vectors(
            index_root,
            catalog,
            query_vectors=np.asarray([[1, 0]], dtype=np.float32),
            video_ids=["a", "b"],
            limit=5,
        )

    assert {hit["video_id"] for hit in hits} == {"a", "b"}
    assert sum(hit["video_id"] == "a" for hit in hits) == 1


def test_voice_search_rejects_coverage_gap_when_fallback_is_disabled(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    index_root = tmp_path / "indexes"
    save_video_speaker_index(
        index_root,
        "a",
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    save_video_speaker_index(
        index_root,
        "b",
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )

    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service.milvus_fallback_enabled", return_value=False),
        patch(
            "app.speaker_service._voice_search_vectors_milvus",
            return_value=([], {"a"}),
        ),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="video\\(s\\): b"):
            voice_search_vectors(
                index_root,
                catalog,
                query_vectors=np.asarray([[1, 0]], dtype=np.float32),
                video_ids=["a", "b"],
                limit=5,
            )
