from pathlib import Path
from unittest.mock import patch, Mock
import re

import numpy as np
import pytest
from fastapi import HTTPException
from app.api import speaker_routes
from app.catalog.db import Catalog
from app.indexing.modalities.speaker.speaker import load_speaker_index, save_speaker_index
from app.identity.speaker_service import (
    SpeakerMilvusCoverageError,
    _load_speaker_data,
    _speaker_data_from_milvus,
    _texts_from_milvus,
    speaker_utterance_embedding,
    video_speakers,
    voice_search,
    voice_search_vectors,
)
from app.platform import context


def _make_speaker_data(vectors: np.ndarray) -> dict:
    """Build a speaker data dict for the panel path via the offline NPZ loader.

    Note: load_speaker_index() still carries ``track_embeddings`` (offline
    recovery artifact), whereas the online _speaker_data_from_milvus() no longer
    returns it — video_speakers only needs ``track_representative_indices`` and
    the utterance arrays, which both paths provide. See
    test_speaker_data_from_milvus_drops_track_embeddings for the online contract.
    """
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
        patch("app.identity.speaker_service._speaker_data_from_milvus", return_value=speaker_data),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
    ):
        payload = video_speakers(tmp_path / "indexes", catalog, "a")

    assert payload["tracks"][0]["label"] == "Host"
    assert payload["tracks"][0]["utterance_indices"] == [0, 1]
    assert payload["utterances"][1]["searchable"] is False


def test_speaker_data_from_milvus_drops_track_embeddings(tmp_path):
    """Online panel path no longer materialises the dead track_embeddings matrix.

    _speaker_data_from_milvus previously returned a per-track centroid matrix that
    video_speakers only ever used for its length (the centroids themselves are
    re-computed over the overlay-corrected membership in _rank_speaker_utterances).
    The cleanup drops that matrix; the representative indices — which ARE consumed
    as auto_representatives — must stay correct, and video_speakers must be
    unaffected. Two utterances of track 0 + one of track 1.
    """
    rows = [
        _milvus_speaker_row(0, [1.0, 0.0]),
        _milvus_speaker_row(1, [0.0, 1.0]),
        _milvus_speaker_row(2, [1.0, 0.0]),
    ]
    # Two utterances on track 0, one on track 1 (override the default 1:1 mapping).
    rows[0]["track_id"] = 0
    rows[1]["track_id"] = 0
    rows[2]["track_id"] = 1

    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        data = _speaker_data_from_milvus("vid")

    # The dead centroid matrix is gone; representatives (per-track argmax) remain.
    assert "track_embeddings" not in data
    assert data["track_representative_indices"].shape == (2,)
    # Track 0's representative is one of its two members {0, 1}; track 1's sole
    # member is utterance 2, so its representative is 2.
    assert int(data["track_representative_indices"][0]) in (0, 1)
    assert int(data["track_representative_indices"][1]) == 2

    # video_speakers still builds correctly off the trimmed dict (track count is
    # now derived from track_representative_indices, not track_embeddings).
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, "vid", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", return_value=data),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
    ):
        payload = video_speakers(tmp_path / "indexes", catalog, "vid")

    assert {track["track_id"] for track in payload["tracks"]} == {0, 1}


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
        patch("app.identity.speaker_service._speaker_data_from_milvus", side_effect=_mock_speaker),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
        patch("app.identity.speaker_service._published_asset_version", return_value="7"),
        patch("app.vector_store.milvus.milvus_client.ensure_milvus_reachable", return_value=None),
        patch("app.vector_store.milvus.milvus_client.get_milvus_client", return_value=mock_client),
    ):
        hits = voice_search(
            tmp_path / "indexes", catalog, query_video_id="a", query_utterance_index=0, limit=3
        )

    # Speaker retrieval now trusts the Milvus COSINE _distance directly (the
    # two-phase client-side re-score was removed). score therefore equals the
    # mock hit's distance (0.99) exactly, not the ~0.99995 the old path derived
    # by re-normalising the [0.99, 0.01] embedding — that discrepancy only ever
    # existed because the mock's distance and embedding were mutually inconsistent.
    assert hits[0]["video_id"] == "b"
    assert hits[0]["score"] == pytest.approx(0.99)
    assert {hit["video_id"] for hit in hits[:2]} == {"a", "b"}
    assert all("text" in hit and "clip_url" in hit for hit in hits)


def test_voice_search_skips_videos_without_published_speaker(tmp_path):
    """A no-speech video (0 utterances) never publishes a speaker asset_version.

    Such a video is a legal catalog member (decode_status=complete, utterances=0),
    but _published_asset_version(video_id, "speaker") raises for it. The cross-video
    voice-search loop must SKIP those videos rather than let one of them abort the
    whole request — the regression that surfaced as a 503 "speaker version is not
    published" after an index rebuild introduced a no-speech video.
    """
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    data_a = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[.99, .01], [-1, 0]], dtype=np.float32))
    # Video "c" has no speech → no published speaker version.
    make_video(catalog, "c", np.asarray([[0, 1], [1, 0]], dtype=np.float32))

    def _mock_speaker(video_id: str, **_kwargs):
        return {"a": data_a}[video_id]  # only the query video is loaded

    def _mock_published(video_id: str, modality: str) -> str:
        if video_id == "c":
            raise SpeakerMilvusCoverageError(
                f"Milvus {modality} version is not published for video {video_id}"
            )
        return "7"

    mock_client = Mock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    def mock_search_side_effect(data, anns_field, param, limit, expr, output_fields, timeout=None):
        match = re.search(r'video_id\s*==\s*["\']([^"\']+)["\']', expr)
        video_id = match.group(1) if match else ""
        if video_id == "a":
            hit = Mock()
            hit.distance = 1.0
            hit.entity = Mock()
            hit.entity.get = lambda field, default=None: {
                "utterance_idx": 0, "start_ms": 0, "end_ms": 1000,
                "track_id": 0, "asr_chunk_idx": 0, "_distance": 1.0,
            }.get(field, default)
            return [[hit]]
        if video_id == "b":
            hit = Mock()
            hit.distance = 0.8
            hit.entity = Mock()
            hit.entity.get = lambda field, default=None: {
                "utterance_idx": 0, "start_ms": 0, "end_ms": 1000,
                "track_id": 0, "asr_chunk_idx": 0, "_distance": 0.8,
            }.get(field, default)
            return [[hit]]
        # Video "c" must never reach Milvus search — it is skipped upstream.
        raise AssertionError(f"search must not be called for skipped video {video_id!r}")

    mock_collection.search.side_effect = mock_search_side_effect

    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", side_effect=_mock_speaker),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
        patch("app.identity.speaker_service._published_asset_version", side_effect=_mock_published),
        patch("app.vector_store.milvus.milvus_client.ensure_milvus_reachable", return_value=None),
        patch("app.vector_store.milvus.milvus_client.get_milvus_client", return_value=mock_client),
    ):
        # Must not raise SpeakerMilvusCoverageError despite "c" being unpublished.
        hits = voice_search(
            tmp_path / "indexes", catalog, query_video_id="a", query_utterance_index=0, limit=5
        )

    returned_videos = {hit["video_id"] for hit in hits}
    assert "c" not in returned_videos          # skipped, contributed nothing
    assert returned_videos == {"b"}            # "a"/utt0 is the excluded query source


def test_speaker_utterance_embedding_uses_primary_loader(tmp_path):
    data = _make_speaker_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    )
    with patch("app.identity.speaker_service._load_speaker_data", return_value=data) as load:
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
    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="sparse or duplicated"):
            _speaker_data_from_milvus("video-1")


def test_speaker_data_from_milvus_checks_expected_utterance_count():
    rows = [_milvus_speaker_row(0, [1.0, 0.0])]
    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="expected 2, got 1"):
            _speaker_data_from_milvus("video-1", expected_utterances=2)


def test_load_speaker_data_propagates_milvus_coverage_error(tmp_path):
    path = tmp_path / "video-1" / "speaker.npz"
    with patch(
        "app.identity.speaker_service._speaker_data_from_milvus",
        side_effect=SpeakerMilvusCoverageError("incomplete"),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="incomplete"):
            _load_speaker_data(path, "video-1")


def test_texts_from_milvus_propagates_storage_failure():
    with patch(
        "app.identity.speaker_service.ensure_milvus_reachable",
        side_effect=ConnectionError("connection refused"),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="ASR text is unavailable"):
            _texts_from_milvus("video-1")


def test_texts_from_milvus_accepts_a_successful_empty_query():
    client = Mock()
    client.collection_for.return_value.query.return_value = []
    with (
        patch("app.identity.speaker_service.ensure_milvus_reachable"),
        patch("app.identity.speaker_service._published_asset_version", return_value="7"),
        patch("app.identity.speaker_service.get_milvus_client", return_value=client),
    ):
        assert _texts_from_milvus("video-1") == []


def test_speaker_route_returns_503_for_milvus_coverage_error(monkeypatch, tmp_path):
    monkeypatch.setattr(context, "catalog", Mock(get_video=Mock(return_value={"id": "video-1"})))
    monkeypatch.setattr(context, "settings", Mock(index_dir=tmp_path / "indexes"))
    monkeypatch.setattr(
        speaker_routes,
        "video_speakers",
        Mock(side_effect=SpeakerMilvusCoverageError("ASR publication missing")),
    )

    with pytest.raises(HTTPException) as exc_info:
        speaker_routes.get_video_speakers("video-1")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "ASR publication missing"


def test_voice_search_returns_only_milvus_results(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    index_root = tmp_path / "indexes"
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
        patch(
            "app.identity.speaker_service._voice_search_vectors_milvus",
            return_value=([milvus_hit], {"a"}),
        ),
    ):
        hits = voice_search_vectors(
            index_root,
            catalog,
            query_vectors=np.asarray([[1, 0]], dtype=np.float32),
            video_ids=["a", "b"],
            limit=5,
        )

    assert hits == [milvus_hit]
