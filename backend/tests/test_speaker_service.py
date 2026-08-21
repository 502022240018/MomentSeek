from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import pytest
from fastapi import HTTPException
from app.api import speaker_routes
from app.catalog.db import Catalog
from app.platform import context
from app.identity.speaker_service import (
    SpeakerMilvusCoverageError,
    _attach_voice_hit_texts,
    _load_speaker_data,
    _speaker_data_from_milvus,
    _speaker_preview_bounds_ms,
    _texts_from_milvus,
    _voice_search_vectors_milvus,
    encode_voice_reference_file,
    speaker_utterance_embedding,
    video_speakers,
    voice_search,
    voice_search_vectors,
)


def test_speaker_preview_expands_short_evidence_without_changing_evidence_bounds():
    assert _speaker_preview_bounds_ms(1000, 1500, duration_seconds=10.0) == (0, 4000)
    assert _speaker_preview_bounds_ms(9000, 9500, duration_seconds=10.0) == (6000, 10000)


def test_uploaded_voice_reference_is_normalized_and_cleans_temporary_wav(tmp_path):
    source = tmp_path / "reference.webm"
    source.write_bytes(b"media")
    settings = SimpleNamespace(
        app_model_dir=Path("/models"),
        speaker_model_repo="speaker-repo",
        speaker_model_cache_dir="speaker-cache",
        speaker_device="cpu",
        resolve_path=lambda value: value,
    )
    encoded = np.zeros((1, 192), dtype=np.float32)
    encoded[0, :2] = [3.0, 4.0]

    with (
        patch("app.identity.speaker_service.subprocess.run") as ffmpeg,
        patch(
            "app.indexing.modalities.speaker.speaker.encode_voice_query",
            return_value=encoded,
        ) as encode,
    ):
        ffmpeg.return_value = SimpleNamespace(returncode=0, stderr="")
        result = encode_voice_reference_file(settings, source)

    assert np.allclose(result[0, :2], [0.6, 0.8])
    assert not source.with_suffix(".voice.wav").exists()
    assert "-ar" in ffmpeg.call_args.args[0]
    assert "16000" in ffmpeg.call_args.args[0]
    encode.assert_called_once()


def test_uploaded_voice_reference_cleans_temporary_wav_when_encoding_fails(tmp_path):
    source = tmp_path / "reference.mp4"
    source.write_bytes(b"media")
    wav_path = source.with_suffix(".voice.wav")
    wav_path.write_bytes(b"temporary")
    settings = SimpleNamespace(
        app_model_dir=Path("/models"),
        speaker_model_repo="speaker-repo",
        speaker_model_cache_dir="speaker-cache",
        speaker_device="cpu",
        resolve_path=lambda value: value,
    )

    with (
        patch(
            "app.identity.speaker_service.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stderr=""),
        ),
        patch(
            "app.indexing.modalities.speaker.speaker.encode_voice_query",
            side_effect=RuntimeError("encoder failed"),
        ),
        pytest.raises(RuntimeError, match="encoder failed"),
    ):
        encode_voice_reference_file(settings, source)

    assert not wav_path.exists()


def _make_speaker_data(vectors: np.ndarray) -> dict:
    """Build the same in-memory contract returned by the Milvus loader."""
    embeddings = np.asarray(vectors, dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    count = len(embeddings)
    return {
        "utterance_embeddings": embeddings,
        "utterance_times_ms": np.asarray(
            [[index * 2000, index * 2000 + 1000] for index in range(count)],
            dtype=np.int32,
        ),
        "utterance_refs": np.asarray(
            [[index, index] for index in range(count)], dtype=np.int32
        ),
        "track_representative_indices": np.arange(count, dtype=np.int32),
    }


_ASR_TEXTS = ["first sentence", "second sentence"]


def make_video(
    catalog: Catalog,
    video_id: str,
    vectors: np.ndarray,
    *,
    publish_speaker: bool = True,
):
    catalog.create_video({
        "id": video_id, "name": video_id, "file_path": f"/tmp/{video_id}.mp4",
        "duration": 10, "fps": 25, "width": 100, "height": 100, "status": "ready",
    })
    count = len(vectors)
    catalog.publish_modality(
        video_id,
        "asr",
        asset_version="asr-7",
        row_count=count,
        metadata={"retrieval_chunks": count},
    )
    if publish_speaker:
        catalog.publish_modality(
            video_id,
            "speaker",
            asset_version="speaker-7",
            row_count=count,
            metadata={
                "utterances": count,
                "tracks": count,
                "source_asr_asset_version": "asr-7",
            },
        )
    return _make_speaker_data(vectors)


def _milvus_speaker_row(index: int, embedding: list[float]) -> dict:
    return {
        "utterance_idx": index,
        "start_ms": index * 1000,
        "end_ms": (index + 1) * 1000,
        "asr_chunk_idx": index,
        "track_id": index,
        "embedding": embedding,
    }


def _speaker_vector(axis: int = 0) -> list[float]:
    vector = [0.0] * 192
    vector[axis] = 1.0
    return vector


def test_video_speakers_applies_mutable_sqlite_overlay(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    speaker_data = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    catalog.upsert_video_speaker("a", 0, display_name="Host")
    catalog.upsert_utterance_override("a", 1, 0, False)

    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", return_value=speaker_data),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
    ):
        payload = video_speakers(catalog, "a")

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
        _milvus_speaker_row(0, _speaker_vector(0)),
        _milvus_speaker_row(1, _speaker_vector(1)),
        _milvus_speaker_row(2, _speaker_vector(0)),
    ]
    # Two utterances on track 0, one on track 1 (override the default 1:1 mapping).
    rows[0]["track_id"] = 0
    rows[1]["track_id"] = 0
    rows[2]["track_id"] = 1

    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        data = _speaker_data_from_milvus(Mock(), "vid")

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
    make_video(
        catalog,
        "vid",
        np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
    )
    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", return_value=data),
        patch(
            "app.identity.speaker_service._texts_from_milvus",
            return_value=[*_ASR_TEXTS, "third sentence"],
        ),
    ):
        payload = video_speakers(catalog, "vid")

    assert {track["track_id"] for track in payload["tracks"]} == {0, 1}


def test_voice_search_matches_individual_utterances(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    data_a = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    data_b = make_video(catalog, "b", np.asarray([[.99, .01], [-1, 0]], dtype=np.float32))

    def _mock_speaker(_catalog, video_id: str, **_kwargs):
        return {"a": data_a, "b": data_b}[video_id]

    # Mock Milvus connection and search results
    mock_client = Mock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    def mock_search_side_effect(data, anns_field, param, limit, expr, output_fields, timeout=None):
        assert 'video_id == "a"' in expr and 'video_id == "b"' in expr
        hits = []
        for video_id, utterance_idx, distance in (("a", 0, 1.0), ("a", 1, 0.0), ("b", 0, 0.99)):
            hit = Mock()
            hit.distance = distance
            hit.entity = Mock()
            hit.entity.get = lambda field, default=None, values={
                "video_id": video_id,
                "asset_version": "7",
                "utterance_idx": utterance_idx,
                "start_ms": utterance_idx * 2000,
                "end_ms": utterance_idx * 2000 + 1000,
                "track_id": utterance_idx,
                "asr_chunk_idx": utterance_idx,
            }: values.get(field, default)
            hits.append(hit)
        return [hits for _ in data]

    mock_collection.search.side_effect = mock_search_side_effect

    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", side_effect=_mock_speaker),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
        patch("app.identity.speaker_service._published_asset_version", return_value="7"),
        patch("app.vector_store.milvus.milvus_client.ensure_milvus_reachable", return_value=None),
        patch("app.vector_store.milvus.milvus_client.get_milvus_client", return_value=mock_client),
    ):
        hits = voice_search(
            catalog, query_video_id="a", query_utterance_index=0, limit=3
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
    mock_collection.search.assert_called_once()
    expression = mock_collection.search.call_args.kwargs["expr"]
    assert 'video_id == "a"' in expression and 'video_id == "b"' in expression


def test_voice_search_skips_videos_without_published_speaker(tmp_path):
    """A video without a usable speaker publication is skipped.

    Such a video is a legal catalog member, but its speaker publication is absent.
    _published_asset_version(catalog, video_id, "speaker") raises for it. The cross-video
    voice-search loop must SKIP those videos rather than let one of them abort the
    whole request — the regression that surfaced as a 503 "speaker version is not
    published" after an index rebuild introduced a no-speech video.
    """
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    data_a = make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[.99, .01], [-1, 0]], dtype=np.float32))
    # Video "c" has no speech → no published speaker version.
    make_video(
        catalog,
        "c",
        np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        publish_speaker=False,
    )

    def _mock_speaker(_catalog, video_id: str, **_kwargs):
        return {"a": data_a}[video_id]  # only the query video is loaded

    def _mock_published(_catalog, video_id: str, modality: str) -> str:
        if video_id == "c":
            raise SpeakerMilvusCoverageError(
                f"Milvus {modality} version is not published for video {video_id}"
            )
        return "7"

    mock_client = Mock()
    mock_collection = Mock()
    mock_client.collection_for.return_value = mock_collection

    def mock_search_side_effect(data, anns_field, param, limit, expr, output_fields, timeout=None):
        assert 'video_id == "a"' in expr and 'video_id == "b"' in expr
        assert 'video_id == "c"' not in expr
        hits = []
        for video_id, distance in (("a", 1.0), ("b", 0.8)):
            hit = Mock()
            hit.distance = distance
            hit.entity = Mock()
            hit.entity.get = lambda field, default=None, values={
                "video_id": video_id,
                "asset_version": "7",
                "utterance_idx": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "track_id": 0,
                "asr_chunk_idx": 0,
            }: values.get(field, default)
            hits.append(hit)
        return [hits for _ in data]

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
            catalog, query_video_id="a", query_utterance_index=0, limit=5
        )

    returned_videos = {hit["video_id"] for hit in hits}
    assert "c" not in returned_videos          # skipped, contributed nothing
    assert returned_videos == {"b"}            # "a"/utt0 is the excluded query source
    mock_collection.search.assert_called_once()


def test_speaker_utterance_embedding_uses_primary_loader():
    catalog = Mock()
    data = _make_speaker_data(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    )
    with patch("app.identity.speaker_service._load_speaker_data", return_value=data) as load:
        vector = speaker_utterance_embedding(catalog, "video-1", 1)

    load.assert_called_once_with(catalog, "video-1")
    np.testing.assert_allclose(vector, np.asarray([0, 1], dtype=np.float32))


def test_speaker_data_from_milvus_rejects_sparse_utterance_indices():
    rows = [
        _milvus_speaker_row(0, [1.0, 0.0]),
        _milvus_speaker_row(2, [0.0, 1.0]),
    ]
    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="sparse or duplicated"):
            _speaker_data_from_milvus(Mock(), "video-1")


def test_speaker_data_from_milvus_checks_expected_utterance_count():
    rows = [_milvus_speaker_row(0, [1.0, 0.0])]
    with patch("app.identity.speaker_service._milvus_rows", return_value=rows):
        with pytest.raises(SpeakerMilvusCoverageError, match="expected 2, got 1"):
            _speaker_data_from_milvus(Mock(), "video-1", expected_utterances=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_ms", None),
        ("end_ms", 0),
        ("asr_chunk_idx", -1),
        ("track_id", -1),
    ],
)
def test_speaker_data_from_milvus_rejects_invalid_row_metadata(field, value):
    row = _milvus_speaker_row(0, _speaker_vector())
    row[field] = value
    with patch("app.identity.speaker_service._milvus_rows", return_value=[row]):
        with pytest.raises(SpeakerMilvusCoverageError, match="invalid metadata"):
            _speaker_data_from_milvus(Mock(), "video-1")


@pytest.mark.parametrize(
    "embedding",
    [
        [1.0, 0.0],
        [0.0] * 192,
        [float("nan")] + [0.0] * 191,
    ],
)
def test_speaker_data_from_milvus_rejects_invalid_embedding(embedding):
    row = _milvus_speaker_row(0, embedding)
    with patch("app.identity.speaker_service._milvus_rows", return_value=[row]):
        with pytest.raises(SpeakerMilvusCoverageError, match="embeddings"):
            _speaker_data_from_milvus(Mock(), "video-1")


def test_video_speakers_rejects_speaker_reference_to_missing_asr_chunk(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    data = make_video(catalog, "video-1", np.asarray([[1, 0]], dtype=np.float32))
    data["utterance_refs"][0, 0] = 2
    with (
        patch("app.identity.speaker_service._speaker_data_from_milvus", return_value=data),
        patch("app.identity.speaker_service._texts_from_milvus", return_value=["only"]),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="missing ASR chunk"):
            video_speakers(catalog, "video-1")


def test_load_speaker_data_propagates_milvus_coverage_error():
    catalog = Mock()
    catalog.get_modality_publication.side_effect = lambda _video_id, modality: {
        "asr": {
            "status": "ready",
            "asset_version": "asr-7",
            "row_count": 1,
        },
        "speaker": {
            "status": "ready",
            "asset_version": "speaker-7",
            "row_count": 1,
            "utterances": 1,
            "source_asr_asset_version": "asr-7",
        },
    }[modality]
    with patch(
        "app.identity.speaker_service._speaker_data_from_milvus",
        side_effect=SpeakerMilvusCoverageError("incomplete"),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="incomplete"):
            _load_speaker_data(catalog, "video-1")


def test_load_rejects_catalog_utterance_row_count_mismatch_before_milvus(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video({
        "id": "video-1",
        "name": "video.mp4",
        "file_path": "/tmp/video.mp4",
        "duration": 10,
        "fps": 25,
        "width": 100,
        "height": 100,
        "status": "ready",
    })
    catalog.publish_modality(
        "video-1",
        "asr",
        asset_version="asr-7",
        row_count=1,
    )
    catalog.publish_modality(
        "video-1",
        "speaker",
        asset_version="speaker-7",
        row_count=1,
        metadata={
            "utterances": 2,
            "source_asr_asset_version": "asr-7",
        },
    )

    with patch("app.identity.speaker_service._speaker_data_from_milvus") as load:
        with pytest.raises(SpeakerMilvusCoverageError, match="count mismatch"):
            _load_speaker_data(catalog, "video-1")
    load.assert_not_called()


def test_load_speaker_data_rejects_asr_generation_mismatch(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video({
        "id": "video-1",
        "name": "video.mp4",
        "file_path": "/tmp/video.mp4",
        "duration": 10,
        "fps": 25,
        "width": 100,
        "height": 100,
        "status": "ready",
    })
    catalog.publish_modality(
        "video-1", "asr", asset_version="asr-new", row_count=1,
    )
    catalog.publish_modality(
        "video-1",
        "speaker",
        asset_version="speaker-old",
        row_count=1,
        metadata={
            "utterances": 1,
            "source_asr_asset_version": "asr-old",
        },
    )

    with patch("app.identity.speaker_service._speaker_data_from_milvus") as load:
        with pytest.raises(SpeakerMilvusCoverageError, match="publication mismatch"):
            _load_speaker_data(catalog, "video-1")
    load.assert_not_called()


def test_texts_from_milvus_propagates_storage_failure():
    catalog = Mock()
    catalog.get_modality_publication.return_value = {
        "status": "ready",
        "asset_version": "asr-7",
        "row_count": 1,
    }
    with patch(
        "app.identity.speaker_service.ensure_milvus_reachable",
        side_effect=ConnectionError("connection refused"),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="ASR text is unavailable"):
            _texts_from_milvus(catalog, "video-1")


def test_texts_from_milvus_rejects_rows_incomplete_against_catalog():
    catalog = Mock()
    catalog.get_modality_publication.return_value = {
        "status": "ready",
        "asset_version": "asr-7",
        "row_count": 1,
    }
    client = Mock()
    collection = Mock()
    del collection.query_iterator
    collection.query.return_value = []
    client.collection_for.return_value = collection
    with (
        patch("app.identity.speaker_service.ensure_milvus_reachable"),
        patch("app.identity.speaker_service.get_milvus_client", return_value=client),
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="incomplete"):
            _texts_from_milvus(catalog, "video-1")


def test_speaker_route_returns_503_for_milvus_coverage_error(monkeypatch):
    catalog = Mock(get_video=Mock(return_value={"id": "video-1"}))
    service = Mock(side_effect=SpeakerMilvusCoverageError("ASR publication missing"))
    monkeypatch.setattr(context, "catalog", catalog)
    monkeypatch.setattr(
        speaker_routes,
        "video_speakers",
        service,
    )

    with pytest.raises(HTTPException) as exc_info:
        speaker_routes.get_video_speakers("video-1")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "ASR publication missing"
    service.assert_called_once_with(catalog, "video-1")


def test_voice_search_returns_only_milvus_results(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, "b", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
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
                return_value=[milvus_hit],
        ),
    ):
        hits = voice_search_vectors(
            catalog,
            query_vectors=np.asarray([[1, 0]], dtype=np.float32),
            video_ids=["a", "b"],
            limit=5,
        )

    assert hits == [milvus_hit]


def test_voice_search_explicit_empty_scope_does_not_connect_to_milvus(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")

    hits = _voice_search_vectors_milvus(
        catalog,
        queries=np.asarray([[1.0] + [0.0] * 191], dtype=np.float32),
        video_ids=[],
        limit=5,
        exclude=None,
    )

    assert hits == []


def test_voice_hit_texts_reads_asr_once_per_video():
    hits = [
        {"video_id": "a", "asr_chunk_index": 0},
        {"video_id": "a", "asr_chunk_index": 1},
        {"video_id": "b", "asr_chunk_index": 0},
    ]
    with patch(
        "app.identity.speaker_service._texts_from_milvus",
        side_effect=lambda _catalog, video_id: {
            "a": ["a0", "a1"],
            "b": ["b0"],
        }[video_id],
    ) as load:
        _attach_voice_hit_texts(Mock(), hits, limit=3)

    assert load.call_count == 2
    assert [hit["text"] for hit in hits] == ["a0", "a1", "b0"]


def test_voice_hit_texts_rejects_missing_asr_reference():
    hits = [{"video_id": "a", "asr_chunk_index": 2}]
    with patch(
        "app.identity.speaker_service._texts_from_milvus",
        return_value=["only"],
    ):
        with pytest.raises(SpeakerMilvusCoverageError, match="missing ASR chunk"):
            _attach_voice_hit_texts(Mock(), hits, limit=1)
