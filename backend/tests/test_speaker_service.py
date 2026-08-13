from pathlib import Path
from unittest.mock import patch

import numpy as np
from app.db import Catalog
from app.indexing.speaker import load_speaker_index, save_speaker_index
from app.speaker_service import video_speakers, voice_search


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

    def _mock_speaker(video_id: str):
        return {"a": data_a, "b": data_b}[video_id]

    with (
        patch("app.speaker_service.milvus_read_enabled", return_value=True),
        patch("app.speaker_service._speaker_data_from_milvus", side_effect=_mock_speaker),
        patch("app.speaker_service._texts_from_milvus", return_value=_ASR_TEXTS),
    ):
        hits = voice_search(
            tmp_path / "indexes", catalog, query_video_id="a", query_utterance_index=0, limit=3
        )

    assert hits[0]["score"] > .99
    assert {hit["video_id"] for hit in hits[:2]} == {"a", "b"}
    assert all("text" in hit and "clip_url" in hit for hit in hits)
