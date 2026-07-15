from pathlib import Path

import numpy as np

from app.db import Catalog
from app.indexing.speaker import save_speaker_index
from app.speaker_service import video_speakers, voice_search


def make_video(catalog: Catalog, root: Path, video_id: str, vectors: np.ndarray):
    catalog.create_video({
        "id": video_id, "name": video_id, "file_path": str(root / f"{video_id}.mp4"),
        "duration": 10, "fps": 25, "width": 100, "height": 100, "status": "ready",
    })
    directory = root / "indexes" / video_id
    directory.mkdir(parents=True)
    np.savez_compressed(
        directory / "asr.npz",
        chunk_times_ms=np.asarray([[0, 1000], [2000, 3000]], dtype=np.int32),
        texts=np.asarray(["first sentence", "second sentence"]),
        embeddings=np.empty((0, 0), dtype=np.float16),
        embedding_chunk_indices=np.empty((0,), dtype=np.int32),
    )
    save_speaker_index(
        directory / "speaker.npz",
        utterance_times_ms=np.asarray([[0, 1000], [2000, 3000]]),
        utterance_embeddings=vectors,
        asr_chunk_indices=np.asarray([0, 1]), auto_track_indices=np.asarray([0, 1]),
    )


def test_video_speakers_applies_mutable_sqlite_overlay(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, tmp_path, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    catalog.upsert_video_speaker("a", 0, display_name="Host")
    catalog.upsert_utterance_override("a", 1, 0, False)

    payload = video_speakers(tmp_path / "indexes", catalog, "a")

    assert payload["tracks"][0]["label"] == "Host"
    assert payload["tracks"][0]["utterance_indices"] == [0, 1]
    assert payload["utterances"][1]["searchable"] is False


def test_voice_search_matches_individual_utterances(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    make_video(catalog, tmp_path, "a", np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    make_video(catalog, tmp_path, "b", np.asarray([[.99, .01], [-1, 0]], dtype=np.float32))

    hits = voice_search(
        tmp_path / "indexes", catalog, query_video_id="a", query_utterance_index=0, limit=3
    )

    assert hits[0]["score"] > .99
    assert {hit["video_id"] for hit in hits[:2]} == {"a", "b"}
    assert all("text" in hit and "clip_url" in hit for hit in hits)
