from unittest.mock import patch

import numpy as np

from app.api.entity_routes import add_entity_voice_sample
from app.api.schemas import VoiceSampleRequest
from app.catalog.db import Catalog
from app.identity.speaker_service import entity_voice_embeddings
from app.platform import context


def test_voice_sample_stores_embedding_blob_without_npz(monkeypatch, tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_entity({
        "id": "person-1",
        "name": "Person",
        "reference_path": "",
        "embedding_path": None,
    })
    monkeypatch.setattr(context, "catalog", catalog)
    vector = np.arange(192, dtype=np.float32)

    with patch(
        "app.identity.speaker_service.speaker_utterance_embedding",
        return_value=vector,
    ) as embedding:
        sample = add_entity_voice_sample(
            "person-1",
            VoiceSampleRequest(video_id="video-1", utterance_index=3),
        )

    embedding.assert_called_once_with(catalog, "video-1", 3)
    assert sample["embedding_path"] == ""
    assert list(tmp_path.rglob("*.npz")) == []
    with catalog.connect() as connection:
        row = connection.execute(
            "SELECT embedding_path, voice_embedding FROM voice_samples WHERE id=?",
            (sample["id"],),
        ).fetchone()
    assert row["embedding_path"] == ""
    np.testing.assert_array_equal(np.frombuffer(row["voice_embedding"], dtype=np.float32), vector)


def test_entity_voice_embeddings_validates_and_normalizes_all_samples(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_entity({
        "id": "person-1",
        "name": "Person",
        "reference_path": "",
        "embedding_path": None,
    })
    for index, axis in enumerate((0, 1)):
        vector = np.zeros(192, dtype=np.float32)
        vector[axis] = 2.0
        catalog.create_voice_sample({
            "id": f"voice-{index}",
            "entity_id": "person-1",
            "source_type": "utterance",
            "source_video_id": "video-1",
            "source_utterance_index": index,
            "audio_path": None,
            "embedding_path": "",
            "embedding_space": "speaker",
            "voice_embedding": vector.tobytes(),
        })

    vectors = entity_voice_embeddings(catalog, "person-1")

    assert vectors.shape == (2, 192)
    assert "voice_embedding" not in catalog.list_voice_samples("person-1")[0]
    assert isinstance(
        catalog.list_voice_sample_embeddings("person-1")[0]["voice_embedding"], bytes
    )
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[0, 0] == 1.0
    assert vectors[1, 1] == 1.0


def test_entity_voice_embeddings_rejects_corrupt_sample_set(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_entity({
        "id": "person-1",
        "name": "Person",
        "reference_path": "",
        "embedding_path": None,
    })
    catalog.create_voice_sample({
        "id": "voice-bad",
        "entity_id": "person-1",
        "source_type": "utterance",
        "source_video_id": "video-1",
        "source_utterance_index": 0,
        "audio_path": None,
        "embedding_path": "",
        "embedding_space": "speaker",
        "voice_embedding": np.zeros(192, dtype=np.float32).tobytes(),
    })

    with np.testing.assert_raises_regex(ValueError, "零向量"):
        entity_voice_embeddings(catalog, "person-1")
