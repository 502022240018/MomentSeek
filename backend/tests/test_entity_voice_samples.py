from unittest.mock import patch

import numpy as np

from app.api.entity_routes import add_entity_voice_sample
from app.api.schemas import VoiceSampleRequest
from app.catalog.db import Catalog
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
