"""Unit coverage for the one-time Face IVF_FLAT/L2 -> DiskANN/COSINE migration."""
from __future__ import annotations

from unittest.mock import MagicMock

from pymilvus.exceptions import IndexNotExistException

from scripts import migrate_face_diskann_index as migration


def test_index_details_handles_collection_without_an_index():
    collection = MagicMock()
    collection.index.side_effect = IndexNotExistException(message="missing")

    assert migration._index_details(collection) == {
        "index_type": None,
        "metric_type": None,
    }


def test_replace_vector_index_preserves_collection_and_uses_current_config(monkeypatch):
    collection = MagicMock()
    config = {
        "index_type": "DISKANN",
        "metric_type": "COSINE",
        "params": {"max_degree": 56},
    }
    monkeypatch.setattr(migration, "get_collection_index_config", lambda _: config)

    migration._replace_vector_index(collection)

    collection.release.assert_called_once_with()
    collection.drop_index.assert_called_once_with()
    collection.create_index.assert_called_once_with(
        field_name="embedding",
        index_params=config,
    )
    collection.load.assert_called_once_with()


def test_collection_state_requires_diskann_and_cosine(monkeypatch):
    collection = MagicMock()
    collection.num_entities = 12
    collection.index.return_value.params = {
        "index_type": "DISKANN",
        "metric_type": "L2",
    }
    monkeypatch.setattr(migration.utility, "has_collection", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(migration, "Collection", lambda *_args, **_kwargs: collection)

    state = migration._collection_state()

    assert state["ready"] is False
    assert state["row_count"] == 12
    assert state["index_type"] == "DISKANN"
    assert state["metric_type"] == "L2"
