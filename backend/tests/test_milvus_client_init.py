from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.vector_store.milvus.milvus_client import MilvusClient


def test_initialization_failure_never_marks_singleton_ready():
    client = object.__new__(MilvusClient)
    client._ready = False
    settings = SimpleNamespace(
        milvus_host="milvus",
        milvus_port=19530,
        milvus_query_timeout_seconds=5.0,
    )

    with (
        patch(
            "app.vector_store.milvus.milvus_client.get_settings",
            return_value=settings,
        ),
        patch(
            "app.vector_store.milvus.milvus_client._runtime_collection_layout",
            return_value=({}, {}),
        ),
        patch("app.vector_store.milvus.milvus_client.connections.connect") as connect,
        patch("app.vector_store.milvus.milvus_client.connections.disconnect") as disconnect,
        patch.object(
            client,
            "_init_collections",
            side_effect=RuntimeError("incompatible collection"),
        ),
        pytest.raises(RuntimeError, match="incompatible collection"),
    ):
        client.__init__()

    assert client._ready is False
    connect.assert_called_once()
    disconnect.assert_called_once_with(alias="default")


def test_successful_initialization_marks_client_ready_only_after_validation():
    client = object.__new__(MilvusClient)
    client._ready = False
    settings = SimpleNamespace(
        milvus_host="milvus",
        milvus_port=19530,
        milvus_query_timeout_seconds=5.0,
    )
    ready_during_validation: list[bool] = []

    def validate() -> None:
        ready_during_validation.append(client._ready)

    with (
        patch(
            "app.vector_store.milvus.milvus_client.get_settings",
            return_value=settings,
        ),
        patch(
            "app.vector_store.milvus.milvus_client._runtime_collection_layout",
            return_value=({}, {}),
        ),
        patch("app.vector_store.milvus.milvus_client.connections.connect"),
        patch.object(client, "_init_collections", side_effect=validate),
    ):
        client.__init__()

    assert ready_during_validation == [False]
    assert client._ready is True
