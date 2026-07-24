import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_process_exit_indexer_mode_is_normalized_for_backward_compatibility():
    settings = Settings(_env_file=None, indexer_mode="process_exit")

    assert settings.indexer_mode == "subprocess"


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ({"indexer_mode": "subprocess"}, "process_exit"),
        ({"indexer_mode": "daemon", "indexer_idle_timeout_seconds": 300}, "idle_release"),
        ({"indexer_mode": "daemon", "indexer_idle_timeout_seconds": 0}, "resident"),
    ),
)
def test_model_idle_policy_is_derived_from_effective_runtime_settings(values, expected):
    assert Settings(_env_file=None, **values).model_idle_policy == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (("indexer_mode", "typo"), ("npu_worker_mode", "shared")),
)
def test_invalid_worker_modes_fail_during_settings_load(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_milvus_query_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, milvus_query_timeout_seconds=0)
