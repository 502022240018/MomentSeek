import pytest
from pydantic import ValidationError

from app.core.settings import Settings


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


@pytest.mark.parametrize("field", ("milvus_asr_collection", "milvus_speaker_collection"))
@pytest.mark.parametrize("name", ("", "9asr", "asr-v2", "asr/v2", "a" * 256))
def test_invalid_milvus_collection_names_fail(field, name):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: name})


def test_milvus_asr_collection_name_is_normalized():
    settings = Settings(
        _env_file=None,
        milvus_asr_collection="  asr_embeddings_planner_v2  ",
    )

    assert settings.milvus_asr_collection == "asr_embeddings_planner_v2"


def test_milvus_speaker_collection_name_is_normalized():
    settings = Settings(
        _env_file=None,
        milvus_speaker_collection="  speaker_embeddings_diskann_v2  ",
    )

    assert settings.milvus_speaker_collection == "speaker_embeddings_diskann_v2"


def test_milvus_query_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, milvus_query_timeout_seconds=0)


def test_color_grading_request_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, color_grading_request_timeout_seconds=0)


def test_milvus_search_video_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, milvus_search_video_batch_size=0)


def test_milvus_search_max_workers_is_safely_bounded():
    assert Settings(_env_file=None).milvus_search_max_workers == 1
    assert Settings(_env_file=None, milvus_search_max_workers=8).milvus_search_max_workers == 8
    with pytest.raises(ValidationError):
        Settings(_env_file=None, milvus_search_max_workers=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, milvus_search_max_workers=9)


def test_visual_priority_is_enabled_by_default_and_can_be_disabled():
    assert Settings(_env_file=None).search_visual_priority_enabled is True
    assert Settings(_env_file=None, search_visual_priority_enabled=False).search_visual_priority_enabled is False


@pytest.mark.parametrize("value", (0, 16_384))
def test_visual_ann_top_k_is_bounded_by_milvus_limit(value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, visual_ann_top_k=value)


def test_visual_ann_top_k_accepts_supported_bounds():
    assert Settings(_env_file=None, visual_ann_top_k=1).visual_ann_top_k == 1
    assert (
        Settings(_env_file=None, visual_ann_top_k=16_383).visual_ann_top_k
        == 16_383
    )


def test_visual_ann_segment_top_n_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, visual_ann_segment_top_n=0)
