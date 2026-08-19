from types import SimpleNamespace

import pytest

from app.core.model_pool import ModelPool
from app.core.settings import Settings
from app.indexing.stage_executor import execute_stage


def test_asr_speaker_postprocessing_is_identical_with_and_without_pool(monkeypatch, tmp_path):
    import app.indexing.modalities.asr.asr as asr
    import app.indexing.modalities.speaker.speaker as speaker
    import app.indexing.stage_executor as stage_executor

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        asr_semantic_enabled=False,
        speaker_device="cpu",
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"fake")
    video = {"id": "video-1", "file_path": str(video_path), "duration": 1.0}
    catalog = stage_executor.Catalog(settings.db_path)
    catalog.create_video(
        {
            **video,
            "name": "video.mp4",
            "fps": 25.0,
            "width": 640,
            "height": 480,
            "status": "ready",
        }
    )
    options = {"asr_speaker_enabled": True}
    speaker_calls = []

    def fake_build_asr_index(**kwargs):
        assert "output_path" not in kwargs
        return {
            "engine": kwargs["engine"],
            "retrieval_chunks": 1,
            "milvus_rows": 1,
        }

    def fake_build_speaker_index(**kwargs):
        assert "asr_path" not in kwargs
        assert "output_path" not in kwargs
        assert kwargs["asr_asset_version"] == "test-version"
        speaker_calls.append(kwargs)
        return {"tracks": 2, "utterances": 2, "milvus_rows": 2}

    client = SimpleNamespace(
        count_video_modality_version=lambda video_id, modality, version: {
            "asr": 1,
            "speaker": 2,
        }[modality]
    )
    milvus_ctx = SimpleNamespace(
        video_id="video-1",
        asset_version="test-version",
        client=client,
    )

    monkeypatch.setattr(asr, "build_asr_index", fake_build_asr_index)
    monkeypatch.setattr(speaker, "build_speaker_index", fake_build_speaker_index)
    monkeypatch.setattr(
        stage_executor,
        "_setup_milvus_context",
        lambda *args: milvus_ctx,
    )

    process_exit_result = execute_stage("asr", video, options, settings)
    pool = ModelPool(idle_timeout=0)
    try:
        daemon_result = execute_stage("asr", video, options, settings, pool)
    finally:
        pool.shutdown()

    assert process_exit_result["speaker"]["tracks"] == 2
    assert daemon_result["speaker"]["tracks"] == 2
    assert len(speaker_calls) == 2

    assert catalog.get_modality_publication("video-1", "asr")["row_count"] == 1
    speaker_publication = catalog.get_modality_publication("video-1", "speaker")
    assert speaker_publication["row_count"] == 2
    assert speaker_publication["source_asr_asset_version"] == "test-version"


def test_asr_speaker_failure_keeps_both_previous_publications(monkeypatch, tmp_path):
    import app.indexing.modalities.asr.asr as asr
    import app.indexing.modalities.speaker.speaker as speaker
    import app.indexing.stage_executor as stage_executor

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        asr_semantic_enabled=False,
        speaker_device="cpu",
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"fake")
    video = {"id": "video-1", "file_path": str(video_path), "duration": 1.0}
    catalog = stage_executor.Catalog(settings.db_path)
    catalog.create_video({
        **video, "name": "video.mp4", "fps": 25.0,
        "width": 640, "height": 480, "status": "ready",
    })
    catalog.publish_modalities(
        "video-1",
        [
            {"modality": "asr", "asset_version": "old-asr", "row_count": 1},
            {
                "modality": "speaker",
                "asset_version": "old-speaker",
                "row_count": 1,
                "metadata": {
                    "utterances": 1,
                    "source_asr_asset_version": "old-asr",
                },
            },
        ],
    )

    monkeypatch.setattr(
        asr,
        "build_asr_index",
        lambda **_kwargs: {"engine": "fake", "milvus_rows": 2},
    )

    def fail_speaker(**_kwargs):
        raise RuntimeError("speaker build failed")

    monkeypatch.setattr(speaker, "build_speaker_index", fail_speaker)
    milvus_ctx = SimpleNamespace(
        video_id="video-1",
        asset_version="new-generation",
        client=SimpleNamespace(
            count_video_modality_version=lambda *_args: 2,
        ),
    )
    monkeypatch.setattr(
        stage_executor, "_setup_milvus_context", lambda *args: milvus_ctx,
    )

    with pytest.raises(RuntimeError, match="speaker build failed"):
        execute_stage(
            "asr", video, {"asr_speaker_enabled": True}, settings,
        )

    assert catalog.get_modality_publication("video-1", "asr")["asset_version"] == "old-asr"
    speaker_publication = catalog.get_modality_publication("video-1", "speaker")
    assert speaker_publication["asset_version"] == "old-speaker"
    assert speaker_publication["status"] == "ready"


def test_asr_only_publish_disables_previous_speaker_atomically(monkeypatch, tmp_path):
    import app.indexing.modalities.asr.asr as asr
    import app.indexing.stage_executor as stage_executor

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        asr_semantic_enabled=False,
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"fake")
    video = {"id": "video-1", "file_path": str(video_path), "duration": 1.0}
    catalog = stage_executor.Catalog(settings.db_path)
    catalog.create_video({
        **video, "name": "video.mp4", "fps": 25.0,
        "width": 640, "height": 480, "status": "ready",
    })
    catalog.publish_modalities(
        "video-1",
        [
            {"modality": "asr", "asset_version": "old-asr", "row_count": 1},
            {
                "modality": "speaker",
                "asset_version": "old-speaker",
                "row_count": 1,
                "metadata": {
                    "utterances": 1,
                    "source_asr_asset_version": "old-asr",
                },
            },
        ],
    )
    monkeypatch.setattr(
        asr,
        "build_asr_index",
        lambda **_kwargs: {"engine": "fake", "milvus_rows": 2},
    )
    milvus_ctx = SimpleNamespace(
        video_id="video-1",
        asset_version="new-asr",
        client=SimpleNamespace(
            count_video_modality_version=lambda *_args: 2,
        ),
    )
    monkeypatch.setattr(
        stage_executor, "_setup_milvus_context", lambda *args: milvus_ctx,
    )

    execute_stage("asr", video, {}, settings)

    assert catalog.get_modality_publication("video-1", "asr")["asset_version"] == "new-asr"
    speaker_publication = catalog.get_modality_publication("video-1", "speaker")
    assert speaker_publication["asset_version"] == "old-speaker"
    assert speaker_publication["status"] == "disabled"
    assert catalog.get_video("video-1")["indexed_modalities"] == ["asr"]


def test_standalone_speaker_records_ready_asr_source_version(monkeypatch, tmp_path):
    import app.indexing.modalities.speaker.speaker as speaker
    import app.indexing.stage_executor as stage_executor

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        speaker_device="cpu",
        milvus_enabled=True,
    )
    settings.ensure_dirs()
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"fake")
    video = {"id": "video-1", "file_path": str(video_path), "duration": 1.0}
    catalog = stage_executor.Catalog(settings.db_path)
    catalog.create_video({
        **video, "name": "video.mp4", "fps": 25.0,
        "width": 640, "height": 480, "status": "ready",
    })
    catalog.publish_modality(
        "video-1", "asr", asset_version="ready-asr", row_count=1,
    )

    def fake_build_speaker_index(**kwargs):
        assert kwargs["asr_asset_version"] == "ready-asr"
        return {"tracks": 1, "utterances": 1, "milvus_rows": 1}

    monkeypatch.setattr(speaker, "build_speaker_index", fake_build_speaker_index)
    milvus_ctx = SimpleNamespace(
        video_id="video-1",
        asset_version="speaker-generation",
        client=SimpleNamespace(
            count_video_modality_version=lambda *_args: 1,
        ),
    )
    monkeypatch.setattr(
        stage_executor, "_setup_milvus_context", lambda *args: milvus_ctx,
    )

    execute_stage("speaker", video, {}, settings)

    publication = catalog.get_modality_publication("video-1", "speaker")
    assert publication["asset_version"] == "speaker-generation"
    assert publication["source_asr_asset_version"] == "ready-asr"


def test_execute_stage_rejects_disabled_milvus(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        milvus_enabled=False,
    )
    video = {"id": "video-1", "file_path": str(tmp_path / "video.mp4")}

    with pytest.raises(RuntimeError, match="Milvus-only indexing requires"):
        execute_stage("visual", video, {}, settings)
