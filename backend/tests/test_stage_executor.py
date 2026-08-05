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
    options = {"asr_speaker_enabled": True}
    speaker_calls = []

    def fake_build_asr_index(**kwargs):
        with open(kwargs["output_path"], "wb") as output:
            output.write(b"fake-asr-index")
        return {"engine": kwargs["engine"], "retrieval_chunks": 1}

    def fake_build_speaker_index(**kwargs):
        speaker_calls.append(kwargs)
        return {"tracks": 2}

    monkeypatch.setattr(asr, "build_asr_index", fake_build_asr_index)
    monkeypatch.setattr(speaker, "build_speaker_index", fake_build_speaker_index)
    monkeypatch.setattr(stage_executor, "_setup_milvus_context", lambda *args: None)

    process_exit_result = execute_stage("asr", video, options, settings)
    pool = ModelPool(idle_timeout=0)
    try:
        daemon_result = execute_stage("asr", video, options, settings, pool)
    finally:
        pool.shutdown()

    assert process_exit_result["speaker"] == {"tracks": 2}
    assert daemon_result["speaker"] == {"tracks": 2}
    assert len(speaker_calls) == 2


def test_execute_stage_rejects_disabled_milvus(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        milvus_enabled=False,
    )
    video = {"id": "video-1", "file_path": str(tmp_path / "video.mp4")}

    import pytest

    with pytest.raises(RuntimeError, match="Milvus-only indexing requires"):
        execute_stage("visual", video, {}, settings)
