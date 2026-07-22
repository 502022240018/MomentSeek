from app.model_pool import ModelPool
from app.settings import Settings
from app.stage_executor import execute_stage


def test_asr_speaker_postprocessing_is_identical_with_and_without_pool(monkeypatch, tmp_path):
    import app.indexing.asr as asr
    import app.indexing.speaker as speaker

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        asr_semantic_enabled=False,
        speaker_device="cpu",
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

    process_exit_result = execute_stage("asr", video, options, settings)
    pool = ModelPool(idle_timeout=0)
    try:
        daemon_result = execute_stage("asr", video, options, settings, pool)
    finally:
        pool.shutdown()

    assert process_exit_result["speaker"] == {"tracks": 2}
    assert daemon_result["speaker"] == {"tracks": 2}
    assert len(speaker_calls) == 2
