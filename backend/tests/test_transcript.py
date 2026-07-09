import json
import subprocess
import sys
import types

import numpy as np
import pytest

from app.indexing import asr
from app.indexing.asr import build_asr_index, load_sidecar
from app.search import lexical_score


def test_load_json_and_srt(tmp_path):
    json_path = tmp_path / "demo.json"
    json_path.write_text(
        json.dumps({"segments": [{"start": 1, "end": 3, "text": "电影投资"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_sidecar(json_path) == [{"start_time": 1.0, "end_time": 3.0, "text": "电影投资"}]

    srt_path = tmp_path / "demo.srt"
    srt_path.write_text("1\n00:00:04,500 --> 00:00:07,000\n欢迎来到节目\n", encoding="utf-8")
    chunks = load_sidecar(srt_path)
    assert chunks[0]["start_time"] == 4.5
    assert chunks[0]["end_time"] == 7.0


def test_lexical_score_supports_chinese_and_substrings():
    assert lexical_score("电影投资", "今天我们聊一聊电影投资的趋势") == 1
    assert lexical_score("电影投", "电影投资") > 0.5
    assert lexical_score("完全不同", "电影投资") < 0.5
    assert lexical_score("台湾", "臺灣") == 1
    assert lexical_score("这里有台词", "這裏有臺詞") == 1


def test_asr_index_handles_video_without_audio(tmp_path, monkeypatch):
    def no_audio(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(asr, "extract_audio", no_audio)
    output_path = tmp_path / "asr.npz"
    semantic_path = tmp_path / "asr_semantic.npz"
    semantic_path.write_bytes(b"stale")

    result = build_asr_index(
        video_path=str(tmp_path / "silent.mp4"),
        output_path=str(output_path),
        working_dir=str(tmp_path / "work"),
        engine="whisper",
        model_name="tiny",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        semantic_output_path=str(semantic_path),
    )

    assert result["engine"] == "no_audio"
    assert result["chunks"] == 0
    assert result["decode_status"] == "empty"
    with np.load(output_path, allow_pickle=False) as data:
        assert data["chunk_times_ms"].shape == (0, 2)
        assert data["texts"].tolist() == []
        assert data["embeddings"].shape == (0, 0)
        assert data["embedding_chunk_indices"].tolist() == []
    assert not semantic_path.exists()


def test_whisper_forces_transcribe_task_and_records_detected_language(tmp_path, monkeypatch):
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **options):
            captured["audio_shape"] = audio.shape
            captured["options"] = options
            return {
                "language": "zh",
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": " 你好 "},
                ],
            }

    fake_whisper = types.SimpleNamespace(
        load_model=lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(asr, "load_wav_mono", lambda _path: np.zeros((16000,), dtype=np.float32))

    chunks, metadata = asr._whisper(
        str(tmp_path / "audio.wav"),
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        language="auto",
        local_files_only=False,
    )

    assert chunks == [{"start_time": 1.0, "end_time": 2.0, "text": "你好"}]
    assert captured["options"]["task"] == "transcribe"
    assert metadata["task"] == "transcribe"
    assert metadata["requested_language"] == "auto"
    assert metadata["detected_language"] == "zh"


def test_whisper_zh_language_does_not_pass_initial_prompt(tmp_path, monkeypatch):
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **options):
            captured["options"] = options
            return {
                "language": "zh",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello"},
                ],
            }

    fake_whisper = types.SimpleNamespace(
        load_model=lambda *_args, **_kwargs: FakeModel(),
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(asr, "load_wav_mono", lambda _path: np.zeros((16000,), dtype=np.float32))

    asr._whisper(
        str(tmp_path / "audio.wav"),
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        language="zh",
        local_files_only=False,
    )

    assert captured["options"]["task"] == "transcribe"
    assert captured["options"]["language"] == "zh"
    assert "initial_prompt" not in captured["options"]


def test_whisper_local_files_only_requires_cached_pt(tmp_path):
    with pytest.raises(FileNotFoundError, match="本地 Whisper 模型缺失"):
        asr._whisper(
            str(tmp_path / "audio.wav"),
            model_name="small",
            device="cpu",
            model_dir=str(tmp_path / "models"),
            language="zh",
            local_files_only=True,
        )


def test_funasr_local_files_only_requires_cached_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="本地 ModelScope/FunASR 模型缺失"):
        asr._funasr(
            str(tmp_path / "audio.wav"),
            "iic/SenseVoiceSmall",
            "cpu",
            model_root=tmp_path / "models" / "funasr",
            local_files_only=True,
        )


def test_sensevoice_funasr_uses_timestamp_flags_without_external_punc(tmp_path, monkeypatch):
    calls = {}

    def fake_resolver(_root, model_name, *, local_files_only=True):
        return {
            "iic/SenseVoiceSmall": "/models/funasr/sensevoice",
            "fsmn-vad": "/models/funasr/vad",
            "ct-punc": "/models/funasr/punc",
        }[model_name]

    class FakeAutoModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [{"text": "你好世界", "timestamp": [[1000, 1200], [1200, 1400], [1400, 1600], [1600, 1800]]}]

    fake_funasr = types.SimpleNamespace(AutoModel=FakeAutoModel)
    fake_postprocess = types.SimpleNamespace(rich_transcription_postprocess=lambda text: text)
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.setitem(sys.modules, "funasr.utils", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", fake_postprocess)
    monkeypatch.setattr(asr, "resolve_modelscope_model_source", fake_resolver)

    chunks = asr._funasr(
        str(tmp_path / "audio.wav"),
        "iic/SenseVoiceSmall",
        "cuda",
        model_root=tmp_path / "models" / "funasr",
        local_files_only=True,
        language="zh",
    )

    assert calls["init"]["model"] == "/models/funasr/sensevoice"
    assert calls["init"]["vad_model"] == "/models/funasr/vad"
    assert "punc_model" not in calls["init"]
    assert calls["init"]["vad_kwargs"] == {"max_single_segment_time": 30000}
    assert calls["generate"]["output_timestamp"] is True
    assert calls["generate"]["return_time_stamps"] is True
    assert calls["generate"]["merge_vad"] is True
    assert "sentence_timestamp" not in calls["generate"]
    assert chunks == [
        {
            "item_id": 0,
            "start_ms": 1000,
            "end_ms": 1800,
            "text": "你好世界",
            "source": "funasr_timestamp",
        }
    ]


def test_sensevoice_silero_strategy_uses_external_vad_groups(tmp_path, monkeypatch):
    calls = {"generate": []}

    def fake_resolver(_root, model_name, *, local_files_only=True):
        return {
            "iic/SenseVoiceSmall": "/models/funasr/sensevoice",
            "fsmn-vad": "/models/funasr/vad",
            "ct-punc": "/models/funasr/punc",
        }[model_name]

    class FakeAutoModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            if "0000" in str(kwargs["input"]):
                return [{"text": "first sentence.", "timestamp": [[0, 300], [400, 700], [800, 1100], [1200, 1500], [1600, 1900], [2000, 2300], [2400, 2700], [2800, 3100], [3200, 3500], [3600, 3900], [4000, 4300], [4400, 4700], [4800, 5100]]}]
            return [{"text": "second sentence.", "timestamp": [[0, 300], [400, 700], [800, 1100], [1200, 1500], [1600, 1900], [2000, 2300], [2400, 2700], [2800, 3100], [3200, 3500], [3600, 3900], [4000, 4300], [4400, 4700], [4800, 5100], [5200, 5500]]}]

    def fake_get_speech_timestamps(_audio, _model, **_kwargs):
        return [
            {"start": 0, "end": 6 * 16000},
            {"start": int(6.5 * 16000), "end": 11 * 16000},
            {"start": 15 * 16000, "end": 19 * 16000},
        ]

    fake_funasr = types.SimpleNamespace(AutoModel=FakeAutoModel)
    fake_silero = types.SimpleNamespace(
        load_silero_vad=lambda: object(),
        get_speech_timestamps=fake_get_speech_timestamps,
    )
    fake_postprocess = types.SimpleNamespace(rich_transcription_postprocess=lambda text: text)
    monkeypatch.setitem(sys.modules, "funasr", fake_funasr)
    monkeypatch.setitem(sys.modules, "silero_vad", fake_silero)
    monkeypatch.setitem(sys.modules, "funasr.utils", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", fake_postprocess)
    monkeypatch.setattr(asr, "resolve_modelscope_model_source", fake_resolver)
    monkeypatch.setattr(asr, "load_wav_mono", lambda _path: np.zeros((20 * 16000,), dtype=np.float32))

    chunks = asr._funasr(
        str(tmp_path / "audio.wav"),
        "iic/SenseVoiceSmall",
        "cuda",
        model_root=tmp_path / "models" / "funasr",
        local_files_only=True,
        language="zh",
        vad_strategy="silero_12s",
        temp_dir=tmp_path / "clips",
    )

    assert calls["init"]["model"] == "/models/funasr/sensevoice"
    assert "vad_model" not in calls["init"]
    assert len(calls["generate"]) == 2
    assert all(call["merge_vad"] is False for call in calls["generate"])
    assert all(call["output_timestamp"] is True for call in calls["generate"])
    assert chunks[0]["start_ms"] == 0
    assert chunks[1]["start_ms"] == 15000
    assert [chunk["text"] for chunk in chunks] == ["first sentence.", "second sentence."]


def test_funasr_timestamped_text_parser_returns_raw_item_without_duration_split():
    text = "hello world. next part."
    timed = [char for char in text if char.isalnum()]
    timestamps = [[index * 1000, index * 1000 + 800] for index in range(len(timed))]

    chunks = asr._parse_funasr_chunks(
        [{"text": text, "timestamp": timestamps}],
        is_sensevoice=True,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello world. next part."
    assert chunks[0]["start_ms"] == 0
    assert chunks[0]["end_ms"] > 12000


def test_paraformer_funasr_keeps_sentence_timestamp_and_punc(tmp_path, monkeypatch):
    calls = {}

    def fake_resolver(_root, model_name, *, local_files_only=True):
        return {
            "paraformer-zh": "/models/funasr/paraformer",
            "fsmn-vad": "/models/funasr/vad",
            "ct-punc": "/models/funasr/punc",
        }[model_name]

    class FakeAutoModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [{"sentence_info": [{"start": 500, "end": 1600, "text": "你好"}]}]

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))
    monkeypatch.setattr(asr, "resolve_modelscope_model_source", fake_resolver)

    chunks = asr._funasr(
        str(tmp_path / "audio.wav"),
        "paraformer-zh",
        "cpu",
        model_root=tmp_path / "models" / "funasr",
        local_files_only=True,
        language="zh",
    )

    assert calls["init"]["punc_model"] == "/models/funasr/punc"
    assert calls["generate"]["sentence_timestamp"] is True
    assert "output_timestamp" not in calls["generate"]
    assert chunks == [
        {
            "item_id": 0,
            "start_ms": 500,
            "end_ms": 1600,
            "text": "你好",
            "source": "funasr_sentence",
        }
    ]


def test_auto_engine_with_auto_language_uses_whisper_language_detection(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")
    funasr_calls = []

    monkeypatch.setattr(asr, "extract_audio", lambda *_args, **_kwargs: audio_path)

    def fake_funasr(*_args, **_kwargs):
        funasr_calls.append(True)
        return [{"start_time": 0.0, "end_time": 1.0, "text": "错误中文路径"}]

    def fake_whisper(*_args, **_kwargs):
        return (
            [{"start_time": 0.0, "end_time": 1.0, "text": "hello world"}],
            {"task": "transcribe", "requested_language": "auto", "detected_language": "en"},
        )

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        return {
            "embeddings": np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "_funasr", fake_funasr)
    monkeypatch.setattr(asr, "_whisper", fake_whisper)
    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="auto",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        language="auto",
        semantic_model="fake-semantic",
    )

    assert funasr_calls == []
    assert result["engine"] == "whisper"
    assert result["detected_language"] == "en"


def test_faster_whisper_engine_uses_faster_whisper_path(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")
    calls = []

    monkeypatch.setattr(asr, "extract_audio", lambda *_args, **_kwargs: audio_path)

    def fake_faster_whisper(*_args, **_kwargs):
        calls.append(True)
        return (
            [{"start_time": 0.0, "end_time": 1.0, "text": "hello from turbo"}],
            {"task": "transcribe", "requested_language": "en", "detected_language": "en"},
        )

    def fail_whisper(*_args, **_kwargs):
        raise AssertionError("OpenAI Whisper should not be used for faster-whisper engine")

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        return {
            "embeddings": np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0], dtype=np.int32),
            "semantic_chunks": 1,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "_faster_whisper", fake_faster_whisper, raising=False)
    monkeypatch.setattr(asr, "_whisper", fail_whisper)
    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="faster-whisper",
        model_name="turbo",
        device="cpu",
        model_dir=str(tmp_path / "models" / "whisper"),
        faster_whisper_model_dir=str(tmp_path / "models" / "faster-whisper"),
        language="en",
        semantic_model="fake-semantic",
    )

    assert calls == [True]
    assert result["engine"] == "faster-whisper"
    assert result["model"] == "turbo"
    assert result["detected_language"] == "en"


def test_sidecar_asr_index_postprocesses_short_fragments_and_preserves_schema(tmp_path, monkeypatch):
    sidecar = tmp_path / "demo.srt"
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\n今天\n\n"
        "2\n00:00:00,800 --> 00:00:01,200\n我们聊一本书\n\n"
        "3\n00:00:03,200 --> 00:00:03,700\n下一段\n",
        encoding="utf-8",
    )

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        return {
            "embeddings": np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float16),
            "embedding_chunk_indices": np.asarray(
                [index for index, chunk in enumerate(chunks) if chunk.get("semantic_eligible", True)],
                dtype=np.int32,
            ),
            "semantic_chunks": len(chunks),
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="sidecar",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        sidecar_path=str(sidecar),
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "asr.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "chunk_times_ms",
            "texts",
            "chunk_emotions",
            "chunk_audio_events",
            "embeddings",
            "embedding_chunk_indices",
        }
        assert data["chunk_times_ms"].tolist() == [[0, 1200], [3200, 3700]]
        assert data["texts"].tolist() == ["今天我们聊一本书", "下一段"]
        assert data["embedding_chunk_indices"].tolist() == [0, 1]
    assert result["raw_chunks"] == 3
    assert result["chunks"] == 2
    assert result["postprocess_stats"]["merged_chunks"] == 1


def test_sidecar_asr_pipeline_does_not_repair_cjk_boundary_across_gap(tmp_path, monkeypatch):
    sidecar = tmp_path / "broken.srt"
    sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n孤\n\n"
        "2\n00:00:04,940 --> 00:00:06,000\n独敏感又倔强。\n",
        encoding="utf-8",
    )

    def fake_semantic_arrays(**kwargs):
        chunks = kwargs["chunks"]
        assert [chunk["text"] for chunk in chunks] == ["孤", "独敏感又倔强。"]
        return {
            "embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            "embedding_chunk_indices": np.asarray([0, 1], dtype=np.int32),
            "semantic_chunks": 2,
            "semantic_model": "fake-semantic",
            "semantic_device": "cpu",
            "semantic_status": "complete",
        }

    monkeypatch.setattr(asr, "build_text_semantic_arrays", fake_semantic_arrays, raising=False)

    result = build_asr_index(
        video_path=str(tmp_path / "video.mp4"),
        output_path=str(tmp_path / "asr.npz"),
        working_dir=str(tmp_path / "work"),
        engine="sidecar",
        model_name="small",
        device="cpu",
        model_dir=str(tmp_path / "models"),
        sidecar_path=str(sidecar),
        semantic_enabled=True,
        semantic_model="fake-semantic",
    )

    with np.load(tmp_path / "asr.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "chunk_times_ms",
            "texts",
            "chunk_emotions",
            "chunk_audio_events",
            "embeddings",
            "embedding_chunk_indices",
        }
        assert data["texts"].tolist() == ["孤", "独敏感又倔强。"]
    assert result["raw_items"] == 2
    assert result["retrieval_chunks"] == 2
    assert result["chunk_builder_stats"]["word_boundary_repairs"] == 0
    assert result["chunk_builder_stats"]["fake_gap_repairs"] == 0


def test_settings_default_asr_language_is_auto():
    from app.settings import Settings

    settings = Settings(app_data_dir="runtime-test")

    assert settings.asr_language == "auto"
    assert settings.asr_debug_artifacts is False
    assert settings.asr_save_raw_transcript is False
    assert settings.asr_vad_strategy == "silero_12s"
