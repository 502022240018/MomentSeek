import importlib.util
import json
from pathlib import Path


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "asr_eval_report.py"
    spec = importlib.util.spec_from_file_location("asr_eval_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_summary(name: str, total_s: float, audio_s: float, cer: float, recall2: float):
    return {
        "name": name,
        "timing": {
            "total_seconds": total_s,
            "decode_total_seconds": total_s - 5,
        },
        "speed": {
            "audio_seconds": audio_s,
            "x_total": audio_s / total_s,
        },
        "aggregate": {
            "sample_count": 2,
            "failed_samples": 0,
            "audio_seconds": audio_s,
            "global_cer": {"cer": cer},
            "window30_cer": {"cer": cer + 0.1},
            "local": {"2000": {"avg_recall": recall2, "avg_f1": 0.33}},
        },
    }


def _sample_summary(elapsed_s: float, cer: float = 0.2, recall2: float = 0.8):
    return {
        "elapsed_seconds": elapsed_s,
        "metrics": {
            "global_cer": {"cer": cer},
            "local": {"2000": {"avg_recall": recall2}},
        },
    }


def test_build_markdown_report_reads_overall_metrics(tmp_path):
    module = _load_report_module()
    eval_dir = tmp_path / "eval"
    _write_json(
        eval_dir / "summary.json",
        {
            "setup": {"samples": []},
            "runs": [
                _run_summary("faster_whisper_small_builtin_vad_zh", 100.0, 3600.0, 0.25, 0.70),
                _run_summary("faster_whisper_turbo_builtin_vad_zh", 80.0, 3600.0, 0.20, 0.80),
            ],
        },
    )

    markdown = module.build_markdown(eval_dir)

    assert "# ASR Evaluation Report" in markdown
    assert "| faster_whisper_small_builtin_vad_zh | 2 | 1.000 | 100.1 | 36.00 | 0.250 | 0.350 | 0.700 | 0.330 | 0 |" not in markdown
    assert "| faster_whisper_small_builtin_vad_zh | 2 | 1.000 | 100.0 | 36.00 | 0.250 | 0.350 | 0.700 | 0.330 | 0 |" in markdown
    assert "| faster_whisper_turbo_builtin_vad_zh | 2 | 1.000 | 80.0 | 45.00 | 0.200 | 0.300 | 0.800 | 0.330 | 0 |" in markdown


def test_build_markdown_report_compares_sample_runtime_to_baseline(tmp_path):
    module = _load_report_module()
    eval_dir = tmp_path / "rerun"
    baseline_dir = tmp_path / "full"
    sample_id = "asr_v1_wenet_meeting_002"
    run_name = "faster_whisper_turbo_builtin_vad_zh"

    _write_json(eval_dir / "summary.json", {"setup": {"samples": [{"sample_id": sample_id}]}, "runs": [_run_summary(run_name, 90.0, 1800.0, 0.18, 0.84)]})
    _write_json(baseline_dir / "summary.json", {"setup": {"samples": [{"sample_id": sample_id}]}, "runs": [_run_summary(run_name, 200.0, 1800.0, 0.19, 0.82)]})
    _write_json(eval_dir / "runs" / run_name / "samples" / sample_id / "summary.json", _sample_summary(149.2, 0.176, 0.84))
    _write_json(baseline_dir / "runs" / run_name / "samples" / sample_id / "summary.json", _sample_summary(419.6, 0.176, 0.84))

    markdown = module.build_markdown(eval_dir, baseline_dir=baseline_dir)

    assert "## Sample Runtime Comparison" in markdown
    assert "| asr_v1_wenet_meeting_002 | faster_whisper_turbo_builtin_vad_zh | 149.2 | 419.6 | -64.4 | 0.176 | 0.840 |" in markdown
