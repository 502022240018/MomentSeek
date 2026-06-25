import json

from app.indexing.asr import load_sidecar
from app.search import lexical_score


def test_load_json_and_srt(tmp_path):
    json_path = tmp_path / "demo.json"
    json_path.write_text(json.dumps({"segments": [{"start": 1, "end": 3, "text": "电影投资"}]}), encoding="utf-8")
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

