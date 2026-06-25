import json

from app.db import Catalog
from app.search import Candidate, SearchEngine, _groups
from app.settings import Settings


def test_visual_adjacent_segments_remain_separate():
    candidates = [
        Candidate("video-1", 0, 5, 0.95, "visual"),
        Candidate("video-1", 5, 10, 0.94, "visual"),
        Candidate("video-1", 10, 15, 0.93, "visual"),
    ]

    groups = _groups(candidates, gap=2, max_duration=15)

    assert [(group[0].start_time, group[-1].end_time) for group in groups] == [(0, 5), (5, 10), (10, 15)]


def test_asr_adjacent_segments_can_merge():
    candidates = [
        Candidate("video-1", 10, 13, 1.0, "asr"),
        Candidate("video-1", 14, 17, 1.0, "asr"),
    ]

    groups = _groups(candidates, gap=2, max_duration=15)

    assert len(groups) == 1
    assert min(item.start_time for item in groups[0]) == 10
    assert max(item.end_time for item in groups[0]) == 17


def test_asr_search_returns_merged_playable_moment(tmp_path):
    settings = Settings(app_data_dir=tmp_path / "runtime", app_model_dir=tmp_path / "models")
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"not-needed-for-search")
    catalog.create_video({
        "id": "video-1", "name": "interview.mp4", "file_path": str(video_path),
        "duration": 60, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.update_video("video-1", indexed_modalities=["asr"])
    index_dir = settings.index_dir / "video-1"
    index_dir.mkdir(parents=True)
    (index_dir / "asr.json").write_text(json.dumps({
        "chunks": [
            {"start_time": 10, "end_time": 13, "text": "我们正在讨论电影投资"},
            {"start_time": 14, "end_time": 17, "text": "电影投资需要长期判断"},
            {"start_time": 40, "end_time": 42, "text": "今天天气很好"},
        ]
    }, ensure_ascii=False), encoding="utf-8")

    results = SearchEngine(settings, catalog).search("电影投资", None, ["asr"], ["video-1"])
    assert len(results) == 1
    assert results[0]["start_time"] == 10
    assert results[0]["end_time"] == 17
    assert results[0]["media_url"] == "/api/videos/video-1/media"
