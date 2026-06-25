import json

from app.db import Catalog
from app.search import SearchEngine
from app.settings import Settings


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

