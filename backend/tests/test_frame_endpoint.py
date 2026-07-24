from pathlib import Path

from fastapi.testclient import TestClient

from app.db import Catalog
from app.settings import Settings


def _client(tmp_path, monkeypatch):
    import app.main as main

    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "catalog", catalog)
    return main, settings, catalog, TestClient(main.app)


def _register_video(settings, catalog, duration=20.0):
    video_path = settings.upload_dir / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video")
    catalog.create_video({
        "id": "video-1",
        "name": "clip.mp4",
        "file_path": str(video_path.resolve()),
        "duration": duration,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "status": "ready",
    })
    return video_path


def test_frame_endpoint_extracts_and_caches(tmp_path, monkeypatch):
    main, settings, catalog, client = _client(tmp_path, monkeypatch)
    _register_video(settings, catalog)

    calls = []

    def fake_extract(video_path, output_path, ms, **_kwargs):
        calls.append(int(ms))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg marker
        return Path(output_path)

    monkeypatch.setattr(main, "extract_frame", fake_extract)

    response = client.get("/api/videos/video-1/frame", params={"ms": 5000})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "max-age=86400" in response.headers.get("cache-control", "")
    assert calls == [5000]

    # Second request for the same timestamp is served from disk cache, no re-extract.
    again = client.get("/api/videos/video-1/frame", params={"ms": 5000})
    assert again.status_code == 200
    assert calls == [5000]

    cached = settings.frame_cache_dir / "video-1" / f"{5000:012d}.jpg"
    assert cached.exists()


def test_frame_endpoint_clamps_ms_to_duration(tmp_path, monkeypatch):
    main, settings, catalog, client = _client(tmp_path, monkeypatch)
    _register_video(settings, catalog, duration=10.0)

    captured = []

    def fake_extract(video_path, output_path, ms, **_kwargs):
        captured.append(int(ms))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\xff\xd8\xff\xd9")
        return Path(output_path)

    monkeypatch.setattr(main, "extract_frame", fake_extract)

    response = client.get("/api/videos/video-1/frame", params={"ms": 999999})
    assert response.status_code == 200
    # Clamped to duration (10_000ms) - 1.
    assert captured == [9999]


def test_frame_endpoint_unknown_video_404(tmp_path, monkeypatch):
    main, settings, catalog, client = _client(tmp_path, monkeypatch)
    response = client.get("/api/videos/missing/frame", params={"ms": 0})
    assert response.status_code == 404


def test_frame_endpoint_rejects_negative_ms(tmp_path, monkeypatch):
    main, settings, catalog, client = _client(tmp_path, monkeypatch)
    _register_video(settings, catalog)
    response = client.get("/api/videos/video-1/frame", params={"ms": -1})
    assert response.status_code == 422
