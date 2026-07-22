import time

from app.db import Catalog
from app.indexer_daemon import execute_job
from app.indexer_daemon import indexer_singleton_lock
from app.model_pool import ModelPool
from app.settings import Settings


def test_pool_caches_by_key():
    pool = ModelPool(idle_timeout=999, reap_interval=999)
    try:
        calls = []
        factory = lambda: (calls.append(1), object())[1]
        a = pool.get("clip", factory)
        b = pool.get("clip", factory)
        assert a is b
        assert len(calls) == 1  # built once, reused
    finally:
        pool.shutdown()


def test_pool_separate_keys():
    pool = ModelPool(idle_timeout=999, reap_interval=999)
    try:
        a = pool.get("clip", lambda: "C")
        b = pool.get("face", lambda: "F")
        assert a == "C" and b == "F"
        assert set(pool.keys()) == {"clip", "face"}
    finally:
        pool.shutdown()


def test_pool_evicts_idle_and_frees_then_rebuilds():
    freed = []
    pool = ModelPool(idle_timeout=0.05, reap_interval=999, on_free=freed.append)
    try:
        pool.get("clip", lambda: "model-1")
        time.sleep(0.12)
        assert pool.evict_idle() == ["clip"]
        assert pool.keys() == []
        assert freed == ["model-1"]
        # a later request rebuilds a fresh instance
        assert pool.get("clip", lambda: "model-2") == "model-2"
    finally:
        pool.shutdown()


def test_pool_keeps_recently_used():
    pool = ModelPool(idle_timeout=10, reap_interval=999)
    try:
        pool.get("clip", lambda: "m")
        assert pool.evict_idle() == []  # used just now, not idle
        assert pool.keys() == ["clip"]
    finally:
        pool.shutdown()


def test_pool_non_positive_timeout_keeps_models_resident():
    freed = []
    pool = ModelPool(idle_timeout=0, reap_interval=999, on_free=freed.append)
    value = object()
    pool.get("face", lambda: value)

    assert pool.evict_idle() == []
    assert pool.keys() == ["face"]
    assert freed == []

    pool.shutdown()
    assert freed == [value]


def test_shutdown_frees_all():
    freed = []
    pool = ModelPool(idle_timeout=999, reap_interval=999, on_free=freed.append)
    pool.get("a", lambda: "x")
    pool.get("b", lambda: "y")
    pool.shutdown()
    assert set(freed) == {"x", "y"}
    assert pool.keys() == []


def test_indexer_singleton_lock_rejects_second_owner_and_releases(tmp_path):
    path = tmp_path / "indexer-daemon.lock"
    with indexer_singleton_lock(path) as first:
        with indexer_singleton_lock(path) as second:
            assert first is True
            assert second is False
    with indexer_singleton_lock(path) as acquired_after_release:
        assert acquired_after_release is True


def test_daemon_dispatches_stage_to_isolated_pool(tmp_path):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        npu_worker_mode="isolated",
    )
    settings.ensure_dirs()
    catalog = Catalog(settings.db_path)
    video_path = settings.upload_dir / "video.mp4"
    video_path.write_bytes(b"fake")
    catalog.create_video({
        "id": "video-1",
        "name": "demo.mp4",
        "file_path": str(video_path),
        "duration": 1.0,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "status": "uploaded",
    })
    catalog.create_job({
        "id": "job-1",
        "video_id": "video-1",
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "modalities": ["visual"],
        "options": {"visual_sample_fps": 1.0},
    })

    class FakeIsolatedPool:
        def __init__(self):
            self.calls = []

        def run_stage(self, stage, video, options):
            self.calls.append((stage, video["id"], options))
            return {"frames": 2, "isolated_worker_pid": 456}

    pool = FakeIsolatedPool()
    execute_job("job-1", settings, catalog, pool)

    job = catalog.get_job("job-1")
    assert pool.calls == [("visual", "video-1", {"visual_sample_fps": 1.0})]
    assert job["status"] == "completed"
    assert job["metrics"]["stages"]["visual"]["isolated_worker_pid"] == 456
