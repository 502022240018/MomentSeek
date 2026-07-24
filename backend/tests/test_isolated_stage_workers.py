from types import SimpleNamespace

import pytest

import app.isolated_stage_workers as workers
from app.settings import get_settings


class _FakeConnection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.closed = False

    def recv(self):
        return next(self.messages)

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class _FakeModelPool:
    instances = []

    def __init__(self, idle_timeout):
        self.idle_timeout = idle_timeout
        self.closed = False
        self.__class__.instances.append(self)

    def keys(self):
        return ["warm-model"]

    def shutdown(self):
        self.closed = True


def test_isolated_worker_serves_one_stage_and_keeps_its_pool(monkeypatch):
    connection = _FakeConnection([
        {
            "operation": "run",
            "request_id": "request-1",
            "video": {"id": "video-1"},
            "options": {"visual_model": "siglip2"},
        },
        {"operation": "shutdown"},
    ])
    _FakeModelPool.instances.clear()
    monkeypatch.setattr(workers, "ModelPool", _FakeModelPool)
    monkeypatch.setattr(
        workers,
        "get_settings",
        lambda: SimpleNamespace(indexer_idle_timeout_seconds=0),
    )
    calls = []
    monkeypatch.setattr(
        workers,
        "_execute_stage",
        lambda stage, video, options, pool: calls.append((stage, video, options, pool)) or {"frames": 3},
    )

    workers.isolated_stage_worker_main("visual", connection)

    assert [item["type"] for item in connection.sent] == ["ready", "result"]
    assert connection.sent[1]["result"] == {"frames": 3}
    assert connection.sent[1]["warm"] == ["warm-model"]
    assert calls[0][:3] == ("visual", {"id": "video-1"}, {"visual_model": "siglip2"})
    assert _FakeModelPool.instances[0].closed is True
    assert connection.closed is True


def test_isolated_worker_exits_after_context_error(monkeypatch):
    connection = _FakeConnection([{
        "operation": "run",
        "request_id": "request-1",
        "video": {"id": "video-1"},
        "options": {},
    }])
    _FakeModelPool.instances.clear()
    monkeypatch.setattr(workers, "ModelPool", _FakeModelPool)
    monkeypatch.setattr(
        workers,
        "get_settings",
        lambda: SimpleNamespace(indexer_idle_timeout_seconds=0),
    )

    def fail(*_args):
        raise RuntimeError("stream is not in current ctx, error code is 107003")

    monkeypatch.setattr(workers, "_execute_stage", fail)

    workers.isolated_stage_worker_main("visual", connection)

    assert [item["type"] for item in connection.sent] == ["ready", "error"]
    assert connection.sent[1]["retryable"] is True
    assert "107003" in connection.sent[1]["message"]
    assert _FakeModelPool.instances[0].closed is True


def test_isolated_pool_reuses_worker_per_modality(monkeypatch):
    created = []

    class FakeWorker:
        def __init__(self, stage, start_timeout_seconds):
            self.stage = stage
            self.pid = 100 + len(created)
            self.alive = True
            self.stopped = False
            created.append(self)

        def run(self, video, options):
            return {"stage": self.stage, "video": video["id"]}

        def stop(self):
            self.alive = False
            self.stopped = True

    monkeypatch.setattr(workers, "IsolatedStageWorker", FakeWorker)
    pool = workers.IsolatedStageWorkerPool(max_attempts=2)

    first = pool.run_stage("visual", {"id": "v1"}, {})
    second = pool.run_stage("visual", {"id": "v2"}, {})
    face = pool.run_stage("face", {"id": "v3"}, {})

    assert len(created) == 2
    assert first["isolated_worker_attempts"] == 1
    assert second["video"] == "v2"
    assert face["stage"] == "face"
    assert set(pool.keys()) == {"visual:pid=100", "face:pid=101"}
    pool.shutdown()
    assert all(item.stopped for item in created)


def test_isolated_pool_restarts_once_for_context_error(monkeypatch):
    created = []

    class FakeWorker:
        def __init__(self, stage, start_timeout_seconds):
            self.stage = stage
            self.pid = 200 + len(created)
            self.alive = True
            created.append(self)

        def run(self, video, options):
            if len(created) == 1:
                raise workers.IsolatedWorkerError("107003", retryable=True)
            return {"frames": 9}

        def stop(self):
            self.alive = False

    monkeypatch.setattr(workers, "IsolatedStageWorker", FakeWorker)
    pool = workers.IsolatedStageWorkerPool(max_attempts=2)

    result = pool.run_stage("visual", {"id": "v1"}, {})

    assert result == {"frames": 9, "isolated_worker_attempts": 2}
    assert len(created) == 2
    pool.shutdown()


def test_isolated_pool_does_not_retry_application_error(monkeypatch):
    created = []

    class FakeWorker:
        def __init__(self, stage, start_timeout_seconds):
            self.stage = stage
            self.pid = 300
            self.alive = True
            created.append(self)

        def run(self, video, options):
            raise workers.IsolatedWorkerError("bad video", retryable=False, remote_traceback="remote failure")

        def stop(self):
            self.alive = False

    monkeypatch.setattr(workers, "IsolatedStageWorker", FakeWorker)
    pool = workers.IsolatedStageWorkerPool(max_attempts=2)

    with pytest.raises(RuntimeError, match="remote failure"):
        pool.run_stage("visual", {"id": "v1"}, {})

    assert len(created) == 1


def test_real_isolated_workers_have_distinct_processes(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("APP_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NPU_ENABLED", "false")
    get_settings.cache_clear()
    visual = workers.IsolatedStageWorker("visual", start_timeout_seconds=15)
    ocr = workers.IsolatedStageWorker("ocr", start_timeout_seconds=15)
    try:
        visual.start()
        ocr.start()
        assert visual.alive is True
        assert ocr.alive is True
        assert visual.pid is not None
        assert ocr.pid is not None
        assert visual.pid != ocr.pid
    finally:
        visual.stop()
        ocr.stop()
        get_settings.cache_clear()
