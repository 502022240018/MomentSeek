import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

from app.color_grading import ColorGradingManager
from app.db import Catalog
from app.settings import Settings
from fastapi.testclient import TestClient


class FakeClient:
    def __init__(self, *, submit_response=None, task_response=None):
        self.submit_response = submit_response
        self.task_response = task_response
        self.submitted = None

    def health(self):
        return {
            "status": "ok",
            "model_loaded": True,
            "database_connected": True,
            "device": "npu:0",
        }

    def submit(self, payload):
        self.submitted = payload
        return self.submit_response

    def get(self, task_id):
        return self.task_response


def _settings(tmp_path, *, enabled=True):
    settings = Settings(
        _env_file=None,
        app_data_dir=tmp_path / "runtime",
        app_model_dir=tmp_path / "models",
        color_grading_enabled=enabled,
    )
    settings.ensure_dirs()
    return settings


def _video(catalog, settings, video_id, name):
    path = settings.upload_dir / f"{video_id}.mp4"
    path.write_bytes(b"video")
    return catalog.create_video(
        {
            "id": video_id,
            "name": name,
            "file_path": str(path.resolve()),
            "duration": 10,
            "fps": 25,
            "width": 1280,
            "height": 720,
            "status": "uploaded",
        }
    )


def test_submit_translates_catalog_ids_to_shared_runtime_paths(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    source = _video(catalog, settings, "source", "source.mp4")
    reference = _video(catalog, settings, "reference", "reference.mp4")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "input_video_id": source["id"],
            "reference_type": "video",
            "reference_video_id": reference["id"],
        }
    )
    manager = ColorGradingManager(settings, catalog)
    manager.client = FakeClient(
        submit_response={
            "task_id": "26905f42-555c-43a5-ae28-a09bfe5fb792",
            "status": "queued",
            "created_at": "2026-07-27T08:00:00Z",
        }
    )

    task = manager.submit("local-task")

    assert task["status"] == "queued"
    assert task["external_task_id"] == "26905f42-555c-43a5-ae28-a09bfe5fb792"
    assert manager.client.submitted == {
        "input_video": source["file_path"],
        "ref_video": reference["file_path"],
        "ncc": False,
    }


def test_submit_forwards_enabled_ncc_to_upstream(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    source = _video(catalog, settings, "source", "source.mp4")
    reference = _video(catalog, settings, "reference", "reference.mp4")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "input_video_id": source["id"],
            "reference_type": "video",
            "reference_video_id": reference["id"],
            "ncc": True,
        }
    )
    manager = ColorGradingManager(settings, catalog)
    manager.client = FakeClient(
        submit_response={
            "task_id": "26905f42-555c-43a5-ae28-a09bfe5fb792",
            "status": "queued",
            "created_at": "2026-07-27T08:00:00Z",
        }
    )

    task = manager.submit("local-task")

    assert task["ncc"] is True
    assert manager.client.submitted["ncc"] is True


def test_catalog_migrates_color_grading_option_and_runtime_columns(tmp_path):
    database = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE color_grading_tasks (
               id TEXT PRIMARY KEY,
               input_video_id TEXT NOT NULL,
               reference_video_id TEXT,
               status TEXT NOT NULL,
               created_at TEXT NOT NULL
            )"""
        )

    catalog = Catalog(database)

    with catalog.connect() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(color_grading_tasks)"
            ).fetchall()
        }
    assert {"ncc", "started_at", "completed_at"} <= columns


def test_successful_upstream_task_is_finalized_into_platform_result(
    monkeypatch,
    tmp_path,
):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    source = _video(catalog, settings, "source", "source.mp4")
    external_id = "26905f42-555c-43a5-ae28-a09bfe5fb792"
    upstream_dir = settings.color_grading_upstream_dir / external_id
    upstream_dir.mkdir(parents=True)
    upstream_video = upstream_dir / "graded.mp4"
    upstream_lut = upstream_dir / "grading.cube"
    upstream_video.write_bytes(b"graded-video")
    upstream_lut.write_text("LUT_3D_SIZE 16", encoding="utf-8")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "external_task_id": external_id,
            "input_video_id": source["id"],
            "reference_type": "image",
            "reference_image_path": str(
                settings.color_grading_reference_dir / "unused.jpg"
            ),
            "status": "running",
            "stage": "running",
        }
    )
    manager = ColorGradingManager(settings, catalog)
    manager.client = FakeClient(
        task_response={
            "task_id": external_id,
            "status": "succeeded",
            "queue_position": None,
            "output_video": str(upstream_video.resolve()),
            "output_lut": str(upstream_lut.resolve()),
            "error_code": None,
            "error_message": None,
            "started_at": "2026-07-27T08:01:00Z",
            "completed_at": "2026-07-27T08:04:30Z",
        }
    )
    monkeypatch.setattr(manager, "_has_audio", lambda path: False)
    monkeypatch.setattr(
        "app.color_grading.probe_video",
        lambda path: SimpleNamespace(
            duration=10,
            fps=25,
            width=1280,
            height=720,
        ),
    )

    task = manager.sync("local-task")

    stored = catalog.get_color_grading_task("local-task")
    final_path = Path(stored["final_video_path"])
    assert task["status"] == "succeeded"
    assert final_path.read_bytes() == b"graded-video"
    assert task["media_url"].endswith("/media")
    assert task["lut_url"].endswith("/lut")
    assert task["started_at"] == "2026-07-27T08:01:00Z"
    assert task["completed_at"] == "2026-07-27T08:04:30Z"
    assert "final_video_path" not in task
    assert "upstream_output_video" not in task


def test_upstream_result_outside_shared_output_root_is_rejected(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    source = _video(catalog, settings, "source", "source.mp4")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    lut = settings.color_grading_upstream_dir / "grading.cube"
    lut.write_text("lut", encoding="utf-8")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "external_task_id": "26905f42-555c-43a5-ae28-a09bfe5fb792",
            "input_video_id": source["id"],
            "reference_type": "image",
            "reference_image_path": str(
                settings.color_grading_reference_dir / "unused.jpg"
            ),
            "status": "running",
            "stage": "running",
        }
    )
    manager = ColorGradingManager(settings, catalog)
    manager.client = FakeClient(
        task_response={
            "status": "succeeded",
            "queue_position": None,
            "output_video": str(outside),
            "output_lut": str(lut),
            "error_code": None,
            "error_message": None,
        }
    )

    task = manager.sync("local-task")

    assert task["status"] == "failed"
    assert "不在允许目录" in task["error_message"]


def test_finalization_preserves_source_audio_when_ffmpeg_is_available(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    source_path = settings.upload_dir / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.4",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(source_path),
        ],
        check=True,
    )
    catalog.create_video(
        {
            "id": "source",
            "name": "source.mp4",
            "file_path": str(source_path.resolve()),
            "duration": 1,
            "fps": 10,
            "width": 160,
            "height": 90,
            "status": "uploaded",
        }
    )
    external_id = "26905f42-555c-43a5-ae28-a09bfe5fb792"
    upstream_dir = settings.color_grading_upstream_dir / external_id
    upstream_dir.mkdir(parents=True)
    upstream_video = upstream_dir / "graded.mp4"
    upstream_lut = upstream_dir / "grading.cube"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=1",
            "-c:v",
            "mpeg4",
            "-an",
            str(upstream_video),
        ],
        check=True,
    )
    upstream_lut.write_text("LUT_3D_SIZE 16", encoding="utf-8")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "external_task_id": external_id,
            "input_video_id": "source",
            "reference_type": "image",
            "reference_image_path": str(
                settings.color_grading_reference_dir / "unused.jpg"
            ),
            "status": "running",
            "stage": "running",
        }
    )
    manager = ColorGradingManager(settings, catalog)
    manager.client = FakeClient(
        task_response={
            "status": "succeeded",
            "queue_position": None,
            "output_video": str(upstream_video.resolve()),
            "output_lut": str(upstream_lut.resolve()),
            "error_code": None,
            "error_message": None,
        }
    )

    task = manager.sync("local-task")

    final_path = Path(catalog.get_color_grading_task("local-task")["final_video_path"])
    assert task["status"] == "succeeded"
    assert manager._has_audio(final_path) is True


def test_disabled_capability_does_not_contact_upstream(tmp_path):
    settings = _settings(tmp_path, enabled=False)
    manager = ColorGradingManager(settings, Catalog(settings.db_path))

    assert manager.capability() == {
        "enabled": False,
        "available": False,
        "reason": "当前部署未启用视频仿色",
        "model_loaded": False,
        "database_connected": False,
        "device": None,
    }


def test_active_task_prevents_source_or_reference_video_deletion(tmp_path):
    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    _video(catalog, settings, "source", "source.mp4")
    _video(catalog, settings, "reference", "reference.mp4")
    catalog.create_color_grading_task(
        {
            "id": "local-task",
            "input_video_id": "source",
            "reference_type": "video",
            "reference_video_id": "reference",
            "status": "queued",
            "stage": "queued",
        }
    )

    assert catalog.has_active_color_grading_tasks("source") is True
    assert catalog.has_active_color_grading_tasks("reference") is True
    catalog.update_color_grading_task(
        "local-task",
        status="succeeded",
        stage="completed",
    )
    assert catalog.has_active_color_grading_tasks("source") is False


def test_video_reference_task_endpoint_persists_platform_mapping(
    monkeypatch,
    tmp_path,
):
    from app import main

    settings = _settings(tmp_path)
    catalog = Catalog(settings.db_path)
    _video(catalog, settings, "source", "source.mp4")
    _video(catalog, settings, "reference", "reference.mp4")

    class FakeManager:
        @staticmethod
        def capability():
            return {
                "enabled": True,
                "available": True,
                "reason": None,
            }

        @staticmethod
        def submit(task_id):
            catalog.update_color_grading_task(
                task_id,
                external_task_id="26905f42-555c-43a5-ae28-a09bfe5fb792",
                status="queued",
                stage="queued",
                upstream_status="queued",
            )
            return catalog.get_color_grading_task(task_id)

    monkeypatch.setattr(main, "settings", settings)
    monkeypatch.setattr(main, "catalog", catalog)
    monkeypatch.setattr(main, "_color_grading_manager", FakeManager)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/color-grading/tasks",
            data={
                "input_video_id": "source",
                "reference_type": "video",
                "ref_video_id": "reference",
                "ncc": "true",
            },
        )

    assert response.status_code == 202
    task = response.json()
    assert task["status"] == "queued"
    stored = catalog.get_color_grading_task(task["id"])
    assert stored["input_video_id"] == "source"
    assert stored["reference_video_id"] == "reference"
    assert stored["ncc"] == 1
