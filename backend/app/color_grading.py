from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from app.media import probe_video

ACTIVE_STATUSES = {"submitting", "queued", "running", "finalizing"}
TERMINAL_STATUSES = {"succeeded", "failed", "submission_unknown"}
UPSTREAM_STATUSES = {"queued", "running", "succeeded", "failed"}
MAX_REFERENCE_IMAGE_BYTES = 25 * 1024 * 1024


class ColorGradingError(RuntimeError):
    pass


class ColorGradingTransportError(ColorGradingError):
    pass


class ColorGradingHTTPError(ColorGradingError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ColorGradingProtocolError(ColorGradingError):
    pass


class ColorGradingClient:
    def __init__(self, base_url: str, timeout_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        return self._request(
            "GET",
            "/health",
            timeout_seconds=min(self.timeout_seconds, 2.0),
        )

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/tasks", payload)

    def get(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/tasks/{task_id}")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds or self.timeout_seconds,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            try:
                detail = json.loads(raw.decode("utf-8")).get("detail")
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                detail = None
            raise ColorGradingHTTPError(
                error.code,
                detail or f"仿色服务请求失败 ({error.code})",
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ColorGradingTransportError(f"无法连接仿色服务：{error}") from error
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ColorGradingProtocolError("仿色服务返回了无效 JSON") from error
        if not isinstance(result, dict):
            raise ColorGradingProtocolError("仿色服务响应必须是 JSON 对象")
        return result


class ColorGradingManager:
    _finalization_lock = threading.RLock()

    def __init__(self, settings, catalog):
        self.settings = settings
        self.catalog = catalog
        self.client = ColorGradingClient(
            settings.color_grading_base_url,
            settings.color_grading_request_timeout_seconds,
        )

    def capability(self) -> dict[str, Any]:
        if not self.settings.color_grading_enabled:
            return {
                "enabled": False,
                "available": False,
                "reason": "当前部署未启用视频仿色",
                "model_loaded": False,
                "database_connected": False,
                "device": None,
            }
        try:
            health = self.client.health()
        except ColorGradingError as error:
            return {
                "enabled": True,
                "available": False,
                "reason": str(error),
                "model_loaded": False,
                "database_connected": False,
                "device": None,
            }
        model_loaded = bool(health.get("model_loaded"))
        database_connected = bool(health.get("database_connected"))
        available = health.get("status") == "ok" and model_loaded and database_connected
        return {
            "enabled": True,
            "available": available,
            "reason": None if available else "仿色模型或任务数据库尚未就绪",
            "model_loaded": model_loaded,
            "database_connected": database_connected,
            "device": health.get("device"),
        }

    def validate_reference_image(self, path: Path) -> None:
        if path.stat().st_size > MAX_REFERENCE_IMAGE_BYTES:
            raise ValueError("参考图片不能超过 25 MB")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError("无法解析参考图片") from error

    def submit(self, task_id: str) -> dict:
        if not self.settings.color_grading_enabled:
            raise ColorGradingError("当前部署未启用视频仿色")
        task = self._require_task(task_id)
        input_video = self._require_video(task["input_video_id"])
        payload: dict[str, Any] = {
            "input_video": str(self._allowed_runtime_file(input_video["file_path"])),
        }
        if task["reference_type"] == "image":
            payload["ref_image"] = str(
                self._allowed_runtime_file(task["reference_image_path"])
            )
        else:
            reference = self._require_video(task["reference_video_id"])
            payload["ref_video"] = str(
                self._allowed_runtime_file(reference["file_path"])
            )
        try:
            response = self.client.submit(payload)
            external_task_id = str(response["task_id"])
            status = str(response["status"])
            if status not in UPSTREAM_STATUSES:
                raise ColorGradingProtocolError(f"未知的仿色任务状态：{status}")
            uuid.UUID(external_task_id)
        except ColorGradingHTTPError as error:
            self.catalog.update_color_grading_task(
                task_id,
                status="failed",
                stage="submitting",
                error_code=f"UpstreamHTTP{error.status_code}",
                error_message=error.detail,
            )
        except ColorGradingTransportError as error:
            # POST may have reached the upstream service before the connection
            # failed. Do not retry automatically and risk duplicate NPU work.
            self.catalog.update_color_grading_task(
                task_id,
                status="submission_unknown",
                stage="submitting",
                error_code="SubmissionUnknown",
                error_message=str(error),
            )
        except (KeyError, ValueError, ColorGradingProtocolError) as error:
            self.catalog.update_color_grading_task(
                task_id,
                status="failed",
                stage="submitting",
                error_code="UpstreamProtocolError",
                error_message=str(error),
            )
        else:
            self.catalog.update_color_grading_task(
                task_id,
                external_task_id=external_task_id,
                status=status,
                stage=status,
                upstream_status=status,
                error_code=None,
                error_message=None,
            )
        return self.decorate(self._require_task(task_id))

    def sync(self, task_id: str) -> dict:
        task = self._require_task(task_id)
        if not self.settings.color_grading_enabled:
            return self.decorate(task)
        if task["status"] in TERMINAL_STATUSES:
            return self.decorate(task)
        if task["status"] == "finalizing":
            self._finalize(task_id)
            return self.decorate(self._require_task(task_id))
        if not task["external_task_id"]:
            return self.decorate(task)
        try:
            response = self.client.get(task["external_task_id"])
            upstream_status = str(response["status"])
            if upstream_status not in UPSTREAM_STATUSES:
                raise ColorGradingProtocolError(
                    f"未知的仿色任务状态：{upstream_status}"
                )
        except ColorGradingError as error:
            self.catalog.update_color_grading_task(
                task_id,
                error_code="UpstreamUnavailable",
                error_message=str(error),
            )
            return self.decorate(self._require_task(task_id))

        common = {
            "upstream_status": upstream_status,
            "queue_position": response.get("queue_position"),
            "error_code": response.get("error_code"),
            "error_message": response.get("error_message"),
        }
        if upstream_status == "failed":
            self.catalog.update_color_grading_task(
                task_id,
                status="failed",
                stage="failed",
                **common,
            )
        elif upstream_status in {"queued", "running"}:
            self.catalog.update_color_grading_task(
                task_id,
                status=upstream_status,
                stage=upstream_status,
                **common,
            )
        else:
            output_video = response.get("output_video")
            output_lut = response.get("output_lut")
            self.catalog.update_color_grading_task(
                task_id,
                upstream_output_video=output_video,
                output_lut_path=output_lut,
                **common,
            )
            if self.catalog.claim_color_grading_finalization(task_id):
                self._finalize(task_id)
        return self.decorate(self._require_task(task_id))

    def list(self, *, refresh: bool = True) -> list[dict]:
        tasks = self.catalog.list_color_grading_tasks()
        if refresh and self.settings.color_grading_enabled:
            try:
                health = self.client.health()
            except ColorGradingError:
                return [self.decorate(task) for task in tasks]
            if (
                health.get("status") != "ok"
                or not health.get("model_loaded")
                or not health.get("database_connected")
            ):
                return [self.decorate(task) for task in tasks]
            refreshed = []
            for task in tasks:
                if task["status"] in ACTIVE_STATUSES:
                    refreshed.append(self.sync(task["id"]))
                else:
                    refreshed.append(self.decorate(task))
            return refreshed
        return [self.decorate(task) for task in tasks]

    def import_result(self, task_id: str) -> dict:
        task = self._require_task(task_id)
        if task["status"] != "succeeded":
            raise ColorGradingError("只有已完成的仿色结果可以加入素材库")
        if task["imported_video_id"]:
            existing = self.catalog.get_video(task["imported_video_id"])
            if existing:
                return existing
        source_path = self._allowed_result_file(task["final_video_path"])
        original = self._require_video(task["input_video_id"])
        video_id = uuid.uuid4().hex
        destination = self.settings.upload_dir / f"{video_id}.mp4"
        shutil.copy2(source_path, destination)
        try:
            info = probe_video(destination)
            created = self.catalog.create_video(
                {
                    "id": video_id,
                    "name": f"{Path(original['name']).stem}（仿色）.mp4",
                    "file_path": str(destination.resolve()),
                    "duration": info.duration,
                    "fps": info.fps,
                    "width": info.width,
                    "height": info.height,
                    "status": "uploaded",
                }
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self.catalog.update_color_grading_task(
            task_id,
            imported_video_id=video_id,
        )
        return created

    def media_path(self, task_id: str) -> Path:
        task = self._require_task(task_id)
        if task["status"] != "succeeded":
            raise ColorGradingError("仿色结果尚未生成")
        return self._allowed_result_file(task["final_video_path"])

    def lut_path(self, task_id: str) -> Path:
        task = self._require_task(task_id)
        if task["status"] != "succeeded":
            raise ColorGradingError("LUT 尚未生成")
        return self._allowed_upstream_file(task["output_lut_path"])

    def reference_path(self, task_id: str) -> Path:
        task = self._require_task(task_id)
        if task["reference_type"] != "image":
            raise ColorGradingError("该任务使用的是参考视频")
        return self._allowed_reference_file(task["reference_image_path"])

    def decorate(self, task: dict) -> dict:
        item = dict(task)
        for internal_path in (
            "reference_image_path",
            "upstream_output_video",
            "output_lut_path",
            "final_video_path",
        ):
            item.pop(internal_path, None)
        input_video = self.catalog.get_video(task["input_video_id"])
        reference_video = (
            self.catalog.get_video(task["reference_video_id"])
            if task["reference_video_id"]
            else None
        )
        item["input_video_name"] = (
            input_video["name"] if input_video else task["input_video_id"]
        )
        item["reference_video_name"] = (
            reference_video["name"] if reference_video else None
        )
        item["media_url"] = (
            f"/api/color-grading/tasks/{task['id']}/media"
            if task["status"] == "succeeded"
            else None
        )
        item["lut_url"] = (
            f"/api/color-grading/tasks/{task['id']}/lut"
            if task["status"] == "succeeded"
            else None
        )
        item["reference_url"] = (
            f"/api/color-grading/tasks/{task['id']}/reference"
            if task["reference_type"] == "image"
            else None
        )
        return item

    def _finalize(self, task_id: str) -> None:
        with self._finalization_lock:
            task = self._require_task(task_id)
            if task["status"] == "succeeded":
                return
            result_dir = self.settings.color_grading_result_dir / task_id
            result_dir.mkdir(parents=True, exist_ok=True)
            final_path = result_dir / "final.mp4"
            partial_path = result_dir / "final.partial.mp4"
            partial_path.unlink(missing_ok=True)
            try:
                upstream_video = self._allowed_upstream_file(
                    task["upstream_output_video"]
                )
                upstream_lut = self._allowed_upstream_file(task["output_lut_path"])
                input_video = self._require_video(task["input_video_id"])
                input_path = self._allowed_runtime_file(input_video["file_path"])
                original_has_audio = self._has_audio(input_path)
                if original_has_audio:
                    process = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(upstream_video),
                            "-i",
                            str(input_path),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0?",
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-movflags",
                            "+faststart",
                            "-shortest",
                            str(partial_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.returncode != 0:
                        raise RuntimeError(
                            process.stderr.strip() or "FFmpeg 合并音轨失败"
                        )
                else:
                    shutil.copy2(upstream_video, partial_path)
                source_info = probe_video(input_path)
                result_info = probe_video(partial_path)
                if result_info.width <= 0 or result_info.height <= 0:
                    raise RuntimeError("仿色结果没有有效视频流")
                tolerance = max(2.0, source_info.duration * 0.03)
                if (
                    source_info.duration > 0
                    and abs(result_info.duration - source_info.duration) > tolerance
                ):
                    raise RuntimeError("仿色结果时长与原视频差异过大")
                if original_has_audio and not self._has_audio(partial_path):
                    raise RuntimeError("仿色结果未保留原视频音轨")
                partial_path.replace(final_path)
                self.catalog.update_color_grading_task(
                    task_id,
                    status="succeeded",
                    stage="completed",
                    final_video_path=str(final_path.resolve()),
                    output_lut_path=str(upstream_lut),
                    error_code=None,
                    error_message=None,
                )
            except Exception as error:  # noqa: BLE001 - persist every task failure
                partial_path.unlink(missing_ok=True)
                self.catalog.update_color_grading_task(
                    task_id,
                    status="failed",
                    stage="finalizing",
                    error_code=type(error).__name__,
                    error_message=str(error),
                )

    @staticmethod
    def _has_audio(path: Path) -> bool:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or f"无法探测音轨：{path}")
        try:
            return bool(json.loads(process.stdout).get("streams"))
        except json.JSONDecodeError as error:
            raise RuntimeError("ffprobe 返回了无效 JSON") from error

    def _require_task(self, task_id: str) -> dict:
        task = self.catalog.get_color_grading_task(task_id)
        if not task:
            raise KeyError("仿色任务不存在")
        return task

    def _require_video(self, video_id: str | None) -> dict:
        video = self.catalog.get_video(video_id or "")
        if not video:
            raise KeyError("视频不存在")
        return video

    def _allowed_runtime_file(self, value: str | Path | None) -> Path:
        return self._allowed_file(value, self.settings.app_data_dir)

    def _allowed_reference_file(self, value: str | Path | None) -> Path:
        return self._allowed_file(
            value,
            self.settings.color_grading_reference_dir,
        )

    def _allowed_upstream_file(self, value: str | Path | None) -> Path:
        return self._allowed_file(
            value,
            self.settings.color_grading_upstream_dir,
        )

    def _allowed_result_file(self, value: str | Path | None) -> Path:
        return self._allowed_file(
            value,
            self.settings.color_grading_result_dir,
        )

    @staticmethod
    def _allowed_file(value: str | Path | None, root: Path) -> Path:
        if not value:
            raise ValueError("缺少文件路径")
        path = Path(value).resolve(strict=True)
        root = Path(root).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"文件不在允许目录中：{path}") from error
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
        return path
