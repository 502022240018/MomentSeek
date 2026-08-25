from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  duration REAL NOT NULL DEFAULT 0,
  fps REAL NOT NULL DEFAULT 0,
  width INTEGER NOT NULL DEFAULT 0,
  height INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'uploaded',
  indexed_modalities TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS video_modality_publications (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  modality TEXT NOT NULL,
  asset_version TEXT NOT NULL,
  row_count INTEGER NOT NULL DEFAULT 0,
  model_key TEXT,
  semantic_model_key TEXT,
  embedding_space TEXT,
  status TEXT NOT NULL DEFAULT 'ready',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  published_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, modality)
);
CREATE INDEX IF NOT EXISTS video_modality_publications_status_idx
  ON video_modality_publications(status, modality, video_id);
CREATE TABLE IF NOT EXISTS folders (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS video_folders (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  folder_id TEXT NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, folder_id)
);
CREATE INDEX IF NOT EXISTS video_folders_folder_video_idx ON video_folders(folder_id, video_id);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'queued',
  stage TEXT NOT NULL DEFAULT 'queued',
  progress REAL NOT NULL DEFAULT 0,
  modalities TEXT NOT NULL DEFAULT '[]',
  options TEXT NOT NULL DEFAULT '{}',
  metrics TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  worker_pid INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  reference_path TEXT NOT NULL,
  embedding_path TEXT,
  face_embedding BLOB,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS video_speakers (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  track_id INTEGER NOT NULL,
  display_name TEXT,
  representative_utterance_index INTEGER,
  hidden INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(video_id, track_id)
);
CREATE TABLE IF NOT EXISTS utterance_overrides (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  utterance_index INTEGER NOT NULL,
  corrected_track_id INTEGER,
  searchable INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY(video_id, utterance_index)
);
CREATE TABLE IF NOT EXISTS speaker_identity_bindings (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  track_id INTEGER NOT NULL,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, track_id)
);
CREATE TABLE IF NOT EXISTS face_identity_bindings (
  video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  asset_version TEXT NOT NULL,
  group_version TEXT NOT NULL,
  group_idx INTEGER NOT NULL,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(video_id, asset_version, group_version, group_idx)
);
CREATE TABLE IF NOT EXISTS voice_samples (
  id TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL,
  source_video_id TEXT,
  source_utterance_index INTEGER,
  audio_path TEXT,
  embedding_path TEXT NOT NULL,
  embedding_space TEXT NOT NULL,
  voice_embedding BLOB,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS color_grading_tasks (
  id TEXT PRIMARY KEY,
  external_task_id TEXT UNIQUE,
  input_video_id TEXT NOT NULL,
  reference_type TEXT NOT NULL,
  reference_video_id TEXT,
  reference_image_path TEXT,
  ncc INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'submitting',
  stage TEXT NOT NULL DEFAULT 'submitting',
  upstream_status TEXT,
  queue_position INTEGER,
  upstream_output_video TEXT,
  output_lut_path TEXT,
  final_video_path TEXT,
  imported_video_id TEXT,
  error_code TEXT,
  error_message TEXT,
  started_at TEXT,
  completed_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS color_grading_tasks_status_idx
  ON color_grading_tasks(status, created_at);
CREATE INDEX IF NOT EXISTS color_grading_tasks_input_video_idx
  ON color_grading_tasks(input_video_id);
CREATE INDEX IF NOT EXISTS color_grading_tasks_reference_video_idx
  ON color_grading_tasks(reference_video_id);
CREATE TABLE IF NOT EXISTS milvus_cleanup_queue (
  video_id TEXT PRIMARY KEY,
  last_error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Catalog:
    DEFAULT_FOLDER_ID = "__default__"
    DEFAULT_FOLDER_NAME = "默认文件夹"
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_columns(connection)

    @staticmethod
    def _ensure_columns(connection: sqlite3.Connection) -> None:
        # Ensure jobs.metrics column
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
        if "metrics" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN metrics TEXT NOT NULL DEFAULT '{}'")

        # Ensure entities.face_embedding column (Phase 3)
        entity_columns = {row["name"] for row in connection.execute("PRAGMA table_info(entities)").fetchall()}
        if "face_embedding" not in entity_columns:
            connection.execute("ALTER TABLE entities ADD COLUMN face_embedding BLOB")

        # Ensure voice_samples.voice_embedding column (Phase 3)
        voice_columns = {row["name"] for row in connection.execute("PRAGMA table_info(voice_samples)").fetchall()}
        if "voice_embedding" not in voice_columns:
            connection.execute("ALTER TABLE voice_samples ADD COLUMN voice_embedding BLOB")

        face_binding_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(face_identity_bindings)"
            ).fetchall()
        }
        if "group_version" not in face_binding_columns:
            # Group indices are generation-local. Preserve legacy bindings as
            # unversioned audit rows, but never apply them to a new generation.
            connection.execute(
                "ALTER TABLE face_identity_bindings "
                "RENAME TO face_identity_bindings_legacy"
            )
            connection.execute(
                """CREATE TABLE face_identity_bindings (
                   video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                   asset_version TEXT NOT NULL,
                   group_version TEXT NOT NULL,
                   group_idx INTEGER NOT NULL,
                   entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                   PRIMARY KEY(video_id, asset_version, group_version, group_idx)
                   )"""
            )
            connection.execute(
                """INSERT INTO face_identity_bindings(
                   video_id,asset_version,group_version,group_idx,entity_id,created_at
                   ) SELECT video_id,asset_version,'',group_idx,entity_id,created_at
                   FROM face_identity_bindings_legacy"""
            )
            connection.execute("DROP TABLE face_identity_bindings_legacy")

        grading_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(color_grading_tasks)"
            ).fetchall()
        }
        if "ncc" not in grading_columns:
            connection.execute(
                "ALTER TABLE color_grading_tasks "
                "ADD COLUMN ncc INTEGER NOT NULL DEFAULT 0"
            )
        if "started_at" not in grading_columns:
            connection.execute(
                "ALTER TABLE color_grading_tasks ADD COLUMN started_at TEXT"
            )
        if "completed_at" not in grading_columns:
            connection.execute(
                "ALTER TABLE color_grading_tasks ADD COLUMN completed_at TEXT"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create_video(self, record: dict) -> dict:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO videos(id,name,file_path,duration,fps,width,height,status)
                   VALUES(:id,:name,:file_path,:duration,:fps,:width,:height,:status)""",
                record,
            )
        return self.get_video(record["id"])

    def list_videos(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC, id ASC").fetchall()
            videos = [self._decode_video(row) for row in rows]
            self._attach_video_folders(connection, videos)
            self._attach_video_publications(connection, videos)
        return videos

    def get_video(self, video_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
            if not row:
                return None
            video = self._decode_video(row)
            self._attach_video_folders(connection, [video])
            self._attach_video_publications(connection, [video])
        return video

    def publish_modality(
        self,
        video_id: str,
        modality: str,
        *,
        asset_version: str,
        row_count: int,
        metadata: dict | None = None,
        status: str = "ready",
    ) -> dict:
        """Publish one verified Milvus version through the atomic batch API."""
        publications = self.publish_modalities(
            video_id,
            [
                {
                    "modality": modality,
                    "asset_version": asset_version,
                    "row_count": row_count,
                    "metadata": metadata,
                    "status": status,
                }
            ],
        )
        return publications[str(modality).strip().casefold()]

    def publish_modalities(
        self,
        video_id: str,
        publications: list[dict],
        *,
        disable_modalities: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, dict]:
        """Atomically switch one or more verified Milvus publication pointers.

        ``disable_modalities`` changes existing rows to ``disabled`` without
        fabricating an asset version when a channel has never been published.
        The compatibility ``videos.indexed_modalities`` column is recomputed
        from ready publication rows in the same transaction.
        """
        allowed_modalities = {"visual", "face", "asr", "speaker", "ocr"}
        prepared: list[dict] = []
        seen: set[str] = set()
        for publication in publications:
            item = dict(publication)
            modality = str(item.get("modality") or "").strip().casefold()
            if modality not in allowed_modalities:
                raise ValueError(f"未知索引模态: {modality}")
            if modality in seen:
                raise ValueError(f"同一批次不能重复发布模态: {modality}")
            seen.add(modality)
            asset_version = str(item.get("asset_version") or "").strip()
            if not asset_version:
                raise ValueError("asset_version 不能为空")
            row_count = int(item.get("row_count"))
            if row_count < 0:
                raise ValueError("row_count 不能为负数")
            status = str(item.get("status") or "ready").strip().casefold()
            if status not in {"ready", "disabled"}:
                raise ValueError(f"不支持的发布状态: {status}")
            payload = dict(item.get("metadata") or {})
            prepared.append(
                {
                    "modality": modality,
                    "asset_version": asset_version,
                    "row_count": row_count,
                    "model_key": payload.get("model_key"),
                    "semantic_model_key": payload.get("semantic_model_key"),
                    "embedding_space": payload.get("embedding_space"),
                    "status": status,
                    "metadata_json": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )

        disabled = {
            str(modality).strip().casefold()
            for modality in (disable_modalities or ())
        }
        unknown_disabled = disabled - allowed_modalities
        if unknown_disabled:
            raise ValueError(
                f"未知索引模态: {', '.join(sorted(unknown_disabled))}"
            )
        overlap = seen & disabled
        if overlap:
            raise ValueError(
                f"同一批次不能同时发布并禁用模态: {', '.join(sorted(overlap))}"
            )
        if not prepared and not disabled:
            raise ValueError("至少发布或禁用一个模态")

        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            video_row = connection.execute(
                "SELECT 1 FROM videos WHERE id=?", (video_id,)
            ).fetchone()
            if not video_row:
                raise KeyError(f"视频不存在: {video_id}")
            for item in prepared:
                connection.execute(
                    """INSERT INTO video_modality_publications(
                           video_id,modality,asset_version,row_count,model_key,
                           semantic_model_key,embedding_space,status,metadata_json
                       ) VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(video_id,modality) DO UPDATE SET
                         asset_version=excluded.asset_version,
                         row_count=excluded.row_count,
                         model_key=excluded.model_key,
                         semantic_model_key=excluded.semantic_model_key,
                         embedding_space=excluded.embedding_space,
                         status=excluded.status,
                         metadata_json=excluded.metadata_json,
                         published_at=CURRENT_TIMESTAMP,
                         updated_at=CURRENT_TIMESTAMP""",
                    (
                        video_id,
                        item["modality"],
                        item["asset_version"],
                        item["row_count"],
                        str(item["model_key"])
                        if item["model_key"] is not None
                        else None,
                        str(item["semantic_model_key"])
                        if item["semantic_model_key"] is not None
                        else None,
                        str(item["embedding_space"])
                        if item["embedding_space"] is not None
                        else None,
                        item["status"],
                        item["metadata_json"],
                    ),
                )
            if disabled:
                marks = ",".join("?" for _ in disabled)
                connection.execute(
                    f"""UPDATE video_modality_publications
                        SET status='disabled',updated_at=CURRENT_TIMESTAMP
                        WHERE video_id=? AND modality IN ({marks})""",
                    (video_id, *sorted(disabled)),
                )
            ready_rows = connection.execute(
                """SELECT modality FROM video_modality_publications
                   WHERE video_id=? AND status='ready' ORDER BY modality""",
                (video_id,),
            ).fetchall()
            indexed = [str(row["modality"]) for row in ready_rows]
            connection.execute(
                """UPDATE videos SET indexed_modalities=?,updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (json.dumps(indexed), video_id),
            )
            affected = sorted(seen | disabled)
            marks = ",".join("?" for _ in affected)
            rows = connection.execute(
                f"""SELECT * FROM video_modality_publications
                    WHERE video_id=? AND modality IN ({marks})""",
                (video_id, *affected),
            ).fetchall()
        return {
            item["modality"]: item
            for row in rows
            if (item := self._decode_publication(row))
        }

    def get_modality_publication(self, video_id: str, modality: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM video_modality_publications
                   WHERE video_id=? AND modality=?""",
                (video_id, modality),
            ).fetchone()
        return self._decode_publication(row) if row else None

    def list_modality_publications(
        self,
        video_ids: list[str] | None = None,
    ) -> list[dict]:
        with self.connect() as connection:
            if video_ids is None:
                rows = connection.execute(
                    """SELECT * FROM video_modality_publications
                       ORDER BY video_id,modality"""
                ).fetchall()
            elif not video_ids:
                rows = []
            else:
                unique_ids = list(dict.fromkeys(video_ids))
                marks = ",".join("?" for _ in unique_ids)
                rows = connection.execute(
                    f"""SELECT * FROM video_modality_publications
                        WHERE video_id IN ({marks}) ORDER BY video_id,modality""",
                    unique_ids,
                ).fetchall()
        return [self._decode_publication(row) for row in rows]

    def list_folders(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT f.*, COUNT(vf.video_id) AS video_count
                   FROM folders f LEFT JOIN video_folders vf ON vf.folder_id=f.id
                   GROUP BY f.id ORDER BY f.created_at DESC, f.id DESC"""
            ).fetchall()
            default_count = connection.execute(
                """SELECT COUNT(*) AS count FROM videos v WHERE NOT EXISTS
                   (SELECT 1 FROM video_folders vf WHERE vf.video_id=v.id)"""
            ).fetchone()["count"]
        return [{"id": self.DEFAULT_FOLDER_ID, "name": self.DEFAULT_FOLDER_NAME,
                 "kind": "default", "video_count": default_count},
                *[{**dict(row), "kind": "user"} for row in rows]]

    def create_folder(self, name: str) -> dict:
        folder_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("INSERT INTO folders(id,name) VALUES(?,?)", (folder_id, name))
        return self.get_folder(folder_id)

    def get_folder(self, folder_id: str) -> dict | None:
        if folder_id == self.DEFAULT_FOLDER_ID:
            return self.list_folders()[0]
        with self.connect() as connection:
            row = connection.execute(
                """SELECT f.*, COUNT(vf.video_id) AS video_count FROM folders f
                   LEFT JOIN video_folders vf ON vf.folder_id=f.id WHERE f.id=? GROUP BY f.id""", (folder_id,)
            ).fetchone()
        return {**dict(row), "kind": "user"} if row else None

    def rename_folder(self, folder_id: str, name: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("UPDATE folders SET name=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (name, folder_id))
        return cursor.rowcount > 0

    def delete_folder(self, folder_id: str) -> int | None:
        with self.connect() as connection:
            if not connection.execute("SELECT 1 FROM folders WHERE id=?", (folder_id,)).fetchone():
                return None
            count = connection.execute("SELECT COUNT(*) AS count FROM video_folders WHERE folder_id=?", (folder_id,)).fetchone()["count"]
            connection.execute("DELETE FROM video_folders WHERE folder_id=?", (folder_id,))
            connection.execute("DELETE FROM folders WHERE id=?", (folder_id,))
        return count

    def update_video_folders(self, video_ids: list[str], folder_ids: list[str], operation: str) -> None:
        video_ids, folder_ids = list(dict.fromkeys(video_ids)), list(dict.fromkeys(folder_ids))
        if not video_ids:
            raise ValueError("请至少选择一个视频")
        if any(folder_id == self.DEFAULT_FOLDER_ID for folder_id in folder_ids):
            raise ValueError("默认文件夹无需手动加入")
        with self.connect() as connection:
            marks = ",".join("?" for _ in video_ids)
            found = connection.execute(f"SELECT COUNT(*) AS count FROM videos WHERE id IN ({marks})", video_ids).fetchone()["count"]
            if found != len(video_ids):
                raise ValueError("包含不存在的视频")
            if folder_ids:
                folder_marks = ",".join("?" for _ in folder_ids)
                found = connection.execute(f"SELECT COUNT(*) AS count FROM folders WHERE id IN ({folder_marks})", folder_ids).fetchone()["count"]
                if found != len(folder_ids):
                    raise ValueError("包含不存在的文件夹")
            if operation == "replace":
                connection.execute(f"DELETE FROM video_folders WHERE video_id IN ({marks})", video_ids)
            elif operation == "remove":
                if folder_ids:
                    folder_marks = ",".join("?" for _ in folder_ids)
                    connection.execute(f"DELETE FROM video_folders WHERE video_id IN ({marks}) AND folder_id IN ({folder_marks})", [*video_ids, *folder_ids])
                return
            if operation not in {"add", "replace"}:
                raise ValueError("不支持的文件夹操作")
            connection.executemany("INSERT OR IGNORE INTO video_folders(video_id,folder_id) VALUES(?,?)", [(video_id, folder_id) for video_id in video_ids for folder_id in folder_ids])

    def resolve_video_scope(self, video_ids: list[str] | None, folder_ids: list[str] | None) -> list[str] | None:
        if video_ids is None and folder_ids is None:
            return None
        selected = list(dict.fromkeys(video_ids or []))
        requested = list(dict.fromkeys(folder_ids or []))
        with self.connect() as connection:
            user_ids = [item for item in requested if item != self.DEFAULT_FOLDER_ID]
            if user_ids:
                marks = ",".join("?" for _ in user_ids)
                found = connection.execute(f"SELECT COUNT(*) AS count FROM folders WHERE id IN ({marks})", user_ids).fetchone()["count"]
                if found != len(user_ids):
                    raise ValueError("包含不存在的文件夹")
                selected.extend(row["video_id"] for row in connection.execute(f"SELECT video_id FROM video_folders WHERE folder_id IN ({marks})", user_ids).fetchall())
            if self.DEFAULT_FOLDER_ID in requested:
                selected.extend(row["id"] for row in connection.execute("SELECT v.id FROM videos v WHERE NOT EXISTS (SELECT 1 FROM video_folders vf WHERE vf.video_id=v.id)").fetchall())
        return list(dict.fromkeys(selected))

    def update_video(self, video_id: str, **values) -> None:
        if "indexed_modalities" in values:
            raise ValueError(
                "indexed_modalities is publication-controlled; use publish_modality()"
            )
        allowed = {"name", "status", "duration", "fps", "width", "height"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        clause = ",".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE videos SET {clause},updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (*values.values(), video_id),
            )

    def delete_video(self, video_id: str) -> bool:
        # Keep explicit cleanup for databases created by older builds where
        # foreign-key enforcement may not have been enabled on every connection.
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE video_id=?", (video_id,))
            connection.execute("DELETE FROM video_folders WHERE video_id=?", (video_id,))
            connection.execute(
                "DELETE FROM video_modality_publications WHERE video_id=?",
                (video_id,),
            )
            cursor = connection.execute("DELETE FROM videos WHERE id=?", (video_id,))
            return cursor.rowcount > 0

    def create_color_grading_task(self, record: dict) -> dict:
        payload = {
            "id": record["id"],
            "external_task_id": record.get("external_task_id"),
            "input_video_id": record["input_video_id"],
            "reference_type": record["reference_type"],
            "reference_video_id": record.get("reference_video_id"),
            "reference_image_path": record.get("reference_image_path"),
            "ncc": int(bool(record.get("ncc", False))),
            "status": record.get("status", "submitting"),
            "stage": record.get("stage", "submitting"),
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
        }
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO color_grading_tasks(
                   id,external_task_id,input_video_id,reference_type,
                   reference_video_id,reference_image_path,ncc,status,stage,
                   started_at,completed_at
                   ) VALUES(
                   :id,:external_task_id,:input_video_id,:reference_type,
                   :reference_video_id,:reference_image_path,:ncc,:status,:stage,
                   :started_at,:completed_at
                   )""",
                payload,
            )
        return self.get_color_grading_task(record["id"])

    def get_color_grading_task(self, task_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM color_grading_tasks WHERE id=?",
                (task_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_color_grading_tasks(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM color_grading_tasks ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def update_color_grading_task(self, task_id: str, **values) -> None:
        allowed = {
            "external_task_id",
            "status",
            "stage",
            "upstream_status",
            "queue_position",
            "upstream_output_video",
            "output_lut_path",
            "final_video_path",
            "imported_video_id",
            "error_code",
            "error_message",
            "started_at",
            "completed_at",
        }
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        clause = ",".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"""UPDATE color_grading_tasks
                    SET {clause},updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (*values.values(), task_id),
            )

    def claim_color_grading_finalization(self, task_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE color_grading_tasks
                   SET status='finalizing',stage='finalizing',
                       error_code=NULL,error_message=NULL,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status IN ('queued','running')""",
                (task_id,),
            )
        return cursor.rowcount == 1

    def recover_color_grading_finalizations(self) -> int:
        """Make interrupted local mux operations eligible for retry."""
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE color_grading_tasks
                   SET status='running',
                       stage='awaiting_result',
                       updated_at=CURRENT_TIMESTAMP
                   WHERE status='finalizing'"""
            )
        return cursor.rowcount

    def has_active_color_grading_tasks(self, video_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT 1 FROM color_grading_tasks
                   WHERE (input_video_id=? OR reference_video_id=?)
                     AND status IN ('submitting','queued','running','finalizing')
                   LIMIT 1""",
                (video_id, video_id),
            ).fetchone()
        return row is not None

    def enqueue_milvus_cleanup(self, video_id: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO milvus_cleanup_queue(video_id,last_error,attempts)
                   VALUES(?,?,1)
                   ON CONFLICT(video_id) DO UPDATE SET
                     last_error=excluded.last_error,
                     attempts=milvus_cleanup_queue.attempts+1,
                     updated_at=CURRENT_TIMESTAMP""",
                (video_id, error),
            )

    def list_milvus_cleanup_queue(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM milvus_cleanup_queue ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_milvus_cleanup(self, video_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM milvus_cleanup_queue WHERE video_id=?",
                (video_id,),
            )

    def create_job(self, record: dict) -> dict:
        payload = dict(record)
        payload["modalities"] = json.dumps(payload.get("modalities", []))
        payload["options"] = json.dumps(payload.get("options", {}), ensure_ascii=False)
        payload["metrics"] = json.dumps(payload.get("metrics", {}), ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO jobs(id,video_id,status,stage,progress,modalities,options,metrics)
                   VALUES(:id,:video_id,:status,:stage,:progress,:modalities,:options,:metrics)""",
                payload,
            )
        return self.get_job(record["id"])

    def get_job(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._decode_job(row) if row else None

    def list_jobs(self, video_id: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM jobs", ()
        if video_id:
            query, args = query + " WHERE video_id=?", (video_id,)
        query += " ORDER BY created_at DESC"
        with self.connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return [self._decode_job(row) for row in rows]

    def next_queued_job(self) -> dict | None:
        """Return the oldest queued job without loading the full job history."""
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM jobs WHERE status='queued'
                   ORDER BY created_at ASC, id ASC LIMIT 1"""
            ).fetchone()
        return self._decode_job(row) if row else None

    def update_job(self, job_id: str, **values) -> None:
        allowed = {"status", "stage", "progress", "error", "worker_pid", "metrics"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        if "metrics" in values:
            values["metrics"] = json.dumps(values["metrics"], ensure_ascii=False)
        clause = ",".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {clause},updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (*values.values(), job_id),
            )

    def claim_queued_job(self, job_id: str, *, worker_pid: int | None = None) -> bool:
        """Atomically transition one queued job to running.

        Cancellation and queue consumers can race across processes. The status
        predicate ensures a cancelled job can never be revived by a stale worker.
        """
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET status='running',stage='starting',progress=0.01,
                   error=NULL,worker_pid=?,updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status='queued'""",
                (worker_pid, job_id),
            )
        return cursor.rowcount == 1

    def create_entity(self, record: dict) -> dict:
        values = {**record, "face_embedding": record.get("face_embedding")}
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO entities(id,name,reference_path,embedding_path,face_embedding) VALUES(:id,:name,:reference_path,:embedding_path,:face_embedding)",
                values,
            )
        return self.get_entity(record["id"])

    def update_entity_embedding(self, entity_id: str, embedding_path: str, face_embedding: bytes | None = None) -> None:
        with self.connect() as connection:
            if face_embedding is not None:
                connection.execute(
                    "UPDATE entities SET embedding_path=?, face_embedding=? WHERE id=?",
                    (embedding_path, face_embedding, entity_id)
                )
            else:
                connection.execute("UPDATE entities SET embedding_path=? WHERE id=?", (embedding_path, entity_id))

    # Binary columns that must never be sent to API clients as raw bytes —
    # they are stored as BLOBs for internal use only.  Strip them from any
    # dict returned to callers so FastAPI / Pydantic do not attempt UTF-8
    # serialisation of the raw float32 buffers.
    _ENTITY_BLOB_FIELDS     = frozenset({"face_embedding"})
    _VOICE_SAMPLE_BLOB_FIELDS = frozenset({"voice_embedding"})

    @staticmethod
    def _strip(row: dict, fields: frozenset) -> dict:
        return {k: v for k, v in row.items() if k not in fields}

    def list_entities(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT e.*, COUNT(v.id) AS voice_sample_count FROM entities e
                   LEFT JOIN voice_samples v ON v.entity_id=e.id GROUP BY e.id ORDER BY e.name"""
            ).fetchall()
        return [self._strip(dict(row), self._ENTITY_BLOB_FIELDS) for row in rows]

    def get_entity(self, entity_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        return self._strip(dict(row), self._ENTITY_BLOB_FIELDS) if row else None

    def rename_entity(self, entity_id: str, name: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("UPDATE entities SET name=? WHERE id=?", (name, entity_id))
            return cursor.rowcount > 0

    def delete_entity(self, entity_id: str) -> bool:
        # Foreign keys are not enabled on every legacy connection, so delete
        # dependent mutable records explicitly.
        with self.connect() as connection:
            connection.execute("DELETE FROM speaker_identity_bindings WHERE entity_id=?", (entity_id,))
            connection.execute("DELETE FROM face_identity_bindings WHERE entity_id=?", (entity_id,))
            connection.execute("DELETE FROM voice_samples WHERE entity_id=?", (entity_id,))
            cursor = connection.execute("DELETE FROM entities WHERE id=?", (entity_id,))
            return cursor.rowcount > 0

    def find_entity_in_text(self, text: str) -> dict | None:
        lowered = text.casefold()
        matches = [entity for entity in self.list_entities() if entity["name"].casefold() in lowered]
        return max(matches, key=lambda item: len(item["name"]), default=None)

    def speaker_overlays(self, video_id: str) -> dict:
        with self.connect() as connection:
            speakers = connection.execute(
                "SELECT * FROM video_speakers WHERE video_id=?", (video_id,)
            ).fetchall()
            utterances = connection.execute(
                "SELECT * FROM utterance_overrides WHERE video_id=?", (video_id,)
            ).fetchall()
            bindings = connection.execute(
                "SELECT * FROM speaker_identity_bindings WHERE video_id=?", (video_id,)
            ).fetchall()
        return {
            "speakers": {int(row["track_id"]): dict(row) for row in speakers},
            "utterances": {int(row["utterance_index"]): dict(row) for row in utterances},
            "bindings": {int(row["track_id"]): dict(row) for row in bindings},
        }

    def upsert_video_speaker(self, video_id: str, track_id: int, **values) -> None:
        display_name = values.get("display_name")
        representative = values.get("representative_utterance_index")
        hidden = 1 if values.get("hidden", False) else 0
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO video_speakers(video_id,track_id,display_name,representative_utterance_index,hidden)
                   VALUES(?,?,?,?,?) ON CONFLICT(video_id,track_id) DO UPDATE SET
                   display_name=COALESCE(excluded.display_name,video_speakers.display_name),
                   representative_utterance_index=COALESCE(excluded.representative_utterance_index,video_speakers.representative_utterance_index),
                   hidden=excluded.hidden""",
                (video_id, track_id, display_name, representative, hidden),
            )

    def upsert_utterance_override(
        self, video_id: str, utterance_index: int, corrected_track_id: int | None, searchable: bool = True
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO utterance_overrides(video_id,utterance_index,corrected_track_id,searchable)
                   VALUES(?,?,?,?) ON CONFLICT(video_id,utterance_index) DO UPDATE SET
                   corrected_track_id=excluded.corrected_track_id,searchable=excluded.searchable""",
                (video_id, utterance_index, corrected_track_id, 1 if searchable else 0),
            )

    def bind_speaker_identity(self, video_id: str, track_id: int, entity_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO speaker_identity_bindings(video_id,track_id,entity_id) VALUES(?,?,?)
                   ON CONFLICT(video_id,track_id) DO UPDATE SET entity_id=excluded.entity_id""",
                (video_id, track_id, entity_id),
            )

    def face_identity_bindings(
        self,
        video_id: str,
        asset_version: str,
        group_version: str,
    ) -> dict[int, dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT b.group_idx,b.entity_id,e.name AS entity_name
                   FROM face_identity_bindings b JOIN entities e ON e.id=b.entity_id
                   WHERE b.video_id=? AND b.asset_version=? AND b.group_version=?""",
                (video_id, asset_version, group_version),
            ).fetchall()
        return {int(row["group_idx"]): dict(row) for row in rows}

    def bind_face_identity(
        self,
        video_id: str,
        asset_version: str,
        group_version: str,
        group_idx: int,
        entity_id: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO face_identity_bindings(
                   video_id,asset_version,group_version,group_idx,entity_id
                   ) VALUES(?,?,?,?,?)
                   ON CONFLICT(video_id,asset_version,group_version,group_idx)
                   DO UPDATE SET entity_id=excluded.entity_id""",
                (video_id, asset_version, group_version, group_idx, entity_id),
            )

    def create_voice_sample(self, record: dict) -> dict:
        values = {**record, "voice_embedding": record.get("voice_embedding")}
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO voice_samples(
                   id,entity_id,source_type,source_video_id,source_utterance_index,audio_path,embedding_path,embedding_space,voice_embedding
                   ) VALUES(:id,:entity_id,:source_type,:source_video_id,:source_utterance_index,:audio_path,:embedding_path,:embedding_space,:voice_embedding)""",
                values,
            )
        return self.get_voice_sample(record["id"])

    def get_voice_sample(self, sample_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM voice_samples WHERE id=?", (sample_id,)).fetchone()
        return self._strip(dict(row), self._VOICE_SAMPLE_BLOB_FIELDS) if row else None

    def list_voice_samples(self, entity_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_samples WHERE entity_id=? ORDER BY created_at", (entity_id,)
            ).fetchall()
        return [self._strip(dict(row), self._VOICE_SAMPLE_BLOB_FIELDS) for row in rows]

    def list_voice_sample_embeddings(self, entity_id: str) -> list[dict]:
        """Return the minimum private payload needed by trusted voice services.

        Public entity APIs continue to use ``list_voice_samples`` and never see
        the raw BLOB.  Keeping this as a separate, explicitly named method makes
        accidental serialization much harder than adding an ``include_blob``
        switch to the public-shaped result.
        """
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT id,voice_embedding FROM voice_samples
                   WHERE entity_id=? ORDER BY created_at,id""",
                (entity_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _attach_video_folders(connection: sqlite3.Connection, videos: list[dict]) -> None:
        if not videos:
            return
        ids = [video["id"] for video in videos]
        marks = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"""SELECT vf.video_id, f.id, f.name FROM video_folders vf
                JOIN folders f ON f.id=vf.folder_id WHERE vf.video_id IN ({marks})
                ORDER BY f.name COLLATE NOCASE""", ids
        ).fetchall()
        grouped: dict[str, list[dict]] = {video_id: [] for video_id in ids}
        for row in rows:
            grouped[row["video_id"]].append({"id": row["id"], "name": row["name"]})
        for video in videos:
            video["folders"] = grouped[video["id"]]
            video["folder_ids"] = [folder["id"] for folder in video["folders"]]

    @classmethod
    def _attach_video_publications(
        cls,
        connection: sqlite3.Connection,
        videos: list[dict],
    ) -> None:
        if not videos:
            return
        ids = [video["id"] for video in videos]
        marks = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"""SELECT * FROM video_modality_publications
                WHERE video_id IN ({marks}) ORDER BY video_id,modality""",
            ids,
        ).fetchall()
        grouped: dict[str, dict[str, dict]] = {video_id: {} for video_id in ids}
        for row in rows:
            publication = cls._decode_publication(row)
            grouped[publication["video_id"]][publication["modality"]] = publication
        for video in videos:
            publications = grouped[video["id"]]
            video["index_publications"] = publications
            # Publications are the online source of truth.  The stored JSON
            # column remains transactionally maintained for old binaries and
            # direct SQL diagnostics, but a stale compatibility flag must not
            # change what retrieval can see.
            video["indexed_modalities"] = sorted(
                modality
                for modality, publication in publications.items()
                if publication.get("status") == "ready"
            )

    @staticmethod
    def _decode_publication(row: sqlite3.Row) -> dict:
        item = dict(row)
        metadata = json.loads(item.pop("metadata_json") or "{}")
        item["metadata"] = metadata
        # Expose metadata keys at the publication boundary so callers can use
        # one compact mapping without coupling themselves to SQLite storage.
        for key, value in metadata.items():
            item.setdefault(key, value)
        return item

    @staticmethod
    def _decode_video(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["indexed_modalities"] = json.loads(item["indexed_modalities"] or "[]")
        return item

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["modalities"] = json.loads(item["modalities"] or "[]")
        item["options"] = json.loads(item["options"] or "{}")
        item["metrics"] = json.loads(item.get("metrics") or "{}")
        return item
