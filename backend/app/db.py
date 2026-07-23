from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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
"""


class Catalog:
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

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
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
            rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [self._decode_video(row) for row in rows]

    def get_video(self, video_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        return self._decode_video(row) if row else None

    def update_video(self, video_id: str, **values) -> None:
        allowed = {"name", "status", "indexed_modalities", "duration", "fps", "width", "height"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        if "indexed_modalities" in values:
            values["indexed_modalities"] = json.dumps(values["indexed_modalities"])
        clause = ",".join(f"{key}=?" for key in values)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE videos SET {clause},updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (*values.values(), video_id),
            )

    def delete_video(self, video_id: str) -> bool:
        # connect() does not enable PRAGMA foreign_keys, so the jobs cascade would
        # not fire; delete the video's jobs explicitly in the same transaction.
        with self.connect() as connection:
            connection.execute("DELETE FROM jobs WHERE video_id=?", (video_id,))
            cursor = connection.execute("DELETE FROM videos WHERE id=?", (video_id,))
            return cursor.rowcount > 0

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
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO entities(id,name,reference_path,embedding_path,face_embedding) VALUES(:id,:name,:reference_path,:embedding_path,:face_embedding)",
                record,
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

    def create_voice_sample(self, record: dict) -> dict:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO voice_samples(
                   id,entity_id,source_type,source_video_id,source_utterance_index,audio_path,embedding_path,embedding_space,voice_embedding
                   ) VALUES(:id,:entity_id,:source_type,:source_video_id,:source_utterance_index,:audio_path,:embedding_path,:embedding_space,:voice_embedding)""",
                record,
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
