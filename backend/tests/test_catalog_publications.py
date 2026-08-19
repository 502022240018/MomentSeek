from __future__ import annotations

import json

import pytest

from app.catalog.db import Catalog


def _catalog_with_video(tmp_path) -> Catalog:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video({
        "id": "video-1",
        "name": "Example",
        "file_path": "uploads/example.mp4",
        "duration": 12.5,
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "status": "uploaded",
    })
    return catalog


def test_publish_modality_is_atomic_with_index_visibility(tmp_path):
    catalog = _catalog_with_video(tmp_path)

    publication = catalog.publish_modality(
        "video-1",
        "visual",
        asset_version="attempt-a",
        row_count=25,
        metadata={
            "model_key": "siglip2-so400m-patch14-384",
            "embedding_space": "siglip2-image-text",
            "sample_fps": 2.0,
        },
    )

    assert publication["asset_version"] == "attempt-a"
    assert publication["row_count"] == 25
    assert publication["model_key"] == "siglip2-so400m-patch14-384"
    video = catalog.get_video("video-1")
    assert video["indexed_modalities"] == ["visual"]
    assert video["index_publications"]["visual"]["sample_fps"] == 2.0


def test_publications_are_independent_per_modality(tmp_path):
    catalog = _catalog_with_video(tmp_path)
    catalog.publish_modality(
        "video-1", "visual", asset_version="visual-a", row_count=2,
        metadata={"model_key": "visual-model"},
    )
    catalog.publish_modality(
        "video-1", "asr", asset_version="asr-a", row_count=0,
        metadata={"model_key": "asr-model", "semantic_status": "empty"},
    )
    catalog.publish_modality(
        "video-1", "visual", asset_version="visual-b", row_count=3,
        metadata={"model_key": "visual-model-v2"},
    )

    video = catalog.get_video("video-1")
    assert video["indexed_modalities"] == ["asr", "visual"]
    assert video["index_publications"]["visual"]["asset_version"] == "visual-b"
    assert video["index_publications"]["asr"]["asset_version"] == "asr-a"
    assert video["index_publications"]["asr"]["row_count"] == 0


def test_publish_modality_rejects_invalid_or_missing_targets(tmp_path):
    catalog = _catalog_with_video(tmp_path)

    with pytest.raises(ValueError, match="row_count"):
        catalog.publish_modality(
            "video-1", "visual", asset_version="bad", row_count=-1,
        )
    with pytest.raises(KeyError, match="视频不存在"):
        catalog.publish_modality(
            "missing", "visual", asset_version="attempt", row_count=1,
        )

    assert catalog.list_modality_publications() == []


def test_delete_video_removes_publications(tmp_path):
    catalog = _catalog_with_video(tmp_path)
    catalog.publish_modality(
        "video-1", "speaker", asset_version="speaker-a", row_count=4,
    )

    assert catalog.delete_video("video-1") is True
    assert catalog.list_modality_publications() == []


def test_publish_modalities_switches_asr_and_speaker_together(tmp_path):
    catalog = _catalog_with_video(tmp_path)
    catalog.publish_modalities(
        "video-1",
        [
            {
                "modality": "asr",
                "asset_version": "generation-2",
                "row_count": 3,
                "metadata": {"retrieval_chunks": 3},
            },
            {
                "modality": "speaker",
                "asset_version": "generation-2",
                "row_count": 2,
                "metadata": {
                    "utterances": 2,
                    "source_asr_asset_version": "generation-2",
                },
            },
        ],
    )

    video = catalog.get_video("video-1")
    assert video["indexed_modalities"] == ["asr", "speaker"]
    assert video["index_publications"]["asr"]["asset_version"] == "generation-2"
    assert video["index_publications"]["speaker"]["asset_version"] == "generation-2"


def test_online_modalities_are_derived_from_publications_and_repair_stored_drift(tmp_path):
    catalog = _catalog_with_video(tmp_path)
    catalog.publish_modality(
        "video-1", "visual", asset_version="visual-a", row_count=1,
    )
    with catalog.connect() as connection:
        connection.execute(
            "UPDATE videos SET indexed_modalities=? WHERE id=?",
            (json.dumps(["asr"]), "video-1"),
        )

    # The compatibility column is stale, but online reads trust ready
    # publication rows and therefore expose only visual.
    assert catalog.get_video("video-1")["indexed_modalities"] == ["visual"]
    with catalog.connect() as connection:
        stored = connection.execute(
            "SELECT indexed_modalities FROM videos WHERE id='video-1'"
        ).fetchone()["indexed_modalities"]
    assert json.loads(stored) == ["asr"]

    # Any later publication recomputes and repairs the stored compatibility
    # flag from the authoritative publication rows in the same transaction.
    catalog.publish_modality(
        "video-1", "face", asset_version="face-a", row_count=0,
    )
    with catalog.connect() as connection:
        repaired = connection.execute(
            "SELECT indexed_modalities FROM videos WHERE id='video-1'"
        ).fetchone()["indexed_modalities"]
    assert json.loads(repaired) == ["face", "visual"]
