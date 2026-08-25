import numpy as np

from app.catalog.db import Catalog
from app.identity.face_gallery import (
    cluster_face_tracks,
    face_group_arrays,
    face_group_model_version,
    select_major_face_groups,
)


def _unit(*values: float) -> np.ndarray:
    vector = np.zeros(512, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


def test_cluster_face_tracks_merges_identity_and_orders_by_importance():
    alice = _unit(1.0, 0.0, 0.0)
    alice_profile = _unit(0.94, 0.20, 0.0)
    bob = _unit(0.0, 1.0, 0.0)
    groups = cluster_face_tracks(
        np.stack([alice, bob, alice_profile]),
        np.asarray([[0, 4000, 1000], [5000, 6000, 5200], [8000, 13000, 9000]]),
        qualities=np.asarray([0.7, 0.9, 0.95]),
        bboxes=np.asarray([[.1, .1, .4, .5], [.2, .1, .5, .6], [.12, .1, .45, .55]]),
        detection_counts=np.asarray([5, 2, 7]),
        cosine_threshold=0.52,
    )

    assert len(groups) == 2
    assert groups[0].track_indices == (0, 2)
    assert groups[0].representative_track_idx == 2
    assert groups[0].duration_ms == 9000
    assert groups[0].occurrence_count == 12
    assert groups[1].track_indices == (1,)


def test_face_group_arrays_are_shape_stable_for_empty_and_nonempty():
    assert face_group_arrays([])["group_embeddings"].shape == (0, 512)
    group = cluster_face_tracks(
        np.stack([_unit(1.0, 0.0)]),
        np.asarray([[100, 900, 500]]),
    )
    arrays = face_group_arrays(group)
    assert arrays["group_embeddings"].shape == (1, 512)
    assert arrays["group_times_ms"].tolist() == [[100, 900, 500]]


def test_importance_score_remains_ordered_beyond_old_saturation_point():
    identity = _unit(1.0, 0.0)
    groups = cluster_face_tracks(
        np.stack([identity, identity]),
        np.asarray([[0, 31_000, 1_000], [50_000, 150_000, 60_000]]),
        qualities=np.asarray([0.5, 0.5]),
        detection_counts=np.asarray([21, 80]),
        # Deliberately keep the two observations separate while exercising the
        # long-duration ranking formula.
        cosine_threshold=1.0,
    )
    # Exact equality still merges at threshold=1; compare independent people.
    if len(groups) == 1:
        second = _unit(0.0, 1.0)
        groups = cluster_face_tracks(
            np.stack([identity, second]),
            np.asarray([[0, 31_000, 1_000], [50_000, 150_000, 60_000]]),
            qualities=np.asarray([0.5, 0.5]),
            detection_counts=np.asarray([21, 80]),
            cosine_threshold=0.52,
        )
    assert groups[0].duration_ms == 100_000
    assert groups[0].importance_score > groups[1].importance_score


def test_select_major_face_groups_filters_noise_and_caps_results():
    vectors = np.stack([_unit(1.0, 0.0), _unit(0.0, 1.0), _unit(0.0, 0.0, 1.0)])
    groups = cluster_face_tracks(
        vectors,
        np.asarray([[0, 6_000, 1_000], [7_000, 7_500, 7_100], [8_000, 8_800, 8_100]]),
        detection_counts=np.asarray([1, 5, 1]),
    )
    selected = select_major_face_groups(
        groups,
        limit=2,
        min_duration_ms=3_000,
        min_occurrence_count=3,
    )
    assert len(selected) == 2
    assert {item.duration_ms for item in selected} == {6_000, 500}


def test_face_group_model_version_is_deterministic_and_threshold_specific():
    assert face_group_model_version(0.52) == "major-people-v2:cosine=0.520"
    assert face_group_model_version(0.46) != face_group_model_version(0.52)


def test_face_identity_binding_tracks_published_asset_version(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.create_video({
        "id": "video-1", "name": "demo.mp4", "file_path": str(tmp_path / "demo.mp4"),
        "duration": 10, "fps": 25, "width": 1280, "height": 720, "status": "ready",
    })
    catalog.create_entity({
        "id": "person-1", "name": "Alice", "reference_path": "", "embedding_path": None,
    })
    catalog.bind_face_identity("video-1", "face-v1", "groups-v1", 3, "person-1")
    catalog.bind_face_identity("video-1", "face-v2", "groups-v2", 3, "person-1")

    bindings = catalog.face_identity_bindings("video-1", "face-v2", "groups-v2")
    assert bindings[3]["entity_name"] == "Alice"
    assert catalog.delete_entity("person-1") is True
    assert catalog.face_identity_bindings("video-1", "face-v2", "groups-v2") == {}
