import numpy as np

from app.indexing.speaker import _adaptive_turn_units, load_speaker_index, save_speaker_index


def test_speaker_index_uses_five_array_schema_and_builds_track_cache(tmp_path):
    output = tmp_path / "speaker.npz"
    result = save_speaker_index(
        output,
        utterance_times_ms=np.asarray([[0, 1000], [1200, 2600], [3000, 4500]]),
        utterance_embeddings=np.asarray([[1, 0, 0], [.9, .1, 0], [0, 1, 0]], dtype=np.float32),
        asr_chunk_indices=np.asarray([2, 3, 4]),
        auto_track_indices=np.asarray([0, 0, 1]),
    )

    data = load_speaker_index(output)

    assert set(data) == {
        "utterance_embeddings", "utterance_times_ms", "utterance_refs",
        "track_embeddings", "track_representative_indices",
    }
    assert data["utterance_refs"].tolist() == [[2, 0], [3, 0], [4, 1]]
    assert data["track_embeddings"].shape == (2, 3)
    assert data["track_representative_indices"].tolist() == [0, 2]
    assert result == {"utterances": 3, "tracks": 2, "embedding_dim": 3}


def test_speaker_index_represents_missing_track_as_minus_one(tmp_path):
    output = tmp_path / "speaker.npz"
    save_speaker_index(
        output,
        utterance_times_ms=np.asarray([[0, 500]]),
        utterance_embeddings=np.asarray([[1, 0]], dtype=np.float32),
        asr_chunk_indices=np.asarray([0]),
        auto_track_indices=np.asarray([-1]),
    )
    data = load_speaker_index(output)
    assert data["track_embeddings"].shape == (0, 2)
    assert data["track_representative_indices"].shape == (0,)


def test_adaptive_turns_follow_speaker_and_asr_boundaries():
    times, refs, tracks = _adaptive_turn_units(
        [[0.0, 1.5], [0.75, 2.25], [1.5, 3.0], [2.25, 3.75]],
        np.asarray([0, 0, 1, 1]),
        np.asarray([[0, 1800], [1800, 4000]], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )

    assert times.tolist() == [[0, 1800], [1875, 3750]]
    assert refs.tolist() == [0, 1]
    assert tracks.tolist() == [0, 1]
