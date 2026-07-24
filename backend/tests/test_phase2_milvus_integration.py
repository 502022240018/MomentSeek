"""Phase 2 unit tests with mocked Milvus client."""
from unittest.mock import MagicMock, Mock, patch
import numpy as np


def test_texts_from_milvus_success():
    """Test successful text retrieval from Milvus."""
    from app.speaker_service import _texts_from_milvus

    # Mock the Milvus client
    with patch(
        "app.speaker_service.ensure_milvus_reachable"
    ), patch("app.speaker_service.get_milvus_client") as mock_get_client:
        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection
        mock_get_client.return_value = mock_client

        # Simulate Milvus returning ASR texts
        mock_collection.query.return_value = [
            {"segment_idx": 0, "text": "Hello world"},
            {"segment_idx": 2, "text": "Third chunk"},
            {"segment_idx": 1, "text": "Second chunk"},
        ]

        result = _texts_from_milvus("test_video_id")

        # Should return sorted and dense list
        assert result == ["Hello world", "Second chunk", "Third chunk"]

        # Verify Milvus was called correctly
        mock_client.collection_for.assert_called_once_with("asr")
        mock_collection.query.assert_called_once()


def test_texts_from_milvus_empty():
    """Test Milvus returning no data."""
    from app.speaker_service import _texts_from_milvus

    with patch(
        "app.speaker_service.ensure_milvus_reachable"
    ), patch("app.speaker_service.get_milvus_client") as mock_get_client:
        mock_client = MagicMock()
        mock_collection = Mock()
        mock_client.collection_for.return_value = mock_collection
        mock_get_client.return_value = mock_client

        mock_collection.query.return_value = []

        result = _texts_from_milvus("test_video_id")

        assert result == []


def test_texts_from_milvus_connection_error():
    """Test Milvus connection failure returns empty list."""
    from app.speaker_service import _texts_from_milvus

    with patch(
        "app.speaker_service.ensure_milvus_reachable"
    ), patch("app.speaker_service.get_milvus_client") as mock_get_client:
        mock_get_client.side_effect = Exception("Connection failed")

        result = _texts_from_milvus("test_video_id")

        # Should return empty on error (caller will fall back to NPZ)
        assert result == []


def test_texts_with_milvus_read_enabled():
    """Test _texts() uses Milvus when read is enabled."""
    from app.speaker_service import _texts
    from pathlib import Path

    with patch('app.speaker_service.milvus_read_enabled') as mock_read_enabled, \
         patch('app.speaker_service._texts_from_milvus') as mock_from_milvus:

        mock_read_enabled.return_value = True
        mock_from_milvus.return_value = ["Milvus text 1", "Milvus text 2"]

        # NPZ should not be loaded
        result = _texts(Path("/fake/asr.npz"), "test_video_id")

        assert result == ["Milvus text 1", "Milvus text 2"]
        mock_from_milvus.assert_called_once_with("test_video_id")


def test_texts_falls_back_to_npz_when_milvus_empty():
    """Test _texts() falls back to NPZ when Milvus returns empty."""
    from app.speaker_service import _texts
    from pathlib import Path
    import tempfile

    # Create a temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        np.savez(tmp.name, texts=np.array(["NPZ text 1", "NPZ text 2"]))
        tmp_path = Path(tmp.name)

    try:
        with patch('app.speaker_service.milvus_read_enabled') as mock_read_enabled, \
             patch('app.speaker_service._texts_from_milvus') as mock_from_milvus:

            mock_read_enabled.return_value = True
            mock_from_milvus.return_value = []  # Empty Milvus result

            result = _texts(tmp_path, "test_video_id")

            # Should fall back to NPZ
            assert result == ["NPZ text 1", "NPZ text 2"]
    finally:
        tmp_path.unlink()


def test_texts_uses_npz_when_read_disabled():
    """Test _texts() uses NPZ when Milvus read is disabled."""
    from app.speaker_service import _texts
    from pathlib import Path
    import tempfile

    # Create a temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        np.savez(tmp.name, texts=np.array(["NPZ only"]))
        tmp_path = Path(tmp.name)

    try:
        with patch('app.speaker_service.milvus_read_enabled') as mock_read_enabled, \
             patch('app.speaker_service._texts_from_milvus') as mock_from_milvus:

            mock_read_enabled.return_value = False

            result = _texts(tmp_path, "test_video_id")

            # Should use NPZ directly
            assert result == ["NPZ only"]
            # Milvus should not be called
            mock_from_milvus.assert_not_called()
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
