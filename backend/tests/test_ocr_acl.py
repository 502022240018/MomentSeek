from pathlib import Path

import pytest

from app.indexing.ocr_acl import _choose_covering_shape, _shape_from_name


def test_shape_from_om_filename():
    path = Path("PP-OCRv6_det_small-1x3x736x1312.om")
    assert _shape_from_name(path) == (1, 3, 736, 1312)


def test_choose_smallest_covering_shape():
    shapes = {
        (1, 3, 736, 1312): Path("small.om"),
        (1, 3, 736, 1760): Path("wide.om"),
    }
    shape, path = _choose_covering_shape(shapes, (1, 3, 736, 1280))
    assert shape == (1, 3, 736, 1312)
    assert path == Path("small.om")


def test_choose_shape_fails_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="没有可覆盖输入"):
        _choose_covering_shape(
            {(1, 3, 736, 1312): Path("small.om")},
            (1, 3, 736, 1760),
        )
