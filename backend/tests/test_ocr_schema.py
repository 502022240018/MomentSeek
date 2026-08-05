"""Unit tests for the OCR hybrid schema and legacy-schema guard."""
from __future__ import annotations

import sys

from types import SimpleNamespace

import pytest

from pymilvus import DataType, FunctionType
from app.vector_store.milvus.milvus_client import _validate_existing_ocr_collection
from app.vector_store.milvus.milvus_schema import (
    EMBEDDING_DIMS,
    _TEXT_LEN,
    create_ocr_schema,
    truncate_text_for_milvus,
)


class TestOCRSchema:
    """Test OCR schema configuration."""

    def test_ocr_schema_creation(self):
        """Test that OCR schema can be created."""
        schema = create_ocr_schema()
        assert schema is not None
        assert schema.description == "OCR: DiskANN + BM25 hybrid search"
        print("✓ OCR schema created successfully")

    def test_ocr_schema_fields(self):
        """Test that OCR schema has all required fields."""
        schema = create_ocr_schema()
        field_names = [field.name for field in schema.fields]

        required_fields = [
            "pk", "video_id", "asset_version", "model_version",
            "frame_idx", "region_idx", "frame_ms", "start_ms", "end_ms",
            "avg_box_score", "text", "has_embedding", "embedding", "sparse_embedding"
        ]

        for field_name in required_fields:
            assert field_name in field_names, f"Missing field: {field_name}"

        print(f"✓ All {len(required_fields)} required fields present")

    def test_text_field_analyzer(self):
        """Test that text field has chinese analyzer enabled."""
        schema = create_ocr_schema()

        text_field = None
        for field in schema.fields:
            if field.name == "text":
                text_field = field
                break

        assert text_field is not None, "text field not found"
        assert text_field.dtype == DataType.VARCHAR
        assert text_field.params.get("enable_analyzer") is True

        analyzer_params = text_field.params.get("analyzer_params")
        assert analyzer_params is not None, "analyzer_params not set"
        # analyzer_params might be a string or dict depending on pymilvus version
        if isinstance(analyzer_params, dict):
            assert analyzer_params.get("type") == "chinese", "analyzer type should be 'chinese'"
        else:
            assert "chinese" in str(analyzer_params), "analyzer should be 'chinese'"

        print("✓ Text field has chinese analyzer enabled")

    def test_text_limit_is_utf8_safe_and_bounded(self):
        """The 5 KB Milvus cap must not split a multibyte character."""
        assert _TEXT_LEN == 5000
        result = truncate_text_for_milvus("中" * 2000)
        assert len(result.encode("utf-8")) <= _TEXT_LEN
        assert result == "中" * 1666
        assert truncate_text_for_milvus("a" * 3000) == "a" * 2000

    def test_sparse_embedding_field(self):
        """Test that sparse_embedding field is configured correctly."""
        schema = create_ocr_schema()

        sparse_field = None
        for field in schema.fields:
            if field.name == "sparse_embedding":
                sparse_field = field
                break

        assert sparse_field is not None, "sparse_embedding field not found"
        assert sparse_field.dtype == DataType.SPARSE_FLOAT_VECTOR
        assert sparse_field.is_function_output is True

        print("✓ sparse_embedding field configured correctly")

    def test_embedding_dimension(self):
        """Test that embedding field has correct dimension."""
        schema = create_ocr_schema()

        embedding_field = None
        for field in schema.fields:
            if field.name == "embedding":
                embedding_field = field
                break

        assert embedding_field is not None, "embedding field not found"
        assert embedding_field.dtype == DataType.FLOAT_VECTOR
        assert embedding_field.params.get("dim") == EMBEDDING_DIMS["ocr"]
        assert EMBEDDING_DIMS["ocr"] == 384, "OCR embedding dimension should be 384"

        print(f"✓ Embedding dimension correct: {EMBEDDING_DIMS['ocr']}")

    def test_bm25_function(self):
        """Test that BM25 function is configured correctly."""
        schema = create_ocr_schema()

        assert hasattr(schema, 'functions'), "Schema should have functions attribute"
        assert schema.functions is not None, "Schema functions should not be None"
        assert len(schema.functions) == 1, "Should have exactly 1 function"

        bm25_func = schema.functions[0]
        assert bm25_func.name == "bm25_ocr"
        assert bm25_func.type == FunctionType.BM25
        assert bm25_func.input_field_names == ["text"]
        assert bm25_func.output_field_names == ["sparse_embedding"]

        print("✓ BM25 function configured correctly")
        print(f"  Name: {bm25_func.name}")
        print(f"  Type: {bm25_func.type}")
        print(f"  Input: {bm25_func.input_field_names}")
        print(f"  Output: {bm25_func.output_field_names}")

    def test_has_embedding_field(self):
        """Test that has_embedding field is present with correct default."""
        schema = create_ocr_schema()

        has_embedding_field = None
        for field in schema.fields:
            if field.name == "has_embedding":
                has_embedding_field = field
                break

        assert has_embedding_field is not None, "has_embedding field not found"
        assert has_embedding_field.dtype == DataType.BOOL
        # default_value might be a proto object, check the value representation
        default_val = has_embedding_field.default_value
        assert default_val is not None and str(default_val).strip().lower() in ['true', 'bool_data: true']

        print("✓ has_embedding field configured correctly")

    def test_legacy_collection_requires_rebuild_before_search(self):
        col = SimpleNamespace(
            schema=SimpleNamespace(
                fields=[SimpleNamespace(name="text"), SimpleNamespace(name="embedding")],
                functions=[],
            ),
            indexes=[SimpleNamespace(field_name="embedding")],
        )

        with pytest.raises(RuntimeError, match="Drop and rebuild the OCR Milvus index"):
            _validate_existing_ocr_collection(col)

    def test_hybrid_collection_passes_startup_validation(self):
        col = SimpleNamespace(
            schema=SimpleNamespace(
                fields=[SimpleNamespace(name=name) for name in {
                    "text", "embedding", "sparse_embedding", "has_embedding",
                }],
                functions=[SimpleNamespace(name="bm25_ocr")],
            ),
            indexes=[
                SimpleNamespace(field_name="embedding"),
                SimpleNamespace(field_name="sparse_embedding"),
            ],
        )

        _validate_existing_ocr_collection(col)


def main():
    """Run schema tests manually."""
    print("="*70)
    print("OCR Schema Tests")
    print("="*70)

    test = TestOCRSchema()

    try:
        print("\n1. Testing schema creation...")
        test.test_ocr_schema_creation()

        print("\n2. Testing required fields...")
        test.test_ocr_schema_fields()

        print("\n3. Testing text field analyzer...")
        test.test_text_field_analyzer()

        print("\n4. Testing sparse_embedding field...")
        test.test_sparse_embedding_field()

        print("\n5. Testing embedding dimension...")
        test.test_embedding_dimension()

        print("\n6. Testing BM25 function...")
        test.test_bm25_function()

        print("\n7. Testing has_embedding field...")
        test.test_has_embedding_field()

        print("\n" + "="*70)
        print("✓ All schema tests passed!")
        print("="*70)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
