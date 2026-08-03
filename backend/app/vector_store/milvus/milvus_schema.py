"""Milvus collection schemas, versioned primary-key builders, and dimension probes.

Primary-key format (deterministic, idempotent upsert):
    {video_id}#{asset_ver}#{model_ver}#{modality}#{segment_id}

asset_ver  — caller-supplied rebuild counter / source-file hash.
             Incrementing it creates a *new* set of records; the old set
             can be cleaned up after the new one is validated.
model_ver  — identifies the embedding model; changing it also produces
             new records rather than silently overwriting old ones.

Embedding dimensions:
    visual   — 1152   (SigLIP2-so400m-384)
    asr      — 384    (paraphrase-multilingual-MiniLM-L12-v2, NOT 768)
    ocr      — 384    (same model as asr)
    face     — 512    (InsightFace buffalo_l)
    speaker  — 192    (3D-Speaker CAM++)

Dimensions are validated at runtime via probe_embedding_dim() before
collection creation; a mismatch raises ValueError with an actionable hint.
"""
from __future__ import annotations

from typing import Dict

from pymilvus import CollectionSchema, DataType, FieldSchema

# ---------------------------------------------------------------------------
# Authoritative embedding dimensions — update when switching models.
# These are schema-creation values; probe_embedding_dim() cross-checks them.
# ---------------------------------------------------------------------------
EMBEDDING_DIMS: Dict[str, int] = {
    "visual": 1152,   # SigLIP2-so400m-384 image encoder
    "asr":     384,   # paraphrase-multilingual-MiniLM-L12-v2 (384, not 768)
    "ocr":     384,   # same text model as asr
    "face":    512,   # InsightFace buffalo_l normed_embedding
    "speaker": 192,   # 3D-Speaker CAM++ speaker encoder
}

MODEL_VERSIONS: Dict[str, str] = {
    "visual":  "siglip2-so400m-v1",
    "asr":     "paraphrase-multilingual-minilm-l12-v2-v1",
    "ocr":     "paraphrase-multilingual-minilm-l12-v2-v1",
    "face":    "insightface-buffalo-l-v1",
    "speaker": "3dspeaker-campplus-zh-en-192-v1",
}

# Max varchar lengths
_PK_LEN = 512
_VID_LEN = 255
_VER_LEN = 64
_TEXT_LEN = 2000


# ---------------------------------------------------------------------------
# Primary-key builders
# ---------------------------------------------------------------------------

def make_pk(video_id: str, asset_ver: str, model_ver: str, modality: str, segment_id: str) -> str:
    """Build a deterministic primary key; all parts are sanitised to avoid #."""
    parts = [video_id, asset_ver, model_ver, modality, segment_id]
    return "#".join(part.replace("#", "_") for part in parts)


def visual_pk(video_id: str, asset_ver: str, frame_idx: int, model_ver: str = MODEL_VERSIONS["visual"]) -> str:
    return make_pk(video_id, asset_ver, model_ver, "visual", f"f{frame_idx:08d}")


def asr_pk(video_id: str, asset_ver: str, segment_idx: int, model_ver: str = MODEL_VERSIONS["asr"]) -> str:
    return make_pk(video_id, asset_ver, model_ver, "asr", f"s{segment_idx:08d}")


def ocr_pk(video_id: str, asset_ver: str, frame_idx: int, region_idx: int, model_ver: str = MODEL_VERSIONS["ocr"]) -> str:
    return make_pk(video_id, asset_ver, model_ver, "ocr", f"f{frame_idx:08d}r{region_idx:04d}")


def face_pk(video_id: str, asset_ver: str, track_idx: int, model_ver: str = MODEL_VERSIONS["face"]) -> str:
    return make_pk(video_id, asset_ver, model_ver, "face", f"t{track_idx:08d}")


def speaker_pk(video_id: str, asset_ver: str, utterance_idx: int, model_ver: str = MODEL_VERSIONS["speaker"]) -> str:
    return make_pk(video_id, asset_ver, model_ver, "speaker", f"u{utterance_idx:08d}")


# ---------------------------------------------------------------------------
# Runtime dimension probe — call before creating a collection to catch mismatches
# ---------------------------------------------------------------------------

def probe_embedding_dim(modality: str, actual_dim: int) -> None:
    """Raise ValueError if actual_dim contradicts the schema constant.

    Call this once after extracting a real embedding vector from the model, so
    dimension mismatches are caught at index-build time rather than at upsert time.
    """
    expected = EMBEDDING_DIMS.get(modality)
    if expected is None:
        raise ValueError(f"Unknown modality for dimension probe: {modality!r}")
    if actual_dim != expected:
        raise ValueError(
            f"Embedding dimension mismatch for modality={modality!r}: "
            f"model produced {actual_dim}d, schema expects {expected}d. "
            f"Update EMBEDDING_DIMS['{modality}'] in milvus_schema.py and "
            f"drop+recreate the collection."
        )


# ---------------------------------------------------------------------------
# Collection schemas
# ---------------------------------------------------------------------------

def _common_fields(modality: str) -> list[FieldSchema]:
    """Fields shared by all collections."""
    return [
        FieldSchema("pk",            DataType.VARCHAR, max_length=_PK_LEN, is_primary=True),
        FieldSchema("video_id",      DataType.VARCHAR, max_length=_VID_LEN),
        FieldSchema("asset_version", DataType.VARCHAR, max_length=_VER_LEN),
        FieldSchema("model_version", DataType.VARCHAR, max_length=_VER_LEN),
    ]


def create_visual_schema() -> CollectionSchema:
    fields = _common_fields("visual") + [
        FieldSchema("frame_idx",        DataType.INT64),
        FieldSchema("timestamp_ms",     DataType.INT64),
        # segment_id: which segment this frame belongs to.
        # For fixed-window strategy: bucket = timestamp_ms // segment_ms.
        # For shot strategy: index into explicit segment boundary array.
        FieldSchema("segment_id",       DataType.INT64,  default_value=-1),
        # segment_start_ms / segment_end_ms: explicit bounds for shot-based segments.
        # -1 means fixed-window — callers compute from segment_id * segment_ms.
        FieldSchema("segment_start_ms", DataType.INT64,  default_value=-1),
        FieldSchema("segment_end_ms",   DataType.INT64,  default_value=-1),
        FieldSchema("embedding",        DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["visual"]),
    ]
    return CollectionSchema(fields, description="Visual frame embeddings (SigLIP2)")


def create_asr_schema() -> CollectionSchema:
    fields = _common_fields("asr") + [
        FieldSchema("segment_idx",   DataType.INT64),
        FieldSchema("start_ms",      DataType.INT64),
        FieldSchema("end_ms",        DataType.INT64),
        FieldSchema("text",          DataType.VARCHAR, max_length=_TEXT_LEN),
        # True  → row carries a real semantic embedding.
        # False → lexical-only chunk; embedding is a zero-vector placeholder.
        # default_value=True keeps backward compatibility: old rows (written before
        # this field existed) are assumed to have real embeddings.
        FieldSchema("has_embedding", DataType.BOOL,  default_value=True),
        FieldSchema("embedding",     DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["asr"]),
    ]
    return CollectionSchema(fields, description="ASR chunk embeddings — all chunks stored; has_embedding=False for lexical-only rows")


def create_ocr_schema() -> CollectionSchema:
    fields = _common_fields("ocr") + [
        FieldSchema("frame_idx",     DataType.INT64),
        # region_idx kept for schema compatibility; always 0 in the new design
        # (one row per frame regardless of embedding presence).
        FieldSchema("region_idx",    DataType.INT64),
        FieldSchema("frame_ms",      DataType.INT64),
        # Aggregated OCR text for all boxes in this frame (enables lexical scoring).
        FieldSchema("text",          DataType.VARCHAR, max_length=_TEXT_LEN, default_value=""),
        # Frame display window (from frame_windows_ms in the NPZ).
        FieldSchema("start_ms",      DataType.INT64,  default_value=-1),
        FieldSchema("end_ms",        DataType.INT64,  default_value=-1),
        # Mean OCR confidence across boxes in this frame.
        FieldSchema("avg_box_score", DataType.FLOAT,  default_value=0.0),
        # True  → row carries a real semantic embedding.
        # False → lexical-only frame; embedding is a zero-vector placeholder.
        # default_value=True keeps backward compatibility with old indexed rows.
        FieldSchema("has_embedding", DataType.BOOL,   default_value=True),
        FieldSchema("embedding",     DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["ocr"]),
    ]
    return CollectionSchema(fields, description="OCR frame embeddings — all frames stored; has_embedding=False for lexical-only rows")


def create_face_schema() -> CollectionSchema:
    fields = _common_fields("face") + [
        FieldSchema("track_idx",    DataType.INT64),
        FieldSchema("start_ms",     DataType.INT64),
        FieldSchema("end_ms",       DataType.INT64),
        FieldSchema("best_ms",      DataType.INT64),
        FieldSchema("embedding",    DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["face"]),
    ]
    return CollectionSchema(fields, description="Face track embeddings (InsightFace)")


def create_speaker_schema() -> CollectionSchema:
    fields = _common_fields("speaker") + [
        FieldSchema("utterance_idx",  DataType.INT64),
        FieldSchema("start_ms",       DataType.INT64),
        FieldSchema("end_ms",         DataType.INT64),
        FieldSchema("asr_chunk_idx",  DataType.INT64),
        FieldSchema("track_id",       DataType.INT64),
        FieldSchema("embedding",      DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMS["speaker"]),
    ]
    return CollectionSchema(fields, description="Speaker utterance embeddings (3D-Speaker CAM++)")
