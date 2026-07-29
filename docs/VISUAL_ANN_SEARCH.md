# Visual ANN Search

## Overview

Visual search uses ANN (Approximate Nearest Neighbor) to efficiently recall candidate frames from large video collections, replacing the previous full-query approach that fetched all embeddings.

**Key benefits:**
- 60-80% latency reduction
- No network transfer of full embedding set
- Multi-query aggregation preserves semantic correctness
- Results go directly to VLM reranking (no z-score normalization needed)

## Architecture

```
Query Embeddings (N subqueries)
    ↓
ANN Search (HNSW or DiskANN)
    ↓
Top-K Frames per Subquery
    ↓
Multi-Query Aggregation (0.65*mean + 0.35*min)
    ↓
Segment-Level Aggregation (mean of top-N frames, N configurable)
    ↓
Candidates → VLM Reranking
```

## Configuration

### Environment Variables

```bash
# Index type
VISUAL_USE_DISKANN=false  # false=HNSW (memory), true=DiskANN (disk)

# Recall size
VISUAL_ANN_TOP_K=500  # Candidates per subquery (300-1000 recommended)

# Segment aggregation
VISUAL_ANN_SEGMENT_TOP_N=3  # Top-N frames per segment for scoring (3-10 recommended)
```

### Index Types

**HNSW (default):**
- In-memory index
- Fast query (<50ms)
- Requires ~4GB RAM for 100K frames
- Uses `ef` parameter (≥ top_k)

**DiskANN:**
- Disk-based index
- Slower query (~100-200ms)
- Low memory footprint
- Uses `search_list` parameter (≥ top_k)

## Implementation

### Core Function

```python
from app.indexing.milvus_search_visual_v2 import milvus_visual_candidates_ann

candidates = milvus_visual_candidates_ann(
    client=milvus_client,
    video_id="video_123",
    query_texts=[query_embedding_1, query_embedding_2],  # Multi-query support
    limit=20,
    profile="balanced",  # "precision", "balanced", "recall"
)
```

### Multi-Query Aggregation

For queries with multiple subqueries (e.g., "red car AND mountain scenery"):

1. **Per-Frame Aggregation**: For each frame with scores `[s1, s2, ...]` across queries:
   - Single query: use score directly
   - Multi-query: `0.65 * mean(scores) + 0.35 * min(scores)`
   - This ensures "simultaneously satisfying multiple constraints"

2. **Per-Segment Aggregation**: Mean of top-N frame scores per segment (N configurable via `VISUAL_ANN_SEGMENT_TOP_N`, default=3)

### Profile Modes

- **precision**: Strict thresholds, returns `limit` candidates
- **balanced** (default): Moderate thresholds, returns `limit` candidates
- **recall**: Relaxed thresholds, returns up to 500 candidates

## Index Management

### Rebuild Index

When switching between HNSW and DiskANN:

```bash
python backend/scripts/rebuild_visual_index.py
```

### Verification

The system automatically verifies index type matches configuration on startup. Mismatches raise:

```
MilvusVisualSearchError: Index type mismatch: config expects DISKANN but collection has HNSW.
Run backend/scripts/rebuild_visual_index.py to rebuild.
```

## Migration from Legacy

### Changes

**Removed:**
- Distribution sampling (no longer needed with VLM reranking)
- Z-score normalization
- Percentile calculation
- Legacy full-query fallback
- `milvus_fallback_enabled` setting
- `visual_sample_size`, `visual_sample_strategy` settings
- **Visual from `BULK_QUERY_FIELDS`** (critical fix - see below)

**Simplified:**
- Direct ANN recall → aggregation → candidates
- Raw cosine scores mapped via `visual_confidence()`
- Candidate fields reduced to essentials

### Critical Bug Fix: Bulk Prefetch Removal

**Problem Identified (2026-07-29):**

The v2 ANN implementation replaced the legacy full-query approach, but `"visual"` remained in `BULK_QUERY_FIELDS`, causing the platform to execute a costly no-op prefetch:

```
SearchEngine.search()
  → iterates BULK_QUERY_FIELDS (including "visual")
  → query_rows_for_videos("visual", ..., ["embedding", ...])
  → query_iterator fetches ALL visual rows with embeddings
  → passes rows to milvus_visual_candidates()
  → rows are IGNORED (ANN issues its own collection.search())
```

**Trigger Conditions:**
- Milvus read enabled
- Visual modality requested
- Video has Visual index
- Video routed to Milvus

→ **Every production Visual search executed full bulk fetch before ANN.**

**Fix Applied:**

1. **Removed `"visual"` from `BULK_QUERY_FIELDS`** (`backend/app/indexing/milvus_search.py:103`)
   - Added comment explaining why Visual is intentionally absent
   - Only ASR and OCR remain in bulk prefetch

2. **Cleaned up call site** (`backend/app/search.py:1536`)
   - Removed deprecated `duration_ms`, `segment_ms`, `rows` arguments from `milvus_visual_candidates()` call
   - Now passes only active parameters: `profile`, `limit`, `profiler`

3. **Added regression test** (`backend/tests/test_search.py`)
   - `test_visual_ann_does_not_bulk_fetch_rows()`
   - Asserts `_query_rows_for_videos` is never called with `modality="visual"`
   - Prevents this issue from returning

**Impact:**
- Eliminates O(N) embedding fetch before ANN (where N = total frames in video)
- Latency improvement varies with video length (larger videos benefit more)
- No functional change to search results

### Candidate Fields

Current output:
```python
Candidate(
    video_id="video_123",
    start_time=10.5,
    end_time=15.5,
    score=0.82,  # visual_confidence(cosine)
    modality="visual",
    raw_score=0.64,  # raw cosine from ANN
    evidence="[milvus_ann] score=0.64 · rank=0.82 · 12 frames · 2 queries",
    unit_type="segment",
    unit_id=42,
)
```

## Performance

### Typical Metrics

| Metric | Full-Query (Legacy) | ANN (Current) |
|--------|---------------------|---------------|
| Latency | 200-500ms | 50-100ms |
| Network Transfer | 2-4MB | <100KB |
| Memory | High | Low-Medium |
| Accuracy | 100% | ~85-95% |

### Tuning

**Increase `visual_ann_top_k` if:**
- Recall is too low
- Missing relevant segments
- Videos are very long (>1 hour)

**Decrease `visual_ann_top_k` if:**
- Latency is too high
- Getting too many false positives

**Increase `visual_ann_segment_top_n` if:**
- Want more robust segment scores (less sensitive to outlier frames)
- Segments have many candidate frames
- Recommended range: 5-10 for dense results

**Decrease `visual_ann_segment_top_n` if:**
- Want more sensitive detection (favor peak frames)
- Segments have few candidate frames
- Recommended range: 1-3 for sparse results

## Testing

### Unit Tests

```bash
pytest backend/tests/integration/test_visual_optimization.py -v
```

### Manual Testing

```python
from app.indexing.milvus_client import MilvusClient
from app.indexing.milvus_search import milvus_visual_candidates
import numpy as np

client = MilvusClient()
query = np.random.randn(1152).astype(np.float32)
query = query / np.linalg.norm(query)

candidates = milvus_visual_candidates(
    client=client,
    video_id="test_video",
    query=query,
    limit=20,
)

for c in candidates[:5]:
    print(f"{c.start_time:.1f}s - {c.end_time:.1f}s: {c.score:.3f}")
```

## Troubleshooting

### High Latency

- Check `visual_ann_top_k` (lower if too high)
- Verify index type matches configuration
- Consider switching HNSW ↔ DiskANN

### Low Recall

- Increase `visual_ann_top_k`
- Use `profile="recall"` mode
- Verify index is built correctly

### Index Mismatch Error

```bash
# Rebuild index to match configuration
python backend/scripts/rebuild_visual_index.py
```

## References

- Implementation: `backend/app/indexing/milvus_search_visual_v2.py`
- Settings: `backend/app/settings.py`
- Milvus HNSW: https://milvus.io/docs/index.md#HNSW
- Milvus DiskANN: https://milvus.io/docs/disk_index.md
