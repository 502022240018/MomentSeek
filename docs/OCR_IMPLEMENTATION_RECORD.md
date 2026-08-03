# OCR DiskANN + BM25 Hybrid Retrieval

## Scope

OCR retrieval uses Milvus only in production:

- Dense semantic retrieval: `embedding` with a DISKANN index and IP metric.
- Lexical retrieval: `text` is analysed by Milvus and converted to
  `sparse_embedding` by the `bm25_ocr` Function.
- Fusion: `WeightedRanker(semantic_weight, lexical_weight)`.

The implementation lives in:

| Responsibility | Path |
| --- | --- |
| Schema | `backend/app/vector_store/milvus/milvus_schema.py` |
| Collection and indexes | `backend/app/vector_store/milvus/milvus_client.py` |
| Hybrid retrieval | `backend/app/vector_store/milvus/milvus_search.py` |
| Search orchestration | `backend/app/retrieval/search.py` |
| Settings | `backend/app/core/settings.py` |

## Production requirements

Production OCR requests do not fall back to NPZ. Set:

```bash
MILVUS_ENABLED=true
MILVUS_READ_ENABLED=true
MILVUS_WRITE_ENABLED=true
MILVUS_FALLBACK_ENABLED=false
```

The relevant tuning settings are:

```bash
OCR_HYBRID_RECALL_SIZE=100
OCR_LEXICAL_WEIGHT=0.7
OCR_DISKANN_SEARCH_LIST=100
```

`OCR_LEXICAL_WEIGHT` is validated in `[0, 1]`; semantic weight is computed as
`1 - OCR_LEXICAL_WEIGHT`.

## Required collection schema

`ocr_embeddings` must contain:

- scalar metadata: `pk`, `video_id`, `asset_version`, `model_version`,
  `frame_idx`, `region_idx`, `frame_ms`, `start_ms`, `end_ms`, and
  `avg_box_score`;
- `text` with the built-in `chinese` analyzer;
- `has_embedding`, `embedding`, and Function output `sparse_embedding`;
- the `bm25_ocr` BM25 Function;
- a DISKANN index on `embedding` and a BM25 sparse inverted index on
  `sparse_embedding`.

The application validates an existing OCR collection during startup. A legacy
collection fails fast with an instruction to rebuild, rather than accepting
traffic and failing on a user's first query.

## One-time upgrade procedure

Adding a Function output field or sparse index is not an in-place Milvus schema
migration. Before deploying this release, ensure that every OCR asset can be
rebuilt from its source video or retained development NPZ artifact.

1. Stop OCR indexing and search traffic for the deployment window.
2. Verify a recoverable source for each OCR asset.
3. Deploy the application image containing the hybrid schema, but keep the app
   service stopped until the legacy collection is removed.
4. Drop the legacy collection from a one-off app container:

   ```bash
   docker compose run --rm --no-deps app \
     python -c "import os; from pymilvus import connections, utility; connections.connect(host=os.environ['MILVUS_HOST'], port=os.environ['MILVUS_PORT']); utility.drop_collection('ocr_embeddings')"
   ```

5. Start the app once so it creates and validates the new collection.
6. Rebuild every OCR index. Existing NPZ files may be supplied as an offline
   migration input:

   ```bash
   docker compose exec app python -m app.vector_store.milvus.rebuild_ocr_from_npz
   ```

   The command defaults to `APP_DATA_DIR/indexes`; use `--dry-run` to list
   assets first or `--index-root` to point at another retained index directory.
7. Check that the collection has both indexes and `bm25_ocr`, then issue a
   BM25-only and a hybrid OCR query.

Do not drop the collection automatically from application startup code.

## Query behaviour

`milvus_ocr_candidates_hybrid()` has three explicit paths:

| Input | Retrieval path |
| --- | --- |
| text and semantic embedding | DISKANN + BM25 hybrid search |
| text without semantic embedding | BM25-only search |
| semantic embedding without text | DISKANN-only search |

Dense requests filter `has_embedding == True`. BM25 requests include
lexical-only OCR frames so recognized text remains searchable even when a
semantic embedding is unavailable.

## Tests and CI

- Unit schema and legacy-collection validation: `backend/tests/test_ocr_schema.py`.
- Live Milvus integration tests: `backend/tests/integration/test_ocr_hybrid_search.py`.

The integration test creates an isolated temporary collection, uses the
production schema and index configuration, and validates BM25-only, dense-only,
and hybrid queries. It reads `MILVUS_HOST` and `MILVUS_PORT`; the GitHub Actions
workflow supplies `127.0.0.1:19530`.

Run locally:

```bash
cd backend
python -m pytest tests/test_ocr_schema.py -q
python -m pytest tests/integration/test_ocr_hybrid_search.py -m integration -q -o addopts=-ra
```
