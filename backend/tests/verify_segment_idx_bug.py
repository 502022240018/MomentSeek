"""Verify the segment_idx mapping issue described in PHASE2_VERIFICATION_RESULT.md"""

# Simulate ASR semantic indexing process

# Original ASR chunks
chunks = [
    {"text": "chunk0", "semantic_eligible": True},
    {"text": "", "semantic_eligible": True},        # Empty, skipped
    {"text": "chunk2", "semantic_eligible": True},
    {"text": "", "semantic_eligible": True},        # Empty, skipped
    {"text": "chunk4", "semantic_eligible": True},
]

# Build indexed list (as in text_semantic.py)
indexed = [
    (index, str(chunk.get("text", "")).strip())
    for index, chunk in enumerate(chunks)
    if chunk.get("semantic_eligible", True) and str(chunk.get("text", "")).strip()
]

print("=" * 70)
print("ASR Semantic Indexing Process")
print("=" * 70)
print()

print("Original chunks:")
for i, chunk in enumerate(chunks):
    print(f"  chunks[{i}] = {chunk['text']!r}")
print()

print("Indexed (filtered) chunks:")
for item in indexed:
    print(f"  indexed: index={item[0]}, text={item[1]!r}")
print()

# Extract indices and texts (as saved in NPZ)
embedding_chunk_indices = [item[0] for item in indexed]
texts_all = [chunk["text"] for chunk in chunks]  # All chunks saved in NPZ

print("NPZ storage:")
print(f"  embedding_chunk_indices = {embedding_chunk_indices}")
print(f"  texts (all chunks)      = {texts_all}")
print()

print("=" * 70)
print("Current Milvus Writing (WRONG)")
print("=" * 70)
print()

print("Current code: segment_idx = embed_idx")
rows_wrong = []
for embed_idx, chunk_idx in enumerate(embedding_chunk_indices):
    row = {
        "segment_idx": embed_idx,           # ❌ WRONG: using embed_idx
        "text": texts_all[chunk_idx],
        "actual_chunk": chunk_idx,
    }
    rows_wrong.append(row)
    print(f"  Row {embed_idx}: segment_idx={embed_idx}, text={texts_all[chunk_idx]!r}, actual_chunk={chunk_idx}")

print()
print("Milvus data (WRONG):")
print("  segment_idx | text     | Should be chunk")
print("  ------------|----------|----------------")
for row in rows_wrong:
    match = "✓" if row["segment_idx"] == row["actual_chunk"] else "✗"
    print(f"  {row['segment_idx']:11} | {row['text']:8} | {row['actual_chunk']} {match}")

print()
print("=" * 70)
print("Phase 2 Reading (based on WRONG data)")
print("=" * 70)
print()

# Simulate _texts_from_milvus() reading
max_idx = max(r["segment_idx"] for r in rows_wrong)
texts_from_milvus_wrong = [
    next((r["text"] for r in rows_wrong if r["segment_idx"] == i), "")
    for i in range(max_idx + 1)
]

print(f"_texts_from_milvus() returns (length={len(texts_from_milvus_wrong)}):")
for i, text in enumerate(texts_from_milvus_wrong):
    expected = texts_all[i] if i < len(texts_all) else "(out of range)"
    match = "✓" if text == expected else "✗"
    print(f"  texts[{i}] = {text!r:10} (expected: {expected!r:10}) {match}")

print()
print("=" * 70)
print("Correct Milvus Writing (FIXED)")
print("=" * 70)
print()

print("Fixed code: segment_idx = chunk_idx")
rows_fixed = []
for embed_idx, chunk_idx in enumerate(embedding_chunk_indices):
    row = {
        "segment_idx": chunk_idx,           # ✅ FIXED: using chunk_idx
        "text": texts_all[chunk_idx],
        "actual_chunk": chunk_idx,
    }
    rows_fixed.append(row)
    print(f"  Row {embed_idx}: segment_idx={chunk_idx}, text={texts_all[chunk_idx]!r}, actual_chunk={chunk_idx}")

print()
print("Milvus data (FIXED):")
print("  segment_idx | text     | Chunk")
print("  ------------|----------|------")
for row in rows_fixed:
    print(f"  {row['segment_idx']:11} | {row['text']:8} | {row['actual_chunk']} ✓")

print()
print("=" * 70)
print("Phase 2 Reading (based on FIXED data)")
print("=" * 70)
print()

# Simulate _texts_from_milvus() with fixed data
max_idx_fixed = max(r["segment_idx"] for r in rows_fixed)
texts_from_milvus_fixed = [
    next((r["text"] for r in rows_fixed if r["segment_idx"] == i), "")
    for i in range(max_idx_fixed + 1)
]

print(f"_texts_from_milvus() returns (length={len(texts_from_milvus_fixed)}):")
for i, text in enumerate(texts_from_milvus_fixed):
    expected = texts_all[i] if i < len(texts_all) else "(out of range)"
    match = "✓" if text == expected else "✗"
    print(f"  texts[{i}] = {text!r:10} (expected: {expected!r:10}) {match}")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

wrong_count = sum(1 for i, text in enumerate(texts_from_milvus_wrong) if i < len(texts_all) and text != texts_all[i])
fixed_count = sum(1 for i, text in enumerate(texts_from_milvus_fixed) if i < len(texts_all) and text != texts_all[i])

print(f"Current implementation (segment_idx = embed_idx):")
print(f"  - Mismatches: {wrong_count}/{len(texts_all)}")
print(f"  - Verification: {'❌ FAILED' if wrong_count > 0 else '✓ PASSED'}")
print()

print(f"Fixed implementation (segment_idx = chunk_idx):")
print(f"  - Mismatches: {fixed_count}/{len(texts_all)}")
print(f"  - Verification: {'✓ PASSED' if fixed_count == 0 else '❌ FAILED'}")
print()

if wrong_count > 0:
    print("⚠️  CRITICAL BUG CONFIRMED!")
    print("   The current code uses embed_idx instead of chunk_idx for segment_idx,")
    print("   causing text misalignment in Speaker service.")
    print()
    print("   Fix required: Change line 165 in milvus_indexer.py")
    print("   FROM: 'segment_idx': embed_idx,")
    print("   TO:   'segment_idx': chunk_idx,")
else:
    print("✓ No bug found")
