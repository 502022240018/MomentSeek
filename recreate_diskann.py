#!/usr/bin/env python3
"""重建visual_embeddings集合以使用DiskANN索引"""
import sys
sys.path.insert(0, 'backend')

from app.indexing.milvus_client import get_milvus_client, reset_milvus_client
from pymilvus import connections, utility, Collection

print("Step 1: Connecting to Milvus...")
connections.connect(host="127.0.0.1", port=19531)

print("\nStep 2: Checking existing collection...")
if utility.has_collection("visual_embeddings"):
    print("  Found existing visual_embeddings collection")
    col = Collection("visual_embeddings")
    indexes = col.indexes
    print(f"  Current index type: {indexes[0].params.get('index_type') if indexes else 'None'}")

    print("\nStep 3: Dropping collection...")
    col.drop()
    print("  ✓ Collection dropped")
else:
    print("  No existing collection found")

print("\nStep 4: Reinitializing MilvusClient with DiskANN config...")
reset_milvus_client()
client = get_milvus_client()

print("\nStep 5: Verifying new index...")
col = Collection("visual_embeddings")
indexes = col.indexes

if indexes:
    idx = indexes[0]
    index_type = idx.params.get('index_type')
    print(f"  ✓ Collection recreated")
    print(f"  Index type: {index_type}")
    print(f"  Metric type: {idx.params.get('metric_type')}")
    print(f"  Parameters: {idx.params.get('params')}")

    if index_type == "DISKANN":
        print("\n✅ DiskANN index successfully created!")
    else:
        print(f"\n⚠️  Warning: Index type is {index_type}, not DISKANN")
else:
    print("  ✗ No index found")

print("\nStep 6: Collection stats...")
stats = client.stats("visual_embeddings")
print(f"  Entities: {stats['num_entities']}")
print(f"  Loaded: {stats['loaded']}")
