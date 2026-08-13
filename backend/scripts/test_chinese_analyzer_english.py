"""
Test whether Milvus chinese analyzer handles English text properly
"""
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility
)

# Connect to Milvus
connections.connect("default", host="localhost", port=19531)

# Test collection name
test_col_name = "test_chinese_english_analyzer"

# Drop if exists
if utility.has_collection(test_col_name):
    utility.drop_collection(test_col_name)
    print(f"Dropped existing collection: {test_col_name}")

# Create test collection with chinese analyzer
fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("text", DataType.VARCHAR, max_length=1000,
                enable_analyzer=True,
                analyzer_params={"type": "chinese"}),
    FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
]

schema = CollectionSchema(fields, enable_dynamic_field=False)

# Add BM25 function
from pymilvus import Function, FunctionType
bm25_function = Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_embedding"],
)
schema.add_function(bm25_function)

col = Collection(test_col_name, schema)
print(f"Created collection: {test_col_name}")

# Insert test data - mixed Chinese and English
test_texts = [
    "这是纯中文文本",
    "This is pure English text",
    "这是混合Chinese and English文本",
    "工资salary薪水payment",
    "machine learning机器学习",
    "Hello world你好世界",
]

data = [{"text": t} for t in test_texts]
col.insert(data)
col.flush()
print(f"Inserted {len(test_texts)} test records")

# Create index
col.create_index(
    "sparse_embedding",
    {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
)
col.load()
print("Index created and collection loaded")

# Test queries
test_queries = [
    "工资",           # Pure Chinese
    "salary",         # Pure English
    "machine",        # English word
    "学习",           # Chinese word
    "Hello",          # English greeting
    "世界",           # Chinese word
]

from pymilvus import AnnSearchRequest
print("\n" + "="*60)
print("BM25 Search Results:")
print("="*60)

for query in test_queries:
    req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_embedding",
        param={"metric_type": "BM25"},
        limit=3,
    )

    results = col.hybrid_search(
        reqs=[req],
        rerank=None,
        limit=3,
        output_fields=["text"]
    )

    print(f"\nQuery: '{query}'")
    if results and results[0]:
        for i, hit in enumerate(results[0], 1):
            print(f"  {i}. [score={hit.score:.4f}] {hit.entity.get('text')}")
    else:
        print("  No results")

# Cleanup
utility.drop_collection(test_col_name)
print(f"\n✓ Test complete, collection dropped")
