#!/usr/bin/env python3
"""检查OCR collection的schema配置"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.settings import Settings
from app.indexing.milvus_client import MilvusClient

settings = Settings()
client = MilvusClient(settings)

try:
    col = client.collection_for_name("ocr_embeddings")

    print("="*60)
    print("OCR Collection Schema检查")
    print("="*60)

    # 获取schema
    schema = col.schema

    # 查找text字段
    text_field = None
    for field in schema.fields:
        if field.name == "text":
            text_field = field
            break

    if text_field:
        print(f"\n✓ 找到text字段")
        print(f"  类型: {text_field.dtype}")
        print(f"  最大长度: {text_field.params.get('max_length', 'N/A')}")
        print(f"  enable_analyzer: {text_field.params.get('enable_analyzer', False)}")

        analyzer_params = text_field.params.get('analyzer_params')
        if analyzer_params:
            print(f"  ✓ analyzer_params: {analyzer_params}")
        else:
            print(f"  ❌ analyzer_params: 未配置（使用默认分析器）")
    else:
        print("\n❌ 未找到text字段")

    # 查找sparse_embedding字段
    sparse_field = None
    for field in schema.fields:
        if field.name == "sparse_embedding":
            sparse_field = field
            break

    if sparse_field:
        print(f"\n✓ 找到sparse_embedding字段")
        print(f"  类型: {sparse_field.dtype}")
        print(f"  is_function_output: {sparse_field.is_function_output}")
    else:
        print("\n❌ 未找到sparse_embedding字段")

    # 检查functions
    if hasattr(schema, 'functions') and schema.functions:
        print(f"\n✓ Collection有{len(schema.functions)}个Function:")
        for func in schema.functions:
            print(f"  - {func.name}: {func.type}")
            print(f"    input: {func.input_field_names}")
            print(f"    output: {func.output_field_names}")
    else:
        print("\n❌ Collection没有配置Function")

    print("\n" + "="*60)

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
