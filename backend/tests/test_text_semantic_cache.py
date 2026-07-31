from app.indexing.text_semantic import _hf_cached_snapshot_path


def test_text_semantic_cache_accepts_hub_layout(tmp_path):
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    snapshot = (
        tmp_path
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    assert _hf_cached_snapshot_path(tmp_path, model_name) == snapshot
