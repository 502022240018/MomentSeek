from app.indexing.milvus_search import query_rows_for_videos
from app.retrieval_metrics import RetrievalProfiler


class FakeIterator:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.closed = False

    def next(self):
        return next(self.pages)

    def close(self):
        self.closed = True


class FakeField:
    def __init__(self, name):
        self.name = name


class FakeCollection:
    name = "visual"

    def __init__(self, pages):
        self.schema = type("Schema", (), {
            "fields": [
                FakeField("video_id"),
                FakeField("frame_idx"),
                FakeField("embedding"),
            ]
        })()
        self.pages = pages
        self.calls = []
        self.iterator = None

    def query_iterator(self, **kwargs):
        self.calls.append(kwargs)
        self.iterator = FakeIterator(self.pages)
        return self.iterator


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def collection_for(self, modality):
        assert modality == "visual"
        return self.collection


def test_query_rows_for_videos_uses_one_iterator_and_groups_rows():
    collection = FakeCollection([
        [
            {"video_id": "v1", "frame_idx": 0, "embedding": [1.0, 0.0]},
            {"video_id": "v2", "frame_idx": 0, "embedding": [0.0, 1.0]},
        ],
        [{"video_id": "v1", "frame_idx": 1, "embedding": [0.9, 0.1]}],
        [],
    ])
    profiler = RetrievalProfiler()

    grouped = query_rows_for_videos(
        FakeClient(collection),
        "visual",
        ["v1", "v2"],
        ["frame_idx", "embedding"],
        profiler,
    )

    assert [row["frame_idx"] for row in grouped["v1"]] == [0, 1]
    assert [row["frame_idx"] for row in grouped["v2"]] == [0]
    assert len(collection.calls) == 1
    assert collection.calls[0]["expr"] == 'video_id in ["v1", "v2"]'
    assert collection.iterator.closed is True
    snapshot = profiler.snapshot()
    assert snapshot["counters"]["milvus"]["visual_requests"] == 1
    assert snapshot["counters"]["milvus"]["visual_rows"] == 3
