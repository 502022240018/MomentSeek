from app.identity.face_gallery_service import _query_all


class FakeIterator:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.closed = False

    def next(self):
        return next(self.pages)

    def close(self):
        self.closed = True


class FakeCollection:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.iterator = None

    def query_iterator(self, **kwargs):
        self.calls.append(kwargs)
        self.iterator = FakeIterator(self.pages)
        return self.iterator


def test_query_all_reads_every_face_page_and_closes_iterator():
    collection = FakeCollection([
        [{"track_idx": 0}, {"track_idx": 1}],
        [{"track_idx": 2}],
    ])

    rows = _query_all(
        collection,
        expr='video_id == "video-1"',
        output_fields=["track_idx"],
    )

    assert [row["track_idx"] for row in rows] == [0, 1, 2]
    assert collection.calls[0]["batch_size"] == 2000
    assert collection.iterator.closed is True
