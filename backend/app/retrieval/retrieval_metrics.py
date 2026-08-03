from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class RetrievalProfiler:
    """Per-request retrieval timings without global mutable state."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._timings: dict[str, dict[str, float]] = defaultdict(dict)
        self._counters: dict[str, dict[str, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    @contextmanager
    def span(self, category: str, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_seconds(category, name, time.perf_counter() - started)

    def add_seconds(self, category: str, name: str, seconds: float) -> None:
        with self._lock:
            values = self._timings[category]
            values[name] = values.get(name, 0.0) + float(seconds)

    def increment(self, category: str, name: str, value: int = 1) -> None:
        with self._lock:
            values = self._counters[category]
            values[name] = values.get(name, 0) + int(value)

    def snapshot(self) -> dict:
        with self._lock:
            timing = {
                category: {
                    name: round(value, 6)
                    for name, value in sorted(values.items())
                }
                for category, values in sorted(self._timings.items())
                if values
            }
            counters = {
                category: dict(sorted(values.items()))
                for category, values in sorted(self._counters.items())
                if values
            }
        return {
            "elapsed_seconds": round(time.perf_counter() - self._started, 6),
            "timing": timing,
            "counters": counters,
        }
