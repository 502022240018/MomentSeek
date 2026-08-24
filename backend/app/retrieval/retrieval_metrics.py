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
        self._timing_samples: dict[str, dict[str, list[float]]] = defaultdict(dict)
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
        duration = float(seconds)
        with self._lock:
            values = self._timings[category]
            values[name] = values.get(name, 0.0) + duration
            samples = self._timing_samples[category]
            samples.setdefault(name, []).append(duration)

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
            timing_stats = {
                category: {
                    name: _timing_stats(samples)
                    for name, samples in sorted(values.items())
                }
                for category, values in sorted(self._timing_samples.items())
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
            "timing_stats": timing_stats,
            "counters": counters,
        }


def _timing_stats(samples: list[float]) -> dict[str, float | int]:
    """Summarise repeated spans while keeping the legacy summed timing intact."""
    values = sorted(samples)
    total = sum(values)
    return {
        "count": len(values),
        "total": round(total, 6),
        "mean": round(total / len(values), 6),
        "min": round(values[0], 6),
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "max": round(values[-1], 6),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (
        sorted_values[upper] - sorted_values[lower]
    ) * fraction
