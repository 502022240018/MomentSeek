from __future__ import annotations

import gc
import threading
import time
from typing import Callable


class ModelPool:
    """Process-local cache of heavy models with idle-timeout eviction.

    Indexing models (CLIP ~8.6s, InsightFace ~2s, Whisper ~4s on 910B) plus the
    one-off NPU kernel compile are otherwise paid on every job because each stage
    runs in a fresh `process_exit` subprocess. Keeping the models resident across
    jobs removes that ~14.5s/job. A background reaper frees idle entries (and the
    ~2.3GB HBM they hold) after `idle_timeout` seconds, so on a shared NPU card we
    only occupy memory while actively indexing — "warm pool + idle release".

    Thread-safety: `get` may be called concurrently; building happens outside the
    lock (load is slow) and a double-check avoids two threads caching the same key.
    """

    def __init__(
        self,
        idle_timeout: float = 300.0,
        reap_interval: float = 30.0,
        on_free: Callable[[object], None] | None = None,
    ):
        self._idle_timeout = idle_timeout
        self._on_free = on_free
        self._entries: dict[str, list] = {}  # key -> [obj, last_used_monotonic]
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop, args=(reap_interval,), name="model-pool-reaper", daemon=True
        )
        self._reaper.start()

    def get(self, key: str, factory: Callable[[], object]) -> object:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry[1] = time.monotonic()
                return entry[0]
        obj = factory()  # load outside the lock; it is slow
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:  # another thread loaded it meanwhile
                entry[1] = time.monotonic()
                self._free(obj)  # drop our duplicate
                return entry[0]
            self._entries[key] = [obj, time.monotonic()]
            return obj

    def evict_idle(self) -> list[str]:
        cutoff = time.monotonic() - self._idle_timeout
        evicted = []
        with self._lock:
            for key in list(self._entries):
                obj, last_used = self._entries[key]
                if last_used <= cutoff:
                    del self._entries[key]
                    evicted.append((key, obj))
        for key, obj in evicted:
            self._free(obj)
        return [key for key, _ in evicted]

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for obj, _ in entries:
            self._free(obj)

    def _reap_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.evict_idle()
            except Exception:
                pass

    def _free(self, obj: object) -> None:
        if self._on_free is not None:
            try:
                self._on_free(obj)
            except Exception:
                pass
        del obj
        gc.collect()
        _empty_device_cache()


def _empty_device_cache() -> None:
    """Release cached NPU/CUDA blocks back to the device after eviction."""
    try:
        import torch
    except Exception:
        return
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
            return
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
