from __future__ import annotations

import gc
import threading
import time
from typing import Callable


class ModelPool:
    """Process-local cache of heavy models with idle-timeout eviction.

    Daemon workers use one pool per process to reuse expensive model loads and
    accelerator compilation across jobs. A positive ``idle_timeout`` evicts idle
    entries; a non-positive value keeps them resident until worker shutdown.

    Thread-safety: ``get`` may be called concurrently; building happens outside
    the lock and a double-check avoids caching duplicate instances.
    """

    def __init__(
        self,
        idle_timeout: float = 300.0,
        reap_interval: float = 30.0,
        on_free: Callable[[object], None] | None = None,
    ):
        self._idle_timeout = idle_timeout
        self._on_free = on_free
        self._entries: dict[str, list] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop,
            args=(reap_interval,),
            name="model-pool-reaper",
            daemon=True,
        )
        self._reaper.start()

    def get(self, key: str, factory: Callable[[], object]) -> object:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry[1] = time.monotonic()
                return entry[0]
        obj = factory()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry[1] = time.monotonic()
                self._free(obj)
                return entry[0]
            self._entries[key] = [obj, time.monotonic()]
            return obj

    def evict_idle(self) -> list[str]:
        if self._idle_timeout <= 0:
            return []
        cutoff = time.monotonic() - self._idle_timeout
        evicted = []
        with self._lock:
            for key in list(self._entries):
                obj, last_used = self._entries[key]
                if last_used <= cutoff:
                    del self._entries[key]
                    evicted.append((key, obj))
        for _, obj in evicted:
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
        else:
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
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
