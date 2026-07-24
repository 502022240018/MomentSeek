"""Batch buffer for efficient Milvus upsert operations."""

from typing import List, Dict, Callable, Optional
import time
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class BatchBuffer:
    """
    Batch buffer for efficient Milvus upsert operations.

    Accumulates data and automatically flushes when:
    - Buffer size reaches batch_size threshold
    - Timeout has elapsed since last flush

    Supports deterministic primary keys for idempotent upsert.
    Thread-safe for concurrent additions.

    Args:
        collection:       Milvus Collection instance.
        batch_size:       Number of records to accumulate before auto-flush.
        timeout_seconds:  Max seconds to wait before auto-flush.
        pk_generator:     Optional function to generate deterministic primary key
                          from a data dict.
        upsert_fn:        Optional callable ``(collection, rows) -> None`` that
                          performs the actual upsert.  Defaults to a direct
                          ``collection.upsert(rows)`` call.  Inject
                          ``milvus_indexer._upsert_with_retry`` (or any wrapper)
                          here to add retry / back-pressure without coupling this
                          module to milvus_indexer.
    """

    def __init__(
        self,
        collection,
        batch_size: int = 100,
        timeout_seconds: float = 5.0,
        pk_generator: Optional[Callable[[Dict], str]] = None,
        upsert_fn: Optional[Callable] = None,
    ):
        self.collection = collection
        self.batch_size = batch_size
        self.timeout = timeout_seconds
        self.pk_generator = pk_generator
        # Default: direct upsert (no retry).  Callers may inject a retry wrapper.
        self._upsert_fn: Callable = upsert_fn or (lambda rows: collection.upsert(rows))
        self.buffer: List[Dict] = []
        self.last_flush = time.time()
        self.lock = Lock()
        self._total_flushed = 0

    def add(self, data: Dict) -> None:
        """
        Add a record to the buffer.

        Automatically generates primary key if pk_generator is provided.
        Automatically flushes if batch_size or timeout threshold is reached.

        Args:
            data: Record data dictionary
        """
        with self.lock:
            # Generate deterministic primary key
            if self.pk_generator and "pk" not in data:
                data["pk"] = self.pk_generator(data)

            self.buffer.append(data)

            # Auto-flush on threshold or timeout
            if len(self.buffer) >= self.batch_size or \
               time.time() - self.last_flush > self.timeout:
                self._flush_internal()

    def _flush_internal(self) -> None:
        """Internal flush implementation (must be called with lock held)."""
        if not self.buffer:
            return

        try:
            self._upsert_fn(list(self.buffer))
            count = len(self.buffer)
            self._total_flushed += count
            logger.debug(f"Flushed {count} records to {self.collection.name} "
                        f"(total: {self._total_flushed})")

        except Exception as e:
            logger.error(f"Failed to flush batch to {self.collection.name}: {e}")
            raise
        finally:
            self.buffer.clear()
            self.last_flush = time.time()

    def flush(self) -> None:
        """Manually flush all pending records."""
        with self.lock:
            self._flush_internal()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures all records are flushed."""
        try:
            self.flush()
        except Exception as e:
            logger.error(f"Error during context manager exit: {e}")
            # Don't suppress the original exception if one exists
            if exc_type is None:
                raise
        return False

    @property
    def pending_count(self) -> int:
        """Get number of records waiting to be flushed."""
        with self.lock:
            return len(self.buffer)

    @property
    def total_flushed(self) -> int:
        """Get total number of records flushed so far."""
        return self._total_flushed
