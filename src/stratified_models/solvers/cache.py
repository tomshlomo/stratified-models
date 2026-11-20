"""An explicitly shared, byte-bounded least-recently-used solver cache."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from threading import RLock

import numpy as np

from stratified_models.solvers.types import CacheStatistics


def _size(value: object) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(_size(item) for item in value)
    if isinstance(value, (float, int, bool, np.number)):
        return 8
    msg = "Specify nbytes explicitly for cache entries without NumPy array storage."
    raise TypeError(msg)


def _freeze(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


class SolverCache:
    """Caches numerical operators by exact, caller-supplied dependency keys.

    A cache does not own warm starts. Its byte budget counts stored numerical
    payloads; Python key/container overhead is not included. Calls are thread-safe,
    but two concurrent cache misses may independently compute the same factor.
    """

    def __init__(self, *, max_bytes: int = 256 * 1024 * 1024) -> None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            msg = "max_bytes must be a nonnegative integer."
            raise ValueError(msg)
        self._max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[object, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._factorizations = 0
        self._evictions = 0
        self._lock = RLock()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def get(self, key: Hashable) -> object | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: Hashable, value: object, *, nbytes: int | None = None) -> bool:
        """Store an owned immutable payload, or return False when over budget."""
        size = _size(value) if nbytes is None else nbytes
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            msg = "nbytes must be a nonnegative integer."
            raise ValueError(msg)
        with self._lock:
            if size > self._max_bytes or self._max_bytes == 0:
                return False
            owned = _freeze(value)
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            while self._entries and self._bytes + size > self._max_bytes:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._bytes -= evicted_size
                self._evictions += 1
            self._entries[key] = (owned, size)
            self._bytes += size
            return True

    def record_factorization(self) -> None:
        with self._lock:
            self._factorizations += 1

    def snapshot(self) -> CacheStatistics:
        with self._lock:
            return CacheStatistics(
                hits=self._hits,
                misses=self._misses,
                factorizations=self._factorizations,
                bytes_used=self._bytes,
                entries=len(self._entries),
                evictions=self._evictions,
            )
