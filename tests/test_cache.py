"""Solver cache ownership, bounds, and key isolation."""

import numpy as np
import pytest

from stratified_models.solvers.cache import SolverCache
from stratified_models.solvers.types import ADMM, QuadraticStatistics


def test_cache_enforces_lru_budget_and_dependency_keys() -> None:
    cache = SolverCache(max_bytes=32)
    first = np.array([1.0, 2.0])
    assert cache.put(("design-a", 1.0), first)
    assert cache.put(("design-b", 1.0), first)
    assert cache.get(("design-a", 2.0)) is None
    np.testing.assert_array_equal(cache.get(("design-a", 1.0)), first)
    assert cache.put(("design-c", 1.0), first)
    assert cache.get(("design-b", 1.0)) is None
    snapshot = cache.snapshot()
    assert snapshot.bytes_used == 32
    assert snapshot.evictions == 1
    assert snapshot.hits == 1


def test_cache_owns_arrays_and_skips_oversize_payloads() -> None:
    cache = SolverCache(max_bytes=16)
    original = np.array([1.0, 2.0])
    assert cache.put("factor", original)
    original[0] = 100.0
    stored = cache.get("factor")
    assert isinstance(stored, np.ndarray)
    assert stored[0] == 1.0
    assert not stored.flags.writeable
    assert not cache.put("too-large", np.zeros(3))
    assert cache.snapshot().bytes_used == 16


def test_quadratic_statistics_owns_validated_arrays() -> None:
    xtx = np.ones((1, 1, 1))
    statistics = QuadraticStatistics(xtx=xtx, xty=np.ones((1, 1)), yty=1.0)
    xtx[0, 0, 0] = 2.0
    assert statistics.xtx[0, 0, 0] == 1.0
    assert not statistics.xtx.flags.writeable
    with pytest.raises(ValueError, match="xty"):
        QuadraticStatistics(xtx=xtx, xty=np.ones((2, 1)), yty=1.0)


@pytest.mark.parametrize(
    "kwargs", [{"rho": 0.0}, {"atol": float("nan")}, {"max_passes": -1}]
)
def test_admm_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ADMM(**kwargs)
