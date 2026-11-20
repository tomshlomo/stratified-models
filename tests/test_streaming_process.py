"""Fresh interpreters expose native allocation growth hidden by tracemalloc."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "profile_streaming.py"
_MIB = 1024 * 1024


def _profile(
    samples: int, *, fit: bool = False, batch_size: int = 4_096
) -> dict[str, object]:
    environment = dict(os.environ)
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        str(_SCRIPT),
        "--samples",
        str(samples),
        "--batch-size",
        str(batch_size),
        "--fit" if fit else "--prepare-only",
    ]
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    result = json.loads(completed.stdout)
    assert isinstance(result, dict)
    return result


@pytest.mark.skipif(sys.platform == "win32", reason="Unix resource RSS measurement")
@pytest.mark.integration
def test_fresh_process_rss_stays_bounded_for_two_million_streamed_rows() -> None:
    small = _profile(20_000)
    large = _profile(2_000_000)
    for profile in (small, large):
        samples = cast("int", profile["samples"])
        assert profile["source_passes"] == 2
        assert profile["rows_scanned"] == 2 * samples
        assert cast("int", profile["largest_batch"]) <= 4_096
        assert profile["active_iterators"] == 0
        assert profile["max_active_iterators"] == 1
        preparation = cast("dict[str, object]", profile["preparation"])
        assert preparation["statistics_bytes"] == 0
        assert profile["fit"] is None
    assert small["model"] == large["model"]
    small_peak = cast("int", small["peak_rss_bytes"])
    large_peak = cast("int", large["peak_rss_bytes"])
    # Fixed batch/work-state memory, with 32 MiB plus interpreter startup noise.
    # Retaining the large numerical sample matrix alone would cost ~137 MiB.
    allowance = 32 * _MIB + max(8 * _MIB, small_peak // 20)
    assert large_peak <= small_peak + allowance, (small, large, allowance)


@pytest.mark.skipif(sys.platform == "win32", reason="Unix resource RSS measurement")
def test_profiler_also_executes_real_streamed_fitting() -> None:
    profile = _profile(2_000, fit=True, batch_size=128)
    fitting = cast("dict[str, object]", profile["fit"])
    assert fitting["converged"] is True
    assert cast("int", fitting["source_passes"]) > 0
    assert cast("int", profile["rows_scanned"]) > 4_000
    assert profile["largest_batch"] == 128
    assert profile["active_iterators"] == 0


def test_profiler_rejects_nonpositive_source_sizes() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", "0"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert "positive integer" in completed.stderr
