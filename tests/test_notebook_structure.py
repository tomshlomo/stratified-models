"""Execute the default notebook journeys and their substantive review checks.

These integration tests are explicit because optional notebook dependencies and
reactive execution are unnecessary for a core-only installation.
"""

from __future__ import annotations

import os
import runpy
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_NOTEBOOK_TESTS") != "1",
        reason="Set RUN_NOTEBOOK_TESTS=1 to execute the optional notebook journeys.",
    ),
]


@pytest.mark.parametrize(
    "filename",
    (
        "01_graph_regression.py",
        "04_solver_diagnostics.py",
        "05_streaming.py",
        "06_workflow.py",
    ),
)
def test_default_notebook_journey(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("marimo")
    pytest.importorskip("matplotlib")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    namespace = runpy.run_path(
        str(root / "examples" / filename), run_name="notebook_review"
    )
    _, definitions = namespace["app"].run()
    checks = definitions["review_checks"]
    assert isinstance(checks, Mapping)
    assert checks, f"{filename} did not expose its review contract"
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    ("filename", "values"),
    (
        (
            "04_solver_diagnostics.py",
            {
                "task_control": "Binary logistic",
                "l1_control": True,
                "reference_control": False,
                "rho_control": 0.0,
            },
        ),
        (
            "05_streaming.py",
            {
                "stream_task": "Binary logistic",
                "source_control": "Polars",
                "batch_control": 17,
            },
        ),
        (
            "06_workflow.py",
            {"tuning_form": {"trials": 2}, "threshold_control": 0.8},
        ),
    ),
)
def test_notebook_control_journey(
    filename: str,
    values: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("marimo")
    pytest.importorskip("matplotlib")
    if filename == "05_streaming.py":
        pytest.importorskip("polars")
    if filename == "06_workflow.py":
        pytest.importorskip("optuna")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    namespace = runpy.run_path(
        str(root / "examples" / filename), run_name="notebook_review_variant"
    )
    _, definitions = namespace["app"].run(
        defs={name: SimpleNamespace(value=value) for name, value in values.items()}
    )
    checks = definitions["review_checks"]
    assert isinstance(checks, Mapping)
    assert all(checks.values()), checks
    if filename == "06_workflow.py":
        from stratified_models.tuning import TuningSuccess

        assert isinstance(definitions["tuning_result"], TuningSuccess)
