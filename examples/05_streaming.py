"""Verify one fitted objective across in-memory and replayable sample sources."""

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="full",
    app_title="One objective, in memory or in batches",
)


@app.cell(hide_code=True)
def _():
    import importlib.util
    import json
    import resource
    import tempfile
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from _support import (
        COLORS, classification_metrics, diagnostics_summary, fit_cost_summary,
        plot_style, regression_metrics, require_success, style_axes,
        synthetic_classification, synthetic_regression,
    )
    from stratified_models import (
        ADMM, L2, AddIntercept, Axis, BinaryLogisticSpec,
        ConjugateGradient, FixedWidthBins, Graph, Laplacian,
        NewtonSolver, Pipeline, PreparationOptions, QuadraticSolver,
        RegressionSpec, Standardize, compose, fit, fit_batches,
    )
    from stratified_models.data import from_pandas, from_polars, parquet_source
    return (
        AddIntercept,
        Axis,
        BinaryLogisticSpec,
        COLORS,
        ConjugateGradient,
        FixedWidthBins,
        Graph,
        L2,
        Laplacian,
        NewtonSolver,
        Path,
        Pipeline,
        PreparationOptions,
        QuadraticSolver,
        RegressionSpec,
        Standardize,
        classification_metrics,
        compose,
        diagnostics_summary,
        fit,
        fit_batches,
        fit_cost_summary,
        from_pandas,
        from_polars,
        importlib,
        mo,
        np,
        parquet_source,
        pd,
        plot_style,
        plt,
        regression_metrics,
        require_success,
        resource,
        style_axes,
        synthetic_classification,
        synthetic_regression,
        tempfile,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One objective, in memory or in batches

    **Regression and logistic fitting · explicit composition · replayable sources**

    Batch boundaries describe execution, not a different statistical model.
    The loss is the sum of every sample's contribution divided by the **global
    training count**. Averaging the means of unequal batches would be wrong.

    Below, the same fixed snapshot is fitted in memory and through pandas,
    Polars, or Parquet batches. Standardization is learned from training data
    by a separate composable transform. Sufficient-stat acceleration is
    deliberately disabled so numerical evaluations really replay the source.

    Out-of-core support bounds memory in the number of samples. The graph,
    coefficients, and solver workspace must still fit the configured model budget.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {"Public API: one recipe, two execution entry points": mo.md(
            "```python\n"
            "in_memory = fit(recipe, X, y, solver=solver)\n\n"
            "source = parquet_source(\n"
            "    'training.parquet', target='target', batch_size=8192,\n"
            ")\n"
            "streamed = fit_batches(recipe, source, solver=solver)\n\n"
            "# Pandas and Polars also adapt to the same replayable-source contract.\n"
            "source = from_pandas(X, y, batch_size=8192)\n"
            "source = from_polars(frame, targets, batch_size=8192)\n"
            "```\n\n"
            "A custom source supplies a stable snapshot identifier, column schema, "
            "row count when known, and a fresh closable `batches()` iterator. "
            "Training batches contain aligned feature columns, targets, and row IDs. "
            "The optimizer consumes validated numerical batches independently of the storage engine."
        )}
    )
    return


@app.cell(hide_code=True)
def _(importlib, mo):
    _source_options = ["pandas"]
    if importlib.util.find_spec("polars") is not None:
        _source_options.append("Polars")
    if importlib.util.find_spec("pyarrow") is not None:
        _source_options.append("Parquet")
    source_control = mo.ui.dropdown(
        options=_source_options,
        value="Parquet" if "Parquet" in _source_options else "pandas",
        label="Replayable source adapter",
    )
    stream_task = mo.ui.dropdown(
        options=["Squared regression", "Binary logistic"],
        value="Squared regression", label="Task",
    )
    batch_control = mo.ui.dropdown(
        options=[17, 37, 64, 128], value=37,
        label="Rows per batch · deliberately uneven tails",
    )
    mo.hstack([stream_task, source_control, batch_control], justify="start", gap=2)
    return batch_control, source_control, stream_task


@app.cell(hide_code=True)
def _(
    AddIntercept,
    Axis,
    BinaryLogisticSpec,
    ConjugateGradient,
    Graph,
    L2,
    Laplacian,
    NewtonSolver,
    Pipeline,
    QuadraticSolver,
    RegressionSpec,
    Standardize,
    compose,
    stream_task,
    synthetic_classification,
    synthetic_regression,
):
    if stream_task.value == "Squared regression":
        stream_train, stream_y, stream_test, stream_test_y, _ = synthetic_regression(
            seed=59, n_groups=4, train_per_group=36,
            test_per_group=35, missing_group=None,
        )
        _stream_spec = RegressionSpec
        stream_solver = QuadraticSolver(linear_solver=ConjugateGradient(tol=1e-9))
    else:
        stream_train, stream_y, stream_test, stream_test_y = synthetic_classification(
            seed=59, n_groups=4, n_train=144, n_test=220,
        )
        _stream_spec = BinaryLogisticSpec
        stream_solver = NewtonSolver(atol=1e-6, rtol=1e-6)
    stream_axis = Axis(name="group", nodes=tuple(range(4)))
    stream_features = ("x", "noise", "intercept")
    stream_recipe = Pipeline(
        preprocessing=compose(
            Standardize(columns=("x", "noise")),
            AddIntercept(name="intercept"),
        ),
        model=_stream_spec(
            features=stream_features,
            axes=(stream_axis,),
            penalties=(
                L2(name="shrinkage", weight=0.02, features=stream_features),
                Laplacian(name="smoothing", weight=0.01, graph=Graph.path(axis=stream_axis), features=stream_features),
            ),
        ),
    )
    untouched_stream_train = stream_train.copy(deep=True)
    return (
        stream_recipe,
        stream_solver,
        stream_test,
        stream_test_y,
        stream_train,
        stream_y,
        untouched_stream_train,
    )


@app.cell(hide_code=True)
def _(
    Path,
    batch_control,
    from_pandas,
    from_polars,
    parquet_source,
    source_control,
    stream_train,
    stream_y,
    tempfile,
):
    quick_directory = tempfile.TemporaryDirectory(prefix="stratified-quick-stream-")
    if source_control.value == "pandas":
        selected_source = from_pandas(stream_train, stream_y, batch_size=batch_control.value)
        selected_features = from_pandas(stream_train, batch_size=batch_control.value)
    elif source_control.value == "Polars":
        import polars as _pl

        _polars_frame = _pl.from_pandas(stream_train)
        selected_source = from_polars(_polars_frame, stream_y.to_numpy(), batch_size=batch_control.value)
        selected_features = from_polars(_polars_frame, batch_size=batch_control.value)
    else:
        _parquet_path = Path(quick_directory.name) / "training.parquet"
        stream_train.assign(target=stream_y.to_numpy()).to_parquet(_parquet_path, index=False)
        selected_source = parquet_source(_parquet_path, target="target", batch_size=batch_control.value)
        selected_features = parquet_source(
            _parquet_path, columns=("x", "noise", "group"),
            batch_size=batch_control.value,
        )
    observed_batch_sizes = tuple(len(batch.targets) for batch in selected_source.batches())
    return observed_batch_sizes, selected_features, selected_source


@app.cell(hide_code=True)
def _(
    PreparationOptions,
    fit,
    fit_batches,
    require_success,
    selected_source,
    stream_recipe,
    stream_solver,
    stream_train,
    stream_y,
):
    streaming_options = PreparationOptions(
        batch_size=64,
        statistics_max_bytes=0,
        max_model_bytes=128 * 1024 * 1024,
    )
    in_memory_result = fit(
        stream_recipe, stream_train, stream_y,
        solver=stream_solver, options=streaming_options,
    )
    batched_result = fit_batches(
        stream_recipe, selected_source,
        solver=stream_solver, options=streaming_options,
    )
    in_memory_model = require_success(in_memory_result)
    batched_model = require_success(batched_result)
    return batched_model, batched_result, in_memory_model, in_memory_result


@app.cell(hide_code=True)
def _(
    batched_model,
    batched_result,
    classification_metrics,
    diagnostics_summary,
    fit_cost_summary,
    in_memory_model,
    in_memory_result,
    mo,
    np,
    observed_batch_sizes,
    pd,
    regression_metrics,
    stream_task,
    stream_test,
    stream_test_y,
):
    if stream_task.value == "Squared regression":
        _memory_predictions = in_memory_model.predict(stream_test)
        _batch_predictions = batched_model.predict(stream_test)
        _metric_function = regression_metrics
    else:
        _memory_predictions = in_memory_model.predict_proba(stream_test)
        _batch_predictions = batched_model.predict_proba(stream_test)
        _metric_function = classification_metrics
    prediction_difference = float(np.max(np.abs(_memory_predictions - _batch_predictions)))
    coefficient_difference = float(np.max(np.abs(in_memory_model.coefficients - batched_model.coefficients)))
    source_comparison = pd.DataFrame(
        [
            {"execution": name, **_metric_function(stream_test_y, prediction), **diagnostics_summary(result.diagnostics), **fit_cost_summary(result)}
            for name, prediction, result in (
                ("In memory", _memory_predictions, in_memory_result),
                ("Replayable batches", _batch_predictions, batched_result),
            )
        ]
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## Equivalent fits on one snapshot

                The observed batch lengths are **{observed_batch_sizes}**.
                The final batch is included with its actual sample count.

                **Maximum prediction difference:** `{prediction_difference:.3g}` ·
                **maximum coefficient difference:** `{coefficient_difference:.3g}`
                """
            ),
            mo.ui.table(source_comparison.round(7), selection=None, pagination=False),
            mo.callout(
                "Preparation and numerical-solve costs are shown separately, with their total passes. "
                "Fitting transforms and validating the source require preparation scans. "
                "Logistic optimization may replay the data many times; batches do not imply a one-pass or online algorithm.",
                kind="info",
            ),
        ]
    )
    return coefficient_difference, prediction_difference


@app.cell(hide_code=True)
def _(
    COLORS,
    batched_model,
    in_memory_model,
    np,
    plot_style,
    plt,
    stream_task,
    stream_test,
    stream_test_y,
    style_axes,
):
    with plot_style():
        batch_figure, _axes = plt.subplots(1, 2, figsize=(12, 3.8))
        if stream_task.value == "Squared regression":
            _a = in_memory_model.predict(stream_test)
            _b = batched_model.predict(stream_test)
            _axes[0].scatter(_a, _b, s=15, alpha=0.55, color=COLORS["stratified"])
            _limit = (min(float(_a.min()), float(_b.min())), max(float(_a.max()), float(_b.max())))
            _axes[0].plot(_limit, _limit, "--", color=COLORS["reference"])
            style_axes(_axes[0], title="Predictions agree across source adapters", xlabel="In-memory prediction", ylabel="Batched prediction")
            _residual = stream_test_y.to_numpy() - _b
            _axes[1].scatter(_b, _residual, s=15, alpha=0.5, color=COLORS["stratified"])
            _axes[1].axhline(0, ls="--", color=COLORS["reference"])
            style_axes(_axes[1], title="Independent test residuals", xlabel="Prediction", ylabel="Observed − predicted")
        else:
            _a = in_memory_model.predict_proba(stream_test)[:, 1]
            _b = batched_model.predict_proba(stream_test)[:, 1]
            _axes[0].scatter(_a, _b, s=15, alpha=0.55, color=COLORS["stratified"])
            _axes[0].plot([0, 1], [0, 1], "--", color=COLORS["reference"])
            style_axes(_axes[0], title="Positive-class probabilities agree", xlabel="In-memory probability", ylabel="Batched probability")
            for _label, _color in ((0, COLORS["independent"]), (1, COLORS["stratified"])):
                _axes[1].hist(_b[stream_test_y.to_numpy() == _label], bins=np.linspace(0, 1, 13), alpha=0.55, color=_color, label=f"Observed class {_label}")
            style_axes(_axes[1], title="Held-out probability distributions", xlabel="Positive-class probability", ylabel="Rows")
            _axes[1].legend()
        batch_figure.tight_layout()
    batch_figure
    return


@app.cell(hide_code=True)
def _(batched_model, mo, np, selected_features, stream_task, stream_train):
    if stream_task.value == "Squared regression":
        with batched_model.predict_batches(selected_features) as _prediction_stream:
            _prediction_chunks = tuple(batch.values for batch in _prediction_stream)
        _direct_prediction = batched_model.predict(stream_train)
    else:
        with batched_model.predict_proba_batches(selected_features) as _prediction_stream:
            _prediction_chunks = tuple(batch.values for batch in _prediction_stream)
        _direct_prediction = batched_model.predict_proba(stream_train)
    prediction_stream_agrees = bool(np.allclose(np.concatenate(_prediction_chunks), _direct_prediction, atol=1e-9, rtol=1e-9))
    mo.md(
        "## Prediction also streams\n\n"
        "`predict_batches` and `predict_proba_batches` yield prediction chunks with row identifiers. "
        "A context manager closes the underlying reader if a consumer stops early. "
        f"The quick fixture verifies equality with in-memory prediction: **{prediction_stream_agrees}**. "
        "Here the small prediction chunks are concatenated only to check equivalence; "
        "a production sink would write each chunk immediately."
    )
    return (prediction_stream_agrees,)


@app.cell(hide_code=True)
def _(FixedWidthBins, compose, from_pandas, mo, np, pd):
    binning_component = FixedWidthBins(
        column="age", output="age_band", origin=0.0, width=10.0, n_bins=11,
    )
    _bin_data = pd.DataFrame({"age": [0.0, 9.9, 10.0, 35.0, 79.9, 109.9]})
    _bin_source = from_pandas(_bin_data)
    fitted_binning = compose(binning_component).fit(_bin_source)
    _binned_batches = tuple(fitted_binning.wrap_source(_bin_source).batches())
    observed_bins = np.concatenate([batch.columns["age_band"] for batch in _binned_batches])
    binning_table = _bin_data.assign(age_band=observed_bins)
    mo.vstack(
        [
            mo.md(
                "## Preparation is ordinary composition\n\n"
                "The fitted standardizer belongs to the pipeline, not the numerical solver. "
                "Fixed-width bins can likewise be an explicit component with a finite domain. "
                "This object declares the half-open age domain [0, 110); out-of-range values "
                "fail explicitly. Learned quantile boundaries and clusters are separate "
                "in-memory or pre-fitted components."
            ),
            mo.md(f"```python\n{binning_component!r}\n```"),
            mo.ui.table(binning_table, selection=None, pagination=False),
        ]
    )
    return (observed_bins,)


@app.cell(hide_code=True)
def _(mo):
    large_run_form = mo.ui.dictionary(
        {
            "rows": mo.ui.dropdown(options=[100_000, 1_000_000], value=100_000, label="Generated file-backed rows"),
            "batch_size": mo.ui.dropdown(options=[2048, 8192], value=8192, label="Rows per read"),
        }
    ).form(submit_button_label="Generate and fit the larger Parquet experiment")
    mo.vstack(
        [
            mo.md(
                "## A deliberate larger run\n\n"
                "This optional run writes a reproducible dataset in bounded chunks and fits "
                "it through Parquet scans. It requires the `parquet` extra, included in `common`. "
                "The configuration is submitted explicitly, so changing an unsubmitted "
                "control does not trigger another expensive fit."
            ),
            large_run_form,
        ]
    )
    return (large_run_form,)


@app.cell(hide_code=True)
def _(
    AddIntercept,
    Axis,
    ConjugateGradient,
    Graph,
    L2,
    Laplacian,
    Path,
    Pipeline,
    PreparationOptions,
    QuadraticSolver,
    RegressionSpec,
    compose,
    diagnostics_summary,
    fit_batches,
    fit_cost_summary,
    importlib,
    large_run_form,
    mo,
    np,
    parquet_source,
    pd,
    require_success,
    resource,
    tempfile,
    time,
):
    if large_run_form.value is None:
        large_run_output = mo.callout("The larger experiment has not been run. All quick checks above are complete.", kind="neutral")
        large_directory = None
    elif importlib.util.find_spec("pyarrow") is None:
        large_run_output = mo.callout("Install the Parquet/notebook dependencies to run this experiment.", kind="warn")
        large_directory = None
    else:
        import pyarrow as _pa
        import pyarrow.parquet as _pq

        _configuration = large_run_form.value
        _n_rows = int(_configuration["rows"])
        _batch_rows = int(_configuration["batch_size"])
        large_directory = tempfile.TemporaryDirectory(prefix="stratified-large-stream-")
        _path = Path(large_directory.name) / "large.parquet"
        _started = time.perf_counter()
        with _pq.ParquetWriter(
            _path,
            _pa.schema([("x", _pa.float64()), ("group", _pa.int64()), ("target", _pa.float64())]),
        ) as _writer:
            for _start in range(0, _n_rows, 16_384):
                _ids = np.arange(_start, min(_start + 16_384, _n_rows), dtype=np.int64)
                _groups = _ids % 6
                _x = np.sin(_ids * 0.731) + np.cos(_ids * 0.017)
                _y = (1 + 0.4 * np.sin(_groups)) * _x + 0.2 * np.cos(_groups) + 0.1 * np.sin(_ids * 2.31)
                _writer.write_table(_pa.table({"x": _x, "group": _groups, "target": _y}))
        _axis = Axis(name="group", nodes=tuple(range(6)))
        _features = ("x", "intercept")
        _large_recipe = Pipeline(
            preprocessing=compose(AddIntercept(name="intercept")),
            model=RegressionSpec(
                features=_features, axes=(_axis,),
                penalties=(
                    L2(name="shrinkage", weight=0.002, features=_features),
                    Laplacian(name="smoothing", weight=0.01, graph=Graph.path(axis=_axis), features=_features),
                ),
            ),
        )
        _large_result = fit_batches(
            _large_recipe,
            parquet_source(_path, target="target", batch_size=_batch_rows),
            solver=QuadraticSolver(linear_solver=ConjugateGradient(tol=1e-8)),
            options=PreparationOptions(statistics_max_bytes=0, batch_size=_batch_rows),
        )
        require_success(_large_result)
        _record = {
            "rows": _n_rows, "read batch": _batch_rows,
            "coefficient count": 12, "Parquet MiB": _path.stat().st_size / 1024 ** 2,
            "generation + fit seconds": time.perf_counter() - _started,
            "process high-water RSS MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **diagnostics_summary(_large_result.diagnostics),
            **fit_cost_summary(_large_result),
        }
        large_run_output = mo.vstack(
            [
                mo.ui.table(pd.DataFrame([_record]).round(4), selection=None, pagination=False),
                mo.callout(
                    "The RSS number is this notebook process's high-water mark, including "
                    "imports and previous cells. It is an observation, not an isolated memory "
                    "benchmark. The package's separate memory tests compare fresh processes "
                    "at fixed batch/model sizes as sample count grows.",
                    kind="info",
                ),
            ]
        )
    large_run_output
    return


@app.cell(hide_code=True)
def _(
    batched_result,
    coefficient_difference,
    in_memory_result,
    observed_batch_sizes,
    observed_bins,
    prediction_difference,
    prediction_stream_agrees,
    stream_train,
    untouched_stream_train,
):
    review_checks = {
        "both numerical solves converged": in_memory_result.diagnostics.converged and batched_result.diagnostics.converged,
        "uneven batches include every row": sum(observed_batch_sizes) == len(stream_train),
        "prediction partitions preserve the fitted objective": prediction_difference < 2e-3,
        "coefficient partitions preserve the fitted objective": coefficient_difference < 2e-3,
        "prediction stream matches in-memory prediction": prediction_stream_agrees,
        "input dataframe remains unchanged": stream_train.equals(untouched_stream_train),
        "numerical replay actually occurred": batched_result.diagnostics.source_passes > 0,
        "fixed-width bins obey the declared boundaries": tuple(observed_bins) == (0, 0, 1, 3, 7, 10),
    }
    assert all(review_checks.values()), review_checks
    return


if __name__ == "__main__":
    app.run()
