import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="A map of local housing models")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # A map of local housing models

    **Where should a model share information?** Nearby houses need not have identical
    price relationships, yet fitting an independent regression in every small region
    wastes information. A geographical grid gives us a concrete way to express proximity.

    Explore a small **offline synthetic adaptation**, or explicitly run a King County
    experiment following [Tuck, Barratt & Boyd (2021), §5.2, pp.19–21](https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf#page=19).
    The original-data preset uses a 50 × 50 grid, nine covariates plus an intercept,
    natural log sale prices, five folds, and an independently fitted scaler in every fold.
    """)
    return


@app.cell
def _():
    import json
    import platform
    import time
    from dataclasses import asdict
    from importlib.metadata import version
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, train_test_split

    from _paper_data import (
        AUTHOR_CODE_URL,
        HOUSE_LATITUDE,
        HOUSE_LONGITUDE,
        KING_COUNTY,
        DataIntegrityError,
        inclusive_bin_width,
        load_house_prices,
        mean_penalty_weight,
        regression_metrics,
        synthetic_house_prices,
    )
    from _support import diagnostics_summary, require_success, style_axes
    from stratified_models import (
        ADMM,
        L2,
        AddIntercept,
        AutoSolver,
        Axis,
        FitFailure,
        FixedWidthBins,
        Graph,
        Laplacian,
        Pipeline,
        RegressionSpec,
        Standardize,
        compose,
        fit,
    )

    return (
        ADMM,
        AUTHOR_CODE_URL,
        AddIntercept,
        AutoSolver,
        Axis,
        DataIntegrityError,
        FitFailure,
        FixedWidthBins,
        Graph,
        HOUSE_LATITUDE,
        HOUSE_LONGITUDE,
        KING_COUNTY,
        KFold,
        L2,
        Laplacian,
        Path,
        Pipeline,
        RandomForestRegressor,
        RegressionSpec,
        Standardize,
        asdict,
        compose,
        diagnostics_summary,
        fit,
        inclusive_bin_width,
        json,
        load_house_prices,
        mean_penalty_weight,
        np,
        pd,
        platform,
        plt,
        regression_metrics,
        require_success,
        style_axes,
        synthetic_house_prices,
        time,
        train_test_split,
        version,
    )


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        ["Quick · synthetic adaptation", "Paper method · original King County data"],
        value="Quick · synthetic adaptation",
        label="Dataset and evaluation",
    )
    solver_choice = mo.ui.dropdown(
        ["Explicit AutoSolver", "ADMM"],
        value="Explicit AutoSolver",
        label="Solver strategy",
    )
    graph_scale = mo.ui.dropdown(
        {
            "No coupling": 0.0,
            "0.1 × paper coupling": 0.1,
            "Paper coupling": 1.0,
            "10 × paper coupling": 10.0,
        },
        value="Paper coupling",
        label="Spatial smoothing",
    )
    candidate_budget = mo.ui.dropdown(
        ["One published setting per model", "Three settings per model"],
        value="One published setting per model",
        label="Equal search budget",
    )
    quick_grid = mo.ui.slider(
        start=3, stop=8, value=4, step=1, label="Synthetic grid side length"
    )
    allow_download = mo.ui.checkbox(
        value=False, label="Allow the verified King County download if uncached"
    )
    run_paper = mo.ui.run_button(label="Run original-data five-fold evaluation")
    mo.vstack(
        [
            mo.hstack([preset, solver_choice, graph_scale]),
            mo.hstack([quick_grid, candidate_budget]),
            mo.hstack([allow_download, run_paper]),
            mo.md(
                "The original-data grid is always **50 × 50**. Its full evaluation includes 100-tree forests and is intentionally opt-in."
            ),
        ]
    )
    return (
        allow_download,
        candidate_budget,
        graph_scale,
        preset,
        quick_grid,
        run_paper,
        solver_choice,
    )


@app.cell
def _(
    DataIntegrityError,
    Path,
    allow_download,
    load_house_prices,
    mo,
    preset,
    quick_grid,
    run_paper,
    synthetic_house_prices,
):
    paper_mode = preset.value == "Paper method · original King County data"
    if paper_mode:
        mo.stop(
            not run_paper.value,
            mo.callout(
                "Review the source and controls, then press Run to evaluate the original data.",
                kind="info",
            ),
        )
        try:
            dataset = load_house_prices(
                cache_dir=Path(".cache/stratified-models/paper-data"),
                allow_download=allow_download.value,
            )
        except (FileNotFoundError, DataIntegrityError, ImportError) as _error:
            mo.stop(True, mo.callout(str(_error), kind="danger"))
    else:
        dataset = synthetic_house_prices()
    grid_size = 50 if paper_mode else int(quick_grid.value)
    mo.vstack(
        [
            mo.callout(dataset.name, kind="info"),
            mo.md(
                f"**{len(dataset.frame):,} sales · {len(dataset.features)} covariates + intercept · {grid_size**2:,} local models.** "
                "The target is already in log-price units; the optimizer does not transform it."
            ),
            mo.ui.table(dataset.frame.head(8), selection=None),
        ]
    )
    return dataset, grid_size, paper_mode


@app.cell
def _(mo):
    mo.md(r"""
    ## A geographical model is an explicit composition

    Two fixed-width bin transforms create latitude and longitude labels. Their domains
    are declared constants tied to the reviewed data snapshot. We do not learn grid
    boundaries, standardization parameters, or model parameters from validation folds.
    Latitude and longitude select the local model; they are not among its nine regressors.

    The local L2 penalty excludes the intercept. Both geographical graph penalties include
    it, allowing a location-dependent baseline price. The same graph weight is applied
    to each grid direction.

    **Objective units matter.** The authors' code uses summed squared residuals; this
    package uses `sum(residual²) / (2 * n_train)`. With matching half-scaled L2 and graph
    penalties, the equivalent new weights are **`paper_weight / (2 * n_train)`**.
    The target is natural log price throughout: reported RMSE is not a dollar error.

    These half factors were checked independently: Appendix A, p.28 defines the local
    sum-of-squares function with one half, and the inspected source implements
    `lambda * sum(theta²) / 2`; equation (2), p.7 puts one half on undirected graph
    energy. Section 5.2 does not itself display `r(theta)`. If it were interpreted as
    unhalved local L2 instead, **only** the package L2 weight would double. This preset
    records the appendix/source convention explicitly.
    """)
    return


@app.cell
def _(
    AddIntercept,
    Axis,
    FixedWidthBins,
    Graph,
    HOUSE_LATITUDE,
    HOUSE_LONGITUDE,
    L2,
    Laplacian,
    Pipeline,
    RegressionSpec,
    Standardize,
    compose,
    dataset,
    graph_scale,
    grid_size,
    inclusive_bin_width,
    mean_penalty_weight,
    np,
):
    def housing_recipe(
        model_name: str, *, n_train: int, multiplier: float
    ) -> Pipeline[RegressionSpec]:
        _features = (*dataset.features, "intercept")
        _penalties: list[L2 | Laplacian] = [
            L2(
                name="local_l2",
                features=dataset.features,
                weight=mean_penalty_weight(
                    0.001 * multiplier, n_train=n_train, loss_multiplier=0.5
                ),
            )
        ]
        _axes = ()
        _steps = []
        if model_name != "Common":
            _latitude = Axis(name="latitude_bin", nodes=tuple(range(grid_size)))
            _longitude = Axis(name="longitude_bin", nodes=tuple(range(grid_size)))
            _axes = (_latitude, _longitude)
            for _column, _output, _domain in (
                ("lat", "latitude_bin", HOUSE_LATITUDE),
                ("long", "longitude_bin", HOUSE_LONGITUDE),
            ):
                _width = inclusive_bin_width(_domain, n_bins=grid_size)
                _steps.append(
                    FixedWidthBins(
                        column=_column,
                        output=_output,
                        origin=_domain[0],
                        width=_width,
                        n_bins=grid_size,
                    )
                )
            if model_name == "Stratified":
                for _name, _axis in (
                    ("latitude_graph", _latitude),
                    ("longitude_graph", _longitude),
                ):
                    _penalties.append(
                        Laplacian(
                            name=_name,
                            graph=Graph.path(axis=_axis),
                            features=_features,
                            weight=mean_penalty_weight(
                                18.5 * graph_scale.value * multiplier,
                                n_train=n_train,
                                loss_multiplier=0.5,
                            ),
                        )
                    )
        _steps.extend(
            (Standardize(columns=dataset.features), AddIntercept(name="intercept"))
        )
        return Pipeline(
            preprocessing=compose(*_steps),
            model=RegressionSpec(
                features=_features, axes=_axes, penalties=tuple(_penalties)
            ),
        )

    return (housing_recipe,)


@app.cell
def _(
    ADMM,
    AutoSolver,
    FitFailure,
    KFold,
    RandomForestRegressor,
    candidate_budget,
    dataset,
    diagnostics_summary,
    fit,
    housing_recipe,
    np,
    paper_mode,
    pd,
    regression_metrics,
    require_success,
    solver_choice,
    time,
    train_test_split,
):
    _x, _y = dataset.x, dataset.y
    _indices = np.arange(len(_y))
    if paper_mode:
        folds = tuple(KFold(n_splits=5, shuffle=True, random_state=17).split(_indices))
    else:
        folds = (train_test_split(_indices, test_size=0.3, random_state=17),)
    multipliers = (
        (0.3, 1.0, 3.0)
        if candidate_budget.value == "Three settings per model"
        else (1.0,)
    )
    chosen_solver = (
        ADMM(max_iterations=3000, atol=1e-6, rtol=1e-5)
        if solver_choice.value == "ADMM"
        else AutoSolver()
    )
    housing_rows = []
    housing_diagnostics = []
    _started = time.perf_counter()
    for _model_name in ("Common", "Separate", "Stratified", "Random forest"):
        for _candidate, _multiplier in enumerate(multipliers):
            for _fold, (_train, _validation) in enumerate(folds):
                if _model_name == "Random forest":
                    _forest = RandomForestRegressor(
                        n_estimators=100 if paper_mode else 30,
                        min_samples_leaf=(1, 3, 8)[_candidate],
                        random_state=17,
                        n_jobs=1,
                    )
                    _forest_features = [*dataset.features, "lat", "long"]
                    _forest.fit(_x.iloc[_train][_forest_features], _y[_train])
                    _prediction = _forest.predict(
                        _x.iloc[_validation][_forest_features]
                    )
                    _converged = True
                else:
                    _recipe = housing_recipe(
                        _model_name, n_train=len(_train), multiplier=_multiplier
                    )
                    _result = fit(
                        _recipe, _x.iloc[_train], _y[_train], solver=chosen_solver
                    )
                    housing_diagnostics.append(
                        {
                            "model": _model_name,
                            "multiplier": _multiplier,
                            "fold": _fold,
                            **diagnostics_summary(_result.diagnostics),
                        }
                    )
                    _converged = not isinstance(_result, FitFailure)
                    if not _converged:
                        housing_rows.append(
                            {
                                "model": _model_name,
                                "multiplier": _multiplier,
                                "fold": _fold,
                                "rmse": np.nan,
                                "r2": np.nan,
                                "converged": False,
                                "n_train": len(_train),
                            }
                        )
                        continue
                    _prediction = _result.predictor.predict(_x.iloc[_validation])
                housing_rows.append(
                    {
                        "model": _model_name,
                        "multiplier": _multiplier,
                        "fold": _fold,
                        **regression_metrics(_y[_validation], _prediction),
                        "converged": _converged,
                        "n_train": len(_train),
                    }
                )
    housing_results = pd.DataFrame(housing_rows)
    housing_summary = (
        housing_results.groupby(["model", "multiplier"], sort=False)
        .agg(
            rmse=("rmse", "mean"),
            fold_sd=("rmse", "std"),
            r2=("r2", "mean"),
            successful_folds=("converged", "sum"),
        )
        .reset_index()
    )
    _complete = housing_summary[
        (housing_summary["successful_folds"] == len(folds))
        & (housing_summary["model"] == "Stratified")
    ]
    if len(_complete) == 0:
        raise RuntimeError(
            "No stratified setting converged on every fold. Inspect diagnostics before interpreting the scores."
        )
    selected_multiplier = float(_complete.sort_values("rmse").iloc[0]["multiplier"])
    final_housing = fit(
        housing_recipe("Stratified", n_train=len(_y), multiplier=selected_multiplier),
        _x,
        _y,
        solver=chosen_solver,
    )
    fitted_regressor = require_success(final_housing)
    housing_elapsed = time.perf_counter() - _started
    assert "price" not in dataset.features
    assert "target" not in dataset.features
    assert np.isfinite(_y).all()
    return (
        chosen_solver,
        final_housing,
        fitted_regressor,
        folds,
        housing_diagnostics,
        housing_elapsed,
        housing_results,
        housing_summary,
        multipliers,
        selected_multiplier,
    )


@app.cell
def _(folds, housing_elapsed, housing_summary, mo, paper_mode):
    _best = (
        housing_summary[housing_summary["successful_folds"] == len(folds)]
        .sort_values("rmse")
        .groupby("model", sort=False)
        .head(1)
    )
    mo.vstack(
        [
            mo.md(
                f"## Measured validation results\n\n**{housing_elapsed:.2f} seconds** for "
                + (
                    "the original-data five-fold run."
                    if paper_mode
                    else "the synthetic single-holdout run."
                )
            ),
            mo.ui.table(_best, selection=None),
            mo.accordion(
                {
                    "All candidate settings and successful fold counts": mo.ui.table(
                        housing_summary, selection=None
                    )
                }
            ),
            mo.callout(
                "Every family uses the same folds and candidate count. The forest receives raw latitude/longitude as in the paper. "
                "With three candidates, forest minimum leaf sizes are 1, 3, and 8; the three stratified candidates scale all penalties together. "
                "A selected CV score uses the same folds as selection and is not an independent post-tuning test score.",
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(
    HOUSE_LATITUDE,
    HOUSE_LONGITUDE,
    dataset,
    fitted_regressor,
    folds,
    grid_size,
    housing_summary,
    np,
    plt,
    style_axes,
):
    housing_figure, _axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    _best = (
        housing_summary[housing_summary["successful_folds"] == len(folds)]
        .sort_values("rmse")
        .groupby("model", sort=False)
        .head(1)
    )
    _axes[0].bar(
        _best["model"],
        _best["rmse"],
        yerr=_best["fold_sd"].fillna(0),
        color=["#8595a8", "#e49b54", "#4778bd", "#66a58b"],
        capsize=3,
    )
    _axes[0].tick_params(axis="x", rotation=20)
    style_axes(
        _axes[0],
        title="Validation RMSE · lower is better",
        ylabel="Natural log sale-price units",
    )
    _coefficients = np.asarray(fitted_regressor.coefficients).reshape(
        grid_size, grid_size, len(dataset.features) + 1
    )
    for _ax, _feature in zip(_axes[1:], ("waterfront", "intercept"), strict=True):
        _index = (*dataset.features, "intercept").index(_feature)
        _surface = _coefficients[:, :, _index]
        _image = _ax.imshow(
            _surface,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            extent=(*HOUSE_LONGITUDE, *HOUSE_LATITUDE),
        )
        style_axes(
            _ax,
            title=f"{_feature.capitalize()} coefficient",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        housing_figure.colorbar(_image, ax=_ax, shrink=0.8)
    housing_figure
    return (housing_figure,)


@app.cell
def _(
    AUTHOR_CODE_URL,
    KING_COUNTY,
    asdict,
    chosen_solver,
    dataset,
    final_housing,
    folds,
    graph_scale,
    grid_size,
    housing_diagnostics,
    housing_elapsed,
    housing_summary,
    json,
    mo,
    multipliers,
    paper_mode,
    pd,
    platform,
    selected_multiplier,
    version,
):
    housing_manifest = {
        "experiment": "King County §5.2",
        "preset": "original-data paper method"
        if paper_mode
        else "synthetic adaptation",
        "provenance": dataset.provenance.as_dict(),
        "features": dataset.features,
        "seed": 17,
        "fold_positions_after_declared_preprocessing": [
            {"train": _train.tolist(), "validation": _validation.tolist()}
            for _train, _validation in folds
        ],
        "geographic_domains": {
            "latitude": [47.1559, 47.7776],
            "longitude": [-122.519, -121.646],
        },
        "grid_side_length": grid_size,
        "axis_order": ["latitude_bin", "longitude_bin"],
        "solver": {
            "name": type(chosen_solver).__name__,
            "configuration": asdict(chosen_solver),
        },
        "candidate_multipliers": multipliers,
        "selected_stratified_multiplier": selected_multiplier,
        "graph_multiplier": graph_scale.value,
        "summed_loss_base_penalties": {
            "local_l2": 0.001,
            "latitude_graph": 18.5,
            "longitude_graph": 18.5,
        },
        "local_penalty_convention": "lambda/2 * sum(theta^2), supported by Appendix A p28 and inspected author sum_squares_reg; section 5.2 does not restate r(theta)",
        "elapsed_seconds": housing_elapsed,
        "forest": {
            "trees": 100 if paper_mode else 30,
            "candidate_minimum_leaf_sizes": [1, 3, 8][: len(multipliers)],
            "n_jobs": 1,
        },
        "loss": "half mean squared log-price residual; summed unhalved objective penalties / (2 training fold N)",
        "versions": {
            "python": platform.python_version(),
            **{
                _package: version(_package)
                for _package in (
                    "stratified-models",
                    "numpy",
                    "jax",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "marimo",
                )
            },
        },
    }
    _ledger = pd.DataFrame(
        [
            {
                "Item": "Dataset",
                "This run": dataset.provenance.kind,
                "Deviation / interpretation": "Synthetic observations are an adaptation, never evidence of reproducing King County results.",
            },
            {
                "Item": "Geography",
                "This run": f"{grid_size} × {grid_size}; coupling × {graph_scale.value:g}",
                "Deviation / interpretation": "Original-data bins use reviewed fixed extents with a floating-point-inclusive upper boundary.",
            },
            {
                "Item": "Objective",
                "This run": "Paper penalties / (2 × training N)",
                "Deviation / interpretation": "Matches unhalved summed square loss to this implementation's half-mean square loss.",
            },
            {
                "Item": "Source script",
                "This run": "Paper specification takes priority",
                "Deviation / interpretation": "The script uses one holdout, 50 trees, different weights, and penalizes the intercept.",
            },
            {
                "Item": "Fold identity",
                "This run": "Five seeded folds in original-data mode",
                "Deviation / interpretation": "The original fold assignments were not published; exact reported RMSE is not promised.",
            },
            {
                "Item": "Parameter count",
                "This run": f"{grid_size**2 * (len(dataset.features) + 1):,}",
                "Deviation / interpretation": "Report the actual coefficient count; the paper's table count differs from its stated 50 × 50 × 10 setup.",
            },
            {
                "Item": "Selected setting",
                "This run": f"Penalty multiplier {selected_multiplier:g}",
                "Deviation / interpretation": "One setting evaluates published weights; three settings perform a small equal-count search.",
            },
        ]
    )
    mo.vstack(
        [
            mo.md("## Read the map alongside its provenance"),
            mo.md(
                "These are coefficients of the model refitted on all observations for inspection. The waterfront coefficient is per standardized unit; "
                "the intercept is evaluated at mean training covariates. Neither map is an isolated causal effect."
            ),
            mo.ui.table(_ledger, selection=None),
            mo.hstack(
                [
                    mo.download(
                        json.dumps(
                            housing_manifest, indent=2, allow_nan=False
                        ).encode(),
                        filename="king_county_run_manifest.json",
                        label="Download exact run settings and folds",
                    ),
                    mo.download(
                        housing_summary.to_csv(index=False).encode(),
                        filename="king_county_measured_scores.csv",
                        label="Download measured scores",
                    ),
                ]
            ),
            mo.accordion(
                {
                    "Every numerical fit": mo.ui.table(
                        pd.DataFrame(housing_diagnostics), selection=None
                    ),
                    "Final fitted model": mo.md(
                        f"Solver `{final_housing.diagnostics.solver}` · `{final_housing.diagnostics.termination}` · {final_housing.diagnostics.iterations} iterations"
                    ),
                    "Data attribution and checksum": mo.md(
                        f"[King County source]({KING_COUNTY.citation_url}) · [{KING_COUNTY.license_name}]({KING_COUNTY.license_url})  \n"
                        f"CSV SHA256: `{dataset.provenance.sha256}`  \n[Inspected author code]({AUTHOR_CODE_URL})  \n"
                        + "  \n".join(dataset.provenance.notes)
                    ),
                    "Runtime versions": mo.md(
                        f"Python `{platform.python_version()}` · NumPy `{version('numpy')}` · JAX `{version('jax')}` · scikit-learn `{version('scikit-learn')}`"
                    ),
                }
            ),
            mo.md(
                "**Try next:** increase grid resolution in the quick adaptation, then change smoothing. "
                "Watch the separate model's validation error and the coupled coefficient maps together. "
                "Use the solver notebook to inspect the numerical cost of a larger graph."
            ),
        ]
    )
    return (housing_manifest,)


if __name__ == "__main__":
    app.run()
