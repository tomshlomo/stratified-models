import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="Age, sex, and borrowing strength")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Age, sex, and borrowing strength

    **Review the classification API, then revisit a published experiment.** A separate
    logistic regression at every age and sex is flexible, but the data at each node can
    be scarce. A graph encourages adjacent ages to share information while retaining
    differences supported by observations.

    The default is a small, seeded, **synthetic adaptation**. It runs offline and contains
    no patient records. The original-data preset implements the statistical method in
    [Tuck, Barratt & Boyd (2021), §5.1, pp.18–19](https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf#page=18),
    with verified UCI data and five folds. Its results are computed here; published
    numbers are reference evidence, never substituted for a run.
    """)
    return


@app.cell(hide_code=True)
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
    from sklearn.model_selection import StratifiedKFold, train_test_split

    from _paper_data import (
        AUTHOR_CODE_URL,
        MESOTHELIOMA,
        DataIntegrityError,
        binary_metrics,
        load_mesothelioma,
        mean_penalty_weight,
        synthetic_mesothelioma,
    )
    from _support import diagnostics_summary, require_success, style_axes
    from stratified_models import (
        ADMM,
        L2,
        AddIntercept,
        Axis,
        BinaryLogisticSpec,
        FitFailure,
        Graph,
        Laplacian,
        NewtonSolver,
        Pipeline,
        Standardize,
        compose,
        fit,
    )

    return (
        ADMM,
        AUTHOR_CODE_URL,
        AddIntercept,
        Axis,
        BinaryLogisticSpec,
        DataIntegrityError,
        FitFailure,
        Graph,
        L2,
        Laplacian,
        MESOTHELIOMA,
        NewtonSolver,
        Path,
        Pipeline,
        Standardize,
        StratifiedKFold,
        asdict,
        binary_metrics,
        compose,
        diagnostics_summary,
        fit,
        json,
        load_mesothelioma,
        mean_penalty_weight,
        np,
        pd,
        platform,
        plt,
        require_success,
        style_axes,
        synthetic_mesothelioma,
        time,
        train_test_split,
        version,
    )


@app.cell(hide_code=True)
def _(mo):
    preset = mo.ui.dropdown(
        ["Quick · synthetic adaptation", "Paper method · original UCI data"],
        value="Quick · synthetic adaptation",
        label="Dataset and evaluation",
    )
    solver_choice = mo.ui.dropdown(
        ["Newton-CG", "ADMM"],
        value="Newton-CG",
        label="Explicit solver",
    )
    graph_scale = mo.ui.dropdown(
        {
            "No coupling": 0.0,
            "0.1 × paper coupling": 0.1,
            "Paper coupling": 1.0,
            "10 × paper coupling": 10.0,
        },
        value="Paper coupling",
        label="Graph strength",
    )
    candidate_budget = mo.ui.dropdown(
        ["One published setting per model", "Three settings per model"],
        value="One published setting per model",
        label="Equal search budget",
    )
    penalize_intercept = mo.ui.checkbox(
        value=True,
        label="Include the classification intercept in local L2",
    )
    allow_download = mo.ui.checkbox(
        value=False, label="Allow the verified UCI data download if uncached"
    )
    run_paper = mo.ui.run_button(label="Run original-data five-fold evaluation")
    mo.vstack(
        [
            mo.hstack([preset, solver_choice, graph_scale, candidate_budget]),
            mo.hstack([penalize_intercept, allow_download, run_paper]),
            mo.md(
                "Changing the original-data controls does not launch the full experiment until you press **Run**."
            ),
        ]
    )
    return (
        allow_download,
        candidate_budget,
        graph_scale,
        penalize_intercept,
        preset,
        run_paper,
        solver_choice,
    )


@app.cell(hide_code=True)
def _(
    DataIntegrityError,
    Path,
    allow_download,
    load_mesothelioma,
    mo,
    preset,
    run_paper,
    synthetic_mesothelioma,
):
    paper_mode = preset.value == "Paper method · original UCI data"
    if paper_mode:
        mo.stop(
            not run_paper.value,
            mo.callout(
                "Original-data evaluation is paused. Review the source and download setting, then press Run.",
                kind="info",
            ),
        )
        try:
            dataset = load_mesothelioma(
                cache_dir=Path(".cache/stratified-models/paper-data"),
                allow_download=allow_download.value,
            )
        except (FileNotFoundError, DataIntegrityError, ImportError) as _error:
            mo.stop(True, mo.callout(str(_error), kind="danger"))
    else:
        dataset = synthetic_mesothelioma()
    age_nodes = tuple(range(19, 86)) if paper_mode else tuple(range(30, 42))
    mo.vstack(
        [
            mo.callout(dataset.name, kind="info"),
            mo.md(
                f"**{len(dataset.frame):,} observations · {len(dataset.features)} covariates · {2 * len(age_nodes)} strata.** "
                "Age and sex select coefficients; they are not duplicated as regressors."
            ),
            mo.ui.table(dataset.frame.head(8), selection=None),
        ]
    )
    return age_nodes, dataset, paper_mode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The recipe makes the modeling choices visible

    Each model receives the same training observations and fold assignments. **Common**
    has one coefficient vector. **Separate** has age × sex vectors with no graph penalty.
    **Stratified** adds an age path and a single sex edge. Graph smoothing includes the
    intercept. The explicit mask control includes it in local L2 by default, giving a
    finite local optimum even when a node's observed outcomes all belong to one class.
    The paper does not explicitly resolve this classification mask; switch it off to
    measure that sensitivity. The house-price experiment explicitly excludes it.

    `Standardize` is a recipe, not a pre-fitted object: every call to `fit` learns its
    mean and population variance from that training fold. Original-data categorical
    domains are declared explicitly. The outcome and the leaking `diagnosis method`
    column never enter the feature list.

    The paper uses **summed** logistic loss. This package uses its global mean, so a
    published penalty weight becomes `weight / n_train`, separately in every fold.
    Both L2 and graph penalties use the same one-half convention as the paper.
    """)
    return


@app.cell(hide_code=True)
def _(
    AddIntercept,
    Axis,
    BinaryLogisticSpec,
    Graph,
    L2,
    Laplacian,
    Pipeline,
    Standardize,
    age_nodes,
    compose,
    dataset,
    graph_scale,
    mean_penalty_weight,
    penalize_intercept,
):
    def classification_recipe(
        model_name: str, *, n_train: int, multiplier: float
    ) -> Pipeline[BinaryLogisticSpec]:
        _features = (*dataset.features, "intercept")
        _local = {"Common": 0.004, "Separate": 12.9, "Stratified": 0.52}[model_name]
        _penalties: list[L2 | Laplacian] = [
            L2(
                name="local_l2",
                weight=mean_penalty_weight(
                    _local * multiplier, n_train=n_train, loss_multiplier=1.0
                ),
                features=_features if penalize_intercept.value else dataset.features,
            )
        ]
        _axes = ()
        if model_name != "Common":
            _sex = Axis(name="sex", nodes=("Male", "Female"))
            _age = Axis(name="age", nodes=age_nodes)
            _axes = (_sex, _age)
            if model_name == "Stratified":
                for _name, _axis, _weight in (
                    ("sex_graph", _sex, 10.0),
                    ("age_graph", _age, 500.0),
                ):
                    _penalties.append(
                        Laplacian(
                            name=_name,
                            graph=Graph.path(axis=_axis),
                            features=_features,
                            weight=mean_penalty_weight(
                                _weight * graph_scale.value * multiplier,
                                n_train=n_train,
                                loss_multiplier=1.0,
                            ),
                        )
                    )
        return Pipeline(
            preprocessing=compose(
                Standardize(columns=dataset.features), AddIntercept(name="intercept")
            ),
            model=BinaryLogisticSpec(
                features=_features, axes=_axes, penalties=tuple(_penalties)
            ),
        )

    return (classification_recipe,)


@app.cell(hide_code=True)
def _(
    ADMM,
    FitFailure,
    NewtonSolver,
    StratifiedKFold,
    binary_metrics,
    candidate_budget,
    classification_recipe,
    dataset,
    diagnostics_summary,
    fit,
    np,
    paper_mode,
    pd,
    require_success,
    solver_choice,
    time,
    train_test_split,
):
    _x, _y = dataset.x, dataset.y
    _rows = np.arange(len(_y))
    if paper_mode:
        folds = tuple(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=41).split(_rows, _y)
        )
    else:
        _train, _validation = train_test_split(
            _rows, test_size=0.3, stratify=_y, random_state=41
        )
        folds = ((_train, _validation),)
    multipliers = (
        (0.3, 1.0, 3.0)
        if candidate_budget.value == "Three settings per model"
        else (1.0,)
    )
    chosen_solver = (
        ADMM(max_iterations=2500, atol=1e-6, rtol=1e-5)
        if solver_choice.value == "ADMM"
        else NewtonSolver()
    )
    classification_rows = []
    classification_diagnostics = []
    _started = time.perf_counter()
    for _model_name in ("Common", "Separate", "Stratified"):
        for _multiplier in multipliers:
            for _fold, (_train, _validation) in enumerate(folds):
                _recipe = classification_recipe(
                    _model_name, n_train=len(_train), multiplier=_multiplier
                )
                _result = fit(
                    _recipe, _x.iloc[_train], _y[_train], solver=chosen_solver
                )
                classification_diagnostics.append(
                    {
                        "model": _model_name,
                        "multiplier": _multiplier,
                        "fold": _fold,
                        **diagnostics_summary(_result.diagnostics),
                    }
                )
                if isinstance(_result, FitFailure):
                    classification_rows.append(
                        {
                            "model": _model_name,
                            "multiplier": _multiplier,
                            "fold": _fold,
                            "anll": np.nan,
                            "error": np.nan,
                            "converged": False,
                            "n_train": len(_train),
                        }
                    )
                    continue
                _prediction = _result.predictor.decision_function(_x.iloc[_validation])
                _metrics = binary_metrics(_y[_validation], _prediction)
                classification_rows.append(
                    {
                        "model": _model_name,
                        "multiplier": _multiplier,
                        "fold": _fold,
                        **_metrics,
                        "converged": True,
                        "n_train": len(_train),
                    }
                )
    classification_results = pd.DataFrame(classification_rows)
    classification_summary = (
        classification_results.groupby(["model", "multiplier"], sort=False)
        .agg(
            anll=("anll", "mean"),
            fold_sd=("anll", "std"),
            error=("error", "mean"),
            successful_folds=("converged", "sum"),
        )
        .reset_index()
    )
    _complete = classification_summary[
        classification_summary["successful_folds"] == len(folds)
    ]
    if len(_complete[_complete["model"] == "Stratified"]) == 0:
        raise RuntimeError(
            "No stratified setting converged on every fold. Inspect the diagnostics and increase the explicit solve budget."
        )
    _best = _complete[_complete["model"] == "Stratified"].sort_values("anll").iloc[0]
    selected_multiplier = float(_best["multiplier"])
    final_classification = fit(
        classification_recipe(
            "Stratified", n_train=len(_y), multiplier=selected_multiplier
        ),
        _x,
        _y,
        solver=chosen_solver,
    )
    fitted_classifier = require_success(final_classification)
    classification_diagnostics.append(
        {
            "model": "Stratified · final refit",
            "multiplier": selected_multiplier,
            "fold": "all",
            **diagnostics_summary(final_classification.diagnostics),
        }
    )
    classification_elapsed = time.perf_counter() - _started
    assert "diagnosis method" not in dataset.features
    assert "target" not in dataset.features
    assert set(np.unique(_y)).issubset({0.0, 1.0})
    return (
        chosen_solver,
        classification_diagnostics,
        classification_elapsed,
        classification_results,
        classification_summary,
        fitted_classifier,
        folds,
        multipliers,
        selected_multiplier,
    )


@app.cell(hide_code=True)
def _(classification_elapsed, classification_summary, folds, mo, paper_mode, pd):
    _best_rows = (
        classification_summary[classification_summary["successful_folds"] == len(folds)]
        .sort_values("anll")
        .groupby("model", sort=False)
        .head(1)
    )
    mo.vstack(
        [
            mo.md(
                f"## Measured results\n\nCompleted in **{classification_elapsed:.2f} s**. "
                + (
                    "Original-data five-fold evaluation."
                    if paper_mode
                    else "Synthetic single-holdout quick evaluation."
                )
            ),
            mo.ui.table(_best_rows, selection=None),
            mo.accordion(
                {
                    "All candidate settings and successful fold counts": mo.ui.table(
                        classification_summary, selection=None
                    )
                }
            ),
            mo.callout(
                "The table reports measured validation scores. With multiple candidates, these same folds select the settings; "
                "the selected score is not an independent estimate after tuning. Fold SD describes fold variation, not a confidence interval.",
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    age_nodes,
    classification_summary,
    dataset,
    fitted_classifier,
    folds,
    np,
    plt,
    style_axes,
):
    classification_figure, _axes = plt.subplots(
        1, 2, figsize=(12, 4.2), layout="constrained"
    )
    _best = (
        classification_summary[classification_summary["successful_folds"] == len(folds)]
        .sort_values("anll")
        .groupby("model", sort=False)
        .head(1)
    )
    _axes[0].bar(
        _best["model"],
        _best["anll"],
        yerr=_best["fold_sd"].fillna(0),
        color=["#8595a8", "#e49b54", "#4778bd"],
        capsize=4,
    )
    style_axes(
        _axes[0],
        title="Validation loss · lower is better",
        ylabel="Mean negative log likelihood",
    )
    _coefficient = np.asarray(fitted_classifier.coefficients).reshape(
        2, len(age_nodes), len(dataset.features) + 1
    )
    _glucose = dataset.features.index("glucose")
    for _index, (_sex, _color) in enumerate(
        (("Male", "#4778bd"), ("Female", "#c77891"))
    ):
        _axes[1].plot(
            age_nodes,
            _coefficient[_index, :, _glucose],
            color=_color,
            label=_sex,
            linewidth=2,
        )
    style_axes(
        _axes[1],
        title="Glucose coefficient · model refitted on all rows",
        xlabel="Age",
        ylabel="Coefficient per training-standardized unit",
    )
    _axes[1].legend(frameon=False)
    classification_figure
    return (classification_figure,)


@app.cell(hide_code=True)
def _(
    AUTHOR_CODE_URL,
    MESOTHELIOMA,
    age_nodes,
    asdict,
    chosen_solver,
    classification_diagnostics,
    classification_elapsed,
    classification_summary,
    dataset,
    folds,
    graph_scale,
    json,
    mo,
    multipliers,
    paper_mode,
    penalize_intercept,
    pd,
    platform,
    selected_multiplier,
    version,
):
    classification_manifest = {
        "experiment": "mesothelioma §5.1",
        "preset": "original-data paper method"
        if paper_mode
        else "synthetic adaptation",
        "provenance": dataset.provenance.as_dict(),
        "features": dataset.features,
        "axes": {"sex": ["Male", "Female"], "age": list(age_nodes)},
        "seed": 41,
        "fold_positions_after_declared_preprocessing": [
            {"train": _train.tolist(), "validation": _validation.tolist()}
            for _train, _validation in folds
        ],
        "solver": {
            "name": type(chosen_solver).__name__,
            "configuration": asdict(chosen_solver),
        },
        "candidate_multipliers": multipliers,
        "selected_stratified_multiplier": selected_multiplier,
        "graph_multiplier": graph_scale.value,
        "local_l2_includes_intercept": penalize_intercept.value,
        "summed_loss_base_penalties": {
            "Common": {"local_l2": 0.004},
            "Separate": {"local_l2": 12.9},
            "Stratified": {"local_l2": 0.52, "sex_graph": 10.0, "age_graph": 500.0},
        },
        "elapsed_seconds": classification_elapsed,
        "loss": "mean binary logistic; summed-objective penalty weights / training fold N",
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
                "Item": "Data",
                "This run": dataset.provenance.kind,
                "Interpretation": "Synthetic mode is an adaptation; original mode verifies the exact XLSX checksum.",
            },
            {
                "Item": "Splits",
                "This run": "Seed 41; stratified folds",
                "Interpretation": "Original fold assignments were not published; exact table matching is not promised.",
            },
            {
                "Item": "Objective",
                "This run": "Mean logistic; weights / training N",
                "Interpretation": "Equivalent objective units to the paper's summed loss.",
            },
            {
                "Item": "Author script",
                "This run": "Follow paper specification",
                "Interpretation": "The script uses a 10% holdout and omits the paper's local L2 penalty.",
            },
            {
                "Item": "Coefficient representation",
                "This run": "One binary log-odds vector",
                "Interpretation": "The source implementation uses two softmax vectors. Do not copy its L2/graph weights without the factor-of-two conversion.",
            },
            {
                "Item": "Classification intercept mask",
                "This run": "Included in local L2"
                if penalize_intercept.value
                else "Excluded from local L2",
                "Interpretation": "The paper is ambiguous here. Inclusion gives finite local optima for single-class nodes; the control exposes the sensitivity.",
            },
            {
                "Item": "Search",
                "This run": f"Selected multiplier {selected_multiplier:g}",
                "Interpretation": "All three families have the same number of candidate settings and the same folds.",
            },
        ]
    )
    mo.vstack(
        [
            mo.md("## Reproduction ledger and solver evidence"),
            mo.ui.table(_ledger, selection=None),
            mo.hstack(
                [
                    mo.download(
                        json.dumps(
                            classification_manifest, indent=2, allow_nan=False
                        ).encode(),
                        filename="mesothelioma_run_manifest.json",
                        label="Download exact run settings and folds",
                    ),
                    mo.download(
                        classification_summary.to_csv(index=False).encode(),
                        filename="mesothelioma_measured_scores.csv",
                        label="Download measured scores",
                    ),
                ]
            ),
            mo.accordion(
                {
                    "Every fit's numerical diagnostics": mo.ui.table(
                        pd.DataFrame(classification_diagnostics), selection=None
                    ),
                    "Source, license, and exact data fingerprint": mo.md(
                        f"[UCI source]({MESOTHELIOMA.citation_url}) · [{MESOTHELIOMA.license_name}]({MESOTHELIOMA.license_url})  \n"
                        f"Raw XLSX SHA256: `{dataset.provenance.sha256}`  \n"
                        f"[Inspected author code]({AUTHOR_CODE_URL})  \n"
                        + "  \n".join(dataset.provenance.notes)
                    ),
                    "Runtime versions": mo.md(
                        f"Python `{platform.python_version()}` · NumPy `{version('numpy')}` · JAX `{version('jax')}` · marimo `{version('marimo')}`"
                    ),
                }
            ),
            mo.md(
                "**Try next:** vary coupling while holding the solver fixed; compare the validation score with the smoothness of the age curves. "
                "Then switch solvers and inspect convergence evidence. A visually smooth curve alone does not establish predictive value."
            ),
        ]
    )
    return (classification_manifest,)


if __name__ == "__main__":
    app.run()
