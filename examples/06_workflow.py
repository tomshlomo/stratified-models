"""A composable classification workflow from fitting through a checked artifact."""

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="full",
    app_title="From a fitted pipeline to a reusable artifact",
)


@app.cell
def _():
    import hashlib
    import json
    import tempfile
    from collections.abc import Mapping
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import (
        ConfusionMatrixDisplay, average_precision_score,
        precision_recall_curve, roc_auc_score, roc_curve,
    )

    from _support import (
        COLORS, OffsetFeature, calibration_bins, classification_metrics,
        diagnostics_summary, package_versions, plot_style, require_success,
        style_axes, synthetic_classification,
    )
    from stratified_models import (
        ADMM, L1, L2, AddIntercept, Axis, BinaryLogisticSpec,
        FitFailure, Graph, Laplacian, Pipeline, Standardize,
        compose, fit,
    )
    from stratified_models.artifacts import (
        CodecRegistry, ComponentCodec, load_binary_classifier, save_artifact,
    )
    from stratified_models.data import from_pandas
    from stratified_models.errors import ArtifactError, StratifiedModelsError
    from stratified_models.sklearn import StratifiedBinaryClassifier
    from stratified_models.tuning import (
        BinaryLogLoss, Fold, LogUniform, Minimize, TrialBudget,
        TuningSuccess, WeightSpace, tune,
    )
    return (
        ADMM,
        AddIntercept,
        ArtifactError,
        Axis,
        BinaryLogLoss,
        BinaryLogisticSpec,
        COLORS,
        CodecRegistry,
        ComponentCodec,
        ConfusionMatrixDisplay,
        FitFailure,
        Fold,
        Graph,
        L1,
        L2,
        Laplacian,
        LogUniform,
        Mapping,
        Minimize,
        OffsetFeature,
        Path,
        Pipeline,
        Standardize,
        StratifiedBinaryClassifier,
        StratifiedModelsError,
        TrialBudget,
        TuningSuccess,
        WeightSpace,
        average_precision_score,
        calibration_bins,
        classification_metrics,
        clone,
        compose,
        diagnostics_summary,
        fit,
        from_pandas,
        hashlib,
        json,
        load_binary_classifier,
        mo,
        np,
        package_versions,
        pd,
        plot_style,
        plt,
        precision_recall_curve,
        require_success,
        roc_auc_score,
        roc_curve,
        save_artifact,
        style_axes,
        synthetic_classification,
        tempfile,
        tune,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # From a fitted pipeline to a reusable artifact

    **Binary probabilities · explicit tuning · sklearn · immutable prediction state**

    This notebook follows one complete experiment: compose preprocessing,
    fit a classifier, inspect held-out probabilities, select weights on
    training-only folds, and save a versioned predictor. It also shows the
    deliberate boundary between invalid input exceptions and unsuccessful
    numerical fits.

    The default run is offline and seeded. It fits a small model and verifies
    artifact and sklearn round trips. Hyperparameter search is an explicit,
    separately submitted operation; the final test snapshot is never used
    to select its weights.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {"Public API: handle a fitted outcome and persist prediction state": mo.md(
            "```python\n"
            "result = fit(recipe, X_train, y_train, solver=ADMM())\n"
            "match result:\n"
            "    case FitSuccess(predictor=classifier, diagnostics=info):\n"
            "        probabilities = classifier.predict_proba(X_test)\n"
            "        save_artifact(classifier, 'artifacts/classifier')\n"
            "        restored = load_binary_classifier('artifacts/classifier')\n"
            "    case FitFailure(reason=reason, diagnostics=info):\n"
            "        report_failure(reason, info)\n\n"
            "```\n\n"
            "The task-specific loader returns a classifier with its probability "
            "interface. The workflow below executes these operations using a temporary artifact directory."
        )}
    )
    return


@app.cell
def _(mo):
    workflow_graph_weight = mo.ui.slider(
        start=-4.0, stop=-1.0, step=0.5, value=-2.0,
        label="log₁₀ graph weight", show_value=True,
    )
    mo.hstack([workflow_graph_weight], justify="start")
    return (workflow_graph_weight,)


@app.cell
def _(
    ADMM,
    AddIntercept,
    Axis,
    BinaryLogisticSpec,
    Graph,
    L1,
    L2,
    Laplacian,
    Pipeline,
    Standardize,
    compose,
    fit,
    require_success,
    synthetic_classification,
    workflow_graph_weight,
):
    workflow_x, workflow_y, final_test_x, final_test_y = synthetic_classification(seed=71)
    original_workflow_x = workflow_x.copy(deep=True)
    workflow_axis = Axis(name="group", nodes=tuple(range(5)))
    workflow_features = ("x", "noise", "intercept")
    workflow_recipe = Pipeline(
        preprocessing=compose(
            Standardize(columns=("x", "noise")),
            AddIntercept(name="intercept"),
        ),
        model=BinaryLogisticSpec(
            features=workflow_features,
            axes=(workflow_axis,),
            penalties=(
                L1(name="sparsity", weight=0.004, features=("x", "noise")),
                L2(name="shrinkage", weight=0.02, features=workflow_features),
                Laplacian(
                    name="smoothing", weight=10.0 ** workflow_graph_weight.value,
                    graph=Graph.path(axis=workflow_axis), features=workflow_features,
                ),
            ),
        ),
    )
    workflow_solver = ADMM(max_iterations=1600, atol=2e-6, rtol=2e-5)
    workflow_result = fit(workflow_recipe, workflow_x, workflow_y, solver=workflow_solver)
    classifier = require_success(workflow_result)
    held_out_probabilities = classifier.predict_proba(final_test_x)
    return (
        classifier,
        final_test_x,
        final_test_y,
        held_out_probabilities,
        original_workflow_x,
        workflow_recipe,
        workflow_result,
        workflow_solver,
        workflow_x,
        workflow_y,
    )


@app.cell
def _(
    average_precision_score,
    classification_metrics,
    diagnostics_summary,
    final_test_y,
    held_out_probabilities,
    mo,
    pd,
    roc_auc_score,
    workflow_result,
):
    probability_metrics = {
        **classification_metrics(final_test_y, held_out_probabilities),
        "ROC AUC": roc_auc_score(final_test_y, held_out_probabilities[:, 1]),
        "Average precision": average_precision_score(final_test_y, held_out_probabilities[:, 1]),
    }
    mo.vstack(
        [
            mo.md("## Inspect probabilities, not only classification accuracy"),
            mo.ui.table(pd.DataFrame([probability_metrics]).round(5), selection=None, pagination=False),
            mo.accordion({"Fit certificate and computation": mo.ui.table(pd.DataFrame([diagnostics_summary(workflow_result.diagnostics)]), selection=None, pagination=False)}),
            mo.md(
                "Log loss and Brier score evaluate probabilities. ROC AUC and average "
                "precision evaluate ranking. A probability model can rank observations "
                "well and still be poorly calibrated; these are separate questions."
            ),
        ]
    )
    return (probability_metrics,)


@app.cell
def _(
    COLORS,
    calibration_bins,
    final_test_y,
    held_out_probabilities,
    np,
    plot_style,
    plt,
    precision_recall_curve,
    roc_curve,
    style_axes,
):
    _positive = held_out_probabilities[:, 1]
    _false_positive, _true_positive, _ = roc_curve(final_test_y, _positive)
    _precision, _recall, _ = precision_recall_curve(final_test_y, _positive)
    calibration_table = calibration_bins(final_test_y, _positive)
    with plot_style():
        probability_figure, _axes = plt.subplots(1, 3, figsize=(14, 3.8))
        _axes[0].plot(_false_positive, _true_positive, color=COLORS["stratified"])
        _axes[0].plot([0, 1], [0, 1], "--", color=COLORS["reference"], alpha=0.65)
        style_axes(_axes[0], title="Discrimination", xlabel="False-positive rate", ylabel="True-positive rate")
        _axes[1].plot(_recall, _precision, color=COLORS["stratified"])
        _axes[1].axhline(float(np.mean(final_test_y)), ls="--", color=COLORS["reference"], label="Test prevalence")
        style_axes(_axes[1], title="Precision and recall", xlabel="Recall", ylabel="Precision")
        _axes[1].legend(fontsize=8)
        _fraction = calibration_table["observed fraction"].to_numpy()
        _count = calibration_table["count"].to_numpy()
        _z = 1.96
        _denominator = 1 + _z ** 2 / _count
        _center = (_fraction + _z ** 2 / (2 * _count)) / _denominator
        _half_width = _z * np.sqrt(_fraction * (1 - _fraction) / _count + _z ** 2 / (4 * _count ** 2)) / _denominator
        _axes[2].plot([0, 1], [0, 1], "--", color=COLORS["reference"], alpha=0.65)
        _axes[2].errorbar(
            calibration_table["mean probability"], _fraction,
            yerr=np.vstack((_fraction - (_center - _half_width), (_center + _half_width) - _fraction)),
            fmt="o", color=COLORS["stratified"], capsize=3,
        )
        _axes[2].set(xlim=(0, 1), ylim=(0, 1))
        style_axes(_axes[2], title="Calibration · binned Wilson intervals", xlabel="Mean predicted probability", ylabel="Observed positive fraction")
        probability_figure.tight_layout()
    probability_figure
    return


@app.cell
def _(mo):
    threshold_control = mo.ui.slider(
        start=0.1, stop=0.9, step=0.05, value=0.5,
        label="Illustrative decision threshold", show_value=True,
    )
    mo.vstack(
        [
            mo.md(
                "## Decision thresholds do not require refitting\n\n"
                "Move this threshold to see decision counts change while fitted "
                "probabilities remain fixed. Choose an operational threshold using "
                "validation data and application costs, not by optimizing this final test plot."
            ),
            threshold_control,
        ]
    )
    return (threshold_control,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    classifier,
    final_test_x,
    final_test_y,
    plot_style,
    plt,
    threshold_control,
):
    threshold_labels = classifier.predict(final_test_x, threshold=threshold_control.value)
    with plot_style():
        threshold_figure, _threshold_ax = plt.subplots(figsize=(4.5, 3.7))
        ConfusionMatrixDisplay.from_predictions(
            final_test_y, threshold_labels, ax=_threshold_ax,
            colorbar=False, cmap="Blues", display_labels=("Negative", "Positive"),
        )
        _threshold_ax.set_title(f"Threshold = {threshold_control.value:.2f}", loc="left", pad=12)
        threshold_figure.tight_layout()
    threshold_figure
    return


@app.cell
def _(mo):
    tuning_form = mo.ui.dictionary(
        {"trials": mo.ui.dropdown(options=[2, 6, 20], value=2, label="Explicit trial budget")}
    ).form(submit_button_label="Run fold-isolated weight selection")
    mo.vstack(
        [
            mo.md(
                "## Optional tuning is an explicit experiment\n\n"
                "The search receives raw train/validation sources. Every fold fits "
                "its own standardizer and owns a separate warm-start session. "
                "Numerical caches can be shared only when their dependencies match. "
                "The metric pools all validation rows; unequal fold sizes are not "
                "silently averaged. A failed numerical trial is reported."
            ),
            tuning_form,
        ]
    )
    return (tuning_form,)


@app.cell
def _(
    BinaryLogLoss,
    Fold,
    LogUniform,
    Minimize,
    TrialBudget,
    TuningSuccess,
    WeightSpace,
    from_pandas,
    mo,
    np,
    pd,
    tune,
    tuning_form,
    workflow_recipe,
    workflow_solver,
    workflow_x,
    workflow_y,
):
    if tuning_form.value is None:
        tuning_output = mo.callout("No search has been run. The default model above uses its explicitly supplied weights.", kind="neutral")
        tuning_result = None
    else:
        _assignment = np.arange(len(workflow_x)) % 2
        _folds = tuple(
            Fold(
                train=from_pandas(workflow_x.iloc[_assignment != _fold], workflow_y.iloc[_assignment != _fold], batch_size=64),
                validation=from_pandas(workflow_x.iloc[_assignment == _fold], workflow_y.iloc[_assignment == _fold], batch_size=64),
            )
            for _fold in range(2)
        )
        tuning_result = tune(
            workflow_recipe,
            folds=_folds,
            search=WeightSpace(
                ranges={
                    workflow_recipe.weight("sparsity"): LogUniform(low=1e-4, high=0.03),
                    workflow_recipe.weight("smoothing"): LogUniform(low=1e-4, high=0.1),
                }
            ),
            metric=Minimize(metric=BinaryLogLoss()),
            solver=workflow_solver,
            budget=TrialBudget(trials=int(tuning_form.value["trials"])),
            seed=71,
        )
        _trial_rows = [
            {"trial": report.number, "status": report.status, "validation log loss": report.score, **report.weights}
            for report in tuning_result.reports
        ]
        if isinstance(tuning_result, TuningSuccess):
            _message = f"Selected validation log loss: **{tuning_result.best_score:.5f}**. The result contains a new immutable recipe. Final refitting is a separate explicit operation."
        else:
            _message = f"The tuning operation returned a typed failure: `{tuning_result.reason}`."
        tuning_output = mo.vstack(
            [
                mo.md(_message),
                mo.ui.table(pd.DataFrame(_trial_rows), selection=None, pagination=False),
            ]
        )
    tuning_output
    return


@app.cell
def _(
    StratifiedBinaryClassifier,
    classification_metrics,
    classifier,
    clone,
    final_test_x,
    final_test_y,
    mo,
    np,
    pd,
    workflow_recipe,
    workflow_solver,
    workflow_x,
    workflow_y,
):
    adapter_template = StratifiedBinaryClassifier(recipe=workflow_recipe, solver=workflow_solver)
    sklearn_estimator = clone(adapter_template)
    _named_labels = np.where(workflow_y.to_numpy() == 1, "positive", "negative")
    sklearn_estimator.fit(workflow_x, _named_labels)
    sklearn_probabilities = sklearn_estimator.predict_proba(final_test_x)
    sklearn_matches_native = bool(np.allclose(sklearn_probabilities, classifier.predict_proba(final_test_x), atol=2e-5, rtol=2e-5))
    mo.vstack(
        [
            mo.md(
                "## sklearn compatibility is a separate adapter\n\n"
                "The adapter follows sklearn's mutable `fit` convention and maps "
                "arbitrary binary labels to the native zero/one task. The recipe "
                "and fitted native predictor remain immutable. `clone` creates an "
                "independent estimator; it does not share fitted preprocessing."
            ),
            mo.ui.table(
                pd.DataFrame([{
                    "classes": str(tuple(sklearn_estimator.classes_)),
                    "unfitted template remains unfitted": not hasattr(adapter_template, "predictor_"),
                    "probabilities match native fit": sklearn_matches_native,
                    **classification_metrics(final_test_y, sklearn_probabilities),
                }]).round(6),
                selection=None, pagination=False,
            ),
        ]
    )
    return (sklearn_matches_native,)


@app.cell
def _(
    Path,
    classifier,
    final_test_x,
    hashlib,
    json,
    load_binary_classifier,
    mo,
    np,
    save_artifact,
    tempfile,
):
    artifact_directory = tempfile.TemporaryDirectory(prefix="stratified-artifact-")
    artifact_path = Path(artifact_directory.name) / "classifier"
    save_artifact(classifier, artifact_path)
    restored_classifier = load_binary_classifier(artifact_path)
    manifest = json.loads((artifact_path / "manifest.json").read_text())
    artifact_prediction_error = float(np.max(np.abs(restored_classifier.predict_proba(final_test_x) - classifier.predict_proba(final_test_x))))
    artifact_checksum_matches = hashlib.sha256((artifact_path / "arrays.npz").read_bytes()).hexdigest() == manifest["payload"]["sha256"]
    _manifest_view = {
        "format": manifest["format"],
        "version": manifest["version"],
        "task": manifest["model"]["task"],
        "weights": manifest["weights"],
        "array metadata": manifest["arrays"],
        "payload checksum verified": artifact_checksum_matches,
        "maximum probability round-trip error": artifact_prediction_error,
    }
    mo.vstack(
        [
            mo.md(
                "## A fitted predictor is a portable prediction artifact\n\n"
                "The directory contains a versioned JSON manifest and numerical arrays. "
                "It records feature/axis order and fitted transform parameters. "
                "Task-specific loading returns a classifier, preserving its typed "
                "probability interface. Sources, training rows, caches, and warm starts "
                "are not part of this inference artifact."
            ),
            mo.md("```json\n" + json.dumps(_manifest_view, indent=2) + "\n```"),
        ]
    )
    return artifact_checksum_matches, artifact_prediction_error


@app.cell
def _(
    AddIntercept,
    ArtifactError,
    CodecRegistry,
    ComponentCodec,
    Mapping,
    OffsetFeature,
    Path,
    Pipeline,
    Standardize,
    compose,
    final_test_x,
    fit,
    load_binary_classifier,
    mo,
    np,
    require_success,
    save_artifact,
    tempfile,
    workflow_recipe,
    workflow_solver,
    workflow_x,
    workflow_y,
):
    def _encode_offset(value: OffsetFeature) -> dict[str, str | float]:
        return {"column": value.column, "offset": value.offset}

    def _decode_offset(state: Mapping[str, object]) -> OffsetFeature:
        _column = state["column"]
        _offset = state["offset"]
        if not isinstance(_column, str) or not isinstance(_offset, (int, float)):
            raise ValueError("Invalid offset component state.")
        return OffsetFeature(column=_column, offset=float(_offset))

    custom_registry = CodecRegistry(
        codecs=(ComponentCodec(
            name="notebooks.offset-feature.v1", component_type=OffsetFeature,
            encode=_encode_offset, decode=_decode_offset,
        ),)
    )
    custom_recipe = Pipeline(
        preprocessing=compose(
            OffsetFeature(column="x", offset=0.25),
            Standardize(columns=("x", "noise")),
            AddIntercept(name="intercept"),
        ),
        model=workflow_recipe.model,
    )
    custom_result = fit(custom_recipe, workflow_x, workflow_y, solver=workflow_solver)
    custom_classifier = require_success(custom_result)
    custom_directory = tempfile.TemporaryDirectory(prefix="stratified-custom-codec-")
    custom_path = Path(custom_directory.name) / "classifier"
    save_artifact(custom_classifier, custom_path, registry=custom_registry)
    try:
        load_binary_classifier(custom_path)
    except ArtifactError:
        missing_codec_rejected = True
    else:
        missing_codec_rejected = False
    restored_custom = load_binary_classifier(custom_path, registry=custom_registry)
    custom_round_trip = bool(np.allclose(custom_classifier.predict_proba(final_test_x), restored_custom.predict_proba(final_test_x), atol=1e-13, rtol=1e-13))
    mo.vstack(
        [
            mo.md(
                "## External components compose and serialize explicitly\n\n"
                "`OffsetFeature` is a tiny external fitted-transform implementation: "
                "it shifts one column and preserves rows and targets. It participates "
                "through the transform protocol without inheriting a library class. "
                "Its codec stores only a column name and finite offset. Loading "
                "requires the explicitly supplied registry; artifact metadata does "
                "not import or install executable components."
            ),
            mo.callout(
                f"Missing registry rejected: **{missing_codec_rejected}** · "
                f"custom-component prediction round trip: **{custom_round_trip}**",
                kind="success" if missing_codec_rejected and custom_round_trip else "danger",
            ),
        ]
    )
    return custom_round_trip, missing_codec_rejected


@app.cell
def _(
    ADMM,
    FitFailure,
    StratifiedModelsError,
    classifier,
    final_test_x,
    fit,
    mo,
    np,
    pd,
    workflow_recipe,
    workflow_x,
    workflow_y,
):
    _invalid_inputs = {
        "Missing regression feature": final_test_x.drop(columns="x"),
        "Unknown stratum": final_test_x.assign(group=99),
        "Nonfinite feature": final_test_x.assign(x=np.nan),
    }
    _error_rows = []
    for _description, _invalid_frame in _invalid_inputs.items():
        try:
            classifier.predict_proba(_invalid_frame)
        except StratifiedModelsError as _error:
            _error_rows.append({"input": _description, "exception": type(_error).__name__, "message": str(_error)})
        else:
            raise AssertionError(f"Invalid input was silently accepted: {_description}")
    budget_failure = fit(workflow_recipe, workflow_x, workflow_y, solver=ADMM(max_iterations=1))
    assert isinstance(budget_failure, FitFailure)
    mo.vstack(
        [
            mo.md(
                "## Errors and unsuccessful solves have different contracts\n\n"
                "Invalid configuration, schema, data, or replay raises a specific "
                "exception as soon as it is encountered. A valid optimization that "
                "exhausts its budget returns the typed failure branch with diagnostics. "
                "Neither path silently returns a plausible-looking successful model."
            ),
            mo.ui.table(pd.DataFrame(_error_rows), selection=None, pagination=False),
            mo.callout(f"Deliberate one-iteration solve: `{budget_failure.reason}`", kind="warn"),
        ]
    )
    return


@app.cell
def _(
    artifact_checksum_matches,
    artifact_prediction_error,
    classifier,
    custom_round_trip,
    final_test_x,
    held_out_probabilities,
    missing_codec_rejected,
    np,
    original_workflow_x,
    sklearn_matches_native,
    workflow_result,
    workflow_x,
):
    _second_prediction = classifier.predict_proba(final_test_x)
    review_checks = {
        "native optimization converged": workflow_result.diagnostics.converged,
        "probability rows sum to one": bool(np.allclose(held_out_probabilities.sum(axis=1), 1.0)),
        "repeated prediction is stable": bool(np.array_equal(held_out_probabilities, _second_prediction)),
        "fitting preserved input frame": workflow_x.equals(original_workflow_x),
        "coefficient storage is read-only": not classifier.coefficients.flags.writeable,
        "builtin artifact round trip preserves probabilities": artifact_prediction_error < 1e-12,
        "artifact array checksum matches": artifact_checksum_matches,
        "custom codec round trip preserves probabilities": custom_round_trip,
        "custom codec must be supplied explicitly": missing_codec_rejected,
        "sklearn wrapper matches native probabilities": sklearn_matches_native,
    }
    assert all(review_checks.values()), review_checks
    return (review_checks,)


@app.cell
def _(json, mo, package_versions, probability_metrics, review_checks):
    workflow_record = {
        "seed": 71,
        "task": "binary logistic; global mean loss; explicit L1/L2/graph penalties",
        "evaluation": "independent synthetic test snapshot; tuning uses training-only folds",
        "versions": package_versions(),
        "test metrics": probability_metrics,
        "checks": review_checks,
    }
    mo.accordion({"Reproducibility and executable checks": mo.md("```json\n" + json.dumps(workflow_record, indent=2) + "\n```")})
    return


if __name__ == "__main__":
    app.run()
