"""An interactive introduction to graph-regularized stratified regression."""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", app_title="Borrowing strength across a graph")


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from _support import (
        COLORS,
        diagnostics_summary,
        package_versions,
        plot_style,
        regression_metrics,
        require_success,
        style_axes,
        synthetic_regression,
    )
    from stratified_models import (
        ADMM,
        L1,
        L2,
        AddIntercept,
        Axis,
        Graph,
        Laplacian,
        Pipeline,
        RegressionSpec,
        compose,
        fit,
    )
    return (
        ADMM,
        AddIntercept,
        Axis,
        COLORS,
        Graph,
        L1,
        L2,
        Laplacian,
        Pipeline,
        RegressionSpec,
        compose,
        diagnostics_summary,
        fit,
        json,
        mo,
        np,
        package_versions,
        pd,
        plot_style,
        plt,
        regression_metrics,
        require_success,
        style_axes,
        synthetic_regression,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Borrowing strength across a graph

    **A complete regression experiment · CPU · seeded synthetic data · ADMM**

    A stratified model gives each group its own regression coefficients.
    Graph regularization encourages connected groups to agree. This sits
    between fitting one model for everybody and fitting every group independently.

    This experiment makes the trade-off visible: the true coefficient field is
    known, one group has **no training observations**, and every method is
    evaluated on the same independent test snapshot.

    Explore the graph and its strength, inspect the estimated coefficients,
    and distinguish predictive accuracy from coefficient smoothing and sparsity.
    """)
    return


@app.cell
def _(graph_weight, l1_weight, l2_weight, mo):
    _api_example = f'''features = ("x", "noise", "intercept")
    recipe = Pipeline(
    preprocessing=compose(AddIntercept(name="intercept")),
    model=RegressionSpec(
        features=features,
        axes=(selected_graph.axis,),
        penalties=(
            L1(name="sparsity", weight={l1_weight:.6g}, features=("x", "noise")),
            L2(name="shrinkage", weight={l2_weight:.6g}, features=features),
            Laplacian(name="smoothing", weight={graph_weight:.6g},
                      graph=selected_graph, features=features),
        ),
    ),
    )
    result = fit(recipe, train_x, train_y, solver=ADMM())'''
    mo.accordion(
        {"Public API: an explicit composition and solver": mo.md(
            "The selected graph is defined by the control below. The model and "
            "transform recipes are immutable. A fit returns a typed success or failure.\n\n"
            "```python\n" + _api_example + "\n```"
        )}
    )
    return


@app.cell
def _(mo):
    graph_choice = mo.ui.dropdown(
        options=["Path", "Ring", "Deliberately mismatched"],
        value="Ring",
        label="Relationships between groups",
    )
    smoothing_log10 = mo.ui.slider(
        start=-4.0,
        stop=0.0,
        step=0.5,
        value=-2.0,
        label="log₁₀ graph weight",
        show_value=True,
    )
    sparsity_log10 = mo.ui.slider(
        start=-4.0,
        stop=-0.5,
        step=0.5,
        value=-2.5,
        label="log₁₀ L1 weight",
        show_value=True,
    )
    sparsity_enabled = mo.ui.checkbox(value=False, label="Include an L1 penalty")
    mo.hstack(
        [
            mo.vstack([graph_choice, sparsity_enabled]),
            mo.vstack([smoothing_log10, sparsity_log10]),
        ],
        justify="start",
        gap=3,
    )
    return graph_choice, smoothing_log10, sparsity_enabled, sparsity_log10


@app.cell
def _(synthetic_regression):
    train_x, train_y, test_x, test_y, true_coefficients = synthetic_regression()
    untouched_train = train_x.copy(deep=True)
    training_counts = train_x.groupby("group").size().reindex(range(8), fill_value=0)
    return (
        test_x,
        test_y,
        train_x,
        train_y,
        training_counts,
        true_coefficients,
        untouched_train,
    )


@app.cell
def _(
    ADMM,
    AddIntercept,
    Axis,
    Graph,
    L1,
    L2,
    Laplacian,
    Pipeline,
    RegressionSpec,
    compose,
    fit,
    graph_choice,
    require_success,
    smoothing_log10,
    sparsity_enabled,
    sparsity_log10,
    train_x,
    train_y,
):
    group_axis = Axis(name="group", nodes=tuple(range(8)))
    if graph_choice.value == "Path":
        selected_graph = Graph.path(axis=group_axis)
    elif graph_choice.value == "Ring":
        selected_graph = Graph(
            axis=group_axis,
            edges=tuple((i, (i + 1) % 8, 1.0) for i in range(8)),
        )
    else:
        _order = (0, 4, 1, 5, 2, 6, 3, 7)
        selected_graph = Graph(
            axis=group_axis,
            edges=tuple((_order[i], _order[(i + 1) % 8], 1.0) for i in range(8)),
        )

    graph_weight = 10.0 ** smoothing_log10.value
    l1_weight = 10.0 ** sparsity_log10.value if sparsity_enabled.value else 0.0
    l2_weight = 0.004
    regression_features = ("x", "noise", "intercept")

    def _recipe(*, pooled: bool, smooth: bool):
        _multiplier = 8 if pooled else 1
        _penalties = (
            L1(name="sparsity", weight=l1_weight * _multiplier, features=("x", "noise")),
            L2(name="shrinkage", weight=l2_weight * _multiplier, features=regression_features),
        )
        if smooth:
            _penalties += (
                Laplacian(
                    name="smoothing",
                    weight=graph_weight,
                    graph=selected_graph,
                    features=regression_features,
                ),
            )
        return Pipeline(
            preprocessing=compose(AddIntercept(name="intercept")),
            model=RegressionSpec(
                features=regression_features,
                axes=() if pooled else (group_axis,),
                penalties=_penalties,
            ),
        )

    demonstration_solver = ADMM(max_iterations=1200, atol=2e-5, rtol=2e-4)
    fitted_results = {
        "pooled": fit(_recipe(pooled=True, smooth=False), train_x, train_y, solver=demonstration_solver),
        "independent": fit(_recipe(pooled=False, smooth=False), train_x, train_y, solver=demonstration_solver),
        "stratified": fit(_recipe(pooled=False, smooth=True), train_x, train_y, solver=demonstration_solver),
    }
    fitted_models = {name: require_success(result) for name, result in fitted_results.items()}
    return (
        fitted_models,
        fitted_results,
        graph_weight,
        l1_weight,
        l2_weight,
        regression_features,
        selected_graph,
    )


@app.cell
def _(graph_weight, l1_weight, l2_weight, mo, training_counts):
    mo.vstack(
        [
            mo.md(
                f"""
                ## The experiment currently shown

                **Graph weight:** `{graph_weight:.4g}` · **L1:** `{l1_weight:.4g}` ·
                **L2:** `{l2_weight:.4g}` · **training rows:** `{int(training_counts.sum())}`

                Group 4 is absent from training. Its stratified coefficients can be
                informed by neighbors; its independent fit is identified by the local
                L2 penalty. The irrelevant `noise` feature has true coefficient zero.
                """
            ),
            mo.callout(
                "For this deliberately empty-group example, L2 includes the intercept. "
                "The pooled model uses eight times the local penalty weights, matching "
                "the penalty of a constant coefficient field across eight strata.",
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(
    COLORS,
    fitted_models,
    np,
    plot_style,
    plt,
    regression_features,
    style_axes,
    true_coefficients,
):
    with plot_style():
        coefficient_figure, coefficient_axes = plt.subplots(1, 3, figsize=(13, 3.7))
        for _j, (_feature, _ax) in enumerate(zip(regression_features, coefficient_axes, strict=True)):
            _ax.plot(range(8), true_coefficients[:, _j], "--", color=COLORS["truth"], label="Truth")
            for _name, _model in fitted_models.items():
                _coefficients = np.asarray(_model.coefficients)
                if len(_coefficients) == 1:
                    _coefficients = np.repeat(_coefficients, 8, axis=0)
                _ax.plot(range(8), _coefficients[:, _j], "o-", ms=4, color=COLORS[_name], label=_name.title())
            _ax.axvspan(3.7, 4.3, color=COLORS["accent"], alpha=0.13)
            _ax.set_xticks(range(8))
            style_axes(_ax, title=_feature, xlabel="Group", ylabel="Coefficient")
        coefficient_axes[0].legend(loc="best")
        coefficient_figure.suptitle("Estimated coefficients and known truth", x=0.07, ha="left", fontsize=15, y=1.03)
        coefficient_figure.tight_layout()
    coefficient_figure
    return


@app.cell
def _(
    COLORS,
    fitted_models,
    np,
    pd,
    plot_style,
    plt,
    regression_metrics,
    selected_graph,
    style_axes,
    test_x,
    test_y,
    training_counts,
):
    test_predictions = {name: model.predict(test_x) for name, model in fitted_models.items()}
    score_table = pd.DataFrame(
        [{"Model": name.title(), **regression_metrics(test_y, prediction)} for name, prediction in test_predictions.items()]
    )
    with plot_style():
        evidence_figure, (_graph_ax, _error_ax) = plt.subplots(1, 2, figsize=(12.5, 3.9))
        _angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        _positions = np.column_stack((np.cos(_angles), np.sin(_angles)))
        for _left, _right, _weight in selected_graph.edges:
            _ends = _positions[[int(_left), int(_right)]]
            _graph_ax.plot(_ends[:, 0], _ends[:, 1], color="#c3cdd5", lw=2, zorder=1)
        _graph_ax.scatter(
            _positions[:, 0], _positions[:, 1], s=300,
            c=training_counts.to_numpy(), cmap="YlGnBu", vmin=0,
            vmax=max(1, int(training_counts.max())), edgecolor="white", linewidth=1.5, zorder=2,
        )
        for _node, (_xx, _yy) in enumerate(_positions):
            _text_color = "white" if training_counts.loc[_node] > training_counts.max() / 2 else "#273746"
            _graph_ax.text(_xx, _yy, str(_node), ha="center", va="center", color=_text_color, fontsize=9)
        _graph_ax.set_aspect("equal")
        _graph_ax.set_axis_off()
        _graph_ax.set_title("Graph structure · pale node 4 has no training data", loc="left", fontsize=11)
        for _offset, (_name, _prediction) in enumerate(test_predictions.items()):
            _rmse = [
                float(np.sqrt(np.mean((test_y.to_numpy()[test_x["group"].to_numpy() == _group] - _prediction[test_x["group"].to_numpy() == _group]) ** 2)))
                for _group in range(8)
            ]
            _error_ax.plot(range(8), _rmse, "o-", color=COLORS[_name], label=_name.title())
        _error_ax.axvspan(3.7, 4.3, color=COLORS["accent"], alpha=0.13)
        _error_ax.set_xticks(range(8))
        style_axes(_error_ax, title="Held-out prediction error by group", xlabel="Group", ylabel="Test RMSE · lower is better")
        _error_ax.legend()
        evidence_figure.tight_layout()
    evidence_figure
    return score_table, test_predictions


@app.cell
def _(diagnostics_summary, fitted_results, mo, pd, score_table):
    solve_table = pd.DataFrame(
        [{"model": name, **diagnostics_summary(result.diagnostics)} for name, result in fitted_results.items()]
    )
    mo.vstack(
        [
            mo.md("## Predictive evidence and solve quality\n\nCompare held-out errors first, then inspect whether the requested optimizations actually converged."),
            mo.ui.table(score_table.round(5), selection=None, pagination=False),
            mo.accordion({"Solver diagnostics": mo.ui.table(solve_table, selection=None, pagination=False)}),
            mo.md(
                "A stronger graph penalty reduces differences between connected coefficients. "
                "It can reduce variance in sparse groups, but excessive smoothing introduces bias. "
                "A wrong graph may make predictions worse. L1 can remove irrelevant coefficients; "
                "that is a different effect from smoothing or improving test error."
            ),
        ]
    )
    return


@app.cell
def _(
    fitted_models,
    fitted_results,
    np,
    test_predictions,
    train_x,
    training_counts,
    untouched_train,
):
    review_checks = {
        "all requested solves converged": all(result.diagnostics.converged for result in fitted_results.values()),
        "input frame was not modified": train_x.equals(untouched_train),
        "held-out predictions are finite": all(np.isfinite(values).all() for values in test_predictions.values()),
        "group 4 genuinely had zero training rows": int(training_counts.loc[4]) == 0,
        "declared missing group remains in model": len(fitted_models["stratified"].coefficients) == 8,
    }
    assert all(review_checks.values()), review_checks
    return (review_checks,)


@app.cell
def _(
    graph_weight,
    json,
    l1_weight,
    l2_weight,
    mo,
    package_versions,
    review_checks,
):
    run_record = {
        "seed": 17,
        "data": "seeded synthetic smooth coefficient field",
        "objective": "global mean squared loss / 2 + local and graph penalties",
        "weights": {"graph": graph_weight, "l1": l1_weight, "l2": l2_weight},
        "versions": package_versions(),
        "checks": review_checks,
    }
    mo.accordion({"Reproducibility and executable checks": mo.md("```json\n" + json.dumps(run_record, indent=2) + "\n```")})
    return


if __name__ == "__main__":
    app.run()
