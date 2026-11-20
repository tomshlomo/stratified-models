"""Inspect solver compatibility, convergence, warm starts, and factor reuse."""

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="full",
    app_title="Solvers, convergence, and reusable work",
)


@app.cell
def _():
    import importlib.util

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from _support import (
        COLORS,
        diagnostics_summary,
        plot_style,
        require_success,
        style_axes,
        synthetic_classification,
        synthetic_regression,
    )
    from stratified_models import (
        ADMM,
        L1,
        L2,
        AddIntercept,
        AutoSolver,
        Axis,
        CVXPYSolver,
        ConjugateGradient,
        DirectSolve,
        FitFailure,
        Graph,
        Laplacian,
        NewtonSolver,
        Pipeline,
        QuadraticSolver,
        RegressionSpec,
        BinaryLogisticSpec,
        SolveSession,
        SolverCache,
        compose,
        prepare,
    )
    return (
        ADMM,
        AddIntercept,
        AutoSolver,
        Axis,
        BinaryLogisticSpec,
        COLORS,
        CVXPYSolver,
        ConjugateGradient,
        DirectSolve,
        FitFailure,
        Graph,
        L1,
        L2,
        Laplacian,
        NewtonSolver,
        Pipeline,
        QuadraticSolver,
        RegressionSpec,
        SolveSession,
        SolverCache,
        compose,
        diagnostics_summary,
        importlib,
        mo,
        np,
        pd,
        plot_style,
        plt,
        prepare,
        require_success,
        style_axes,
        synthetic_classification,
        synthetic_regression,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Solvers, convergence, and reusable work

    **One objective · several numerical methods · explicit resource costs**

    Solver choice is explicit. Quadratic objectives admit a linear solve;
    smooth logistic objectives admit Newton-CG; ADMM also supports the
    nonsmooth L1 penalty. `AutoSolver()` is an explicit, documented selection
    policy rather than a hidden change to the objective.

    This notebook compares converged solutions, opens the ADMM diagnostics,
    and separates **factorization reuse** from **warm-start reuse**.
    All comparisons below use the same prepared dataset and objective.
    """)
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "The common objective and the ADMM split": mo.md(
                r"""
                Every compatible solver minimizes

                $$
                \frac{1}{N}\sum_i \ell_i(\theta_{g_i})
                +\lambda_1\lVert M_1\theta\rVert_1
                +\frac{\lambda_2}{2}\lVert M_2\theta\rVert_2^2
                +\frac{\gamma}{2}\theta^T L\theta.
                $$

                The feature masks state which coefficients receive each penalty.
                ADMM keeps copies for the sample loss, local penalties, and graph
                quadratic, then enforces consensus. These copies form one
                product-space variable in a two-block consensus formulation.

                The local proximal update is soft thresholding and shrinkage.
                The graph update solves a positive-definite linear system.
                Logistic loss needs an iterative inner solve; its certified
                proximal-error bound is plotted against the requested accuracy.
                """
            ),
            "Explicit solver selection": mo.md(
                "```python\n"
                "ADMM(atol=1e-6, rtol=1e-5)\n"
                "QuadraticSolver(linear_solver=DirectSolve())\n"
                "QuadraticSolver(linear_solver=ConjugateGradient())\n"
                "NewtonSolver()\n"
                "AutoSolver()  # explicit deterministic selection policy\n"
                "CVXPYSolver()  # optional small-problem reference\n"
                "```"
            ),
        }
    )
    return


@app.cell
def _(importlib, mo):
    task_control = mo.ui.dropdown(
        options=["Squared regression", "Binary logistic"],
        value="Squared regression", label="Data loss",
    )
    l1_control = mo.ui.checkbox(value=False, label="Add a nonzero L1 penalty")
    reference_control = mo.ui.checkbox(
        value=importlib.util.find_spec("cvxpy") is not None,
        label="Include the optional CVXPY reference",
    )
    rho_control = mo.ui.slider(
        start=-1.0, stop=1.0, step=0.5, value=0.0,
        label="log₁₀ initial ADMM ρ", show_value=True,
    )
    mo.hstack(
        [mo.vstack([task_control, l1_control, reference_control]), rho_control],
        justify="start", gap=3,
    )
    return l1_control, reference_control, rho_control, task_control


@app.cell
def _(
    AddIntercept,
    Axis,
    BinaryLogisticSpec,
    Graph,
    L1,
    L2,
    Laplacian,
    Pipeline,
    RegressionSpec,
    compose,
    l1_control,
    prepare,
    synthetic_classification,
    synthetic_regression,
    task_control,
):
    if task_control.value == "Squared regression":
        solver_x, solver_y, _, _, _ = synthetic_regression(
            seed=41, n_groups=5, train_per_group=28,
            test_per_group=4, missing_group=None,
        )
        _specification = RegressionSpec
    else:
        solver_x, solver_y, _, _ = synthetic_classification(
            seed=41, n_groups=5, n_train=220, n_test=20,
        )
        _specification = BinaryLogisticSpec
    solver_axis = Axis(name="group", nodes=tuple(range(5)))
    solver_features = ("x", "noise", "intercept")
    solver_recipe = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=_specification(
            features=solver_features,
            axes=(solver_axis,),
            penalties=(
                L1(name="sparsity", weight=0.012 if l1_control.value else 0.0, features=("x", "noise")),
                L2(name="shrinkage", weight=0.02, features=solver_features),
                Laplacian(name="smoothing", weight=0.025, graph=Graph.path(axis=solver_axis), features=solver_features),
            ),
        ),
    )
    prepared_problem = prepare(solver_recipe, solver_x, solver_y)
    return (prepared_problem,)


@app.cell
def _(
    ADMM,
    AutoSolver,
    CVXPYSolver,
    ConjugateGradient,
    DirectSolve,
    NewtonSolver,
    QuadraticSolver,
    SolveSession,
    SolverCache,
    l1_control,
    prepared_problem,
    reference_control,
    require_success,
    rho_control,
    task_control,
):
    solvers = {
        "ADMM": ADMM(max_iterations=1600, atol=1e-6, rtol=1e-5, rho=10.0 ** rho_control.value),
        "Auto": AutoSolver(),
    }
    if not l1_control.value:
        solvers["Newton-CG"] = NewtonSolver()
        if task_control.value == "Squared regression":
            solvers["Direct"] = QuadraticSolver(linear_solver=DirectSolve())
            solvers["CG"] = QuadraticSolver(linear_solver=ConjugateGradient())
    if reference_control.value:
        solvers["CVXPY"] = CVXPYSolver()

    comparison_results = {}
    for _name, _solver in solvers.items():
        _session = SolveSession(prepared_problem, solver=_solver, cache=SolverCache())
        _result = _session.solve()
        require_success(_result)
        comparison_results[_name] = _result
    reference_name = "CVXPY" if "CVXPY" in comparison_results else "ADMM"
    reference_objective = comparison_results[reference_name].diagnostics.objective.total
    return comparison_results, reference_name, reference_objective


@app.cell
def _(
    comparison_results,
    diagnostics_summary,
    l1_control,
    mo,
    pd,
    prepared_problem,
    reference_name,
    reference_objective,
    task_control,
):
    comparison_table = pd.DataFrame(
        [
            {
                "requested": name,
                **diagnostics_summary(result.diagnostics),
                "objective difference": result.diagnostics.objective.total - reference_objective,
            }
            for name, result in comparison_results.items()
        ]
    )
    _compatibility = (
        "L1 is active: direct quadratic and smooth Newton methods are intentionally excluded."
        if l1_control.value
        else (
            "The complete objective is quadratic: direct and matrix-free linear solves are eligible."
            if task_control.value == "Squared regression"
            else "Logistic loss is smooth but not quadratic: Newton-CG is eligible; a one-shot linear solve is not."
        )
    )
    mo.vstack(
        [
            mo.md("## Compatible methods solve the same problem"),
            mo.md(
                f"Shared preparation took **{prepared_problem.preparation.elapsed_seconds:.4g} s**, "
                f"**{prepared_problem.preparation.source_passes} source pass(es)** and "
                f"**{prepared_problem.preparation.rows_scanned:,} rows scanned**, retaining "
                f"**{prepared_problem.preparation.statistics_bytes / 1024:.3g} KiB** of sufficient statistics. "
                "This one-time cost is separate from the per-solve counters below."
            ),
            mo.callout(_compatibility, kind="info"),
            mo.ui.table(comparison_table.round(7), selection=None, pagination=False),
            mo.md(
                f"Objective differences use **{reference_name}** as the displayed reference. "
                "These are small interactive measurements after shared data preparation, "
                "including any first-use compilation. They are not a controlled throughput benchmark. "
                "Compare solution quality and source passes as well as wall time."
            ),
        ]
    )
    return


@app.cell
def _(
    COLORS,
    comparison_results,
    np,
    plot_style,
    plt,
    reference_objective,
    style_axes,
):
    admm_diagnostics = comparison_results["ADMM"].diagnostics
    admm_history = admm_diagnostics.history
    _iteration = np.array([row.iteration for row in admm_history])
    with plot_style():
        convergence_figure, _axes = plt.subplots(1, 3, figsize=(14, 3.8))
        _axes[0].plot(_iteration, [row.objective for row in admm_history], color=COLORS["ADMM"])
        _axes[0].axhline(reference_objective, ls="--", color=COLORS["reference"], label="Final reference")
        style_axes(_axes[0], title="Objective at the consensus iterate", xlabel="Outer iteration", ylabel="Mean loss + penalties")
        _axes[0].legend()
        for _field, _threshold, _label, _color in (
            ("primal_residual", "primal_threshold", "Primal", COLORS["ADMM"]),
            ("dual_residual", "dual_threshold", "Dual", COLORS["CG"]),
        ):
            _values = np.array([getattr(row, _field) for row in admm_history], dtype=float)
            _limits = np.array([getattr(row, _threshold) for row in admm_history], dtype=float)
            _axes[1].semilogy(_iteration, np.maximum(_values, 1e-16), color=_color, label=f"{_label} residual")
            _axes[1].semilogy(_iteration, np.maximum(_limits, 1e-16), "--", color=_color, alpha=0.7, label=f"{_label} threshold")
        style_axes(_axes[1], title="Both residual tests matter", xlabel="Outer iteration", ylabel="Residual norm · log scale")
        _axes[1].legend(fontsize=8)
        _inner = np.array([row.inner_residual for row in admm_history], dtype=float)
        _axes[2].semilogy(_iteration, np.maximum(_inner, 1e-16), color=COLORS["Direct"], label="Proximal error bound")
        _inner_bounds = np.array([getattr(row, "inner_threshold", np.nan) for row in admm_history], dtype=float)
        if np.isfinite(_inner_bounds).any():
            _axes[2].semilogy(_iteration, np.maximum(_inner_bounds, 1e-16), "--", color=COLORS["reference"], label="Requested accuracy")
        style_axes(_axes[2], title="Certified accuracy of the proximal subproblems", xlabel="Outer iteration", ylabel="Proximal error bound · log scale")
        _axes[2].legend(fontsize=8)
        convergence_figure.tight_layout()
    convergence_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ADMM's raw objective need not decrease on every iteration. A flat objective
    alone is not a convergence certificate: primal agreement, dual progress,
    and adequately solved inner problems are checked together.

    The loss is a **global sample mean**. Every solver includes the same local
    and graph terms, with the same coefficient masks. Unsupported objectives
    are rejected rather than silently dropping penalties.

    ## Warm starts and factorization caches are different

    The next three runs solve exactly the same prepared problem. The first
    starts with a new cache; the second uses a fresh session sharing the cache;
    the third reuses the first session's solution as well. A new session is the
    explicit way to start without the previous optimization history.
    """)
    return


@app.cell
def _(
    ADMM,
    SolveSession,
    SolverCache,
    diagnostics_summary,
    pd,
    prepared_problem,
    require_success,
):
    shared_cache = SolverCache(max_bytes=32 * 1024 * 1024)
    retained_session = SolveSession(
        prepared_problem,
        solver=ADMM(max_iterations=1600, atol=1e-6, rtol=1e-5),
        cache=shared_cache,
    )
    cold_result = retained_session.solve()
    cache_result = SolveSession(
        prepared_problem,
        solver=ADMM(max_iterations=1600, atol=1e-6, rtol=1e-5),
        cache=shared_cache,
    ).solve()
    warm_result = retained_session.solve()
    for _reused_result in (cold_result, cache_result, warm_result):
        require_success(_reused_result)
    reuse_table = pd.DataFrame(
        [
            {"run": name, **diagnostics_summary(result.diagnostics)}
            for name, result in (
                ("Cold session + empty cache", cold_result),
                ("Fresh session + populated cache", cache_result),
                ("Reused session + populated cache", warm_result),
            )
        ]
    )
    return (
        cache_result,
        cold_result,
        retained_session,
        reuse_table,
        warm_result,
    )


@app.cell
def _(mo, reuse_table):
    mo.vstack(
        [
            mo.ui.table(reuse_table.round(7), selection=None, pagination=False),
            mo.callout(
                "A direct linear solver primarily benefits from factorization reuse. "
                "CG, Newton, and ADMM can also benefit from a good initial iterate. "
                "Changing a matrix invalidates its exact numeric factor; a still-useful "
                "preconditioner is a separate, explicit reuse opportunity.",
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(
    ADMM,
    FitFailure,
    SolveSession,
    diagnostics_summary,
    mo,
    pd,
    prepared_problem,
):
    limited_result = SolveSession(
        prepared_problem, solver=ADMM(max_iterations=1),
    ).solve()
    assert isinstance(limited_result, FitFailure)
    mo.vstack(
        [
            mo.md("## Failure is a typed outcome, not a plausible-looking fitted model"),
            mo.callout(
                f"A deliberate one-iteration budget produced `{limited_result.reason}`. "
                "The failure branch carries diagnostics and does not expose a successfully fitted predictor.",
                kind="warn",
            ),
            mo.ui.table(pd.DataFrame([diagnostics_summary(limited_result.diagnostics)]), selection=None, pagination=False),
        ]
    )
    return (limited_result,)


@app.cell
def _(mo):
    weight_path_button = mo.ui.run_button(label="Run a regularization path with warm starts")
    mo.vstack(
        [
            mo.md(
                "## A sequence of related objectives\n\n"
                "This optional experiment changes only the graph weight. It records the "
                "actual objective, residual and reuse statistics for every solve. "
                "Only matching numerical operators may reuse exact factorizations."
            ),
            weight_path_button,
        ]
    )
    return (weight_path_button,)


@app.cell
def _(
    diagnostics_summary,
    mo,
    pd,
    require_success,
    retained_session,
    weight_path_button,
):
    if weight_path_button.value:
        _path_rows = []
        for _gamma in (0.001, 0.005, 0.025, 0.1):
            _path_result = retained_session.solve(weight_overrides={"smoothing": _gamma})
            require_success(_path_result)
            _path_rows.append({"graph weight": _gamma, **diagnostics_summary(_path_result.diagnostics)})
        weight_path_output = mo.ui.table(pd.DataFrame(_path_rows), selection=None, pagination=False)
    else:
        weight_path_output = mo.callout("The additional path has not been run. The default comparison above is complete.", kind="neutral")
    weight_path_output
    return


@app.cell
def _(
    FitFailure,
    cache_result,
    cold_result,
    comparison_results,
    limited_result,
    np,
    reference_objective,
    require_success,
    warm_result,
):
    _objectives = [result.diagnostics.objective.total for result in comparison_results.values()]
    _cold_prediction = np.asarray(require_success(cold_result).coefficients)
    review_checks = {
        "all compatible methods converged": all(result.diagnostics.converged for result in comparison_results.values()),
        "objectives agree to reference tolerance": max(abs(value - reference_objective) for value in _objectives) < 5e-4 * max(1.0, abs(reference_objective)),
        "cache reuse preserves coefficients": np.allclose(_cold_prediction, require_success(cache_result).coefficients, atol=3e-4, rtol=3e-4),
        "warm reuse preserves coefficients": np.allclose(_cold_prediction, require_success(warm_result).coefficients, atol=3e-4, rtol=3e-4),
        "exhausted budget returns the failure branch": isinstance(limited_result, FitFailure),
    }
    assert all(review_checks.values()), review_checks
    return


if __name__ == "__main__":
    app.run()
