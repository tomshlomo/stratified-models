# Stratified Models

Fit related regression or binary classification models across a graph. Local L1/L2 penalties control model complexity; Laplacian penalties encourage neighboring strata to have similar coefficients.

The library provides immutable model recipes and fitted predictors, explicit preprocessing composition, replayable sample sources, typed numerical outcomes, ADMM and specialized solvers, and versioned prediction artifacts. Samples can exceed memory; graph and coefficient work state must currently fit CPU memory.

## Explore the notebooks

The six marimo notebooks are the main guided review of the library. They include interactive experiments, numerical checks, solver comparisons, streaming workflows, and quick/full paper reproduction presets.

```sh
uv sync --extra common --group dev
uv run marimo edit examples/01_graph_regression.py
```

| Notebook | Explore |
| --- | --- |
| [Graph regression](examples/01_graph_regression.py) | Graph structure, borrowing strength, sparsity, pooled and independent baselines |
| [Mesothelioma classification](examples/02_mesothelioma.py) | Probabilities, coefficient curves, and the original-data classification experiment |
| [House prices](examples/03_house_prices.py) | Spatial coefficients and a King County paper-method preset |
| [Solver diagnostics](examples/04_solver_diagnostics.py) | ADMM residuals, direct/CG/Newton comparisons, warm starts and factorization reuse |
| [Streaming and composition](examples/05_streaming.py) | Equivalent in-memory and batched fitting, pandas/Polars/Parquet, memory and data passes |
| [Experiment to artifact](examples/06_workflow.py) | Calibration, explicit tuning, sklearn, persistence and failure handling |

Build executed HTML previews with `uv run python scripts/check_notebooks.py`. It bundles the frontend assets for offline review. Serve the directory with `python -m http.server 8000 --bind 127.0.0.1 --directory notebook_exports` and open `http://127.0.0.1:8000`. Defaults are small, seeded and offline. Full paper runs require an explicit action. See the [notebook review guide](docs/notebooks.md) and [paper reproduction details](docs/paper_reproductions.md).

## Fit a model

Install the core from this checkout with `pip install .`. The following example needs only core dependencies:

```python
import numpy as np

from stratified_models import (
    L2, AddIntercept, Axis, FitFailure, Graph, Laplacian,
    Pipeline, QuadraticSolver, RegressionSpec, Standardize,
    compose, fit,
)

x = np.linspace(-2, 2, 120)
regions = np.where(np.arange(len(x)) % 2 == 0, "north", "south")
X = {"x": x, "region": regions}
y = 1 + 2 * x + 0.5 * (regions == "south")

region = Axis(name="region", nodes=("north", "south"))
recipe = Pipeline(
    preprocessing=compose(
        Standardize(columns=("x",)),
        AddIntercept(name="intercept"),
    ),
    model=RegressionSpec(
        features=("x", "intercept"),
        axes=(region,),
        penalties=(
            L2(name="ridge", weight=0.01, features=("x",)),
            Laplacian(
                name="space", weight=0.1,
                graph=Graph.path(axis=region),
                features=("x", "intercept"),
            ),
        ),
    ),
)

result = fit(recipe, X, y, solver=QuadraticSolver())
if isinstance(result, FitFailure):
    raise RuntimeError(result.reason.message)

predictor = result.predictor
predictions = predictor.predict(X)
```

For binary targets encoded as zero and one, use `BinaryLogisticSpec` and a compatible solver. Its fitted predictor exposes `predict_proba`, `decision_function`, and thresholded `predict`. Constructors and source validation raise specific exceptions for invalid inputs; unsuccessful numerical solves return `FitFailure`. The failure-first guard above preserves task-specific result types in current type checkers.

## Select the numerical method explicitly

Pass `ADMM()` for the full supported objective family, including nonzero L1 penalties. `QuadraticSolver(linear_solver=DirectSolve(...))` and `QuadraticSolver(linear_solver=ConjugateGradient(...))` handle squared loss with L2 and graph penalties. `NewtonSolver()` handles smooth squared/logistic objectives. Optional `CVXPYSolver()` provides small independent reference problems.

`AutoSolver()` is an explicit strategy: nonzero L1 selects ADMM; smooth logistic selects Newton; quadratic problems select direct solving when the estimated system fits its memory limit and CG otherwise. Uncoupled strata use independent small direct blocks rather than one dense global matrix. Diagnostics explain the choice. Incompatible explicit solver/objective combinations raise before optimization. There is no implicit fallback after numerical failure.

The objective uses **global mean sample loss**, not a sum or a mean of batch means:

\[
\frac1N\sum_i\ell_i(\theta_{g_i})
+\sum_r\lambda_r r(\theta)
+\frac12\sum_j\gamma_j\theta^\mathsf{T}L_j\theta.
\]

Squared loss is half squared error. Each penalty declares its feature mask, including intercept treatment. Reproducing a paper's summed objective requires converting its penalty weights; the paper notebooks show the relevant conventions.

## Stream samples and reuse computation

```python
from stratified_models import ADMM, fit_batches, parquet_source

source = parquet_source("training.parquet", target="outcome", batch_size=8192)
result = fit_batches(recipe, source, solver=ADMM())
```

Sources produce aligned batches and replay one stable snapshot. Preprocessing fits independently of the optimizer. Streaming regression can prepare per-stratum sufficient statistics when their quadratic feature-size storage fits the preparation budget. `PreparationOptions(statistics_max_bytes=0)` disables these preparation-time statistics; CG or ADMM then evaluates the loss through sample replays. A direct solver can assemble its required quadratic statistics within its own configured memory limit. Logistic fitting uses repeated scans. Prediction also offers closable batch streams.

For repeated weights, use `prepare`/`prepare_batches` and `SolveSession`. Sessions retain warm-start state for one prepared dataset. Create a fresh session for a cold start. Explicitly shared `SolverCache` objects reuse compatible factors independently of warm starts. Cache keys distinguish data, normalization, graph structure, weights, penalty parameters and precision as applicable. Cache payloads are byte-bounded; active work buffers and Python container overhead are separate.

`InitialGuess.from_predictor(...)` transfers a primal starting point explicitly. Changed standardization can be transported through supported affine coordinate conversions; changed stratum meaning is rejected. Cross-validation keeps fold-specific preprocessing and warm starts separate while sharing compatible cached computations. Tuning accepts raw training/validation source pairs and never materializes all validation predictions to compute built-in metrics.

`result.preparation` reports preparation passes, rows, time and statistics storage. `result.diagnostics` reports that individual numerical solve, including its residuals, data passes, work and cache statistics. Re-solving does not charge preparation again.

## Integrations and artifacts

Optional extras are `pandas`, `polars`, `parquet`, `sklearn`, `tuning`, `cvxpy`, and `notebooks`. `pip install ".[common]"` installs the common bundle. Importing the core does not import these integrations. The separate `stratified_models.sklearn` estimators follow sklearn's mutable fit/clone conventions and delegate to the immutable native API.

`save_artifact` writes a versioned JSON manifest and checksummed numerical arrays. Task-specific `load_regressor` and `load_binary_classifier` preserve return types. Built-in transforms round-trip; custom components supply explicit codecs. Artifacts contain prediction state, not sample sources or optimizer sessions.

## Development and scope

```sh
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
RUN_NOTEBOOK_TESTS=1 uv run pytest tests/test_notebook_structure.py
uv run python scripts/check_notebooks.py
uv build
```

Core and common-extra installations are checked separately. Tests cover independent convex references, batch partition invariance, source replay, immutable outputs, fold isolation, cache invalidation, numerical failure, resource growth, sklearn cloning and artifact corruption. Notebook HTML execution and visual review complement these tests.

GPU acceleration, eigen-stratified model representations, Gaussian/Poisson models, and distributed execution remain later work. The earlier NumPy and JAX implementations are retained under `legacy/` as historical references and are excluded from the installed package and tests. The public API intentionally breaks compatibility with those unfinished versions.

This project is based on work by Jonathan Tuck, Shane Barratt and Stephen Boyd:

- [A distributed method for fitting Laplacian regularized stratified models](https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf).
- [Fitting Laplacian regularized stratified Gaussian models](https://web.stanford.edu/~boyd/papers/pdf/cov_strat_models.pdf).
- [Eigen-stratified models](https://web.stanford.edu/~boyd/papers/eigen_strat.html).

See [the implementation contracts](docs/architecture.md) for ownership, mathematical conventions and release acceptance criteria.
