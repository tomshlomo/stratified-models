# Stratified Models notebook review

The six marimo notebooks are executable experiments covering the public library API, numerical correctness, resource costs, and statistical evaluation. Each opens with a small seeded run. Expensive searches, larger files, and paper-method reproductions require an explicit action.

## Start the review

Install the project and its common integrations from the checkout:

```sh
uv sync --extra common --group dev
uv run --extra common marimo edit examples/01_graph_regression.py
```

Use `marimo run` instead of `marimo edit` for an application view. Each source is an ordinary Python marimo notebook; run commands from the repository root. Notebook helpers are local to `examples/` and do not belong to the installed numerical core.

To execute all quick defaults and build rendered previews:

```sh
uv run --extra common python scripts/check_notebooks.py
```

The exporter writes a review index at `notebook_exports/index.html` and bundles the matching marimo frontend assets beside it. Serve that directory locally so browser modules load normally:

```sh
uv run python -m http.server 8080 --bind 127.0.0.1 --directory notebook_exports
```

Open `http://127.0.0.1:8080` for the review index. The HTML files preserve the measured default outputs and need no external assets. Open the Python notebook in marimo to change controls or submit a full experiment. The exporter checks notebook structure first and applies a per-notebook execution timeout.

## Review sequence

| Notebook | Main question | Evidence and interactions |
| --- | --- | --- |
| [01 Graph regression](../examples/01_graph_regression.py) | What does graph smoothing do? | Compare a known coefficient field with pooled, independent, and stratified fits. Change graph structure, graph strength, and L1 sparsity. One group has no training observations. Inspect coefficient curves, graph structure, per-group held-out error, and numerical certificates. |
| [02 Mesothelioma classification](../examples/02_mesothelioma.py) | Can the age and sex graph improve probability estimates? | Start with a clearly labelled synthetic illustration, then explicitly load the original dataset and run the paper-method protocol. Inspect held-out probability scores, errors, and coefficient curves. |
| [03 House prices](../examples/03_house_prices.py) | How do spatially related regressions compare with pooled and separate models? | Explore spatial coefficient maps and held-out log-price errors. The full preset follows the paper's grid and five-fold methodology with recorded differences and scaling conventions. |
| [04 Solver diagnostics](../examples/04_solver_diagnostics.py) | Do eligible methods solve the same objective, and where does their work go? | Compare ADMM, direct and CG quadratic methods, Newton-CG, automatic selection, and optional CVXPY. Toggle quadratic/logistic loss and L1 eligibility. Inspect residual thresholds, inner certificates, typed nonconvergence, independent cache sharing, warm starts, and a regularization path. |
| [05 Streaming and composition](../examples/05_streaming.py) | Does changing sample storage or batch size change the fitted objective? | Fit the same regression or logistic snapshot in memory and through pandas, Polars, or Parquet batches. Disable sufficient statistics to exercise actual replays. Compare coefficients and predictions, inspect fixed-width bin boundaries, and optionally generate a larger file in bounded chunks. |
| [06 Experiment to artifact](../examples/06_workflow.py) | Can a fitted pipeline be selected, reused, and persisted reliably? | Inspect classifier discrimination and calibration. Change the decision threshold without refitting. Run explicit fold-isolated tuning, clone the sklearn adapter, inspect a versioned artifact, round-trip an external transform through its codec, and exercise input errors and typed numerical failure. |

## Scientific conventions

All native objectives use a global mean sample loss, followed by explicitly weighted local and graph penalties. Unequal batches contribute their actual row counts. A graph edge has one declared nonnegative weight, and the axis owns the label order used by both graph operators and data encoding.

The notebooks state which coefficients receive local penalties and graph smoothing. The synthetic missing-group experiment penalizes its intercept locally to identify the independent empty-group baseline. Paper presets follow their specified intercept convention instead.

Pooled and stratified models can assign different total local penalties to an otherwise constant coefficient field. Notebook 01 scales the pooled penalty by the number of strata so that this comparison is explicit. Held-out accuracy is never guaranteed to improve for every graph or regularization weight.

Calibration, ranking, coefficient sparsity, graph smoothness, and prediction error measure different properties. Classification notebooks display proper probability scores as well as decision metrics. Decision thresholds should be selected on validation data and application costs; moving a test-set slider is an illustration, not an optimization protocol.

## Solvers and reusable work

ADMM is eligible for every supported objective, including nonzero L1 penalties. Direct and CG quadratic solvers require a quadratic objective. Newton-CG requires a smooth objective. `AutoSolver` is explicitly selected and records its deterministic choice.

The solver notebook separates three runs of the same prepared problem: a cold session with an empty cache, a fresh session using a populated cache, and a reused session. Fresh sessions start independently while compatible factorization entries may be shared. An unchanged exact matrix can reuse its factorization; changed matrices require an appropriate recomputation or separately supported update.

Interactive wall times include relevant first-use compilation and are not controlled hardware benchmarks. The comparison tables also expose convergence, stationarity, source passes, rows scanned, and factorization counts. ADMM objective values may be nonmonotone; its residual and inner-accuracy tests establish convergence.

## Data and resource behavior

Quick defaults use deterministic local synthetic snapshots and make no network requests. Paper data loading is explicit and records source provenance. The paper notebooks distinguish synthetic illustrations from original-data reproductions; see [paper reproduction notes](paper_reproductions.md).

Streaming examples change batch partitions of one already-defined snapshot. They do not regenerate a different random dataset for each batch size. The numerical solver can scan a source repeatedly, while transforms and validation can require additional preparation scans.

The optional larger Parquet experiment writes data in bounded chunks. Its displayed process RSS is a high-water measurement including imports and prior notebook activity, not an isolated memory bound. Separate fresh-process resource checks measure growth at fixed batch and model sizes. Reproduce the preparation measurements, then include numerical fitting, with:

```sh
uv run python scripts/profile_streaming.py --samples 20000 --prepare-only
uv run python scripts/profile_streaming.py --samples 2000000 --prepare-only
uv run python scripts/profile_streaming.py --samples 20000 --fit
```

Preparation counters include preprocessing and validation scans; numerical solve counters account for subsequent evaluation. Preparation statistics occupy a separately reported byte budget. The graph, coefficients, and work buffers remain subject to a model-memory budget.

Prediction streams carry row identifiers. They are consumed incrementally in production. The quick equivalence checks concatenate only small fixtures to compare against ordinary in-memory predictions.

## Reproducibility and verification

The review notebooks expose a `review_checks` mapping with executable assertions. Checks cover reference objectives, batch equivalence, input immutability, finite probabilities, numerical failure branches, sklearn cloning, artifact checksums, and fitted-component round trips as applicable. Run the four general workflow journeys with:

```sh
RUN_NOTEBOOK_TESTS=1 uv run --extra common python -m pytest tests/test_notebook_structure.py
```

Structural checks and HTML execution are also available without running those pytest journeys:

```sh
uv run --extra common python scripts/check_notebooks.py --check-only
uv run --extra common python scripts/check_notebooks.py --output-dir notebook_exports
```

After execution, inspect the rendered pages for figure readability, error outputs, active settings, and distinctions between requested and completed expensive runs. Fixed seeds, package versions, effective configurations, and stated evaluation boundaries make numerical evidence reviewable. Full paper runs additionally record data hashes, folds, tolerances, and methodology deviations.
