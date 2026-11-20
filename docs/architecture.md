# Stratified Models implementation contracts

The first release fits Laplacian regularized stratified squared-error and binary logistic models with L1 and L2 penalties. It supports replayable sample batches and ordinary in-memory data. Coefficients and graphs initially fit CPU memory. GPU execution and eigen-stratified coefficient representations are later milestones.

## Agreed user-facing behavior

- Immutable validated recipes, axes, graphs, fitted transforms, predictors and diagnostics. Composition uses protocols and small components rather than inheritance.
- Preprocessing is outside the numerical optimizer. Explicit transform chains support streaming standardization, a constant intercept feature and fixed-width bins. More elaborate transforms may run in memory or be supplied pre-fitted.
- The objective is globally mean sample loss plus explicitly weighted penalties, independent of batch boundaries. No implicit sample-count rescaling of weights.
- Ordinary `fit(recipe, X, y, solver=...)` and explicit `fit_batches(recipe, source, solver=...)` return `FitSuccess(predictor, diagnostics) | FitFailure(reason, diagnostics)`. Invalid configurations, schemas, data, and source replay raise specific exceptions. Numerical nonconvergence and exhausted solve budgets return typed failures. Cancellation and programming errors propagate.
- `solver` is required. `AutoSolver()` is an explicit deterministic strategy: nonzero L1 selects ADMM; quadratic objectives select direct or CG by configured resource limits; smooth logistic selects Newton-CG. Record the chosen method and reason. No implicit fallback.
- ADMM is a first-class production solver. Also support direct/CG quadratic solvers, smooth Newton-CG and optional small CVXPY references.
- One authoritative `Axis` owns ordered labels. Graphs and encoders refer to that axis; unknown labels fail explicitly. Intercepts are explicit features, omitted from local penalties in examples and included in graph smoothing.
- A prepared problem is immutable and bound to one data snapshot. A session owns its warm-start state. Sessions have no required context manager and no reset method; a fresh instance starts independently.
- An explicit bounded solver cache is independent of warm starts and may be shared by sessions. Exact factors are reused only for matching operator dependencies. Compiled kernels take numerical weights as arguments. Changed data must not accidentally reuse old loss statistics or factors.
- Cross-validation fits preprocessing independently per fold. Each fold has its own warm-start history across parameter trials. Compatible numerical caches may be shared. Cross-fold coefficient initialization is explicit, coordinate checked and recorded.
- Optional explicit Optuna tuning, optional sklearn adapter, core plus integration extras and a common bundle.
- Versioned array/JSON artifacts persist fitted prediction state. Built-in codecs are available; custom components use an explicit codec registry. Inference artifacts do not contain sources, training rows or optimizer caches.

## Shared implementation interfaces

These internal contracts coordinate modules; public examples may add concise factories.

`specs.py` defines frozen `Axis(name, nodes)`, `Graph(axis, edges)` with `Graph.path(axis=...)`, named `L1`, `L2` and `Laplacian` penalties (`name`, finite nonnegative `weight`, tuple `features`, and `graph` for Laplacian), `RegressionSpec(features, axes, penalties)`, `BinaryLogisticSpec(features, axes, penalties)`, `Pipeline(preprocessing, model)` and `WeightRef`. Feature and axis names are strings. Binary numerical targets are zero or one. `Pipeline.weight(name)` resolves and validates a penalty reference.

`data.py` defines owned read-only NumPy `FeatureBatch(columns, row_ids)` and `TrainingBatch(columns, targets, row_ids)`. Columns are an immutable mapping from strings to one-dimensional arrays. `NumericBatch(x, y, nodes, row_ids)` has a floating two-dimensional design matrix, floating target vector and int64 flattened node codes. Source protocols expose `snapshot_id`, `n_rows` (possibly unknown), `column_names`, and `batches()` returning a fresh closable iterator. Supervised and feature-only sources have distinct batch types. Array/pandas/Polars adapters align rows positionally and reject ambiguous X/y indexes. File readers close in iterator finalizers. Sources must replay a stable dataset. Parquet support is an optional import.

`preprocessing.py` defines frozen `Standardize(columns)`, `AddIntercept(name)`, `FixedWidthBins(column, output, origin, width, n_bins)`, `TransformChain(steps)` and `compose(*steps)`. Fitting a transform recipe returns a separate fitted transform. Fitted chains transform feature batches without mutation and preserve row identity/targets. Standardization uses population variance and a scale of one for constant columns. Fixed bin grids have a finite declared domain and reject out-of-range values. Serialization exposes fitted parameter data, not sklearn objects.

Root-owned preparation exposes a numerical problem protocol with `task` (`regression` or `binary`), `axes`, `axis_sizes`, `features`, `penalties`, `n_samples`, `n_nodes`, `n_features`, `fingerprint`, `design_fingerprint`, optional `statistics`, and `iter_numeric()` yielding validated `NumericBatch` values. Each replay verifies its row-count/snapshot contract. Quadratic statistics are unnormalized sums `xtx` (nodes, features, features), `xty` (nodes, features), and scalar `yty`; loss and derivatives divide by `n_samples`. Statistics are optional and memory budgeted. There are no sample-retaining per-node function closures.

`solvers/` owns solver configuration, numerical problem protocol and statistics types, bounded `SolverCache`, numerical results and immutable diagnostics. Public numerical entry point: `solve(problem, *, solver, weights, cache, initial=None, state=None)`. Numerical success contains coefficients, diagnostics and optional solver-specific continuation state; numerical failure contains a typed reason and diagnostics. Return arrays must remain valid after later solves. Root high-level fitting wraps numerical success into a task-specific fitted predictor.

Diagnostics include selected solver/reason, convergence/termination, objective components, stationarity measure, iterations, elapsed time, source passes and rows scanned, cache hits/misses/factorization count, and immutable iteration history. ADMM history exposes primal/dual residuals and thresholds, inner optimality residuals and penalty values. Timings are measured after relevant device synchronization.

## Numerical foundation

Use ADMM product-space consensus over mean loss, combined local L1/L2 and the graph quadratic sum. Local prox is exact soft threshold plus shrinkage. Graph prox solves an SPD system by direct factorization or matrix-free CG. Loss prox uses sufficient statistics for quadratic data when budget permits; otherwise stream its operator. Logistic loss prox uses damped Newton-CG with certified inner accuracy. Inner errors decrease to the requested accuracy; exhausted inner budgets are failures. Correct dual scaling, residual stopping and finite-value checks replace the old objective-extrapolation heuristic.

Batch loss/gradient/Hessian-vector kernels use JAX; source scans and convergence orchestration remain on the host. Bound asynchronous buffering and compiled shape variants. No process-global logging or JAX configuration changes at import. CPU float64 is the correctness default, with execution-scoped precision configuration.

## Review notebooks and release gates

Six substantive marimo notebooks cover graph regression, mesothelioma classification, King County spatial regression, solver diagnostics/reuse, streaming/composition, and tuning/sklearn/artifacts. Each has fixed seeds, purposeful reactive controls, clear figures and narrative, meaningful correctness assertions, fast default execution and explicit expensive runs. Deliver editable sources and rendered HTML previews with an index. Headless execution and visual inspection gate release.

Paper presets reproduce statistical objectives, preprocessing, splits and evaluation using this implementation. They do not claim to reproduce the authors' precise optimizer trajectory. Match the paper specification and document discrepancies in original example scripts. Convert summed-loss penalty weights to mean-loss units using each training fold's sample count and matching half factors. Reproduction results record data hashes, folds, versions, settings, tolerances and deviations.

Acceptance checks compare independent CVXPY objectives, gradients and stationarity; uneven batch partitions; sufficient-stat versus streamed regression; graph reordering and absent/disconnected nodes; zero penalties and rank deficiency; extreme logits; input immutability; fold isolation; cache invalidation; bounded memory as sample count grows; clean core/extras installations; sklearn clone semantics; artifact round trips; all notebook quick presets; and selected full paper presets.
