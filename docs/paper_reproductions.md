# Paper experiments and notebook provenance

The editable marimo notebooks are the main way to review the statistical API and its
results:

- [Age, sex, and borrowing strength](../examples/02_mesothelioma.py): binary
  classification, age × sex graphs, explicit ADMM/Newton selection, fold-specific
  preprocessing, and coefficient curves.
- [A map of local housing models](../examples/03_house_prices.py): spatial regression,
  fixed binning, coefficient maps, common/separate/stratified baselines, and random forests.

Both start with a small seeded **synthetic adaptation**, require no network access,
and display measured scores. Generated observations are never described as the original
data or evidence of reproducing a published result. Switch presets and press the explicit
run button to evaluate the original datasets. Downloading a missing dataset requires its
own opt-in checkbox. A valid cached snapshot can be used without another download.

From the repository root, install the notebook extra and open a notebook:

```sh
uv sync --extra notebooks
uv run marimo edit examples/02_mesothelioma.py
uv run marimo edit examples/03_house_prices.py
```

The notebooks expose equal candidate-count budgets, graph strengths and solver choices.
One candidate evaluates the published penalty setting. Three candidates scale each
family's penalties by 0.3, 1, and 3. The forest uses minimum leaf sizes 1, 3, and 8.
This is a small, declared comparison rather than an exhaustive optimization claim.
Original-data evaluation uses five folds; quick mode uses one held-out split.

Each notebook offers downloadable JSON containing its data fingerprint, exact fold
positions, seed, features, solver configuration, grid/coupling settings, candidate budget
and dependency versions. Measured score tables are downloadable as CSV. Numerical
diagnostics are visible for every numerical fit. Coefficient figures use a final model
refitted on all observations and are labeled separately from validation results.

## Primary sources

Jonathan Tuck, Shane Barratt, and Stephen Boyd, *A Distributed Method for Fitting
Laplacian Regularized Stratified Models*, JMLR 22 (2021), 1–37:

- [Paper and code landing page](https://web.stanford.edu/~boyd/papers/strat_models.html)
- [Final paper PDF](https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf)
- [Inspected author implementation, commit 0108114](https://github.com/cvxgrp/strat_models/tree/0108114afe1265262f345bf93a5bc147f6cb146c)

Our paper presets reproduce the stated statistical setup using this package. They do
not claim to reproduce the original optimizer's trajectory, implementation bugs,
unpublished fold assignments or precise reported table entries. The notebooks make
discrepancies visible rather than silently using whichever source is easiest to execute.

## Mesothelioma classification

The experiment follows §5.1, pp.18–19: logistic regression with local squared L2
regularization, 134 age × sex strata, an age path over 19 through 85, and an edge between
the two sex categories. The original-data preset uses all 324 observations, one-hot
encoding with declared categorical domains, and fold-fitted standardization. It removes
`diagnosis method`, which leaks the outcome, along with the outcome itself from regressors.
Age and sex are stratification columns rather than regressors.

The classification intercept mask is explicit: the default includes it in local L2,
ensuring a finite optimum at nodes whose observations all belong to one class. The
classification experiment does not explicitly specify that mask, so this is a documented
interpretation rather than a claim about the authors' unpublished settings. Toggle the
mask to compare the sensitivity; it is recorded in each run manifest. Housing follows
its paper's explicit instruction to exclude the intercept from local L2.

The three families are common, separate, and graph-regularized models. The notebook
reports average negative log likelihood and classification error, and plots the glucose
coefficient over age for each sex. All models share the same stratified five-fold split
with seed 41. The fold assignments are a reproducible choice here; the authors' original
assignments were not published.

Published summed-loss weights used by the one-candidate preset:

| Family | Local L2 | Sex graph | Age graph |
| --- | ---: | ---: | ---: |
| Common | 0.004 | — | — |
| Separate | 12.9 | 0 | 0 |
| Stratified | 0.52 | 10 | 500 |

Dataset attribution: Abdullah Tanrikulu and Orhan Er (2012),
[Mesothelioma's disease data set, UCI DOI 10.24432/C5903T](https://doi.org/10.24432/C5903T),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [Current UCI data card](https://archive.ics.uci.edu/dataset/351/mesothelioma+s+disease+data+set)
- [Download archive](https://archive.ics.uci.edu/static/public/351/mesothelioma+s+disease+data+set.zip)
- Raw member: `Mesothelioma data set.xlsx`
- SHA256: `cb3a7b5b9352ff30224f5080769db031339dd78c5a4ffb6860cb2a2a54cba73f`

The inspected [author script](https://github.com/cvxgrp/strat_models/blob/0108114afe1265262f345bf93a5bc147f6cb146c/examples/mesothelioma.py)
uses a 10% holdout and omits the local regularization reported by the paper. Its logistic
implementation represents two classes with two softmax vectors. This package uses one
log-odds vector. If matching that source parameterization, the minimum-norm softmax
vectors are `(-beta/2, beta/2)`: their summed squared norms and graph energies are half
those of the single-vector representation. Copying its penalty numbers unchanged would
therefore change the model. Our preset follows the paper's single-vector logistic model.

## King County house prices

The experiment follows §5.2, pp.19–21. Predict natural log sale price from nine covariates:
bedrooms, bathrooms, living area, lot area, floors, waterfront, condition, grade, and year
built. An explicit intercept makes ten coefficients per location. Latitude and longitude
select a cell of the 50 × 50 graph. L2 excludes the intercept; geographical smoothing
includes it. The paper-setting local weight is 0.001 in each family, with graph weight
18.5 for the stratified family.

The data filter `longitude <= -121.6` follows the authors' script. It retains 21,596 of
21,613 sales. The declared geographic domain is latitude `[47.1559, 47.7776]` and longitude
`[-122.519, -121.646]`, tied to the verified snapshot. Equal-width grids are fixed before
fold evaluation. A two-unit floating-point expansion of the upper coordinate includes the maximum
coordinate; observations on an internal bin boundary follow this package's documented
bin convention. All statistical scaling is fitted independently on each training fold.

All four families use the same shuffled five-fold split with seed 17. The random forest
receives the nine covariates and raw latitude/longitude; its full preset has 100 trees.
The score is RMSE in **natural log-price units**, accompanied by R². Fold standard
deviations show variation across folds and are not confidence intervals. Multiple-candidate
scores also serve model selection; they are not an independent post-selection test.

Dataset: [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction),
May 2014 through May 2015, whose source card declares
[CC0](https://creativecommons.org/publicdomain/zero/1.0/).

- [Public download endpoint](https://www.kaggle.com/api/v1/datasets/download/harlfoxem/housesalesprediction)
- Raw member: `kc_house_data.csv`
- SHA256: `d0875baa0251b21d4bdc9d2ae940a4fe0bb6009824f23dd0e2a5b2bf04557b7e`

The inspected [author script](https://github.com/cvxgrp/strat_models/blob/0108114afe1265262f345bf93a5bc147f6cb146c/examples/house.py)
uses one holdout, 50 forest trees, different penalty weights, and penalizes the intercept.
The paper's parameter count also differs from its stated grid × feature dimensions;
this notebook reports its actual coefficient count. Modern forest implementations,
unknown original folds and boundary details further preclude promising exact published
RMSE. A measured agreement claim must identify these choices and the run evidence.

## Loss and penalty units

For a paper objective `sum(loss_i) + lambda * R`, an equivalent globally mean objective
has weight `lambda / N`, where `N` is the current training fold's sample count. If the
new per-sample loss is also multiplied by `c`, the equivalent weight is `c * lambda / N`.
This rescales the complete objective by a positive constant; it does not change its
minimizer. It does change numerical objective values, residual scales and suitable
solver tolerances.

The package uses:

```text
regression loss = sum((prediction - target)^2) / (2 N)
binary loss     = sum(log(1 + exp(score)) - target * score) / N
local L2        = lambda / 2 * sum(coefficients^2)
graph penalty   = gamma / 2 * sum_undirected_edges(w * coefficient_difference^2)
```

The conventions are checked separately, rather than inferred from the phrase “squared
regularization.” Section 5.2 names the house loss and regularizer but does not display an
experiment-specific formula for `r`. Appendix A, p.28, explicitly gives its sum-of-squares
function as `f(theta) = ||theta||² / 2`, with proximal shrinkage `nu / (t + 1)`.
The inspected [author regularizer](https://github.com/cvxgrp/strat_models/blob/0108114afe1265262f345bf93a5bc147f6cb146c/strat_models/regularizers.py#L39)
implements `lambda * sum(theta²) / 2`. Its
[squared-loss proximal solve](https://github.com/cvxgrp/strat_models/blob/0108114afe1265262f345bf93a5bc147f6cb146c/strat_models/losses.py#L164)
independently confirms **unhalved** summed squared residuals. Equation (2), p.7,
places one half on the sum over undirected graph edges.

We therefore declare the following source-supported interpretation explicitly:

| Experiment | Paper/source local penalty | Package L2 weight | Package graph weight |
| --- | --- | --- | --- |
| House prices | `lambda * sum(theta²) / 2` | `lambda / (2N)` | `gamma / (2N)` |
| Binary classification | `lambda * sum(theta²) / 2` | `lambda / N` | `gamma / N` |

If one instead interpreted the paper's house local penalty as **unhalved**
`lambda * sum(theta²)`, its package L2 weight would be `lambda / N`, while its graph
weight would still be `gamma / (2N)`. That is a different declared convention from the
inspected author regularizer, and is not silently substituted here. The notebook's
published-setting preset follows the half-scaled local regularizer supported by the
appendix and source; it does not claim that §5.2 alone uniquely settles the notation.

The house preset consequently passes `c=0.5`; logistic uses `c=1`. Graph edges are counted once. Penalty weights
are never divided by batch count, per-node sample count, or number of strata. Conversion
is explicit in each notebook's recipe builder, including the final full-data refit.

## Verification and boundaries

The data helper checks extracted-member hashes, rejects changed cached data, bounds
download/archive size, writes caches atomically and never downloads without opt-in.
Its offline tests cover these contracts, target leakage exclusion, declared category
domains, geography/target preparation, deterministic adaptations and objective-unit
equivalence. No live-data download is part of the ordinary unit suite.

Release verification also executes the quick notebooks headlessly and inspects their
rendered figures. Original-data presets are explicit integration experiments: retain
their downloaded manifests, measured scores and numerical diagnostics when reporting
results. Failure to converge is visible and disqualifies a candidate from selection.

Small measured original-data summaries are tracked in
[reproduction_results.json](reproduction_results.json) and
[reproduction_results.csv](reproduction_results.csv). They include the classification
intercept-mask sensitivity. Each record distinguishes completed statistical evaluation
from numerical convergence of every fit; a failed candidate has null scores rather than
an accepted approximate prediction. Full fold manifests and diagnostic tables live in
the ignored local `notebook_exports/evidence/` delivery directory. No patient feature
rows are included in these summaries or manifests.

The original paper's Senate experiment smooths probability parameters under box
constraints; logit smoothing would be a different model. The Chicago crime experiment
uses Poisson loss and a graph with millions of strata. Neither is presented as a faithful
v1 reproduction. L1 extensions in other review notebooks are library demonstrations,
not alterations silently attributed to these two published experiments.
