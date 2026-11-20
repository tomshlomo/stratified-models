# Stratified Models · Notebook gallery

**[Open the hosted notebook gallery](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/00_review_gallery.py)**

Six executed quick-run notebooks explore graph regularization, regression and classification, solver behavior, streaming data, and reusable fitted artifacts.

**The saved default outputs are read-only.** Live sliders, reruns, and larger experiments require a Python server with marimo and JAX. The hosted views below follow the preview branch; the source links and local setup use the pinned implementation revision.

[Review pull request #5](https://github.com/tomshlomo/stratified-models/pull/5) · [Notebook guide](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/docs/notebooks.md) · [Source snapshot `39f8289`](https://github.com/tomshlomo/stratified-models/tree/39f8289a3631e0477735450c2e7f731fb305ac2e)

## Explore the notebooks

1. **[Graph regression](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/01_graph_regression.py)**  
   See how graph structure, regularization and a missing training group shape coefficients and held-out error. Compare pooled, independent and stratified fits against known truth.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/01_graph_regression.py)

2. **[Mesothelioma classification](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/02_mesothelioma.py)**  
   Review probability estimates, held-out scores and age/sex coefficient curves. The quick run is synthetic; the source includes the original-data, paper-method protocol.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/02_mesothelioma.py)

3. **[Spatial house prices](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/03_house_prices.py)**  
   Inspect spatial coefficient maps and held-out log-price errors against pooled and separate models. The synthetic quick run introduces the full paper-method preset available in the source.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/03_house_prices.py)

4. **[Solver diagnostics and reuse](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/04_solver_diagnostics.py)**  
   Compare eligible solvers on the same objective. Inspect ADMM residual thresholds, numerical certificates, typed nonconvergence, warm starts and shared numerical caches.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/04_solver_diagnostics.py)

5. **[Streaming and composition](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/05_streaming.py)**  
   Follow one data snapshot through in-memory and batched fitting with pandas, Polars or Parquet. Compare results, explicit preprocessing, bin boundaries and data-replay costs.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/05_streaming.py)

6. **[From experiment to artifact](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/06_workflow.py)**  
   Review discrimination and calibration, fold-isolated tuning, sklearn cloning and versioned prediction artifacts. See fitted components round-trip and failures stay explicit.  
   [Python source](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/examples/06_workflow.py)

<details>
<summary><strong>Run with live controls</strong></summary>

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands in a terminal:

```sh
git clone https://github.com/tomshlomo/stratified-models.git
cd stratified-models
git checkout 39f8289a3631e0477735450c2e7f731fb305ac2e
uv sync --frozen --extra common --group dev
uv run --extra common marimo run examples/01_graph_regression.py
```

Open the local address printed by marimo. Change the final filename to explore another notebook. Use `marimo edit` instead of `marimo run` to inspect or change its Python cells. Larger experiments and external data downloads require explicit actions inside the notebooks.

</details>

[Scientific conventions and verification](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/docs/notebooks.md) · [Paper reproduction protocols](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/docs/paper_reproductions.md)

## Refresh the saved previews

From the publication branch, execute the quick defaults and extract the saved outputs:

```sh
uv sync --extra common --group dev
uv run python scripts/check_notebooks.py
uv run python scripts/export_notebook_sessions.py \
  --source-dir examples --output-dir examples/__marimo__/session \
  notebook_exports/[0-9][0-9]_*.html
```

The extractor checks the source hashes and restores every cell before writing a snapshot. Review the generated outputs, then explicitly add the selected JSON files with `git add -f examples/__marimo__/session/*.py.json`; this directory is otherwise ignored. Publishing these files to `notebook-previews` updates the linked molab views. The numerical implementation in this initial publication is commit `39f8289a3631e0477735450c2e7f731fb305ac2e`.
