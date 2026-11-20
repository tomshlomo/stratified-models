import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stratified Models
    ### Six experiments to explore and review

    Follow graph regularization from coefficient fields to classification,
    optimization, streaming data, and reusable fitted artifacts.

    **These hosted notebooks show saved, executed quick-run outputs.**
    Treat their controls as read-only. Live sliders, reruns, original-data
    reproductions, and full-size presets require a Python server with marimo
    and JAX. The local instructions below open that experience.

    [Review pull request #5](https://github.com/tomshlomo/stratified-models/pull/5)
    · [Notebook guide](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/docs/notebooks.md)
    · [Source snapshot `39f8289`](https://github.com/tomshlomo/stratified-models/tree/39f8289a3631e0477735450c2e7f731fb305ac2e)

    ---

    ## Choose a notebook

    ### 01 · [Graph regression](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/01_graph_regression.py)

    See how graph structure, regularization, and a missing training group
    shape coefficients and held-out error. Compare pooled, independent,
    and stratified fits against known truth.

    ### 02 · [Mesothelioma classification](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/02_mesothelioma.py)

    Review probability estimates, held-out scores, and age/sex coefficient
    curves. The quick run is synthetic; the source includes an explicit
    original-data, paper-method protocol.

    ### 03 · [Spatial house prices](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/03_house_prices.py)

    Inspect spatial coefficient maps and log-price errors against pooled
    and separate models. The synthetic quick run introduces the full
    paper-method preset available in the source.

    ### 04 · [Solver diagnostics and reuse](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/04_solver_diagnostics.py)

    Compare eligible solvers on the same objective. Inspect ADMM residual
    thresholds, numerical certificates, typed nonconvergence, warm starts,
    and shared numerical caches.

    ### 05 · [Streaming and composition](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/05_streaming.py)

    Follow one data snapshot through in-memory and batched fitting with
    pandas, Polars, or Parquet. Compare results, explicit preprocessing,
    bin boundaries, and data-replay costs.

    ### 06 · [From experiment to artifact](https://molab.marimo.io/github/tomshlomo/stratified-models/blob/notebook-previews/examples/06_workflow.py)

    Review discrimination and calibration, fold-isolated tuning, sklearn
    cloning, and versioned prediction artifacts. See fitted components
    round-trip and failures stay explicit.

    ---

    ## Run with live controls

    Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
    then run these commands in a terminal. They select the implementation
    revision used for this review.

    ```sh
    git clone https://github.com/tomshlomo/stratified-models.git
    cd stratified-models
    git checkout 39f8289a3631e0477735450c2e7f731fb305ac2e
    uv sync --frozen --extra common --group dev
    uv run --extra common marimo run examples/01_graph_regression.py
    ```

    Open the local address printed by marimo. Change the final filename
    to explore another notebook. Use `marimo edit` instead of `marimo run`
    to inspect or change its Python cells. Larger experiments and external
    data downloads require explicit actions inside the notebooks.

    [Paper reproduction protocols](https://github.com/tomshlomo/stratified-models/blob/39f8289a3631e0477735450c2e7f731fb305ac2e/docs/paper_reproductions.md)
    · [Browse the six notebook sources](https://github.com/tomshlomo/stratified-models/tree/39f8289a3631e0477735450c2e7f731fb305ac2e/examples)
    """)
    return


if __name__ == "__main__":
    app.run()
