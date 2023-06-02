# stratified-models
A python package for training stratified machine learning models with laplacian regularization.
\
Based on and inspired by these great papers:
* [Tuck, Jonathan, Shane Barratt, and Stephen Boyd. "A distributed method for fitting Laplacian regularized stratified models."](https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf)
* [Tuck, Jonathan, and Stephen Boyd. "Fitting Laplacian regularized stratified Gaussian models."](https://web.stanford.edu/~boyd/papers/pdf/cov_strat_models.pdf)
* [Tuck, Jonathan, and Stephen Boyd. "Eigen-stratified models."](https://web.stanford.edu/~boyd/papers/eigen_strat.html)

A work in progress.

## Todos:

### CI/CD:
* fix cvxpy installation in nox
* run pre-commit in nox (or, if it's too hard, in ci)
* dependabot
* uncomment `stratified_models/__init__.py`?
* coverage report and badge
* pack

### Accelerations:
* replace pandas with dask/polars
* replace numpy with dask arrays/jax/torch
* implement a multi-threaded/processing version of ADMM (rust or dask?)
* implement prox of graph and path graphs using FFT


### Usability:
* util to transform a continuous feature to discrete and automatically get the right path graph for it.
  * constant width
  * percentile
* util to normalize target and regressors before optimization
* util to create common models:
  * ridge
  * lasso
  * ols
  * logistic regression
  * svm? (would require to compute the prox for the hinge loss)
* integration with optuna for hyperparameter optimization out of the box
* util for featureless prediction (non parametric)

### Algorithms:
* a score function that approximates the leave one out cross validation, or SURE. relevant papers:
  * [Tractable Evaluation of Stein’s Unbiased Risk
Estimator with Convex Regularizers](https://web.stanford.edu/~boyd/papers/pdf/sure_tractable_eval.pdf)
  * [Optimizing Approximate Leave-one-out Cross-validation to
Tune Hyperparameters](https://arxiv.org/pdf/2011.10218.pdf)
  * [Approximate Leave-One-Out for High-Dimensional
Non-Differentiable Learning Problems](https://arxiv.org/pdf/1810.02716.pdf)
* more losses:
  * logistic
  * poisson negative log likelihood
  * hinge
  * huber
* (major) probabilistic and non-scalar predictions
  * normal
  * CDF
  * logits
* more options for proxes of networkx laplacians:
  * sparse eigh
  * cg with diagonal preconditioner
* (major) eigen stratified models (constrain theta to low graph-frequencies)
* (major) graph learning, e.g [Joint Graph Learning and Model Fitting in Laplacian
Regularized Stratified Models](https://arxiv.org/pdf/2305.02573.pdf)
* smart initialization strategies:
  * no stratification
  * no stratification, then train a strat model using only the prediction
