# Benchmark Optimization Notes

## Baseline
- Config: kmeans_k=10, house_age_bins=5, n_folds=3, n_trials=5
- Tuning time: 101.0s
- Test R²: 0.5401 (baseline LR: 0.5758)

## Experiments

### 1. Add jax.jit to objective/grad in solve() ✓
- **Result**: 101s → 42s (2.4x speedup)
- **Key insight**: Must warm up cached_property (SparseQuadraticForm.a_jax) before jitting to avoid tracer leaks
- **Files changed**: `src/stratified_models/solvers/newton.py`

### 2. Add jax.jit to CG matvec ?
- **Result on small benchmark**: 42s → 46s (slower)
- **Reason**: JIT compilation overhead per Newton step outweighs benefits when only 2 iterations needed
- **Hypothesis**: May help on larger problems with more CG iterations
- **Status**: Kept enabled - worth having for larger problems

## Observations
- Stratified model currently underperforms baseline LR (likely needs more trials or tuned hyperparameter ranges)
- Newton converges in 2 iterations for this problem (quadratic loss + graph regularization)

## TODO
- Tune hyperparameter search ranges
- Try more trials
- Profile to find remaining bottlenecks
- Re-test matvec jit with larger problems (more strata → more CG iterations)
