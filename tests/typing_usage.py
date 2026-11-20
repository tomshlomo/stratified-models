"""Public typing contracts; checked explicitly by CI rather than executed."""

from typing import assert_type

from stratified_models import (
    L2,
    AddIntercept,
    AutoSolver,
    BinaryLogisticSpec,
    Choice,
    FitFailure,
    FitResult,
    FittedBinaryClassifier,
    FittedRegressor,
    Pipeline,
    RegressionSpec,
    SolveSession,
    WeightSpace,
    compose,
    fit,
    fit_batches,
    prepare,
    prepare_batches,
    source_from_data,
)


def public_fit_types() -> None:
    features = {"x": [-1.0, 0.0, 1.0, 2.0]}
    targets = [0.0, 1.0, 0.0, 1.0]
    regression = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=RegressionSpec(
            features=("x", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x",)),),
        ),
    )
    binary = Pipeline(
        preprocessing=compose(AddIntercept(name="intercept")),
        model=BinaryLogisticSpec(
            features=("x", "intercept"),
            penalties=(L2(name="ridge", weight=0.1, features=("x",)),),
        ),
    )
    reg_result = fit(regression, features, targets, solver=AutoSolver())
    binary_result = fit(binary, features, targets, solver=AutoSolver())
    assert_type(reg_result, FitResult[FittedRegressor])
    assert_type(binary_result, FitResult[FittedBinaryClassifier])
    if isinstance(reg_result, FitFailure):
        assert_type(reg_result, FitFailure)
    else:
        assert_type(reg_result.predictor, FittedRegressor)
    if isinstance(binary_result, FitFailure):
        assert_type(binary_result, FitFailure)
    else:
        assert_type(binary_result.predictor, FittedBinaryClassifier)

    prepared_regression = prepare(regression, features, targets)
    prepared_binary = prepare(binary, features, targets)
    reg_session = SolveSession(prepared_regression, solver=AutoSolver())
    binary_session = SolveSession(prepared_binary, solver=AutoSolver())
    reg_repeated = reg_session.solve(weight_overrides={"ridge": 0.2})
    binary_repeated = binary_session.solve(
        weight_overrides={binary.weight("ridge"): 0.2}
    )
    assert_type(reg_repeated, FitResult[FittedRegressor])
    assert_type(binary_repeated, FitResult[FittedBinaryClassifier])
    if not isinstance(reg_repeated, FitFailure):
        assert_type(reg_repeated.predictor, FittedRegressor)
    if not isinstance(binary_repeated, FitFailure):
        assert_type(binary_repeated.predictor, FittedBinaryClassifier)

    source = source_from_data(features, targets)
    reg_batches = fit_batches(regression, source, solver=AutoSolver())
    binary_batches = fit_batches(binary, source, solver=AutoSolver())
    assert_type(reg_batches, FitResult[FittedRegressor])
    assert_type(binary_batches, FitResult[FittedBinaryClassifier])
    assert_type(
        SolveSession(prepare_batches(regression, source), solver=AutoSolver()).solve(),
        FitResult[FittedRegressor],
    )
    assert_type(
        SolveSession(prepare_batches(binary, source), solver=AutoSolver()).solve(),
        FitResult[FittedBinaryClassifier],
    )
    WeightSpace(ranges={"ridge": Choice(values=(0.0, 0.1))})
    WeightSpace(ranges={regression.weight("ridge"): Choice(values=(0.0, 0.1))})
