"""Prediction artifacts round-trip and reject corrupt or incompatible state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import attrs
import numpy as np
import pytest

from stratified_models.artifacts import (
    CodecRegistry,
    ComponentCodec,
    load_artifact,
    load_binary_classifier,
    load_regressor,
    save_artifact,
)
from stratified_models.data import FeatureBatch, TrainingBatch
from stratified_models.errors import ArtifactError
from stratified_models.predict import FittedBinaryClassifier, FittedRegressor
from stratified_models.preprocessing import (
    AddIntercept,
    FittedAddIntercept,
    FittedFixedWidthBins,
    FittedStandardize,
    FittedTransformChain,
    FixedWidthBins,
    Standardize,
    TransformChain,
)
from stratified_models.specs import (
    L1,
    L2,
    Axis,
    BinaryLogisticSpec,
    Graph,
    Laplacian,
    Pipeline,
    RegressionSpec,
)


def _predictor(*, binary: bool = False) -> FittedRegressor | FittedBinaryClassifier:
    axis = Axis(name="bin", nodes=(0, 1))
    penalties = (
        L1(name="sparse", weight=0.1, features=("x",)),
        L2(name="ridge", weight=0.2, features=("x",)),
        Laplacian(
            name="smooth",
            weight=0.3,
            features=("x", "intercept"),
            graph=Graph.path(axis=axis),
        ),
    )
    constructor = BinaryLogisticSpec if binary else RegressionSpec
    model = constructor(features=("x", "intercept"), axes=(axis,), penalties=penalties)
    recipe = TransformChain(
        steps=(
            FixedWidthBins(
                column="position", output="bin", origin=0.0, width=1.0, n_bins=2
            ),
            Standardize(columns=("x",)),
            AddIntercept(name="intercept"),
        )
    )
    transformers = FittedTransformChain(
        steps=(
            FittedFixedWidthBins(
                column="position", output="bin", origin=0.0, width=1.0, n_bins=2
            ),
            FittedStandardize(
                columns=("x",), means=np.array([2.0]), scales=np.array([3.0])
            ),
            FittedAddIntercept(name="intercept"),
        )
    )
    predictor_type = FittedBinaryClassifier if binary else FittedRegressor
    return predictor_type(
        pipeline=Pipeline(preprocessing=recipe, model=model),
        transformers=transformers,
        coefficients=np.array([[1.0, 2.0], [-2.0, -1.0]]),
        weights={"sparse": 0.4, "ridge": 0.5, "smooth": 0.6},
    )


@pytest.mark.parametrize("binary", [False, True])
def test_builtin_prediction_state_round_trips(tmp_path: Path, binary: bool) -> None:
    predictor = _predictor(binary=binary)
    path = save_artifact(predictor, tmp_path / "predictor")
    restored = load_artifact(path)
    data = {"x": np.array([1.0, 4.0]), "position": np.array([0.2, 1.3])}
    np.testing.assert_array_equal(restored.predict(data), predictor.predict(data))
    np.testing.assert_array_equal(restored.coefficients, predictor.coefficients)
    assert restored.weights == predictor.weights
    assert restored.training_diagnostics is None
    assert not restored.coefficients.flags.writeable
    manifest = json.loads((path / "manifest.json").read_text())
    assert set(manifest) == {
        "format",
        "version",
        "model",
        "preprocessing",
        "transformers",
        "weights",
        "coefficients",
        "arrays",
        "payload",
    }
    if binary:
        assert isinstance(predictor, FittedBinaryClassifier)
        restored_classifier = load_binary_classifier(path)
        np.testing.assert_array_equal(
            restored_classifier.predict_proba(data), predictor.predict_proba(data)
        )
        with pytest.raises(ArtifactError, match="not a regressor"):
            load_regressor(path)
    else:
        assert isinstance(load_regressor(path), FittedRegressor)
        with pytest.raises(ArtifactError, match="not a binary classifier"):
            load_binary_classifier(path)


def test_payload_corruption_and_unknown_version_are_explicit(tmp_path: Path) -> None:
    path = save_artifact(_predictor(), tmp_path / "predictor")
    payload = path / "arrays.npz"
    payload.write_bytes(payload.read_bytes() + b"corruption")
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(path)
    save_artifact(_predictor(), path, overwrite=True)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="version"):
        load_artifact(path)


def test_header_shape_and_memory_budget_are_checked(tmp_path: Path) -> None:
    path = save_artifact(_predictor(), tmp_path / "predictor")
    with pytest.raises(ArtifactError, match="budget"):
        load_artifact(path, max_bytes=8)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arrays"]["coefficients"]["shape"] = [1, 4]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="header"):
        load_artifact(path)


@dataclass(frozen=True)
class Shift:
    offset: float

    @property
    def fingerprint(self) -> str:
        return f"test-shift:{self.offset}"

    def output_columns(self, columns: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(columns)

    def transform[B: (FeatureBatch, TrainingBatch)](self, batch: B) -> B:
        return attrs.evolve(
            batch, columns={**batch.columns, "x": batch.columns["x"] + self.offset}
        )


def test_custom_components_require_an_explicit_registry(tmp_path: Path) -> None:
    shift = Shift(2.0)
    model = RegressionSpec(features=("x",))
    predictor = FittedRegressor(
        pipeline=Pipeline(preprocessing=TransformChain(steps=(shift,)), model=model),
        transformers=FittedTransformChain(steps=(shift,)),
        coefficients=np.array([[3.0]]),
        weights={},
    )
    with pytest.raises(ArtifactError, match="explicit codec"):
        save_artifact(predictor, tmp_path / "predictor")
    codec = ComponentCodec(
        name="test.shift.v1",
        component_type=Shift,
        encode=lambda value: {"offset": value.offset},
        decode=lambda state: Shift(float(state["offset"])),
    )
    registry = CodecRegistry(codecs=(codec,))
    path = save_artifact(predictor, tmp_path / "predictor", registry=registry)
    with pytest.raises(ArtifactError, match="supply it explicitly"):
        load_regressor(path)
    restored = load_regressor(path, registry=registry)
    np.testing.assert_array_equal(restored.predict({"x": np.array([1.0])}), [9.0])
