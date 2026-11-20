import attrs
import numpy as np
import pytest

from stratified_models.errors import UnknownNodeError, ValidationError
from stratified_models.preprocessing import TransformChain
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


def test_axis_preserves_declared_label_order_and_rejects_unknown_labels() -> None:
    labels = ["west", "east"]
    axis = Axis(name="region", nodes=labels)
    labels.reverse()
    assert axis.nodes == ("west", "east")
    np.testing.assert_array_equal(axis.encode(["east", "west", "east"]), [1, 0, 1])
    with pytest.raises(UnknownNodeError, match="unknown node"):
        axis.encode(["south"])
    with pytest.raises(UnknownNodeError):
        Axis(name="code", nodes=(1, 2)).encode([True])
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        axis.name = "changed"


def test_graph_edges_use_labels_and_path_obeys_axis_order() -> None:
    axis = Axis(name="region", nodes=("z", "a", "m"))
    graph = Graph.path(axis=axis, weight=2.0)
    assert graph.edges == (("z", "a", 2.0), ("a", "m", 2.0))
    assert graph.indexed_edges == ((0, 1, 2.0), (1, 2, 2.0))
    assert Graph.path(axis=Axis(name="singleton", nodes=(0,))).edges == ()


@pytest.mark.parametrize(
    "edges",
    [
        ((0, 0, 1.0),),
        ((0, 1, -1.0),),
        ((0, 1, np.nan),),
        ((0, 2, 1.0),),
        ((0, 1, 1.0), (1, 0, 2.0)),
    ],
)
def test_graph_rejects_invalid_edges(edges: tuple) -> None:
    with pytest.raises(ValidationError):
        Graph(axis=Axis(name="group", nodes=(0, 1)), edges=edges)


def test_penalty_masks_and_graph_coordinates_are_validated() -> None:
    axis = Axis(name="region", nodes=("a", "b"))
    reordered = Axis(name="region", nodes=("b", "a"))
    with pytest.raises(ValidationError, match="unknown features"):
        RegressionSpec(
            features=("x",),
            penalties=(L2(name="ridge", weight=1.0, features=("typo",)),),
        )
    with pytest.raises(ValidationError, match="node order"):
        RegressionSpec(
            features=("x",),
            axes=(axis,),
            penalties=(
                Laplacian(
                    name="smooth",
                    weight=1.0,
                    features=("x",),
                    graph=Graph.path(axis=reordered),
                ),
            ),
        )
    for invalid in (-1.0, np.inf, np.nan):
        with pytest.raises(ValidationError, match="finite and nonnegative"):
            L1(name="lasso", weight=invalid, features=("x",))
    with pytest.raises(ValidationError, match="must not be empty"):
        L2(name="ridge", weight=1.0, features=())


def test_pipeline_weight_reference_is_checked_and_model_inputs_are_snapshots() -> None:
    features = ["x", "intercept"]
    penalties = [L1(name="lasso", weight=0.1, features=("x",))]
    model = BinaryLogisticSpec(features=features, penalties=penalties)
    pipeline = Pipeline(preprocessing=TransformChain(), model=model)
    features.append("bad")
    penalties.clear()
    assert model.features == ("x", "intercept")
    assert len(model.penalties) == 1
    assert model.task == "binary"
    assert pipeline.weight("lasso").name == "lasso"
    with pytest.raises(ValidationError, match="unknown penalty"):
        pipeline.weight("typo")
    with pytest.raises(ValidationError, match="unique"):
        RegressionSpec(features=("x", "x"))
