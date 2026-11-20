"""Immutable, validated model and graph definitions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from stratified_models.errors import SchemaError, UnknownNodeError, ValidationError

if TYPE_CHECKING:
    from stratified_models.preprocessing import TransformChain

type NodeLabel = str | int
type Edge = tuple[NodeLabel, NodeLabel, float]


def _tuple[T](values: Iterable[T]) -> tuple[T, ...]:
    return tuple(values)


def validate_name(value: str, *, kind: str = "name") -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        message = f"{kind} must be a nonempty string without surrounding whitespace"
        raise ValidationError(message)


def validate_names(
    values: Sequence[str], *, kind: str, allow_empty: bool = False
) -> None:
    if not values and not allow_empty:
        message = f"{kind} must not be empty"
        raise ValidationError(message)
    for value in values:
        validate_name(value, kind=kind)
    if len(set(values)) != len(values):
        message = f"{kind} must contain unique names"
        raise ValidationError(message)


def _label(value: object) -> NodeLabel:
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        message = "axis labels must be strings or integers (excluding booleans)"
        raise ValidationError(message)
    return value


def _labels(values: Iterable[NodeLabel]) -> tuple[NodeLabel, ...]:
    return tuple(_label(value) for value in values)


@attrs.frozen(kw_only=True)
class Axis:
    name: str
    nodes: tuple[NodeLabel, ...] = attrs.field(converter=_labels)

    def __attrs_post_init__(self) -> None:
        validate_name(self.name, kind="axis name")
        if not self.nodes:
            message = "an axis must contain at least one node"
            raise ValidationError(message)
        if len(set(self.nodes)) != len(self.nodes):
            message = f"axis {self.name!r} contains duplicate node labels"
            raise ValidationError(message)

    @property
    def size(self) -> int:
        return len(self.nodes)

    def index(self, label: NodeLabel) -> int:
        try:
            label = _label(label)
            return self.nodes.index(label)
        except (ValueError, ValidationError) as error:
            message = f"unknown node {label!r} for axis {self.name!r}"
            raise UnknownNodeError(message) from error

    def encode(self, values: ArrayLike) -> NDArray[np.int64]:
        observed = np.asarray(values, dtype=object)
        if observed.ndim != 1:
            message = f"axis {self.name!r} requires one-dimensional labels"
            raise SchemaError(message)
        lookup = {label: index for index, label in enumerate(self.nodes)}
        encoded = np.empty(len(observed), dtype=np.int64)
        for row, value in enumerate(observed):
            label = value.item() if isinstance(value, np.generic) else value
            try:
                label = _label(label)
                encoded[row] = lookup[label]
            except (KeyError, TypeError, ValidationError) as error:
                message = f"unknown node {label!r} at row {row} for axis {self.name!r}"
                raise UnknownNodeError(message) from error
        return np.frombuffer(encoded.tobytes(), dtype=np.int64)


def _edges(values: Iterable[Edge]) -> tuple[Edge, ...]:
    edges = []
    for edge in values:
        if len(edge) != 3:
            message = "each edge must be (left node, right node, weight)"
            raise ValidationError(message)
        left, right, weight = edge
        edges.append((_label(left), _label(right), float(weight)))
    return tuple(edges)


@attrs.frozen(kw_only=True)
class Graph:
    axis: Axis
    edges: tuple[Edge, ...] = attrs.field(converter=_edges)

    def __attrs_post_init__(self) -> None:
        seen: set[frozenset[NodeLabel]] = set()
        for left, right, weight in self.edges:
            self.axis.index(left)
            self.axis.index(right)
            if left == right:
                message = "self-loop edges are not supported"
                raise ValidationError(message)
            pair = frozenset((left, right))
            if pair in seen:
                message = f"duplicate undirected graph edge {left!r}, {right!r}"
                raise ValidationError(message)
            seen.add(pair)
            if not math.isfinite(weight) or weight < 0:
                message = "graph edge weights must be finite and nonnegative"
                raise ValidationError(message)

    @property
    def indexed_edges(self) -> tuple[tuple[int, int, float], ...]:
        lookup = {node: index for index, node in enumerate(self.axis.nodes)}
        return tuple(
            (lookup[left], lookup[right], weight) for left, right, weight in self.edges
        )

    @classmethod
    def path(cls, *, axis: Axis, weight: float = 1.0) -> Graph:
        if not math.isfinite(weight) or weight < 0:
            message = "graph edge weights must be finite and nonnegative"
            raise ValidationError(message)
        return cls(
            axis=axis,
            edges=tuple(
                (left, right, weight)
                for left, right in zip(axis.nodes[:-1], axis.nodes[1:], strict=True)
            ),
        )


def _validate_penalty(name: str, weight: float, features: tuple[str, ...]) -> None:
    validate_name(name, kind="penalty name")
    validate_names(features, kind="penalty features")
    if not math.isfinite(weight) or weight < 0:
        message = "penalty weights must be finite and nonnegative"
        raise ValidationError(message)


@attrs.frozen(kw_only=True)
class L1:
    name: str
    weight: float = attrs.field(converter=float)
    features: tuple[str, ...] = attrs.field(converter=_tuple)

    def __attrs_post_init__(self) -> None:
        _validate_penalty(self.name, self.weight, self.features)


@attrs.frozen(kw_only=True)
class L2:
    name: str
    weight: float = attrs.field(converter=float)
    features: tuple[str, ...] = attrs.field(converter=_tuple)

    def __attrs_post_init__(self) -> None:
        _validate_penalty(self.name, self.weight, self.features)


@attrs.frozen(kw_only=True)
class Laplacian:
    name: str
    weight: float = attrs.field(converter=float)
    features: tuple[str, ...] = attrs.field(converter=_tuple)
    graph: Graph

    def __attrs_post_init__(self) -> None:
        _validate_penalty(self.name, self.weight, self.features)


type Penalty = L1 | L2 | Laplacian


def _validate_model(
    features: tuple[str, ...], axes: tuple[Axis, ...], penalties: tuple[Penalty, ...]
) -> None:
    validate_names(features, kind="model features")
    validate_names(
        tuple(axis.name for axis in axes), kind="axis names", allow_empty=True
    )
    validate_names(
        tuple(penalty.name for penalty in penalties),
        kind="penalty names",
        allow_empty=True,
    )
    feature_set = set(features)
    axes_by_name = {axis.name: axis for axis in axes}
    for penalty in penalties:
        unknown = set(penalty.features).difference(feature_set)
        if unknown:
            message = f"penalty {penalty.name!r} references unknown features: {sorted(unknown)}"
            raise ValidationError(message)
        if (
            isinstance(penalty, Laplacian)
            and axes_by_name.get(penalty.graph.axis.name) != penalty.graph.axis
        ):
            message = f"graph axis for penalty {penalty.name!r} does not match the declared model axis and node order"
            raise ValidationError(message)


@attrs.frozen(kw_only=True)
class RegressionSpec:
    features: tuple[str, ...] = attrs.field(converter=_tuple)
    axes: tuple[Axis, ...] = attrs.field(converter=_tuple, factory=tuple)
    penalties: tuple[Penalty, ...] = attrs.field(converter=_tuple, factory=tuple)

    def __attrs_post_init__(self) -> None:
        _validate_model(self.features, self.axes, self.penalties)

    @property
    def task(self) -> Literal["regression"]:
        return "regression"


@attrs.frozen(kw_only=True)
class BinaryLogisticSpec:
    features: tuple[str, ...] = attrs.field(converter=_tuple)
    axes: tuple[Axis, ...] = attrs.field(converter=_tuple, factory=tuple)
    penalties: tuple[Penalty, ...] = attrs.field(converter=_tuple, factory=tuple)

    def __attrs_post_init__(self) -> None:
        _validate_model(self.features, self.axes, self.penalties)

    @property
    def task(self) -> Literal["binary"]:
        return "binary"


type ModelSpec = RegressionSpec | BinaryLogisticSpec


@attrs.frozen(kw_only=True)
class WeightRef:
    name: str

    def __attrs_post_init__(self) -> None:
        validate_name(self.name, kind="penalty reference")


@attrs.frozen(kw_only=True)
class Pipeline[M: ModelSpec]:
    preprocessing: TransformChain
    model: M

    def weight(self, name: str) -> WeightRef:
        if name not in {penalty.name for penalty in self.model.penalties}:
            message = f"unknown penalty {name!r}"
            raise ValidationError(message)
        return WeightRef(name=name)
