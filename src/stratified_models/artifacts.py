"""Versioned prediction artifacts containing JSON and checked numerical arrays.

Artifacts never import classes named in their manifest. Custom components are
decoded only through codecs explicitly supplied by the caller.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from stratified_models.errors import ArtifactError, ValidationError
from stratified_models.predict import FittedBinaryClassifier, FittedRegressor
from stratified_models.preprocessing import (
    AddIntercept,
    FittedAddIntercept,
    FittedFixedWidthBins,
    FittedStandardize,
    FittedTransform,
    FittedTransformChain,
    FixedWidthBins,
    Standardize,
    TransformChain,
    TransformRecipe,
)
from stratified_models.specs import (
    L1,
    L2,
    Axis,
    BinaryLogisticSpec,
    Graph,
    Laplacian,
    ModelSpec,
    Penalty,
    Pipeline,
    RegressionSpec,
)

type JsonValue = (
    None | bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue]
)
type Predictor = FittedRegressor | FittedBinaryClassifier

_FORMAT = "stratified-models-predictor"
_VERSION = 1
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024


class Codec(Protocol):
    """An explicitly installed codec; artifact files cannot install codecs."""

    @property
    def name(self) -> str: ...

    def matches(self, value: object) -> bool: ...

    def encode_value(self, value: object) -> Mapping[str, JsonValue]: ...

    def decode_value(self, state: Mapping[str, JsonValue]) -> object: ...


@dataclass(frozen=True, kw_only=True)
class ComponentCodec[T]:
    name: str
    component_type: type[T]
    encode: Callable[[T], Mapping[str, JsonValue]]
    decode: Callable[[Mapping[str, JsonValue]], T]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            msg = "A custom codec needs a nonempty, stable name."
            raise ArtifactError(msg)

    def matches(self, value: object) -> bool:
        return type(value) is self.component_type

    def encode_value(self, value: object) -> Mapping[str, JsonValue]:
        if not isinstance(value, self.component_type):
            msg = f"Codec {self.name!r} received an incompatible component."
            raise ArtifactError(msg)
        return self.encode(value)

    def decode_value(self, state: Mapping[str, JsonValue]) -> T:
        component = self.decode(state)
        if not isinstance(component, self.component_type):
            msg = f"Codec {self.name!r} returned an incompatible component."
            raise ArtifactError(msg)
        return component


@dataclass(frozen=True, kw_only=True)
class CodecRegistry:
    codecs: tuple[Codec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "codecs", tuple(self.codecs))
        names = [codec.name for codec in self.codecs]
        if len(set(names)) != len(names):
            msg = "Custom codec names must be unique."
            raise ArtifactError(msg)

    def encode(self, component: object) -> dict[str, JsonValue]:
        matches = [codec for codec in self.codecs if codec.matches(component)]
        if len(matches) != 1:
            msg = f"Provide one explicit codec for component {type(component).__qualname__!r}."
            raise ArtifactError(msg)
        codec = matches[0]
        return {
            "type": "custom",
            "codec": codec.name,
            "state": dict(codec.encode_value(component)),
        }

    def decode(self, name: str, state: Mapping[str, JsonValue]) -> object:
        for codec in self.codecs:
            if codec.name == name:
                return codec.decode_value(state)
        msg = f"Artifact requires custom codec {name!r}; supply it explicitly."
        raise ArtifactError(msg)


_BUILTIN_REGISTRY = CodecRegistry()


def _mapping(value: object, name: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        msg = f"Artifact {name} must be a JSON object."
        raise ArtifactError(msg)
    return cast("dict[str, JsonValue]", value)


def _sequence(value: object, name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        msg = f"Artifact {name} must be a JSON array."
        raise ArtifactError(msg)
    return cast("list[JsonValue]", value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        msg = f"Artifact {name} must be a string."
        raise ArtifactError(msg)
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(_string(item, name) for item in _sequence(value, name))


def _number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        msg = f"Artifact {name} must be a finite number."
        raise ArtifactError(msg)
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Artifact {name} must be an integer."
        raise ArtifactError(msg)
    return value


def _invalid_json_constant(value: str) -> JsonValue:
    msg = f"Artifact JSON contains nonfinite constant {value!r}."
    raise ArtifactError(msg)


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for name, value in pairs:
        if name in result:
            msg = f"Artifact JSON repeats key {name!r}."
            raise ArtifactError(msg)
        result[name] = value
    return result


def _node(value: object) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        msg = "Artifact axis nodes must be strings or integers."
        raise ArtifactError(msg)
    return value


class _Encoder:
    def __init__(self, registry: CodecRegistry) -> None:
        self.registry = registry
        self.arrays: dict[str, NDArray] = {}

    def array(self, value: NDArray, *, name: str | None = None) -> str:
        key = name if name is not None else f"array_{len(self.arrays)}"
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            msg = "Artifact arrays must have finite numerical values."
            raise ArtifactError(msg)
        self.arrays[key] = array
        return key

    def step(self, step: object) -> dict[str, JsonValue]:
        if isinstance(step, FittedStandardize):
            return {
                "type": "fitted_standardize",
                "columns": list(step.columns),
                "means": self.array(step.means),
                "scales": self.array(step.scales),
            }
        if isinstance(step, Standardize):
            return {"type": "standardize", "columns": list(step.columns)}
        if isinstance(step, (AddIntercept, FittedAddIntercept)):
            return {
                "type": "fitted_intercept"
                if isinstance(step, FittedAddIntercept)
                else "intercept",
                "name": step.name,
            }
        if isinstance(step, (FixedWidthBins, FittedFixedWidthBins)):
            return {
                "type": "fitted_bins"
                if isinstance(step, FittedFixedWidthBins)
                else "bins",
                "column": step.column,
                "output": step.output,
                "origin": step.origin,
                "width": step.width,
                "n_bins": step.n_bins,
            }
        return self.registry.encode(step)


def _model_manifest(model: ModelSpec) -> dict[str, JsonValue]:
    penalties: list[JsonValue] = []
    for penalty in model.penalties:
        item: dict[str, JsonValue] = {
            "type": "l1"
            if isinstance(penalty, L1)
            else "l2"
            if isinstance(penalty, L2)
            else "laplacian",
            "name": penalty.name,
            "weight": penalty.weight,
            "features": list(penalty.features),
        }
        if isinstance(penalty, Laplacian):
            item["graph"] = {
                "axis": penalty.graph.axis.name,
                "edges": [list(edge) for edge in penalty.graph.edges],
            }
        penalties.append(item)
    return {
        "task": model.task,
        "features": list(model.features),
        "axes": [{"name": axis.name, "nodes": list(axis.nodes)} for axis in model.axes],
        "penalties": penalties,
    }


def _decode_model(value: object) -> ModelSpec:
    state = _mapping(value, "model")
    axes: list[Axis] = []
    for item in _sequence(state["axes"], "axes"):
        axis = _mapping(item, "axis")
        axes.append(
            Axis(
                name=_string(axis["name"], "axis name"),
                nodes=tuple(
                    _node(node) for node in _sequence(axis["nodes"], "axis nodes")
                ),
            )
        )
    by_name = {axis.name: axis for axis in axes}
    penalties: list[Penalty] = []
    for item in _sequence(state["penalties"], "penalties"):
        penalty = _mapping(item, "penalty")
        name = _string(penalty["name"], "penalty name")
        weight = _number(penalty["weight"], "penalty weight")
        features = _strings(penalty["features"], "penalty features")
        if penalty["type"] == "l1":
            penalties.append(L1(name=name, weight=weight, features=features))
        elif penalty["type"] == "l2":
            penalties.append(L2(name=name, weight=weight, features=features))
        elif penalty["type"] == "laplacian":
            graph = _mapping(penalty["graph"], "graph")
            axis_name = _string(graph["axis"], "graph axis")
            edges = []
            for edge in _sequence(graph["edges"], "graph edges"):
                parts = _sequence(edge, "graph edge")
                if len(parts) != 3:
                    msg = "Artifact graph edges need two nodes and one weight."
                    raise ArtifactError(msg)
                edges.append(
                    (_node(parts[0]), _node(parts[1]), _number(parts[2], "edge weight"))
                )
            penalties.append(
                Laplacian(
                    name=name,
                    weight=weight,
                    features=features,
                    graph=Graph(axis=by_name[axis_name], edges=tuple(edges)),
                )
            )
        else:
            msg = "Unsupported artifact penalty type."
            raise ArtifactError(msg)
    features = _strings(state["features"], "features")
    if state["task"] == "regression":
        return RegressionSpec(
            features=features, axes=tuple(axes), penalties=tuple(penalties)
        )
    if state["task"] == "binary":
        return BinaryLogisticSpec(
            features=features, axes=tuple(axes), penalties=tuple(penalties)
        )
    msg = "Unsupported artifact prediction task."
    raise ArtifactError(msg)


def _decode_step(
    value: object, arrays: Mapping[str, NDArray], registry: CodecRegistry
) -> object:
    state = _mapping(value, "transform")
    kind = state["type"]
    if kind in {"standardize", "fitted_standardize"}:
        columns = _strings(state["columns"], "standardization columns")
        if kind == "standardize":
            return Standardize(columns=columns)
        return FittedStandardize(
            columns=columns,
            means=arrays[_string(state["means"], "means array")],
            scales=arrays[_string(state["scales"], "scales array")],
        )
    if kind in {"intercept", "fitted_intercept"}:
        name = _string(state["name"], "intercept name")
        return (
            AddIntercept(name=name)
            if kind == "intercept"
            else FittedAddIntercept(name=name)
        )
    if kind in {"bins", "fitted_bins"}:
        constructor = FixedWidthBins if kind == "bins" else FittedFixedWidthBins
        return constructor(
            column=_string(state["column"], "bin column"),
            output=_string(state["output"], "bin output"),
            origin=_number(state["origin"], "bin origin"),
            width=_number(state["width"], "bin width"),
            n_bins=_integer(state["n_bins"], "bin count"),
        )
    if kind == "custom":
        return registry.decode(
            _string(state["codec"], "custom codec"),
            _mapping(state["state"], "custom state"),
        )
    msg = "Unsupported artifact transform type."
    raise ArtifactError(msg)


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_artifact(
    predictor: Predictor,
    path: str | Path,
    *,
    registry: CodecRegistry = _BUILTIN_REGISTRY,
    overwrite: bool = False,
) -> Path:
    """Save prediction state to a directory, without training data or caches."""
    if not isinstance(predictor, (FittedRegressor, FittedBinaryClassifier)):
        msg = "Only fitted native predictors can be persisted."
        raise ArtifactError(msg)
    destination = Path(path)
    if destination.exists() and not destination.is_dir():
        msg = "An artifact destination must be a directory."
        raise ArtifactError(msg)
    if not overwrite and any(
        (destination / name).exists() for name in ("manifest.json", "arrays.npz")
    ):
        msg = "Artifact already exists; pass overwrite=True to replace it."
        raise ArtifactError(msg)
    encoder = _Encoder(registry)
    encoder.array(predictor.coefficients, name="coefficients")
    preprocessing = [
        encoder.step(step) for step in predictor.pipeline.preprocessing.steps
    ]
    fitted = [encoder.step(step) for step in predictor.transformers.steps]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in encoder.arrays.items():
            array_buffer = io.BytesIO()
            np.save(array_buffer, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", array_buffer.getvalue())
    payload = buffer.getvalue()
    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "model": _model_manifest(predictor.pipeline.model),
        "preprocessing": preprocessing,
        "transformers": fitted,
        "weights": dict(predictor.weights),
        "coefficients": "coefficients",
        "arrays": {
            name: {"shape": list(array.shape), "dtype": array.dtype.str}
            for name, array in encoder.arrays.items()
        },
        "payload": {
            "file": "arrays.npz",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    try:
        encoded_manifest = json.dumps(
            manifest, allow_nan=False, ensure_ascii=False, indent=2
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        msg = "A component codec returned state that is not finite JSON data."
        raise ArtifactError(msg) from error
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination / "arrays.npz", payload)
    _atomic_write(destination / "manifest.json", encoded_manifest)
    return destination


def _checked_arrays(
    payload: bytes, metadata: object, max_bytes: int
) -> dict[str, NDArray]:
    descriptions = _mapping(metadata, "arrays")
    arrays: dict[str, NDArray] = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != len(descriptions) or len(
            {member.filename for member in members}
        ) != len(members):
            msg = "Artifact array entries do not match their manifest."
            raise ArtifactError(msg)
        if sum(member.file_size for member in members) > max_bytes:
            msg = "Artifact arrays exceed the configured memory budget."
            raise ArtifactError(msg)
        for name, description in descriptions.items():
            if not name or not name.replace("_", "").isalnum():
                msg = "Artifact array names must be simple identifiers."
                raise ArtifactError(msg)
            expected = _mapping(description, "array description")
            shape = tuple(
                _integer(part, "array dimension")
                for part in _sequence(expected["shape"], "array shape")
            )
            dtype = np.dtype(_string(expected["dtype"], "array dtype"))
            if (
                any(size < 0 for size in shape)
                or dtype.kind not in "fiu"
                or math.prod(shape) * dtype.itemsize > max_bytes
            ):
                msg = "Artifact array shape or dtype is unsupported or exceeds its budget."
                raise ArtifactError(msg)
            member = archive.getinfo(f"{name}.npy")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                elif version == (2, 0):
                    actual_shape, _, actual_dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
                else:
                    msg = "Unsupported NumPy array format version."
                    raise ArtifactError(msg)
                if (
                    actual_shape != shape
                    or actual_dtype != dtype
                    or stream.tell() + math.prod(shape) * dtype.itemsize
                    != member.file_size
                ):
                    msg = (
                        "Artifact array header does not match its declared shape/dtype."
                    )
                    raise ArtifactError(msg)
            with archive.open(member) as stream:
                array = np.load(stream, allow_pickle=False)
            if not isinstance(array, np.ndarray) or not np.isfinite(array).all():
                msg = "Artifact arrays must contain finite numerical values."
                raise ArtifactError(msg)
            array.flags.writeable = False
            arrays[name] = array
    return arrays


def _load(path: Path, registry: CodecRegistry, max_bytes: int) -> Predictor:
    manifest_path = path / "manifest.json"
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        msg = "Artifact manifest exceeds the supported size."
        raise ArtifactError(msg)
    manifest = _mapping(
        json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_invalid_json_constant,
            object_pairs_hook=_unique_json_object,
        ),
        "manifest",
    )
    if (
        manifest.get("format") != _FORMAT
        or _integer(manifest.get("version"), "format version") != _VERSION
    ):
        msg = "Unsupported artifact format or version."
        raise ArtifactError(msg)
    payload_description = _mapping(manifest["payload"], "payload")
    if payload_description["file"] != "arrays.npz":
        msg = "Artifact payload must be arrays.npz in the artifact directory."
        raise ArtifactError(msg)
    payload_path = path / "arrays.npz"
    if payload_path.stat().st_size > max_bytes:
        msg = "Artifact payload exceeds the configured memory budget."
        raise ArtifactError(msg)
    payload = payload_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != payload_description["sha256"]:
        msg = "Artifact payload checksum does not match its manifest."
        raise ArtifactError(msg)
    arrays = _checked_arrays(payload, manifest["arrays"], max_bytes)
    model = _decode_model(manifest["model"])
    recipe_steps = tuple(
        _decode_step(step, arrays, registry)
        for step in _sequence(manifest["preprocessing"], "preprocessing")
    )
    if any(
        not isinstance(step, (FittedTransform, TransformRecipe))
        for step in recipe_steps
    ):
        msg = (
            "Decoded preprocessing components do not implement the transform protocol."
        )
        raise ArtifactError(msg)
    fitted_steps = tuple(
        _decode_step(step, arrays, registry)
        for step in _sequence(manifest["transformers"], "transformers")
    )
    if any(not isinstance(step, FittedTransform) for step in fitted_steps):
        msg = (
            "Decoded fitted components do not implement the fitted-transform protocol."
        )
        raise ArtifactError(msg)
    transformers = FittedTransformChain(
        steps=cast("tuple[FittedTransform, ...]", fitted_steps)
    )
    preprocessing = TransformChain(
        steps=cast("tuple[FittedTransform | TransformRecipe, ...]", recipe_steps)
    )
    weights = {
        name: _number(weight, "effective weight")
        for name, weight in _mapping(manifest["weights"], "weights").items()
    }
    if set(weights) != {penalty.name for penalty in model.penalties} or any(
        weight < 0 for weight in weights.values()
    ):
        msg = "Artifact effective weights must match its named nonnegative penalties."
        raise ArtifactError(msg)
    coefficients = arrays[_string(manifest["coefficients"], "coefficients array")]
    expected_shape = (math.prod(axis.size for axis in model.axes), len(model.features))
    if coefficients.shape != expected_shape or coefficients.dtype.kind != "f":
        msg = "Artifact coefficient shape/dtype does not match its model schema."
        raise ArtifactError(msg)
    if isinstance(model, RegressionSpec):
        return FittedRegressor(
            pipeline=Pipeline(preprocessing=preprocessing, model=model),
            transformers=transformers,
            coefficients=coefficients,
            weights=weights,
            training_diagnostics=None,
        )
    return FittedBinaryClassifier(
        pipeline=Pipeline(preprocessing=preprocessing, model=model),
        transformers=transformers,
        coefficients=coefficients,
        weights=weights,
        training_diagnostics=None,
    )


def load_artifact(
    path: str | Path,
    *,
    registry: CodecRegistry = _BUILTIN_REGISTRY,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> Predictor:
    """Load checked prediction state; no pickles or dynamic imports are used."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        msg = "max_bytes must be a positive integer."
        raise ArtifactError(msg)
    try:
        return _load(Path(path), registry, max_bytes)
    except ArtifactError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        EOFError,
        RecursionError,
        zipfile.BadZipFile,
        ValidationError,
    ) as error:
        msg = f"Invalid prediction artifact: {error}"
        raise ArtifactError(msg) from error


def load_regressor(
    path: str | Path,
    *,
    registry: CodecRegistry = _BUILTIN_REGISTRY,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> FittedRegressor:
    predictor = load_artifact(path, registry=registry, max_bytes=max_bytes)
    if not isinstance(predictor, FittedRegressor):
        msg = "Artifact contains a binary classifier, not a regressor."
        raise ArtifactError(msg)
    return predictor


def load_binary_classifier(
    path: str | Path,
    *,
    registry: CodecRegistry = _BUILTIN_REGISTRY,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> FittedBinaryClassifier:
    predictor = load_artifact(path, registry=registry, max_bytes=max_bytes)
    if not isinstance(predictor, FittedBinaryClassifier):
        msg = "Artifact contains a regressor, not a binary classifier."
        raise ArtifactError(msg)
    return predictor
