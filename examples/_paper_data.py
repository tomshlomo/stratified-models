"""Verified data snapshots and deterministic offline adaptations for the paper notebooks."""

from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import attrs
import numpy as np
import pandas as pd
from numpy.typing import NDArray

PAPER_URL = "https://web.stanford.edu/~boyd/papers/pdf/strat_models.pdf"
AUTHOR_COMMIT = "0108114afe1265262f345bf93a5bc147f6cb146c"
AUTHOR_CODE_URL = (
    f"https://github.com/cvxgrp/strat_models/tree/{AUTHOR_COMMIT}/examples"
)
MAX_ARCHIVE_BYTES = 12_000_000
HOUSE_FEATURES = (
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
    "grade",
    "yr_built",
)
HOUSE_LATITUDE = (47.1559, 47.7776)
HOUSE_LONGITUDE = (-122.519, -121.646)
MESOTHELIOMA_CATEGORIES = MappingProxyType(
    {
        "city": tuple(range(9)),
        "keep side": tuple(range(3)),
        "type of MM": tuple(range(3)),
        "habit of cigarette": tuple(range(4)),
    }
)


class DataIntegrityError(ValueError):
    """A downloaded or cached snapshot differs from the reviewed source data."""


@attrs.frozen(kw_only=True)
class DataArtifact:
    name: str
    url: str
    member: str
    sha256: str
    citation_url: str
    license_name: str
    license_url: str


MESOTHELIOMA = DataArtifact(
    name="mesothelioma",
    url="https://archive.ics.uci.edu/static/public/351/mesothelioma+s+disease+data+set.zip",
    member="Mesothelioma data set.xlsx",
    sha256="cb3a7b5b9352ff30224f5080769db031339dd78c5a4ffb6860cb2a2a54cba73f",
    citation_url="https://doi.org/10.24432/C5903T",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
)
KING_COUNTY = DataArtifact(
    name="king_county",
    url="https://www.kaggle.com/api/v1/datasets/download/harlfoxem/housesalesprediction",
    member="kc_house_data.csv",
    sha256="d0875baa0251b21d4bdc9d2ae940a4fe0bb6009824f23dd0e2a5b2bf04557b7e",
    citation_url="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction",
    license_name="CC0 (dataset card)",
    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
)


@attrs.frozen(kw_only=True)
class Provenance:
    kind: Literal["original_data", "synthetic_adaptation"]
    citation_url: str
    sha256: str
    n_raw: int
    n_used: int
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return attrs.asdict(self)


@attrs.frozen(kw_only=True)
class PaperDataset:
    name: str
    frame: pd.DataFrame
    features: tuple[str, ...]
    provenance: Provenance

    @property
    def x(self) -> pd.DataFrame:
        return self.frame.drop(columns="target").copy()

    @property
    def y(self) -> NDArray[np.float64]:
        return self.frame["target"].to_numpy(dtype=np.float64, copy=True)


def verified_bytes(content: bytes, artifact: DataArtifact) -> bytes:
    observed = hashlib.sha256(content).hexdigest()
    if observed != artifact.sha256:
        message = (
            f"{artifact.name} checksum mismatch: expected {artifact.sha256}, "
            f"received {observed}. The source may have changed; review it before use."
        )
        raise DataIntegrityError(message)
    return content


def extract_verified(archive: bytes, artifact: DataArtifact) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        member = zipped.getinfo(artifact.member)
        if member.file_size > MAX_ARCHIVE_BYTES:
            message = f"{artifact.name} exceeds the notebook data size limit"
            raise DataIntegrityError(message)
        return verified_bytes(zipped.read(member), artifact)


def fetch_artifact(
    artifact: DataArtifact,
    *,
    cache_dir: Path,
    allow_download: bool,
) -> bytes:
    """Use a checked local snapshot, downloading only after explicit opt-in."""
    destination = cache_dir / artifact.member
    if destination.exists():
        return verified_bytes(destination.read_bytes(), artifact)
    if not allow_download:
        message = (
            f"{artifact.name} is not cached. Enable the notebook's explicit data download "
            f"control to retrieve {artifact.citation_url}."
        )
        raise FileNotFoundError(message)
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "stratified-models-paper-notebooks/1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        archive = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(archive) > MAX_ARCHIVE_BYTES:
        message = f"{artifact.name} exceeds the notebook archive size limit"
        raise DataIntegrityError(message)
    raw = extract_verified(archive, artifact)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return raw


def prepare_mesothelioma(raw: pd.DataFrame, *, sha256: str) -> PaperDataset:
    """Apply declared categorical domains; fitted scaling remains inside each CV fold."""
    data = raw.copy(deep=True)
    data.columns = data.columns.str.strip()
    required = {
        "age",
        "gender",
        "diagnosis method",
        "class of diagnosis",
        "glucose",
        *MESOTHELIOMA_CATEGORIES,
    }
    if not required.issubset(data.columns):
        message = f"Mesothelioma snapshot is missing columns: {sorted(required - set(data.columns))}"
        raise DataIntegrityError(message)
    data = data.apply(pd.to_numeric, errors="raise")
    if data.isna().any().any():
        message = "Mesothelioma snapshot contains missing values"
        raise DataIntegrityError(message)
    ages = data["age"].to_numpy(dtype=float)
    if not np.all((ages == np.floor(ages)) & (ages >= 19) & (ages <= 85)):
        message = (
            "Mesothelioma age values fall outside the declared integer axis 19..85"
        )
        raise DataIntegrityError(message)
    if not set(data["gender"].unique()).issubset({0, 1}):
        message = "Mesothelioma gender values must use the source codes 0 and 1"
        raise DataIntegrityError(message)
    if not set(data["class of diagnosis"].unique()).issubset({1, 2}):
        message = "Mesothelioma outcomes must use the source codes 1 and 2"
        raise DataIntegrityError(message)
    target = data["class of diagnosis"].to_numpy(dtype=float) - 1.0
    sex = data["gender"].map({1: "Male", 0: "Female"})
    engineered = data.drop(
        columns=["age", "gender", "diagnosis method", "class of diagnosis"]
    )
    for column, categories in MESOTHELIOMA_CATEGORIES.items():
        if not set(engineered[column].unique()).issubset(categories):
            message = f"Mesothelioma {column!r} contains undeclared category codes"
            raise DataIntegrityError(message)
        engineered[column] = pd.Categorical(engineered[column], categories=categories)
    engineered = pd.get_dummies(
        engineered, columns=list(MESOTHELIOMA_CATEGORIES), dtype=float
    )
    features = tuple(str(column) for column in engineered.columns)
    engineered["age"] = ages.astype(np.int64)
    engineered["sex"] = sex.to_numpy()
    engineered["target"] = target
    return PaperDataset(
        name="UCI Mesothelioma · original data",
        frame=engineered,
        features=features,
        provenance=Provenance(
            kind="original_data",
            citation_url=MESOTHELIOMA.citation_url,
            sha256=sha256,
            n_raw=len(raw),
            n_used=len(engineered),
            notes=(
                "Removed diagnosis method because it leaks the outcome.",
                "Categorical vocabularies are fixed schema domains, not fitted from validation folds.",
                "Age and sex select strata; all numeric scaling is fitted within each training fold.",
            ),
        ),
    )


def load_mesothelioma(*, cache_dir: Path, allow_download: bool) -> PaperDataset:
    raw = fetch_artifact(
        MESOTHELIOMA, cache_dir=cache_dir, allow_download=allow_download
    )
    return prepare_mesothelioma(
        pd.read_excel(io.BytesIO(raw), engine="openpyxl"), sha256=MESOTHELIOMA.sha256
    )


def prepare_house_prices(raw: pd.DataFrame, *, sha256: str) -> PaperDataset:
    required = {"price", "lat", "long", *HOUSE_FEATURES}
    if not required.issubset(raw.columns):
        message = f"King County snapshot is missing columns: {sorted(required - set(raw.columns))}"
        raise DataIntegrityError(message)
    frame = raw.loc[
        raw["long"] <= -121.6, [*HOUSE_FEATURES, "lat", "long", "price"]
    ].copy()
    numeric = frame.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(frame["price"].to_numpy() <= 0):
        message = "King County features must be finite and prices strictly positive"
        raise DataIntegrityError(message)
    for column, domain in (("lat", HOUSE_LATITUDE), ("long", HOUSE_LONGITUDE)):
        if not frame[column].between(*domain, inclusive="both").all():
            message = f"King County {column} lies outside the reviewed fixed geographic domain"
            raise DataIntegrityError(message)
    frame["target"] = np.log(frame.pop("price").to_numpy(dtype=float))
    return PaperDataset(
        name="King County · original data",
        frame=frame,
        features=HOUSE_FEATURES,
        provenance=Provenance(
            kind="original_data",
            citation_url=KING_COUNTY.citation_url,
            sha256=sha256,
            n_raw=len(raw),
            n_used=len(frame),
            notes=(
                "Longitude <= -121.6 matches the authors' source script.",
                "Target is natural log sale price; no target standardization is applied.",
                "The fixed geographic domain comes from the reviewed snapshot, not a validation-fold fit.",
                "Numeric scaling is fitted independently inside each training fold.",
            ),
        ),
    )


def load_house_prices(*, cache_dir: Path, allow_download: bool) -> PaperDataset:
    raw = fetch_artifact(
        KING_COUNTY, cache_dir=cache_dir, allow_download=allow_download
    )
    return prepare_house_prices(pd.read_csv(io.BytesIO(raw)), sha256=KING_COUNTY.sha256)


def synthetic_mesothelioma(*, seed: int = 41, n_rows: int = 240) -> PaperDataset:
    """Generate an offline teaching dataset, unrelated to individual patient records."""
    rng = np.random.default_rng(seed)
    ages = rng.integers(30, 42, size=n_rows)
    sexes = rng.choice(np.array(["Male", "Female"]), size=n_rows)
    glucose = rng.normal(size=n_rows)
    exposure = rng.normal(size=n_rows)
    logits = -0.6 + (0.7 + 0.7 * np.sin((ages - 30) / 3)) * glucose + 0.4 * exposure
    logits += np.where(sexes == "Male", 0.2, -0.2)
    probabilities = 1 / (1 + np.exp(-logits))
    frame = pd.DataFrame(
        {
            "age": ages,
            "sex": sexes,
            "glucose": glucose,
            "exposure": exposure,
            "target": rng.binomial(1, probabilities).astype(float),
        }
    )
    return PaperDataset(
        name="Synthetic age × sex classification · adaptation",
        frame=frame,
        features=("glucose", "exposure"),
        provenance=Provenance(
            kind="synthetic_adaptation",
            citation_url=PAPER_URL,
            sha256="not applicable: generated data",
            n_raw=n_rows,
            n_used=n_rows,
            notes=(
                f"Seed {seed}; generated observations, not UCI patient data.",
                "Demonstrates the graph structure; does not reproduce the paper's metrics.",
            ),
        ),
    )


def synthetic_house_prices(*, seed: int = 17, n_rows: int = 320) -> PaperDataset:
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(*HOUSE_LATITUDE, size=n_rows)
    longitude = rng.uniform(*HOUSE_LONGITUDE, size=n_rows)
    bedrooms = rng.integers(1, 6, size=n_rows).astype(float)
    sqft_living = rng.lognormal(mean=7.4, sigma=0.35, size=n_rows)
    waterfront = rng.binomial(1, 0.15, size=n_rows).astype(float)
    coast = (longitude - HOUSE_LONGITUDE[0]) / (HOUSE_LONGITUDE[1] - HOUSE_LONGITUDE[0])
    target = 10.8 + 0.00065 * sqft_living + 0.1 * bedrooms
    target += (0.6 - 0.5 * coast) * waterfront + 0.35 * np.sin(8 * latitude)
    target += rng.normal(scale=0.12, size=n_rows)
    frame = pd.DataFrame(
        {
            "lat": latitude,
            "long": longitude,
            "bedrooms": bedrooms,
            "sqft_living": sqft_living,
            "waterfront": waterfront,
            "target": target,
        }
    )
    return PaperDataset(
        name="Synthetic spatial log-price regression · adaptation",
        frame=frame,
        features=("bedrooms", "sqft_living", "waterfront"),
        provenance=Provenance(
            kind="synthetic_adaptation",
            citation_url=PAPER_URL,
            sha256="not applicable: generated data",
            n_raw=n_rows,
            n_used=n_rows,
            notes=(
                f"Seed {seed}; generated houses, not King County sales.",
                "Reduced features and grid are a quick educational adaptation.",
            ),
        ),
    )


def mean_penalty_weight(
    weight: float,
    *,
    n_train: int,
    loss_multiplier: float,
) -> float:
    """Convert a paper sum-loss weight when the new per-sample loss is multiplied by c."""
    if n_train <= 0 or not math.isfinite(weight) or weight < 0:
        message = (
            "a nonnegative finite paper weight and positive training count are required"
        )
        raise ValueError(message)
    if not math.isfinite(loss_multiplier) or loss_multiplier <= 0:
        message = "loss_multiplier must be positive and finite"
        raise ValueError(message)
    return loss_multiplier * weight / n_train


def inclusive_bin_width(domain: tuple[float, float], *, n_bins: int) -> float:
    """Keep reviewed maximum coordinates inside a fixed half-open grid."""
    lower, upper = domain
    if n_bins <= 0 or not (
        math.isfinite(lower) and math.isfinite(upper) and lower < upper
    ):
        message = (
            "a finite increasing geographic domain and positive bin count are required"
        )
        raise ValueError(message)
    expanded_upper = float(np.nextafter(np.nextafter(upper, np.inf), np.inf))
    return (expanded_upper - lower) / n_bins


def binary_metrics(
    target: NDArray[np.float64], scores: NDArray[np.float64]
) -> dict[str, float]:
    return {
        "anll": float(np.mean(np.logaddexp(0.0, scores) - target * scores)),
        "error": float(np.mean((scores >= 0) != target)),
    }


def regression_metrics(
    target: NDArray[np.float64], prediction: NDArray[np.float64]
) -> dict[str, float]:
    residual = target - prediction
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - float(np.sum(residual**2)) / denominator
        if denominator
        else float("nan"),
    }
