"""Paper-data integrity and feature-engineering contracts; no external network required."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import Request

import numpy as np
import pandas as pd
import pytest

from examples._paper_data import (
    HOUSE_FEATURES,
    HOUSE_LATITUDE,
    HOUSE_LONGITUDE,
    DataArtifact,
    DataIntegrityError,
    binary_metrics,
    extract_verified,
    fetch_artifact,
    inclusive_bin_width,
    mean_penalty_weight,
    prepare_house_prices,
    prepare_mesothelioma,
    regression_metrics,
    synthetic_house_prices,
    synthetic_mesothelioma,
)


@pytest.fixture
def artifact() -> DataArtifact:
    return DataArtifact(
        name="fixture",
        url="https://example.invalid/data.zip",
        member="data.csv",
        sha256=hashlib.sha256(b"x,y\n1,2\n").hexdigest(),
        citation_url="https://example.invalid/source",
        license_name="test fixture",
        license_url="https://example.invalid/license",
    )


def archive_bytes(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", content)
    return buffer.getvalue()


def test_download_requires_explicit_opt_in_and_checks_existing_cache(
    artifact: DataArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        pytest.fail("network access occurred without explicit opt-in")

    monkeypatch.setattr(
        "examples._paper_data.urllib.request.urlopen", unexpected_network
    )
    with pytest.raises(FileNotFoundError, match="explicit data download"):
        fetch_artifact(artifact, cache_dir=tmp_path, allow_download=False)
    cached = tmp_path / artifact.member
    cached.write_bytes(b"x,y\n1,2\n")
    assert (
        fetch_artifact(artifact, cache_dir=tmp_path, allow_download=False)
        == cached.read_bytes()
    )
    cached.write_bytes(b"changed")
    with pytest.raises(DataIntegrityError, match="checksum mismatch"):
        fetch_artifact(artifact, cache_dir=tmp_path, allow_download=False)


def test_only_verified_member_is_cached_after_opt_in(
    artifact: DataArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"x,y\n1,2\n"
    requests: list[str] = []

    def download(request: Request, *, timeout: int) -> io.BytesIO:
        requests.append(request.full_url)
        assert timeout > 0
        return io.BytesIO(archive_bytes(content))

    monkeypatch.setattr("examples._paper_data.urllib.request.urlopen", download)
    result = fetch_artifact(artifact, cache_dir=tmp_path / "cache", allow_download=True)
    assert result == content
    assert requests == [artifact.url]
    assert list((tmp_path / "cache").iterdir()) == [
        tmp_path / "cache" / artifact.member
    ]


def test_changed_archive_contents_are_rejected(artifact: DataArtifact) -> None:
    with pytest.raises(DataIntegrityError, match="checksum mismatch"):
        extract_verified(archive_bytes(b"unreviewed replacement"), artifact)


def test_mesothelioma_removes_outcome_leak_and_preserves_declared_categories() -> None:
    raw = pd.DataFrame(
        {
            "age": [19.0, 85.0],
            "gender": [1, 0],
            "city": [0, 1],
            "keep side": [0, 1],
            "type of MM": [0, 1],
            "habit of cigarette": [0, 1],
            "diagnosis method": [1, 2],
            "class of diagnosis": [1, 2],
            "glucose": [90, 110],
        }
    )
    original = raw.copy(deep=True)
    prepared = prepare_mesothelioma(raw, sha256="fixture")
    assert prepared.frame["sex"].tolist() == ["Male", "Female"]
    np.testing.assert_array_equal(prepared.y, [0, 1])
    assert "diagnosis method" not in prepared.frame.columns
    assert "class of diagnosis" not in prepared.frame.columns
    assert "age" not in prepared.features
    assert "sex" not in prepared.features
    assert "target" not in prepared.features
    assert "city_8" in prepared.features
    assert prepared.frame["city_8"].eq(0).all()
    pd.testing.assert_frame_equal(raw, original)
    with pytest.raises(DataIntegrityError, match="category"):
        prepare_mesothelioma(raw.assign(city=99), sha256="fixture")


def test_house_target_filter_and_covariates_match_declared_experiment() -> None:
    raw = pd.DataFrame({feature: [1.0, 2.0, 3.0] for feature in HOUSE_FEATURES})
    raw["lat"] = [47.3, 47.6, 47.4]
    raw["long"] = [-122.4, -121.8, -121.5]
    raw["price"] = [100_000.0, 400_000.0, 200_000.0]
    original = raw.copy(deep=True)
    prepared = prepare_house_prices(raw, sha256="fixture")
    assert prepared.features == HOUSE_FEATURES
    assert prepared.provenance.n_raw == 3
    assert prepared.provenance.n_used == 2
    np.testing.assert_allclose(prepared.y, np.log([100_000.0, 400_000.0]))
    assert "price" not in prepared.frame.columns
    assert "target" not in prepared.x.columns
    pd.testing.assert_frame_equal(raw, original)
    with pytest.raises(DataIntegrityError, match="strictly positive"):
        prepare_house_prices(raw.assign(price=0), sha256="fixture")


@pytest.mark.parametrize("loss_multiplier", [1.0, 0.5])
def test_mean_conversion_preserves_complete_objective(loss_multiplier: float) -> None:
    residual = np.array([-2.0, 1.0, 3.0, -1.0])
    coefficients = np.array([0.3, -0.7])
    original_weight = 4.2
    old_loss = float(np.sum(residual**2))
    penalty = float(0.5 * np.sum(coefficients**2))
    converted = mean_penalty_weight(
        original_weight, n_train=len(residual), loss_multiplier=loss_multiplier
    )
    new_objective = loss_multiplier * old_loss / len(residual) + converted * penalty
    expected = loss_multiplier * (old_loss + original_weight * penalty) / len(residual)
    assert new_objective == pytest.approx(expected)


def test_unhalved_local_penalty_requires_its_own_factor_conversion() -> None:
    n_train = 11
    residual_sum = 7.3
    squared_norm = 2.7
    graph_energy = 1.9
    local_weight, graph_weight = 0.6, 3.2
    original = (
        residual_sum + local_weight * squared_norm + graph_weight * graph_energy / 2
    )
    converted_local = mean_penalty_weight(
        2 * local_weight, n_train=n_train, loss_multiplier=0.5
    )
    converted_graph = mean_penalty_weight(
        graph_weight, n_train=n_train, loss_multiplier=0.5
    )
    converted = (
        residual_sum / (2 * n_train)
        + converted_local * squared_norm / 2
        + converted_graph * graph_energy / 2
    )
    assert converted_local == pytest.approx(local_weight / n_train)
    assert converted_graph == pytest.approx(graph_weight / (2 * n_train))
    assert converted == pytest.approx(original / (2 * n_train))


def test_offline_adaptations_are_replayable_and_explicitly_labeled() -> None:
    for factory in (synthetic_mesothelioma, synthetic_house_prices):
        first = factory(seed=123, n_rows=40)
        second = factory(seed=123, n_rows=40)
        assert first.provenance.kind == "synthetic_adaptation"
        assert "adaptation" in first.name
        pd.testing.assert_frame_equal(first.frame, second.frame)
        assert np.isfinite(first.y).all()


def test_metrics_use_stable_logistic_loss_and_log_price_rmse() -> None:
    target = np.array([0.0, 1.0])
    scores = np.array([-1_000.0, 1_000.0])
    metrics = binary_metrics(target, scores)
    assert metrics["anll"] == pytest.approx(0.0)
    assert metrics["error"] == 0.0
    regression = regression_metrics(np.array([10.0, 12.0]), np.array([11.0, 11.0]))
    assert regression == {"rmse": 1.0, "r2": 0.0}


@pytest.mark.parametrize("domain", [HOUSE_LATITUDE, HOUSE_LONGITUDE])
@pytest.mark.parametrize("n_bins", [3, 50, 256])
def test_fixed_geographic_grid_includes_both_reviewed_extrema(
    domain: tuple[float, float],
    n_bins: int,
) -> None:
    width = inclusive_bin_width(domain, n_bins=n_bins)
    assert domain[0] + width * n_bins > domain[1]
    encoded = np.floor((np.asarray(domain) - domain[0]) / width).astype(int)
    np.testing.assert_array_equal(encoded, [0, n_bins - 1])
