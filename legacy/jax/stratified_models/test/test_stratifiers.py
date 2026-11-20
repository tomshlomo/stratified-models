from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from stratified_models.stratifiers import (
    BinningStratifier,
    ConstantWidthBinning,
    KMeansStratifierFitter,
    QuantilesBinning,
)


def test_constant_width_binning_respects_min_bins() -> None:
    values = pd.DataFrame({"feature": [0.0, 10.0]})
    values.name = "feature"
    fitter = ConstantWidthBinning(
        out_feature_name="bin",
        min_width=3.0,
        min_num_bins=5,
    )

    stratifier = fitter.fit(values)

    assert stratifier.graph.size == 5
    assert stratifier.out_feature_name == "bin"
    assert stratifier.bin_edges.shape[0] == 4


def test_quantiles_binning_edges_match_quantiles() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    fitter = QuantilesBinning(out_feature_name="qbin", n_bins=2)

    edges = fitter.get_bin_edges(x)

    expected = np.quantile(x, np.linspace(0, 1, 3))[1:-1]
    assert np.allclose(edges, expected)


def test_binning_stratifier_assigns_bins() -> None:
    graph_name = "group"
    stratifier = BinningStratifier(
        bin_edges=np.array([0.5]),
        graph=(
            ConstantWidthBinning(
                out_feature_name=graph_name,
                min_width=1.0,
                min_num_bins=2,
            )
            .fit(_named_frame(graph_name, [0.0, 1.0]))
            .graph
        ),
        in_feature_name="value",
    )
    data = pd.DataFrame({"value": [0.1, 0.6]}, index=[10, 11])

    out = stratifier.transform(data)

    assert list(out) == [0, 1]
    assert out.name == graph_name
    assert list(out.index) == [10, 11]


def test_kmeans_stratifier_uses_requested_features() -> None:
    x = pd.DataFrame(
        {
            "a": [0.0, 0.1, 10.0, 10.1, 0.0, 10.0, 0.1, 9.9],
            "b": [0.0, 0.2, 10.0, 9.9, 10.0, 0.0, 9.8, 0.1],
            "extra": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    kmeans = KMeans(n_clusters=4, n_init=1, random_state=0)
    fitter = KMeansStratifierFitter(out_feature_name="cluster", kmeans=kmeans)

    stratifier = fitter.fit(x[["a", "b"]])
    predicted = stratifier.transform(x)

    expected = kmeans.predict(x[["a", "b"]])
    assert np.array_equal(predicted.to_numpy(), expected)
    assert stratifier.graph.size == 4
    assert stratifier.out_feature_name == "cluster"


def _named_frame(name: str, values: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame({name: values})
    frame.name = name
    return frame
