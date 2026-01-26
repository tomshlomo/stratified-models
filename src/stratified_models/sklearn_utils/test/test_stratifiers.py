import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from stratified_models.sklearn_utils.stratifiers import (
    BinningStratifier,
    KMeansStratifier,
)


def test_binning_stratifier_wrapper() -> None:
    x_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    stratifier = BinningStratifier(
        bin_width=2.0,
        n_bins=None,
        strategy="uniform",
    )
    stratifier.fit(x_df)

    transformed = stratifier.transform(x_df)
    assert transformed.columns.tolist() == ["feature"]
    np.testing.assert_array_equal(transformed["feature"].values, [0, 0, 1, 1])


def test_kmeans_stratifier_wrapper() -> None:
    x_df = pd.DataFrame({"feature": [1.0, 2.0, 10.0, 11.0]})
    stratifier = KMeansStratifier(
        n_clusters=2,
        random_state=42,
        n_init="auto",
        kmeans=None,
    )
    stratifier.fit(x_df)

    transformed = stratifier.transform(x_df)
    labels = transformed["feature"].to_numpy()
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    stratifier = KMeansStratifier(
        n_clusters=None,
        random_state=None,
        n_init="auto",
        kmeans=kmeans,
    )
    stratifier.fit(x_df)
