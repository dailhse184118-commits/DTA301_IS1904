"""Preprocessing, PCA, and cluster-tendency utilities for SinD Stage 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class PreprocessingSpec:
    """One candidate preprocessing pipeline."""

    name: str
    winsorize: bool
    scaler: str


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip selected columns to training-set quantile bounds.

    The transformer follows the scikit-learn API so that quantile bounds are
    learned only from training folds during cross-validation.
    """

    def __init__(
        self,
        feature_names: list[str],
        continuous_features: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        self.lower_bounds_ = frame[self.continuous_features].quantile(
            self.lower_quantile
        )
        self.upper_bounds_ = frame[self.continuous_features].quantile(
            self.upper_quantile
        )
        return self

    def transform(self, X):
        frame = self._to_frame(X).copy()
        frame.loc[:, self.continuous_features] = frame[
            self.continuous_features
        ].clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis="columns",
        )
        return frame.to_numpy(dtype=float)

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
            frame.columns = self.feature_names
            return frame
        return pd.DataFrame(X, columns=self.feature_names)


def apply_preprocessing(
    data: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
    spec: PreprocessingSpec,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
):
    """Fit one candidate preprocessing pipeline on the supplied data."""
    X = data[feature_columns].copy()
    winsorizer = None

    if spec.winsorize:
        winsorizer = QuantileWinsorizer(
            feature_names=feature_columns,
            continuous_features=continuous_features,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        X_processed = winsorizer.fit_transform(X)
    else:
        X_processed = X.to_numpy(dtype=float)

    if spec.scaler == "standard":
        scaler = StandardScaler()
    elif spec.scaler == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler: {spec.scaler}")

    X_scaled = scaler.fit_transform(X_processed)
    pca = PCA(n_components=0.90, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    return {
        "X_raw": X.to_numpy(dtype=float),
        "X_processed": X_processed,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "winsorizer": winsorizer,
        "scaler": scaler,
        "pca": pca,
    }


def hopkins_statistic(
    X: np.ndarray,
    sample_size: int = 1000,
    random_state: int = 42,
) -> float:
    """Calculate Hopkins statistic in the model input space.

    Values near 0.5 indicate spatial randomness. Values substantially above
    0.5 indicate non-random, clusterable structure under this definition.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 3:
        raise ValueError("Hopkins statistic requires a 2D array with >=3 rows.")

    rng = np.random.default_rng(random_state)
    m = min(sample_size, max(2, len(X) - 1))
    sample_indices = rng.choice(len(X), size=m, replace=False)

    nearest = NearestNeighbors(n_neighbors=2).fit(X)
    sampled_distances = nearest.kneighbors(
        X[sample_indices],
        n_neighbors=2,
        return_distance=True,
    )[0][:, 1]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    random_points = rng.uniform(mins, maxs, size=(m, X.shape[1]))
    random_distances = nearest.kneighbors(
        random_points,
        n_neighbors=1,
        return_distance=True,
    )[0][:, 0]

    numerator = float(random_distances.sum())
    denominator = numerator + float(sampled_distances.sum())
    return numerator / denominator if denominator else 0.5


def pca_loading_table(
    pca: PCA,
    feature_columns: Iterable[str],
    pipeline_name: str,
) -> pd.DataFrame:
    """Return loading coefficients for interpretable PCA reporting."""
    feature_columns = list(feature_columns)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    columns = [f"PC{index + 1}" for index in range(pca.n_components_)]
    table = pd.DataFrame(loadings, columns=columns)
    table.insert(0, "feature", feature_columns)
    table.insert(0, "pipeline", pipeline_name)
    return table
