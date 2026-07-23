"""Cross-city and cross-recording validation utilities for SinD Stage 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import RobustScaler


CORE_FEATURES = [
    "mean_speed_mps",
    "max_speed_mps",
    "speed_std_mps",
    "mean_long_acc_mps2",
    "max_acceleration_mps2",
    "max_deceleration_mps2",
    "acceleration_std_mps2",
    "mean_abs_jerk_mps3",
    "observed_stop_transition_count",
    "stopped_time_ratio",
]

CONTINUOUS_FEATURES = CORE_FEATURES[:8]
PROFILE_IDS = [1, 2, 3, 4]


@dataclass(frozen=True)
class ValidationConfig:
    """End-to-end preprocessing and clustering settings."""

    winsor_lower_quantile: float = 0.01
    winsor_upper_quantile: float = 0.99
    pca_variance_threshold: float = 0.90
    n_clusters: int = 4
    n_init: int = 20
    random_state: int = 42


class TrainOnlyPreprocessor:
    """Fit winsorization, scaling, and PCA using training rows only."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.lower_bounds_: pd.Series | None = None
        self.upper_bounds_: pd.Series | None = None
        self.scaler_: RobustScaler | None = None
        self.pca_: PCA | None = None

    def fit(self, data: pd.DataFrame) -> "TrainOnlyPreprocessor":
        features = data[CORE_FEATURES].copy()

        self.lower_bounds_ = features[CONTINUOUS_FEATURES].quantile(
            self.config.winsor_lower_quantile
        )
        self.upper_bounds_ = features[CONTINUOUS_FEATURES].quantile(
            self.config.winsor_upper_quantile
        )

        winsorized = self._apply_winsorization(features)

        self.scaler_ = RobustScaler()
        scaled = self.scaler_.fit_transform(winsorized)

        self.pca_ = PCA(
            n_components=self.config.pca_variance_threshold,
            svd_solver="full",
        )
        self.pca_.fit(scaled)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if (
            self.lower_bounds_ is None
            or self.upper_bounds_ is None
            or self.scaler_ is None
            or self.pca_ is None
        ):
            raise RuntimeError("Preprocessor must be fitted before transform.")

        features = data[CORE_FEATURES].copy()
        winsorized = self._apply_winsorization(features)
        scaled = self.scaler_.transform(winsorized)
        return self.pca_.transform(scaled)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def _apply_winsorization(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        output = features.copy()
        output[CONTINUOUS_FEATURES] = output[
            CONTINUOUS_FEATURES
        ].clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis="columns",
        )
        return output

    @property
    def retained_variance(self) -> float:
        if self.pca_ is None:
            raise RuntimeError("PCA is not fitted.")
        return float(self.pca_.explained_variance_ratio_.sum())

    @property
    def n_components(self) -> int:
        if self.pca_ is None:
            raise RuntimeError("PCA is not fitted.")
        return int(self.pca_.n_components_)


def align_clusters_to_reference(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
    profile_ids: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Align arbitrary cluster IDs to reference profile IDs using Hungarian matching."""
    if profile_ids is None:
        profile_ids = PROFILE_IDS

    candidate_ids = sorted(np.unique(candidate_labels).tolist())
    contingency = np.zeros(
        (len(candidate_ids), len(profile_ids)),
        dtype=int,
    )

    for row_index, candidate_id in enumerate(candidate_ids):
        for column_index, profile_id in enumerate(profile_ids):
            contingency[row_index, column_index] = int(
                (
                    (candidate_labels == candidate_id)
                    & (reference_labels == profile_id)
                ).sum()
            )

    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {
        int(candidate_ids[row]): int(profile_ids[column])
        for row, column in zip(row_ind, col_ind)
    }

    aligned = np.array(
        [mapping.get(int(label), int(label)) for label in candidate_labels],
        dtype=int,
    )
    return aligned, mapping


def fit_end_to_end_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: ValidationConfig,
) -> dict[str, object]:
    """Fit all preprocessing and K-Means steps on training data only."""
    preprocessor = TrainOnlyPreprocessor(config)
    X_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)

    model = KMeans(
        n_clusters=config.n_clusters,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    raw_train_labels = model.fit_predict(X_train)
    raw_test_labels = model.predict(X_test)

    aligned_train_labels, mapping = align_clusters_to_reference(
        train_data["profile_id"].to_numpy(dtype=int),
        raw_train_labels,
    )
    aligned_test_labels = np.array(
        [mapping[int(label)] for label in raw_test_labels],
        dtype=int,
    )

    return {
        "preprocessor": preprocessor,
        "model": model,
        "train_labels": aligned_train_labels,
        "test_labels": aligned_test_labels,
        "cluster_mapping": mapping,
    }


def evaluate_held_out_labels(
    reference: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int | bool]:
    """Evaluate held-out labels against the full-data reference partition."""
    reference = np.asarray(reference, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    present_reference = sorted(np.unique(reference).tolist())
    present_predicted = sorted(np.unique(predicted).tolist())

    counts = pd.Series(predicted).value_counts()
    return {
        "ari": float(adjusted_rand_score(reference, predicted)),
        "aligned_accuracy": float(accuracy_score(reference, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(reference, predicted)
        ),
        "macro_f1_present_reference": float(
            f1_score(
                reference,
                predicted,
                labels=present_reference,
                average="macro",
                zero_division=0,
            )
        ),
        "reference_profiles_present": int(len(present_reference)),
        "predicted_profiles_present": int(len(present_predicted)),
        "all_reference_profiles_recovered": bool(
            set(present_reference).issubset(set(present_predicted))
        ),
        "smallest_predicted_profile_count": int(counts.min()),
        "smallest_predicted_profile_percentage": float(
            counts.min() / len(predicted) * 100
        ),
    }


def global_profile_reference(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return global profile medians and feature IQR values."""
    profile_medians = data.groupby("profile_id")[CORE_FEATURES].median()
    feature_iqr = (
        data[CORE_FEATURES].quantile(0.75)
        - data[CORE_FEATURES].quantile(0.25)
    ).replace(0, 1.0)
    return profile_medians, feature_iqr


def semantic_profile_consistency(
    data: pd.DataFrame,
    predicted_labels: np.ndarray,
    global_profile_medians: pd.DataFrame,
    global_feature_iqr: pd.Series,
) -> pd.DataFrame:
    """Compare predicted profile medians with global reference medians."""
    working = data[CORE_FEATURES].copy()
    working["predicted_profile_id"] = np.asarray(
        predicted_labels,
        dtype=int,
    )

    center = global_profile_medians.median()

    rows: list[dict[str, float | int]] = []
    for profile_id, group in working.groupby(
        "predicted_profile_id",
        observed=True,
    ):
        predicted_median = group[CORE_FEATURES].median()
        reference_median = global_profile_medians.loc[profile_id]

        normalized_difference = (
            (predicted_median - reference_median).abs()
            / global_feature_iqr
        )

        predicted_vector = (
            (predicted_median - center)
            / global_feature_iqr
        )
        reference_vector = (
            (reference_median - center)
            / global_feature_iqr
        )

        if (
            predicted_vector.std(ddof=0) > 0
            and reference_vector.std(ddof=0) > 0
        ):
            profile_correlation = float(
                np.corrcoef(
                    predicted_vector.to_numpy(),
                    reference_vector.to_numpy(),
                )[0, 1]
            )
        else:
            profile_correlation = np.nan

        rows.append(
            {
                "profile_id": int(profile_id),
                "trajectory_count": int(len(group)),
                "median_normalized_mae": float(
                    normalized_difference.mean()
                ),
                "median_normalized_max_error": float(
                    normalized_difference.max()
                ),
                "profile_vector_correlation": profile_correlation,
            }
        )

    return pd.DataFrame(rows)
