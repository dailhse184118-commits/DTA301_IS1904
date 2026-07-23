"""Reusable clustering evaluation helpers for the SinD full-dataset study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Shared settings for fair clustering evaluation."""

    silhouette_sample_size: int = 5000
    random_state: int = 42
    noise_label: int = -1


def evaluate_partition(
    X: np.ndarray,
    labels: np.ndarray,
    config: EvaluationConfig | None = None,
) -> dict[str, float | int]:
    """Evaluate a partition while treating -1 as density-model noise."""
    if config is None:
        config = EvaluationConfig()

    labels = np.asarray(labels)
    clustered_mask = labels != config.noise_label
    X_clustered = X[clustered_mask]
    labels_clustered = labels[clustered_mask]

    unique_clusters, counts = np.unique(labels_clustered, return_counts=True)
    n_clusters = len(unique_clusters)
    n_total = len(labels)
    n_clustered = int(clustered_mask.sum())
    n_noise = n_total - n_clustered

    result: dict[str, float | int] = {
        "n_observations": n_total,
        "n_clustered": n_clustered,
        "n_noise": n_noise,
        "coverage": n_clustered / n_total if n_total else 0.0,
        "noise_percentage": n_noise / n_total * 100 if n_total else 0.0,
        "n_clusters": n_clusters,
        "smallest_cluster_count": int(counts.min()) if len(counts) else 0,
        "largest_cluster_count": int(counts.max()) if len(counts) else 0,
        "smallest_cluster_percentage": (
            counts.min() / n_total * 100 if len(counts) else 0.0
        ),
        "largest_cluster_percentage": (
            counts.max() / n_total * 100 if len(counts) else 0.0
        ),
    }

    if n_clusters < 2 or n_clustered <= n_clusters:
        result.update(
            {
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
            }
        )
        return result

    sample_size = min(config.silhouette_sample_size, n_clustered)
    result["silhouette"] = float(
        silhouette_score(
            X_clustered,
            labels_clustered,
            sample_size=sample_size,
            random_state=config.random_state,
        )
    )
    result["davies_bouldin"] = float(
        davies_bouldin_score(X_clustered, labels_clustered)
    )
    result["calinski_harabasz"] = float(
        calinski_harabasz_score(X_clustered, labels_clustered)
    )
    return result


def normalized_cluster_entropy(
    labels: np.ndarray,
    noise_label: int = -1,
) -> float:
    """Return cluster-size entropy normalized to [0, 1]."""
    labels = np.asarray(labels)
    labels = labels[labels != noise_label]
    _, counts = np.unique(labels, return_counts=True)

    if len(counts) <= 1:
        return 0.0

    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(counts)))


def feature_separation_eta_squared(
    data: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    noise_label: int = -1,
) -> pd.DataFrame:
    """Measure between-cluster separation for original interpretable features."""
    labels = np.asarray(labels)
    mask = labels != noise_label
    working = data.loc[mask, feature_columns].copy()
    working["cluster"] = labels[mask]

    rows: list[dict[str, float | str]] = []
    for feature in feature_columns:
        values = working[feature]
        grand_mean = values.mean()
        total_ss = float(((values - grand_mean) ** 2).sum())

        between_ss = 0.0
        for _, group in working.groupby("cluster", observed=True):
            group_values = group[feature]
            between_ss += (
                len(group_values)
                * float((group_values.mean() - grand_mean) ** 2)
            )

        eta_squared = between_ss / total_ss if total_ss > 0 else 0.0
        rows.append(
            {
                "feature": feature,
                "eta_squared": float(eta_squared),
            }
        )

    return pd.DataFrame(rows)


def pairwise_seed_stability(label_sets: list[np.ndarray]) -> dict[str, float]:
    """Summarize all pairwise ARIs across repeated seeded fits."""
    scores: list[float] = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            scores.append(
                adjusted_rand_score(label_sets[i], label_sets[j])
            )

    if not scores:
        return {
            "seed_ari_mean": np.nan,
            "seed_ari_min": np.nan,
            "seed_ari_std": np.nan,
        }

    values = np.asarray(scores, dtype=float)
    return {
        "seed_ari_mean": float(values.mean()),
        "seed_ari_min": float(values.min()),
        "seed_ari_std": float(values.std(ddof=0)),
    }


def robust_profile_scores(
    data: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    noise_label: int = -1,
) -> pd.DataFrame:
    """Return cluster medians relative to the global median and IQR."""
    labels = np.asarray(labels)
    mask = labels != noise_label
    working = data.loc[mask, feature_columns].copy()
    working["cluster"] = labels[mask]

    global_median = working[feature_columns].median()
    global_iqr = (
        working[feature_columns].quantile(0.75)
        - working[feature_columns].quantile(0.25)
    ).replace(0, 1.0)

    medians = working.groupby("cluster", observed=True)[feature_columns].median()
    robust_scores = (medians - global_median) / global_iqr
    robust_scores.index.name = "cluster"
    return robust_scores.reset_index()
