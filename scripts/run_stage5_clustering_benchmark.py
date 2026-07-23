"""Run Stage 5 clustering-model benchmark on selected Stage 4 PCA scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import (  # noqa: E402
    EvaluationConfig,
    evaluate_partition,
    feature_separation_eta_squared,
    normalized_cluster_entropy,
    pairwise_seed_stability,
)

RANDOM_STATE = 42
PCA_COLUMNS = ["PC1", "PC2", "PC3", "PC4", "PC5"]
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


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def add_interpretability_metrics(
    row: dict[str, object],
    features: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, object]:
    eta = feature_separation_eta_squared(
        features,
        labels,
        CORE_FEATURES,
    )
    row["mean_feature_eta_squared"] = float(eta["eta_squared"].mean())
    row["max_feature_eta_squared"] = float(eta["eta_squared"].max())
    row["features_eta_squared_ge_0_10"] = int(
        (eta["eta_squared"] >= 0.10).sum()
    )
    row["cluster_entropy"] = normalized_cluster_entropy(labels)
    return row


def model_row(
    family: str,
    config_name: str,
    X: np.ndarray,
    labels: np.ndarray,
    features: pd.DataFrame,
    evaluation_config: EvaluationConfig,
    **parameters: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_family": family,
        "config_name": config_name,
        **parameters,
    }
    row.update(evaluate_partition(X, labels, evaluation_config))
    return add_interpretability_metrics(row, features, labels)


def stratified_sample_indices(
    city_labels: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Create a city-stratified sample for non-scalable algorithms."""
    sample_size = min(sample_size, len(city_labels))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state,
    )
    indices, _ = next(
        splitter.split(
            np.zeros(len(city_labels)),
            city_labels.astype(str),
        )
    )
    return np.sort(indices)


def fit_seed_labels(
    family: str,
    X: np.ndarray,
    n_clusters: int,
    seeds: list[int],
) -> list[np.ndarray]:
    label_sets: list[np.ndarray] = []

    for seed in seeds:
        if family == "KMeans":
            model = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=seed,
            )
            labels = model.fit_predict(X)
        elif family == "MiniBatchKMeans":
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                n_init=10,
                max_iter=300,
                random_state=seed,
            )
            labels = model.fit_predict(X)
        elif family == "GaussianMixture":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                n_init=2,
                reg_covar=1e-6,
                max_iter=300,
                random_state=seed,
            )
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported seeded family: {family}")

        label_sets.append(labels)

    return label_sets


def bootstrap_predictive_stability(
    family: str,
    X: np.ndarray,
    reference_labels: np.ndarray,
    n_clusters: int,
    repetitions: int = 10,
    sample_fraction: float = 0.80,
    random_state: int = 1200,
) -> pd.DataFrame:
    """Refit predictive clusterers on bootstrap samples and score full labels."""
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    sample_size = int(round(sample_fraction * len(X)))

    for repetition in range(repetitions):
        sample_indices = rng.choice(
            len(X),
            size=sample_size,
            replace=True,
        )

        seed = random_state + repetition
        if family == "KMeans":
            model = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=seed,
            )
        elif family == "MiniBatchKMeans":
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                n_init=10,
                max_iter=300,
                random_state=seed,
            )
        elif family == "GaussianMixture":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                n_init=2,
                reg_covar=1e-6,
                max_iter=300,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported predictive family: {family}")

        model.fit(X[sample_indices])
        predicted = model.predict(X)
        _, counts = np.unique(predicted, return_counts=True)

        rows.append(
            {
                "repetition": repetition + 1,
                "ari_vs_reference": adjusted_rand_score(
                    reference_labels,
                    predicted,
                ),
                "smallest_cluster_count": int(counts.min()),
                "smallest_cluster_percentage": float(
                    counts.min() / len(X) * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def subsample_density_stability(
    estimator_factory,
    X: np.ndarray,
    reference_labels: np.ndarray,
    repetitions: int = 8,
    sample_fraction: float = 0.80,
    random_state: int = 2200,
) -> pd.DataFrame:
    """Refit density models on subsamples and compare common observations."""
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    sample_size = int(round(sample_fraction * len(X)))

    for repetition in range(repetitions):
        indices = np.sort(
            rng.choice(
                len(X),
                size=sample_size,
                replace=False,
            )
        )
        model = estimator_factory()
        labels = model.fit_predict(X[indices])
        reference_subset = reference_labels[indices]

        rows.append(
            {
                "repetition": repetition + 1,
                "ari_vs_reference_on_subsample": adjusted_rand_score(
                    reference_subset,
                    labels,
                ),
                "coverage": float((labels != -1).mean()),
                "n_clusters": int(
                    len(set(labels)) - (1 if -1 in labels else 0)
                ),
            }
        )

    return pd.DataFrame(rows)


def leave_one_city_out_stability(
    family: str,
    X: np.ndarray,
    cities: pd.Series,
    reference_labels: np.ndarray,
    n_clusters: int,
) -> pd.DataFrame:
    """Fit without one city, then predict all observations and compare."""
    rows: list[dict[str, object]] = []

    for held_out_city in sorted(cities.unique()):
        train_mask = cities.ne(held_out_city).to_numpy()

        if family == "KMeans":
            model = KMeans(
                n_clusters=n_clusters,
                n_init=8,
                random_state=RANDOM_STATE,
            )
        elif family == "MiniBatchKMeans":
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1024,
                n_init=8,
                max_iter=300,
                random_state=RANDOM_STATE,
            )
        elif family == "GaussianMixture":
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                n_init=1,
                reg_covar=1e-6,
                max_iter=300,
                random_state=RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unsupported family: {family}")

        model.fit(X[train_mask])
        predicted_all = model.predict(X)
        held_labels = predicted_all[~train_mask]
        _, held_counts = np.unique(held_labels, return_counts=True)

        rows.append(
            {
                "held_out_city": held_out_city,
                "ari_vs_full_reference_all_rows": adjusted_rand_score(
                    reference_labels,
                    predicted_all,
                ),
                "held_out_smallest_cluster_percentage": float(
                    held_counts.min() / len(held_labels) * 100
                ),
                "held_out_clusters_present": int(len(held_counts)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    logs_dir = PROJECT_ROOT / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(logs_dir / "stage5_clustering_benchmark.log")
    warnings.filterwarnings("ignore", category=UserWarning)

    pca_scores = pd.read_csv(
        processed_dir / "stage4_selected_pca_scores.csv"
    )
    feature_data = pd.read_csv(
        processed_dir / "sind_full_core_behavior_features.csv"
    )

    merged = pca_scores.merge(
        feature_data[
            ["trajectory_uid", *CORE_FEATURES]
        ],
        on="trajectory_uid",
        how="left",
        validate="one_to_one",
    )

    if merged[CORE_FEATURES].isna().any().any():
        raise ValueError("Feature merge created missing values.")

    X = merged[PCA_COLUMNS].to_numpy(dtype=float)
    evaluation_config = EvaluationConfig(
        silhouette_sample_size=1500,
        random_state=RANDOM_STATE,
    )

    benchmark_rows: list[dict[str, object]] = []
    assignment_columns = merged[
        ["trajectory_uid", "city", "recording_id", "track_id"]
    ].copy()
    fitted_labels: dict[str, np.ndarray] = {}

    started = time.perf_counter()

    # ---------------------------------------------------------
    # K-Means: full data, k = 2..12
    # ---------------------------------------------------------
    logging.info("Running K-Means sweep.")
    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=8,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        name = f"kmeans_k{k}"
        fitted_labels[name] = labels
        row = model_row(
            "KMeans",
            name,
            X,
            labels,
            merged,
            evaluation_config,
            n_clusters_requested=k,
            inertia=float(model.inertia_),
        )
        benchmark_rows.append(row)

    # ---------------------------------------------------------
    # MiniBatch K-Means: full data, k = 2..12
    # ---------------------------------------------------------
    logging.info("Running MiniBatch K-Means sweep.")
    for k in range(2, 9):
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            n_init=8,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        name = f"minibatch_k{k}"
        fitted_labels[name] = labels
        row = model_row(
            "MiniBatchKMeans",
            name,
            X,
            labels,
            merged,
            evaluation_config,
            n_clusters_requested=k,
            inertia=float(model.inertia_),
        )
        benchmark_rows.append(row)

    # ---------------------------------------------------------
    # Gaussian mixture: full data, components = 2..10
    # ---------------------------------------------------------
    logging.info("Running Gaussian Mixture sweep.")
    for k in range(2, 7):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=1,
            reg_covar=1e-6,
            max_iter=300,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X)
        name = f"gmm_k{k}"
        fitted_labels[name] = labels
        row = model_row(
            "GaussianMixture",
            name,
            X,
            labels,
            merged,
            evaluation_config,
            n_clusters_requested=k,
            bic=float(model.bic(X)),
            aic=float(model.aic(X)),
            converged=bool(model.converged_),
        )
        benchmark_rows.append(row)

    # ---------------------------------------------------------
    # Agglomerative clustering on a city-stratified sample only
    # ---------------------------------------------------------
    logging.info("Running Agglomerative sample benchmark.")
    agglomerative_indices = stratified_sample_indices(
        merged["city"],
        sample_size=4000,
        random_state=RANDOM_STATE,
    )
    X_agg = X[agglomerative_indices]
    merged_agg = merged.iloc[agglomerative_indices].reset_index(drop=True)

    for k in range(2, 7):
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )
        labels = model.fit_predict(X_agg)
        name = f"agglomerative_sample_k{k}"
        row = model_row(
            "AgglomerativeSample",
            name,
            X_agg,
            labels,
            merged_agg,
            evaluation_config,
            n_clusters_requested=k,
            sample_size=len(X_agg),
            full_dataset_supported=False,
        )
        benchmark_rows.append(row)

    # ---------------------------------------------------------
    # Density-model benchmark on a city-stratified sample
    # ---------------------------------------------------------
    # DBSCAN can become computationally expensive when high-eps settings
    # create very large neighborhoods. Density models are therefore screened
    # on a fixed city-stratified sample, while scalable predictive models are
    # evaluated on all 19,948 trajectories.
    logging.info("Running density-model sample benchmark.")
    density_indices = stratified_sample_indices(
        merged["city"],
        sample_size=7000,
        random_state=RANDOM_STATE + 10,
    )
    X_density = X[density_indices]
    merged_density = merged.iloc[density_indices].reset_index(drop=True)
    density_assignments = merged.iloc[density_indices][
        ["trajectory_uid", "city", "recording_id", "track_id"]
    ].reset_index(drop=True)

    dbscan_configs: list[dict[str, float | int]] = []
    for min_samples in [15, 40, 80]:
        neighbors = NearestNeighbors(
            n_neighbors=min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        distances, _ = neighbors.fit(X_density).kneighbors(X_density)
        kth_distances = distances[:, -1]

        for quantile in [0.80, 0.90, 0.95]:
            eps = float(np.quantile(kth_distances, quantile))
            dbscan_configs.append(
                {
                    "min_samples": min_samples,
                    "eps_quantile": quantile,
                    "eps": eps,
                }
            )

    for config in dbscan_configs:
        model = DBSCAN(
            eps=config["eps"],
            min_samples=int(config["min_samples"]),
            n_jobs=-1,
        )
        labels = model.fit_predict(X_density)
        name = (
            f"dbscan_sample_ms{config['min_samples']}"
            f"_q{str(config['eps_quantile']).replace('.', '')}"
        )
        density_assignments[name] = labels
        row = model_row(
            "DBSCAN_Sample",
            name,
            X_density,
            labels,
            merged_density,
            evaluation_config,
            min_samples=int(config["min_samples"]),
            eps=float(config["eps"]),
            eps_quantile=float(config["eps_quantile"]),
            sample_size=len(X_density),
            full_dataset_supported=False,
        )
        benchmark_rows.append(row)

    for min_cluster_size in [150, 400, 800]:
        for min_samples in [15, 50]:
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                allow_single_cluster=False,
                n_jobs=-1,
            )
            labels = model.fit_predict(X_density)
            name = (
                f"hdbscan_sample_mcs{min_cluster_size}"
                f"_ms{min_samples}"
            )
            density_assignments[name] = labels
            row = model_row(
                "HDBSCAN_Sample",
                name,
                X_density,
                labels,
                merged_density,
                evaluation_config,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                sample_size=len(X_density),
                full_dataset_supported=False,
            )
            benchmark_rows.append(row)

    density_assignments.to_csv(
        processed_dir / "stage5_density_sample_assignments.csv",
        index=False,
    )

    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(
        tables_dir / "stage5_model_benchmark.csv",
        index=False,
    )

    for family, filename in [
        ("KMeans", "stage5_kmeans_k_sweep.csv"),
        ("MiniBatchKMeans", "stage5_minibatch_k_sweep.csv"),
        ("GaussianMixture", "stage5_gmm_k_sweep.csv"),
        ("AgglomerativeSample", "stage5_agglomerative_sample_sweep.csv"),
        ("DBSCAN_Sample", "stage5_dbscan_sample_grid.csv"),
        ("HDBSCAN_Sample", "stage5_hdbscan_sample_grid.csv"),
    ]:
        benchmark.loc[
            benchmark["model_family"].eq(family)
        ].to_csv(tables_dir / filename, index=False)

    elapsed = time.perf_counter() - started
    logging.info("Stage 5 model sweep completed in %.1f seconds.", elapsed)

    print("\n=== STAGE 5 MODEL SWEEP COMPLETE ===")
    print(f"Observations: {len(X):,}")
    print(f"Configurations evaluated: {len(benchmark):,}")
    print(f"Benchmark table: {tables_dir / 'stage5_model_benchmark.csv'}")
    print("Run run_stage5_kmeans_stability.py next, then finalize_stage5_selection.py.")


if __name__ == "__main__":
    main()
