"""Run Stage 4 preprocessing, PCA, cluster tendency, and K-Means benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    PreprocessingSpec,
    QuantileWinsorizer,
    apply_preprocessing,
    hopkins_statistic,
    pca_loading_table,
)

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

SPECS = [
    PreprocessingSpec(
        name="A_winsor_standard",
        winsorize=True,
        scaler="standard",
    ),
    PreprocessingSpec(
        name="B_raw_robust",
        winsorize=False,
        scaler="robust",
    ),
    PreprocessingSpec(
        name="C_winsor_robust",
        winsorize=True,
        scaler="robust",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "sind_full_core_behavior_features.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    return parser.parse_args()


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


def build_cv_pipeline(spec: PreprocessingSpec) -> Pipeline:
    steps = []
    if spec.winsorize:
        steps.append(
            (
                "winsor",
                QuantileWinsorizer(
                    feature_names=CORE_FEATURES,
                    continuous_features=CONTINUOUS_FEATURES,
                    lower_quantile=0.01,
                    upper_quantile=0.99,
                ),
            )
        )
    else:
        steps.append(
            (
                "identity",
                FunctionTransformer(
                    lambda X: np.asarray(X, dtype=float),
                    validate=False,
                ),
            )
        )

    scaler = StandardScaler() if spec.scaler == "standard" else RobustScaler()
    steps.extend(
        [
            ("scaler", scaler),
            ("pca", PCA(n_components=0.90, svd_solver="full")),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return Pipeline(steps)


def metric_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    return ranked if higher_is_better else 1.0 - ranked + (1.0 / len(series))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    tables_dir = output_root / "outputs" / "tables"
    figures_dir = output_root / "outputs" / "figures"
    processed_dir = output_root / "data" / "processed"
    logs_dir = output_root / "logs"
    for folder in [tables_dir, figures_dir, processed_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    configure_logging(logs_dir / "stage4_preprocessing_benchmark.log")
    started = time.perf_counter()

    data = pd.read_csv(args.feature_path)
    X_frame = data[CORE_FEATURES]
    y_city = data["city"].astype(str)

    if data["trajectory_uid"].duplicated().any():
        raise ValueError("Duplicate trajectory_uid detected.")
    if X_frame.isna().any().any() or np.isinf(X_frame.to_numpy()).any():
        raise ValueError("Invalid values detected in core features.")

    logging.info("Loaded %d trajectories.", len(data))

    fitted = {}
    pca_variance_rows = []
    loading_tables = []
    hopkins_rows = []
    winsor_bounds_rows = []
    city_predictability_rows = []
    kmeans_rows = []

    for spec in SPECS:
        logging.info("Fitting preprocessing pipeline %s", spec.name)
        result = apply_preprocessing(
            data=data,
            feature_columns=CORE_FEATURES,
            continuous_features=CONTINUOUS_FEATURES,
            spec=spec,
        )
        fitted[spec.name] = result
        pca = result["pca"]
        X_pca = result["X_pca"]

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        for index, (ratio, cumulative_ratio) in enumerate(
            zip(pca.explained_variance_ratio_, cumulative), start=1
        ):
            pca_variance_rows.append(
                {
                    "pipeline": spec.name,
                    "component": f"PC{index}",
                    "component_number": index,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance": float(cumulative_ratio),
                }
            )
        loading_tables.append(
            pca_loading_table(pca, CORE_FEATURES, spec.name)
        )

        if result["winsorizer"] is not None:
            win = result["winsorizer"]
            for feature in CONTINUOUS_FEATURES:
                raw = data[feature]
                winsor_bounds_rows.append(
                    {
                        "pipeline": spec.name,
                        "feature": feature,
                        "lower_bound_1pct": float(win.lower_bounds_[feature]),
                        "upper_bound_99pct": float(win.upper_bounds_[feature]),
                        "below_lower_count": int((raw < win.lower_bounds_[feature]).sum()),
                        "above_upper_count": int((raw > win.upper_bounds_[feature]).sum()),
                    }
                )

        for seed in range(10):
            hopkins_rows.append(
                {
                    "pipeline": spec.name,
                    "repeat": seed + 1,
                    "hopkins_statistic": hopkins_statistic(
                        X_pca,
                        sample_size=1000,
                        random_state=100 + seed,
                    ),
                }
            )

        logging.info("Running city-predictability diagnostic for %s", spec.name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_pipeline = build_cv_pipeline(spec)
        scores = cross_val_score(
            cv_pipeline,
            X_frame,
            y_city,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=1,
        )
        city_predictability_rows.append(
            {
                "pipeline": spec.name,
                "balanced_accuracy_mean": float(scores.mean()),
                "balanced_accuracy_std": float(scores.std(ddof=1)),
                "chance_balanced_accuracy": 1.0 / y_city.nunique(),
            }
        )

        logging.info("Running preliminary K-Means grid for %s", spec.name)
        for k in range(2, 11):
            reference = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20,
                max_iter=500,
            )
            labels = reference.fit_predict(X_pca)
            counts = np.bincount(labels, minlength=k)

            seed_aris = []
            for seed in [7, 17, 27, 37, 47]:
                candidate = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=10,
                    max_iter=500,
                )
                candidate_labels = candidate.fit_predict(X_pca)
                seed_aris.append(adjusted_rand_score(labels, candidate_labels))

            kmeans_rows.append(
                {
                    "pipeline": spec.name,
                    "k": k,
                    "pca_components": int(X_pca.shape[1]),
                    "pca_retained_variance": float(cumulative[-1]),
                    "inertia": float(reference.inertia_),
                    "silhouette": float(
                        silhouette_score(
                            X_pca,
                            labels,
                            sample_size=min(4000, len(X_pca)),
                            random_state=42,
                        )
                    ),
                    "davies_bouldin": float(davies_bouldin_score(X_pca, labels)),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(X_pca, labels)
                    ),
                    "minimum_cluster_size": int(counts.min()),
                    "maximum_cluster_size": int(counts.max()),
                    "minimum_cluster_share": float(counts.min() / len(labels)),
                    "maximum_cluster_share": float(counts.max() / len(labels)),
                    "seed_stability_ari_mean": float(np.mean(seed_aris)),
                    "seed_stability_ari_min": float(np.min(seed_aris)),
                }
            )

    pca_variance = pd.DataFrame(pca_variance_rows)
    pca_loadings = pd.concat(loading_tables, ignore_index=True)
    hopkins = pd.DataFrame(hopkins_rows)
    winsor_bounds = pd.DataFrame(winsor_bounds_rows)
    city_predictability = pd.DataFrame(city_predictability_rows)
    kmeans_metrics = pd.DataFrame(kmeans_rows)

    # Within-pipeline composite score to identify each pipeline's strongest k.
    ranked_parts = []
    for pipeline_name, group in kmeans_metrics.groupby("pipeline", sort=False):
        group = group.copy()
        group["silhouette_rank_score"] = metric_percentile(group["silhouette"], True)
        group["dbi_rank_score"] = metric_percentile(group["davies_bouldin"], False)
        group["ch_rank_score"] = metric_percentile(group["calinski_harabasz"], True)
        group["stability_rank_score"] = metric_percentile(
            group["seed_stability_ari_mean"], True
        )
        group["balance_rank_score"] = metric_percentile(
            group["minimum_cluster_share"], True
        )
        group["within_pipeline_composite"] = (
            0.35 * group["silhouette_rank_score"]
            + 0.25 * group["dbi_rank_score"]
            + 0.15 * group["ch_rank_score"]
            + 0.20 * group["stability_rank_score"]
            + 0.05 * group["balance_rank_score"]
        )
        ranked_parts.append(group)
    kmeans_metrics = pd.concat(ranked_parts, ignore_index=True)

    best_rows = (
        kmeans_metrics.sort_values(
            ["pipeline", "within_pipeline_composite", "silhouette"],
            ascending=[True, False, False],
        )
        .groupby("pipeline", as_index=False)
        .head(1)
        .copy()
    )

    hopkins_summary = (
        hopkins.groupby("pipeline", as_index=False)["hopkins_statistic"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "hopkins_mean",
                "std": "hopkins_std",
                "min": "hopkins_min",
                "max": "hopkins_max",
            }
        )
    )

    benchmark = best_rows.merge(hopkins_summary, on="pipeline", how="left")
    benchmark = benchmark.merge(city_predictability, on="pipeline", how="left")

    benchmark["silhouette_selection_score"] = metric_percentile(
        benchmark["silhouette"], True
    )
    benchmark["dbi_selection_score"] = metric_percentile(
        benchmark["davies_bouldin"], False
    )
    benchmark["stability_selection_score"] = metric_percentile(
        benchmark["seed_stability_ari_mean"], True
    )
    benchmark["hopkins_selection_score"] = metric_percentile(
        benchmark["hopkins_mean"], True
    )
    benchmark["city_risk_selection_score"] = metric_percentile(
        benchmark["balanced_accuracy_mean"], False
    )
    benchmark["balance_selection_score"] = metric_percentile(
        benchmark["minimum_cluster_share"], True
    )
    benchmark["overall_selection_score"] = (
        0.30 * benchmark["silhouette_selection_score"]
        + 0.20 * benchmark["dbi_selection_score"]
        + 0.20 * benchmark["stability_selection_score"]
        + 0.10 * benchmark["hopkins_selection_score"]
        + 0.15 * benchmark["city_risk_selection_score"]
        + 0.05 * benchmark["balance_selection_score"]
    )
    benchmark = benchmark.sort_values(
        "overall_selection_score", ascending=False
    ).reset_index(drop=True)

    selected = benchmark.iloc[0]
    selected_pipeline = str(selected["pipeline"])
    selected_k = int(selected["k"])
    selected_result = fitted[selected_pipeline]
    selected_X = selected_result["X_pca"]

    # Bootstrap stability for the selected preprocessing/k candidate.
    reference = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=30,
        max_iter=500,
    )
    reference_labels = reference.fit_predict(selected_X)
    rng = np.random.default_rng(2026)
    bootstrap_rows = []
    sample_size = int(round(0.80 * len(selected_X)))
    for repeat in range(10):
        sample_indices = rng.choice(len(selected_X), size=sample_size, replace=True)
        model = KMeans(
            n_clusters=selected_k,
            random_state=1000 + repeat,
            n_init=20,
            max_iter=500,
        )
        model.fit(selected_X[sample_indices])
        predicted_full = model.predict(selected_X)
        bootstrap_rows.append(
            {
                "repeat": repeat + 1,
                "adjusted_rand_index_vs_reference": adjusted_rand_score(
                    reference_labels, predicted_full
                ),
            }
        )
    bootstrap_stability = pd.DataFrame(bootstrap_rows)

    # Save selected PCA scores and preliminary K-Means labels for traceability.
    score_columns = [f"PC{i + 1}" for i in range(selected_X.shape[1])]
    selected_scores = data[
        ["trajectory_uid", "city", "recording_id", "track_id"]
    ].copy()
    for i, column in enumerate(score_columns):
        selected_scores[column] = selected_X[:, i]
    selected_scores["preliminary_kmeans_label"] = reference_labels
    selected_scores.to_csv(
        processed_dir / "stage4_selected_pca_scores.csv", index=False
    )

    selection_config = {
        "selected_pipeline": selected_pipeline,
        "selected_preliminary_k": selected_k,
        "selection_score": float(selected["overall_selection_score"]),
        "winsorize_continuous_features": bool(
            next(spec for spec in SPECS if spec.name == selected_pipeline).winsorize
        ),
        "winsor_lower_quantile": 0.01,
        "winsor_upper_quantile": 0.99,
        "scaler": next(
            spec.scaler for spec in SPECS if spec.name == selected_pipeline
        ),
        "pca_variance_threshold": 0.90,
        "pca_components": int(selected_X.shape[1]),
        "pca_retained_variance": float(selected["pca_retained_variance"]),
        "preliminary_k_only": True,
        "note": (
            "The selected k is a Stage 4 preliminary benchmark candidate, not "
            "the final clustering decision. Stage 5 must compare full model "
            "families and validation results."
        ),
    }

    # Output tables.
    pca_variance.to_csv(tables_dir / "stage4_pca_explained_variance.csv", index=False)
    pca_loadings.to_csv(tables_dir / "stage4_pca_loadings.csv", index=False)
    hopkins.to_csv(tables_dir / "stage4_hopkins_repeats.csv", index=False)
    hopkins_summary.to_csv(tables_dir / "stage4_hopkins_summary.csv", index=False)
    winsor_bounds.to_csv(tables_dir / "stage4_winsorization_bounds.csv", index=False)
    city_predictability.to_csv(
        tables_dir / "stage4_city_predictability.csv", index=False
    )
    kmeans_metrics.to_csv(
        tables_dir / "stage4_preliminary_kmeans_metrics.csv", index=False
    )
    benchmark.to_csv(
        tables_dir / "stage4_preprocessing_benchmark_summary.csv", index=False
    )
    bootstrap_stability.to_csv(
        tables_dir / "stage4_selected_bootstrap_stability.csv", index=False
    )
    (tables_dir / "stage4_selected_preprocessing_config.json").write_text(
        json.dumps(selection_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Figures: one chart per diagnostic.
    fig, ax = plt.subplots(figsize=(9, 5))
    for pipeline_name, group in pca_variance.groupby("pipeline"):
        ax.plot(
            group["component_number"],
            group["cumulative_explained_variance"],
            marker="o",
            label=pipeline_name,
        )
    ax.axhline(0.90, linestyle="--")
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_pca_cumulative_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(hopkins_summary["pipeline"], hopkins_summary["hopkins_mean"])
    ax.axhline(0.50, linestyle="--")
    ax.set_title("Hopkins cluster-tendency statistic")
    ax.set_xlabel("Preprocessing pipeline")
    ax.set_ylabel("Mean Hopkins statistic")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_hopkins_by_pipeline.png", dpi=180)
    plt.close(fig)

    for metric, title, ylabel in [
        ("silhouette", "Preliminary K-Means silhouette", "Silhouette"),
        ("davies_bouldin", "Preliminary K-Means Davies–Bouldin", "Davies–Bouldin"),
        ("seed_stability_ari_mean", "K-Means seed stability", "Mean ARI"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for pipeline_name, group in kmeans_metrics.groupby("pipeline"):
            ax.plot(group["k"], group[metric], marker="o", label=pipeline_name)
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"stage4_{metric}_by_pipeline.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        city_predictability["pipeline"],
        city_predictability["balanced_accuracy_mean"],
    )
    ax.axhline(1.0 / y_city.nunique(), linestyle="--")
    ax.set_title("City predictability from transformed features")
    ax.set_xlabel("Preprocessing pipeline")
    ax.set_ylabel("Cross-validated balanced accuracy")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_city_predictability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sample_rng = np.random.default_rng(42)
    sample_indices = sample_rng.choice(
        len(selected_scores), size=min(6000, len(selected_scores)), replace=False
    )
    sample = selected_scores.iloc[sample_indices]
    for city_name, group in sample.groupby("city"):
        ax.scatter(group["PC1"], group["PC2"], s=8, alpha=0.45, label=city_name)
    ax.set_title(f"Selected PCA space by city: {selected_pipeline}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stage4_selected_pca_by_city.png", dpi=180)
    plt.close(fig)

    logging.info(
        "Selected preprocessing: %s; preliminary k=%d", selected_pipeline, selected_k
    )
    logging.info("Completed Stage 4 in %.2f seconds", time.perf_counter() - started)

    print("\n=== STAGE 4 COMPLETE ===")
    print(benchmark[[
        "pipeline",
        "k",
        "pca_components",
        "pca_retained_variance",
        "silhouette",
        "davies_bouldin",
        "seed_stability_ari_mean",
        "hopkins_mean",
        "balanced_accuracy_mean",
        "overall_selection_score",
    ]].to_string(index=False))
    print("\nSelected:", json.dumps(selection_config, ensure_ascii=False, indent=2))
    print("\nBootstrap ARI mean:", bootstrap_stability["adjusted_rand_index_vs_reference"].mean())
    print("Bootstrap ARI minimum:", bootstrap_stability["adjusted_rand_index_vs_reference"].min())


if __name__ == "__main__":
    main()
