"""Apply bootstrap stability gates and finalize the Stage 4 preprocessing choice."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import PreprocessingSpec, apply_preprocessing  # noqa: E402

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

SPECS = {
    "A_winsor_standard": PreprocessingSpec(
        "A_winsor_standard", True, "standard"
    ),
    "B_raw_robust": PreprocessingSpec(
        "B_raw_robust", False, "robust"
    ),
    "C_winsor_robust": PreprocessingSpec(
        "C_winsor_robust", True, "robust"
    ),
}


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    score = series.rank(method="average", pct=True)
    return score if higher_is_better else 1.0 - score + (1.0 / len(series))


def main() -> None:
    tables = PROJECT_ROOT / "outputs" / "tables"
    processed = PROJECT_ROOT / "data" / "processed"

    internal = pd.read_csv(
        tables / "stage4_preprocessing_benchmark_summary.csv"
    )
    boot = pd.read_csv(
        tables / "stage4_all_candidates_bootstrap_stability.csv"
    )

    boot_summary = (
        boot.groupby(["pipeline", "k"], as_index=False)
        .agg(
            bootstrap_ari_mean=("ari", "mean"),
            bootstrap_ari_median=("ari", "median"),
            bootstrap_ari_min=("ari", "min"),
            bootstrap_unstable_repeats=("ari", lambda values: int((values < 0.80).sum())),
            bootstrap_min_cluster_share_mean=("min_cluster_share", "mean"),
            bootstrap_min_cluster_share_min=("min_cluster_share", "min"),
        )
    )
    boot_summary.to_csv(
        tables / "stage4_candidate_bootstrap_summary.csv", index=False
    )

    final = internal.merge(boot_summary, on=["pipeline", "k"], how="left")
    final["passes_bootstrap_stability_gate"] = (
        (final["bootstrap_ari_min"] >= 0.80)
        & (final["bootstrap_unstable_repeats"] == 0)
        & (final["bootstrap_min_cluster_share_min"] >= 0.02)
    )

    final["silhouette_score_component"] = percentile_score(final["silhouette"], True)
    final["dbi_score_component"] = percentile_score(final["davies_bouldin"], False)
    final["bootstrap_mean_component"] = percentile_score(
        final["bootstrap_ari_mean"], True
    )
    final["bootstrap_min_component"] = percentile_score(
        final["bootstrap_ari_min"], True
    )
    final["city_risk_component"] = percentile_score(
        final["balanced_accuracy_mean"], False
    )
    final["cluster_balance_component"] = percentile_score(
        final["minimum_cluster_share"], True
    )

    final["stability_gated_selection_score"] = (
        0.25 * final["silhouette_score_component"]
        + 0.15 * final["dbi_score_component"]
        + 0.25 * final["bootstrap_mean_component"]
        + 0.20 * final["bootstrap_min_component"]
        + 0.10 * final["city_risk_component"]
        + 0.05 * final["cluster_balance_component"]
    )
    final.loc[
        ~final["passes_bootstrap_stability_gate"],
        "stability_gated_selection_score",
    ] = -1.0

    final = final.sort_values(
        ["passes_bootstrap_stability_gate", "stability_gated_selection_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected = final.iloc[0]
    selected_pipeline = str(selected["pipeline"])
    selected_k = int(selected["k"])

    data = pd.read_csv(
        processed / "sind_full_core_behavior_features.csv"
    )
    result = apply_preprocessing(
        data=data,
        feature_columns=CORE_FEATURES,
        continuous_features=CONTINUOUS_FEATURES,
        spec=SPECS[selected_pipeline],
    )
    X_pca = result["X_pca"]
    model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=30,
        max_iter=500,
    )
    labels = model.fit_predict(X_pca)

    selected_scores = data[
        ["trajectory_uid", "city", "recording_id", "track_id"]
    ].copy()
    for index in range(X_pca.shape[1]):
        selected_scores[f"PC{index + 1}"] = X_pca[:, index]
    selected_scores["preliminary_kmeans_label"] = labels
    selected_scores.to_csv(
        processed / "stage4_selected_pca_scores.csv", index=False
    )

    selected_bootstrap = boot.loc[
        (boot["pipeline"] == selected_pipeline)
        & (boot["k"] == selected_k)
    ].copy()
    selected_bootstrap.to_csv(
        tables / "stage4_selected_bootstrap_stability.csv", index=False
    )

    config = {
        "selected_pipeline": selected_pipeline,
        "selected_preliminary_k": selected_k,
        "selection_basis": "Internal metrics followed by mandatory bootstrap stability gate",
        "winsorize_continuous_features": SPECS[selected_pipeline].winsorize,
        "winsor_lower_quantile": 0.01,
        "winsor_upper_quantile": 0.99,
        "scaler": SPECS[selected_pipeline].scaler,
        "pca_variance_threshold": 0.90,
        "pca_components": int(X_pca.shape[1]),
        "pca_retained_variance": float(selected["pca_retained_variance"]),
        "bootstrap_ari_mean": float(selected["bootstrap_ari_mean"]),
        "bootstrap_ari_min": float(selected["bootstrap_ari_min"]),
        "bootstrap_unstable_repeats": int(selected["bootstrap_unstable_repeats"]),
        "preliminary_k_only": True,
        "rejected_internal_winner": "B_raw_robust",
        "rejection_reason": (
            "Although B_raw_robust had the strongest preliminary silhouette and Hopkins scores, "
            "2 of 20 bootstrap resamples collapsed, its minimum ARI fell to approximately 0.001, "
            "and the smallest predicted cluster fell to one trajectory. It therefore failed the "
            "mandatory stability gate."
        ),
        "sensitivity_baseline": "A_winsor_standard",
        "note": (
            "C_winsor_robust is the Stage 4 primary preprocessing baseline. "
            "The preliminary k=2 result is not the final behavioral-profile decision. "
            "Stage 5 must compare K-Means, density-based, mixture, and hierarchical models, "
            "including more interpretable k values."
        ),
    }

    final.to_csv(
        tables / "stage4_preprocessing_selection_final.csv", index=False
    )
    (tables / "stage4_selected_preprocessing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(final[[
        "pipeline",
        "k",
        "silhouette",
        "davies_bouldin",
        "bootstrap_ari_mean",
        "bootstrap_ari_min",
        "bootstrap_unstable_repeats",
        "bootstrap_min_cluster_share_min",
        "passes_bootstrap_stability_gate",
        "stability_gated_selection_score",
    ]].to_string(index=False))
    print("\nSelected configuration:\n" + json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
