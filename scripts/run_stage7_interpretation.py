"""Run Stage 7 final profile interpretation and decision-support analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretation import (  # noqa: E402
    CORE_FEATURES,
    PROFILE_NAMES,
    attention_dimension_table,
    chi_square_effect,
    clean_tianjin_metadata,
    read_vehicle_track,
    robust_profile_scores,
    standardized_residuals,
    wilson_interval,
)


PCA_COLUMNS = ["PC1", "PC2", "PC3", "PC4", "PC5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final profile interpretation, Tianjin metadata analysis, "
            "representative trajectories, and decision-support tables."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/data"),
        help="Directory containing the private recording ZIP archives.",
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


def load_data() -> pd.DataFrame:
    processed = PROJECT_ROOT / "data" / "processed"

    features = pd.read_csv(
        processed / "sind_full_core_behavior_features.csv"
    )
    assignments = pd.read_csv(
        processed / "stage5_final_cluster_assignments.csv"
    )

    assignment_columns = [
        "trajectory_uid",
        "cluster_raw",
        "profile_id",
        "profile_name",
        *PCA_COLUMNS,
    ]

    merged = features.merge(
        assignments[assignment_columns],
        on="trajectory_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(features):
        raise ValueError(
            "Feature table and final assignments do not match one-to-one."
        )
    return merged


def profile_summary_tables(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = (
        data.groupby(["profile_id", "profile_name"])
        .size()
        .reset_index(name="trajectory_count")
    )
    counts["percentage"] = (
        counts["trajectory_count"] / len(data) * 100
    )

    medians = (
        data.groupby(["profile_id", "profile_name"])[
            CORE_FEATURES
        ]
        .median()
        .reset_index()
    )
    means = (
        data.groupby(["profile_id", "profile_name"])[
            CORE_FEATURES
        ]
        .mean()
        .reset_index()
    )

    summary = counts.merge(
        medians,
        on=["profile_id", "profile_name"],
        validate="one_to_one",
    )
    return summary, medians, means


def build_tianjin_tables(
    data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tianjin = clean_tianjin_metadata(
        data.loc[data["city"].eq("Tianjin")].copy()
    )

    # CrossType with Others preserved for transparency.
    cross_counts = pd.crosstab(
        [
            tianjin["profile_id"],
            tianjin["profile_name"],
        ],
        tianjin["CrossType_clean"],
        dropna=False,
    )
    cross_pct = (
        cross_counts.div(cross_counts.sum(axis=1), axis=0)
        * 100
    )

    cross_long = (
        cross_counts.stack()
        .rename("trajectory_count")
        .reset_index()
    )
    cross_pct_long = (
        cross_pct.stack()
        .rename("profile_percentage")
        .reset_index()
    )
    cross_long = cross_long.merge(
        cross_pct_long,
        on=[
            "profile_id",
            "profile_name",
            "CrossType_clean",
        ],
        validate="one_to_one",
    )

    # Primary association excludes sparse "Others".
    cross_primary = pd.crosstab(
        tianjin.loc[
            tianjin["CrossType_clean"].ne("Others"),
            "profile_id",
        ],
        tianjin.loc[
            tianjin["CrossType_clean"].ne("Others"),
            "CrossType_clean",
        ],
    ).reindex(index=[1, 2, 3, 4], fill_value=0)

    cross_all = pd.crosstab(
        tianjin["profile_id"],
        tianjin["CrossType_clean"],
    ).reindex(index=[1, 2, 3, 4], fill_value=0)

    cross_tests = pd.DataFrame(
        [
            {
                "association": "Profile x CrossType (excluding Others)",
                **chi_square_effect(cross_primary),
            },
            {
                "association": "Profile x CrossType (including Others)",
                **chi_square_effect(cross_all),
            },
        ]
    )

    cross_residuals = standardized_residuals(
        cross_primary
    )
    cross_residuals.index.name = "profile_id"
    cross_residuals = (
        cross_residuals.reset_index()
        .melt(
            id_vars="profile_id",
            var_name="CrossType_clean",
            value_name="standardized_residual",
        )
    )
    cross_residuals["profile_name"] = cross_residuals[
        "profile_id"
    ].map(PROFILE_NAMES)

    # Violation categories after trailing-space normalization.
    violation_counts = pd.crosstab(
        [
            tianjin["profile_id"],
            tianjin["profile_name"],
        ],
        tianjin["Signal_Violation_clean"],
        dropna=False,
    )
    violation_pct = (
        violation_counts.div(
            violation_counts.sum(axis=1),
            axis=0,
        )
        * 100
    )

    violation_long = (
        violation_counts.stack()
        .rename("trajectory_count")
        .reset_index()
    )
    violation_pct_long = (
        violation_pct.stack()
        .rename("profile_percentage")
        .reset_index()
    )
    violation_long = violation_long.merge(
        violation_pct_long,
        on=[
            "profile_id",
            "profile_name",
            "Signal_Violation_clean",
        ],
        validate="one_to_one",
    )

    binary_table = pd.crosstab(
        tianjin["profile_id"],
        tianjin["Any_Violation"],
    ).reindex(
        index=[1, 2, 3, 4],
        columns=[False, True],
        fill_value=0,
    )

    category_table = pd.crosstab(
        tianjin["profile_id"],
        tianjin["Signal_Violation_clean"],
    ).reindex(index=[1, 2, 3, 4], fill_value=0)

    violation_tests = pd.DataFrame(
        [
            {
                "association": "Profile x Any signal violation",
                **chi_square_effect(binary_table),
            },
            {
                "association": "Profile x Violation category",
                **chi_square_effect(category_table),
            },
        ]
    )

    overall_rate = float(tianjin["Any_Violation"].mean())
    violation_summary_rows: list[dict[str, object]] = []

    for (profile_id, profile_name), group in tianjin.groupby(
        ["profile_id", "profile_name"],
        sort=True,
    ):
        total = len(group)
        violations = int(group["Any_Violation"].sum())
        rate = violations / total
        lower, upper = wilson_interval(violations, total)

        violation_summary_rows.append(
            {
                "profile_id": int(profile_id),
                "profile_name": profile_name,
                "tianjin_trajectories": int(total),
                "violations": violations,
                "violation_rate": rate,
                "violation_rate_pct": rate * 100,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "relative_to_overall": (
                    rate / overall_rate
                    if overall_rate > 0
                    else np.nan
                ),
            }
        )

    violation_summary = pd.DataFrame(
        violation_summary_rows
    )

    violation_residuals = standardized_residuals(
        binary_table
    )
    violation_residuals.index.name = "profile_id"
    violation_residuals = (
        violation_residuals.reset_index()
        .melt(
            id_vars="profile_id",
            var_name="Any_Violation",
            value_name="standardized_residual",
        )
    )
    violation_residuals["profile_name"] = violation_residuals[
        "profile_id"
    ].map(PROFILE_NAMES)

    return {
        "tianjin_clean": tianjin,
        "cross_long": cross_long,
        "cross_tests": cross_tests,
        "cross_residuals": cross_residuals,
        "violation_long": violation_long,
        "violation_tests": violation_tests,
        "violation_summary": violation_summary,
        "violation_residuals": violation_residuals,
    }


def representative_trajectories(
    data: pd.DataFrame,
    raw_root: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    model = joblib.load(
        PROJECT_ROOT
        / "models"
        / "stage5_kmeans_k4_pca_model.joblib"
    )
    centers = model.cluster_centers_

    working = data.copy()
    working["distance_to_centroid"] = np.linalg.norm(
        working[PCA_COLUMNS].to_numpy()
        - centers[working["cluster_raw"].to_numpy()],
        axis=1,
    )

    representatives = (
        working.sort_values("distance_to_centroid")
        .groupby(
            ["profile_id", "profile_name"],
            as_index=False,
            sort=True,
        )
        .head(1)
        .sort_values("profile_id")
        .copy()
    )

    output_rows: list[dict[str, object]] = []

    for _, row in representatives.iterrows():
        archive_path = raw_root / row["source_archive"]
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Representative archive missing: {archive_path}"
            )

        track = read_vehicle_track(
            archive_path=archive_path,
            track_id=int(row["track_id"]),
        )

        profile_id = int(row["profile_id"])
        profile_slug = (
            str(row["profile_name"])
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        track_output = (
            PROJECT_ROOT
            / "outputs"
            / "tables"
            / f"stage7_representative_profile_{profile_id}_timeseries.csv"
        )
        track[
            [
                "track_id",
                "frame_id",
                "time_s",
                "x",
                "y",
                "speed_mps",
                "a_lon",
                "jerk_mps3",
            ]
        ].to_csv(track_output, index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(track["x"], track["y"])
        ax.scatter(
            [track["x"].iloc[0]],
            [track["y"].iloc[0]],
            marker="o",
            label="Start",
        )
        ax.scatter(
            [track["x"].iloc[-1]],
            [track["y"].iloc[-1]],
            marker="x",
            label="End",
        )
        ax.set_title(
            f"Representative path — {row['profile_name']}"
        )
        ax.set_xlabel("x coordinate (m)")
        ax.set_ylabel("y coordinate (m)")
        ax.legend()
        ax.axis("equal")
        plt.tight_layout()
        path_figure = (
            figures_dir
            / f"stage7_representative_{profile_slug}_path.png"
        )
        plt.savefig(
            path_figure,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(track["time_s"], track["speed_mps"])
        ax.axhline(
            0.5,
            linestyle="--",
            label="Stop threshold: 0.5 m/s",
        )
        ax.set_title(
            f"Representative speed — {row['profile_name']}"
        )
        ax.set_xlabel("Observed time (s)")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        plt.tight_layout()
        speed_figure = (
            figures_dir
            / f"stage7_representative_{profile_slug}_speed.png"
        )
        plt.savefig(
            speed_figure,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(track["time_s"], track["a_lon"])
        ax.axhline(0, linestyle="--")
        ax.set_title(
            f"Representative longitudinal acceleration — "
            f"{row['profile_name']}"
        )
        ax.set_xlabel("Observed time (s)")
        ax.set_ylabel("Longitudinal acceleration (m/s²)")
        plt.tight_layout()
        acceleration_figure = (
            figures_dir
            / f"stage7_representative_{profile_slug}_acceleration.png"
        )
        plt.savefig(
            acceleration_figure,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        output_rows.append(
            {
                "profile_id": profile_id,
                "profile_name": row["profile_name"],
                "trajectory_uid": row["trajectory_uid"],
                "city": row["city"],
                "recording_id": row["recording_id"],
                "track_id": int(row["track_id"]),
                "source_archive": row["source_archive"],
                "distance_to_centroid": float(
                    row["distance_to_centroid"]
                ),
                "trajectory_duration_s": float(
                    row["trajectory_duration_s"]
                ),
                "travel_distance_m": float(
                    row["travel_distance_m"]
                ),
                "mean_speed_mps": float(row["mean_speed_mps"]),
                "max_speed_mps": float(row["max_speed_mps"]),
                "max_acceleration_mps2": float(
                    row["max_acceleration_mps2"]
                ),
                "max_deceleration_mps2": float(
                    row["max_deceleration_mps2"]
                ),
                "stopped_time_ratio": float(
                    row["stopped_time_ratio"]
                ),
                "path_figure": str(path_figure.relative_to(PROJECT_ROOT)),
                "speed_figure": str(
                    speed_figure.relative_to(PROJECT_ROOT)
                ),
                "acceleration_figure": str(
                    acceleration_figure.relative_to(PROJECT_ROOT)
                ),
                "timeseries_file": str(
                    track_output.relative_to(PROJECT_ROOT)
                ),
            }
        )

    return pd.DataFrame(output_rows)


def create_figures(
    profile_summary: pd.DataFrame,
    robust_scores: pd.DataFrame,
    tianjin_tables: dict[str, pd.DataFrame],
    attention: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        profile_summary["profile_name"],
        profile_summary["trajectory_count"],
    )
    ax.set_title("Final behavioral-profile sizes")
    ax.set_xlabel("Behavioral profile")
    ax.set_ylabel("Trajectory count")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "stage7_profile_sizes.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    heatmap = robust_scores.set_index(
        "profile_name"
    )[CORE_FEATURES]
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(
        heatmap.to_numpy(),
        aspect="auto",
    )
    ax.set_xticks(range(len(CORE_FEATURES)))
    ax.set_xticklabels(CORE_FEATURES, rotation=90)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    fig.colorbar(
        image,
        ax=ax,
        label="Median difference / global IQR",
    )
    ax.set_title(
        "Behavioral profile signatures in robust standardized units"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / "stage7_profile_robust_score_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    cross = tianjin_tables["cross_long"].pivot(
        index="profile_name",
        columns="CrossType_clean",
        values="profile_percentage",
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(cross))
    for column in cross.columns:
        values = cross[column].to_numpy()
        ax.bar(
            cross.index,
            values,
            bottom=bottom,
            label=column,
        )
        bottom += values
    ax.set_title("Tianjin crossing-type distribution by profile")
    ax.set_xlabel("Behavioral profile")
    ax.set_ylabel("Percentage within profile")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        figures_dir / "stage7_tianjin_crosstype_by_profile.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    violation = tianjin_tables[
        "violation_summary"
    ].sort_values("profile_id")
    errors = np.vstack(
        [
            violation["violation_rate"]
            - violation["wilson_95_lower"],
            violation["wilson_95_upper"]
            - violation["violation_rate"],
        ]
    ) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        violation["profile_name"],
        violation["violation_rate_pct"],
        yerr=errors,
        capsize=4,
    )
    ax.set_title(
        "Tianjin signal-violation rate by behavioral profile"
    )
    ax.set_xlabel("Behavioral profile")
    ax.set_ylabel("Any signal violation (%)")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "stage7_tianjin_violation_rate_by_profile.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    attention_plot = attention.set_index(
        "profile_name"
    )[
        [
            "dynamic_maneuver_priority",
            "signal_queue_priority",
            "speed_adjustment_priority",
        ]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        attention_plot.to_numpy(),
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    ax.set_xticks(range(len(attention_plot.columns)))
    ax.set_xticklabels(
        [
            "Dynamic maneuver",
            "Signal/queue",
            "Speed adjustment",
        ]
    )
    ax.set_yticks(range(len(attention_plot.index)))
    ax.set_yticklabels(attention_plot.index)
    fig.colorbar(
        image,
        ax=ax,
        label="Within-profile ranking score (0–100)",
    )
    ax.set_title(
        "Operational-attention dimensions by behavioral profile"
    )
    plt.tight_layout()
    plt.savefig(
        figures_dir / "stage7_attention_dimension_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    logs_dir = PROJECT_ROOT / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        logs_dir / "stage7_interpretation.log"
    )

    started = time.perf_counter()
    data = load_data()

    profile_summary, profile_medians, profile_means = (
        profile_summary_tables(data)
    )
    robust_scores = robust_profile_scores(data)
    tianjin_tables = build_tianjin_tables(data)

    attention = attention_dimension_table(
        profile_medians=profile_medians,
        violation_summary=tianjin_tables[
            "violation_summary"
        ],
    )

    representatives = representative_trajectories(
        data=data,
        raw_root=args.raw_root,
        figures_dir=figures_dir,
    )

    # Save profile tables.
    profile_summary.to_csv(
        tables_dir / "stage7_profile_summary.csv",
        index=False,
    )
    profile_medians.to_csv(
        tables_dir / "stage7_profile_medians.csv",
        index=False,
    )
    profile_means.to_csv(
        tables_dir / "stage7_profile_means.csv",
        index=False,
    )
    robust_scores.to_csv(
        tables_dir / "stage7_profile_robust_scores.csv",
        index=False,
    )

    # Save Tianjin tables.
    tianjin_tables["cross_long"].to_csv(
        tables_dir / "stage7_tianjin_crosstype_by_profile.csv",
        index=False,
    )
    tianjin_tables["cross_tests"].to_csv(
        tables_dir / "stage7_tianjin_crosstype_association.csv",
        index=False,
    )
    tianjin_tables["cross_residuals"].to_csv(
        tables_dir / "stage7_tianjin_crosstype_residuals.csv",
        index=False,
    )
    tianjin_tables["violation_long"].to_csv(
        tables_dir / "stage7_tianjin_violation_categories_by_profile.csv",
        index=False,
    )
    tianjin_tables["violation_tests"].to_csv(
        tables_dir / "stage7_tianjin_violation_association.csv",
        index=False,
    )
    tianjin_tables["violation_summary"].to_csv(
        tables_dir / "stage7_tianjin_violation_rate_by_profile.csv",
        index=False,
    )
    tianjin_tables["violation_residuals"].to_csv(
        tables_dir / "stage7_tianjin_violation_residuals.csv",
        index=False,
    )

    attention.to_csv(
        tables_dir / "stage7_decision_support_matrix.csv",
        index=False,
    )
    representatives.to_csv(
        tables_dir / "stage7_representative_trajectories.csv",
        index=False,
    )

    create_figures(
        profile_summary=profile_summary,
        robust_scores=robust_scores,
        tianjin_tables=tianjin_tables,
        attention=attention,
        figures_dir=figures_dir,
    )

    stage6_summary = pd.read_csv(
        tables_dir / "stage6_validation_summary.csv"
    )
    stage5_config_path = (
        tables_dir / "stage5_selected_model_config.json"
    )
    with open(
        stage5_config_path,
        encoding="utf-8",
    ) as stream:
        stage5_config = json.load(stream)

    final_evidence = pd.DataFrame(
        [
            {
                "evidence_category": "Coverage",
                "metric": "Trajectory coverage",
                "value": stage5_config["coverage"],
                "interpretation": (
                    "All 19,948 eligible trajectories receive a profile."
                ),
            },
            {
                "evidence_category": "Internal structure",
                "metric": "Silhouette sample 5000",
                "value": stage5_config["silhouette_sample_5000"],
                "interpretation": (
                    "Moderate separation; profiles overlap and should not "
                    "be presented as perfectly isolated classes."
                ),
            },
            {
                "evidence_category": "Seed stability",
                "metric": "Minimum pairwise ARI",
                "value": stage5_config[
                    "production_seed_pairwise_ari_min"
                ],
                "interpretation": (
                    "The four-profile partition is highly stable to "
                    "K-Means initialization."
                ),
            },
            {
                "evidence_category": "Bootstrap stability",
                "metric": "Minimum bootstrap ARI",
                "value": stage5_config["bootstrap_ari_min"],
                "interpretation": (
                    "The partition remains reproducible under resampled "
                    "trajectory composition."
                ),
            },
            {
                "evidence_category": "Cross-city validation",
                "metric": "Mean leave-one-city-out ARI",
                "value": float(
                    stage6_summary.loc[
                        stage6_summary[
                            "validation_level"
                        ].eq("leave_one_city_out"),
                        "ari_mean",
                    ].iloc[0]
                ),
                "interpretation": (
                    "The structure transfers across cities, with stronger "
                    "context sensitivity in Changchun."
                ),
            },
            {
                "evidence_category": "Cross-recording validation",
                "metric": "Median leave-one-recording-out ARI",
                "value": float(
                    stage6_summary.loc[
                        stage6_summary[
                            "validation_level"
                        ].eq("leave_one_recording_out"),
                        "ari_median",
                    ].iloc[0]
                ),
                "interpretation": (
                    "The structure is highly reproducible across recording "
                    "sessions."
                ),
            },
        ]
    )
    final_evidence.to_csv(
        tables_dir / "stage7_final_model_evidence.csv",
        index=False,
    )

    config = {
        "stage": 7,
        "selected_model": "KMeans k=4",
        "profile_names": PROFILE_NAMES,
        "tianjin_metadata_scope": (
            "CrossType and Signal_Violation_Behavior are interpreted "
            "only for Tianjin and were not clustering inputs."
        ),
        "decision_support_warning": (
            "Attention dimensions are descriptive operational priorities, "
            "not crash probabilities or causal risk estimates."
        ),
        "representative_selection": (
            "Nearest observed trajectory to each K-Means centroid in the "
            "five-dimensional PCA space."
        ),
    }
    with open(
        tables_dir / "stage7_interpretation_config.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(config, stream, indent=2)

    elapsed = time.perf_counter() - started
    logging.info(
        "Stage 7 completed in %.2f seconds.",
        elapsed,
    )

    print("\n=== STAGE 7 INTERPRETATION COMPLETE ===")
    print(f"Trajectories interpreted: {len(data):,}")
    print("\nProfile summary:")
    print(
        profile_summary[
            [
                "profile_id",
                "profile_name",
                "trajectory_count",
                "percentage",
            ]
        ].to_string(index=False)
    )
    print("\nTianjin violation summary:")
    print(
        tianjin_tables["violation_summary"].to_string(
            index=False
        )
    )
    print("\nAssociation tests:")
    print(
        pd.concat(
            [
                tianjin_tables["cross_tests"],
                tianjin_tables["violation_tests"],
            ],
            ignore_index=True,
        ).to_string(index=False)
    )
    print("\nDecision-support matrix:")
    print(
        attention[
            [
                "profile_id",
                "profile_name",
                "dynamic_maneuver_priority",
                "signal_queue_priority",
                "speed_adjustment_priority",
                "tianjin_violation_rate_pct",
                "attention_tier",
            ]
        ].to_string(index=False)
    )
    print(f"\nElapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
