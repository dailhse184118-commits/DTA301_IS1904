"""Generate aggregate EDA tables and figures from the trajectory feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import calculate_iqr_outlier_summary, validate_eda_input

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sind_full_core_behavior_features.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = PROJECT_ROOT / "outputs/tables"
    figures = PROJECT_ROOT / "outputs/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.feature_path)
    validate_eda_input(data, CORE_FEATURES, expected_recordings=56)

    statistics = data[CORE_FEATURES].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T
    statistics["skewness"] = data[CORE_FEATURES].skew()
    statistics["zero_percentage"] = data[CORE_FEATURES].eq(0).mean() * 100
    statistics.index.name = "feature"
    statistics.reset_index().to_csv(
        tables / "feature_descriptive_statistics.csv", index=False
    )

    calculate_iqr_outlier_summary(data, CORE_FEATURES).to_csv(
        tables / "feature_outlier_audit.csv", index=False
    )

    correlation = data[CORE_FEATURES].corr(method="spearman")
    correlation.to_csv(tables / "spearman_correlation.csv")

    city_summary = (
        data.groupby("city")[CORE_FEATURES]
        .median()
        .reset_index()
    )
    city_summary.to_csv(tables / "city_feature_medians.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(CORE_FEATURES)))
    ax.set_yticks(range(len(CORE_FEATURES)))
    ax.set_xticklabels(CORE_FEATURES, rotation=90)
    ax.set_yticklabels(CORE_FEATURES)
    fig.colorbar(image, ax=ax)
    ax.set_title("Spearman correlation among core features")
    plt.tight_layout()
    plt.savefig(figures / "stage3_spearman_correlation.png", dpi=180)
    plt.close(fig)

    print(f"EDA complete for {len(data):,} trajectories.")


if __name__ == "__main__":
    main()
