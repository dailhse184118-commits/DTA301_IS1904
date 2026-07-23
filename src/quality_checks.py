"""Quality rules for passenger-car trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityConfig:
    """Configurable, conservative trajectory-quality thresholds."""

    minimum_duration_s: float = 5.0
    stationary_max_travel_distance_m: float = 0.5
    stationary_max_speed_mps: float = 0.1


def add_quality_flags(
    trajectory_summary: pd.DataFrame,
    config: QualityConfig,
) -> pd.DataFrame:
    """Add transparent quality flags and an exclusion reason.

    A trajectory is considered stationary only when it traveled no more than
    the configured distance AND never exceeded the configured speed. This
    conservative conjunction avoids removing ordinary vehicles that simply
    waited at a red light for part of their observed trajectory.
    """
    output = trajectory_summary.copy()

    output["flag_duration_below_minimum"] = (
        output["trajectory_duration_s"] < config.minimum_duration_s
    )
    output["flag_stationary_full_trajectory"] = (
        (
            output["travel_distance_m"]
            <= config.stationary_max_travel_distance_m
        )
        & (
            output["max_speed_mps"]
            <= config.stationary_max_speed_mps
        )
    )

    def build_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["flag_duration_below_minimum"]:
            reasons.append(
                f"duration_below_{config.minimum_duration_s:g}s"
            )
        if row["flag_stationary_full_trajectory"]:
            reasons.append("stationary_full_trajectory")
        return "; ".join(reasons)

    output["exclusion_reason"] = output.apply(build_reason, axis=1)
    output["is_modeling_eligible"] = output["exclusion_reason"].eq("")
    return output


def validate_feature_table(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Fail loudly if a final modeling table is not safe to use."""
    if features["trajectory_uid"].duplicated().any():
        duplicates = int(features["trajectory_uid"].duplicated().sum())
        raise ValueError(
            f"Final feature table contains {duplicates} duplicate trajectory_uid values."
        )

    missing = int(features[feature_columns].isna().sum().sum())
    if missing:
        raise ValueError(
            f"Final feature table contains {missing} missing feature values."
        )

    numeric = features[feature_columns].select_dtypes(include="number")
    if not numeric.empty:
        import numpy as np

        infinite = int(np.isinf(numeric.to_numpy()).sum())
        if infinite:
            raise ValueError(
                f"Final feature table contains {infinite} infinite values."
            )

    if len(features) != features["trajectory_uid"].nunique():
        raise ValueError(
            "Each modeling row must represent exactly one unique trajectory."
        )
