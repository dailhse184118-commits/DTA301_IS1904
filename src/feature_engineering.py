"""Vectorized trajectory-level feature engineering for SinD vehicles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import RecordingArchive
from .quality_checks import QualityConfig, add_quality_flags


CORE_FEATURE_COLUMNS = [
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


def _population_std_from_sums(
    value_sum: pd.Series,
    squared_sum: pd.Series,
    count: pd.Series,
) -> pd.Series:
    """Calculate population standard deviation without slow group apply."""
    mean = value_sum / count
    variance = (squared_sum / count) - mean.pow(2)
    return np.sqrt(variance.clip(lower=0))


def engineer_recording_features(
    vehicle_data: pd.DataFrame,
    recording: RecordingArchive,
    quality_config: QualityConfig,
    stop_speed_threshold_mps: float = 0.5,
) -> pd.DataFrame:
    """Convert all passenger-car frames in one recording into one row per car."""
    cars = vehicle_data.loc[
        vehicle_data["agent_type"].astype(str).str.strip().str.lower().eq("car")
    ].copy()

    if cars.empty:
        return pd.DataFrame()

    cars = cars.sort_values(
        ["track_id", "timestamp_ms", "frame_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    cars["speed_mps"] = np.hypot(cars["vx"], cars["vy"])
    cars["speed_squared"] = cars["speed_mps"].pow(2)
    cars["long_acc_squared"] = cars["a_lon"].pow(2)

    grouped = cars.groupby("track_id", sort=False, observed=True)
    cars["dt_s"] = grouped["timestamp_ms"].diff() / 1000.0
    cars["jerk_mps3"] = grouped["a_lon"].diff() / cars["dt_s"]
    cars.loc[cars["dt_s"].le(0), "jerk_mps3"] = np.nan
    cars["abs_jerk_mps3"] = cars["jerk_mps3"].abs()

    cars["step_distance_m"] = np.hypot(
        grouped["x"].diff(),
        grouped["y"].diff(),
    ).fillna(0.0)

    cars["is_stopped"] = cars["speed_mps"] < stop_speed_threshold_mps
    previous_stopped = grouped["is_stopped"].shift()
    cars["observed_stop_transition"] = (
        previous_stopped.eq(False) & cars["is_stopped"]
    ).astype("int8")

    summary = grouped.agg(
        first_timestamp_ms=("timestamp_ms", "min"),
        last_timestamp_ms=("timestamp_ms", "max"),
        frame_count=("frame_id", "size"),
        mean_speed_mps=("speed_mps", "mean"),
        max_speed_mps=("speed_mps", "max"),
        speed_sum=("speed_mps", "sum"),
        speed_squared_sum=("speed_squared", "sum"),
        mean_long_acc_mps2=("a_lon", "mean"),
        raw_max_long_acc_mps2=("a_lon", "max"),
        raw_min_long_acc_mps2=("a_lon", "min"),
        long_acc_sum=("a_lon", "sum"),
        long_acc_squared_sum=("long_acc_squared", "sum"),
        mean_abs_jerk_mps3=("abs_jerk_mps3", "mean"),
        observed_stop_transition_count=(
            "observed_stop_transition",
            "sum",
        ),
        stopped_time_ratio=("is_stopped", "mean"),
        travel_distance_m=("step_distance_m", "sum"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        y_min=("y", "min"),
        y_max=("y", "max"),
    ).reset_index()

    summary["trajectory_duration_s"] = (
        summary["last_timestamp_ms"] - summary["first_timestamp_ms"]
    ) / 1000.0

    summary["speed_std_mps"] = _population_std_from_sums(
        summary["speed_sum"],
        summary["speed_squared_sum"],
        summary["frame_count"],
    )
    summary["acceleration_std_mps2"] = _population_std_from_sums(
        summary["long_acc_sum"],
        summary["long_acc_squared_sum"],
        summary["frame_count"],
    )

    summary["max_acceleration_mps2"] = (
        summary["raw_max_long_acc_mps2"].clip(lower=0)
    )
    summary["max_deceleration_mps2"] = (
        -summary["raw_min_long_acc_mps2"].clip(upper=0)
    )
    summary["position_span_m"] = np.hypot(
        summary["x_max"] - summary["x_min"],
        summary["y_max"] - summary["y_min"],
    )

    # A one-frame trajectory has no defined jerk; zero means no observed change.
    summary["mean_abs_jerk_mps3"] = (
        summary["mean_abs_jerk_mps3"].fillna(0.0)
    )

    summary.insert(0, "city", recording.city)
    summary.insert(1, "recording_id", recording.recording_id)
    summary.insert(
        2,
        "trajectory_uid",
        (
            summary["city"]
            + "__"
            + summary["recording_id"]
            + "__"
            + summary["track_id"].astype(str)
        ),
    )
    summary["source_archive"] = recording.archive_path.name

    summary = add_quality_flags(summary, quality_config)

    drop_columns = [
        "speed_sum",
        "speed_squared_sum",
        "long_acc_sum",
        "long_acc_squared_sum",
        "raw_max_long_acc_mps2",
        "raw_min_long_acc_mps2",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    ]
    return summary.drop(columns=drop_columns)


def attach_optional_tianjin_metadata(
    features: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach optional post-cluster reference variables without using them as features."""
    output = features.copy()

    optional_columns = [
        "class",
        "CrossType",
        "Signal_Violation_Behavior",
    ]

    if metadata is None:
        for column in optional_columns:
            output[column] = pd.NA
        return output

    available = ["trackId"] + [
        column for column in optional_columns
        if column in metadata.columns
    ]
    selected = metadata[available].drop_duplicates("trackId").copy()
    selected = selected.rename(columns={"trackId": "track_id"})

    output = output.merge(
        selected,
        on="track_id",
        how="left",
        validate="one_to_one",
    )

    for column in optional_columns:
        if column not in output.columns:
            output[column] = pd.NA

    return output
