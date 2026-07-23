"""Interpretation and decision-support helpers for SinD Stage 7."""

from __future__ import annotations

import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


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

PROFILE_NAMES = {
    1: "Smooth and Steady",
    2: "Stop-and-Go",
    3: "Dynamic Speed Adjustment",
    4: "Acceleration-Intensive",
}


def clean_tianjin_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata labels without changing trajectory-level features."""
    output = data.copy()
    output["CrossType_clean"] = (
        output["CrossType"]
        .astype("string")
        .str.strip()
    )
    output["Signal_Violation_clean"] = (
        output["Signal_Violation_Behavior"]
        .astype("string")
        .str.strip()
        .replace(
            {
                "No violation of traffic lights": "No violation",
                "yellow-light running": "Yellow-light running",
                "red-light running": "Red-light running",
            }
        )
    )
    output["Any_Violation"] = output[
        "Signal_Violation_clean"
    ].ne("No violation")
    return output


def robust_profile_scores(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Express profile medians relative to the global median and IQR."""
    if feature_columns is None:
        feature_columns = CORE_FEATURES

    global_median = data[feature_columns].median()
    global_iqr = (
        data[feature_columns].quantile(0.75)
        - data[feature_columns].quantile(0.25)
    ).replace(0, 1.0)

    medians = data.groupby(
        ["profile_id", "profile_name"],
        observed=True,
    )[feature_columns].median()

    scores = (medians - global_median) / global_iqr
    scores = scores.reset_index()
    return scores


def chi_square_effect(
    table: pd.DataFrame,
) -> dict[str, float | int]:
    """Calculate chi-square association and bias-corrected Cramer's V."""
    observed = table.to_numpy(dtype=float)
    chi2, p_value, degrees_of_freedom, _ = chi2_contingency(
        observed,
        correction=False,
    )

    n = observed.sum()
    rows, columns = observed.shape
    phi2 = chi2 / n

    # Bias correction from Bergsma (2013).
    phi2_corrected = max(
        0.0,
        phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1),
    )
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    columns_corrected = (
        columns - ((columns - 1) ** 2) / max(n - 1, 1)
    )
    denominator = min(
        rows_corrected - 1,
        columns_corrected - 1,
    )
    cramers_v = (
        math.sqrt(phi2_corrected / denominator)
        if denominator > 0
        else 0.0
    )

    return {
        "chi_square": float(chi2),
        "degrees_of_freedom": int(degrees_of_freedom),
        "p_value": float(p_value),
        "cramers_v_bias_corrected": float(cramers_v),
        "n": int(n),
        "rows": int(rows),
        "columns": int(columns),
    }


def standardized_residuals(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Return Pearson standardized residuals for a contingency table."""
    observed = table.to_numpy(dtype=float)
    _, _, _, expected = chi2_contingency(
        observed,
        correction=False,
    )
    residuals = (observed - expected) / np.sqrt(expected)
    return pd.DataFrame(
        residuals,
        index=table.index,
        columns=table.columns,
    )


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan

    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (
        proportion + z**2 / (2 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def find_zip_member(
    archive: zipfile.ZipFile,
    basename: str,
) -> str:
    """Find a file inside a recording archive by basename."""
    for member in archive.namelist():
        if Path(member).name == basename:
            return member
    raise FileNotFoundError(
        f"{basename} not found in {archive.filename}"
    )


def read_vehicle_track(
    archive_path: str | Path,
    track_id: int,
) -> pd.DataFrame:
    """Read one vehicle trajectory from a ZIP without extracting the archive."""
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        member = find_zip_member(
            archive,
            "Veh_smoothed_tracks.csv",
        )
        chunks: list[pd.DataFrame] = []
        with archive.open(member) as stream:
            for chunk in pd.read_csv(
                stream,
                chunksize=250_000,
                low_memory=False,
            ):
                selected = chunk.loc[
                    chunk["track_id"].eq(track_id)
                ].copy()
                if not selected.empty:
                    chunks.append(selected)

    if not chunks:
        raise ValueError(
            f"Track {track_id} not found in {archive_path.name}"
        )

    track = pd.concat(chunks, ignore_index=True)
    track = track.sort_values(
        ["timestamp_ms", "frame_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    track["time_s"] = (
        track["timestamp_ms"] - track["timestamp_ms"].iloc[0]
    ) / 1000.0
    track["speed_mps"] = np.hypot(track["vx"], track["vy"])
    track["dt_s"] = track["time_s"].diff()
    track["jerk_mps3"] = track["a_lon"].diff() / track["dt_s"]
    track.loc[track["dt_s"].le(0), "jerk_mps3"] = np.nan
    return track


def attention_dimension_table(
    profile_medians: pd.DataFrame,
    violation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build transparent operational-attention dimensions.

    These dimensions are descriptive priorities, not crash-risk probabilities.
    """
    profile = profile_medians.set_index(
        ["profile_id", "profile_name"]
    ).copy()

    dynamic_features = [
        "max_acceleration_mps2",
        "max_deceleration_mps2",
        "acceleration_std_mps2",
        "mean_abs_jerk_mps3",
    ]
    signal_features = [
        "observed_stop_transition_count",
        "stopped_time_ratio",
    ]
    speed_adjustment_features = [
        "speed_std_mps",
        "max_speed_mps",
        "max_deceleration_mps2",
    ]

    percentile = profile.rank(
        axis=0,
        method="average",
        pct=True,
    ) * 100

    output = pd.DataFrame(index=profile.index)
    output["dynamic_maneuver_priority"] = percentile[
        dynamic_features
    ].mean(axis=1)
    output["signal_queue_priority"] = percentile[
        signal_features
    ].mean(axis=1)
    output["speed_adjustment_priority"] = percentile[
        speed_adjustment_features
    ].mean(axis=1)

    violation = violation_summary.set_index(
        ["profile_id", "profile_name"]
    )
    output["tianjin_violation_rate_pct"] = (
        violation["violation_rate"] * 100
    )
    output["tianjin_violation_enrichment"] = violation[
        "relative_to_overall"
    ]

    output = output.reset_index()

    def tier(row: pd.Series) -> str:
        if (
            row["dynamic_maneuver_priority"] >= 75
            or row["tianjin_violation_enrichment"] >= 1.25
        ):
            return "High review priority"
        if (
            row["signal_queue_priority"] >= 75
            or row["speed_adjustment_priority"] >= 70
        ):
            return "Operational priority"
        return "Baseline reference"

    output["attention_tier"] = output.apply(tier, axis=1)

    focus = {
        1: (
            "Use as the baseline flow profile; monitor prevalence "
            "and large context shifts rather than intervene by default."
        ),
        2: (
            "Review signal timing, queue formation, stop frequency, "
            "and possible spillback conditions."
        ),
        3: (
            "Review approach-speed consistency, braking zones, "
            "and locations where repeated speed adjustment occurs."
        ),
        4: (
            "Prioritize trajectory-quality verification and review "
            "high-acceleration/high-jerk maneuvers."
        ),
    }
    output["recommended_operational_focus"] = output[
        "profile_id"
    ].map(focus)
    output[
        "interpretation_warning"
    ] = (
        "Descriptive attention priority only; not a crash probability "
        "or causal safety-risk estimate."
    )
    return output
