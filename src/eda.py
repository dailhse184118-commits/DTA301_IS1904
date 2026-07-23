"""Reusable exploratory-data-analysis helpers for the SinD full study."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_iqr_outlier_summary(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return IQR-based diagnostic counts without deleting observations."""
    rows = []
    for feature in feature_columns:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (data[feature] < lower_bound) | (data[feature] > upper_bound)
        rows.append({
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(mask.sum()),
            "outlier_percentage": float(mask.mean() * 100),
        })
    return pd.DataFrame(rows)


def validate_eda_input(data: pd.DataFrame, feature_columns: list[str], expected_recordings: int = 56) -> None:
    """Fail loudly if the EDA input is structurally invalid."""
    if data["trajectory_uid"].duplicated().any():
        raise ValueError("Duplicate trajectory_uid values detected.")
    missing = int(data[feature_columns].isna().sum().sum())
    if missing:
        raise ValueError(f"{missing} missing cells found in core features.")
    infinite = int(np.isinf(data[feature_columns].to_numpy()).sum())
    if infinite:
        raise ValueError(f"{infinite} infinite values found in core features.")
    if data["recording_id"].nunique() != expected_recordings:
        raise ValueError("The expected recording count was not preserved.")
