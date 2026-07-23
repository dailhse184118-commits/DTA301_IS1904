"""Run Stage 6 end-to-end cross-city and cross-recording validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import (  # noqa: E402
    CORE_FEATURES,
    PROFILE_IDS,
    ValidationConfig,
    evaluate_held_out_labels,
    fit_end_to_end_split,
    global_profile_reference,
    semantic_profile_consistency,
)


PROFILE_NAMES = {
    1: "Smooth and Steady",
    2: "Stop-and-Go",
    3: "Dynamic Speed Adjustment",
    4: "Acceleration-Intensive",
}


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


def merge_stage_data() -> pd.DataFrame:
    """Merge Stage 5 reference profiles with original trajectory features."""
    processed = PROJECT_ROOT / "data" / "processed"

    assignments = pd.read_csv(
        processed / "stage5_final_cluster_assignments.csv"
    )
    features = pd.read_csv(
        processed / "sind_full_core_behavior_features.csv"
    )

    keep_assignment = [
        "trajectory_uid",
        "profile_id",
        "profile_name",
    ]
    merged = features.merge(
        assignments[keep_assignment],
        on="trajectory_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(assignments):
        raise ValueError(
            "Stage 5 assignments and feature rows do not match one-to-one."
        )
    if merged[CORE_FEATURES].isna().any().any():
        raise ValueError("Missing core features in Stage 6 input.")
    if merged["profile_id"].isna().any():
        raise ValueError("Missing Stage 5 reference profile labels.")

    merged["profile_id"] = merged["profile_id"].astype(int)
    return merged


def confusion_long(
    reference: np.ndarray,
    predicted: np.ndarray,
    split_type: str,
    split_name: str,
) -> pd.DataFrame:
    """Return a long-form profile confusion table."""
    table = pd.crosstab(
        pd.Series(reference, name="reference_profile_id"),
        pd.Series(predicted, name="predicted_profile_id"),
        dropna=False,
    )
    table = table.reindex(
        index=PROFILE_IDS,
        columns=PROFILE_IDS,
        fill_value=0,
    )
    long = (
        table.stack()
        .rename("trajectory_count")
        .reset_index()
    )
    long.insert(0, "split_type", split_type)
    long.insert(1, "split_name", split_name)
    return long


def main() -> None:
    tables = PROJECT_ROOT / "outputs" / "tables"
    figures = PROJECT_ROOT / "outputs" / "figures"
    logs = PROJECT_ROOT / "logs"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    configure_logging(logs / "stage6_cross_context_validation.log")

    data = merge_stage_data()
    global_medians, global_iqr = global_profile_reference(data)

    city_config = ValidationConfig(
        n_clusters=4,
        n_init=50,
        random_state=42,
    )
    recording_config = ValidationConfig(
        n_clusters=4,
        n_init=20,
        random_state=42,
    )

    started = time.perf_counter()

    # ---------------------------------------------------------
    # 1. Leave-one-city-out end-to-end validation
    # ---------------------------------------------------------
    city_rows: list[dict[str, object]] = []
    city_semantic_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    city_prediction_rows: list[pd.DataFrame] = []

    for held_city in sorted(data["city"].unique()):
        logging.info("LOCO validation: holding out %s", held_city)
        train = data.loc[data["city"].ne(held_city)].reset_index(drop=True)
        test = data.loc[data["city"].eq(held_city)].reset_index(drop=True)

        result = fit_end_to_end_split(
            train_data=train,
            test_data=test,
            config=city_config,
        )
        predicted = result["test_labels"]
        preprocessor = result["preprocessor"]

        metrics = evaluate_held_out_labels(
            test["profile_id"].to_numpy(),
            predicted,
        )
        semantics = semantic_profile_consistency(
            data=test,
            predicted_labels=predicted,
            global_profile_medians=global_medians,
            global_feature_iqr=global_iqr,
        )
        semantics.insert(0, "held_out_city", held_city)
        semantics["profile_name"] = semantics["profile_id"].map(
            PROFILE_NAMES
        )
        city_semantic_rows.append(semantics)

        city_rows.append(
            {
                "held_out_city": held_city,
                "n_train": len(train),
                "n_test": len(test),
                "pca_components": preprocessor.n_components,
                "pca_retained_variance": preprocessor.retained_variance,
                **metrics,
                "semantic_mae_mean": float(
                    semantics["median_normalized_mae"].mean()
                ),
                "semantic_mae_max_profile": float(
                    semantics["median_normalized_mae"].max()
                ),
                "semantic_profile_correlation_mean": float(
                    semantics[
                        "profile_vector_correlation"
                    ].mean()
                ),
                "cluster_mapping": json.dumps(
                    result["cluster_mapping"],
                    sort_keys=True,
                ),
            }
        )

        confusion_rows.append(
            confusion_long(
                test["profile_id"].to_numpy(),
                predicted,
                split_type="leave_one_city_out",
                split_name=held_city,
            )
        )

        predictions = test[
            [
                "trajectory_uid",
                "city",
                "recording_id",
                "track_id",
                "profile_id",
                "profile_name",
            ]
        ].copy()
        predictions["predicted_profile_id"] = predicted
        predictions["predicted_profile_name"] = predictions[
            "predicted_profile_id"
        ].map(PROFILE_NAMES)
        predictions["held_out_city"] = held_city
        city_prediction_rows.append(predictions)

    city_results = pd.DataFrame(city_rows)
    city_results.to_csv(
        tables / "stage6_leave_one_city_out.csv",
        index=False,
    )
    pd.concat(
        city_semantic_rows,
        ignore_index=True,
    ).to_csv(
        tables / "stage6_leave_one_city_out_semantics.csv",
        index=False,
    )
    pd.concat(
        city_prediction_rows,
        ignore_index=True,
    ).to_csv(
        tables / "stage6_leave_one_city_out_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Leave-one-recording-out end-to-end validation
    # ---------------------------------------------------------
    recording_rows: list[dict[str, object]] = []
    recording_semantic_rows: list[pd.DataFrame] = []
    recording_prediction_rows: list[pd.DataFrame] = []

    grouped_recordings = list(
        data.groupby(["city", "recording_id"], sort=True)
    )

    for index, ((city, recording_id), test_source) in enumerate(
        grouped_recordings,
        start=1,
    ):
        logging.info(
            "LORO validation %d/%d: %s / %s",
            index,
            len(grouped_recordings),
            city,
            recording_id,
        )
        test = test_source.reset_index(drop=True)
        train = data.loc[
            ~(
                data["city"].eq(city)
                & data["recording_id"].eq(recording_id)
            )
        ].reset_index(drop=True)

        result = fit_end_to_end_split(
            train_data=train,
            test_data=test,
            config=recording_config,
        )
        predicted = result["test_labels"]
        preprocessor = result["preprocessor"]

        metrics = evaluate_held_out_labels(
            test["profile_id"].to_numpy(),
            predicted,
        )
        semantics = semantic_profile_consistency(
            data=test,
            predicted_labels=predicted,
            global_profile_medians=global_medians,
            global_feature_iqr=global_iqr,
        )
        semantics.insert(0, "city", city)
        semantics.insert(1, "recording_id", recording_id)
        semantics["profile_name"] = semantics["profile_id"].map(
            PROFILE_NAMES
        )
        recording_semantic_rows.append(semantics)

        recording_rows.append(
            {
                "city": city,
                "recording_id": recording_id,
                "n_train": len(train),
                "n_test": len(test),
                "pca_components": preprocessor.n_components,
                "pca_retained_variance": preprocessor.retained_variance,
                **metrics,
                "semantic_mae_mean": float(
                    semantics["median_normalized_mae"].mean()
                ),
                "semantic_profile_correlation_mean": float(
                    semantics[
                        "profile_vector_correlation"
                    ].mean()
                ),
            }
        )

        predictions = test[
            [
                "trajectory_uid",
                "city",
                "recording_id",
                "track_id",
                "profile_id",
                "profile_name",
            ]
        ].copy()
        predictions["predicted_profile_id"] = predicted
        predictions["predicted_profile_name"] = predictions[
            "predicted_profile_id"
        ].map(PROFILE_NAMES)
        recording_prediction_rows.append(predictions)

    recording_results = pd.DataFrame(recording_rows)
    recording_results.to_csv(
        tables / "stage6_leave_one_recording_out.csv",
        index=False,
    )
    pd.concat(
        recording_semantic_rows,
        ignore_index=True,
    ).to_csv(
        tables / "stage6_leave_one_recording_out_semantics.csv",
        index=False,
    )
    pd.concat(
        recording_prediction_rows,
        ignore_index=True,
    ).to_csv(
        tables / "stage6_leave_one_recording_out_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Fixed-final-label descriptive context consistency
    # ---------------------------------------------------------
    city_profile_counts = (
        data.groupby(["city", "profile_id", "profile_name"])
        .size()
        .reset_index(name="trajectory_count")
    )
    city_totals = city_profile_counts.groupby("city")[
        "trajectory_count"
    ].transform("sum")
    city_profile_counts["city_percentage"] = (
        city_profile_counts["trajectory_count"]
        / city_totals
        * 100
    )
    city_profile_counts.to_csv(
        tables / "stage6_final_profile_distribution_by_city.csv",
        index=False,
    )

    recording_profile_counts = (
        data.groupby(
            ["city", "recording_id", "profile_id", "profile_name"]
        )
        .size()
        .reset_index(name="trajectory_count")
    )
    recording_totals = recording_profile_counts.groupby(
        ["city", "recording_id"]
    )["trajectory_count"].transform("sum")
    recording_profile_counts["recording_percentage"] = (
        recording_profile_counts["trajectory_count"]
        / recording_totals
        * 100
    )
    recording_profile_counts.to_csv(
        tables / "stage6_final_profile_distribution_by_recording.csv",
        index=False,
    )

    fixed_city_semantics: list[pd.DataFrame] = []
    for city, subset in data.groupby("city", sort=True):
        semantics = semantic_profile_consistency(
            data=subset.reset_index(drop=True),
            predicted_labels=subset["profile_id"].to_numpy(),
            global_profile_medians=global_medians,
            global_feature_iqr=global_iqr,
        )
        semantics.insert(0, "city", city)
        semantics["profile_name"] = semantics["profile_id"].map(
            PROFILE_NAMES
        )
        fixed_city_semantics.append(semantics)

    fixed_city_semantics_table = pd.concat(
        fixed_city_semantics,
        ignore_index=True,
    )
    fixed_city_semantics_table.to_csv(
        tables / "stage6_fixed_profile_semantics_by_city.csv",
        index=False,
    )

    recording_coverage = (
        data.groupby(["city", "recording_id"])
        .agg(
            trajectory_count=("trajectory_uid", "count"),
            profiles_present=("profile_id", "nunique"),
            smallest_profile_count=(
                "profile_id",
                lambda values: values.value_counts().min(),
            ),
            largest_profile_percentage=(
                "profile_id",
                lambda values: (
                    values.value_counts().max()
                    / len(values)
                    * 100
                ),
            ),
        )
        .reset_index()
    )
    recording_coverage["all_four_profiles_present"] = (
        recording_coverage["profiles_present"].eq(4)
    )
    recording_coverage.to_csv(
        tables / "stage6_recording_profile_coverage.csv",
        index=False,
    )

    # Save confusion tables.
    pd.concat(
        confusion_rows,
        ignore_index=True,
    ).to_csv(
        tables / "stage6_leave_one_city_out_confusion.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Validation summary and decision
    # ---------------------------------------------------------
    city_summary = {
        "validation_level": "leave_one_city_out",
        "splits": int(len(city_results)),
        "ari_mean": float(city_results["ari"].mean()),
        "ari_min": float(city_results["ari"].min()),
        "aligned_accuracy_mean": float(
            city_results["aligned_accuracy"].mean()
        ),
        "aligned_accuracy_min": float(
            city_results["aligned_accuracy"].min()
        ),
        "balanced_accuracy_mean": float(
            city_results["balanced_accuracy"].mean()
        ),
        "semantic_mae_mean": float(
            city_results["semantic_mae_mean"].mean()
        ),
        "semantic_profile_correlation_mean": float(
            city_results[
                "semantic_profile_correlation_mean"
            ].mean()
        ),
        "all_splits_recovered_reference_profiles": bool(
            city_results[
                "all_reference_profiles_recovered"
            ].all()
        ),
    }

    recording_summary = {
        "validation_level": "leave_one_recording_out",
        "splits": int(len(recording_results)),
        "ari_mean": float(recording_results["ari"].mean()),
        "ari_median": float(recording_results["ari"].median()),
        "ari_min": float(recording_results["ari"].min()),
        "ari_10th_percentile": float(
            recording_results["ari"].quantile(0.10)
        ),
        "aligned_accuracy_mean": float(
            recording_results["aligned_accuracy"].mean()
        ),
        "aligned_accuracy_median": float(
            recording_results["aligned_accuracy"].median()
        ),
        "balanced_accuracy_mean": float(
            recording_results["balanced_accuracy"].mean()
        ),
        "semantic_mae_mean": float(
            recording_results["semantic_mae_mean"].mean()
        ),
        "semantic_profile_correlation_mean": float(
            recording_results[
                "semantic_profile_correlation_mean"
            ].mean()
        ),
        "recordings_with_ari_below_0_60": int(
            recording_results["ari"].lt(0.60).sum()
        ),
        "recordings_with_ari_at_least_0_80": int(
            recording_results["ari"].ge(0.80).sum()
        ),
    }

    validation_summary = pd.DataFrame(
        [city_summary, recording_summary]
    )
    validation_summary.to_csv(
        tables / "stage6_validation_summary.csv",
        index=False,
    )

    # Transparent decision rules.
    city_pass = (
        city_summary["ari_mean"] >= 0.70
        and city_summary["ari_min"] >= 0.50
        and city_summary["aligned_accuracy_mean"] >= 0.75
        and city_summary[
            "semantic_profile_correlation_mean"
        ] >= 0.75
    )
    recording_pass = (
        recording_summary["ari_median"] >= 0.80
        and recording_summary["ari_10th_percentile"] >= 0.60
        and recording_summary["aligned_accuracy_mean"] >= 0.80
        and recording_summary[
            "semantic_profile_correlation_mean"
        ] >= 0.75
    )

    if city_pass and recording_pass:
        decision = "Validated with context sensitivity"
        interpretation = (
            "The four-profile solution is reproducible across recordings "
            "and remains meaningfully transferable across cities. "
            "City-specific prevalence and feature shifts must still be "
            "reported rather than treated as identical contexts."
        )
    elif recording_pass:
        decision = "Recording-robust but city-sensitive"
        interpretation = (
            "The four-profile solution is stable across recordings, but "
            "cross-city transfer is weaker. The model should be presented "
            "as a multi-context structure with explicit city effects."
        )
    else:
        decision = "Requires model revision"
        interpretation = (
            "The four-profile solution did not pass the predefined "
            "recording and city validation gates."
        )

    decision_table = pd.DataFrame(
        [
            {
                "selected_model": "KMeans k=4",
                "preprocessing": "C_winsor_robust",
                "validation_decision": decision,
                "city_gate_passed": city_pass,
                "recording_gate_passed": recording_pass,
                "interpretation": interpretation,
            }
        ]
    )
    decision_table.to_csv(
        tables / "stage6_validation_decision.csv",
        index=False,
    )

    config_output = {
        "stage": 6,
        "validation_design": {
            "leave_one_city_out_splits": int(len(city_results)),
            "leave_one_recording_out_splits": int(
                len(recording_results)
            ),
            "train_only_preprocessing": True,
            "winsorization": "1st-99th percentile on eight continuous features",
            "scaler": "RobustScaler",
            "pca": "fit on training rows only, retain >=90% variance",
            "model": "KMeans k=4",
            "cluster_alignment": "Hungarian matching on training rows",
        },
        "decision": decision,
        "city_gate_passed": city_pass,
        "recording_gate_passed": recording_pass,
    }
    with open(
        tables / "stage6_validation_config.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(config_output, stream, indent=2)

    elapsed = time.perf_counter() - started
    logging.info("Stage 6 completed in %.2f seconds.", elapsed)

    print("\n=== STAGE 6 CROSS-CONTEXT VALIDATION COMPLETE ===")
    print(f"Trajectories: {len(data):,}")
    print(f"LOCO splits: {len(city_results)}")
    print(f"LORO splits: {len(recording_results)}")
    print("\nLeave-one-city-out:")
    print(city_results[
        [
            "held_out_city",
            "n_test",
            "ari",
            "aligned_accuracy",
            "balanced_accuracy",
            "semantic_mae_mean",
            "semantic_profile_correlation_mean",
        ]
    ].to_string(index=False))
    print("\nLeave-one-recording-out summary:")
    print(pd.DataFrame([recording_summary]).to_string(index=False))
    print(f"\nDecision: {decision}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
