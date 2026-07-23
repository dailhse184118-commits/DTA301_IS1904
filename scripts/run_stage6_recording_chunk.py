"""Run a slice of Stage 6 leave-one-recording-out validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import (  # noqa: E402
    CORE_FEATURES,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    processed = PROJECT_ROOT / "data" / "processed"
    assignments = pd.read_csv(
        processed / "stage5_final_cluster_assignments.csv"
    )
    features = pd.read_csv(
        processed / "sind_full_core_behavior_features.csv"
    )
    return features.merge(
        assignments[
            ["trajectory_uid", "profile_id", "profile_name"]
        ],
        on="trajectory_uid",
        how="inner",
        validate="one_to_one",
    )


def main() -> None:
    args = parse_args()
    data = load_data()
    global_medians, global_iqr = global_profile_reference(data)
    config = ValidationConfig(
        n_clusters=4,
        n_init=20,
        random_state=42,
    )

    grouped = list(
        data.groupby(["city", "recording_id"], sort=True)
    )
    selected = grouped[args.start:args.end]

    result_rows: list[dict[str, object]] = []
    semantic_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for (city, recording_id), test_source in selected:
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
            config=config,
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
        semantic_rows.append(semantics)

        result_rows.append(
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
        prediction_rows.append(predictions)

    parts = PROJECT_ROOT / "outputs" / "tables" / "stage6_parts"
    suffix = f"{args.start:02d}_{args.end:02d}"

    pd.DataFrame(result_rows).to_csv(
        parts / f"recording_results_{suffix}.csv",
        index=False,
    )
    pd.concat(
        semantic_rows,
        ignore_index=True,
    ).to_csv(
        parts / f"recording_semantics_{suffix}.csv",
        index=False,
    )
    pd.concat(
        prediction_rows,
        ignore_index=True,
    ).to_csv(
        parts / f"recording_predictions_{suffix}.csv",
        index=False,
    )

    print(
        f"Completed recordings {args.start}:{args.end} "
        f"({len(result_rows)} splits)."
    )


if __name__ == "__main__":
    main()
