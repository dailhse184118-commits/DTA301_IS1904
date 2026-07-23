"""Run the SinD multi-recording trajectory-feature pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    discover_recording_archives,
    read_vehicle_data,
    read_vehicle_metadata,
)
from src.feature_engineering import (  # noqa: E402
    CORE_FEATURE_COLUMNS,
    attach_optional_tianjin_metadata,
    engineer_recording_features,
)
from src.quality_checks import (  # noqa: E402
    QualityConfig,
    validate_feature_table,
)


def configure_logging(log_path: Path) -> None:
    """Configure console and file logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one trajectory-level passenger-car feature table from "
            "all SinD recording ZIP archives."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/data"),
        help="Directory that contains the recording ZIP archives.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project directory for processed outputs and logs.",
    )
    parser.add_argument(
        "--minimum-duration-s",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--stationary-max-distance-m",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--stationary-max-speed-mps",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--stop-speed-threshold-mps",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    processed_dir = output_root / "data" / "processed"
    tables_dir = output_root / "outputs" / "tables"
    logs_dir = output_root / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "multi_recording_feature_pipeline.log"
    configure_logging(log_path)

    quality_config = QualityConfig(
        minimum_duration_s=args.minimum_duration_s,
        stationary_max_travel_distance_m=(
            args.stationary_max_distance_m
        ),
        stationary_max_speed_mps=args.stationary_max_speed_mps,
    )

    recordings = discover_recording_archives(args.raw_root)
    if not recordings:
        raise FileNotFoundError(
            f"No SinD recording ZIP archives found under {args.raw_root}"
        )

    logging.info("Discovered %d recording archives.", len(recordings))

    all_summaries: list[pd.DataFrame] = []
    processing_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    started_at = time.perf_counter()

    for index, recording in enumerate(recordings, start=1):
        recording_started = time.perf_counter()
        logging.info(
            "[%d/%d] Processing %s / %s",
            index,
            len(recordings),
            recording.city,
            recording.recording_id,
        )

        try:
            vehicle_data = read_vehicle_data(recording)
            raw_vehicle_rows = len(vehicle_data)
            raw_vehicle_trajectories = int(
                vehicle_data["track_id"].nunique()
            )
            passenger_car_rows = int(
                vehicle_data["agent_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("car")
                .sum()
            )

            summary = engineer_recording_features(
                vehicle_data=vehicle_data,
                recording=recording,
                quality_config=quality_config,
                stop_speed_threshold_mps=(
                    args.stop_speed_threshold_mps
                ),
            )
            metadata = read_vehicle_metadata(recording)
            summary = attach_optional_tianjin_metadata(
                summary,
                metadata,
            )

            all_summaries.append(summary)

            short_count = int(
                summary["flag_duration_below_minimum"].sum()
            )
            stationary_count = int(
                summary["flag_stationary_full_trajectory"].sum()
            )
            overlap_count = int(
                (
                    summary["flag_duration_below_minimum"]
                    & summary["flag_stationary_full_trajectory"]
                ).sum()
            )
            excluded_count = int(
                (~summary["is_modeling_eligible"]).sum()
            )
            valid_count = int(
                summary["is_modeling_eligible"].sum()
            )

            processing_rows.append(
                {
                    "city": recording.city,
                    "recording_id": recording.recording_id,
                    "source_archive": recording.archive_path.name,
                    "raw_vehicle_rows": raw_vehicle_rows,
                    "raw_vehicle_trajectories": (
                        raw_vehicle_trajectories
                    ),
                    "passenger_car_rows": passenger_car_rows,
                    "passenger_car_trajectories": len(summary),
                    "flagged_duration_below_minimum": short_count,
                    "flagged_stationary_full_trajectory": (
                        stationary_count
                    ),
                    "flag_overlap_short_and_stationary": (
                        overlap_count
                    ),
                    "excluded_unique_trajectories": excluded_count,
                    "modeling_eligible_trajectories": valid_count,
                    "tianjin_vehicle_metadata_available": (
                        metadata is not None
                    ),
                    "elapsed_seconds": round(
                        time.perf_counter() - recording_started,
                        3,
                    ),
                }
            )
        except Exception as exc:
            logging.exception(
                "Failed to process %s",
                recording.archive_path.name,
            )
            failures.append(
                {
                    "city": recording.city,
                    "recording_id": recording.recording_id,
                    "source_archive": recording.archive_path.name,
                    "error": repr(exc),
                }
            )

    if failures:
        failure_table = pd.DataFrame(failures)
        failure_path = tables_dir / "recording_failures.csv"
        failure_table.to_csv(failure_path, index=False)
        raise RuntimeError(
            f"{len(failures)} recordings failed. See {failure_path}"
        )

    trajectory_summary = pd.concat(
        all_summaries,
        ignore_index=True,
        sort=False,
    )

    if trajectory_summary["trajectory_uid"].duplicated().any():
        raise ValueError(
            "trajectory_uid is not unique across the full dataset."
        )

    exclusion_log = trajectory_summary.loc[
        ~trajectory_summary["is_modeling_eligible"],
        [
            "trajectory_uid",
            "city",
            "recording_id",
            "track_id",
            "source_archive",
            "trajectory_duration_s",
            "frame_count",
            "travel_distance_m",
            "max_speed_mps",
            "flag_duration_below_minimum",
            "flag_stationary_full_trajectory",
            "exclusion_reason",
        ],
    ].copy()

    model_columns = [
        "trajectory_uid",
        "city",
        "recording_id",
        "track_id",
        "source_archive",
        "trajectory_duration_s",
        "frame_count",
        "travel_distance_m",
        "position_span_m",
        *CORE_FEATURE_COLUMNS,
        "class",
        "CrossType",
        "Signal_Violation_Behavior",
    ]

    model_features = trajectory_summary.loc[
        trajectory_summary["is_modeling_eligible"],
        model_columns,
    ].copy()

    validate_feature_table(
        model_features,
        CORE_FEATURE_COLUMNS,
    )

    processing_summary = pd.DataFrame(processing_rows)
    city_summary = (
        processing_summary.groupby("city", as_index=False)
        .agg(
            recordings=("recording_id", "count"),
            raw_vehicle_rows=("raw_vehicle_rows", "sum"),
            raw_vehicle_trajectories=(
                "raw_vehicle_trajectories",
                "sum",
            ),
            passenger_car_rows=("passenger_car_rows", "sum"),
            passenger_car_trajectories=(
                "passenger_car_trajectories",
                "sum",
            ),
            flagged_duration_below_minimum=(
                "flagged_duration_below_minimum",
                "sum",
            ),
            flagged_stationary_full_trajectory=(
                "flagged_stationary_full_trajectory",
                "sum",
            ),
            flag_overlap_short_and_stationary=(
                "flag_overlap_short_and_stationary",
                "sum",
            ),
            excluded_unique_trajectories=(
                "excluded_unique_trajectories",
                "sum",
            ),
            modeling_eligible_trajectories=(
                "modeling_eligible_trajectories",
                "sum",
            ),
        )
    )

    trajectory_summary_path = (
        processed_dir / "passenger_car_trajectory_summary.csv"
    )
    exclusion_log_path = (
        processed_dir / "trajectory_exclusion_log.csv"
    )
    feature_path = (
        processed_dir / "sind_full_core_behavior_features.csv"
    )
    processing_summary_path = (
        tables_dir / "recording_processing_summary.csv"
    )
    city_summary_path = (
        tables_dir / "city_processing_summary.csv"
    )

    trajectory_summary.to_csv(
        trajectory_summary_path,
        index=False,
    )
    exclusion_log.to_csv(exclusion_log_path, index=False)
    model_features.to_csv(feature_path, index=False)
    processing_summary.to_csv(
        processing_summary_path,
        index=False,
    )
    city_summary.to_csv(city_summary_path, index=False)

    total_elapsed = time.perf_counter() - started_at
    logging.info(
        "Pipeline completed in %.2f seconds.",
        total_elapsed,
    )
    logging.info(
        "Passenger-car trajectories before filtering: %d",
        len(trajectory_summary),
    )
    logging.info(
        "Unique excluded trajectories: %d",
        len(exclusion_log),
    )
    logging.info(
        "Modeling-eligible trajectories: %d",
        len(model_features),
    )
    logging.info("Saved %s", feature_path)

    print("\n=== FULL DATASET FEATURE PIPELINE COMPLETE ===")
    print(city_summary.to_string(index=False))
    print(
        f"\nTotal passenger-car trajectories: "
        f"{len(trajectory_summary):,}"
    )
    print(f"Unique excluded trajectories: {len(exclusion_log):,}")
    print(
        f"Modeling-eligible trajectories: "
        f"{len(model_features):,}"
    )
    print(f"Feature table: {feature_path}")
    print(f"Exclusion log: {exclusion_log_path}")
    print(f"Pipeline log: {log_path}")


if __name__ == "__main__":
    main()
