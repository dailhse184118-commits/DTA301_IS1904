"""Audit SinD recording ZIP archives before feature engineering."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


def infer_city(filename: str) -> str | None:
    name = filename.lower()
    if name.startswith("changchun_"):
        return "Changchun"
    if "_nr_" in name:
        return "Chongqing"
    if name.startswith("xian_"):
        return "Xi'an"
    if re.fullmatch(r"\d+_\d+_\d+\.zip", name):
        return "Tianjin"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def find_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    return next(
        (name for name in archive.namelist() if Path(name).name == basename),
        None,
    )


def main() -> None:
    args = parse_args()
    tables = args.output_root / "outputs/tables"
    tables.mkdir(parents=True, exist_ok=True)

    archives = [
        path for path in sorted(args.raw_root.rglob("*.zip"))
        if infer_city(path.name) is not None
    ]
    if not archives:
        raise FileNotFoundError("No SinD recording ZIP archives were found.")

    rows: list[dict[str, object]] = []
    for path in archives:
        city = infer_city(path.name)
        recording_id = path.stem.replace("(1)", "")

        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            vehicle_member = find_member(archive, "Veh_smoothed_tracks.csv")
            pedestrian_member = find_member(archive, "Ped_smoothed_tracks.csv")
            traffic_members = [
                name for name in archive.namelist()
                if "traffic" in Path(name).name.lower()
            ]
            metadata_member = find_member(archive, "Veh_tracks_meta.csv")

            if vehicle_member is None:
                raise FileNotFoundError(
                    f"Veh_smoothed_tracks.csv missing in {path.name}"
                )

            vehicle_rows = 0
            vehicle_ids: set[object] = set()
            car_ids: set[object] = set()
            missing_cells = 0

            with archive.open(vehicle_member) as stream:
                for chunk in pd.read_csv(
                    stream,
                    usecols=lambda column: column in {
                        "track_id",
                        "frame_id",
                        "timestamp_ms",
                        "agent_type",
                    },
                    chunksize=250_000,
                    low_memory=False,
                ):
                    vehicle_rows += len(chunk)
                    missing_cells += int(chunk.isna().sum().sum())
                    vehicle_ids.update(chunk["track_id"].dropna().unique())
                    car_mask = (
                        chunk["agent_type"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .eq("car")
                    )
                    car_ids.update(
                        chunk.loc[car_mask, "track_id"].dropna().unique()
                    )

            rows.append(
                {
                    "city": city,
                    "recording_id": recording_id,
                    "archive": path.name,
                    "zip_integrity_ok": bad_member is None,
                    "vehicle_rows": vehicle_rows,
                    "vehicle_trajectories": len(vehicle_ids),
                    "passenger_car_trajectories": len(car_ids),
                    "selected_identifier_missing_cells": missing_cells,
                    "has_pedestrian_file": pedestrian_member is not None,
                    "traffic_light_files": len(traffic_members),
                    "has_vehicle_metadata": metadata_member is not None,
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(tables / "initial_recording_inventory.csv", index=False)

    city_summary = (
        audit.groupby("city", as_index=False)
        .agg(
            recordings=("recording_id", "count"),
            vehicle_rows=("vehicle_rows", "sum"),
            vehicle_trajectories=("vehicle_trajectories", "sum"),
            passenger_car_trajectories=("passenger_car_trajectories", "sum"),
            traffic_light_files=("traffic_light_files", "sum"),
        )
    )
    city_summary.to_csv(tables / "initial_city_inventory.csv", index=False)

    print(city_summary.to_string(index=False))
    print(f"\nAudited {len(audit)} recordings.")


if __name__ == "__main__":
    main()
