"""Dataset discovery and loading utilities for SinD recording archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import pandas as pd


VEHICLE_FILE = "Veh_smoothed_tracks.csv"
PEDESTRIAN_FILE = "Ped_smoothed_tracks.csv"
VEHICLE_META_FILE = "Veh_tracks_meta.csv"

REQUIRED_VEHICLE_COLUMNS = [
    "track_id",
    "frame_id",
    "timestamp_ms",
    "agent_type",
    "x",
    "y",
    "vx",
    "vy",
    "yaw_rad",
    "heading_rad",
    "length",
    "width",
    "ax",
    "ay",
    "v_lon",
    "v_lat",
    "a_lon",
    "a_lat",
]


@dataclass(frozen=True)
class RecordingArchive:
    """One SinD recording stored as a ZIP archive."""

    city: str
    recording_id: str
    archive_path: Path


def infer_city_and_recording(filename: str) -> tuple[str, str] | None:
    """Infer city and recording ID from the uploaded archive name."""
    stem = Path(filename).stem.replace("(1)", "")
    lower = stem.lower()

    if lower.startswith("changchun_"):
        return "Changchun", stem
    if "_nr_" in lower:
        return "Chongqing", stem
    if lower.startswith("xian_"):
        return "Xi'an", stem
    if re.fullmatch(r"\d+_\d+_\d+", lower):
        return "Tianjin", stem
    return None


def discover_recording_archives(raw_root: str | Path) -> list[RecordingArchive]:
    """Recursively discover the 56 recording ZIP archives.

    ZIP bundles belonging to reports or code submissions are ignored because
    their names do not match a SinD recording naming convention.
    """
    raw_root = Path(raw_root)
    recordings: list[RecordingArchive] = []

    for path in sorted(raw_root.rglob("*.zip")):
        inferred = infer_city_and_recording(path.name)
        if inferred is None:
            continue
        city, recording_id = inferred
        recordings.append(
            RecordingArchive(
                city=city,
                recording_id=recording_id,
                archive_path=path,
            )
        )

    return recordings


def _find_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    """Find a ZIP member by basename, regardless of its parent directory."""
    for member in archive.namelist():
        if Path(member).name == basename:
            return member
    return None


def normalize_vehicle_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and drop CSV export-index columns."""
    normalized = data.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    export_columns = [
        column for column in normalized.columns
        if column.startswith("Unnamed:")
    ]
    if export_columns:
        normalized = normalized.drop(columns=export_columns)

    missing = [
        column for column in REQUIRED_VEHICLE_COLUMNS
        if column not in normalized.columns
    ]
    if missing:
        raise ValueError(f"Missing required vehicle columns: {missing}")

    return normalized


def read_vehicle_data(recording: RecordingArchive) -> pd.DataFrame:
    """Read and normalize a recording's vehicle trajectory CSV."""
    with zipfile.ZipFile(recording.archive_path) as archive:
        member = _find_member(archive, VEHICLE_FILE)
        if member is None:
            raise FileNotFoundError(
                f"{VEHICLE_FILE} not found in {recording.archive_path.name}"
            )
        with archive.open(member) as stream:
            data = pd.read_csv(stream, low_memory=False)

    return normalize_vehicle_columns(data)


def read_vehicle_metadata(recording: RecordingArchive) -> pd.DataFrame | None:
    """Read optional vehicle metadata.

    The uploaded full package provides this metadata for Tianjin but not for
    Changchun, Chongqing, or Xi'an.
    """
    with zipfile.ZipFile(recording.archive_path) as archive:
        member = _find_member(archive, VEHICLE_META_FILE)
        if member is None:
            return None
        with archive.open(member) as stream:
            metadata = pd.read_csv(stream, low_memory=False)

    metadata.columns = [str(column).strip() for column in metadata.columns]
    return metadata
