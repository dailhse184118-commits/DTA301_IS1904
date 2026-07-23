"""Run the full SinD analysis pipeline from private raw ZIP archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Private directory containing all SinD recording ZIP archives.",
    )
    parser.add_argument(
        "--skip-stage6",
        action="store_true",
        help="Skip the slower cross-context validation stage.",
    )
    parser.add_argument(
        "--skip-stage7",
        action="store_true",
        help="Skip raw representative-trajectory interpretation.",
    )
    return parser.parse_args()


def run(script: str, *arguments: str) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / script),
        *arguments,
    ]
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    raw_root = str(args.raw_root.resolve())

    run("run_stage1_dataset_audit.py", "--raw-root", raw_root)
    run(
        "run_feature_pipeline.py",
        "--raw-root",
        raw_root,
        "--output-root",
        str(PROJECT_ROOT),
    )
    run("run_stage3_eda.py")
    run("run_stage4_preprocessing_benchmark.py")
    run("finalize_stage4_selection.py")
    run("run_stage5_clustering_benchmark.py")
    run("run_stage5_kmeans_stability.py")
    run("run_stage5_production_seed_stability.py")
    run("finalize_stage5_selection.py")

    if not args.skip_stage6:
        run("run_stage6_cross_context_validation.py")

    if not args.skip_stage7:
        run("run_stage7_interpretation.py", "--raw-root", raw_root)

    print("\nFull pipeline completed.")


if __name__ == "__main__":
    main()
