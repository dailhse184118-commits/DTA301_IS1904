# SinD Data Source and Access

## Dataset

SinD is a real-world trajectory dataset for traffic participants at
signalized intersections.

The full research package used in this project contains recordings from
Changchun, Chongqing, Tianjin, and Xi'an.

## Access condition

The dataset was supplied to the student group for academic,
non-commercial research. The raw files are not included in this public
repository.

Do not:

- upload the original recording ZIP archives to GitHub;
- publish access links, passwords, or download codes;
- redistribute the full data to unauthorized users.

Obtain access from the official dataset provider and comply with the provider's
citation and usage requirements.

## Local placement

Store the raw archives outside the repository and pass their parent directory
to the pipeline:

```powershell
python run_pipeline.py --raw-root "D:\PrivateDatasets\SinD"
```

Expected archive naming conventions include:

```text
changchun_*.zip
*_NR_*.zip
8_2_1.zip
xian_*.zip
```

The loader searches recursively and reads `Veh_smoothed_tracks.csv` directly
from each ZIP archive.

## Why processed trajectory files are excluded

The GitHub-ready package excludes row-level derived data and model assignment
files as a conservative response to the dataset's redistribution condition.
Aggregate statistics, code, figures, and report evidence are included.
