# Vehicle Behavior Analysis Using Real-World Trajectory Data

DTA301 – Data Analysis  
Research-Based Learning project using the full SinD trajectory dataset.

## Project summary

This project analyzes passenger-car behavior at signalized intersections using
real-world vehicle trajectories from four cities:

- Changchun
- Chongqing
- Tianjin
- Xi'an

The private full dataset contains 56 recordings and 12,052,293 vehicle-frame
records. The processing pipeline identified 20,704 passenger-car trajectories,
removed 756 short or full-record stationary trajectories, and retained 19,948
trajectories for modeling.

The selected workflow is:

```text
Raw recording ZIP archives
→ trajectory quality filtering
→ ten interpretable behavior features
→ 1st–99th percentile winsorization
→ RobustScaler
→ PCA with five components
→ K-Means with k=4
→ cross-city and cross-recording validation
→ post-cluster interpretation
```

## Final behavioral profiles

| Profile | Trajectories | Share |
|---|---:|---:|
| Smooth and Steady | 7,477 | 37.48% |
| Stop-and-Go | 6,866 | 34.42% |
| Dynamic Speed Adjustment | 4,338 | 21.75% |
| Acceleration-Intensive | 1,267 | 6.35% |

The four-profile model covers all eligible trajectories. It is highly stable
to initialization and strongly reproducible across recordings, while some
profile boundaries remain context-sensitive across cities.

## Public repository and dataset restriction

The SinD dataset was supplied for non-commercial academic research. The raw
recording ZIP archives are intentionally excluded from this repository and
must not be uploaded publicly.

This public-safe repository contains:

- analysis source code;
- reproducible scripts and notebooks;
- aggregate result tables and figures;
- D1–D5 and GitHub submission reports;
- final presentation slides.

It does **not** contain:

- raw SinD trajectory files;
- trajectory-level processed datasets;
- per-trajectory predictions;
- representative raw trajectory time series.

See [DATA_SOURCE.md](DATA_SOURCE.md) before running the project.

## Repository structure

```text
DTA301_IS1904/
├── README.md
├── DATA_SOURCE.md
├── AI_USAGE.md
├── requirements.txt
├── run_pipeline.py
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── notebooks/
├── src/
├── scripts/
├── outputs/
│   ├── figures/
│   └── tables/
├── reports/
├── slides/
└── tests/
```

## Environment setup

Recommended Python version: 3.11–3.13.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Raw data placement

Keep the SinD archives outside the Git repository, for example:

```text
D:\PrivateDatasets\SinD\
├── Changchun\
├── Chongqing\
├── Tianjin\
└── Xi'an\
```

The loader searches recursively, so all 56 recording ZIP files may remain in
city subfolders.

## Run the full pipeline

```powershell
python run_pipeline.py --raw-root "D:\PrivateDatasets\SinD"
```

The full pipeline includes the slower cross-city and cross-recording
validation stages. For a quicker development run:

```powershell
python run_pipeline.py `
  --raw-root "D:\PrivateDatasets\SinD" `
  --skip-stage6 `
  --skip-stage7
```

## Run individual stages

```powershell
python scripts/run_stage1_dataset_audit.py --raw-root "D:\PrivateDatasets\SinD"
python scripts/run_feature_pipeline.py --raw-root "D:\PrivateDatasets\SinD" --output-root .
python scripts/run_stage3_eda.py
python scripts/run_stage4_preprocessing_benchmark.py
python scripts/finalize_stage4_selection.py
python scripts/run_stage5_clustering_benchmark.py
python scripts/run_stage5_kmeans_stability.py
python scripts/run_stage5_production_seed_stability.py
python scripts/finalize_stage5_selection.py
python scripts/run_stage6_cross_context_validation.py
python scripts/run_stage7_interpretation.py --raw-root "D:\PrivateDatasets\SinD"
```

## Core modeling features

```text
mean_speed_mps
max_speed_mps
speed_std_mps
mean_long_acc_mps2
max_acceleration_mps2
max_deceleration_mps2
acceleration_std_mps2
mean_abs_jerk_mps3
observed_stop_transition_count
stopped_time_ratio
```

`CrossType` and `Signal_Violation_Behavior` are not clustering inputs. They are
available only for Tianjin and are used after clustering for interpretation.

## Key evidence

- PCA retained variance: 91.71% using five components.
- Final model: K-Means, k=4, n_init=50.
- Coverage: 100%.
- Minimum production seed ARI: 0.9776.
- Minimum bootstrap ARI: 0.8047.
- Mean leave-one-city-out ARI: 0.8533.
- Median leave-one-recording-out ARI: 0.9774.

The Silhouette score is moderate rather than perfect, so the four profiles
should be presented as interpretable overlapping behavior patterns, not as
ground-truth classes.

## Reports and slides

The `reports/` folder contains D1–D5 and the GitHub/code submission report.
The `slides/` folder contains the final ten-slide presentation.

## AI usage

AI assistance is documented in [AI_USAGE.md](AI_USAGE.md). AI was used for
explanation, code drafting, debugging, document structure, and language
editing. Dataset processing decisions, numerical results, and final claims are
traceable to executable code and saved evidence tables.

## Important interpretation limits

This project supports behavioral monitoring and review prioritization. It does
not estimate crash probability, prove causal safety relationships, or provide
an automated enforcement decision.
