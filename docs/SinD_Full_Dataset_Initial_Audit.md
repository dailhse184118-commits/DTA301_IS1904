# SinD Full Dataset Study v2 — Initial Data Audit

Generated: 2026-07-23 03:22

## 1. Scope received

- Cities: 4
- Recordings: 56
- Dataset ZIP size: approximately 1.00 GB
- Estimated uncompressed CSV size: approximately 3.21 GB
- CSV files inside ZIPs: 236
- Vehicle frame-level rows: 12,052,293
- Vehicle trajectories: 30,369
- Passenger-car trajectories: 20,704
- Pedestrian frame-level rows: 1,291,343
- Pedestrian trajectories: 4,209
- Total recording duration: 17.22 hours

## 2. City summary

| city      |   recordings |   vehicle_rows |   vehicle_trajectories |   car_trajectories |   pedestrian_rows |   pedestrian_trajectories |   recordings_with_vehicle_meta |   traffic_light_files |   total_duration_hours |
|:----------|-------------:|---------------:|-----------------------:|-------------------:|------------------:|--------------------------:|-------------------------------:|----------------------:|-----------------------:|
| Changchun |            8 |        3158641 |                  10372 |               8774 |            151508 |                       765 |                              0 |                     8 |                3.1199  |
| Chongqing |           10 |        3080096 |                   2691 |               2171 |            363203 |                      1157 |                              0 |                     9 |                3.23031 |
| Tianjin   |           23 |        3182723 |                  11375 |               4752 |            606682 |                      1588 |                             23 |                    23 |                7.02305 |
| Xi'an     |           15 |        2630833 |                   5931 |               5007 |            169950 |                       699 |                              0 |                    15 |                3.8437  |

## 3. Core trajectory-data quality

All 56 ZIP archives opened successfully and passed ZIP integrity checks.

Across the required vehicle columns:

- Missing cells: 0
- Infinite numerical cells: 0
- Duplicate `(track_id, frame_id)` rows: 0
- Duplicate `(track_id, timestamp_ms)` rows: 0
- Tracks with inconsistent `agent_type`: 0
- Non-positive timestamp steps within vehicle tracks: 0
- Median sampling interval: 100.1001 ms in every city, approximately 9.99 Hz.

The preliminary trajectory filters identified:

- Passenger-car trajectories shorter than 5 seconds: 368
- Near-stationary passenger-car trajectories: 442

These counts are preliminary flags and may overlap. They are not yet the final number of removed trajectories.

## 4. Schema compatibility

The core vehicle schema is consistent across all four cities:

`track_id, frame_id, timestamp_ms, agent_type, x, y, vx, vy, yaw_rad, heading_rad, length, width, ax, ay, v_lon, v_lat, a_lon, a_lat`

An extra export-index column named `Unnamed: 0` is present and should be dropped.

The pedestrian schema is also consistent across all four cities.

## 5. Metadata limitation

All 23 Tianjin recordings contain:

- `Veh_tracks_meta.csv`
- `Ped_tracks_meta.csv`
- `recoding_metas.csv`

The Tianjin vehicle metadata matched the trajectory data exactly:

- IDs missing from metadata: 0
- Extra metadata IDs: 0
- Frame-count mismatches: 0

Changchun, Chongqing and Xi'an do not contain equivalent vehicle metadata files in the uploaded package. Therefore, variables such as `CrossType` and `Signal_Violation_Behavior` are directly available only for Tianjin.

## 6. Traffic-light audit

- Changchun: 8/8 recordings contain traffic-light data.
- Chongqing: 9/10 recordings contain traffic-light data.
- Tianjin: 23/23 recordings contain traffic-light data.
- Xi'an: 15/15 recordings contain traffic-light data.

Important exceptions:

1. `6_29_NR_3` has no traffic-light CSV.
2. Traffic-light schemas differ by city: 4 columns, 5 columns or 10 columns.
3. Some traffic-light files are not chronologically sorted.
4. `xian_412_m1` contains duplicate frame entries and one row with a missing timestamp.
5. `xian_415_n2` and `xian_415_n4` contain an unlabeled fifth column with non-empty values.
6. Several Changchun files contain timestamp values equal to zero at the beginning or end.

Traffic-light data must therefore be normalized and validated in a separate pipeline before it is used for signal-response analysis.

## 7. Recommended research design

The full study should use two linked analysis layers:

### Layer A — Multi-city core behavior clustering

Use all four cities and only features derivable consistently from vehicle trajectories:

- speed
- acceleration
- deceleration
- jerk
- stop transitions
- stopped-time ratio

This layer supports pooled clustering, cross-recording validation and cross-city validation.

### Layer B — Tianjin enriched interpretation

Use Tianjin metadata for post-cluster interpretation with:

- crossing type
- signal-violation behavior
- recording conditions

This avoids pretending that metadata exists for cities where it was not supplied.

## 8. Required next step

Do not train models yet.

The next stage is to build a reproducible multi-recording loader and create one trajectory-level master table with a unique key:

`trajectory_uid = city + recording_id + track_id`

Then apply documented quality filters, derive the core feature set, and compare city/recording distributions before deciding the final preprocessing and model benchmark.
