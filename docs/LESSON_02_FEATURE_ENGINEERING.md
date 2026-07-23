# Lesson 02 — From Raw Frames to a Trajectory-Level Research Dataset

## 1. What we completed

The full SinD package contains 56 recording ZIP archives and more than 12 million vehicle frame records. This stage created a reproducible pipeline that reads each recording separately, keeps passenger cars, assigns a globally unique trajectory ID, applies transparent quality rules, and calculates one analytical row per trajectory.

The pipeline produced:

- 20,704 passenger-car trajectories before filtering.
- 756 uniquely excluded trajectories.
- 19,948 trajectories eligible for EDA and clustering.
- 0 duplicate trajectory IDs.
- 0 missing values in the ten core modeling features.
- 0 infinite values in the ten core modeling features.

## 2. Why one row per trajectory?

A raw row describes one vehicle at one timestamp. A vehicle can appear in hundreds or thousands of rows. If those rows were clustered directly:

- long trajectories would have more influence than short trajectories;
- neighboring frames would repeat nearly the same information;
- clusters would describe instantaneous states rather than complete observed behaviors.

The project therefore changes the observation unit from one frame to one complete passenger-car trajectory.

## 3. Why `track_id` is not enough

A `track_id` is unique only inside one recording. The same numeric ID may appear in another recording and refer to a different vehicle.

The pipeline creates:

```text
trajectory_uid = city + "__" + recording_id + "__" + track_id
```

This is a data-engineering requirement, not merely a naming preference.

## 4. Cleaning rules

### Rule A — Minimum duration

Trajectories shorter than five seconds are excluded because their speed variation, jerk, and stopping measures are based on too little temporal evidence.

### Rule B — Full-record stationary trajectory

A trajectory is excluded as stationary only when both conditions are satisfied:

- total travelled distance is at most 0.5 m; and
- maximum speed is at most 0.1 m/s.

The use of both conditions is deliberately conservative. A vehicle that waits at a red light but later moves is retained.

### Why an exclusion log matters

Deleting rows without a record makes the research impossible to audit. Every exclusion is stored with its trajectory ID, city, recording, duration, distance, maximum speed, flags, and reason.

## 5. Feature engineering concepts

### Speed

Speed is the magnitude of the horizontal and vertical velocity components:

```text
speed = sqrt(vx² + vy²)
```

It expresses how fast the vehicle moves regardless of direction.

### Longitudinal acceleration

`a_lon` measures acceleration along the vehicle's forward axis. Positive values represent acceleration; negative values represent deceleration.

### Maximum deceleration

The report uses braking magnitude, so the most negative acceleration is converted to a non-negative magnitude.

### Jerk

Jerk is the change in longitudinal acceleration divided by elapsed time. High absolute jerk means acceleration changes abruptly, which may indicate less smooth movement.

### Stop transition

A stop transition is counted when the trajectory changes from moving to stopped. A vehicle already stopped in its first frame contributes to stopped time but does not create an observed moving-to-stopped transition.

### Stopped-time ratio

The ratio is the proportion of observed frames whose speed is below 0.5 m/s. Because all four cities use approximately the same 9.99 Hz sampling rate, the frame proportion is a consistent approximation of the observed time proportion.

## 6. Why the pipeline does not use traffic-light data yet

The motion trajectory schema is consistent across all four cities. Traffic-light files are not:

- one Chongqing recording has no traffic-light CSV;
- schemas differ by city;
- some files are not chronologically sorted;
- some Xi'an files contain duplicate or incomplete rows;
- two Xi'an files contain an unlabeled fifth column.

Separating the core movement pipeline prevents these issues from contaminating the first behavior-clustering dataset.

## 7. Why Tianjin metadata remains optional

Only Tianjin contains `CrossType` and `Signal_Violation_Behavior` metadata in the uploaded package. These variables are retained for later post-cluster interpretation, but they are not included in the core modeling features.

This preserves two principles:

1. The clustering remains unsupervised.
2. The four-city model uses features available consistently in every city.

## 8. Results you should remember

| Item | Result |
|---|---:|
| Cities | 4 |
| Recordings | 56 |
| Passenger-car trajectories before filtering | 20,704 |
| Unique exclusions | 756 |
| Final modeling population | 19,948 |
| Short-only exclusions | 314 |
| Stationary-only exclusions | 388 |
| Short and stationary overlap | 54 |

City-level eligible trajectories:

| City | Eligible trajectories |
|---|---:|
| Changchun | 8,665 |
| Chongqing | 1,928 |
| Tianjin | 4,533 |
| Xi'an | 4,822 |

## 9. Likely lecturer questions

### Why did you not simply concatenate all vehicle CSV files?

Because track IDs restart across recordings, each recording must first receive city and recording identifiers. Direct concatenation could merge different vehicles under the same ID.

### Why exclude trajectories shorter than five seconds?

The purpose is to summarize complete behavior. A very short observation may not include enough information to estimate variability, stopping, or acceleration-change patterns reliably.

### Why is a stationary vehicle excluded when stopping is part of the study?

Only a vehicle that remains nearly stationary for the full observed trajectory is excluded. A vehicle that stops and later moves is retained and contributes to the stop-and-go behavior analysis.

### Why use only passenger cars?

Different participant types have different physical capabilities and movement patterns. Mixing cars, bicycles, motorcycles, buses, and trucks could make clusters reflect vehicle type rather than passenger-car behavior.

### Why not train the clustering model immediately?

The feature table contains extreme values. Formal EDA must first determine whether they are real rare behaviors or artifacts and decide on winsorization, robust scaling, or another treatment.

## 10. Next lesson

The next notebook is **03_full_dataset_eda.ipynb**. It will examine:

- city and recording imbalance;
- trajectory duration and distance;
- distributions of the ten features;
- correlations and redundancy;
- extreme trajectories;
- differences between cities;
- the risk that clusters represent city context rather than behavior.
