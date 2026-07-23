# Lesson 03 — Full-Dataset Exploratory Data Analysis

## 1. What this stage does

This stage studies the 19,948 modeling-eligible passenger-car trajectories before any scaling, PCA, or clustering is applied.

EDA is not a model. Its purpose is to decide whether the data is ready for distance-based algorithms and which preprocessing choices need to be tested.

## 2. Questions answered

1. Is the dataset balanced across cities and recordings?
2. Which features are skewed or contain long tails?
3. How many observations are statistical outliers under the IQR rule?
4. Which exact trajectories create the largest feature values?
5. Are some features strongly redundant?
6. Do feature distributions differ substantially by city?
7. Which preprocessing alternatives should be compared next?

## 3. Data composition

| city      |   trajectory_count |   percentage |
|:----------|-------------------:|-------------:|
| Changchun |               8665 |        43.44 |
| Xi'an     |               4822 |        24.17 |
| Tianjin   |               4533 |        22.72 |
| Chongqing |               1928 |         9.67 |

Changchun contributes the most trajectories and Chongqing the fewest. Pooled clustering may therefore be influenced more strongly by cities with larger samples. Later evaluation must include recording-level and city-level validation.

## 4. Descriptive-statistics findings

- Maximum observed speed: 99.47 m/s.
- Maximum acceleration: 126.19 m/s².
- Maximum deceleration magnitude: 115.74 m/s².
- Maximum observed stop-transition count: 15.

These values are not automatically deleted. They are traced back through `trajectory_uid` for raw-trajectory review.

## 5. IQR outlier diagnostics

The five features with the highest IQR outlier percentages are:

| feature                        |   outlier_count |   outlier_percentage |
|:-------------------------------|----------------:|---------------------:|
| max_acceleration_mps2          |            1669 |                 8.37 |
| mean_long_acc_mps2             |            1289 |                 6.46 |
| mean_abs_jerk_mps3             |             980 |                 4.91 |
| acceleration_std_mps2          |             889 |                 4.46 |
| observed_stop_transition_count |             449 |                 2.25 |

An IQR outlier is a statistical flag, not proof of an error. It may indicate rare real behavior, a boundary artifact, or a tracking/smoothing issue.

## 6. Correlation and feature redundancy

| feature_1                      | feature_2          |   spearman_correlation |   absolute_correlation |
|:-------------------------------|:-------------------|-----------------------:|-----------------------:|
| observed_stop_transition_count | stopped_time_ratio |               0.905939 |               0.905939 |
| mean_speed_mps                 | stopped_time_ratio |              -0.85211  |               0.85211  |

A high-correlation pair is not removed automatically. Feature meaning, PCA loadings, model sensitivity, and interpretability must also be considered.

## 7. City-effect diagnostic

The five features with the strongest rank-based city effect are:

| feature               |   rank_eta_squared_city |
|:----------------------|------------------------:|
| max_speed_mps         |                  0.2241 |
| speed_std_mps         |                  0.1226 |
| max_deceleration_mps2 |                  0.1073 |
| max_acceleration_mps2 |                  0.0763 |
| acceleration_std_mps2 |                  0.0664 |

A larger rank-based eta-squared value means that a greater share of ranked feature variation is associated with city groups. This supports the need for cross-city validation.

## 8. Provisional preprocessing candidates

### Candidate A
1st–99th percentile winsorization on the eight continuous features, followed by StandardScaler.

### Candidate B
No winsorization, followed by RobustScaler.

### Candidate C
1st–99th percentile winsorization on the eight continuous features, followed by RobustScaler.

`observed_stop_transition_count` is preserved as a valid discrete count by default. `stopped_time_ratio` is preserved because it is naturally bounded between zero and one.

## 9. Why not train immediately?

K-Means, DBSCAN, and PCA depend on distances. Scaling changes measurement units, but it does not eliminate the relative influence of extreme observations. The next stage must compare preprocessing candidates using retained PCA variance, cluster tendency, clustering metrics, stability, city sensitivity, and interpretability.

## 10. Likely lecturer questions

### Why use Spearman correlation?
Because several features are skewed and contain extreme values. Spearman is rank-based and does not require a strictly linear relationship.

### Why not delete every boxplot outlier?
Because an outlier may represent rare but real braking or acceleration behavior. Each extreme observation remains traceable to its raw trajectory.

### Why is city imbalance important?
Because pooled clusters may be influenced by the city contributing the largest number of trajectories. Cross-city validation checks whether the profiles generalize.

### Why not scale each city independently?
Doing so could erase genuine contextual differences and would create an unrealistic cross-city evaluation. Preprocessing must be fitted only on training data and then applied unchanged to held-out data.
