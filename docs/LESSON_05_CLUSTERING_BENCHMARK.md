# Lesson 05 — Full Clustering Model Benchmark and Final Profile Selection

## 1. Purpose of Stage 5

Stage 5 compares several unsupervised clustering families on the five PCA components selected in Stage 4. The goal is not to reward the model with the single highest Silhouette score. The goal is to select a structure that is:

- statistically defensible;
- stable when the data composition changes;
- available for every trajectory;
- interpretable in the original ten behavioral features;
- applicable to recordings and cities that were not used during fitting.

The input contains **19,948 passenger-car trajectories**, five PCA components, and no supervised label in the model input.

## 2. Models evaluated

### Full-dataset models

- K-Means, k = 2 to 8
- MiniBatch K-Means, k = 2 to 8
- Gaussian Mixture Model, 2 to 6 components

### Sample-screened models

- Ward Agglomerative Clustering, k = 2 to 6, on a city-stratified sample of 4,000 trajectories
- DBSCAN on a city-stratified sample of 7,000 trajectories
- HDBSCAN on the same 7,000-trajectory sample

Agglomerative and density models were sample-screened because Ward linkage has quadratic memory requirements, while high-epsilon DBSCAN configurations can create extremely large neighborhoods. They remain valid exploratory comparisons, but the study does not pretend they were full-dataset predictive models.

## 3. Shared evaluation metrics

### Silhouette Score

For trajectory i:

```text
s(i) = [b(i) - a(i)] / max[a(i), b(i)]
```

- a(i): average distance to the trajectory's own cluster
- b(i): lowest average distance to another cluster

A larger value indicates better separation and cohesion. Silhouette was calculated on a fixed sample for computational consistency.

### Davies–Bouldin Index

Lower values are better. It penalizes clusters that are internally dispersed and close to one another.

### Calinski–Harabasz Index

Higher values are better. It compares between-cluster dispersion with within-cluster dispersion.

### Coverage and noise

These are mandatory for density models. A model must not appear superior merely because it marks difficult observations as noise and evaluates only the easier remainder.

### Cluster balance

The smallest cluster percentage and normalized cluster entropy are reported. Extremely small clusters may represent a fragile tail rather than a reproducible behavioral profile.

### Original-feature interpretability

Eta-squared is calculated for every original feature. It estimates how much feature variation is associated with cluster membership. This prevents the project from evaluating clusters only in abstract PCA space.

## 4. Why k = 2 was not selected

K-Means with k = 2 achieved the strongest K-Means compactness:

- Silhouette: approximately 0.484 in the common benchmark
- full coverage: 100%
- strong production seed stability

However, k = 2 creates only a broad division, largely separating stop-oriented trajectories from continuous motion. It cannot preserve the additional distinction between smooth motion, moderate dynamic adjustment, and acceleration-intensive behavior.

More importantly, its recording-group subsample stability was weak:

- mean ARI: approximately 0.627
- minimum ARI: approximately 0.035

This means its broad boundary can change substantially when recordings are omitted, even though random-seed stability is high.

## 5. Why k = 5 was not selected

K-Means with k = 5 increases original-feature separation and achieves a slightly lower Davies–Bouldin score than k = 4. However, the additional split is less stable:

- production seed ARI mean: approximately 0.955
- bootstrap ARI mean: approximately 0.687
- leave-one-city-out ARI mean: approximately 0.713
- recording-subsample ARI mean: approximately 0.740

The fifth cluster therefore adds detail, but the detail is not as reproducible as the four-profile solution.

## 6. Why K-Means k = 4 was selected

The final model is:

```text
Stage 4 preprocessing C
→ 1%–99% winsorization of eight continuous features
→ RobustScaler
→ five PCA components
→ K-Means with k = 4
→ n_init = 50
→ random_state = 42
```

Final evidence:

| Criterion | Result |
|---|---:|
| Coverage | 100.00% |
| Silhouette, 5,000-row sample | 0.2963 |
| Davies–Bouldin | 1.2791 |
| Calinski–Harabasz | 8298.68 |
| Production seed ARI mean | 0.9850 |
| Production seed ARI minimum | 0.9776 |
| Bootstrap ARI mean | 0.8847 |
| Bootstrap ARI minimum | 0.8047 |
| Leave-one-city-out ARI mean | 0.8265 |
| Recording-subsample ARI mean | 0.9208 |

K = 4 is not the winner of every individual metric. It is selected because it gives the best combined trade-off between granularity, coverage, stability, and interpretable behavioral meaning.

## 7. Final behavioral profiles

|   profile_id | profile_name             |   trajectory_count |   percentage |
|-------------:|:-------------------------|-------------------:|-------------:|
|            1 | Smooth and Steady        |               7477 |        37.48 |
|            2 | Stop-and-Go              |               6866 |        34.42 |
|            3 | Dynamic Speed Adjustment |               4338 |        21.75 |
|            4 | Acceleration-Intensive   |               1267 |         6.35 |

### Profile 1 — Smooth and Steady

Typical median characteristics:

- mean speed: 6.727 m/s
- speed standard deviation: 1.196 m/s
- maximum acceleration: 0.993 m/s²
- maximum deceleration: 0.799 m/s²
- mean absolute jerk: 0.436 m/s³
- stopped-time ratio: 0

This profile represents continuous motion with the lowest speed variation, acceleration variation, braking magnitude, and jerk.

### Profile 2 — Stop-and-Go

Typical median characteristics:

- mean speed: 2.082 m/s
- speed standard deviation: 2.928 m/s
- observed stop transitions: 1
- stopped-time ratio: 0.591

Its low average speed, high speed variation, and large stopped-time ratio make the interpretation direct and robust.

### Profile 3 — Dynamic Speed Adjustment

Typical median characteristics:

- mean speed: 6.625 m/s
- maximum speed: 10.864 m/s
- speed standard deviation: 2.098 m/s
- maximum acceleration: 1.955 m/s²
- maximum deceleration: 1.927 m/s²
- mean absolute jerk: 0.780 m/s³

This profile is more active than Smooth and Steady but much less extreme than Acceleration-Intensive. Vehicles make repeated speed and acceleration adjustments while generally remaining in motion.

### Profile 4 — Acceleration-Intensive

Typical median characteristics:

- maximum acceleration: 6.843 m/s²
- acceleration standard deviation: 1.529 m/s²
- mean absolute jerk: 1.111 m/s³
- mean longitudinal acceleration: 0.358 m/s²

This is the smallest profile, but it still contains 1,267 trajectories, or 6.35% of the full modeling population. It is not a tiny outlier cluster.

## 8. Why other models were not selected

| candidate                        | decision               | reason                                                                                                                                       |
|:---------------------------------|:-----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| K-Means, k=2                     | Not selected           | Best compactness but too coarse; recording-group stability was much weaker and it collapses distinct dynamic profiles.                       |
| K-Means, k=4                     | Selected primary model | Best balance of full coverage, production seed stability, recording robustness, four interpretable profiles, and manageable complexity.      |
| K-Means, k=5                     | Sensitivity candidate  | Higher feature separation but lower bootstrap, seed, city, and recording stability; fifth cluster adds a less robust split.                  |
| MiniBatch K-Means, k=4           | Not selected           | Similar structure but slightly weaker internal metrics; scalability advantage is unnecessary for 19,948 trajectories.                        |
| Gaussian Mixture, k=4            | Not selected           | Flexible probabilistic model, but substantially lower silhouette and higher Davies–Bouldin score.                                            |
| Agglomerative, k=4 (sample)      | Exploratory only       | Interpretable sample structure, but full 19,948-row Ward clustering is not memory-scalable and has no direct prediction for held-out cities. |
| DBSCAN (sample best compactness) | Complementary only     | High sample silhouette but one cluster is below 1% and 11.17% is noise; sensitive to density parameters.                                     |
| HDBSCAN (sample)                 | Complementary only     | Finds a broad density structure but leaves 39.63% as noise, so it cannot serve as the primary full-coverage profiling model.                 |

### Density-model interpretation

The best DBSCAN sample setting achieved a high Silhouette score, but its smallest cluster represented less than 1% of the full sample and approximately 11% was marked as noise. HDBSCAN produced meaningful broad structure but marked approximately 40% to 56% of the sample as noise depending on the setting.

Density models are therefore valuable as complementary evidence for dense cores and sparse trajectories, but not as the primary system when the research requires a profile for every valid trajectory.

## 9. City composition and generalization

Every profile appears in every city. This is important because no final profile is exclusive to one city.

|   profile_id | profile_name             |   Changchun |   Chongqing |   Tianjin |   Xi'an |
|-------------:|:-------------------------|------------:|------------:|----------:|--------:|
|            1 | Smooth and Steady        |        31.1 |        10.6 |      33.8 |    24.6 |
|            2 | Stop-and-Go              |        48.0 |        12.4 |      16.4 |    23.3 |
|            3 | Dynamic Speed Adjustment |        55.0 |         5.8 |      14.6 |    24.6 |
|            4 | Acceleration-Intensive   |        52.5 |         2.9 |      19.4 |    25.2 |

However, city proportions differ across profiles. The project must not interpret a cluster as universally identical without the next cross-city validation stage.

## 10. Strongest feature separation

| feature                        |   eta_squared |
|:-------------------------------|--------------:|
| stopped_time_ratio             |        0.7630 |
| max_acceleration_mps2          |        0.6076 |
| acceleration_std_mps2          |        0.6048 |
| mean_speed_mps                 |        0.5233 |
| mean_abs_jerk_mps3             |        0.5058 |
| observed_stop_transition_count |        0.4942 |

These values show which original features most strongly distinguish the four profiles.

## 11. Tianjin metadata is post-cluster interpretation only

`CrossType` and `Signal_Violation_Behavior` were merged only after model fitting. They did not influence cluster creation.

This is methodologically important:

```text
motion features → unsupervised clustering → metadata interpretation
```

not:

```text
metadata labels → cluster formation
```

## 12. Code workflow

### Model sweep

```bash
python scripts/run_stage5_clustering_benchmark.py
```

This creates the family-level benchmark tables.

### K-Means stability

```bash
python scripts/run_stage5_kmeans_stability.py
python scripts/run_stage5_production_seed_stability.py
```

The first script tests bootstrap, city omission, and recording-group omission. The second confirms that production-style repeated initialization removes most random-start sensitivity.

### Finalization

```bash
python scripts/finalize_stage5_selection.py
```

This fits K-Means k = 4 with `n_init=50`, saves the model, creates final assignments, builds original-feature profiles, calculates city and Tianjin metadata tables, and selects representative trajectories nearest each PCA centroid.

## 13. Likely lecturer questions

### Did the group select k = 4 because it matched the old sample result?

No. The old result motivated a continuity candidate, but the full-data decision used full-dataset metrics, bootstrap stability, leave-one-city-out testing, recording-group subsampling, cluster balance, and original-feature interpretation.

### Why not choose the highest Silhouette score?

Because high Silhouette can favor a coarse k = 2 split or a density configuration that labels difficult cases as noise. The research requires detailed, stable, full-coverage profiles.

### Is a Silhouette around 0.30 too low?

Not automatically. Real-world trajectory behavior is continuous and overlapping rather than naturally separated into perfectly isolated classes. The score must be interpreted together with stability, feature profiles, balance, and generalization.

### Why is K-Means appropriate when behavior is not spherical?

K-Means is used as the primary interpretable partition after robust preprocessing and PCA. GMM and density models were included specifically to test whether a more flexible geometry produced a clearly superior and stable alternative. They did not provide a better overall solution.

### Why is the Acceleration-Intensive cluster valid if it is the smallest?

It contains 1,267 trajectories, representing 6.35% of the full modeling dataset. It is much larger than a sparse anomaly group and remains interpretable across cities.

## 14. Stage 5 conclusion

The selected primary model is **K-Means with four behavioral profiles**. It provides 100% coverage and a defensible balance of stability and interpretability.

The four profiles are:

1. Smooth and Steady
2. Stop-and-Go
3. Dynamic Speed Adjustment
4. Acceleration-Intensive

The next stage must validate whether these profiles preserve their meanings across individual recordings and held-out cities, not merely whether the same numeric labels appear.
