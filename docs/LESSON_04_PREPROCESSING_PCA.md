# Lesson 04 — Preprocessing Benchmark, PCA, and Cluster Tendency

## 1. Purpose of Stage 4

Stage 3 showed that the trajectory table is structurally clean but contains long right tails and extreme values, especially in maximum speed, acceleration, deceleration, and jerk.

Stage 4 therefore does not begin by choosing a final clustering model. It first compares three preprocessing pipelines under the same data and evaluation conditions.

| Code | Pipeline | Meaning |
|---|---|---|
| A | `A_winsor_standard` | Winsorize the eight continuous features at the 1st and 99th percentiles, then use StandardScaler |
| B | `B_raw_robust` | Keep raw feature values, then use RobustScaler |
| C | `C_winsor_robust` | Winsorize the eight continuous features, then use RobustScaler |

The stop-transition count and stopped-time ratio are not winsorized. The former is a meaningful discrete count and the latter is naturally bounded between zero and one.

## 2. How the code is organized

The reusable preprocessing code is stored in:

```text
src/preprocessing.py
```

The full experiment runner is:

```text
scripts/run_stage4_preprocessing_benchmark.py
```

Bootstrap stability is finalized through:

```text
scripts/finalize_stage4_selection.py
```

The notebook is:

```text
notebooks/04_preprocessing_pca_cluster_tendency.ipynb
```

## 3. Winsorization

For each of the eight continuous features, the training distribution supplies two bounds:

```text
lower bound = 1st percentile
upper bound = 99th percentile
```

Values outside the bounds are clipped to the relevant boundary. The trajectory is not deleted.

For example, under the full-data benchmark:

| Feature | 1st percentile | 99th percentile |
|---|---:|---:|
| Mean speed | 0.4987 | 13.9749 |
| Maximum speed | 4.6268 | 16.6069 |
| Maximum acceleration | 0.1665 | 9.3464 |
| Maximum deceleration | 0.0000 | 4.2558 |
| Mean absolute jerk | 0.1672 | 1.6552 |

Approximately 1% of observations are clipped at each tail where distinct values exist. This reduces the influence of extremes while preserving all 19,948 trajectories.

## 4. StandardScaler and RobustScaler

### StandardScaler

```text
z = (x - mean) / standard deviation
```

It makes features comparable in units but the mean and standard deviation can be influenced by extreme values.

### RobustScaler

```text
z_robust = (x - median) / IQR
```

It uses the median and interquartile range, so it is less sensitive to long tails.

Scaling is required because K-Means, PCA, DBSCAN, and related methods use distances. Without scaling, a feature with a larger numerical range can dominate the analysis.

## 5. PCA process

PCA transforms correlated features into orthogonal principal components. The experiment retains the smallest number of components whose cumulative explained variance reaches at least 90%.

All three pipelines required five components:

| Pipeline | Components | Retained variance |
|---|---:|---:|
| A — Winsor + Standard | 5 | 90.37% |
| B — Raw + Robust | 5 | 90.02% |
| C — Winsor + Robust | 5 | 91.71% |

For selected Pipeline C:

| Component | Individual variance | Cumulative variance |
|---|---:|---:|
| PC1 | 43.51% | 43.51% |
| PC2 | 19.87% | 63.39% |
| PC3 | 13.96% | 77.35% |
| PC4 | 8.93% | 86.28% |
| PC5 | 5.44% | 91.71% |


### Interpreting selected PCA loadings

- **PC1 — Dynamic intensity:** dominated by maximum acceleration, acceleration variability, and jerk.
- **PC2 — Stopping versus continuous movement:** positive loadings for stop transitions and stopped-time ratio, with a negative loading for mean speed.
- **PC3 — Acceleration direction:** contrasts positive mean longitudinal acceleration with braking-related and variability features.
- **PC4 — Speed ceiling and speed variation:** dominated by maximum speed and speed standard deviation.

PCA does not label behaviors by itself. These interpretations describe the feature combinations represented by each component.

## 6. Hopkins cluster tendency

The Hopkins statistic compares nearest-neighbor distances from real observations with distances from uniformly generated reference points.

Under the definition used here:

- around 0.50: approximately random spatial structure;
- substantially above 0.50: non-random and potentially clusterable structure;
- close to 1.00: very strong departure from uniform randomness.

| Pipeline | Mean Hopkins |
|---|---:|
| A_winsor_standard | 0.9401 |
| B_raw_robust | 0.9926 |
| C_winsor_robust | 0.9130 |


All three pipelines show non-random structure. However, Hopkins alone cannot select the preprocessing pipeline. Pipeline B's extremely high value was partly associated with unstable extreme structure, as shown by bootstrap testing.

## 7. Preliminary K-Means benchmark

For each pipeline, K-Means was tested from `k=2` through `k=10`. The experiment recorded:

- Silhouette score: higher is better.
- Davies–Bouldin index: lower is better.
- Calinski–Harabasz index: higher is generally better.
- Minimum and maximum cluster share.
- Seed stability measured by Adjusted Rand Index.

The strongest preliminary candidate inside each pipeline was:

| Pipeline | Preliminary k | Silhouette | DBI | Seed ARI mean | Smallest cluster share |
|---|---:|---:|---:|---:|---:|
| A_winsor_standard | 3 | 0.3414 | 1.1866 | 0.9992 | 15.62% |
| B_raw_robust | 2 | 0.5427 | 0.9714 | 1.0000 | 9.08% |
| C_winsor_robust | 2 | 0.4893 | 0.9755 | 0.9987 | 9.91% |


These figures are only a screening step. A high silhouette at `k=2` may reflect one broad majority group and one small extreme group rather than a useful set of interpretable driving profiles.

## 8. Why random-seed stability was not enough

Pipeline B appeared strongest from silhouette, Hopkins, and repeated random initializations. Its seed ARI was 1.00.

However, changing only the K-Means initialization is a weak stability test because every run still sees the same complete dataset. The project therefore added 20 bootstrap resamples. Each bootstrap model was trained on an 80% resampled dataset and then used to label the full dataset.

| Pipeline | Bootstrap ARI mean | Minimum ARI | Repeats below 0.80 | Smallest observed cluster share |
|---|---:|---:|---:|---:|
| A_winsor_standard | 0.9860 | 0.9657 | 0 | 14.3623% |
| B_raw_robust | 0.8802 | 0.0009 | 2 | 0.0050% |
| C_winsor_robust | 0.9820 | 0.9573 | 0 | 9.2942% |


### Critical finding

Pipeline B failed the mandatory bootstrap gate:

- 2 of 20 bootstrap runs collapsed;
- minimum ARI fell to approximately 0.001;
- the smallest predicted cluster fell to one trajectory.

This means the apparently excellent internal separation depended on a fragile extreme structure. The project rejected Pipeline B rather than selecting it from silhouette alone.

## 9. Final Stage 4 preprocessing decision

The stability-gated comparison selected:

```text
C_winsor_robust
```

Its configuration is:

```text
Winsorize the eight continuous features at 1st–99th percentiles
→ RobustScaler
→ PCA retaining at least 90% variance
→ five principal components retained
```

Evidence:

- PCA retained variance: 91.71%.
- Preliminary silhouette at k=2: 0.4893.
- Preliminary DBI at k=2: 0.9755.
- Bootstrap mean ARI: 0.9820.
- Bootstrap minimum ARI: 0.9573.
- Unstable bootstrap runs: 0 of 20.
- Minimum bootstrap cluster share: 9.29%.

Pipeline A remains the sensitivity baseline because it was also highly stable and uses the conventional winsorization-plus-StandardScaler design.

## 10. City-separation risk

A cross-validated multinomial logistic regression attempted to predict city from the transformed PCA features. Chance balanced accuracy is 0.25.

| Pipeline | Balanced city accuracy |
|---|---:|
| A_winsor_standard | 0.4414 |
| B_raw_robust | 0.4343 |
| C_winsor_robust | 0.4331 |


All values are above chance, so preprocessing does not remove city context completely. This is expected and confirms that Stage 5 and later validation must include held-out recordings and held-out cities.

## 11. Important limitation: k=2 is not the final answer

Pipeline C's strongest internal K-Means screening result occurred at `k=2`, but this is only a broad structural split. The project does not yet claim that there are only two meaningful driving profiles.

Stage 5 must compare:

- K-Means over multiple k values;
- Gaussian Mixture Models;
- Agglomerative clustering;
- DBSCAN and HDBSCAN where feasible;
- model stability and coverage;
- city and recording generalization;
- behavioral interpretability.

The final model may favor a larger number of profiles when interpretability and generalization justify it.

## 12. Likely lecturer questions

### Why did the group choose Pipeline C instead of the pipeline with the highest silhouette?

Because Pipeline B's high silhouette was not reproducible under bootstrap resampling. A model that collapses when the dataset composition changes slightly is not reliable.

### Why use both winsorization and RobustScaler?

Winsorization limits a small number of extreme continuous values, while RobustScaler centers and scales using the median and IQR. Their roles are different. The combination was accepted because it produced strong separation without bootstrap collapse.

### Does winsorization delete aggressive trajectories?

No. All trajectories remain. Only feature values outside the learned 1st and 99th percentile boundaries are clipped.

### Why retain five PCA components?

Five was the smallest number that preserved at least 90% of total variance under all three pipelines. Pipeline C retained 91.71%.

### Does Hopkins prove that K-Means is the correct model?

No. Hopkins indicates non-random spatial structure, but it does not identify the correct algorithm or number of clusters.

### Why is k=2 not final despite the highest preliminary score?

Internal metrics often prefer broad splits. The research objective requires detailed and interpretable behavior profiles, so final selection must combine metrics, stability, coverage, generalization, and interpretation.
