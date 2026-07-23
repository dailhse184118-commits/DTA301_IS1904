# Lesson 06 — Cross-City and Cross-Recording Validation

## 1. Research purpose

Stage 5 selected a four-profile K-Means solution from all 19,948 trajectories.
Stage 6 asks a stricter question:

> Does the same four-profile structure remain reproducible when a complete
> city or a complete recording is excluded from training?

This is not supervised prediction. The Stage 5 partition is a reference
partition, not ground-truth behavior labels. Agreement metrics measure
reproducibility and transferability of the discovered structure.

## 2. Why the validation must be end-to-end

For every split, the following steps are fitted using training rows only:

```text
Training trajectories
→ learn 1st and 99th percentile winsorization bounds
→ fit RobustScaler
→ fit PCA retaining at least 90% variance
→ fit K-Means k=4
→ align arbitrary cluster numbers with reference profile IDs
→ transform and predict the held-out context
```

The held-out city or recording is never used to fit winsorization, scaling,
PCA, or K-Means. This prevents preprocessing leakage.

## 3. Why cluster alignment is required

K-Means cluster numbers have no intrinsic meaning. A new fit may call
Stop-and-Go cluster 0, 1, 2, or 3.

Hungarian matching is applied on training rows to find the one-to-one mapping
that maximizes agreement with the Stage 5 reference profiles. The mapping is
then applied unchanged to the held-out rows.

## 4. Leave-one-city-out results

| held_out_city   |   n_test |   pca_components |   pca_retained_variance |    ari |   aligned_accuracy |   balanced_accuracy |   semantic_mae_mean |   semantic_profile_correlation_mean |
|:----------------|---------:|-----------------:|------------------------:|-------:|-------------------:|--------------------:|--------------------:|------------------------------------:|
| Changchun       |     8665 |                5 |                  0.9116 | 0.6858 |             0.8352 |              0.8500 |              0.1666 |                              0.8499 |
| Chongqing       |     1928 |                5 |                  0.9165 | 0.9658 |             0.9850 |              0.9568 |              0.2167 |                              0.8859 |
| Tianjin         |     4533 |                5 |                  0.9137 | 0.8852 |             0.9565 |              0.9176 |              0.2362 |                              0.8449 |
| Xi'an           |     4822 |                5 |                  0.9299 | 0.8766 |             0.9513 |              0.9404 |              0.1130 |                              0.9137 |

Summary:

- Mean ARI: **0.8533**
- Minimum ARI: **0.6858**
- Mean aligned agreement: **0.9320**
- Mean semantic profile correlation: **0.8736**
- All four profiles were recovered in **4/4** held-out cities.

### Interpretation

Chongqing, Tianjin, and Xi'an transfer strongly. Changchun is the hardest
held-out city, with ARI 0.6858.
This does not destroy the four-profile structure: all four profiles remain
present and aligned agreement remains above 0.83. It shows that Changchun has
a stronger context shift in feature distribution and cluster boundaries.

## 5. Leave-one-recording-out results

All 56 recordings were held out once.

- Mean ARI: **0.9689**
- Median ARI: **0.9774**
- 10th percentile ARI: **0.9485**
- Minimum ARI: **0.6082**
- Mean aligned agreement: **0.9884**
- Recordings with ARI at least 0.80: **55/56**
- Recordings with ARI below 0.60: **0/56**

Lowest results:

| city      | recording_id             |   n_test |    ari |   aligned_accuracy |   balanced_accuracy |
|:----------|:-------------------------|---------:|-------:|-------------------:|--------------------:|
| Changchun | changchun_pudong_507_009 |     1166 | 0.6082 |             0.8362 |              0.7591 |
| Chongqing | 6_29_NR_4                |      161 | 0.8929 |             0.9627 |              0.8889 |
| Changchun | changchun_pudong_507_010 |     1207 | 0.9401 |             0.9760 |              0.9763 |
| Tianjin   | 8_6_2                    |      209 | 0.9402 |             0.9809 |              0.9602 |
| Tianjin   | 8_9_1                    |      208 | 0.9405 |             0.9760 |              0.9730 |
| Tianjin   | 8_3_4                    |      241 | 0.9477 |             0.9834 |              0.9779 |
| Xi'an     | xian_415_n2              |      416 | 0.9494 |             0.9832 |              0.9787 |
| Xi'an     | xian_412_m1              |      350 | 0.9504 |             0.9829 |              0.9844 |

Only `changchun_pudong_507_009` shows a major reduction relative to the other
recordings. Its ARI is still positive and all four profiles are recovered, but
the boundary agreement is weaker. This recording should be discussed as a
context-sensitive case rather than hidden.

## 6. Semantic consistency

ARI checks whether trajectories are grouped similarly. Semantic validation
checks whether a profile still has the same feature meaning.

For each held-out context, the median of each predicted profile is compared
with the global profile median after dividing each feature difference by the
global feature IQR.

- Lower normalized MAE means closer profile medians.
- Higher profile-vector correlation means the pattern of high and low
  behavioral features remains similar.

Mean semantic profile correlations are above 0.79 across the 56 recording
splits and above 0.84 in every city split.

## 7. Profile prevalence by city

| city      | profile_name             |   trajectory_count |   city_percentage |
|:----------|:-------------------------|-------------------:|------------------:|
| Changchun | Smooth and Steady        |               2323 |             26.81 |
| Changchun | Stop-and-Go              |               3293 |             38.00 |
| Changchun | Dynamic Speed Adjustment |               2384 |             27.51 |
| Changchun | Acceleration-Intensive   |                665 |              7.67 |
| Chongqing | Smooth and Steady        |                790 |             40.98 |
| Chongqing | Stop-and-Go              |                848 |             43.98 |
| Chongqing | Dynamic Speed Adjustment |                253 |             13.12 |
| Chongqing | Acceleration-Intensive   |                 37 |              1.92 |
| Tianjin   | Smooth and Steady        |               2527 |             55.75 |
| Tianjin   | Stop-and-Go              |               1127 |             24.86 |
| Tianjin   | Dynamic Speed Adjustment |                633 |             13.96 |
| Tianjin   | Acceleration-Intensive   |                246 |              5.43 |
| Xi'an     | Smooth and Steady        |               1837 |             38.10 |
| Xi'an     | Stop-and-Go              |               1598 |             33.14 |
| Xi'an     | Dynamic Speed Adjustment |               1068 |             22.15 |
| Xi'an     | Acceleration-Intensive   |                319 |              6.62 |

The four profiles occur in all cities, but their prevalence differs. Profile
prevalence therefore reflects both behavioral structure and local traffic
context. The report must not claim that every city has identical behavior
proportions.

## 8. Final Stage 6 decision

**Validated with context sensitivity**

The four-profile solution is strongly reproducible across recordings and remains transferable across held-out cities. Changchun produces the weakest city transfer and one Changchun recording is notably less stable, so city and recording context must remain explicit in the report.

The final claim should be:

> The four profiles are reproducible across recordings and transferable across
> cities, while their frequency and exact boundaries remain context-sensitive.

The claim should not be:

> The four cities have identical driving behavior.

## 9. Code logic in plain language

### Train-only preprocessing

```python
preprocessor.fit(train_data)
X_train = preprocessor.transform(train_data)
X_test = preprocessor.transform(test_data)
```

The held-out data is transformed using the training bounds, training median,
training IQR, and training PCA loadings.

### Cluster alignment

```python
aligned_train, mapping = align_clusters_to_reference(
    train_reference_profiles,
    new_train_clusters,
)
aligned_test = apply_mapping(new_test_clusters, mapping)
```

The test rows do not decide the mapping.

### Agreement metrics

- **ARI:** compares pairs of trajectories and does not depend on cluster number.
- **Aligned accuracy:** percentage matching the reference profile after
  training-only alignment.
- **Balanced accuracy:** gives equal importance to each profile.
- **Semantic normalized MAE:** measures how far profile medians shift.
- **Profile-vector correlation:** checks whether the feature pattern remains
  interpretable in the same way.

## 10. Likely lecturer questions

### Is aligned accuracy a real classification accuracy?

No. There are no ground-truth behavioral labels. It is agreement with the
Stage 5 reference partition after cluster-number alignment.

### Why fit PCA again for every split?

Because using PCA fitted on the full dataset would leak information from the
held-out city or recording into the training process.

### Why is Changchun weaker?

The data supports a context shift, but Stage 6 alone does not prove one causal
reason. Possible factors include different speed distributions, intersection
geometry, traffic density, or recording composition. Those are hypotheses,
not confirmed causes.

### Does one weaker recording invalidate the model?

No. Fifty-five recordings have ARI above 0.89 and the overall median is above
0.97. The weaker recording is transparently reported as a limitation and a
useful case for later trajectory-level inspection.

## 11. Next stage

Stage 7 should perform final interpretation and decision support:

- profile explanations using raw units;
- representative trajectories;
- Tianjin CrossType and Signal_Violation_Behavior analysis;
- city and recording prevalence;
- operational risk-priority logic;
- final model comparison and study conclusions.
