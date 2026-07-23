# Lesson 07 — Final Profile Interpretation and Decision Support

## 1. What Stage 7 does

Stage 7 converts the validated four-cluster solution into understandable
behavioral profiles and a cautious operational decision-support framework.

The stage has four tasks:

1. Explain each profile in the original physical units.
2. Select representative real trajectories near each K-Means centroid.
3. Interpret Tianjin-only `CrossType` and
   `Signal_Violation_Behavior` metadata after clustering.
4. Translate profile evidence into operational review priorities without
   pretending that clustering produces crash probabilities.

## 2. Evidence base

The final model uses:

```text
19,948 eligible passenger-car trajectories
→ 10 trajectory-level behavioral features
→ 1%–99% winsorization on eight continuous features
→ RobustScaler
→ PCA with five components and 91.71% retained variance
→ K-Means with k=4
```

The model assigns every eligible trajectory to one profile. Stage 6 showed:

- mean leave-one-city-out ARI = **0.8533**;
- median leave-one-recording-out ARI = **0.9774**.

The profiles are therefore reproducible, but the internal Silhouette score is
only **0.2963**.
This means the profiles overlap and should not be described as perfectly
separated natural classes.

## 3. Final profile distribution

|   profile_id | profile_name             |   trajectory_count |   percentage |
|-------------:|:-------------------------|-------------------:|-------------:|
|            1 | Smooth and Steady        |               7477 |        37.48 |
|            2 | Stop-and-Go              |               6866 |        34.42 |
|            3 | Dynamic Speed Adjustment |               4338 |        21.75 |
|            4 | Acceleration-Intensive   |               1267 |         6.35 |

## 4. Profile interpretation in raw units

|   profile_id | profile_name             |   mean_speed_mps |   max_speed_mps |   speed_std_mps |   mean_long_acc_mps2 |   max_acceleration_mps2 |   max_deceleration_mps2 |   acceleration_std_mps2 |   mean_abs_jerk_mps3 |   observed_stop_transition_count |   stopped_time_ratio |
|-------------:|:-------------------------|-----------------:|----------------:|----------------:|---------------------:|------------------------:|------------------------:|------------------------:|---------------------:|---------------------------------:|---------------------:|
|            1 | Smooth and Steady        |           6.7272 |          9.3688 |          1.1957 |               0.1518 |                  0.9932 |                  0.7995 |                  0.4895 |               0.4363 |                           0.0000 |               0.0000 |
|            2 | Stop-and-Go              |           2.0817 |          9.8829 |          2.9281 |               0.0777 |                  1.7898 |                  1.7215 |                  0.6491 |               0.3606 |                           1.0000 |               0.5909 |
|            3 | Dynamic Speed Adjustment |           6.6253 |         10.8638 |          2.0979 |               0.2468 |                  1.9552 |                  1.9265 |                  0.9678 |               0.7804 |                           0.0000 |               0.0000 |
|            4 | Acceleration-Intensive   |           6.7068 |         10.7049 |          1.9101 |               0.3582 |                  6.8431 |                  2.1843 |                  1.5291 |               1.1112 |                           0.0000 |               0.0000 |

### Profile 1 — Smooth and Steady

This is the largest profile. It has continuous movement, no typical stopped
time, and the lowest dynamic-control intensity. Its median speed variability,
maximum acceleration, maximum deceleration, and acceleration variability are
all below the overall center.

Operational meaning:

- use it as a baseline flow pattern;
- compare other profiles against it;
- monitor large changes in its prevalence by city or recording.

Important caution:

The word *smooth* is not a safety label. In Tianjin, 3.60% of trajectories in
this profile have a recorded signal-violation label.

### Profile 2 — Stop-and-Go

This profile has the lowest mean speed and the highest stopping dimension.

Key evidence:

- median observed stop-transition count = 1;
- median stopped-time ratio = 0.5909;
- median speed variability = 2.9281 m/s.

Operational meaning:

- inspect queue formation;
- review signal timing and delay;
- identify possible spillback or repeated stopping.

Important caution:

Stopping at a signalized intersection is often expected. This profile should
not be interpreted automatically as unsafe behavior.

### Profile 3 — Dynamic Speed Adjustment

This profile usually remains in motion but changes speed and acceleration more
strongly than Smooth and Steady.

Key evidence:

- median acceleration standard deviation = 0.9678 m/s²;
- median absolute jerk = 0.7804 m/s³;
- median maximum deceleration = 1.9265 m/s².

Operational meaning:

- inspect approach-speed consistency;
- locate braking and speed-adjustment zones;
- compare with geometry, signal phase, and route direction.

Important caution:

Dynamic behavior is descriptive. It does not prove aggression or a safety
violation.

### Profile 4 — Acceleration-Intensive

This is the smallest profile but still contains 1,267 trajectories.

Key evidence:

- median maximum acceleration = 6.8431 m/s²;
- median acceleration variability = 1.5291 m/s²;
- median absolute jerk = 1.1112 m/s³.

Its robust maximum-acceleration score is approximately
**4.95**
global IQRs above the overall median.

Operational meaning:

- prioritize raw-trajectory verification;
- review high-acceleration and high-jerk maneuvers;
- distinguish real maneuvers from tracking or smoothing artifacts.

## 5. Representative trajectories

One observed trajectory is selected for each profile by finding the trajectory
nearest to the K-Means centroid in the five-dimensional PCA space.

|   profile_id | profile_name             | trajectory_uid                           | city      | recording_id             |   track_id |   distance_to_centroid |   trajectory_duration_s |
|-------------:|:-------------------------|:-----------------------------------------|:----------|:-------------------------|-----------:|-----------------------:|------------------------:|
|            1 | Smooth and Steady        | Xi'an__xian_412_n3__341                  | Xi'an     | xian_412_n3              |        341 |                 0.1589 |                 13.3133 |
|            2 | Stop-and-Go              | Xi'an__xian_412_m2__31                   | Xi'an     | xian_412_m2              |         31 |                 0.1691 |                 73.3734 |
|            3 | Dynamic Speed Adjustment | Chongqing__6_29_NR_2__165                | Chongqing | 6_29_NR_2                |        165 |                 0.2514 |                 12.1121 |
|            4 | Acceleration-Intensive   | Changchun__changchun_pudong_507_009__359 | Changchun | changchun_pudong_507_009 |        359 |                 0.5853 |                 22.8228 |

Why this method is useful:

- it selects a model-typical observation rather than the most extreme one;
- the raw path, speed, and acceleration can be inspected;
- the examples are real observed trajectories, not synthetic averages.

Important limitation:

One representative trajectory cannot show the entire range of a profile. It
is an illustration, not a proof that every member behaves identically.

## 6. Tianjin CrossType interpretation

`CrossType` is available only for Tianjin and was not used to form the
clusters.

The primary profile-by-crossing-type association, excluding the sparse
`Others` category, gives:

- chi-square = **428.149**;
- p-value = **2.471e-89**;
- bias-corrected Cramer's V = **0.217**.

The relationship is statistically clear and practically stronger than the
violation relationship.

Important patterns from standardized residuals:

- Stop-and-Go is overrepresented in `StraightCross` and strongly
  underrepresented in `RightTurn`.
- Smooth and Steady is overrepresented in `RightTurn`.
- Dynamic Speed Adjustment is also overrepresented in `RightTurn` and
  `LeftTurn`.
- Acceleration-Intensive is underrepresented in `LeftTurn`.

These are associations, not causal conclusions. The analysis does not prove
that a crossing type causes a behavioral profile.

## 7. Tianjin signal-violation interpretation

|   profile_id | profile_name             |   tianjin_trajectories |   violations |   violation_rate_pct |   wilson_95_lower |   wilson_95_upper |   relative_to_overall |
|-------------:|:-------------------------|-----------------------:|-------------:|---------------------:|------------------:|------------------:|----------------------:|
|            1 | Smooth and Steady        |                   2527 |           91 |               3.6011 |            0.0294 |            0.0440 |                1.1829 |
|            2 | Stop-and-Go              |                   1127 |           11 |               0.9760 |            0.0055 |            0.0174 |                0.3206 |
|            3 | Dynamic Speed Adjustment |                    633 |           26 |               4.1074 |            0.0282 |            0.0595 |                1.3492 |
|            4 | Acceleration-Intensive   |                    246 |           10 |               4.0650 |            0.0222 |            0.0732 |                1.3353 |

The profile-by-any-violation association gives:

- chi-square = **22.280**;
- p-value = **5.705e-05**;
- bias-corrected Cramer's V = **0.065**.

The p-value is small, but Cramer's V is only about 0.065. Therefore:

> The violation distribution differs statistically across profiles, but the
> practical association is weak.

Stop-and-Go has the lowest Tianjin violation rate. Dynamic Speed Adjustment
and Acceleration-Intensive have the highest rates, but their confidence
intervals overlap and the Acceleration-Intensive sample is much smaller.

The correct statement is:

> Tianjin metadata suggests a weak association between behavioral profile and
> recorded signal violations.

The incorrect statement is:

> The clustering model accurately predicts signal violations.

## 8. Decision-support matrix

|   profile_id | profile_name             |   dynamic_maneuver_priority |   signal_queue_priority |   speed_adjustment_priority |   tianjin_violation_rate_pct | attention_tier       | recommended_operational_focus                                                                                   |
|-------------:|:-------------------------|----------------------------:|------------------------:|----------------------------:|-----------------------------:|:---------------------|:----------------------------------------------------------------------------------------------------------------|
|            1 | Smooth and Steady        |                       31.25 |                   50.00 |                       25.00 |                         3.60 | Baseline reference   | Use as the baseline flow profile; monitor prevalence and large context shifts rather than intervene by default. |
|            2 | Stop-and-Go              |                       43.75 |                  100.00 |                       66.67 |                         0.98 | Operational priority | Review signal timing, queue formation, stop frequency, and possible spillback conditions.                       |
|            3 | Dynamic Speed Adjustment |                       75.00 |                   50.00 |                       83.33 |                         4.11 | High review priority | Review approach-speed consistency, braking zones, and locations where repeated speed adjustment occurs.         |
|            4 | Acceleration-Intensive   |                      100.00 |                   50.00 |                       75.00 |                         4.07 | High review priority | Prioritize trajectory-quality verification and review high-acceleration/high-jerk maneuvers.                    |

### How the dimensions are calculated

The profile medians are ranked across the four profiles.

- Dynamic maneuver priority uses maximum acceleration, maximum deceleration,
  acceleration variability, and jerk.
- Signal/queue priority uses stop-transition count and stopped-time ratio.
- Speed-adjustment priority uses speed variability, maximum speed, and maximum
  deceleration.
- Tianjin violation enrichment is displayed separately.

These scores are relative rankings among the four discovered profiles. They
are not externally validated risk scores.

## 9. What the system can support

The model can support:

- selecting representative trajectories for inspection;
- monitoring profile prevalence by city or recording;
- identifying queue-oriented, dynamic-adjustment, and high-acceleration
  contexts;
- prioritizing where an analyst should inspect raw trajectories, signal
  operation, or roadway context;
- summarizing large trajectory datasets into interpretable behavioral groups.

The model cannot currently support:

- estimating crash probability;
- proving causal relationships;
- automatically issuing enforcement decisions;
- generalizing Tianjin violation rates to every city;
- replacing traffic-engineering or safety experts.

## 10. Likely lecturer questions

### Why are metadata variables used only after clustering?

Because the research is unsupervised and because equivalent metadata is not
available for all four cities. Using `CrossType` or violation labels as model
inputs would change the research question and introduce inconsistency.

### Why can a Smooth and Steady trajectory still violate a signal?

The profile describes movement dynamics, not legal compliance. A vehicle may
move smoothly while crossing against a prohibited signal.

### Why is a small p-value not enough?

With 4,533 Tianjin trajectories, even a weak association can become
statistically significant. Cramer's V describes the practical strength and is
small for the violation association.

### Why call the matrix decision support rather than risk prediction?

The matrix organizes descriptive evidence and operational priorities. There
is no crash-outcome label and no causal model, so a risk-probability claim
would not be supported.

### Why choose centroid-nearest representative trajectories?

They are the observed members most typical of the model's cluster center in
PCA space. This is more defensible than manually selecting visually dramatic
examples.

## 11. Stage 7 conclusion

The four-profile solution is interpretable and reproducible:

1. Smooth and Steady — baseline continuous flow.
2. Stop-and-Go — queue and signal-operation attention.
3. Dynamic Speed Adjustment — speed-transition and braking attention.
4. Acceleration-Intensive — high-dynamic maneuver review.

Tianjin metadata strengthens interpretation but does not turn the clusters
into supervised safety labels. CrossType has a clearer relationship with the
profiles than signal-violation behavior. The final system should be presented
as an explainable behavioral monitoring and review-prioritization framework,
not as an automated safety-risk predictor.
