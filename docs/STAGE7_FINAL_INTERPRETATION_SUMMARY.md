# Stage 7 — Final Interpretation Summary

## Final model

- Preprocessing: 1%–99% winsorization on eight continuous features,
  RobustScaler, PCA.
- PCA retained variance: 91.71%.
- Model: K-Means, k=4.
- Eligible trajectories: 19,948.
- Coverage: 100%.
- Validation decision: Validated with context sensitivity.

## Final behavioral profiles

|   profile_id | profile_name             | behavioral_definition                                                                                                | primary_evidence                                                                                                                     | decision_support_role                                                  | important_limitation                                                                                                      |
|-------------:|:-------------------------|:---------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
|            1 | Smooth and Steady        | Continuous movement with relatively low speed variability, acceleration intensity, braking intensity, and jerk.      | Median stopped-time ratio = 0; lowest median acceleration and deceleration intensity among the four profiles.                        | Baseline reference for normal continuous-flow conditions.              | Smooth does not mean risk-free; Tianjin violation rate is 3.60%, so profile name is not a safety label.                   |
|            2 | Stop-and-Go              | Low mean speed with repeated or extended stopping and high within-trajectory speed variation.                        | Median stop-transition count = 1 and stopped-time ratio = 0.591; lowest mean speed.                                                  | Signal timing, queue formation, delay, and spillback review.           | The profile can reflect expected red-signal behavior rather than unsafe driving.                                          |
|            3 | Dynamic Speed Adjustment | Continuous movement with stronger acceleration, braking, acceleration variability, and jerk than the smooth profile. | High median acceleration variability and jerk, with no typical stopped-time contribution.                                            | Approach-speed consistency and braking-zone investigation.             | Dynamic adjustment is descriptive and does not by itself prove aggressive or unsafe driving.                              |
|            4 | Acceleration-Intensive   | A smaller profile distinguished by very high positive acceleration, acceleration variability, and jerk.              | Median maximum acceleration = 6.843 m/s²; robust score for maximum acceleration is nearly five global IQRs above the overall median. | High-acceleration maneuver review and trajectory-quality verification. | The profile is not a crash-risk class; extreme maneuvers and tracking artifacts must be distinguished through raw review. |

## Tianjin metadata findings

- Profile and CrossType association: Cramer's V =
  0.217.
- Profile and any signal violation association: Cramer's V =
  0.065.
- The CrossType relationship is materially stronger.
- The signal-violation association is statistically significant but weak.
- Metadata findings apply to Tianjin only.

## Decision-support conclusion

| profile_name             | attention_tier       | recommended_operational_focus                                                                                   |
|:-------------------------|:---------------------|:----------------------------------------------------------------------------------------------------------------|
| Smooth and Steady        | Baseline reference   | Use as the baseline flow profile; monitor prevalence and large context shifts rather than intervene by default. |
| Stop-and-Go              | Operational priority | Review signal timing, queue formation, stop frequency, and possible spillback conditions.                       |
| Dynamic Speed Adjustment | High review priority | Review approach-speed consistency, braking zones, and locations where repeated speed adjustment occurs.         |
| Acceleration-Intensive   | High review priority | Prioritize trajectory-quality verification and review high-acceleration/high-jerk maneuvers.                    |

The decision-support matrix is descriptive. It is not a crash-risk probability,
causal safety model, or enforcement tool.
