# Router V4.0.1 Nested-OOF Quality-First Report

> Same-campaign internal nested OOF pilot. The former V4.0.0 Final Test is exposed and is not reused as an unbiased test.

## Data and evidence boundary

- Questions/Documents: **780 / 661**.
- Complete six-policy panels: **500**.
- Conditional CIF counts r100/r80/r60/r40/r20: **[11, 33, 48, 33, 38]**.
- Outer Fold rows: **{'0': 156, '1': 156, '2': 156, '3': 156, '4': 156}**; document leakage: **0**.
- Every row is tested once by an ensemble that did not train or calibrate on its document.

## Primary complete-panel routing

| Method | Acceptable | H2O route | Safe aggregate KV saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 66.60% | 0.00% | 0.00% | 0 |
| Fixed R020_I32 | 61.40% | 100.00% | 21.25% | 38 |
| Fixed R040_I32 | 62.20% | 100.00% | 21.48% | 33 |
| Fixed R060_I32 | 63.60% | 100.00% | 18.89% | 27 |
| Fixed R080_I32 | 63.80% | 100.00% | 13.02% | 22 |
| Fixed R100_I32 | 65.80% | 100.00% | 6.56% | 11 |
| Oracle | 66.60% | 47.60% | 19.90% | 0 |
| frozen_conditional_mlp | 65.40% | 30.60% | 1.95% | 7 |
| legacy_absolute_linear | 66.00% | 35.60% | 2.45% | 5 |
| partial_finetune_conditional | 65.40% | 34.00% | 2.57% | 7 |

- Primary: **partial_finetune_conditional**.
- Safe Oracle opportunity capture: **12.94%**.
- All Primary gates passed: **False**.
- Offline risk-coverage points: **53** (diagnostic only; not used to replace the frozen Primary rule).

## Acceptance gates

| Gate | Result |
|---|---:|
| acceptable_matches_fixed_full | False |
| aggregate_safe_saving_at_least_two_percent | True |
| compression_induced_failures_zero | False |
| h2o_route_rate_at_least_ten_percent | True |
| truncated_all_full | True |

## Conditional failure prediction

| Policy | Precision | Recall | F1 | AUROC | AUPRC | Brier | Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL | 40.38% | 75.00% | 0.525 | 0.698 | 0.572 | 0.190 | 252 |
| SAFE_H2O_R020_I32 | 13.99% | 71.05% | 0.234 | 0.689 | 0.207 | 0.285 | 38 |
| SAFE_H2O_R040_I32 | 12.78% | 69.70% | 0.216 | 0.666 | 0.181 | 0.262 | 33 |
| SAFE_H2O_R060_I32 | 12.45% | 68.75% | 0.211 | 0.694 | 0.186 | 0.258 | 48 |
| SAFE_H2O_R080_I32 | 7.73% | 45.45% | 0.132 | 0.606 | 0.088 | 0.187 | 33 |
| SAFE_H2O_R100_I32 | 4.88% | 18.18% | 0.077 | 0.539 | 0.050 | 0.117 | 11 |

## Verification

- Regression tests: V4.0.1 **13**, V4.0.0 **41**, V3.0 **24**, V3 **51**; total **129**, failures **0**.
- CUDA tiny smoke passed; every registered variant has exactly 780 OOF predictions and zero document leakage.
- Artifact manifest hash verification passed. The all-780 research candidate is blocked because the Primary safety gates failed.

## Interpretation

- Primary safety is measured by Full-correct rows that become incorrect after compression; Full-failure rescue is prohibited.
- Missing r100/r40/r20 outcomes are masked. All-780 metrics use the common Full/r80/r60 action subset.
- Context-truncated, protocol-mismatched, invalid, OOD, and high-uncertainty inputs fall back to Full.
- This is not external blind generalization or a production Router.
- `routing_enabled=false`, `production_checkpoint=false`, and `formal_result_eligible=false` remain in force.
