# Router V4.0.2 Shared Candidate Risk Report

> Same-data internal development Nested OOF. This is not a blind or production result.

## Data and architecture

- Questions/Documents: **780 / 661**.
- Complete six-policy panels: **500**.
- Conditional CIF counts: **[11, 33, 48, 33, 38]**.
- Frozen BGE/V3 z64; shared question-policy linear risk scorer; 10-member document bootstrap; cross-fitted calibration.

## Primary Complete-6 result

| Method | Acceptable | H2O route | Safe aggregate saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 66.60% | 0.00% | 0.00% | 0 |
| Fixed R100_I32 | 65.80% | 100.00% | 6.56% | 11 |
| Fixed R080_I32 | 63.80% | 100.00% | 13.02% | 22 |
| Fixed R060_I32 | 63.60% | 100.00% | 18.89% | 27 |
| Fixed R040_I32 | 62.20% | 100.00% | 21.48% | 33 |
| Fixed R020_I32 | 61.40% | 100.00% | 21.25% | 38 |
| V4.0.2 Shared Candidate Risk | 66.20% | 10.40% | 0.67% | 3 |
| Oracle | 66.60% | 47.60% | 19.90% | 0 |

## Complete-6 comparison with frozen V4.0.1 baselines

| Method | Acceptable | H2O route | Safe aggregate saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 66.60% | 0.00% | 0.00% | 0 |
| Fixed R100_I32 | 65.80% | 100.00% | 6.56% | 11 |
| Fixed R080_I32 | 63.80% | 100.00% | 13.02% | 22 |
| Fixed R060_I32 | 63.60% | 100.00% | 18.89% | 27 |
| Fixed R040_I32 | 62.20% | 100.00% | 21.48% | 33 |
| Fixed R020_I32 | 61.40% | 100.00% | 21.25% | 38 |
| legacy_absolute_linear | 66.00% | 35.60% | 2.45% | 5 |
| frozen_conditional_mlp | 65.40% | 30.60% | 1.95% | 7 |
| partial_finetune_conditional | 65.40% | 34.00% | 2.57% | 7 |
| V4.0.2 Shared Candidate Risk | 66.20% | 10.40% | 0.67% | 3 |
| Oracle | 66.60% | 47.60% | 19.90% | 0 |

## Secondary Common-3 result

| Method | Acceptable | H2O route | Safe aggregate saving | CIF |
|---|---:|---:|---:|---:|
| Fixed Full | 67.69% | 0.00% | 0.00% | 0 |
| Fixed R080_I32 | 65.51% | 100.00% | 13.43% | 33 |
| Fixed R060_I32 | 64.23% | 100.00% | 18.90% | 48 |
| V4.0.2 Shared Candidate Risk | 67.31% | 6.92% | 0.48% | 4 |
| Oracle | 67.69% | 47.05% | 15.79% | 0 |

## Safety verdict

- Technical execution gates passed: **True**.
- Original zero-error quality target passed: **False**.
- Quality gates enforced as a training blocker: **False**.
- Override audit reason: **user_authorized_continue_after_nested_oof_review_20260729**.
- Acceptable delta vs Fixed Full: **-0.40 pp**.
- Routed H2O rows / CIF: **52 / 3**.
- One-sided 95% Clopper-Pearson CIF upper bound: **14.24%**.
- Safe Oracle opportunity capture: **3.34%**.
- Policy distribution: **{'FULL': 448, 'SAFE_H2O_R040_I32': 10, 'SAFE_H2O_R080_I32': 24, 'SAFE_H2O_R100_I32': 18}**.
- Full fallback rate: **89.60%**.

The descriptive OOF risk-coverage curve contains a zero-CIF point at 3.80% H2O routing and 0.15% safe saving. This is post-hoc on exposed Outer OOF outcomes and is not used to replace the frozen cross-fitted rules.

## Failure prediction diagnostics

| Policy | AUROC | AUPRC | Recall | Brier | Failure / Observed |
|---|---:|---:|---:|---:|---:|
| FULL | 0.716 | 0.578 | 86.51% | 0.182 | 252 / 780 |
| SAFE_H2O_R020_I32 | 0.677 | 0.162 | 97.06% | 0.297 | 34 / 319 |
| SAFE_H2O_R040_I32 | 0.652 | 0.131 | 86.67% | 0.282 | 30 / 319 |
| SAFE_H2O_R060_I32 | 0.631 | 0.115 | 81.40% | 0.264 | 43 / 505 |
| SAFE_H2O_R080_I32 | 0.584 | 0.078 | 61.29% | 0.200 | 31 / 505 |
| SAFE_H2O_R100_I32 | 0.396 | 0.027 | 50.00% | 0.097 | 10 / 319 |

## Outer-fold stability (Complete-6)

| Outer fold | Rows | Full acceptable | Router acceptable | H2O route | Safe saving | CIF |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 68.00% | 67.00% | 22.00% | 1.95% | 1 |
| 1 | 100 | 70.00% | 70.00% | 8.00% | 0.03% | 0 |
| 2 | 100 | 66.00% | 66.00% | 1.00% | 0.00% | 0 |
| 3 | 100 | 63.00% | 61.00% | 13.00% | 0.75% | 2 |
| 4 | 100 | 66.00% | 67.00% | 8.00% | 0.65% | 0 |

## Common-3 slices

| Slice | Rows | Full acceptable | Router acceptable | H2O route | Safe saving | CIF |
|---|---:|---:|---:|---:|---:|---:|
| category_CompLong | 78 | 16.67% | 16.67% | 0.00% | 0.00% | 0 |
| category_CompShort | 234 | 76.50% | 76.50% | 3.85% | 0.06% | 0 |
| category_SimpLong | 234 | 59.40% | 59.40% | 0.00% | 0.00% | 0 |
| category_SimpShort | 234 | 84.19% | 82.91% | 19.23% | 3.40% | 4 |
| cohort_new200 | 200 | 69.50% | 68.50% | 6.00% | 0.24% | 2 |
| cohort_new500 | 500 | 66.60% | 66.20% | 7.60% | 0.50% | 2 |
| cohort_old80 | 80 | 70.00% | 71.25% | 5.00% | 0.99% | 0 |
| not_testmini_overlap | 626 | 68.05% | 67.57% | 7.19% | 0.43% | 4 |
| not_truncated | 678 | 74.48% | 74.04% | 7.96% | 0.69% | 4 |
| testmini_overlap | 154 | 66.23% | 66.23% | 5.84% | 0.71% | 0 |
| truncated | 102 | 22.55% | 22.55% | 0.00% | 0.00% | 0 |

## All-780 research-only bundle

- Checkpoint: `research_candidate\ensemble.pt`.
- SHA256: `2551427a1cae5bf095b39b7a1b6ccc52719b07c27e7b72774b5062c6f425ca7c`.
- Training rows/epochs/bootstrap members: **780 / 50 / 10**.
- Fitted only after the explicit quality-gate override.
- It is not re-evaluated on the same 780 training rows; Nested OOF remains the performance estimate.
- `research_only=true`, `routing_enabled=false`, `production_checkpoint=false`.

## Gate details

```json
{
  "common3": {
    "acceptable_not_below_fixed_full": false,
    "compression_induced_failures_zero": false,
    "incompatible_all_full": true,
    "positive_aggregate_safe_saving": true,
    "truncated_all_full": true,
    "unobserved_policy_selections_zero": true
  },
  "complete6": {
    "acceptable_not_below_fixed_full": false,
    "compression_induced_failures_zero": false,
    "incompatible_all_full": true,
    "positive_aggregate_safe_saving": true,
    "truncated_all_full": true,
    "unobserved_policy_selections_zero": true
  },
  "document_leakage_zero": true,
  "exact_outer_oof_coverage": true
}
```

## Interpretation

- Correctness is evaluated relative to Fixed Full; H2O rescue is not used to justify compression.
- H2O route rate is reported but is not a pass gate.
- The user-authorized override permits research-only all-data training; it does not erase the observed CIF or make the router safe.
- The 780 outcomes were already exposed before V4.0.2 was designed, so this result cannot establish blind generalization.
- `routing_enabled=false`, `production_checkpoint=false`, and `formal_result_eligible=false` remain mandatory.
