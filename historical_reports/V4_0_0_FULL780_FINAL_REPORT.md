# Router V4.0.0 Full-780 Quality-First Report

> Same-campaign internal held-out pilot; not external blind generalization and not a production Router.

## Data and split

- Questions/Documents: **780 / 661**.
- Train rows: **562**.
- Calibration Core: **100**.
- Final Test Core: **100**.
- Testmini-document overlap: **154** (allowed and provenance-only).
- Context-truncated: **102** (trained; runtime Full fallback).
- Complete six-policy panels: **500**; Full/r80/r60 outcomes cover all 780 rows.
- Full acceptable/unacceptable: **528 / 252**; any-policy acceptable/no-safe: **554 / 226**.
- Full-failure/H2O-rescue: **26** (secondary analysis only).
- Frozen split manifest SHA-256: `330534397bf4d7e193424da121c4740a57eea3bb829d8140894c939f8fcb785a`.

## Feature and model contract

- Feature coverage: **780/780** hybrid1936 and z64.
- Reused/generated features: **626 / 154**.
- 10-row feature equivalence max absolute difference: **5.96e-08** (tolerance 1e-05).
- V3 backbone SHA-256: `01d00e076986e8bbae8a5000b9debf62922abc47f119489e79a14e50407fdd73`.
- Normalizer SHA-256: `713f026f3982ec87df49238e988e084a190f79487c0f8ad099af23a595110048`.
- Model: frozen BGE + frozen V3 `1936->128->64`; `z64+structural16->Linear(80,6)` failure head and `structural16->Linear(16,5)` Peak-KV cost head.
- Training: **562** rows, CUDA, 50 epochs, five grouped-OOF heads plus a five-seed ensemble; masked missing policies.

## Calibration-only frozen rule

- Calibration acceptable rate: **67.00%**; compression-induced failures: **0**.
- Calibration H2O route/full fallback: **13.00% / 87.00%**.
- Calibration safe aggregate Peak-KV saving: **0.53%**.
- Frozen routed policy mix: **FULL=87, R060=13**.
- Truncation, protocol mismatch, invalid/OOD/high-uncertainty input fall back to Full; H2O rescue is disabled.
- Final Test outcomes were not used to select or freeze this rule.

## Final Test routing

| Method | Acceptable | Safe aggregate KV saving | H2O route | Compression failures |
|---|---:|---:|---:|---:|
| Fixed Full | 65.00% | 0.00% | 0.00% | 0 |
| Fixed r020 | 58.00% | 21.84% | 100.00% | 9 |
| Fixed r040 | 57.00% | 18.86% | 100.00% | 9 |
| Fixed r060 | 57.00% | 15.48% | 100.00% | 9 |
| Fixed r080 | 62.00% | 14.38% | 100.00% | 6 |
| Fixed r100 | 62.00% | 5.87% | 100.00% | 3 |
| Learned Router | 65.00% | 0.33% | 7.00% | 0 |
| Oracle | 68.00% | 25.04% | 62.00% | 0 |

- Acceptable delta vs Fixed Full: **+0.00 pp**.
- Added errors vs Fixed Full: **0**.
- Safe Oracle opportunity capture: **1.31%**.
- Final gates passed: **False**.
- Learned policy distribution: **FULL=93, R060=7**; Full fallback **93.00%**.
- Oracle policy distribution: **FULL=38, R020=7, R040=10, R060=4, R080=14, R100=27**.

### Pre-registered acceptance gates

| Gate | Result |
|---|---:|
| Acceptable-rate regret <= 1 pp | True |
| Added compression errors within budget | True |
| Truncated rows all fall back to Full | True |
| Safe aggregate saving > 0 | True |
| H2O route rate >= 10% | False |

## Failure prediction

| Policy | Failure Recall | Failure Precision | F1 | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 85.71% | 37.04% | 0.517 | 0.698 | 0.617 | 0.192 |
| SAFE_H2O_R020_I32 | 100.00% | 42.00% | 0.592 | 0.642 | 0.626 | 0.228 |
| SAFE_H2O_R040_I32 | 100.00% | 43.00% | 0.601 | 0.664 | 0.653 | 0.228 |
| SAFE_H2O_R060_I32 | 97.67% | 46.67% | 0.632 | 0.684 | 0.680 | 0.224 |
| SAFE_H2O_R080_I32 | 100.00% | 38.00% | 0.551 | 0.651 | 0.621 | 0.207 |
| SAFE_H2O_R100_I32 | 100.00% | 38.00% | 0.551 | 0.675 | 0.590 | 0.210 |

## Final Test slices

| Slice | Rows | Full acceptable | Router acceptable | Router H2O route | Safe aggregate saving |
|---|---:|---:|---:|---:|---:|
| SimpShort | 30 | 76.67% | 76.67% | 23.33% | 2.44% |
| SimpLong | 30 | 60.00% | 60.00% | 0.00% | 0.00% |
| CompShort | 30 | 76.67% | 76.67% | 0.00% | 0.00% |
| CompLong | 10 | 10.00% | 10.00% | 0.00% | 0.00% |
| truncated | 12 | 16.67% | 16.67% | 0.00% | 0.00% |
| not_truncated | 88 | 71.59% | 71.59% | 7.95% | 0.46% |
| testmini_overlap | 18 | 77.78% | 77.78% | 22.22% | 1.05% |
| not_testmini_overlap | 82 | 62.20% | 62.20% | 3.66% | 0.21% |
| full_acceptable | 65 | 100.00% | 100.00% | 9.23% | 0.65% |
| full_unacceptable | 35 | 0.00% | 0.00% | 2.86% | 0.00% |

## Routing confusion matrix

Rows are Oracle policies; columns are Learned Router policies.

| Oracle \ Router | Full | r100 | r80 | r60 | r40 | r20 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 36 | 0 | 0 | 2 | 0 | 0 |
| r100 | 23 | 0 | 0 | 4 | 0 | 0 |
| r80 | 13 | 0 | 0 | 1 | 0 | 0 |
| r60 | 4 | 0 | 0 | 0 | 0 | 0 |
| r40 | 10 | 0 | 0 | 0 | 0 | 0 |
| r20 | 7 | 0 | 0 | 0 | 0 | 0 |

## Verification and status

- Regression tests recorded: V4 **41**, V3.0 **24**, V3 **51**; failures **0**.
- Data audit, 780/780 feature coverage, deterministic grouped split, frozen rule, Final Test open-once markers and artifact manifest are present.
- `FULL780_EXPERIMENT_COMPLETE.json` is intentionally absent because the H2O route-rate gate failed. The run is marked `FULL780_EXPERIMENT_FINISHED_WITH_LIMITS.json`.

## Interpretation

- The quality-first objective succeeded on this internal test: the Router matched Fixed Full at 65% acceptable with zero added compression errors.
- Compression is real but small: 7% of rows routed to r60, 0.33% aggregate safe Peak-KV saving, and 1.31% of Oracle safe saving opportunity captured.
- The experiment did not pass its complete acceptance contract because 7% H2O routing is below the frozen 10% minimum. No post-test threshold adjustment or rerun was performed.
- Correctness is measured as absolute answer acceptability; input truncation is reported separately.
- Full-failure/H2O-rescue rows are analysis-only; the deployed decision rule does not attempt rescue.
- Testmini-document overlap was explicitly allowed for this run, so this result must not be presented as testmini or external generalization.
- Unsafe compressed choices receive zero safe saving.
- `routing_enabled=false`, `production_checkpoint=false`, and `formal_result_eligible=false` remain in force.
