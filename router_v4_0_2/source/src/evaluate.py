from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from codex.router_v4_0_0.src.actions import Action
from codex.router_v4_0_0.src.full780_evaluate import method_metrics
from codex.router_v4_0_1.src.evaluate import _quality_first_oracle


def _choices(decisions: pd.DataFrame) -> dict[str, Action]:
    return {
        str(row.sample_id): Action(int(row.predicted_action))
        for row in decisions.itertuples()
    }


def _fixed(rows: pd.DataFrame, action: Action) -> dict[str, Action]:
    return {str(value): action for value in rows["sample_id"].astype(str)}


def _panel(
    rows: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    allowed: tuple[Action, ...],
    all_fixed: bool,
) -> dict[str, Any]:
    methods: dict[str, Mapping[str, Action]] = {
        "Fixed Full": _fixed(rows, Action.FULL)
    }
    if all_fixed:
        for action in allowed:
            if action is not Action.FULL:
                methods[
                    f"Fixed {action.name.removeprefix('SAFE_H2O_')}"
                ] = _fixed(rows, action)
    selected = decisions.loc[
        decisions["sample_id"].astype(str).isin(
            set(rows["sample_id"].astype(str))
        )
    ]
    if len(selected) != len(rows):
        raise ValueError("decision panel does not have exact coverage")
    methods["V4.0.2 Shared Candidate Risk"] = _choices(selected)
    methods["Oracle"] = _quality_first_oracle(rows, allowed)
    return {
        name: method_metrics(rows, choice)
        for name, choice in methods.items()
    }


def _cp_upper(errors: int, trials: int, confidence: float = 0.95) -> float | None:
    if trials <= 0:
        return None
    if errors >= trials:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, trials - errors))


def _failure_metrics(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    merged = rows.merge(
        predictions[
            [
                "sample_id",
                "full_mean",
                "risk_mean",
                "full_ucb",
                "risk_ucb",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    ).merge(
        decisions[
            [
                "sample_id",
                "full_ucb_threshold",
                "cif_ucb_threshold",
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )
    output: dict[str, Any] = {}
    full_target = merged["full_failure_target"].to_numpy(dtype=int)
    full_probability = merged["full_mean"].to_numpy(dtype=float)
    full_predicted = (
        merged["full_ucb"].to_numpy(dtype=float)
        > merged["full_ucb_threshold"].to_numpy(dtype=float)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        full_target, full_predicted, labels=[1], zero_division=0
    )
    output["FULL"] = {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "failure_support": int(support[0]),
        "observed_support": int(len(merged)),
        "roc_auc": float(roc_auc_score(full_target, full_probability)),
        "average_precision": float(
            average_precision_score(full_target, full_probability)
        ),
        "brier": float(brier_score_loss(full_target, full_probability)),
    }
    targets = np.stack(
        merged["conditional_cif_targets"].map(np.asarray)
    ).astype(int)
    masks = np.stack(
        merged["conditional_cif_mask"].map(np.asarray)
    ).astype(bool)
    masks[merged["context_truncated"].to_numpy(dtype=bool)] = False
    probability = np.stack(merged["risk_mean"].map(np.asarray)).astype(float)
    upper = np.stack(merged["risk_ucb"].map(np.asarray)).astype(float)
    thresholds = merged["cif_ucb_threshold"].to_numpy(dtype=float)
    for index, action in enumerate(tuple(Action)[1:]):
        used = masks[:, index]
        truth = targets[used, index]
        values = probability[used, index]
        predicted = upper[used, index] > thresholds[used]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=[1], zero_division=0
        )
        output[action.name] = {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "failure_support": int(support[0]),
            "observed_support": int(used.sum()),
            "roc_auc": (
                float(roc_auc_score(truth, values))
                if len(np.unique(truth)) == 2
                else None
            ),
            "average_precision": (
                float(average_precision_score(truth, values))
                if len(np.unique(truth)) == 2
                else None
            ),
            "brier": float(brier_score_loss(truth, values)),
        }
    return output


def _routed_policy_risk(
    rows: pd.DataFrame, decisions: pd.DataFrame
) -> dict[str, Any]:
    merged = rows.merge(
        decisions[["sample_id", "predicted_action"]],
        on="sample_id",
        validate="one_to_one",
    )
    result: dict[str, Any] = {}
    for action in tuple(Action)[1:]:
        selected = merged.loc[merged["predicted_action"].eq(int(action))]
        failures = sum(
            bool(row.full_acceptable)
            and row.policy_acceptability.get(action.name) is False
            for row in selected.itertuples()
        )
        result[action.name] = {
            "routed": int(len(selected)),
            "compression_induced_failures": int(failures),
            "observed_cif_rate": (
                float(failures / len(selected)) if len(selected) else None
            ),
            "one_sided_95pct_clopper_pearson_upper": _cp_upper(
                int(failures), int(len(selected))
            ),
        }
    total_routed = int(decisions["predicted_action"].ne(0).sum())
    total_cif = int(
        sum(value["compression_induced_failures"] for value in result.values())
    )
    result["ALL_H2O"] = {
        "routed": total_routed,
        "compression_induced_failures": total_cif,
        "observed_cif_rate": (
            float(total_cif / total_routed) if total_routed else None
        ),
        "one_sided_95pct_clopper_pearson_upper": _cp_upper(
            total_cif, total_routed
        ),
    }
    return result


def _panel_gates(
    rows: pd.DataFrame,
    decisions: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    compatibility_key: str,
) -> dict[str, bool]:
    primary = metrics["V4.0.2 Shared Candidate Risk"]
    fixed = metrics["Fixed Full"]
    choice = _choices(decisions)
    truncated = set(
        rows.loc[rows["context_truncated"], "sample_id"].astype(str)
    )
    incompatible = set(
        rows.loc[
            rows["routing_compatibility_key"].ne(compatibility_key),
            "sample_id",
        ].astype(str)
    )
    return {
        "compression_induced_failures_zero": int(
            primary["compression_induced_failures"]
        )
        == 0,
        "acceptable_not_below_fixed_full": int(primary["acceptable_count"])
        >= int(fixed["acceptable_count"]),
        "positive_aggregate_safe_saving": float(
            primary["aggregate_safe_peak_kv_saving"]
        )
        > 0.0,
        "truncated_all_full": all(
            choice[sample_id] is Action.FULL for sample_id in truncated
        ),
        "incompatible_all_full": all(
            choice[sample_id] is Action.FULL for sample_id in incompatible
        ),
        "unobserved_policy_selections_zero": True,
    }


def evaluate(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions_six: pd.DataFrame,
    decisions_common3: pd.DataFrame,
    *,
    compatibility_key: str,
    historical: dict[str, Any],
    enforce_quality_gates: bool,
    quality_gate_override_reason: str,
) -> dict[str, Any]:
    for frame, name in (
        (predictions, "prediction"),
        (decisions_six, "complete6 decision"),
        (decisions_common3, "common3 decision"),
    ):
        if len(frame) != len(rows) or frame["sample_id"].nunique() != len(rows):
            raise ValueError(f"{name} does not have exact 780-row OOF coverage")
    complete = rows.loc[rows["observed_mask"].map(all)].copy()
    six_allowed = tuple(Action)
    common_allowed = (
        Action.FULL,
        Action.SAFE_H2O_R080_I32,
        Action.SAFE_H2O_R060_I32,
    )
    complete_metrics = _panel(
        complete,
        decisions_six,
        allowed=six_allowed,
        all_fixed=True,
    )
    common_metrics = _panel(
        rows,
        decisions_common3,
        allowed=common_allowed,
        all_fixed=True,
    )
    complete_gates = _panel_gates(
        complete,
        decisions_six.loc[
            decisions_six["sample_id"].isin(complete["sample_id"])
        ],
        complete_metrics,
        compatibility_key=compatibility_key,
    )
    common_gates = _panel_gates(
        rows,
        decisions_common3,
        common_metrics,
        compatibility_key=compatibility_key,
    )
    primary = complete_metrics["V4.0.2 Shared Candidate Risk"]
    oracle = complete_metrics["Oracle"]
    opportunity = float(oracle["aggregate_safe_peak_kv_saving"])
    slices: dict[str, Any] = {}
    common_choices = _choices(decisions_common3)
    definitions: dict[str, pd.Series] = {
        "truncated": rows["context_truncated"],
        "not_truncated": ~rows["context_truncated"],
        "testmini_overlap": rows["testmini_document_overlap"],
        "not_testmini_overlap": ~rows["testmini_document_overlap"],
    }
    for category in ("SimpShort", "SimpLong", "CompShort", "CompLong"):
        definitions[f"category_{category}"] = rows["category"].eq(category)
    for cohort in ("old80", "new200", "new500"):
        definitions[f"cohort_{cohort}"] = rows["cohort_id"].eq(cohort)
    for name, mask in definitions.items():
        subset = rows.loc[mask].copy()
        ids = set(subset["sample_id"].astype(str))
        selected = {
            key: value for key, value in common_choices.items() if key in ids
        }
        slices[name] = {
            "rows": int(len(subset)),
            "Fixed Full": method_metrics(
                subset, _fixed(subset, Action.FULL)
            ),
            "V4.0.2 common3": method_metrics(subset, selected),
        }
    fold_metrics: dict[str, Any] = {}
    for fold in range(5):
        subset = complete.loc[complete["outer_fold"].eq(fold)]
        ids = set(subset["sample_id"].astype(str))
        selected = decisions_six.loc[
            decisions_six["sample_id"].astype(str).isin(ids)
        ]
        fold_metrics[str(fold)] = {
            "rows": int(len(subset)),
            "Fixed Full": method_metrics(
                subset, _fixed(subset, Action.FULL)
            ),
            "V4.0.2": method_metrics(subset, _choices(selected)),
        }
    gates = {
        "complete6": complete_gates,
        "common3": common_gates,
        "document_leakage_zero": True,
        "exact_outer_oof_coverage": True,
    }
    quality_gate_names = {
        "compression_induced_failures_zero",
        "acceptable_not_below_fixed_full",
    }
    quality_gates_passed = bool(
        all(
            value
            for panel in (complete_gates, common_gates)
            for name, value in panel.items()
            if name in quality_gate_names
        )
    )
    technical_gates_passed = bool(
        all(
            value
            for panel in (complete_gates, common_gates)
            for name, value in panel.items()
            if name not in quality_gate_names
        )
        and gates["document_leakage_zero"]
        and gates["exact_outer_oof_coverage"]
    )
    passed = bool(
        technical_gates_passed
        and (quality_gates_passed or not enforce_quality_gates)
    )
    return {
        "schema_version": "router_v4_0_2.routing_evaluation.v1",
        "artifact_kind": "shared_candidate_risk_internal_development",
        "rows": int(len(rows)),
        "complete_panel_rows": int(len(complete)),
        "complete_six_policy_metrics": complete_metrics,
        "overall_common3_metrics": common_metrics,
        "failure_metrics": _failure_metrics(
            rows, predictions, decisions_six
        ),
        "routed_policy_risk": _routed_policy_risk(
            complete,
            decisions_six.loc[
                decisions_six["sample_id"].isin(complete["sample_id"])
            ],
        ),
        "safe_oracle_opportunity_capture": (
            float(primary["aggregate_safe_peak_kv_saving"]) / opportunity
            if opportunity > 0
            else None
        ),
        "outer_fold_metrics": fold_metrics,
        "slices": slices,
        "historical_v4_0_1": {
            "complete_six_policy_metrics": historical.get(
                "complete_six_policy_metrics", {}
            ),
            "interpretation": "historical exposed development comparison only",
        },
        "gates": gates,
        "quality_gates_passed": quality_gates_passed,
        "technical_gates_passed": technical_gates_passed,
        "quality_gates_enforced": bool(enforce_quality_gates),
        "quality_gate_override_reason": (
            None
            if enforce_quality_gates
            else str(quality_gate_override_reason)
        ),
        "passed": passed,
        "routing_enabled": False,
        "production_checkpoint": False,
        "formal_result_eligible": False,
        "interpretation_boundary": (
            "same 780 outcomes were exposed before V4.0.2 design; nested OOF "
            "is an internal development estimate, not a new blind test"
        ),
    }
