from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from codex.router_v4_0_0.src.actions import Action


@dataclass(frozen=True)
class DecisionConfig:
    full_ucb_threshold: float
    cif_ucb_threshold: float
    min_expected_saving: float
    max_bootstrap_std: float
    tie_tolerance: float
    routing_compatibility_key: str


def choose_action(
    *,
    full_ucb: float,
    full_std: float,
    risk_ucb: np.ndarray,
    risk_std: np.ndarray,
    cost_mean: np.ndarray,
    cost_std: np.ndarray,
    compatibility_key: str,
    context_truncated: bool,
    config: DecisionConfig,
    allowed_actions: Iterable[Action],
    features_valid: bool = True,
) -> tuple[Action, str]:
    if (
        not features_valid
        or compatibility_key != config.routing_compatibility_key
    ):
        return Action.FULL, "invalid_or_incompatible"
    if context_truncated:
        return Action.FULL, "context_truncated"
    if (
        not np.isfinite(full_ucb)
        or not np.isfinite(full_std)
        or full_ucb > config.full_ucb_threshold
        or full_std > config.max_bootstrap_std
    ):
        return Action.FULL, "full_not_confidently_safe"
    risk = np.asarray(risk_ucb, dtype=float)
    rstd = np.asarray(risk_std, dtype=float)
    cost = np.asarray(cost_mean, dtype=float)
    cstd = np.asarray(cost_std, dtype=float)
    if any(values.shape != (5,) for values in (risk, rstd, cost, cstd)):
        raise ValueError("policy decision tensors must have width 5")
    allowed = set(allowed_actions)
    candidates: list[tuple[Action, float]] = []
    for action in Action:
        if action is Action.FULL or action not in allowed:
            continue
        index = int(action) - 1
        if (
            not np.isfinite(risk[index])
            or not np.isfinite(rstd[index])
            or risk[index] > config.cif_ucb_threshold
            or rstd[index] > config.max_bootstrap_std
        ):
            continue
        if (
            not np.isfinite(cost[index])
            or not np.isfinite(cstd[index])
            or cstd[index] > config.max_bootstrap_std
            or 1.0 - cost[index] < config.min_expected_saving
        ):
            continue
        candidates.append((action, float(cost[index])))
    if not candidates:
        return Action.FULL, "no_safe_compression_candidate"
    minimum = min(value for _, value in candidates)
    tied = [
        item
        for item in candidates
        if item[1] <= minimum + config.tie_tolerance
    ]
    selected = min(tied, key=lambda item: int(item[0]))[0]
    return selected, "safe_lowest_predicted_peak_kv"


def make_decisions(
    questions: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    *,
    config: DecisionConfig,
    allowed_actions: tuple[Action, ...],
    outer_fold: int,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(questions.itertuples()):
        action, reason = choose_action(
            full_ucb=float(calibrated["full_ucb"][index]),
            full_std=float(calibrated["full_std"][index]),
            risk_ucb=calibrated["risk_ucb"][index],
            risk_std=calibrated["risk_std"][index],
            cost_mean=calibrated["cost_mean"][index],
            cost_std=calibrated["cost_std"][index],
            compatibility_key=str(row.routing_compatibility_key),
            context_truncated=bool(row.context_truncated),
            config=config,
            allowed_actions=allowed_actions,
        )
        output.append(
            {
                "sample_id": str(row.sample_id),
                "document_hash": str(row.document_hash),
                "predicted_action": int(action),
                "predicted_action_name": action.name,
                "decision_reason": reason,
                "full_fallback": action is Action.FULL,
                "outer_fold": int(outer_fold),
                "full_ucb_threshold": config.full_ucb_threshold,
                "cif_ucb_threshold": config.cif_ucb_threshold,
            }
        )
    return pd.DataFrame(output)


def _summary(
    questions: pd.DataFrame, decisions: pd.DataFrame
) -> dict[str, Any]:
    merged = questions.merge(
        decisions[["sample_id", "predicted_action"]],
        on="sample_id",
        validate="one_to_one",
    )
    cif = rescue = unobserved = compressed = 0
    full_total = safe_cost_total = 0.0
    for row in merged.itertuples():
        action = Action(int(row.predicted_action))
        full_cost = float(row.policy_peak_kv_bytes[Action.FULL.name])
        full_total += full_cost
        if action is Action.FULL:
            safe_cost_total += full_cost
            continue
        outcome = row.policy_acceptability.get(action.name)
        cost = row.policy_peak_kv_bytes.get(action.name)
        if outcome is None or cost is None:
            unobserved += 1
            safe_cost_total += full_cost
            continue
        compressed += 1
        cif += int(bool(row.full_acceptable) and not bool(outcome))
        rescue += int((not bool(row.full_acceptable)) and bool(outcome))
        safe_cost_total += float(cost) if bool(outcome) else full_cost
    return {
        "compression_induced_failures": int(cif),
        "h2o_rescue_rows": int(rescue),
        "unobserved_selected_rows": int(unobserved),
        "compressed_rows": int(compressed),
        "aggregate_safe_peak_kv_saving": float(
            1.0 - safe_cost_total / full_total if full_total else 0.0
        ),
    }


def _grid(values: np.ndarray, points: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, points))
    return np.unique(np.concatenate(([0.0], quantiles, [1.0])))


def optimize_zero_cif_rule(
    questions: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    *,
    base_config: DecisionConfig,
    allowed_actions: tuple[Action, ...],
    outer_fold: int,
    grid_points: int,
) -> tuple[DecisionConfig, dict[str, Any]]:
    best: tuple[
        tuple[float, int, float, float], DecisionConfig, dict[str, Any]
    ] | None = None
    for full_threshold in _grid(calibrated["full_ucb"], grid_points):
        for cif_threshold in _grid(calibrated["risk_ucb"], grid_points):
            candidate = DecisionConfig(
                full_ucb_threshold=float(full_threshold),
                cif_ucb_threshold=float(cif_threshold),
                min_expected_saving=base_config.min_expected_saving,
                max_bootstrap_std=base_config.max_bootstrap_std,
                tie_tolerance=base_config.tie_tolerance,
                routing_compatibility_key=base_config.routing_compatibility_key,
            )
            decisions = make_decisions(
                questions,
                calibrated,
                config=candidate,
                allowed_actions=allowed_actions,
                outer_fold=outer_fold,
            )
            summary = _summary(questions, decisions)
            if (
                summary["compression_induced_failures"] != 0
                or summary["h2o_rescue_rows"] != 0
                or summary["unobserved_selected_rows"] != 0
            ):
                continue
            score = (
                float(summary["aggregate_safe_peak_kv_saving"]),
                int(summary["compressed_rows"]),
                -float(cif_threshold),
                -float(full_threshold),
            )
            if best is None or score > best[0]:
                best = (score, candidate, summary)
    if best is None:
        raise RuntimeError("no zero-CIF calibration rule exists")
    return best[1], {
        "schema_version": "router_v4_0_2.zero_cif_rule_summary.v1",
        **best[2],
        "decision": asdict(best[1]),
        "method": "crossfit_bootstrap_ucb_zero_cif_no_rescue",
    }

