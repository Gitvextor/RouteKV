from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class BinaryPlatt:
    coefficient: float
    intercept: float


@dataclass(frozen=True)
class SharedPolicyPlatt:
    coefficients: tuple[float, ...]
    intercept: float


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(
        np.asarray(probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6
    )
    return np.log(values / (1.0 - values))


def _full_members(predictions: pd.DataFrame) -> np.ndarray:
    values = np.stack(
        predictions["raw_full_member_probabilities"].map(np.asarray)
    ).astype(np.float64)
    if values.ndim != 2:
        raise ValueError("Full member probabilities must be [N,M]")
    return values


def _risk_members(predictions: pd.DataFrame) -> np.ndarray:
    values = np.stack(
        predictions["raw_cif_member_probabilities"].map(np.asarray)
    ).astype(np.float64)
    if values.ndim != 3 or values.shape[1] != 5:
        raise ValueError("CIF member probabilities must be [N,5,M]")
    return values


def fit_calibrators(
    questions: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[BinaryPlatt, SharedPolicyPlatt]:
    merged = questions.merge(
        predictions,
        on=["sample_id", "document_hash"],
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    full_target = merged["full_failure_target"].to_numpy(dtype=int)
    full_mean = _full_members(merged).mean(axis=1)
    if len(np.unique(full_target)) != 2:
        raise ValueError("Full calibration requires both classes")
    full_model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=3000, random_state=42
    )
    full_model.fit(_logit(full_mean).reshape(-1, 1), full_target)
    full = BinaryPlatt(
        coefficient=float(full_model.coef_[0, 0]),
        intercept=float(full_model.intercept_[0]),
    )

    targets = np.stack(
        merged["conditional_cif_targets"].map(np.asarray)
    ).astype(int)
    masks = np.stack(
        merged["conditional_cif_mask"].map(np.asarray)
    ).astype(bool)
    masks[merged["context_truncated"].to_numpy(dtype=bool)] = False
    risk_mean = _risk_members(merged).mean(axis=2)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row, policy in zip(*np.where(masks)):
        one_hot = np.zeros(5, dtype=np.float64)
        one_hot[policy] = 1.0
        features.append(
            np.concatenate(
                (
                    np.asarray([_logit(risk_mean[row, policy])]),
                    one_hot,
                )
            )
        )
        labels.append(int(targets[row, policy]))
    if len(np.unique(labels)) != 2:
        raise ValueError("CIF calibration requires both classes")
    risk_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=4000,
        random_state=42,
        class_weight="balanced",
    )
    risk_model.fit(np.stack(features), np.asarray(labels))
    risk = SharedPolicyPlatt(
        coefficients=tuple(
            float(value) for value in risk_model.coef_[0]
        ),
        intercept=float(risk_model.intercept_[0]),
    )
    return full, risk


def apply_calibrators(
    predictions: pd.DataFrame,
    full: BinaryPlatt,
    risk: SharedPolicyPlatt,
    *,
    ucb_z: float,
    monotone: bool,
) -> dict[str, np.ndarray]:
    raw_full = _full_members(predictions)
    full_members = 1.0 / (
        1.0 + np.exp(-(full.coefficient * _logit(raw_full) + full.intercept))
    )
    raw_risk = _risk_members(predictions)
    coefficients = np.asarray(risk.coefficients, dtype=np.float64)
    risk_members = np.zeros_like(raw_risk)
    for policy in range(5):
        one_hot = np.zeros((raw_risk.shape[0], raw_risk.shape[2], 5))
        one_hot[:, :, policy] = 1.0
        features = np.concatenate(
            (_logit(raw_risk[:, policy, :])[:, :, None], one_hot), axis=2
        )
        logits = features @ coefficients + risk.intercept
        risk_members[:, policy, :] = 1.0 / (1.0 + np.exp(-logits))
    full_mean = full_members.mean(axis=1)
    full_std = full_members.std(axis=1)
    full_ucb = np.clip(full_mean + float(ucb_z) * full_std, 0.0, 1.0)
    risk_mean = risk_members.mean(axis=2)
    risk_std = risk_members.std(axis=2)
    risk_ucb = np.clip(risk_mean + float(ucb_z) * risk_std, 0.0, 1.0)
    if monotone:
        risk_ucb = np.maximum.accumulate(risk_ucb, axis=1)
    raw_cost = np.stack(
        predictions["predicted_cost_member_ratios"].map(np.asarray)
    ).astype(np.float64)
    return {
        "calibrated_full_member_probabilities": full_members.astype(np.float32),
        "calibrated_cif_member_probabilities": risk_members.astype(np.float32),
        "full_mean": full_mean.astype(np.float32),
        "full_std": full_std.astype(np.float32),
        "full_ucb": full_ucb.astype(np.float32),
        "risk_mean": risk_mean.astype(np.float32),
        "risk_std": risk_std.astype(np.float32),
        "risk_ucb": risk_ucb.astype(np.float32),
        "cost_mean": raw_cost.mean(axis=2).astype(np.float32),
        "cost_std": raw_cost.std(axis=2).astype(np.float32),
    }


def calibration_payload(
    full: BinaryPlatt, risk: SharedPolicyPlatt
) -> dict[str, Any]:
    return {
        "schema_version": "router_v4_0_2.shared_policy_calibration.v1",
        "full_platt": asdict(full),
        "shared_policy_platt": {
            **asdict(risk),
            "coefficients": list(risk.coefficients),
        },
    }

