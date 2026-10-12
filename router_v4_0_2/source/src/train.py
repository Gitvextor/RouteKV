from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import expand_candidates
from .model import ModelConfig, SharedCandidateEnsemble, save_checkpoint


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    weight_decay: float
    batch_size_questions: int
    batch_size_candidates: int
    epochs: int
    gradient_clip_norm: float
    maximum_pos_weight: float
    cost_loss_weight: float
    device: str


def _stack(series: pd.Series, width: int) -> np.ndarray:
    values = np.stack(series.map(np.asarray)).astype(np.float32)
    if values.shape != (len(series), width):
        raise ValueError(f"tensor must be [N,{width}]")
    return values


def _base_weights(questions: pd.DataFrame) -> np.ndarray:
    counts = questions.groupby("document_hash")["sample_id"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype=np.float32)
    return weights / weights.mean()


def _candidate_weights(
    questions: pd.DataFrame, candidates: pd.DataFrame
) -> np.ndarray:
    document_counts = (
        questions.groupby("document_hash")["sample_id"].count().to_dict()
    )
    candidate_counts = candidates.groupby("sample_id")["candidate_id"].count()
    weights = np.asarray(
        [
            1.0
            / float(document_counts[str(row.document_hash)])
            / float(candidate_counts[str(row.sample_id)])
            for row in candidates.itertuples()
        ],
        dtype=np.float32,
    )
    return weights / weights.mean()


def _bootstrap_weights(
    documents: pd.Series, seeds: tuple[int, ...]
) -> np.ndarray:
    values = documents.astype(str).to_numpy()
    unique = np.asarray(sorted(set(values)))
    output = np.zeros((len(values), len(seeds)), dtype=np.float32)
    for member, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        sampled = rng.choice(unique, size=len(unique), replace=True)
        chosen, counts = np.unique(sampled, return_counts=True)
        lookup = dict(zip(chosen.tolist(), counts.astype(float).tolist()))
        output[:, member] = np.asarray(
            [lookup.get(value, 0.0) for value in values], dtype=np.float32
        )
    return output


def _pos_weight(
    targets: np.ndarray, weights: np.ndarray, maximum: float
) -> float:
    positive = float(weights[targets == 1].sum())
    negative = float(weights[targets == 0].sum())
    if positive <= 0 or negative <= 0:
        raise ValueError("binary training target must contain both classes")
    return float(min(maximum, negative / positive))


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denominator = weight.sum().clamp_min(1.0e-8)
    return (loss * weight).sum() / denominator


def _batches(
    count: int, batch_size: int, *, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    return [
        order[start : start + batch_size]
        for start in range(0, count, batch_size)
    ]


def train_ensemble(
    questions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, pd.DataFrame]:
    if train_config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(train_config.device)
    torch.manual_seed(int(model_config.member_seeds[0]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = SharedCandidateEnsemble(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    qx = torch.from_numpy(_stack(questions["base80"], 80)).to(device)
    qy_np = questions["full_failure_target"].to_numpy(dtype=np.float32)
    qy = torch.from_numpy(qy_np).to(device)
    q_base_np = _base_weights(questions)
    q_boot_np = _bootstrap_weights(
        questions["document_hash"], model_config.member_seeds
    )
    q_weight = torch.from_numpy(q_base_np[:, None] * q_boot_np).to(device)
    full_pos_weight = _pos_weight(
        qy_np, q_base_np, train_config.maximum_pos_weight
    )

    risk_rows = candidates.loc[candidates["cif_mask"]].reset_index(drop=True)
    rx = torch.from_numpy(_stack(risk_rows["risk_input166"], 166)).to(device)
    ry_np = risk_rows["conditional_cif_target"].to_numpy(dtype=np.float32)
    ry = torch.from_numpy(ry_np).to(device)
    r_base_np = _candidate_weights(questions, risk_rows)
    r_boot_np = _bootstrap_weights(
        risk_rows["document_hash"], model_config.member_seeds
    )
    r_weight = torch.from_numpy(r_base_np[:, None] * r_boot_np).to(device)
    risk_pos_weight = _pos_weight(
        ry_np, r_base_np, train_config.maximum_pos_weight
    )

    cost_rows = candidates.loc[candidates["cost_mask"]].reset_index(drop=True)
    cx = torch.from_numpy(_stack(cost_rows["cost_input22"], 22)).to(device)
    cy = torch.from_numpy(
        cost_rows["peak_kv_ratio"].to_numpy(dtype=np.float32)
    ).to(device)
    c_base_np = _candidate_weights(questions, cost_rows)
    c_boot_np = _bootstrap_weights(
        cost_rows["document_hash"], model_config.member_seeds
    )
    c_weight = torch.from_numpy(c_base_np[:, None] * c_boot_np).to(device)

    history: list[dict[str, float | int]] = []
    members = model_config.members
    full_pw = torch.tensor(full_pos_weight, device=device)
    risk_pw = torch.tensor(risk_pos_weight, device=device)
    for epoch in range(train_config.epochs):
        model.train()
        full_total = risk_total = cost_total = 0.0
        full_steps = risk_steps = cost_steps = 0
        for indices in _batches(
            len(questions),
            train_config.batch_size_questions,
            seed=10000 + epoch,
        ):
            batch = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.full_logits(qx[batch])
            target = qy[batch, None].expand(-1, members)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=full_pw
            )
            loss = _weighted_mean(losses, q_weight[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm
            )
            optimizer.step()
            full_total += float(loss.detach().cpu())
            full_steps += 1
        for indices in _batches(
            len(risk_rows),
            train_config.batch_size_candidates,
            seed=20000 + epoch,
        ):
            batch = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.risk_logits(rx[batch])
            target = ry[batch, None].expand(-1, members)
            losses = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=risk_pw
            )
            loss = _weighted_mean(losses, r_weight[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm
            )
            optimizer.step()
            risk_total += float(loss.detach().cpu())
            risk_steps += 1
        for indices in _batches(
            len(cost_rows),
            train_config.batch_size_candidates,
            seed=30000 + epoch,
        ):
            batch = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model.cost_ratios(cx[batch])
            target = cy[batch, None].expand(-1, members)
            losses = F.huber_loss(predicted, target, reduction="none")
            loss = train_config.cost_loss_weight * _weighted_mean(
                losses, c_weight[batch]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm
            )
            optimizer.step()
            cost_total += float(loss.detach().cpu())
            cost_steps += 1
        history.append(
            {
                "epoch": epoch + 1,
                "full_loss": full_total / max(full_steps, 1),
                "risk_loss": risk_total / max(risk_steps, 1),
                "weighted_cost_loss": cost_total / max(cost_steps, 1),
                "full_pos_weight": full_pos_weight,
                "risk_pos_weight": risk_pos_weight,
            }
        )
    path = save_checkpoint(
        output_path,
        model,
        metadata={
            **metadata,
            "question_rows": int(len(questions)),
            "risk_candidate_rows": int(len(risk_rows)),
            "cost_candidate_rows": int(len(cost_rows)),
        },
    )
    return path, pd.DataFrame(history)


@torch.no_grad()
def predict_ensemble(
    model: SharedCandidateEnsemble,
    questions: pd.DataFrame,
    *,
    coordinates: list[float],
    device: str,
) -> pd.DataFrame:
    model = model.to(device)
    model.eval()
    base = torch.from_numpy(_stack(questions["base80"], 80)).to(device)
    inference = expand_candidates(
        questions, coordinates, observed_only=False
    )
    risk = torch.from_numpy(_stack(inference["risk_input166"], 166)).to(device)
    cost = torch.from_numpy(_stack(inference["cost_input22"], 22)).to(device)
    full_members = torch.sigmoid(model.full_logits(base)).cpu().numpy()
    risk_members = (
        torch.sigmoid(model.risk_logits(risk))
        .cpu()
        .numpy()
        .reshape(len(questions), 5, model.config.members)
    )
    cost_members = (
        model.cost_ratios(cost)
        .cpu()
        .numpy()
        .reshape(len(questions), 5, model.config.members)
    )
    result = questions[
        [
            "sample_id",
            "document_hash",
            "outer_fold",
            "routing_compatibility_key",
        ]
    ].copy()
    result["raw_full_member_probabilities"] = [
        row.tolist() for row in full_members
    ]
    result["raw_cif_member_probabilities"] = [
        row.tolist() for row in risk_members
    ]
    result["predicted_cost_member_ratios"] = [
        row.tolist() for row in cost_members
    ]
    return result
