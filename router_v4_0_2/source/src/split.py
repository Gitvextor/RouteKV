from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from codex.router_v4_0_1.src.split import _group_features


@dataclass(frozen=True)
class GreedyAssignment:
    assignment: np.ndarray
    objective: float


def _stable_noise(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _greedy_balanced_assignment(
    rows: pd.DataFrame, *, folds: int, seed: int
) -> GreedyAssignment:
    groups, features = _group_features(rows)
    values = groups[features].to_numpy(dtype=float)
    totals = values.sum(axis=0)
    target = totals / float(folds)
    scale = np.maximum(target, 1.0)
    cif_indices = [
        features.index(f"cif_{name}")
        for name in ("r100", "r80", "r60", "r40", "r20")
    ]
    rare_scale = np.maximum(totals[cif_indices], 1.0)
    best: tuple[float, np.ndarray] | None = None
    for attempt in range(256):
        attempt_seed = seed + 7919 * attempt
        rare = (
            values[:, cif_indices] / rare_scale[None, :]
        ).sum(axis=1)
        row_size = values[:, features.index("rows")]
        noise = np.asarray(
            [
                _stable_noise(attempt_seed, str(document))
                for document in groups.index
            ]
        )
        order = np.lexsort((noise, -row_size, -rare))
        fold_values = np.zeros((folds, len(features)), dtype=float)
        assigned = np.full(len(groups), -1, dtype=int)
        for position, group_index in enumerate(order):
            scores: list[tuple[float, float, int]] = []
            for fold in range(folds):
                candidate = fold_values.copy()
                candidate[fold] += values[group_index]
                deviation = float(
                    np.square((candidate - target[None, :]) / scale[None, :]).sum()
                )
                empty_bonus = -1.0e-3 if position < folds and not fold_values[fold].any() else 0.0
                scores.append(
                    (
                        deviation + empty_bonus,
                        _stable_noise(
                            attempt_seed,
                            f"{groups.index[group_index]}:{fold}",
                        ),
                        fold,
                    )
                )
            chosen = min(scores)[2]
            assigned[group_index] = chosen
            fold_values[chosen] += values[group_index]
        cif_counts = fold_values[:, cif_indices]
        valid = bool((cif_counts >= 1.0).all())
        objective = float(
            np.square((fold_values - target[None, :]) / scale[None, :]).sum()
        )
        if valid and (best is None or objective < best[0]):
            best = (objective, assigned.copy())
            if attempt >= 15:
                break
    if best is None:
        raise RuntimeError(
            "deterministic grouped greedy split cannot place every CIF class "
            "in every inner fold"
        )
    lookup = dict(zip(groups.index.astype(str), best[1]))
    return GreedyAssignment(
        assignment=rows["document_hash"].astype(str).map(lookup).to_numpy(),
        objective=float(best[0]),
    )


def build_crossfit_manifest(
    questions: pd.DataFrame,
    frozen_outer: pd.DataFrame,
    *,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = frozen_outer[
        ["sample_id", "document_hash", "category", "cohort_id", "outer_fold"]
    ].copy()
    adjusted = questions.copy()
    masks = np.stack(
        adjusted["conditional_cif_mask"].map(np.asarray)
    ).astype(bool)
    masks[adjusted["context_truncated"].to_numpy(dtype=bool)] = False
    adjusted["conditional_cif_mask"] = [row for row in masks]
    inner_counts: dict[str, dict[str, int]] = {}
    objectives: dict[str, float] = {}
    for outer in range(outer_folds):
        development = adjusted.loc[
            adjusted["outer_fold"].ne(outer)
        ].copy()
        assignment = _greedy_balanced_assignment(
            development,
            folds=inner_folds,
            seed=seed + 1000 + outer,
        )
        lookup = dict(
            zip(
                development["sample_id"].astype(str),
                assignment.assignment.astype(int),
            )
        )
        column = f"outer_{outer}_inner_fold"
        manifest[column] = manifest["sample_id"].astype(str).map(lookup)
        manifest.loc[manifest["outer_fold"].eq(outer), column] = -1
        manifest[column] = manifest[column].astype(int)
        objectives[str(outer)] = assignment.objective
        inner_counts[str(outer)] = {
            str(key): int(value)
            for key, value in pd.Series(assignment.assignment)
            .value_counts()
            .sort_index()
            .items()
        }
    if manifest.groupby("document_hash")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("document leakage across frozen outer folds")
    for outer in range(outer_folds):
        column = f"outer_{outer}_inner_fold"
        development = manifest.loc[manifest["outer_fold"].ne(outer)]
        if development.groupby("document_hash")[column].nunique().max() != 1:
            raise RuntimeError(f"document leakage in inner crossfit {outer}")
    audit = {
        "schema_version": "router_v4_0_2.crossfit_split_audit.v1",
        "rows": int(len(manifest)),
        "documents": int(manifest["document_hash"].nunique()),
        "outer_fold_rows": {
            str(key): int(value)
            for key, value in manifest["outer_fold"]
            .value_counts()
            .sort_index()
            .items()
        },
        "inner_fold_rows": inner_counts,
        "inner_objectives": objectives,
        "document_leakage": 0,
        "passed": True,
    }
    return manifest, audit
