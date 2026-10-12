from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from codex.router_v4_0_0.src.actions import (
    ACTION_SPACE_VERSION,
    H2O_ACTIONS,
)
from codex.router_v4_0_0.src.artifacts import sha256_file


FORBIDDEN_MODEL_FIELDS = frozenset(
    {
        "cohort_id",
        "archive_sha",
        "protocol_fingerprint",
        "provenance_fingerprint",
        "campaign",
        "package_path",
        "source_path",
        "archive_path",
        "document_hash",
        "document_family_hash",
        "policy_acceptability",
        "policy_peak_kv_bytes",
        "failure_targets",
        "observed_mask",
        "conditional_cif_target",
        "peak_kv_ratio",
        "full_acceptable",
        "correct",
        "ground_truth",
    }
)
MODEL_INPUT_ALLOWLIST = frozenset(
    {
        "base80",
        "policy_one_hot5",
        "policy_order_coordinate",
        "base80_x_policy_order",
    }
)
BEHAVIOR_FINGERPRINT_FIELDS = (
    "algorithm_runtime_fingerprint",
    "model_asset_fingerprint",
    "evaluation_protocol",
)


def _check_hash(path: str | Path, expected: str) -> None:
    if sha256_file(path).lower() != str(expected).lower():
        raise RuntimeError(f"frozen input hash mismatch: {path}")


def _canonical_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_inputs(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    inputs = config["inputs"]
    for key in ("source_dataset", "outer_fold_manifest", "input_provenance"):
        _check_hash(inputs[key], inputs[f"{key}_sha256"])
    rows = pd.read_parquet(inputs["source_dataset"])
    folds = pd.read_parquet(inputs["outer_fold_manifest"])
    provenance = json.loads(
        Path(inputs["input_provenance"]).read_text(encoding="utf-8")
    )
    if len(rows) != len(folds):
        raise ValueError("source rows and frozen fold manifest differ")
    merged = rows.merge(
        folds[["sample_id", "document_hash", "outer_fold"]],
        on=["sample_id", "document_hash"],
        validate="one_to_one",
        suffixes=("", "_frozen"),
    )
    if "outer_fold_frozen" in merged:
        if not np.array_equal(
            merged["outer_fold"].to_numpy(),
            merged["outer_fold_frozen"].to_numpy(),
        ):
            raise ValueError("source/frozen outer fold mismatch")
        merged = merged.drop(columns=["outer_fold_frozen"])
    return merged, folds, provenance


def build_compatibility_manifest(
    provenance: dict[str, Any],
) -> dict[str, Any]:
    compatibility = dict(provenance["compatibility"])
    common = dict(compatibility["common_fingerprints"])
    behavior: dict[str, str] = {}
    for field in BEHAVIOR_FINGERPRINT_FIELDS:
        values = list(common.get(field, []))
        if len(values) != 1 or not values[0] or values[0] == "None":
            raise RuntimeError(f"behavior compatibility field is not frozen: {field}")
        behavior[field] = str(values[0])
    if not bool(compatibility.get("compatible")):
        raise RuntimeError("upstream cohort compatibility audit failed")
    key_payload = {
        **behavior,
        "action_space_version": ACTION_SPACE_VERSION,
    }
    return {
        "schema_version": "router_v4_0_2.routing_compatibility.v1",
        "compatible": True,
        "behavior_fields": key_payload,
        "routing_compatibility_key": _canonical_sha(key_payload),
        "cohort_protocol_fingerprints": compatibility[
            "cohort_protocol_fingerprints"
        ],
        "hardware_runtime_fingerprint": common.get(
            "hardware_runtime_fingerprint", []
        ),
        "provenance_fields_are_not_decision_inputs": True,
    }


def _array(series: pd.Series, width: int, dtype: Any) -> np.ndarray:
    values = np.stack(series.map(np.asarray)).astype(dtype)
    if values.shape != (len(series), width):
        raise ValueError(f"expected tensor [N,{width}], got {values.shape}")
    return values


def _base80(rows: pd.DataFrame) -> np.ndarray:
    z64 = _array(rows["z64"], 64, np.float32)
    structural = _array(rows["normalized_structural16"], 16, np.float32)
    values = np.concatenate((z64, structural), axis=1)
    if not np.isfinite(values).all():
        raise ValueError("base80 contains non-finite values")
    return values


def build_question_table(
    rows: pd.DataFrame, compatibility_key: str
) -> pd.DataFrame:
    result = rows.copy()
    result["base80"] = [row for row in _base80(result)]
    result["routing_compatibility_key"] = str(compatibility_key)
    required = {
        "sample_id",
        "document_hash",
        "base80",
        "full_failure_target",
        "context_truncated",
        "routing_compatibility_key",
    }
    if required - set(result):
        raise ValueError("question table contract incomplete")
    return result


def expand_candidates(
    questions: pd.DataFrame,
    coordinates: Iterable[float],
    *,
    observed_only: bool,
) -> pd.DataFrame:
    order = np.asarray(list(coordinates), dtype=np.float32)
    if order.shape != (5,):
        raise ValueError("five policy order coordinates are required")
    targets = _array(questions["conditional_cif_targets"], 5, np.int64)
    masks = _array(questions["conditional_cif_mask"], 5, bool)
    observed = _array(questions["observed_mask"], 6, bool)[:, 1:]
    costs = _array(questions["cost_targets"], 5, np.float32)
    bases = _array(questions["base80"], 80, np.float32)
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(questions.itertuples()):
        for policy_index, action in enumerate(H2O_ACTIONS):
            if observed_only and not observed[row_index, policy_index]:
                continue
            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[policy_index] = 1.0
            coordinate = float(order[policy_index])
            cif_mask = bool(
                masks[row_index, policy_index]
                and not bool(row.context_truncated)
            )
            cost_mask = bool(observed[row_index, policy_index])
            output.append(
                {
                    "candidate_id": f"{row.sample_id}::{int(action)}",
                    "sample_id": str(row.sample_id),
                    "document_hash": str(row.document_hash),
                    "action_id": int(action),
                    "action_space_version": ACTION_SPACE_VERSION,
                    "base80": bases[row_index],
                    "policy_one_hot5": one_hot,
                    "policy_order_coordinate": coordinate,
                    "base80_x_policy_order": bases[row_index] * coordinate,
                    "risk_input166": np.concatenate(
                        (
                            bases[row_index],
                            one_hot,
                            np.asarray([coordinate], dtype=np.float32),
                            bases[row_index] * coordinate,
                        )
                    ).astype(np.float32),
                    "cost_input22": np.concatenate(
                        (
                            bases[row_index, 64:],
                            one_hot,
                            np.asarray([coordinate], dtype=np.float32),
                        )
                    ).astype(np.float32),
                    "conditional_cif_target": int(
                        targets[row_index, policy_index]
                    ),
                    "cif_mask": cif_mask,
                    "peak_kv_ratio": (
                        float(costs[row_index, policy_index])
                        if cost_mask
                        else np.nan
                    ),
                    "cost_mask": cost_mask,
                    "context_truncated": bool(row.context_truncated),
                    "routing_compatibility_key": str(
                        row.routing_compatibility_key
                    ),
                    "outer_fold": int(row.outer_fold),
                }
            )
    result = pd.DataFrame(output)
    if FORBIDDEN_MODEL_FIELDS & MODEL_INPUT_ALLOWLIST:
        raise RuntimeError("forbidden provenance/outcome entered model allowlist")
    if result["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate ID")
    return result


def audit_data(
    config: dict[str, Any],
    questions: pd.DataFrame,
    candidates: pd.DataFrame,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    expected = config["expected"]
    observed = _array(questions["observed_mask"], 6, bool)
    targets = _array(questions["conditional_cif_targets"], 5, np.int64)
    masks = _array(questions["conditional_cif_mask"], 5, bool)
    values = {
        "questions": int(len(questions)),
        "documents": int(questions["document_hash"].nunique()),
        "complete_panels": int(observed.all(axis=1).sum()),
        "policy_observed": [int(value) for value in observed[:, 1:].sum(axis=0)],
        "conditional_cif": [
            int((masks[:, index] & (targets[:, index] == 1)).sum())
            for index in range(5)
        ],
        "full_failures": int(questions["full_failure_target"].sum()),
        "truncated": int(questions["context_truncated"].sum()),
        "testmini_overlap": int(questions["testmini_document_overlap"].sum()),
    }
    passed = all(values[key] == expected[key] for key in values)
    return {
        "schema_version": "router_v4_0_2.data_audit.v1",
        **values,
        "observed_candidate_rows": int(len(candidates)),
        "cif_training_rows": int(candidates["cif_mask"].sum()),
        "cost_training_rows": int(candidates["cost_mask"].sum()),
        "compatibility_key": compatibility["routing_compatibility_key"],
        "forbidden_model_fields_verified": not bool(
            FORBIDDEN_MODEL_FIELDS & MODEL_INPUT_ALLOWLIST
        ),
        "passed": bool(passed and compatibility["compatible"]),
        "routing_enabled": False,
        "production_checkpoint": False,
        "formal_result_eligible": False,
    }

