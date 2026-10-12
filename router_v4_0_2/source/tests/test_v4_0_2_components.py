from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from codex.router_v4_0_0.src.actions import Action
from codex.router_v4_0_2.src.calibration import apply_calibrators, fit_calibrators
from codex.router_v4_0_2.src.config import load_config
from codex.router_v4_0_2.src.data import (
    FORBIDDEN_MODEL_FIELDS,
    MODEL_INPUT_ALLOWLIST,
    audit_data,
    build_compatibility_manifest,
    build_question_table,
    expand_candidates,
    load_frozen_inputs,
)
from codex.router_v4_0_2.src.decision import (
    DecisionConfig,
    choose_action,
    optimize_zero_cif_rule,
)
from codex.router_v4_0_2.src.model import (
    ModelConfig,
    SharedCandidateEnsemble,
    load_checkpoint,
    save_checkpoint,
)
from codex.router_v4_0_2.src.split import build_crossfit_manifest
from codex.router_v4_0_2.src.train import _bootstrap_weights


ROOT = Path(r"G:\2026.08_KV_Cache_Research")
CONFIG = (
    ROOT
    / "codex"
    / "router_v4_0_2"
    / "configs"
    / "shared_candidate_risk_v1.yaml"
)


def _real() -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    config = load_config(CONFIG)
    rows, folds, provenance = load_frozen_inputs(config)
    compatibility = build_compatibility_manifest(provenance)
    questions = build_question_table(
        rows, compatibility["routing_compatibility_key"]
    )
    return config, questions, folds, compatibility


def test_config_is_zero_cif_nonproduction() -> None:
    config = load_config(CONFIG)
    assert config["acceptance"]["maximum_compression_induced_failures"] == 0
    assert config["acceptance"]["enforce_quality_gates"] is False
    assert config["acceptance"]["quality_gate_override_reason"]
    assert "minimum_h2o_route_rate" not in config["acceptance"]
    assert config["status"]["routing_enabled"] is False
    assert config["decision"]["allow_h2o_rescue"] is False


def test_real_counts_candidate_contract_and_masks() -> None:
    config, questions, _, compatibility = _real()
    candidates = expand_candidates(
        questions,
        config["model"]["policy_order_coordinates"],
        observed_only=True,
    )
    audit = audit_data(config, questions, candidates, compatibility)
    assert audit["passed"]
    assert audit["questions"] == 780
    assert audit["documents"] == 661
    assert audit["policy_observed"] == [500, 780, 780, 500, 500]
    assert audit["conditional_cif"] == [11, 33, 48, 33, 38]
    assert candidates["risk_input166"].map(len).eq(166).all()
    assert candidates["cost_input22"].map(len).eq(22).all()
    assert not candidates.loc[candidates["context_truncated"], "cif_mask"].any()


def test_forbidden_provenance_is_not_model_input() -> None:
    assert not (FORBIDDEN_MODEL_FIELDS & MODEL_INPUT_ALLOWLIST)


def test_provenance_protocols_share_behavior_compatibility_key() -> None:
    _, questions, _, compatibility = _real()
    assert compatibility["compatible"]
    assert questions["routing_compatibility_key"].nunique() == 1
    assert questions["protocol_fingerprint"].nunique() == 3


def test_model_shapes_and_checkpoint_equivalence() -> None:
    config = ModelConfig(members=3, member_seeds=(42, 43, 44))
    model = SharedCandidateEnsemble(config).eval()
    base = torch.randn(4, 80)
    risk = torch.randn(7, 166)
    cost = torch.randn(7, 22)
    with torch.no_grad():
        expected = (
            model.full_logits(base),
            model.risk_logits(risk),
            model.cost_ratios(cost),
        )
    assert expected[0].shape == (4, 3)
    assert expected[1].shape == (7, 3)
    assert expected[2].shape == (7, 3)
    directory = Path(__file__).parent / ".checkpoint_equivalence"
    path = directory / "model.pt"
    try:
        path = save_checkpoint(path, model, metadata={"fold": 0})
        loaded, metadata = load_checkpoint(path)
        with torch.no_grad():
            actual = (
                loaded.full_logits(base),
                loaded.risk_logits(risk),
                loaded.cost_ratios(cost),
            )
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)
        assert metadata["production_checkpoint"] is False
    finally:
        if path.exists():
            path.unlink()
        if directory.exists():
            directory.rmdir()


def _decision_config(key: str = "compatible") -> DecisionConfig:
    return DecisionConfig(
        full_ucb_threshold=0.2,
        cif_ucb_threshold=0.2,
        min_expected_saving=0.05,
        max_bootstrap_std=0.1,
        tie_tolerance=0.01,
        routing_compatibility_key=key,
    )


def test_truncated_and_incompatible_fall_back_full() -> None:
    common = dict(
        full_ucb=0.01,
        full_std=0.01,
        risk_ucb=np.zeros(5),
        risk_std=np.zeros(5),
        cost_mean=np.asarray([0.9, 0.8, 0.7, 0.6, 0.5]),
        cost_std=np.zeros(5),
        config=_decision_config(),
        allowed_actions=tuple(Action),
    )
    action, _ = choose_action(
        **common, compatibility_key="compatible", context_truncated=True
    )
    assert action is Action.FULL
    action, _ = choose_action(
        **common, compatibility_key="wrong", context_truncated=False
    )
    assert action is Action.FULL


def test_monotone_ucb_and_calibration_shapes() -> None:
    count, members = 30, 4
    questions = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(count)],
            "document_hash": [f"d{i}" for i in range(count)],
            "full_failure_target": [i % 2 for i in range(count)],
            "conditional_cif_targets": [
                np.asarray([(i + j) % 4 == 0 for j in range(5)], dtype=int)
                for i in range(count)
            ],
            "conditional_cif_mask": [np.ones(5, dtype=bool)] * count,
            "context_truncated": [False] * count,
        }
    )
    predictions = questions[["sample_id", "document_hash"]].copy()
    predictions["raw_full_member_probabilities"] = [
        np.linspace(0.1, 0.4, members) + 0.01 * (i % 2)
        for i in range(count)
    ]
    predictions["raw_cif_member_probabilities"] = [
        np.tile(np.linspace(0.1, 0.4, 5)[:, None], (1, members))
        + 0.01 * (i % 3)
        for i in range(count)
    ]
    predictions["predicted_cost_member_ratios"] = [
        np.tile(np.linspace(0.9, 0.5, 5)[:, None], (1, members))
        for _ in range(count)
    ]
    full, risk = fit_calibrators(questions, predictions)
    values = apply_calibrators(
        predictions, full, risk, ucb_z=1.645, monotone=True
    )
    assert values["risk_ucb"].shape == (count, 5)
    assert np.all(np.diff(values["risk_ucb"], axis=1) >= -1.0e-7)


def test_frozen_outer_and_inner_crossfit_are_grouped() -> None:
    records = []
    for index in range(75):
        records.append(
            {
                "sample_id": f"s{index}",
                "document_hash": f"d{index}",
                "category": (
                    "SimpShort",
                    "SimpLong",
                    "CompShort",
                    "CompLong",
                )[index % 4],
                "cohort_id": ("old80", "new200", "new500")[index % 3],
                "outer_fold": index % 5,
                "context_truncated": False,
                "testmini_document_overlap": index % 7 == 0,
                "full_failure_target": index % 4 == 0,
                "conditional_cif_targets": np.ones(5, dtype=int)
                if index < 30
                else np.zeros(5, dtype=int),
                "conditional_cif_mask": np.ones(5, dtype=bool),
                "observed_mask": np.ones(6, dtype=bool),
            }
        )
    questions = pd.DataFrame(records)
    frozen = questions[
        ["sample_id", "document_hash", "category", "cohort_id", "outer_fold"]
    ].copy()
    manifest, audit = build_crossfit_manifest(
        questions,
        frozen,
        outer_folds=5,
        inner_folds=3,
        seed=42,
    )
    assert audit["passed"]
    assert manifest.groupby("document_hash")["outer_fold"].nunique().max() == 1
    for outer in range(5):
        column = f"outer_{outer}_inner_fold"
        development = manifest.loc[manifest["outer_fold"].ne(outer)]
        assert development.groupby("document_hash")[column].nunique().max() == 1


def test_document_bootstrap_is_grouped_and_deterministic() -> None:
    documents = pd.Series(["d0", "d0", "d1", "d2", "d2"])
    left = _bootstrap_weights(documents, (42, 43, 44))
    right = _bootstrap_weights(documents, (42, 43, 44))
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left[0], left[1])
    np.testing.assert_array_equal(left[3], left[4])
    assert left.shape == (5, 3)
    assert not np.array_equal(left[:, 0], left[:, 1])


def test_zero_cif_optimizer_prefers_only_safe_candidate() -> None:
    action = Action.SAFE_H2O_R080_I32
    safe_outcomes = {
        member.name: (member is Action.FULL or member is action)
        for member in Action
    }
    unsafe_outcomes = dict(safe_outcomes)
    unsafe_outcomes[action.name] = False
    costs = {member.name: 100.0 for member in Action}
    costs[action.name] = 70.0
    questions = pd.DataFrame(
        {
            "sample_id": ["safe", "unsafe"],
            "document_hash": ["d0", "d1"],
            "routing_compatibility_key": ["key", "key"],
            "context_truncated": [False, False],
            "full_acceptable": [True, True],
            "policy_acceptability": [safe_outcomes, unsafe_outcomes],
            "policy_peak_kv_bytes": [costs, costs],
        }
    )
    risk = np.ones((2, 5), dtype=np.float32)
    risk[:, int(action) - 1] = [0.1, 0.9]
    cost = np.ones((2, 5), dtype=np.float32)
    cost[:, int(action) - 1] = 0.7
    calibrated = {
        "full_ucb": np.asarray([0.01, 0.01], dtype=np.float32),
        "full_std": np.zeros(2, dtype=np.float32),
        "risk_ucb": risk,
        "risk_std": np.zeros((2, 5), dtype=np.float32),
        "cost_mean": cost,
        "cost_std": np.zeros((2, 5), dtype=np.float32),
    }
    _, summary = optimize_zero_cif_rule(
        questions,
        calibrated,
        base_config=_decision_config("key"),
        allowed_actions=(Action.FULL, action),
        outer_fold=0,
        grid_points=5,
    )
    assert summary["compression_induced_failures"] == 0
    assert summary["h2o_rescue_rows"] == 0
    assert summary["unobserved_selected_rows"] == 0
    assert summary["compressed_rows"] == 1
    assert summary["aggregate_safe_peak_kv_saving"] > 0


def test_nested_member_arrays_round_trip_through_parquet() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "raw_full_member_probabilities": [[0.1, 0.2]],
            "raw_cif_member_probabilities": [
                np.arange(10, dtype=float).reshape(5, 2).tolist()
            ],
            "predicted_cost_member_ratios": [
                np.arange(10, dtype=float).reshape(5, 2).tolist()
            ],
        }
    )
    directory = Path(__file__).parent / ".parquet_round_trip"
    path = directory / "predictions.parquet"
    directory.mkdir(exist_ok=False)
    try:
        frame.to_parquet(path, index=False)
        restored = pd.read_parquet(path)
        assert np.asarray(
            restored.loc[0, "raw_cif_member_probabilities"].tolist()
        ).shape == (5, 2)
        assert np.asarray(
            restored.loc[0, "predicted_cost_member_ratios"].tolist()
        ).shape == (5, 2)
    finally:
        if path.exists():
            path.unlink()
        if directory.exists():
            directory.rmdir()
