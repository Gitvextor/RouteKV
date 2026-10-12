from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codex.router_v4_0_0.src.actions import Action, action_space_payload
from codex.router_v4_0_0.src.artifacts import (
    build_manifest,
    sha256_file,
    verify_manifest,
    write_json,
    write_parquet,
)

from .calibration import (
    apply_calibrators,
    calibration_payload,
    fit_calibrators,
)
from .config import load_config, output_root
from .data import (
    audit_data as audit_frame,
    build_compatibility_manifest,
    build_question_table,
    expand_candidates,
    load_frozen_inputs,
)
from .decision import (
    DecisionConfig,
    make_decisions,
    optimize_zero_cif_rule,
)
from .evaluate import evaluate
from .model import ModelConfig, load_checkpoint
from .report import render_report
from .split import build_crossfit_manifest
from .train import TrainConfig, predict_ensemble, train_ensemble


def _refresh(root: Path) -> None:
    write_json(root / "artifact_manifest.json", build_manifest(root))


def _require(root: Path, marker: str) -> None:
    if not (root / marker).is_file() or verify_manifest(root):
        raise RuntimeError(f"required frozen stage is missing or changed: {marker}")


def _model_config(config: dict[str, Any]) -> ModelConfig:
    model = config["model"]
    training = config["training"]
    return ModelConfig(
        members=int(training["bootstrap_members"]),
        member_seeds=tuple(int(value) for value in training["bootstrap_seeds"]),
        base_dimension=int(model["base_dimension"]),
        risk_dimension=int(model["risk_dimension"]),
        cost_dimension=int(model["cost_dimension"]),
        cost_hidden=int(model["cost_hidden"]),
    )


def _train_config(
    config: dict[str, Any], device: str, *, epochs: int | None = None
) -> TrainConfig:
    values = config["training"]
    return TrainConfig(
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        batch_size_questions=int(values["batch_size_questions"]),
        batch_size_candidates=int(values["batch_size_candidates"]),
        epochs=int(values["epochs"] if epochs is None else epochs),
        gradient_clip_norm=float(values["gradient_clip_norm"]),
        maximum_pos_weight=float(values["maximum_pos_weight"]),
        cost_loss_weight=float(values["cost_loss_weight"]),
        device=device,
    )


def _base_decision(
    config: dict[str, Any], compatibility_key: str
) -> DecisionConfig:
    values = config["decision"]
    return DecisionConfig(
        full_ucb_threshold=0.0,
        cif_ucb_threshold=0.0,
        min_expected_saving=float(values["min_expected_saving"]),
        max_bootstrap_std=float(values["max_bootstrap_std"]),
        tie_tolerance=float(values["tie_tolerance"]),
        routing_compatibility_key=compatibility_key,
    )


def _sort_pair(
    questions: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = questions.sort_values("sample_id").reset_index(drop=True)
    p = predictions.sort_values("sample_id").reset_index(drop=True)
    if q["sample_id"].tolist() != p["sample_id"].tolist():
        raise ValueError("question/prediction alignment mismatch")
    return q, p


def _train_or_resume(
    questions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint: Path,
    metadata: dict[str, Any],
) -> Path:
    history_path = checkpoint.parent / "history.parquet"
    if checkpoint.is_file() and history_path.is_file():
        model, existing = load_checkpoint(
            checkpoint, map_location=train_config.device
        )
        if model.config != model_config:
            raise RuntimeError("resume checkpoint model config mismatch")
        for key, value in metadata.items():
            if existing.get(key) != value:
                raise RuntimeError(f"resume checkpoint metadata mismatch: {key}")
        return checkpoint
    path, history = train_ensemble(
        questions,
        candidates,
        model_config=model_config,
        train_config=train_config,
        output_path=checkpoint,
        metadata=metadata,
    )
    write_parquet(history_path, history)
    return path


def _enrich_predictions(
    predictions: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    *,
    outer_fold: int,
) -> pd.DataFrame:
    result = predictions.copy()
    for key in (
        "full_mean",
        "full_std",
        "full_ucb",
        "risk_mean",
        "risk_std",
        "risk_ucb",
        "cost_mean",
        "cost_std",
    ):
        values = calibrated[key]
        result[key] = (
            values.astype(np.float32)
            if values.ndim == 1
            else [row.astype(np.float32) for row in values]
        )
    result["outer_fold"] = int(outer_fold)
    return result


def audit_data(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "AUDIT_COMPLETE.json").exists():
        raise RuntimeError("V4.0.2 audit already frozen")
    rows, folds, provenance = load_frozen_inputs(config)
    compatibility = build_compatibility_manifest(provenance)
    questions = build_question_table(
        rows, compatibility["routing_compatibility_key"]
    )
    candidates = expand_candidates(
        questions,
        config["model"]["policy_order_coordinates"],
        observed_only=True,
    )
    audit = audit_frame(config, questions, candidates, compatibility)
    if not audit["passed"]:
        raise RuntimeError(f"V4.0.2 data audit failed: {audit}")
    write_json(root / "data_audit.json", audit)
    write_json(root / "compatibility_manifest.json", compatibility)
    write_json(root / "action_space.json", action_space_payload())
    write_json(
        root / "input_provenance.json",
        {
            "schema_version": "router_v4_0_2.input_provenance.v1",
            "source_dataset_sha256": sha256_file(
                config["inputs"]["source_dataset"]
            ),
            "outer_fold_manifest_sha256": sha256_file(
                config["inputs"]["outer_fold_manifest"]
            ),
            "input_provenance_sha256": sha256_file(
                config["inputs"]["input_provenance"]
            ),
            "source_rows": int(len(rows)),
            "frozen_outer_rows": int(len(folds)),
        },
    )
    write_json(
        root / "AUDIT_COMPLETE.json",
        {
            "schema_version": "router_v4_0_2.audit_complete.v1",
            "passed": True,
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        },
    )
    _refresh(root)
    return audit


def build_candidates(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "AUDIT_COMPLETE.json")
    if (root / "CANDIDATES_COMPLETE.json").exists():
        raise RuntimeError("V4.0.2 candidates already frozen")
    rows, _, provenance = load_frozen_inputs(config)
    compatibility = build_compatibility_manifest(provenance)
    questions = build_question_table(
        rows, compatibility["routing_compatibility_key"]
    )
    candidates = expand_candidates(
        questions,
        config["model"]["policy_order_coordinates"],
        observed_only=True,
    )
    write_parquet(root / "question_dataset.parquet", questions)
    write_parquet(root / "candidate_dataset.parquet", candidates)
    payload = {
        "schema_version": "router_v4_0_2.candidates_complete.v1",
        "questions": int(len(questions)),
        "candidates": int(len(candidates)),
        "cif_training_rows": int(candidates["cif_mask"].sum()),
        "cost_training_rows": int(candidates["cost_mask"].sum()),
        "passed": True,
    }
    write_json(root / "CANDIDATES_COMPLETE.json", payload)
    _refresh(root)
    return payload


def prepare_nested_crossfit(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "CANDIDATES_COMPLETE.json")
    if (root / "CROSSFIT_PREPARED.json").exists():
        raise RuntimeError("V4.0.2 crossfit already frozen")
    questions = pd.read_parquet(root / "question_dataset.parquet")
    frozen = pd.read_parquet(config["inputs"]["outer_fold_manifest"])
    split = config["split"]
    manifest, audit = build_crossfit_manifest(
        questions,
        frozen,
        outer_folds=int(split["outer_folds"]),
        inner_folds=int(split["inner_folds"]),
        seed=int(split["seed"]),
    )
    write_parquet(root / "nested_fold_manifest.parquet", manifest)
    write_json(root / "split_audit.json", audit)
    payload = {
        "schema_version": "router_v4_0_2.crossfit_prepared.v1",
        "fold_manifest_sha256": sha256_file(
            root / "nested_fold_manifest.parquet"
        ),
        "document_leakage": 0,
        "passed": True,
    }
    write_json(root / "CROSSFIT_PREPARED.json", payload)
    _refresh(root)
    return payload


def cuda_tiny_smoke(
    config_path: str | Path, *, device: str
) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "CROSSFIT_PREPARED.json")
    if (root / "CUDA_TINY_SMOKE.json").exists():
        raise RuntimeError("CUDA smoke already complete")
    questions = pd.read_parquet(root / "question_dataset.parquet")
    candidates = pd.read_parquet(root / "candidate_dataset.parquet")
    sample = questions.iloc[:160].copy()
    candidate = candidates.loc[
        candidates["sample_id"].isin(sample["sample_id"])
    ].copy()
    if not candidate["cif_mask"].any():
        sample = questions.copy()
        candidate = candidates.copy()
    path, history = train_ensemble(
        sample,
        candidate,
        model_config=_model_config(config),
        train_config=_train_config(config, device, epochs=1),
        output_path=root / "cuda_tiny_smoke" / "ensemble.pt",
        metadata={"role": "cuda_tiny_smoke"},
    )
    write_parquet(root / "cuda_tiny_smoke" / "history.parquet", history)
    payload = {
        "schema_version": "router_v4_0_2.cuda_tiny_smoke.v1",
        "checkpoint": str(path.relative_to(root)),
        "device": device,
        "passed": True,
    }
    write_json(root / "CUDA_TINY_SMOKE.json", payload)
    _refresh(root)
    return payload


def run_nested_oof(
    config_path: str | Path, *, device: str
) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "CUDA_TINY_SMOKE.json")
    if (root / "NESTED_OOF_COMPLETE.json").exists():
        raise RuntimeError("V4.0.2 Nested OOF already complete")
    questions = pd.read_parquet(root / "question_dataset.parquet")
    candidates = pd.read_parquet(root / "candidate_dataset.parquet")
    manifest = pd.read_parquet(root / "nested_fold_manifest.parquet")
    compatibility = json.loads(
        (root / "compatibility_manifest.json").read_text(encoding="utf-8")
    )
    key = str(compatibility["routing_compatibility_key"])
    coordinates = list(config["model"]["policy_order_coordinates"])
    model_config = _model_config(config)
    train_config = _train_config(config, device)
    calibration_cfg = config["calibration"]
    six_allowed = tuple(Action)
    common_allowed = (
        Action.FULL,
        Action.SAFE_H2O_R080_I32,
        Action.SAFE_H2O_R060_I32,
    )
    inner_all: list[pd.DataFrame] = []
    outer_all: list[pd.DataFrame] = []
    decisions_six: list[pd.DataFrame] = []
    decisions_common: list[pd.DataFrame] = []
    rules: list[dict[str, Any]] = []
    outer_folds = int(config["split"]["outer_folds"])
    inner_folds = int(config["split"]["inner_folds"])
    for outer in range(outer_folds):
        development_ids = set(
            manifest.loc[
                manifest["outer_fold"].ne(outer), "sample_id"
            ].astype(str)
        )
        test_ids = set(
            manifest.loc[
                manifest["outer_fold"].eq(outer), "sample_id"
            ].astype(str)
        )
        development = questions.loc[
            questions["sample_id"].astype(str).isin(development_ids)
        ].copy()
        test = questions.loc[
            questions["sample_id"].astype(str).isin(test_ids)
        ].copy()
        inner_predictions: list[pd.DataFrame] = []
        column = f"outer_{outer}_inner_fold"
        for inner in range(inner_folds):
            hold_ids = set(
                manifest.loc[
                    manifest[column].eq(inner), "sample_id"
                ].astype(str)
            )
            train_ids = development_ids - hold_ids
            train_q = questions.loc[
                questions["sample_id"].astype(str).isin(train_ids)
            ].copy()
            hold_q = questions.loc[
                questions["sample_id"].astype(str).isin(hold_ids)
            ].copy()
            train_c = candidates.loc[
                candidates["sample_id"].astype(str).isin(train_ids)
            ].copy()
            checkpoint = (
                root
                / "fold_models"
                / f"outer_{outer}"
                / f"inner_{inner}"
                / "ensemble.pt"
            )
            path = _train_or_resume(
                train_q,
                train_c,
                model_config=model_config,
                train_config=train_config,
                checkpoint=checkpoint,
                metadata={
                    "role": "inner_crossfit",
                    "outer_fold": outer,
                    "inner_fold": inner,
                },
            )
            model, _ = load_checkpoint(path, map_location=device)
            predicted = predict_ensemble(
                model,
                hold_q,
                coordinates=coordinates,
                device=device,
            )
            predicted["calibration_outer_fold"] = outer
            predicted["inner_fold"] = inner
            inner_predictions.append(predicted)
        inner = pd.concat(inner_predictions, ignore_index=True)
        development, inner = _sort_pair(development, inner)
        if len(inner) != len(development):
            raise RuntimeError("inner crossfit lacks exact development coverage")
        full_calibrator, risk_calibrator = fit_calibrators(
            development, inner
        )
        calibrated_inner = apply_calibrators(
            inner,
            full_calibrator,
            risk_calibrator,
            ucb_z=float(calibration_cfg["ucb_z"]),
            monotone=bool(config["decision"]["monotone_prefix_risk"]),
        )
        complete_mask = development["observed_mask"].map(all).to_numpy(bool)
        complete_development = development.loc[complete_mask].reset_index(
            drop=True
        )
        calibrated_complete = {
            key_name: values[complete_mask]
            for key_name, values in calibrated_inner.items()
        }
        rule, summary = optimize_zero_cif_rule(
            complete_development,
            calibrated_complete,
            base_config=_base_decision(config, key),
            allowed_actions=six_allowed,
            outer_fold=outer,
            grid_points=int(calibration_cfg["grid_points"]),
        )
        rule_payload = {
            "schema_version": "router_v4_0_2.fold_rule.v1",
            "outer_fold": outer,
            "calibration": calibration_payload(
                full_calibrator, risk_calibrator
            ),
            "decision": asdict(rule),
            "calibration_summary": summary,
        }
        write_json(
            root / "fold_calibrators" / f"outer_{outer}.json",
            rule_payload,
        )
        rules.append(rule_payload)
        inner_all.append(inner)

        train_candidates = candidates.loc[
            candidates["sample_id"].astype(str).isin(development_ids)
        ].copy()
        checkpoint = (
            root / "fold_models" / f"outer_{outer}" / "final" / "ensemble.pt"
        )
        path = _train_or_resume(
            development,
            train_candidates,
            model_config=model_config,
            train_config=train_config,
            checkpoint=checkpoint,
            metadata={"role": "outer_final", "outer_fold": outer},
        )
        model, _ = load_checkpoint(path, map_location=device)
        predicted = predict_ensemble(
            model, test, coordinates=coordinates, device=device
        )
        test, predicted = _sort_pair(test, predicted)
        calibrated_test = apply_calibrators(
            predicted,
            full_calibrator,
            risk_calibrator,
            ucb_z=float(calibration_cfg["ucb_z"]),
            monotone=bool(config["decision"]["monotone_prefix_risk"]),
        )
        outer_all.append(
            _enrich_predictions(
                predicted, calibrated_test, outer_fold=outer
            )
        )
        decisions_six.append(
            make_decisions(
                test,
                calibrated_test,
                config=rule,
                allowed_actions=six_allowed,
                outer_fold=outer,
            )
        )
        decisions_common.append(
            make_decisions(
                test,
                calibrated_test,
                config=rule,
                allowed_actions=common_allowed,
                outer_fold=outer,
            )
        )
    inner_frame = pd.concat(inner_all, ignore_index=True)
    outer_frame = pd.concat(outer_all, ignore_index=True)
    six_frame = pd.concat(decisions_six, ignore_index=True)
    common_frame = pd.concat(decisions_common, ignore_index=True)
    if (
        len(outer_frame) != len(questions)
        or outer_frame["sample_id"].nunique() != len(questions)
    ):
        raise RuntimeError("outer OOF coverage is not exact")
    write_parquet(root / "inner_crossfit_predictions.parquet", inner_frame)
    write_parquet(root / "outer_oof_predictions.parquet", outer_frame)
    write_parquet(
        root / "outer_oof_decisions_complete6.parquet", six_frame
    )
    write_parquet(
        root / "outer_oof_decisions_common3.parquet", common_frame
    )
    payload = {
        "schema_version": "router_v4_0_2.nested_oof_complete.v1",
        "outer_predictions": int(len(outer_frame)),
        "inner_crossfit_predictions": int(len(inner_frame)),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "bootstrap_members": model_config.members,
        "fold_rules": rules,
        "passed": True,
        "routing_enabled": False,
        "production_checkpoint": False,
        "formal_result_eligible": False,
    }
    write_json(root / "NESTED_OOF_COMPLETE.json", payload)
    _refresh(root)
    return payload


def evaluate_oof(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "NESTED_OOF_COMPLETE.json")
    questions = pd.read_parquet(root / "question_dataset.parquet")
    predictions = pd.read_parquet(root / "outer_oof_predictions.parquet")
    six = pd.read_parquet(root / "outer_oof_decisions_complete6.parquet")
    common = pd.read_parquet(root / "outer_oof_decisions_common3.parquet")
    compatibility = json.loads(
        (root / "compatibility_manifest.json").read_text(encoding="utf-8")
    )
    historical = json.loads(
        Path(config["inputs"]["historical_evaluation"]).read_text(
            encoding="utf-8"
        )
    )
    evaluation = evaluate(
        questions,
        predictions,
        six,
        common,
        compatibility_key=compatibility["routing_compatibility_key"],
        historical=historical,
        enforce_quality_gates=bool(
            config["acceptance"]["enforce_quality_gates"]
        ),
        quality_gate_override_reason=str(
            config["acceptance"].get("quality_gate_override_reason", "")
        ),
    )
    write_json(root / "routing_evaluation.json", evaluation)
    complete = questions.loc[questions["observed_mask"].map(all)].copy()
    complete_ids = set(complete["sample_id"].astype(str))
    complete_predictions = predictions.loc[
        predictions["sample_id"].astype(str).isin(complete_ids)
    ].copy()
    risk_frame = complete_predictions[
        ["sample_id", "outer_fold", "full_ucb"]
    ].copy()
    risk_frame["minimum_cif_ucb"] = complete_predictions["risk_ucb"].map(
        lambda values: float(np.min(np.asarray(values)))
    )
    risk_frame["maximum_cif_ucb"] = complete_predictions["risk_ucb"].map(
        lambda values: float(np.max(np.asarray(values)))
    )
    all_risk = np.concatenate(
        complete_predictions["risk_ucb"].map(np.asarray).tolist()
    )
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.0]),
                np.quantile(all_risk, np.linspace(0.0, 1.0, 31)),
                np.asarray([1.0]),
            )
        )
    )
    curve_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected_parts: list[pd.DataFrame] = []
        for fold in range(5):
            fold_questions = complete.loc[
                complete["outer_fold"].eq(fold)
            ].sort_values("sample_id").reset_index(drop=True)
            fold_predictions = complete_predictions.loc[
                complete_predictions["outer_fold"].eq(fold)
            ].sort_values("sample_id").reset_index(drop=True)
            rule_payload = json.loads(
                (
                    root / "fold_calibrators" / f"outer_{fold}.json"
                ).read_text(encoding="utf-8")
            )
            frozen = dict(rule_payload["decision"])
            rule = DecisionConfig(
                full_ucb_threshold=float(frozen["full_ucb_threshold"]),
                cif_ucb_threshold=float(threshold),
                min_expected_saving=float(frozen["min_expected_saving"]),
                max_bootstrap_std=float(frozen["max_bootstrap_std"]),
                tie_tolerance=float(frozen["tie_tolerance"]),
                routing_compatibility_key=str(
                    frozen["routing_compatibility_key"]
                ),
            )
            calibrated = {
                key_name: (
                    fold_predictions[key_name].to_numpy(dtype=np.float32)
                    if key_name in {"full_mean", "full_std", "full_ucb"}
                    else np.stack(
                        fold_predictions[key_name].map(np.asarray)
                    ).astype(np.float32)
                )
                for key_name in (
                    "full_mean",
                    "full_std",
                    "full_ucb",
                    "risk_mean",
                    "risk_std",
                    "risk_ucb",
                    "cost_mean",
                    "cost_std",
                )
            }
            selected_parts.append(
                make_decisions(
                    fold_questions,
                    calibrated,
                    config=rule,
                    allowed_actions=tuple(Action),
                    outer_fold=fold,
                )
            )
        selected = pd.concat(selected_parts, ignore_index=True)
        from codex.router_v4_0_0.src.full780_evaluate import method_metrics

        choice = {
            str(row.sample_id): Action(int(row.predicted_action))
            for row in selected.itertuples()
        }
        metric = method_metrics(complete, choice)
        curve_rows.append(
            {
                "cif_ucb_threshold": float(threshold),
                "acceptable_rate": float(metric["acceptable_rate"]),
                "compression_induced_failures": int(
                    metric["compression_induced_failures"]
                ),
                "h2o_route_rate": float(metric["h2o_route_rate"]),
                "aggregate_safe_peak_kv_saving": float(
                    metric["aggregate_safe_peak_kv_saving"]
                ),
            }
        )
    curve = pd.DataFrame(curve_rows)
    write_parquet(root / "risk_coverage_rows.parquet", risk_frame)
    curve.to_csv(root / "risk_coverage_curve.csv", index=False)
    write_json(
        root / "EVALUATION_COMPLETE.json",
        {
            "schema_version": "router_v4_0_2.evaluation_complete.v1",
            "passed": bool(evaluation["passed"]),
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        },
    )
    _refresh(root)
    return evaluation


def render(config_path: str | Path) -> Path:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "EVALUATION_COMPLETE.json")
    return render_report(root)


def fit_research_candidate(
    config_path: str | Path, *, device: str
) -> dict[str, Any]:
    config = load_config(config_path)
    root = output_root(config)
    _require(root, "EVALUATION_COMPLETE.json")
    if (root / "RESEARCH_CANDIDATE_READY.json").exists():
        raise RuntimeError("V4.0.2 research candidate already fitted")
    evaluation = json.loads(
        (root / "routing_evaluation.json").read_text(encoding="utf-8")
    )
    if not evaluation["passed"]:
        failed: list[str] = []
        for panel in ("complete6", "common3"):
            failed.extend(
                f"{panel}.{name}"
                for name, passed in evaluation["gates"][panel].items()
                if not passed
            )
        payload = {
            "schema_version": "router_v4_0_2.research_candidate_blocked.v1",
            "reason": "nested_oof_safety_or_gain_gates_failed",
            "failed_gates": failed,
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        }
        write_json(root / "RESEARCH_CANDIDATE_BLOCKED.json", payload)
        _refresh(root)
        raise RuntimeError("V4.0.2 research candidate blocked by OOF gates")
    if not bool(evaluation.get("quality_gates_passed", False)):
        override = {
            "schema_version": "router_v4_0_2.quality_gate_override.v1",
            "authorized": True,
            "reason": str(evaluation["quality_gate_override_reason"]),
            "quality_gates_enforced": False,
            "quality_gates_passed": False,
            "technical_gates_passed": bool(
                evaluation["technical_gates_passed"]
            ),
            "observed_quality_failures_are_retained": True,
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        }
        write_json(root / "QUALITY_GATE_OVERRIDE.json", override)
        blocked_path = root / "RESEARCH_CANDIDATE_BLOCKED.json"
        if blocked_path.exists():
            blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
            blocked["superseded_for_research_training"] = True
            blocked["superseded_by"] = "QUALITY_GATE_OVERRIDE.json"
            write_json(blocked_path, blocked)
        _refresh(root)
    questions = pd.read_parquet(root / "question_dataset.parquet")
    candidates = pd.read_parquet(root / "candidate_dataset.parquet")
    path, history = train_ensemble(
        questions,
        candidates,
        model_config=_model_config(config),
        train_config=_train_config(config, device),
        output_path=root / "research_candidate" / "ensemble.pt",
        metadata={
            "role": "all780_research_only",
            "external_blind_test_required": True,
        },
    )
    write_parquet(root / "research_candidate" / "history.parquet", history)
    payload = {
        "schema_version": "router_v4_0_2.research_candidate_ready.v1",
        "checkpoint": str(path.relative_to(root)),
        "research_only": True,
        "external_blind_test_required": True,
        "quality_gates_passed": bool(evaluation["quality_gates_passed"]),
        "quality_gate_override": (
            "QUALITY_GATE_OVERRIDE.json"
            if not evaluation["quality_gates_passed"]
            else None
        ),
        "routing_enabled": False,
        "production_checkpoint": False,
        "formal_result_eligible": False,
    }
    write_json(root / "RESEARCH_CANDIDATE_READY.json", payload)
    _refresh(root)
    return payload
