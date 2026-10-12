from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from codex.router_v4_0_0.src.artifacts import (
    build_manifest,
    sha256_file,
    write_json,
)


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _method_rows(metrics: dict[str, dict]) -> list[str]:
    rows = [
        "| Method | Acceptable | H2O route | Safe aggregate saving | CIF |",
        "|---|---:|---:|---:|---:|",
    ]
    order = [
        "Fixed Full",
        "Fixed R100_I32",
        "Fixed R080_I32",
        "Fixed R060_I32",
        "Fixed R040_I32",
        "Fixed R020_I32",
        "legacy_absolute_linear",
        "frozen_conditional_mlp",
        "partial_finetune_conditional",
        "V4.0.2 Shared Candidate Risk",
        "Oracle",
    ]
    for name in order:
        if name not in metrics:
            continue
        value = metrics[name]
        rows.append(
            f"| {name} | {_pct(value['acceptable_rate'])} | "
            f"{_pct(value['h2o_route_rate'])} | "
            f"{_pct(value['aggregate_safe_peak_kv_saving'])} | "
            f"{int(value['compression_induced_failures'])} |"
        )
    return rows


def _failure_rows(metrics: dict[str, dict]) -> list[str]:
    rows = [
        "| Policy | AUROC | AUPRC | Recall | Brier | Failure / Observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, value in metrics.items():
        rows.append(
            f"| {policy} | {float(value['roc_auc']):.3f} | "
            f"{float(value['average_precision']):.3f} | "
            f"{_pct(value['recall'])} | {float(value['brier']):.3f} | "
            f"{int(value['failure_support'])} / "
            f"{int(value['observed_support'])} |"
        )
    return rows


def _fold_rows(metrics: dict[str, dict]) -> list[str]:
    rows = [
        "| Outer fold | Rows | Full acceptable | Router acceptable | "
        "H2O route | Safe saving | CIF |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, value in sorted(metrics.items(), key=lambda item: int(item[0])):
        fixed = value["Fixed Full"]
        router = value["V4.0.2"]
        rows.append(
            f"| {fold} | {int(value['rows'])} | "
            f"{_pct(fixed['acceptable_rate'])} | "
            f"{_pct(router['acceptable_rate'])} | "
            f"{_pct(router['h2o_route_rate'])} | "
            f"{_pct(router['aggregate_safe_peak_kv_saving'])} | "
            f"{int(router['compression_induced_failures'])} |"
        )
    return rows


def _slice_rows(metrics: dict[str, dict]) -> list[str]:
    rows = [
        "| Slice | Rows | Full acceptable | Router acceptable | "
        "H2O route | Safe saving | CIF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        fixed = value["Fixed Full"]
        router = value["V4.0.2 common3"]
        rows.append(
            f"| {name} | {int(value['rows'])} | "
            f"{_pct(fixed['acceptable_rate'])} | "
            f"{_pct(router['acceptable_rate'])} | "
            f"{_pct(router['h2o_route_rate'])} | "
            f"{_pct(router['aggregate_safe_peak_kv_saving'])} | "
            f"{int(router['compression_induced_failures'])} |"
        )
    return rows


def render_report(root: Path) -> Path:
    evaluation = json.loads(
        (root / "routing_evaluation.json").read_text(encoding="utf-8")
    )
    audit = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
    six = evaluation["complete_six_policy_metrics"]
    common = evaluation["overall_common3_metrics"]
    primary = six["V4.0.2 Shared Candidate Risk"]
    fixed = six["Fixed Full"]
    risk = evaluation["routed_policy_risk"]["ALL_H2O"]
    historical = evaluation["historical_v4_0_1"][
        "complete_six_policy_metrics"
    ]
    complete_comparison = {
        **historical,
        "V4.0.2 Shared Candidate Risk": primary,
    }
    complete_comparison["Oracle"] = six["Oracle"]
    curve = pd.read_csv(root / "risk_coverage_curve.csv")
    zero_cif = curve.loc[curve["compression_induced_failures"].eq(0)]
    best_zero_cif = zero_cif.sort_values(
        "aggregate_safe_peak_kv_saving"
    ).iloc[-1]
    candidate_lines: list[str] = []
    ready_path = root / "RESEARCH_CANDIDATE_READY.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        checkpoint = root / ready["checkpoint"]
        history = pd.read_parquet(
            root / "research_candidate" / "history.parquet"
        )
        candidate_lines = [
            "## All-780 research-only bundle",
            "",
            f"- Checkpoint: `{ready['checkpoint']}`.",
            f"- SHA256: `{sha256_file(checkpoint)}`.",
            f"- Training rows/epochs/bootstrap members: "
            f"**780 / {int(history['epoch'].max())} / 10**.",
            "- Fitted only after the explicit quality-gate override.",
            "- It is not re-evaluated on the same 780 training rows; Nested "
            "OOF remains the performance estimate.",
            "- `research_only=true`, `routing_enabled=false`, "
            "`production_checkpoint=false`.",
            "",
        ]
    lines = [
        "# Router V4.0.2 Shared Candidate Risk Report",
        "",
        "> Same-data internal development Nested OOF. This is not a blind or "
        "production result.",
        "",
        "## Data and architecture",
        "",
        f"- Questions/Documents: **{audit['questions']} / {audit['documents']}**.",
        f"- Complete six-policy panels: **{audit['complete_panels']}**.",
        f"- Conditional CIF counts: **{audit['conditional_cif']}**.",
        "- Frozen BGE/V3 z64; shared question-policy linear risk scorer; "
        "10-member document bootstrap; cross-fitted calibration.",
        "",
        "## Primary Complete-6 result",
        "",
        *_method_rows(six),
        "",
        "## Complete-6 comparison with frozen V4.0.1 baselines",
        "",
        *_method_rows(complete_comparison),
        "",
        "## Secondary Common-3 result",
        "",
        *_method_rows(common),
        "",
        "## Safety verdict",
        "",
        f"- Technical execution gates passed: "
        f"**{evaluation['technical_gates_passed']}**.",
        f"- Original zero-error quality target passed: "
        f"**{evaluation['quality_gates_passed']}**.",
        f"- Quality gates enforced as a training blocker: "
        f"**{evaluation['quality_gates_enforced']}**.",
        f"- Override audit reason: "
        f"**{evaluation['quality_gate_override_reason']}**.",
        f"- Acceptable delta vs Fixed Full: "
        f"**{100*(primary['acceptable_rate']-fixed['acceptable_rate']):+.2f} pp**.",
        f"- Routed H2O rows / CIF: **{risk['routed']} / "
        f"{risk['compression_induced_failures']}**.",
        "- One-sided 95% Clopper-Pearson CIF upper bound: "
        f"**{_pct(risk['one_sided_95pct_clopper_pearson_upper']) if risk['one_sided_95pct_clopper_pearson_upper'] is not None else 'N/A'}**.",
        f"- Safe Oracle opportunity capture: "
        f"**{_pct(evaluation['safe_oracle_opportunity_capture'])}**.",
        f"- Policy distribution: **{primary['policy_distribution']}**.",
        f"- Full fallback rate: **{_pct(primary['full_fallback_rate'])}**.",
        "",
        "The descriptive OOF risk-coverage curve contains a zero-CIF point at "
        f"{_pct(best_zero_cif['h2o_route_rate'])} H2O routing and "
        f"{_pct(best_zero_cif['aggregate_safe_peak_kv_saving'])} safe saving. "
        "This is post-hoc on exposed Outer OOF outcomes and is not used to "
        "replace the frozen cross-fitted rules.",
        "",
        "## Failure prediction diagnostics",
        "",
        *_failure_rows(evaluation["failure_metrics"]),
        "",
        "## Outer-fold stability (Complete-6)",
        "",
        *_fold_rows(evaluation["outer_fold_metrics"]),
        "",
        "## Common-3 slices",
        "",
        *_slice_rows(evaluation["slices"]),
        "",
        *candidate_lines,
        "## Gate details",
        "",
        "```json",
        json.dumps(evaluation["gates"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Interpretation",
        "",
        "- Correctness is evaluated relative to Fixed Full; H2O rescue is not "
        "used to justify compression.",
        "- H2O route rate is reported but is not a pass gate.",
        "- The user-authorized override permits research-only all-data "
        "training; it does not erase the observed CIF or make the router safe.",
        "- The 780 outcomes were already exposed before V4.0.2 was designed, "
        "so this result cannot establish blind generalization.",
        "- `routing_enabled=false`, `production_checkpoint=false`, and "
        "`formal_result_eligible=false` remain mandatory.",
    ]
    report = root / "final_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = pd.DataFrame(
        [
            {
                "method": name,
                "acceptable": values["acceptable_rate"],
                "saving": values["aggregate_safe_peak_kv_saving"],
            }
            for name, values in six.items()
            if name in {
                "Fixed Full",
                "Fixed R100_I32",
                "Fixed R080_I32",
                "Fixed R060_I32",
                "Fixed R040_I32",
                "Fixed R020_I32",
                "V4.0.2 Shared Candidate Risk",
                "Oracle",
            }
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(frame["method"], frame["acceptable"])
    axes[0].set_title("Acceptable Rate")
    axes[0].tick_params(axis="x", rotation=55)
    axes[1].bar(frame["method"], frame["saving"])
    axes[1].set_title("Safe Aggregate Peak-KV Saving")
    axes[1].tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(root / "routing_methods_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(
        100.0 * curve["h2o_route_rate"],
        100.0 * curve["aggregate_safe_peak_kv_saving"],
        marker="o",
        markersize=3,
        label="Safe Peak-KV saving",
    )
    failures = curve["compression_induced_failures"].to_numpy()
    unsafe = failures > 0
    axis.scatter(
        100.0 * curve.loc[unsafe, "h2o_route_rate"],
        100.0 * curve.loc[unsafe, "aggregate_safe_peak_kv_saving"],
        color="red",
        s=28,
        label="CIF > 0",
    )
    axis.set_xlabel("H2O Route Rate (%)")
    axis.set_ylabel("Aggregate Safe Peak-KV Saving (%)")
    axis.set_title("V4.0.2 OOF Risk-Coverage Diagnostic")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(root / "risk_coverage_curve.png", dpi=160)
    plt.close(figure)

    write_json(root / "artifact_manifest.json", build_manifest(root))
    status_name = (
        "EXPERIMENT_COMPLETE_INTERNAL.json"
        if evaluation["passed"] and evaluation["quality_gates_passed"]
        else (
            "EXPERIMENT_COMPLETE_INTERNAL_WITH_QUALITY_WARNINGS.json"
            if evaluation["passed"]
            else "EXPERIMENT_FINISHED_WITH_LIMITS.json"
        )
    )
    write_json(
        root / status_name,
        {
            "schema_version": "router_v4_0_2.experiment_status.v1",
            "passed": bool(evaluation["passed"]),
            "technical_gates_passed": bool(
                evaluation["technical_gates_passed"]
            ),
            "quality_gates_passed": bool(
                evaluation["quality_gates_passed"]
            ),
            "quality_gates_enforced": bool(
                evaluation["quality_gates_enforced"]
            ),
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        },
    )
    write_json(root / "artifact_manifest.json", build_manifest(root))
    return report
