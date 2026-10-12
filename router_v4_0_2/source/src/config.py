from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != "router_v4_0_2.config.v1":
        raise ValueError("unexpected V4.0.2 config schema")
    status = config["status"]
    if any(
        bool(status[key])
        for key in (
            "routing_enabled",
            "production_checkpoint",
            "formal_result_eligible",
        )
    ):
        raise ValueError("V4.0.2 must remain a non-production internal pilot")
    if config["decision"]["allow_h2o_rescue"]:
        raise ValueError("H2O rescue must remain disabled")
    acceptance = config["acceptance"]
    if not isinstance(acceptance.get("enforce_quality_gates"), bool):
        raise ValueError("acceptance.enforce_quality_gates must be boolean")
    if not acceptance["enforce_quality_gates"] and not str(
        acceptance.get("quality_gate_override_reason", "")
    ).strip():
        raise ValueError("disabled quality gates require an audit reason")
    seeds = list(config["training"]["bootstrap_seeds"])
    if len(seeds) != int(config["training"]["bootstrap_members"]):
        raise ValueError("bootstrap seed/member mismatch")
    return config


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root"])
