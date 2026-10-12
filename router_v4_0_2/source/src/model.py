from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    members: int = 10
    member_seeds: tuple[int, ...] = tuple(range(42, 52))
    base_dimension: int = 80
    risk_dimension: int = 166
    cost_dimension: int = 22
    cost_hidden: int = 16

    def __post_init__(self) -> None:
        if len(self.member_seeds) != self.members:
            raise ValueError("member seed count mismatch")


class EnsembleLinear(nn.Module):
    def __init__(self, dimension: int, config: ModelConfig) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(dimension, elementwise_affine=False)
        self.weight = nn.Parameter(torch.empty(config.members, dimension))
        self.bias = nn.Parameter(torch.zeros(config.members))
        for index, seed in enumerate(config.member_seeds):
            generator = torch.Generator().manual_seed(int(seed))
            self.weight.data[index].normal_(
                mean=0.0, std=dimension**-0.5, generator=generator
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer(values)
        return torch.einsum("nd,md->nm", normalized, self.weight) + self.bias


class EnsembleCostHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(
            config.cost_dimension, elementwise_affine=False
        )
        self.weight1 = nn.Parameter(
            torch.empty(
                config.members, config.cost_hidden, config.cost_dimension
            )
        )
        self.bias1 = nn.Parameter(
            torch.zeros(config.members, config.cost_hidden)
        )
        self.weight2 = nn.Parameter(
            torch.empty(config.members, config.cost_hidden)
        )
        self.bias2 = nn.Parameter(torch.zeros(config.members))
        for index, seed in enumerate(config.member_seeds):
            generator = torch.Generator().manual_seed(int(seed) + 1000)
            self.weight1.data[index].normal_(
                mean=0.0,
                std=config.cost_dimension**-0.5,
                generator=generator,
            )
            self.weight2.data[index].normal_(
                mean=0.0,
                std=config.cost_hidden**-0.5,
                generator=generator,
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.normalizer(values)
        hidden = torch.einsum(
            "nd,mhd->nmh", normalized, self.weight1
        ) + self.bias1.unsqueeze(0)
        return (
            torch.einsum("nmh,mh->nm", F.gelu(hidden), self.weight2)
            + self.bias2
        )


class SharedCandidateEnsemble(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.full_head = EnsembleLinear(config.base_dimension, config)
        self.risk_head = EnsembleLinear(config.risk_dimension, config)
        self.cost_head = EnsembleCostHead(config)

    def full_logits(self, base80: torch.Tensor) -> torch.Tensor:
        return self.full_head(base80)

    def risk_logits(self, risk166: torch.Tensor) -> torch.Tensor:
        return self.risk_head(risk166)

    def cost_ratios(self, cost22: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cost_head(cost22))


def save_checkpoint(
    path: str | Path,
    model: SharedCandidateEnsemble,
    *,
    metadata: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "router_v4_0_2.checkpoint.v1",
        "model_config": {
            **asdict(model.config),
            "member_seeds": list(model.config.member_seeds),
        },
        "state_dict": model.state_dict(),
        "metadata": {
            **metadata,
            "diagnostic_only": True,
            "routing_enabled": False,
            "production_checkpoint": False,
            "formal_result_eligible": False,
        },
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path, *, map_location: str = "cpu"
) -> tuple[SharedCandidateEnsemble, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != "router_v4_0_2.checkpoint.v1":
        raise ValueError("unexpected V4.0.2 checkpoint schema")
    values = dict(payload["model_config"])
    values["member_seeds"] = tuple(values["member_seeds"])
    model = SharedCandidateEnsemble(ModelConfig(**values))
    model.load_state_dict(payload["state_dict"])
    return model, dict(payload["metadata"])

