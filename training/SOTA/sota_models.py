from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    mode: str


class SOTAMultiExpertFusion(nn.Module):
    """A compact stacked fusion head over frozen image/tabular/fusion experts.

    The base experts carry the heavy visual and tabular representations learned
    by the ablation sweep. This module learns only a metadata-conditioned gate
    and a calibration head over the expert logits, which makes it strong without
    retraining every backbone end to end.
    """

    def __init__(
        self,
        experts: Iterable[tuple[ExpertSpec, nn.Module]],
        metadata_dim: int,
        num_classes: int,
        hidden: int = 192,
        dropout: float = 0.25,
    ):
        super().__init__()
        expert_items = list(experts)
        if not expert_items:
            raise ValueError("SOTAMultiExpertFusion needs at least one expert.")
        self.expert_specs = [spec for spec, _ in expert_items]
        self.experts = nn.ModuleList([module for _, module in expert_items])
        for expert in self.experts:
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

        self.metadata_norm = nn.LayerNorm(metadata_dim)
        self.gate = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(self.experts)),
        )
        self.calibrator = nn.Sequential(
            nn.Linear(len(self.experts) * num_classes + metadata_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden, num_classes),
        )
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def train(self, mode: bool = True):
        super().train(mode)
        for expert in self.experts:
            expert.eval()
        return self

    def _expert_logits(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        needs_input_grad = torch.is_grad_enabled() and (image.requires_grad or metadata.requires_grad)
        logits = []
        context = torch.enable_grad() if needs_input_grad else torch.no_grad()
        with context:
            for spec, expert in zip(self.expert_specs, self.experts):
                if spec.mode == "image_only":
                    out = expert(image)
                elif spec.mode == "tabular_only":
                    out = expert(metadata)
                else:
                    out = expert(image, metadata)
                logits.append(out.float())
        return torch.stack(logits, dim=1)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        metadata = metadata.float()
        expert_logits = self._expert_logits(image, metadata)
        norm_meta = self.metadata_norm(metadata)
        weights = torch.softmax(self.gate(norm_meta), dim=1)
        gated_logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        calibrated_logits = self.calibrator(torch.cat([expert_logits.flatten(1), norm_meta], dim=1))
        alpha = torch.sigmoid(self.mix_logit)
        return alpha * gated_logits + (1.0 - alpha) * calibrated_logits
