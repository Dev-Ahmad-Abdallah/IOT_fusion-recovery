from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

from CNN.CNN import build_cnn
from Transformer.Transformer import build_transformer


class MLPTabular(nn.Module):
    def __init__(self, metadata_dim: int, num_classes: int, hidden: int | None = None, dropout: float = 0.25):
        super().__init__()
        hidden = hidden or max(128, metadata_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.75),
        )
        self.head = nn.Linear(hidden // 2, num_classes)
        self.feature_dim = hidden // 2

    def forward_features(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.encoder(metadata)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(metadata))


class TabTransformer(nn.Module):
    """A compact transformer over tabular columns represented as continuous values."""

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.value_proj = nn.Linear(1, embed_dim)
        self.column_embed = nn.Parameter(torch.zeros(1, metadata_dim, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        self.feature_dim = embed_dim
        nn.init.trunc_normal_(self.column_embed, std=0.02)

    def forward_features(self, metadata: torch.Tensor) -> torch.Tensor:
        x = self.value_proj(metadata.unsqueeze(-1)) + self.column_embed
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.dropout(x)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(metadata))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetClassifier(nn.Module):
    def __init__(self, num_classes: int, base: int = 24, dropout: float = 0.35):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        return self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class ConvTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.norm = nn.LayerNorm(channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x + self.local(x))
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.attn(self.norm(seq))
        return x + seq.transpose(1, 2).reshape(b, c, h, w)


class ConformerTiny(nn.Module):
    def __init__(self, num_classes: int, channels: int = 96):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels // 2, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ConvTransformerBlock(channels, num_heads=4),
            ConvTransformerBlock(channels, num_heads=4),
            ConvTransformerBlock(channels, num_heads=4),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(channels, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CvTStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, heads: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7 if stride > 1 else 3, stride=stride, padding=3 if stride > 1 else 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.block = ConvTransformerBlock(out_channels, num_heads=heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.embed(x))


class CvTTiny(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stage1 = CvTStage(3, 48, stride=4, heads=3)
        self.stage2 = CvTStage(48, 96, stride=2, heads=4)
        self.stage3 = CvTStage(96, 160, stride=2, heads=5)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.25), nn.Linear(160, num_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage3(self.stage2(self.stage1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class EarlyFusion(nn.Module):
    def __init__(self, image_model: nn.Module, metadata_dim: int):
        super().__init__()
        self.image_model = image_model
        self.conditioner = nn.Sequential(nn.Linear(metadata_dim, 64), nn.SiLU(inplace=True), nn.Linear(64, 6))

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        scale_bias = torch.tanh(self.conditioner(metadata)).view(-1, 6, 1, 1)
        scale, bias = scale_bias[:, :3], scale_bias[:, 3:]
        return self.image_model(image * (1.0 + 0.10 * scale) + 0.10 * bias)


class IntermediateFusion(nn.Module):
    def __init__(self, image_model: nn.Module, tabular_model: nn.Module, num_classes: int):
        super().__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        tab_dim = getattr(tabular_model, "feature_dim", num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(num_classes + tab_dim, max(128, num_classes * 24)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(max(128, num_classes * 24), num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_logits = self.image_model(image)
        tab_features = self.tabular_model.forward_features(metadata)
        return self.fusion_head(torch.cat([image_logits, tab_features], dim=1))


class LateFusion(nn.Module):
    def __init__(self, image_model: nn.Module, tabular_model: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.image_model(image) + 0.5 * self.tabular_model(metadata)


class MoEGate(nn.Module):
    def __init__(self, metadata_dim: int, hidden: int = 64, num_experts: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_experts),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(metadata), dim=1)


class TwoExpertMoE(nn.Module):
    def __init__(self, image_expert: nn.Module, tabular_expert: nn.Module, metadata_dim: int):
        super().__init__()
        self.image_expert = image_expert
        self.tabular_expert = tabular_expert
        for param in self.image_expert.parameters():
            param.requires_grad = False
        for param in self.tabular_expert.parameters():
            param.requires_grad = False
        self.gate = MoEGate(metadata_dim, num_experts=2)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_logits = self.image_expert(image)
        tabular_logits = self.tabular_expert(metadata)
        weights = self.gate(metadata)
        return weights[:, 0:1] * image_logits + weights[:, 1:2] * tabular_logits


def build_image_model(name: str, num_classes: int) -> nn.Module:
    if name == "cnn":
        return build_cnn(num_classes=num_classes, base_channels=32)
    if name == "transformer":
        return build_transformer(num_classes=num_classes)
    if name == "unet":
        return UNetClassifier(num_classes=num_classes)
    if name == "efficientnet":
        if timm is None:
            raise RuntimeError("timm is required for EfficientNet.")
        return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    if name == "conformer":
        return ConformerTiny(num_classes=num_classes)
    if name == "cvt":
        return CvTTiny(num_classes=num_classes)
    raise ValueError(f"Unknown image model: {name}")


def build_tabular_model(name: str, metadata_dim: int, num_classes: int) -> nn.Module:
    if name == "mlp":
        return MLPTabular(metadata_dim=metadata_dim, num_classes=num_classes)
    if name == "tabtransformer":
        return TabTransformer(metadata_dim=metadata_dim, num_classes=num_classes)
    raise ValueError(f"Unknown tabular model: {name}")


@dataclass(frozen=True)
class ModelSpec:
    experiment_name: str
    image_model: str | None
    tabular_model: str | None
    mode: str


def build_model(spec: ModelSpec, metadata_dim: int, num_classes: int) -> nn.Module:
    if spec.mode == "image_only":
        assert spec.image_model is not None
        return build_image_model(spec.image_model, num_classes)
    if spec.mode == "tabular_only":
        assert spec.tabular_model is not None
        return build_tabular_model(spec.tabular_model, metadata_dim, num_classes)

    assert spec.image_model is not None and spec.tabular_model is not None
    image_model = build_image_model(spec.image_model, num_classes)
    tabular_model = build_tabular_model(spec.tabular_model, metadata_dim, num_classes)
    if spec.mode == "early_fusion":
        return EarlyFusion(image_model, metadata_dim)
    if spec.mode == "intermediate_fusion":
        return IntermediateFusion(image_model, tabular_model, num_classes)
    if spec.mode == "late_fusion":
        return LateFusion(image_model, tabular_model)
    raise ValueError(f"Unknown mode: {spec.mode}")


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "image_only":
        return model(batch["image"])
    if mode == "tabular_only":
        return model(batch["metadata"])
    return model(batch["image"], batch["metadata"])
