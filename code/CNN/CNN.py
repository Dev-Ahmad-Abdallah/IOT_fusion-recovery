import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, drop_path=0.0, se_reduction=16):
        super().__init__()
        mid_channels = out_channels
        expanded_channels = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(expanded_channels, reduction=se_reduction)
        self.drop_path = StochasticDepth(drop_path)

        self.downsample = None
        if stride != 1 or in_channels != expanded_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
            )

    def forward(self, x):
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act(out + identity)
        return out


class SkinLesionResNet(nn.Module):
    """A scratch-trained ResNet-50-style CNN scaled for PAD-UFES images.

    It keeps the residual bottleneck layout but uses fewer base channels, squeeze-
    excitation, stochastic depth, and classifier dropout to reduce overfitting.
    """

    def __init__(
        self,
        num_classes=6,
        base_channels=32,
        layers=(3, 4, 6, 3),
        dropout=0.45,
        drop_path_rate=0.12,
        se_reduction=16,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = base_channels * 2
        stage_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        total_blocks = sum(layers)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0

        self.layer1 = self._make_stage(stage_channels[0], layers[0], stride=1, drop_rates=drop_rates[cursor:cursor + layers[0]], se_reduction=se_reduction)
        cursor += layers[0]
        self.layer2 = self._make_stage(stage_channels[1], layers[1], stride=2, drop_rates=drop_rates[cursor:cursor + layers[1]], se_reduction=se_reduction)
        cursor += layers[1]
        self.layer3 = self._make_stage(stage_channels[2], layers[2], stride=2, drop_rates=drop_rates[cursor:cursor + layers[2]], se_reduction=se_reduction)
        cursor += layers[2]
        self.layer4 = self._make_stage(stage_channels[3], layers[3], stride=2, drop_rates=drop_rates[cursor:cursor + layers[3]], se_reduction=se_reduction)

        final_channels = stage_channels[-1] * BottleneckBlock.expansion
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(final_channels),
            nn.Dropout(p=dropout),
            nn.Linear(final_channels, num_classes),
        )

        self._init_weights()

    def _make_stage(self, out_channels, blocks, stride, drop_rates, se_reduction):
        layers = []
        for idx in range(blocks):
            layers.append(
                BottleneckBlock(
                    self.in_channels,
                    out_channels,
                    stride=stride if idx == 0 else 1,
                    drop_path=drop_rates[idx],
                    se_reduction=se_reduction,
                )
            )
            self.in_channels = out_channels * BottleneckBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


class MetadataOnlyClassifier(nn.Module):
    def __init__(self, metadata_dim, num_classes):
        super().__init__()
        hidden = max(64, metadata_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, metadata_tensor):
        return self.net(metadata_tensor)


class IntermediateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        image_feature_dim = num_classes
        metadata_feature_dim = max(64, metadata_dim)

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_feature_dim),
            nn.BatchNorm1d(metadata_feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_feature_dim, max(128, num_classes * 16)),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(max(128, num_classes * 16), num_classes),
        )

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_features = self.metadata_encoder(metadata_tensor)
        fused = torch.cat([image_logits, metadata_features], dim=1)
        return self.fusion_head(fused)


class LateFusionClassifier(nn.Module):
    def __init__(self, image_model, metadata_dim, num_classes):
        super().__init__()
        self.image_model = image_model
        self.metadata_model = MetadataOnlyClassifier(metadata_dim, num_classes)

    def forward(self, image_tensor, metadata_tensor):
        image_logits = self.image_model(image_tensor)
        metadata_logits = self.metadata_model(metadata_tensor)
        return 0.5 * image_logits + 0.5 * metadata_logits


def build_cnn(num_classes=6, **kwargs):
    drop_path_rate = kwargs.pop("drop_path_rate", 0.12)
    dropout = kwargs.pop("dropout", min(0.6, max(0.2, float(drop_path_rate) + 0.33)))
    base_channels = kwargs.pop("base_channels", 32)
    layers = kwargs.pop("layers", (3, 4, 6, 3))
    se_reduction = kwargs.pop("se_reduction", 16)
    return SkinLesionResNet(
        num_classes=num_classes,
        base_channels=base_channels,
        layers=layers,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        se_reduction=se_reduction,
        **kwargs,
    )
