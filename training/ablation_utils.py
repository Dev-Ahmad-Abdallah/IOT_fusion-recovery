from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from sklearn.metrics import (
    accuracy_score,
    auc as sklearn_auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


REPO_ROOT = Path(os.getenv("IOT_FUSION_REPO", "/home/zeus/content/IOT_fusion"))
DATA_ROOT = Path(os.getenv("IOT_FUSION_DATA", "/home/zeus/content/dataset"))
TRAIN_CSV = DATA_ROOT / "processed" / "train_tabular_processed.csv"
TEST_CSV = DATA_ROOT / "processed" / "test_tabular_processed.csv"
TRAIN_TENSOR_DIR = DATA_ROOT / "processed_images" / "train_tensors"
TEST_TENSOR_DIR = DATA_ROOT / "processed_images" / "test_tensors"
WANDB_ENV = REPO_ROOT / "W&B.env"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablations"
RESULT_LOGS_DIR = REPO_ROOT / "Results_Logs"
RESULT_FIGURES_DIR = REPO_ROOT / "Results_Figures"
RESULT_XAI_DIR = REPO_ROOT / "Results_XAI"

CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from CNN.CNN import (  # noqa: E402
    IntermediateFusionClassifier as CNNIntermediateFusionClassifier,
    LateFusionClassifier as CNNLateFusionClassifier,
    MetadataOnlyClassifier as CNNMetadataOnlyClassifier,
    build_cnn,
)
from Transformer.Transformer import (  # noqa: E402
    IntermediateFusionClassifier as TransformerIntermediateFusionClassifier,
    LateFusionClassifier as TransformerLateFusionClassifier,
    MetadataOnlyClassifier as TransformerMetadataOnlyClassifier,
    build_transformer,
)


ABLATIONS = [
    ("resnet50", "images_only", "resnet50_img_only"),
    ("cnn", "images_only", "cnn_img_only"),
    ("cnn", "metadata_only", "cnn_meta_only"),
    ("cnn", "early_fusion", "cnn_early_fusion"),
    ("cnn", "intermediate_fusion", "cnn_inter_fusion"),
    ("cnn", "late_fusion", "cnn_late_fusion"),
    ("transformer", "images_only", "transformer_img_only"),
    ("transformer", "metadata_only", "transformer_meta_only"),
    ("transformer", "early_fusion", "transformer_early_fusion"),
    ("transformer", "intermediate_fusion", "transformer_inter_fusion"),
    ("transformer", "late_fusion", "transformer_late_fusion"),
]


@dataclass(frozen=True)
class DataInfo:
    feature_cols: list[str]
    class_names: list[str]
    class_to_idx: dict[str, int]

    @property
    def metadata_dim(self) -> int:
        return len(self.feature_cols)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_wandb_env(path: Path = WANDB_ENV) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _experiment_result_dirs(experiment_name: str) -> tuple[Path, Path, Path]:
    logs_dir = RESULT_LOGS_DIR / experiment_name
    figures_dir = RESULT_FIGURES_DIR / experiment_name
    xai_dir = RESULT_XAI_DIR / experiment_name
    for path in (logs_dir, figures_dir, xai_dir):
        path.mkdir(parents=True, exist_ok=True)
    return logs_dir, figures_dir, xai_dir


def _copy_if_different(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _read_source_csvs() -> pd.DataFrame:
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["tensor_source_split"] = "train"
    test_df = pd.read_csv(TEST_CSV)
    test_df["tensor_source_split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)


def get_data_info() -> DataInfo:
    df = _read_source_csvs()
    feature_cols = [c for c in df.columns if c not in {"img_id", "patient_id", "label_raw", "tensor_source_split"}]
    class_names = sorted(df["label_raw"].astype(str).unique().tolist())
    return DataInfo(
        feature_cols=feature_cols,
        class_names=class_names,
        class_to_idx={name: idx for idx, name in enumerate(class_names)},
    )


def make_split_dataframes(seed: int = 42) -> dict[str, pd.DataFrame]:
    df = _read_source_csvs().reset_index(drop=True)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label_raw"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_raw"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


class FusionTensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, info: DataInfo, input_mode: str, image_transform: nn.Module | None = None):
        self.df = df.reset_index(drop=True)
        self.info = info
        self.input_mode = input_mode
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        label = self.info.class_to_idx[str(row["label_raw"])]
        metadata = torch.tensor(row[self.info.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        batch = {
            "label": torch.tensor(label, dtype=torch.long),
            "metadata_tensor": metadata,
        }
        if self.input_mode != "metadata_only":
            tensor_dir = TRAIN_TENSOR_DIR if row["tensor_source_split"] == "train" else TEST_TENSOR_DIR
            tensor_name = Path(str(row["img_id"])).stem + "_prep.pt"
            image = torch.load(tensor_dir / tensor_name, map_location="cpu").float()
            if self.image_transform is not None:
                image = self.image_transform(image)
            batch["image"] = image
        return batch


class RandomTensorChannelJitter(nn.Module):
    def __init__(self, scale: float = 0.12, shift: float = 0.05, p: float = 0.7):
        super().__init__()
        self.scale = float(scale)
        self.shift = float(shift)
        self.p = float(p)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return image
        channel_shape = (image.shape[0],) + (1,) * (image.ndim - 1)
        scale = image.new_empty(channel_shape).uniform_(1.0 - self.scale, 1.0 + self.scale)
        shift = image.new_empty(channel_shape).uniform_(-self.shift, self.shift)
        return image * scale + shift


class RandomGaussianNoise(nn.Module):
    def __init__(self, std: float = 0.04, p: float = 0.35):
        super().__init__()
        self.std = float(std)
        self.p = float(p)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return image
        return image + torch.randn_like(image) * self.std


def get_train_image_augmentation(enabled: bool = True) -> nn.Module | None:
    if not enabled:
        return None
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.82, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25, fill=0.0),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10), fill=0.0),
            RandomTensorChannelJitter(scale=0.12, shift=0.05, p=0.7),
            transforms.RandomAutocontrast(p=0.25),
            RandomGaussianNoise(std=0.04, p=0.35),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
        ]
    )


def make_loader(
    split: str,
    input_mode: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    info = get_data_info()
    split_dfs = make_split_dataframes(seed=seed)
    dataset = FusionTensorDataset(
        split_dfs[split],
        info,
        input_mode,
        image_transform=get_train_image_augmentation(split == "train" and input_mode != "metadata_only"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


class AdditiveEarlyFusion(nn.Module):
    def __init__(self, image_model: nn.Module, metadata_dim: int):
        super().__init__()
        self.image_model = image_model
        self.film = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 6))

    def forward(self, image_tensor: torch.Tensor, metadata_tensor: torch.Tensor) -> torch.Tensor:
        scale_bias = torch.tanh(self.film(metadata_tensor)).view(-1, 6, 1, 1)
        scale, bias = scale_bias[:, :3], scale_bias[:, 3:]
        fused_image = image_tensor * (1.0 + 0.10 * scale) + 0.10 * bias
        return self.image_model(fused_image)


def build_image_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "cnn":
        return build_cnn(num_classes=num_classes, base_channels=32)
    if model_name == "transformer":
        return build_transformer(num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_ablation_model(model_name: str, input_mode: str, info: DataInfo) -> nn.Module:
    if input_mode == "images_only":
        return build_image_model(model_name, info.num_classes)

    if model_name == "resnet50":
        metadata_cls = CNNMetadataOnlyClassifier
        intermediate_cls = CNNIntermediateFusionClassifier
        late_cls = CNNLateFusionClassifier
    elif model_name == "cnn":
        metadata_cls = CNNMetadataOnlyClassifier
        intermediate_cls = CNNIntermediateFusionClassifier
        late_cls = CNNLateFusionClassifier
    elif model_name == "transformer":
        metadata_cls = TransformerMetadataOnlyClassifier
        intermediate_cls = TransformerIntermediateFusionClassifier
        late_cls = TransformerLateFusionClassifier
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if input_mode == "metadata_only":
        return metadata_cls(metadata_dim=info.metadata_dim, num_classes=info.num_classes)

    image_model = build_image_model(model_name, info.num_classes)
    if input_mode == "early_fusion":
        return AdditiveEarlyFusion(image_model=image_model, metadata_dim=info.metadata_dim)
    if input_mode == "intermediate_fusion":
        return intermediate_cls(image_model=image_model, metadata_dim=info.metadata_dim, num_classes=info.num_classes)
    if input_mode == "late_fusion":
        return late_cls(image_model=image_model, metadata_dim=info.metadata_dim, num_classes=info.num_classes)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, input_mode: str) -> dict[str, torch.Tensor]:
    moved = {"label": batch["label"].to(device, non_blocking=True)}
    if input_mode != "images_only":
        moved["metadata_tensor"] = batch["metadata_tensor"].to(device, non_blocking=True)
    if input_mode != "metadata_only":
        moved["image"] = batch["image"].to(device, non_blocking=True)
    return moved


def forward_by_mode(model: nn.Module, batch: dict[str, torch.Tensor], input_mode: str) -> torch.Tensor:
    if input_mode == "images_only":
        return model(batch["image"])
    if input_mode == "metadata_only":
        return model(batch["metadata_tensor"])
    return model(batch["image"], batch["metadata_tensor"])


def init_wandb_run(run_name: str, model_name: str, input_mode: str, epochs: int, batch_size: int) -> Any:
    load_wandb_env()
    try:
        import wandb
    except Exception as exc:
        print(f"W&B unavailable, continuing without logging: {exc}")
        return None

    try:
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", "iot fusion"),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=run_name,
            tags=["full", model_name, input_mode, "80-10-10"],
            config={
                "model_name": model_name,
                "input_mode": input_mode,
                "data_root": str(DATA_ROOT),
                "epochs": epochs,
                "batch_size": batch_size,
                "split": "stratified 80 train / 10 validation / 10 test",
            },
        )
    except Exception as exc:
        print(f"W&B init failed, continuing without logging: {exc}")
        return None


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    aucs: list[float] = []
    for idx in range(num_classes):
        positives = int(y_bin[:, idx].sum())
        negatives = int(len(y_bin) - positives)
        if positives == 0 or negatives == 0:
            continue
        try:
            auc_value = float(roc_auc_score(y_bin[:, idx], y_prob[:, idx]))
        except ValueError:
            continue
        if not np.isnan(auc_value):
            aucs.append(auc_value)
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred) * 100.0),
        "bacc": float(balanced_accuracy_score(y_true, y_pred) * 100.0),
        "auc": _safe_auc(y_true, y_prob, num_classes),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _class_weights(split_df: pd.DataFrame, info: DataInfo, device: torch.device, power: float = 0.5) -> torch.Tensor:
    counts = split_df["label_raw"].value_counts().reindex(info.class_names).fillna(0).to_numpy(dtype=np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.power(weights, power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _sample_weights(split_df: pd.DataFrame, info: DataInfo) -> torch.Tensor:
    counts = split_df["label_raw"].value_counts().reindex(info.class_names).fillna(0).to_dict()
    weights = split_df["label_raw"].map(lambda label: 1.0 / max(float(counts[label]), 1.0)).to_numpy(dtype=np.float64)
    return torch.tensor(weights, dtype=torch.double)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    input_mode: str,
    amp: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    desc: str,
    num_classes: int,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_seen = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    iterator = tqdm(loader, desc=desc, leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in iterator:
            batch = move_batch(batch, device, input_mode)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and amp)):
                logits = forward_by_mode(model, batch, input_mode)
                loss = criterion(logits, batch["label"])

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            labels = batch["label"].detach()
            probs = torch.softmax(logits.detach(), dim=1)
            preds = probs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

            running_loss = total_loss / max(1, total_seen)
            running_acc = accuracy_score(np.concatenate(all_true), np.concatenate(all_pred)) * 100.0
            iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}%")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    metrics = _classification_metrics(y_true, y_pred, y_prob, num_classes)
    metrics["loss"] = total_loss / max(1, total_seen)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


def save_xai_artifacts(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    input_mode: str,
    info: DataInfo,
    run_name: str,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    paths: dict[str, str] = {}

    xai_batch = {k: v[: min(4, len(v))].detach().clone() for k, v in batch.items()}
    if "image" in xai_batch:
        image = xai_batch["image"].detach().clone().requires_grad_(True)
        local = dict(xai_batch)
        local["image"] = image
        logits = forward_by_mode(model, local, input_mode)
        target = logits.argmax(dim=1)
        score = logits.gather(1, target.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph="metadata_tensor" in xai_batch)
        saliency = image.grad.detach().abs().amax(dim=1).cpu().numpy()
        path = output_dir / f"{run_name}_image_saliency.npy"
        np.save(path, saliency)
        paths["image_saliency"] = str(path)

    if "metadata_tensor" in xai_batch:
        metadata = xai_batch["metadata_tensor"].detach().clone().requires_grad_(True)
        local = dict(xai_batch)
        local["metadata_tensor"] = metadata
        logits = forward_by_mode(model, local, input_mode)
        target = logits.argmax(dim=1)
        score = logits.gather(1, target.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        importance = metadata.grad.detach().abs().mean(dim=0).cpu().numpy()
        top_idx = np.argsort(-importance)[:20]
        path = output_dir / f"{run_name}_metadata_xai_top20.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "feature", "gradient_importance"])
            for rank, idx in enumerate(top_idx, start=1):
                writer.writerow([rank, info.feature_cols[int(idx)], float(importance[int(idx)])])
        paths["metadata_importance"] = str(path)

    model.train()
    return paths


def _save_split_summary(split_dfs: dict[str, pd.DataFrame], path: Path) -> None:
    rows = []
    for split_name, df in split_dfs.items():
        counts = df["label_raw"].value_counts().to_dict()
        row = {"split": split_name, "rows": len(df)}
        row.update({f"class_{k}": v for k, v in counts.items()})
        rows.append(row)
    pd.DataFrame(rows).fillna(0).to_csv(path, index=False)


def _save_epoch_log(history: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in history.to_dict(orient="records"):
        lines.append(
            f"Epoch {int(row['epoch']):03d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% "
            f"train_bacc={row['train_bacc']:.2f}% train_auc={row['train_auc']:.4f} "
            f"train_f1={row['train_f1_macro']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}% "
            f"val_bacc={row['val_bacc']:.2f}% val_auc={row['val_auc']:.4f} "
            f"val_f1={row['val_f1_macro']:.4f} | lr={row['lr']:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_loss_curve(history: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_metric_curve(history: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation accuracy")
    plt.plot(history["epoch"], history["train_auc"] * 100.0, label="Train AUC x100")
    plt.plot(history["epoch"], history["val_auc"] * 100.0, label="Validation AUC x100")
    plt.plot(history["epoch"], history["train_f1_macro"] * 100.0, label="Train macro F1 x100")
    plt.plot(history["epoch"], history["val_f1_macro"] * 100.0, label="Validation macro F1 x100")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Epoch Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], output_path: Path) -> None:
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        if y_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        class_auc = sklearn_auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} AUC={class_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _log_artifacts_to_wandb(wandb_run: Any, artifact_paths: dict[str, Path]) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
        for key, path in artifact_paths.items():
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                wandb_run.log({key: wandb.Image(str(path))})
            else:
                wandb_run.save(str(path))
    except Exception as exc:
        print(f"W&B artifact logging skipped: {exc}")


def run_training(
    model_name: str,
    input_mode: str,
    experiment_name: str,
    *,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    amp: bool = True,
    num_workers: int = 2,
    wandb_enabled: bool = True,
    xai_enabled: bool = True,
    device_name: str = "cuda",
    split_seed: int = 42,
    augment_train: bool = True,
    class_weighted_loss: bool = True,
    class_weight_power: float = 0.5,
    class_balanced_sampler: bool = False,
    early_stopping_patience: int = 12,
    min_epochs: int = 15,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "xai").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    run_logs_dir, run_figures_dir, run_xai_dir = _experiment_result_dirs(experiment_name)
    info = get_data_info()
    split_dfs = make_split_dataframes(seed=split_seed)
    split_summary_path = run_logs_dir / "split_summary.csv"
    _save_split_summary(split_dfs, split_summary_path)
    _copy_if_different(split_summary_path, OUTPUT_DIR / f"{experiment_name}_split_summary.csv")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    wandb_run = init_wandb_run(experiment_name, model_name, input_mode, epochs, batch_size) if wandb_enabled else None
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "augment_train": augment_train,
                "class_weighted_loss": class_weighted_loss,
                "class_weight_power": class_weight_power,
                "class_balanced_sampler": class_balanced_sampler,
                "early_stopping_patience": early_stopping_patience,
                "backbone": "torchvision ResNet50 IMAGENET1K_V2" if model_name == "resnet50" else model_name,
            },
            allow_val_change=True,
        )
    model = build_ablation_model(model_name, input_mode, info).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weight = _class_weights(split_dfs["train"], info, device, power=class_weight_power) if class_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp))

    sampler = None
    train_shuffle = True
    if class_balanced_sampler:
        sampler = WeightedRandomSampler(
            weights=_sample_weights(split_dfs["train"], info),
            num_samples=len(split_dfs["train"]),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        FusionTensorDataset(
            split_dfs["train"],
            info,
            input_mode,
            image_transform=get_train_image_augmentation(augment_train and input_mode != "metadata_only"),
        ),
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        FusionTensorDataset(split_dfs["val"], info, input_mode),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        FusionTensorDataset(split_dfs["test"], info, input_mode),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    metrics_path = run_logs_dir / "epoch_metrics.csv"
    checkpoint_path = OUTPUT_DIR / f"{experiment_name}_best.pth"
    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0
    best_val_auc = -1.0
    epochs_without_improvement = 0
    started = time.time()

    print("Split sizes:", {name: len(df) for name, df in split_dfs.items()})
    print("Class mapping:", info.class_to_idx)

    for epoch in range(1, epochs + 1):
        train = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            input_mode,
            amp,
            optimizer,
            scaler,
            desc=f"{experiment_name} train epoch {epoch}/{epochs}",
            num_classes=info.num_classes,
        )
        val = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            input_mode,
            amp,
            optimizer=None,
            scaler=None,
            desc=f"{experiment_name} val epoch {epoch}/{epochs}",
            num_classes=info.num_classes,
        )
        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_acc": train["acc"],
            "train_bacc": train["bacc"],
            "train_auc": train["auc"],
            "train_f1_macro": train["f1_macro"],
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_bacc": val["bacc"],
            "val_auc": val["auc"],
            "val_f1_macro": val["f1_macro"],
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% "
            f"train_bacc={row['train_bacc']:.2f}% train_auc={row['train_auc']:.4f} train_f1={row['train_f1_macro']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}% "
            f"val_bacc={row['val_bacc']:.2f}% val_auc={row['val_auc']:.4f} val_f1={row['val_f1_macro']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(row | {"epoch": epoch})

        val_f1 = row["val_f1_macro"]
        val_auc = row["val_auc"] if not np.isnan(row["val_auc"]) else -1.0
        improved = False
        if (val_f1 > best_val_f1) or (val_f1 == best_val_f1 and val_auc >= best_val_auc):
            improved = True
            best_val_f1 = val_f1
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": info.class_to_idx,
                    "epoch": epoch,
                    "val_acc": row["val_acc"],
                    "val_bacc": row["val_bacc"],
                    "val_auc": row["val_auc"],
                    "val_f1_macro": row["val_f1_macro"],
                    "input_mode": input_mode,
                    "model_name": model_name,
                    "experiment_name": experiment_name,
                    "split": "stratified 80 train / 10 validation / 10 test",
                },
                checkpoint_path,
            )
        if not improved:
            epochs_without_improvement += 1

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        history[-1]["lr"] = current_lr
        if wandb_run is not None:
            wandb_run.log({"lr": current_lr, "epochs_without_improvement": epochs_without_improvement, "epoch": epoch})

        if epoch >= min_epochs and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no validation macro-F1 improvement for {epochs_without_improvement} epochs.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(metrics_path, index=False)
    epoch_log_path = run_logs_dir / "epoch_log.txt"
    _save_epoch_log(history_df, epoch_log_path)
    _copy_if_different(metrics_path, OUTPUT_DIR / f"{experiment_name}_metrics.csv")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = _run_epoch(
        model,
        test_loader,
        criterion,
        device,
        input_mode,
        amp,
        optimizer=None,
        scaler=None,
        desc=f"{experiment_name} test",
        num_classes=info.num_classes,
    )
    y_true = test["y_true"]
    y_pred = test["y_pred"]
    y_prob = test["y_prob"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(info.num_classes)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(info.num_classes)),
        target_names=info.class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).T

    report_path = run_logs_dir / "classification_report.csv"
    cm_path = run_logs_dir / "confusion_matrix.csv"
    prob_path = run_logs_dir / "test_predictions.csv"
    report_df.to_csv(report_path)
    pd.DataFrame(cm, index=info.class_names, columns=info.class_names).to_csv(cm_path)
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [info.class_names[i] for i in y_true],
            "pred_label": [info.class_names[i] for i in y_pred],
            **{f"prob_{name}": y_prob[:, idx] for idx, name in enumerate(info.class_names)},
        }
    ).to_csv(prob_path, index=False)
    _copy_if_different(report_path, OUTPUT_DIR / f"{experiment_name}_classification_report.csv")
    _copy_if_different(cm_path, OUTPUT_DIR / f"{experiment_name}_confusion_matrix.csv")
    _copy_if_different(prob_path, OUTPUT_DIR / f"{experiment_name}_test_predictions.csv")

    loss_plot_path = run_figures_dir / "loss_curve.png"
    metric_plot_path = run_figures_dir / "epoch_metrics.png"
    cm_plot_path = run_figures_dir / "confusion_matrix.png"
    roc_plot_path = run_figures_dir / "roc_curve.png"
    _plot_loss_curve(history_df, loss_plot_path)
    _plot_metric_curve(history_df, metric_plot_path)
    _plot_confusion_matrix(cm, info.class_names, cm_plot_path)
    _plot_roc_curves(y_true, y_prob, info.class_names, roc_plot_path)
    _copy_if_different(loss_plot_path, OUTPUT_DIR / "plots" / f"{experiment_name}_loss_curve.png")
    _copy_if_different(metric_plot_path, OUTPUT_DIR / "plots" / f"{experiment_name}_epoch_metrics.png")
    _copy_if_different(cm_plot_path, OUTPUT_DIR / "plots" / f"{experiment_name}_confusion_matrix.png")
    _copy_if_different(roc_plot_path, OUTPUT_DIR / "plots" / f"{experiment_name}_roc_curve.png")

    xai_paths: dict[str, str] = {}
    if xai_enabled:
        first_batch = next(iter(test_loader))
        first_batch = move_batch(first_batch, device, input_mode)
        xai_paths = save_xai_artifacts(model, first_batch, input_mode, info, experiment_name, run_xai_dir)
        for path in xai_paths.values():
            _copy_if_different(Path(path), OUTPUT_DIR / "xai" / Path(path).name)

    artifact_paths = {
        "plots/loss_curve": loss_plot_path,
        "plots/epoch_metrics": metric_plot_path,
        "plots/confusion_matrix": cm_plot_path,
        "plots/roc_curve": roc_plot_path,
        "logs/epoch_log": epoch_log_path,
        "logs/epoch_metrics": metrics_path,
        "metrics/classification_report": report_path,
        "metrics/confusion_matrix_csv": cm_path,
        "metrics/test_predictions": prob_path,
    }
    _log_artifacts_to_wandb(wandb_run, artifact_paths)

    final = history[-1]
    result = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "input_mode": input_mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "split": "80 train / 10 validation / 10 test",
        "augment_train": augment_train,
        "class_weighted_loss": class_weighted_loss,
        "class_weight_power": class_weight_power,
        "class_balanced_sampler": class_balanced_sampler,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_bacc": float(checkpoint["val_bacc"]),
        "best_val_auc": float(checkpoint["val_auc"]),
        "best_val_f1_macro": float(checkpoint["val_f1_macro"]),
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_train_bacc": final["train_bacc"],
        "final_train_auc": final["train_auc"],
        "final_train_f1_macro": final["train_f1_macro"],
        "final_val_loss": final["val_loss"],
        "final_val_acc": final["val_acc"],
        "final_val_bacc": final["val_bacc"],
        "final_val_auc": final["val_auc"],
        "final_val_f1_macro": final["val_f1_macro"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_bacc": test["bacc"],
        "test_auc": test["auc"],
        "test_f1_macro": test["f1_macro"],
        "elapsed_sec": final["elapsed_sec"],
        "metrics_path": str(metrics_path),
        "epoch_log_path": str(epoch_log_path),
        "split_summary_path": str(split_summary_path),
        "checkpoint_path": str(checkpoint_path),
        "classification_report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "test_predictions_path": str(prob_path),
        "loss_curve_path": str(loss_plot_path),
        "epoch_metrics_curve_path": str(metric_plot_path),
        "confusion_matrix_plot_path": str(cm_plot_path),
        "roc_curve_path": str(roc_plot_path),
        "xai_paths": repr(xai_paths),
    }
    result_path = run_logs_dir / "result_summary.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)
    _copy_if_different(result_path, OUTPUT_DIR / f"{experiment_name}_result.csv")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_epoch": result["best_epoch"],
                "best_val_acc": result["best_val_acc"],
                "best_val_bacc": result["best_val_bacc"],
                "best_val_auc": result["best_val_auc"],
                "best_val_f1_macro": result["best_val_f1_macro"],
                "test_acc": result["test_acc"],
                "test_bacc": result["test_bacc"],
                "test_auc": result["test_auc"],
                "test_f1_macro": result["test_f1_macro"],
            }
        )
        wandb_run.finish()

    print("Test performance:")
    print(f"test_loss={test['loss']:.4f} test_acc={test['acc']:.2f}% test_bacc={test['bacc']:.2f}% test_auc={test['auc']:.4f} test_f1={test['f1_macro']:.4f}")
    print("Classification report:")
    print(report_df.to_string())
    print(f"{experiment_name} complete: best_epoch={result['best_epoch']}, test_acc={result['test_acc']:.2f}%")
    return result


def parse_ablation_name(name: str) -> tuple[str, str, str]:
    matches = [item for item in ABLATIONS if item[2] == name]
    if not matches:
        raise ValueError(f"Unknown experiment_name {name!r}. Known: {[x[2] for x in ABLATIONS]}")
    return matches[0]
