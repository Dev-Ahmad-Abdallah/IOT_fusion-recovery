from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from PIL import Image
from sklearn.feature_selection import mutual_info_classif
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

REPO_ROOT = Path(os.getenv("IOT_FUSION_REPO", "/home/zeus/content/IOT_fusion"))
DATA_ROOT = Path(os.getenv("IOT_FUSION_DATA", "/home/zeus/content/dataset"))
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from v2_models import ModelSpec, TwoExpertMoE, build_model, forward_model

CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

RESULT_LOGS = REPO_ROOT / "Results_Logs"
RESULT_FIGURES = REPO_ROOT / "Results_Figures"
RESULT_XAI = REPO_ROOT / "Results_XAI"
ONNX_BESTS = REPO_ROOT / "ONNX_Bests"
OUTPUTS = REPO_ROOT / "outputs" / "ablations_v2"

IMAGE_MODELS = ["cnn", "transformer", "unet", "efficientnet", "conformer", "cvt"]
TABULAR_MODELS = ["mlp", "tabtransformer"]
FUSION_MODES = ["early_fusion", "intermediate_fusion", "late_fusion"]
TRACKS = [
    ("processed", "default_balanced", "full"),
    ("processed", "no_imbalance", "reduced"),
    ("processed", "feature_selection", "reduced"),
    ("unprocessed", "default_balanced", "reduced"),
    ("unprocessed", "no_imbalance", "reduced"),
    ("unprocessed", "feature_selection", "reduced"),
]
REDUCED_EXPERIMENTS = {"cnn_img_only", "tabtransformer_tab_only", "cnn_tabtransformer_intermediate_fusion"}


@dataclass(frozen=True)
class Experiment:
    dataset_source: str
    variant: str
    experiment_name: str
    image_model: str | None
    tabular_model: str | None
    mode: str
    scope: str

    @property
    def uses_image(self) -> bool:
        return self.mode != "tabular_only"

    @property
    def uses_metadata(self) -> bool:
        return self.mode != "image_only"

    @property
    def is_moe(self) -> bool:
        return self.mode == "moe"

    @property
    def is_fusion(self) -> bool:
        return self.mode in FUSION_MODES


@dataclass
class DataBundle:
    splits: dict[str, pd.DataFrame]
    feature_cols: list[str]
    metadata_mean: pd.Series
    metadata_std: pd.Series
    image_lookup: dict[str, Path]
    source: str
    variant: str


class AblationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bundle: DataBundle, mode: str, train: bool):
        self.df = df.reset_index(drop=True)
        self.bundle = bundle
        self.mode = mode
        self.train = train
        self.pil_train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.82, 1.0), ratio=(0.9, 1.1), antialias=True),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.08), value=0.0),
            ]
        )
        self.pil_eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tensor_train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(25, fill=0.0),
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.08), value=0.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        if self.bundle.source == "processed":
            tensor_dir = DATA_ROOT / "processed_images" / f"{row['tensor_source_split']}_tensors"
            image = torch.load(tensor_dir / (Path(str(row["img_id"])).stem + "_prep.pt"), map_location="cpu").float()
            if self.train:
                image = self.tensor_train_transform(image)
            return image
        path = self.bundle.image_lookup[Path(str(row["img_id"])).stem]
        image = Image.open(path).convert("RGB")
        return (self.pil_train_transform if self.train else self.pil_eval_transform)(image)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        batch = {"label": torch.tensor(CLASS_TO_IDX[str(row["label_raw"])], dtype=torch.long)}
        if self.mode != "image_only":
            batch["metadata"] = torch.tensor(row[self.bundle.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        if self.mode != "tabular_only":
            batch["image"] = self._load_image(row)
        return batch


def load_wandb_env() -> None:
    env_path = REPO_ROOT / "W&B.env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.strip().startswith("#"):
            continue
        key, value = raw.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def result_dirs(exp: Experiment) -> tuple[Path, Path, Path, Path, Path]:
    parts = (exp.dataset_source, exp.variant, exp.experiment_name)
    logs = RESULT_LOGS.joinpath(*parts)
    figs = RESULT_FIGURES.joinpath(*parts)
    xai = RESULT_XAI.joinpath(*parts)
    onnx = ONNX_BESTS.joinpath(*parts)
    ckpt = OUTPUTS.joinpath(*parts)
    for path in (logs, figs, xai, onnx, ckpt):
        path.mkdir(parents=True, exist_ok=True)
    return logs, figs, xai, onnx, ckpt


def clean_outputs() -> None:
    for path in [RESULT_LOGS, RESULT_FIGURES, RESULT_XAI, ONNX_BESTS, REPO_ROOT / "outputs", REPO_ROOT / "wandb"]:
        if path.exists():
            shutil.rmtree(path)
    for path in [RESULT_LOGS, RESULT_FIGURES, RESULT_XAI, ONNX_BESTS, OUTPUTS]:
        path.mkdir(parents=True, exist_ok=True)


def build_manifest() -> list[Experiment]:
    base: list[ModelSpec] = []
    for image_model in IMAGE_MODELS:
        base.append(ModelSpec(f"{image_model}_img_only", image_model, None, "image_only"))
    for tabular_model in TABULAR_MODELS:
        base.append(ModelSpec(f"{tabular_model}_tab_only", None, tabular_model, "tabular_only"))
    for image_model in IMAGE_MODELS:
        for tabular_model in TABULAR_MODELS:
            for fusion_mode in FUSION_MODES:
                base.append(ModelSpec(f"{image_model}_{tabular_model}_{fusion_mode}", image_model, tabular_model, fusion_mode))

    manifest: list[Experiment] = []
    for source, variant, scope in TRACKS:
        for spec in base:
            if scope == "reduced" and spec.experiment_name not in REDUCED_EXPERIMENTS:
                continue
            manifest.append(Experiment(source, variant, spec.experiment_name, spec.image_model, spec.tabular_model, spec.mode, scope))
        manifest.append(Experiment(source, variant, "moe_best", None, None, "moe", scope))
    return manifest


def _processed_df() -> pd.DataFrame:
    train_df = pd.read_csv(DATA_ROOT / "processed" / "train_tabular_processed.csv")
    train_df["tensor_source_split"] = "train"
    test_df = pd.read_csv(DATA_ROOT / "processed" / "test_tabular_processed.csv")
    test_df["tensor_source_split"] = "test"
    return pd.concat([train_df, test_df], ignore_index=True)


def _raw_df() -> tuple[pd.DataFrame, dict[str, Path]]:
    df = pd.read_csv(REPO_ROOT / "Un-Processed_Data_Set" / "metadata.csv")
    df = df.rename(columns={"diagnostic": "label_raw"})
    image_lookup = {p.stem: p for p in (REPO_ROOT / "Un-Processed_Data_Set").glob("imgs_part_*/*.png")}
    missing = [img for img in df["img_id"].astype(str).map(lambda x: Path(x).stem) if img not in image_lookup]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} unprocessed images; first: {missing[:3]}")
    return df, image_lookup


def _encode_raw_metadata(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[["img_id", "patient_id", "label_raw"]].copy()
    meta = df.drop(columns=["img_id", "patient_id", "lesion_id", "label_raw"], errors="ignore")
    numeric_cols = meta.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in meta.columns if c not in numeric_cols]
    numeric = meta[numeric_cols].copy()
    for col in numeric_cols:
        numeric[col] = numeric[col].fillna(numeric[col].median())
    categorical = pd.get_dummies(meta[cat_cols].fillna("nan").astype(str), prefix=cat_cols, prefix_sep="__", dummy_na=False)
    return pd.concat([keep, numeric.add_prefix("raw_num__"), categorical.add_prefix("raw_cat__")], axis=1)


def _feature_select(df: pd.DataFrame, feature_cols: list[str], k: int = 32) -> list[str]:
    y = df["label_raw"].map(CLASS_TO_IDX).to_numpy()
    x = df[feature_cols].to_numpy(dtype=np.float32)
    k = min(k, len(feature_cols))
    scores = mutual_info_classif(x, y, discrete_features=False, random_state=42)
    order = np.argsort(-scores)[:k]
    return [feature_cols[i] for i in order]


def make_data_bundle(source: str, variant: str, seed: int = 42) -> DataBundle:
    image_lookup: dict[str, Path] = {}
    if source == "processed":
        df = _processed_df()
        feature_cols = [c for c in df.columns if c not in {"img_id", "patient_id", "label_raw", "tensor_source_split"}]
    elif source == "unprocessed":
        raw, image_lookup = _raw_df()
        df = _encode_raw_metadata(raw)
        feature_cols = [c for c in df.columns if c not in {"img_id", "patient_id", "label_raw"}]
    else:
        raise ValueError(f"Unknown source: {source}")

    if variant == "feature_selection":
        feature_cols = _feature_select(df, feature_cols, k=32)

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=df["label_raw"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label_raw"])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1).fillna(1)
    for split in (train_df, val_df, test_df):
        split.loc[:, feature_cols] = (split[feature_cols] - mean) / std
        split.loc[:, feature_cols] = split[feature_cols].fillna(0.0)
    return DataBundle({"train": train_df, "val": val_df, "test": test_df}, feature_cols, mean, std, image_lookup, source, variant)


def make_loaders(exp: Experiment, bundle: DataBundle, batch_size: int, num_workers: int = 2) -> dict[str, DataLoader]:
    dataset_mode = "tabular_only" if exp.mode == "tabular_only" else ("image_only" if exp.mode == "image_only" else "fusion")
    return {
        split: DataLoader(
            AblationDataset(df, bundle, dataset_mode, train=(split == "train")),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        for split, df in bundle.splits.items()
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    y_prob = np.divide(
        y_prob,
        row_sums,
        out=np.full_like(y_prob, 1.0 / y_prob.shape[1], dtype=np.float64),
        where=row_sums > 0,
    )
    try:
        auc_value = float(roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        auc_value = float("nan")
    return {
        "acc": float(accuracy_score(y_true, y_pred) * 100.0),
        "bacc": float(balanced_accuracy_score(y_true, y_pred) * 100.0),
        "auc": auc_value,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    mode: str,
    amp: bool,
    desc: str,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_seen = 0
    all_true, all_pred, all_prob = [], [], []
    iterator = tqdm(loader, desc=desc, leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in iterator:
            batch = move_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda" and amp)):
                logits = forward_model(model, batch, mode)
                loss = criterion(logits, batch["label"])
            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            probs = torch.softmax(logits.detach().float(), dim=1)
            preds = probs.argmax(dim=1)
            labels = batch["label"].detach()
            bs = labels.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            total_seen += bs
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
            iterator.set_postfix(loss=f"{total_loss / max(total_seen, 1):.4f}")
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob).astype(np.float64)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    y_prob = np.divide(
        y_prob,
        row_sums,
        out=np.full_like(y_prob, 1.0 / y_prob.shape[1], dtype=np.float64),
        where=row_sums > 0,
    )
    result = metrics_from_arrays(y_true, y_pred, y_prob)
    result.update({"loss": total_loss / max(total_seen, 1), "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob})
    return result


def class_weights(df: pd.DataFrame, enabled: bool, device: torch.device) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = df["label_raw"].value_counts().reindex(CLASS_NAMES).fillna(0).to_numpy(dtype=np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def default_epochs(exp: Experiment) -> int:
    if exp.is_moe:
        return 18
    if exp.mode == "tabular_only":
        return 35
    return 25


def default_batch_size(exp: Experiment) -> int:
    if exp.mode == "tabular_only":
        return 256
    if exp.image_model == "efficientnet":
        return 48
    return 64


def checkpoint_path(exp: Experiment) -> Path:
    return OUTPUTS / exp.dataset_source / exp.variant / exp.experiment_name / "best.pth"


def load_best_model(exp: Experiment, bundle: DataBundle, device: torch.device) -> nn.Module:
    model = build_model(ModelSpec(exp.experiment_name, exp.image_model, exp.tabular_model, exp.mode), len(bundle.feature_cols), len(CLASS_NAMES)).to(device)
    checkpoint = torch.load(checkpoint_path(exp), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_moe_experts(source: str, variant: str) -> tuple[Experiment, Experiment]:
    manifest = [e for e in build_manifest() if e.dataset_source == source and e.variant == variant and not e.is_moe]
    rows = []
    for exp in manifest:
        path = RESULT_LOGS / source / variant / exp.experiment_name / "result_summary.csv"
        if path.exists():
            rows.append((exp, pd.read_csv(path).iloc[0].to_dict()))
    image_candidates = [(e, r) for e, r in rows if e.mode == "image_only"]
    tab_candidates = [(e, r) for e, r in rows if e.mode == "tabular_only"]
    if not image_candidates or not tab_candidates:
        raise RuntimeError(f"Cannot select MoE experts for {source}/{variant}; missing image or tabular runs.")
    return (
        max(image_candidates, key=lambda item: item[1]["best_val_f1_macro"])[0],
        max(tab_candidates, key=lambda item: item[1]["best_val_f1_macro"])[0],
    )


def instantiate_model(exp: Experiment, bundle: DataBundle, device: torch.device) -> nn.Module:
    if exp.is_moe:
        image_exp, tab_exp = select_moe_experts(exp.dataset_source, exp.variant)
        return TwoExpertMoE(load_best_model(image_exp, bundle, device), load_best_model(tab_exp, bundle, device), len(bundle.feature_cols)).to(device)
    return build_model(ModelSpec(exp.experiment_name, exp.image_model, exp.tabular_model, exp.mode), len(bundle.feature_cols), len(CLASS_NAMES)).to(device)


def save_epoch_log(history: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in history.to_dict(orient="records"):
        lines.append(
            f"Epoch {int(row['epoch']):03d} | "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% train_bacc={row['train_bacc']:.2f}% "
            f"train_auc={row['train_auc']:.4f} train_f1={row['train_f1_macro']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}% val_bacc={row['val_bacc']:.2f}% "
            f"val_auc={row['val_auc']:.4f} val_f1={row['val_f1_macro']:.4f} lr={row['lr']:.8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(history: pd.DataFrame, figs: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_values = history[["train_loss", "val_loss"]].to_numpy(dtype=float)
    finite_losses = loss_values[np.isfinite(loss_values)]
    if finite_losses.size:
        loss_min = float(finite_losses.min())
        loss_max = float(finite_losses.max())
        tick_step = 0.1 if loss_max <= 1.0 else 0.2
        lower = max(0.0, np.floor((loss_min - tick_step) / tick_step) * tick_step)
        upper = np.ceil((loss_max + tick_step) / tick_step) * tick_step
        if upper <= lower:
            upper = lower + tick_step
        plt.ylim(lower, upper)
        plt.gca().yaxis.set_major_locator(MultipleLocator(tick_step))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for key in ["train_acc", "val_acc", "train_bacc", "val_bacc"]:
        plt.plot(history["epoch"], history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Percent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / "epoch_metrics.png", dpi=150)
    plt.close()


def plot_test_figures(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, figs: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figs / "confusion_matrix.png", dpi=150)
    plt.close()

    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(CLASS_NAMES):
        if y_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        plt.plot(fpr, tpr, label=f"{name} AUC={sklearn_auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figs / "roc_curve.png", dpi=150)
    plt.close()


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "image_only":
            return self.model(inputs[0])
        if self.mode == "tabular_only":
            return self.model(inputs[0])
        return self.model(inputs[0], inputs[1])


def export_onnx(model: nn.Module, exp: Experiment, bundle: DataBundle, onnx_dir: Path, device: torch.device) -> None:
    model.eval()
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fastpath_state = None
    try:
        fastpath_state = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        fastpath_state = None
    mode = exp.mode if exp.mode != "moe" else "late_fusion"
    wrapper = ExportWrapper(model, mode).to(device)
    image = torch.randn(1, 3, 224, 224, device=device)
    metadata = torch.randn(1, len(bundle.feature_cols), device=device)
    if exp.mode == "image_only":
        args, input_names = (image,), ["image"]
    elif exp.mode == "tabular_only":
        args, input_names = (metadata,), ["metadata"]
    else:
        args, input_names = (image, metadata), ["image", "metadata"]
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}
    try:
        torch.onnx.export(
            wrapper,
            args,
            onnx_dir / "best_model.onnx",
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
    finally:
        if fastpath_state is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath_state)
    (onnx_dir / "onnx_export_manifest.json").write_text(
        json.dumps({"experiment": asdict(exp), "input_names": input_names, "output": "logits"}, indent=2),
        encoding="utf-8",
    )


def save_xai(model: nn.Module, exp: Experiment, bundle: DataBundle, loaders: dict[str, DataLoader], xai_dir: Path, device: torch.device) -> None:
    model.eval()
    mode = exp.mode if exp.mode != "moe" else "late_fusion"
    batch = move_batch(next(iter(loaders["test"])), device)
    batch = {k: v[: min(4, len(v))].detach().clone() for k, v in batch.items()}
    rows = []
    if exp.uses_image and "image" in batch:
        image = batch["image"].clone().requires_grad_(True)
        local = dict(batch)
        local["image"] = image
        logits = forward_model(model, local, mode)
        target = logits.argmax(dim=1)
        score = logits.gather(1, target.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=exp.uses_metadata)
        sal = image.grad.detach().abs().amax(dim=1).cpu().numpy()
        np.save(xai_dir / "image_gradcam_or_saliency.npy", sal)
        plt.figure(figsize=(8, 3))
        for i in range(min(4, sal.shape[0])):
            plt.subplot(1, 4, i + 1)
            plt.imshow(sal[i], cmap="magma")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(xai_dir / "image_gradcam_or_saliency.png", dpi=150)
        plt.close()
        rows.append({"artifact": "image_gradcam_or_saliency", "method": "gradient saliency fallback"})

    if exp.uses_metadata and "metadata" in batch:
        metadata = batch["metadata"].clone().requires_grad_(True)
        local = dict(batch)
        local["metadata"] = metadata
        logits = forward_model(model, local, mode)
        target = logits.argmax(dim=1)
        score = logits.gather(1, target.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        importance = metadata.grad.detach().abs().mean(dim=0).cpu().numpy()
        df = pd.DataFrame({"feature": bundle.feature_cols, "gradient_importance": importance}).sort_values(
            "gradient_importance", ascending=False
        )
        df.to_csv(xai_dir / "metadata_attribution.csv", index=False)
        top = df.head(20).sort_values("gradient_importance")
        plt.figure(figsize=(9, 5))
        plt.barh(top["feature"], top["gradient_importance"])
        plt.tight_layout()
        plt.savefig(xai_dir / "metadata_attribution.png", dpi=150)
        plt.close()
        rows.append({"artifact": "metadata_attribution", "method": "mean absolute input gradient"})

    if exp.is_fusion or exp.is_moe:
        with torch.no_grad():
            original = torch.softmax(forward_model(model, batch, mode), dim=1)
            pred = original.argmax(dim=1)
            img_mask = dict(batch)
            img_mask["image"] = torch.zeros_like(img_mask["image"])
            meta_mask = dict(batch)
            meta_mask["metadata"] = torch.zeros_like(meta_mask["metadata"])
            img_prob = torch.softmax(forward_model(model, img_mask, mode), dim=1)
            meta_prob = torch.softmax(forward_model(model, meta_mask, mode), dim=1)
            contrib = []
            for i in range(len(pred)):
                cls = pred[i].item()
                contrib.append(
                    {
                        "sample": i,
                        "pred_label": CLASS_NAMES[cls],
                        "original_prob": float(original[i, cls].item()),
                        "image_delta": float(original[i, cls].item() - img_prob[i, cls].item()),
                        "metadata_delta": float(original[i, cls].item() - meta_prob[i, cls].item()),
                    }
                )
        contrib_df = pd.DataFrame(contrib)
        contrib_df.to_csv(xai_dir / "modality_contribution.csv", index=False)
        plt.figure(figsize=(6, 4))
        plt.bar(["image_delta", "metadata_delta"], [contrib_df["image_delta"].mean(), contrib_df["metadata_delta"].mean()])
        plt.tight_layout()
        plt.savefig(xai_dir / "modality_contribution.png", dpi=150)
        plt.close()
        rows.append({"artifact": "modality_contribution", "method": "baseline removal probability delta"})
    pd.DataFrame(rows).to_csv(xai_dir / "xai_sample_index.csv", index=False)


def write_split_summary(bundle: DataBundle, path: Path) -> None:
    rows = []
    for split, df in bundle.splits.items():
        row = {"split": split, "rows": len(df)}
        row.update({f"class_{k}": int(v) for k, v in df["label_raw"].value_counts().to_dict().items()})
        rows.append(row)
    pd.DataFrame(rows).fillna(0).to_csv(path, index=False)


def train_one(exp: Experiment, force: bool = False) -> dict[str, Any]:
    logs, figs, xai_dir, onnx_dir, ckpt_dir = result_dirs(exp)
    if (logs / "result_summary.csv").exists() and not force:
        return pd.read_csv(logs / "result_summary.csv").iloc[0].to_dict()

    load_wandb_env()
    wandb_run = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    bundle = make_data_bundle(exp.dataset_source, exp.variant)
    batch_size = default_batch_size(exp)
    loaders = make_loaders(exp, bundle, batch_size=batch_size)
    model = instantiate_model(exp, bundle, device)
    weight = class_weights(bundle.splits["train"], exp.variant != "no_imbalance", device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05 if exp.variant != "no_imbalance" else 0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4 if exp.mode != "tabular_only" else 5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1, best_auc = -1.0, -1.0
    patience, min_epochs = 8, 8
    epochs_without_improvement = 0
    history = []
    started = time.time()
    write_split_summary(bundle, logs / "split_summary.csv")
    mode = exp.mode if exp.mode != "moe" else "late_fusion"
    try:
        import wandb

        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "iot fusion"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}",
            group=f"{exp.dataset_source}/{exp.variant}",
            job_type="v2_ablation",
            config={**asdict(exp), "batch_size": batch_size, "feature_count": len(bundle.feature_cols)},
            reinit=True,
        )
    except Exception as exc:
        print(f"W&B logging disabled for {exp.experiment_name}: {exc}", flush=True)
        wandb_run = None

    for epoch in range(1, default_epochs(exp) + 1):
        train = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, mode, True, f"{exp.experiment_name} train {epoch}")
        val = run_epoch(model, loaders["val"], criterion, None, None, device, mode, True, f"{exp.experiment_name} val {epoch}")
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
            "elapsed_sec": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name} epoch {epoch:03d}: "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}% "
            f"val_auc={row['val_auc']:.4f} val_f1={row['val_f1_macro']:.4f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}, step=epoch)
        val_auc = row["val_auc"] if not np.isnan(row["val_auc"]) else -1.0
        improved = row["val_f1_macro"] > best_f1 or (row["val_f1_macro"] == best_f1 and val_auc >= best_auc)
        if improved:
            best_f1 = row["val_f1_macro"]
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "experiment": asdict(exp), "feature_cols": bundle.feature_cols, "epoch": epoch, **row},
                ckpt_dir / "best.pth",
            )
        else:
            epochs_without_improvement += 1
        scheduler.step(row["val_f1_macro"])
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(logs / "epoch_metrics.csv", index=False)
    save_epoch_log(history_df, logs / "epoch_log.txt")
    plot_curves(history_df, figs)

    checkpoint = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test = run_epoch(model, loaders["test"], criterion, None, None, device, mode, True, f"{exp.experiment_name} test")
    y_true, y_pred, y_prob = test["y_true"], test["y_pred"], test["y_prob"]
    pd.DataFrame(
        classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    ).T.to_csv(logs / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(logs / "confusion_matrix.csv")
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "pred_label": [CLASS_NAMES[i] for i in y_pred],
            **{f"prob_{name}": y_prob[:, idx] for idx, name in enumerate(CLASS_NAMES)},
        }
    ).to_csv(logs / "test_predictions.csv", index=False)
    plot_test_figures(y_true, y_pred, y_prob, figs)
    save_xai(model, exp, bundle, loaders, xai_dir, device)
    export_onnx(model, exp, bundle, onnx_dir, device)

    final = history[-1]
    result = {
        **asdict(exp),
        "epochs": len(history),
        "batch_size": batch_size,
        "feature_count": len(bundle.feature_cols),
        "imbalance_handling": exp.variant != "no_imbalance",
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_bacc": float(checkpoint["val_bacc"]),
        "best_val_auc": float(checkpoint["val_auc"]),
        "best_val_f1_macro": float(checkpoint["val_f1_macro"]),
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_val_loss": final["val_loss"],
        "final_val_acc": final["val_acc"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "test_bacc": test["bacc"],
        "test_auc": test["auc"],
        "test_f1_macro": test["f1_macro"],
        "checkpoint_path": str(ckpt_dir / "best.pth"),
        "onnx_path": str(onnx_dir / "best_model.onnx"),
    }
    pd.DataFrame([result]).to_csv(logs / "result_summary.csv", index=False)
    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test.items() if isinstance(v, (int, float, np.integer, np.floating))})
        wandb_run.summary.update(result)
        wandb_run.finish()
    return result


def collect_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULT_LOGS.glob("*/*/*/result_summary.csv")):
        try:
            rows.append(pd.read_csv(path).iloc[0].to_dict())
        except Exception:
            continue
    summary = pd.DataFrame(rows)
    RESULT_LOGS.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        sort_cols = [c for c in ["dataset_source", "variant", "experiment_name"] if c in summary.columns]
        summary = summary.sort_values(sort_cols).reset_index(drop=True)
    summary.to_csv(RESULT_LOGS / "ablation_summary.csv", index=False)
    return summary


def verify_outputs(manifest: list[Experiment]) -> dict[str, Any]:
    missing: dict[str, list[str]] = {"summary": [], "checkpoint": [], "onnx": [], "figures": [], "xai": []}
    for exp in manifest:
        base = f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}"
        if not (RESULT_LOGS / base / "result_summary.csv").exists():
            missing["summary"].append(base)
        if not checkpoint_path(exp).exists():
            missing["checkpoint"].append(base)
        if not (ONNX_BESTS / base / "best_model.onnx").exists():
            missing["onnx"].append(base)
        fig_dir = RESULT_FIGURES / base
        if not all((fig_dir / name).exists() for name in ["loss_curve.png", "epoch_metrics.png", "confusion_matrix.png", "roc_curve.png"]):
            missing["figures"].append(base)
        if not (RESULT_XAI / base / "xai_sample_index.csv").exists():
            missing["xai"].append(base)
    report = {
        "expected_runs": len(manifest),
        "completed_summaries": len(manifest) - len(missing["summary"]),
        "completed_onnx": len(manifest) - len(missing["onnx"]),
        "missing": missing,
    }
    (RESULT_LOGS / "verification_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _comparison_notebook(title: str, code: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nExecuted comparison notebook for the v2 ablation grid."),
        nbf.v4.new_code_cell(code),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def generate_comparison_notebooks(execute: bool = True) -> None:
    summary_path = RESULT_LOGS / "ablation_summary.csv"
    common = r"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "Results_Logs" / "ablation_summary.csv").exists():
    ROOT = ROOT.parent
summary = pd.read_csv(ROOT / "Results_Logs" / "ablation_summary.csv")
metric_cols = ["test_acc", "test_bacc", "test_auc", "test_f1_macro"]
metric_labels = {
    "test_acc": "Accuracy (%)",
    "test_bacc": "Balanced accuracy (%)",
    "test_auc": "AUC (%)",
    "test_f1_macro": "Macro-F1 (%)",
    "combined_score": "Combined score (%)",
}
summary["combined_score"] = (
    0.35 * summary["test_f1_macro"].astype(float)
    + 0.25 * (summary["test_bacc"].astype(float) / 100.0)
    + 0.25 * summary["test_auc"].astype(float)
    + 0.10 * (summary["test_acc"].astype(float) / 100.0)
    + 0.05
    * (
        1.0
        / (
            1.0
            + summary["final_val_loss"].astype(float)
            + (summary["final_val_loss"].astype(float) - summary["final_train_loss"].astype(float)).abs()
        )
    )
)

def best_per_track(df):
    # Return one winner per dataset source and variant for compact comparisons.
    return (
        df.sort_values(["combined_score", "test_f1_macro", "test_bacc", "test_auc", "test_acc"], ascending=[False, False, False, False, False])
        .groupby(["dataset_source", "variant"], as_index=False)
        .head(1)
        .sort_values(["dataset_source", "variant"])
        .reset_index(drop=True)
    )

def metric_to_percent(series, metric):
    values = series.astype(float).copy()
    if metric in {"test_auc", "test_f1_macro"}:
        values = values * 100.0
    return values

def plot_metric_bars(df, title, out_name):
    data = df.sort_values("combined_score", ascending=False).copy()
    labels = data["dataset_source"] + " / " + data["variant"] + " / " + data["experiment_name"]
    fig_h = max(4.8, 0.72 * len(data) + 2.0)
    fig, axes = plt.subplots(3, 2, figsize=(13, max(fig_h, 7.0)))
    for ax, metric in zip(axes.ravel(), ["combined_score", *metric_cols]):
        values = metric_to_percent(data[metric], metric)
        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_title(metric_labels[metric])
        ax.set_xlabel("Score; higher is better")
        ax.set_xlim(max(0, values.min() - 5), min(100, values.max() + 5))
        for y, value in enumerate(values):
            ax.text(value + 0.35, y, f"{value:.1f}", va="center", fontsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out = ROOT / "Results_Figures" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.show()
    print(f"saved {out}")
"""
    processed_code = (
        common
        + r"""
df = summary[summary["dataset_source"] == "processed"].copy()
ranked = best_per_track(df)
display(ranked[["variant", "experiment_name", "mode", "image_model", "tabular_model", "combined_score", *metric_cols, "onnx_path"]])
plot_metric_bars(ranked, "Processed dataset: best run in each variant (ranked by combined score)", "compare_processed_best_metrics.png")

display(ranked.sort_values("combined_score", ascending=False))
"""
    )
    unprocessed_code = (
        common
        + r"""
df = summary[summary["dataset_source"] == "unprocessed"].copy()
ranked = best_per_track(df)
display(ranked[["variant", "experiment_name", "mode", "image_model", "tabular_model", "combined_score", *metric_cols, "onnx_path"]])
plot_metric_bars(ranked, "Unprocessed dataset: best run in each variant (ranked by combined score)", "compare_unprocessed_best_metrics.png")

display(ranked.sort_values("combined_score", ascending=False))
"""
    )
    cross_code = (
        common
        + r"""
processed = best_per_track(summary[summary["dataset_source"] == "processed"]).copy()
unprocessed = best_per_track(summary[summary["dataset_source"] == "unprocessed"]).copy()
processed = processed[["variant", "experiment_name", "mode", "image_model", "tabular_model", "combined_score", *metric_cols]].rename(
    columns={
        "experiment_name": "processed_best",
        "mode": "processed_mode",
        "image_model": "processed_image_model",
        "tabular_model": "processed_tabular_model",
    }
)
unprocessed = unprocessed[["variant", "experiment_name", "mode", "image_model", "tabular_model", "combined_score", *metric_cols]].rename(
    columns={
        "experiment_name": "unprocessed_best",
        "mode": "unprocessed_mode",
        "image_model": "unprocessed_image_model",
        "tabular_model": "unprocessed_tabular_model",
    }
)
merged = processed.merge(unprocessed, on="variant", suffixes=("_processed", "_unprocessed"))
for metric in ["combined_score", *metric_cols]:
    merged[f"{metric}_delta"] = merged[f"{metric}_processed"] - merged[f"{metric}_unprocessed"]
display(merged.sort_values("combined_score_delta", ascending=False))

fig, ax = plt.subplots(figsize=(11, 5))
labels = merged["variant"]
delta = merged["combined_score_delta"] * 100.0
ax.barh(labels, delta)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Processed minus unprocessed combined score, percentage points")
ax.set_title("Best processed vs best unprocessed run per variant")
for y, value in enumerate(delta):
    ax.text(value + (0.15 if value >= 0 else -0.15), y, f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right")
fig.tight_layout()
out = ROOT / "Results_Figures" / "compare_best_processed_vs_unprocessed_combined_delta.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
plt.show()
print(f"saved {out}")
"""
    )
    notebooks = {
        REPO_ROOT / "training" / "compare_processed.ipynb": _comparison_notebook("Processed Dataset Comparison", processed_code),
        REPO_ROOT / "training" / "compare_unprocessed.ipynb": _comparison_notebook("Unprocessed Dataset Comparison", unprocessed_code),
        REPO_ROOT / "training" / "compare_processed_vs_unprocessed.ipynb": _comparison_notebook(
            "Processed vs Unprocessed Comparison", cross_code
        ),
    }
    for path, nb in notebooks.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, path)
    if not execute:
        return
    for path in notebooks:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--ExecutePreprocessor.timeout=900",
                    str(path),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
        except Exception as exc:
            print(f"Notebook execution failed for {path.name}: {exc}", flush=True)


def verify_pipeline(export_smoke: bool = False) -> None:
    manifest = build_manifest()
    full = [e for e in manifest if e.dataset_source == "processed" and e.variant == "default_balanced"]
    reduced = [e for e in manifest if not (e.dataset_source == "processed" and e.variant == "default_balanced")]
    assert len(manifest) == 65, f"Expected 65 runs, got {len(manifest)}"
    assert len(full) == 45, f"Expected 45 full-grid runs, got {len(full)}"
    assert len(reduced) == 20, f"Expected 20 reduced runs, got {len(reduced)}"

    processed = make_data_bundle("processed", "default_balanced")
    unprocessed = make_data_bundle("unprocessed", "default_balanced")
    feature_selected = make_data_bundle("processed", "feature_selection")
    assert [len(processed.splits[k]) for k in ["train", "val", "test"]] == [1838, 230, 230]
    assert sum(len(df) for df in unprocessed.splits.values()) == 2298
    assert len(unprocessed.image_lookup) == 2298
    assert len(feature_selected.feature_cols) < len(processed.feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = [
        Experiment("processed", "default_balanced", "cnn_img_only", "cnn", None, "image_only", "full"),
        Experiment("processed", "default_balanced", "tabtransformer_tab_only", None, "tabtransformer", "tabular_only", "full"),
        Experiment(
            "processed",
            "default_balanced",
            "cnn_tabtransformer_intermediate_fusion",
            "cnn",
            "tabtransformer",
            "intermediate_fusion",
            "full",
        ),
    ]
    for exp in samples:
        bundle = make_data_bundle(exp.dataset_source, exp.variant)
        loaders = make_loaders(exp, bundle, batch_size=2, num_workers=0)
        batch = move_batch(next(iter(loaders["train"])), device)
        model = instantiate_model(exp, bundle, device).eval()
        with torch.no_grad():
            logits = forward_model(model, batch, exp.mode)
        assert tuple(logits.shape) == (2, len(CLASS_NAMES)), f"Bad logits for {exp.experiment_name}: {tuple(logits.shape)}"

    moe_image = build_model(ModelSpec("verify_img", "cnn", None, "image_only"), len(processed.feature_cols), len(CLASS_NAMES)).to(device)
    moe_tab = build_model(ModelSpec("verify_tab", None, "tabtransformer", "tabular_only"), len(processed.feature_cols), len(CLASS_NAMES)).to(device)
    moe = TwoExpertMoE(moe_image, moe_tab, len(processed.feature_cols)).to(device).eval()
    loaders = make_loaders(
        Experiment("processed", "default_balanced", "verify_moe", None, None, "moe", "full"),
        processed,
        batch_size=2,
        num_workers=0,
    )
    batch = move_batch(next(iter(loaders["train"])), device)
    with torch.no_grad():
        logits = forward_model(moe, batch, "late_fusion")
    assert tuple(logits.shape) == (2, len(CLASS_NAMES)), f"Bad MoE logits: {tuple(logits.shape)}"

    if export_smoke:
        smoke_dir = ONNX_BESTS / "_verify"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for exp in samples:
            bundle = make_data_bundle(exp.dataset_source, exp.variant)
            model = instantiate_model(exp, bundle, device).eval()
            export_onnx(model, exp, bundle, smoke_dir / exp.experiment_name, device)
        export_onnx(
            moe,
            Experiment("processed", "default_balanced", "verify_moe", None, None, "moe", "full"),
            processed,
            smoke_dir / "verify_moe",
            device,
        )
    print(
        json.dumps(
            {
                "manifest_runs": len(manifest),
                "full_grid_runs": len(full),
                "reduced_runs": len(reduced),
                "processed_split_sizes": {k: len(v) for k, v in processed.splits.items()},
                "unprocessed_split_sizes": {k: len(v) for k, v in unprocessed.splits.items()},
                "processed_feature_count": len(processed.feature_cols),
                "feature_selection_count": len(feature_selected.feature_cols),
                "device": str(device),
                "onnx_export_smoke": export_smoke,
            },
            indent=2,
        ),
        flush=True,
    )


def _filter_start_after(manifest: list[Experiment], start_after: str | None) -> list[Experiment]:
    if start_after is None:
        return manifest
    for index, exp in enumerate(manifest):
        exp_key = f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}"
        if exp_key == start_after or exp.experiment_name == start_after:
            return manifest[index + 1 :]
    raise ValueError(f"start_after marker not found: {start_after}")


def _run_experiment_worker(exp: Experiment, force: bool) -> dict[str, Any]:
    return train_one(exp, force=force)


def run_all(clean: bool = False, force: bool = False, start_after: str | None = None, workers: int = 1) -> None:
    if clean:
        clean_outputs()
    else:
        for path in [RESULT_LOGS, RESULT_FIGURES, RESULT_XAI, ONNX_BESTS, OUTPUTS]:
            path.mkdir(parents=True, exist_ok=True)

    full_manifest = build_manifest()
    manifest = _filter_start_after(full_manifest, start_after)
    status_path = RESULT_LOGS / "_v2_run_status.txt"

    completed = 0
    total = len(manifest)
    workers = max(1, int(workers))
    for source, variant, _scope in TRACKS:
        track = [exp for exp in manifest if exp.dataset_source == source and exp.variant == variant]
        if not track:
            continue
        standard = [exp for exp in track if not exp.is_moe]
        moe = [exp for exp in track if exp.is_moe]
        track_name = f"{source}/{variant}"

        if workers == 1 or len(standard) <= 1:
            for exp in standard:
                exp_key = f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}"
                status_path.write_text(
                    f"RUNNING {completed + 1}/{total} {exp_key}\nupdated={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    encoding="utf-8",
                )
                print(f"\n=== [{completed + 1}/{total}] {exp_key} ===", flush=True)
                train_one(exp, force=force)
                completed += 1
                collect_summaries()
        else:
            print(f"\n=== Parallel track {track_name}: {len(standard)} runs with workers={workers} ===", flush=True)
            pending: dict[concurrent.futures.Future[dict[str, Any]], Experiment] = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                for exp in standard:
                    pending[pool.submit(_run_experiment_worker, exp, force)] = exp
                while pending:
                    status_path.write_text(
                        f"RUNNING {completed}/{total} completed; active {len(pending)} in {track_name}\n"
                        f"updated={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                        encoding="utf-8",
                    )
                    done, _ = concurrent.futures.wait(
                        pending,
                        timeout=30,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        collect_summaries()
                        continue
                    for future in done:
                        exp = pending.pop(future)
                        exp_key = f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}"
                        future.result()
                        completed += 1
                        print(f"=== completed [{completed}/{total}] {exp_key} ===", flush=True)
                        collect_summaries()

        for exp in moe:
            exp_key = f"{exp.dataset_source}/{exp.variant}/{exp.experiment_name}"
            status_path.write_text(
                f"RUNNING {completed + 1}/{total} {exp_key}\nupdated={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
            print(f"\n=== [{completed + 1}/{total}] {exp_key} ===", flush=True)
            train_one(exp, force=force)
            completed += 1
            collect_summaries()

    summary = collect_summaries()
    report = verify_outputs(full_manifest)
    generate_comparison_notebooks(execute=True)
    status_path.write_text(
        f"DONE {report['completed_summaries']}/{report['expected_runs']} summaries, "
        f"{report['completed_onnx']}/{report['expected_runs']} ONNX exports\n"
        f"updated={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    print(summary[["dataset_source", "variant", "experiment_name", "test_f1_macro", "test_auc"]].tail(20), flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 IOT fusion ablations with ONNX, XAI, and MoE.")
    parser.add_argument("--clean", action="store_true", help="Delete old v2 result folders before running.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing run outputs.")
    parser.add_argument("--verify", action="store_true", help="Run import/data/model forward checks and exit.")
    parser.add_argument("--verify-export", action="store_true", help="Also export representative ONNX smoke models during verification.")
    parser.add_argument("--run-one", help="Run a single experiment name from the manifest.")
    parser.add_argument("--dataset-source", choices=["processed", "unprocessed"], default="processed")
    parser.add_argument("--variant", choices=["default_balanced", "no_imbalance", "feature_selection"], default="default_balanced")
    parser.add_argument("--start-after", help="Resume the full manifest after this experiment name or source/variant/experiment key.")
    parser.add_argument("--write-notebooks-only", action="store_true", help="Only generate comparison notebooks without training.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel training workers within each non-MoE track.")
    args = parser.parse_args()

    if args.verify:
        verify_pipeline(export_smoke=args.verify_export)
        return
    if args.write_notebooks_only:
        collect_summaries()
        generate_comparison_notebooks(execute=True)
        return
    if args.run_one:
        matches = [
            exp
            for exp in build_manifest()
            if exp.experiment_name == args.run_one and exp.dataset_source == args.dataset_source and exp.variant == args.variant
        ]
        if not matches:
            raise SystemExit(f"No manifest entry for {args.dataset_source}/{args.variant}/{args.run_one}")
        train_one(matches[0], force=args.force)
        collect_summaries()
        return
    run_all(clean=args.clean, force=args.force, start_after=args.start_after, workers=args.workers)


if __name__ == "__main__":
    main()
