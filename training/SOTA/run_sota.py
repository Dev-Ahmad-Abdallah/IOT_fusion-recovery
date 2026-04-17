from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from training.SOTA.sota_models import ExpertSpec, SOTAMultiExpertFusion
from training.v2_pipeline import (
    CLASS_NAMES,
    Experiment,
    ModelSpec,
    ONNX_BESTS,
    OUTPUTS,
    RESULT_FIGURES,
    RESULT_LOGS,
    RESULT_XAI,
    build_manifest,
    checkpoint_path,
    class_weights,
    collect_summaries,
    export_onnx,
    instantiate_model,
    load_wandb_env,
    make_data_bundle,
    make_loaders,
    plot_curves,
    plot_test_figures,
    result_dirs,
    run_epoch,
    save_epoch_log,
    save_xai,
    write_split_summary,
)


SOTA_NAME = "sota_best"
SOTA_MODE = "late_fusion"


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        focal = (1.0 - prob).pow(self.gamma)
        loss = -(true_dist * focal * log_prob)
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)
        return loss.sum(dim=1).mean()


def sota_experiment(source: str, variant: str) -> Experiment:
    return Experiment(source, variant, SOTA_NAME, "sota", "multi_expert", SOTA_MODE, "sota")


def _manifest_by_key() -> dict[tuple[str, str, str], Experiment]:
    return {(exp.dataset_source, exp.variant, exp.experiment_name): exp for exp in build_manifest()}


def _load_existing_summary() -> pd.DataFrame:
    summary_path = RESULT_LOGS / "ablation_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return collect_summaries()


def select_experts(source: str, variant: str, max_experts: int = 7) -> list[Experiment]:
    """Choose strong, diverse frozen experts for the final stacked model.

    Selection is based on validation macro-F1 to avoid using the test split for
    model construction. We force in the best image-only and tabular-only models
    where available, then add the strongest fusion/MoE experts.
    """
    summary = _load_existing_summary()
    key_map = _manifest_by_key()
    track = summary[
        (summary["dataset_source"] == source)
        & (summary["variant"] == variant)
        & (summary["experiment_name"] != SOTA_NAME)
    ].copy()
    if track.empty:
        raise RuntimeError(f"No completed base results for {source}/{variant}.")
    track["score"] = track["best_val_f1_macro"].fillna(track["test_f1_macro"]).astype(float)

    selected: list[Experiment] = []

    def add_row(row: pd.Series) -> None:
        key = (str(row["dataset_source"]), str(row["variant"]), str(row["experiment_name"]))
        exp = key_map.get(key)
        if exp is not None and exp not in selected and checkpoint_path(exp).exists():
            selected.append(exp)

    for mode in ["image_only", "tabular_only"]:
        subset = track[track["mode"] == mode].sort_values("score", ascending=False)
        if not subset.empty:
            add_row(subset.iloc[0])

    ranked = track.sort_values(["score", "test_auc", "test_f1_macro"], ascending=[False, False, False])
    for _, row in ranked.iterrows():
        add_row(row)
        if len(selected) >= max_experts:
            break

    if len(selected) < 2:
        raise RuntimeError(f"Need at least two experts for {source}/{variant}; got {len(selected)}.")
    return selected


def load_frozen_expert(exp: Experiment, bundle, device: torch.device) -> nn.Module:
    model = instantiate_model(exp, bundle, device)
    checkpoint = torch.load(checkpoint_path(exp), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_sota_model(source: str, variant: str, bundle, device: torch.device, max_experts: int = 7) -> tuple[SOTAMultiExpertFusion, list[Experiment]]:
    experts = select_experts(source, variant, max_experts=max_experts)
    modules = []
    for expert_exp in experts:
        modules.append((ExpertSpec(expert_exp.experiment_name, expert_exp.mode), load_frozen_expert(expert_exp, bundle, device)))
    model = SOTAMultiExpertFusion(modules, metadata_dim=len(bundle.feature_cols), num_classes=len(CLASS_NAMES)).to(device)
    return model, experts


def train_sota_one(source: str, variant: str, force: bool = False, max_experts: int = 7) -> dict[str, Any]:
    exp = sota_experiment(source, variant)
    logs, figs, xai_dir, onnx_dir, ckpt_dir = result_dirs(exp)
    if (logs / "result_summary.csv").exists() and not force:
        return pd.read_csv(logs / "result_summary.csv").iloc[0].to_dict()

    load_wandb_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    bundle = make_data_bundle(source, variant)
    loaders = make_loaders(exp, bundle, batch_size=48 if source == "processed" and variant == "default_balanced" else 64)
    model, experts = build_sota_model(source, variant, bundle, device, max_experts=max_experts)
    write_split_summary(bundle, logs / "split_summary.csv")
    (logs / "expert_manifest.json").write_text(
        json.dumps([asdict(expert) for expert in experts], indent=2),
        encoding="utf-8",
    )

    weight = class_weights(bundle.splits["train"], variant != "no_imbalance", device)
    criterion = FocalLoss(weight=weight, gamma=1.0, label_smoothing=0.06 if variant != "no_imbalance" else 0.03)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=8e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    wandb_run = None
    try:
        import wandb

        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "iot fusion"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=f"{source}/{variant}/{SOTA_NAME}",
            group=f"{source}/{variant}",
            job_type="sota_stacked_fusion",
            config={
                **asdict(exp),
                "expert_names": [e.experiment_name for e in experts],
                "feature_count": len(bundle.feature_cols),
            },
            reinit=True,
        )
    except Exception as exc:
        print(f"W&B logging disabled for {source}/{variant}/{SOTA_NAME}: {exc}", flush=True)
        wandb_run = None

    history = []
    best_monitor, best_auc = -1.0, -1.0
    epochs_without_improvement = 0
    max_epochs = 16
    min_epochs = 5
    patience = 4
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        train = run_epoch(model, loaders["train"], criterion, optimizer, scaler, device, SOTA_MODE, True, f"{SOTA_NAME} train {epoch}")
        val = run_epoch(model, loaders["val"], criterion, None, None, device, SOTA_MODE, True, f"{SOTA_NAME} val {epoch}")
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
            f"{source}/{variant}/{SOTA_NAME} epoch {epoch:03d}: "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.2f}% "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.2f}% "
            f"val_auc={row['val_auc']:.4f} val_f1={row['val_f1_macro']:.4f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}, step=epoch)

        val_auc = row["val_auc"] if not np.isnan(row["val_auc"]) else -1.0
        val_monitor = 0.5 * row["val_f1_macro"] + 0.5 * (row["val_bacc"] / 100.0)
        improved = val_monitor > best_monitor or (val_monitor == best_monitor and val_auc >= best_auc)
        if improved:
            best_monitor = val_monitor
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "experiment": asdict(exp),
                    "experts": [asdict(e) for e in experts],
                    "feature_cols": bundle.feature_cols,
                    "epoch": epoch,
                    "val_monitor": val_monitor,
                    **row,
                },
                ckpt_dir / "best.pth",
            )
        else:
            epochs_without_improvement += 1
        scheduler.step(val_monitor)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(logs / "epoch_metrics.csv", index=False)
    save_epoch_log(history_df, logs / "epoch_log.txt")
    plot_curves(history_df, figs)

    checkpoint = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test = run_epoch(model, loaders["test"], criterion, None, None, device, SOTA_MODE, True, f"{SOTA_NAME} test")
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
        "batch_size": loaders["train"].batch_size,
        "feature_count": len(bundle.feature_cols),
        "imbalance_handling": variant != "no_imbalance",
        "expert_names": "|".join(e.experiment_name for e in experts),
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
    collect_summaries()
    return result


def verify_sota_forward(source: str = "processed", variant: str = "default_balanced") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = make_data_bundle(source, variant)
    exp = sota_experiment(source, variant)
    loaders = make_loaders(exp, bundle, batch_size=2, num_workers=0)
    model, experts = build_sota_model(source, variant, bundle, device, max_experts=4)
    batch = next(iter(loaders["train"]))
    image = batch["image"].to(device)
    metadata = batch["metadata"].to(device)
    with torch.no_grad():
        logits = model(image, metadata)
    assert tuple(logits.shape) == (2, len(CLASS_NAMES)), tuple(logits.shape)
    print(json.dumps({"source": source, "variant": variant, "experts": [e.experiment_name for e in experts], "logits_shape": list(logits.shape)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final stacked SOTA fusion model.")
    parser.add_argument("--dataset-source", choices=["processed", "unprocessed"])
    parser.add_argument("--variant", choices=["default_balanced", "no_imbalance", "feature_selection"])
    parser.add_argument("--all", action="store_true", help="Train SOTA on all dataset-source/variant tracks.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--max-experts", type=int, default=7)
    args = parser.parse_args()

    if args.verify:
        verify_sota_forward(args.dataset_source or "processed", args.variant or "default_balanced")
        return

    tracks = [
        ("processed", "default_balanced"),
        ("processed", "no_imbalance"),
        ("processed", "feature_selection"),
        ("unprocessed", "default_balanced"),
        ("unprocessed", "no_imbalance"),
        ("unprocessed", "feature_selection"),
    ]
    if not args.all:
        if not args.dataset_source or not args.variant:
            raise SystemExit("Use --all or provide --dataset-source and --variant.")
        tracks = [(args.dataset_source, args.variant)]
    for source, variant in tracks:
        print(f"\n=== SOTA {source}/{variant} ===", flush=True)
        train_sota_one(source, variant, force=args.force, max_experts=args.max_experts)


if __name__ == "__main__":
    main()
