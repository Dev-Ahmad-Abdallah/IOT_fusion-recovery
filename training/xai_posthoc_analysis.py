from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ablation_utils import (
    ABLATIONS,
    OUTPUT_DIR,
    REPO_ROOT,
    RESULT_XAI_DIR,
    FusionTensorDataset,
    build_ablation_model,
    forward_by_mode,
    get_data_info,
    make_split_dataframes,
    move_batch,
)


COMPARISON_DIR = RESULT_XAI_DIR / "Comparisons"


@dataclass(frozen=True)
class Experiment:
    model_name: str
    input_mode: str
    experiment_name: str

    @property
    def uses_image(self) -> bool:
        return self.input_mode != "metadata_only"

    @property
    def uses_metadata(self) -> bool:
        return self.input_mode != "images_only"

    @property
    def is_fusion(self) -> bool:
        return self.input_mode in {"early_fusion", "intermediate_fusion", "late_fusion"}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _to_image_array(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().float().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr) - np.nanmin(arr)
    if denom > 1e-8:
        arr = arr / denom
    return np.clip(arr, 0.0, 1.0)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = values - values.min()
    denom = values.max() - values.min()
    if denom > 1e-8:
        values = values / denom
    return values


def _plot_overlay(image: torch.Tensor, heatmap: np.ndarray, title: str, path: Path, cmap: str = "magma") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _to_image_array(image)
    heatmap = _normalize_map(heatmap)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(base)
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap=cmap)
    plt.title("Attribution")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(base)
    plt.imshow(heatmap, cmap=cmap, alpha=0.45)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, top_k: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["abs_value"] = plot_df[value_col].abs()
    plot_df = plot_df.sort_values("abs_value", ascending=False).head(top_k).sort_values("abs_value")
    plt.figure(figsize=(9, max(4, 0.25 * len(plot_df))))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_modality_contribution(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    summary = df.groupby("experiment_name")[["image_delta", "metadata_delta"]].mean().reset_index()
    x = np.arange(len(summary))
    width = 0.38
    plt.figure(figsize=(max(8, 0.7 * len(summary)), 5))
    plt.bar(x - width / 2, summary["image_delta"], width, label="Image removal delta")
    plt.bar(x + width / 2, summary["metadata_delta"], width, label="Metadata removal delta")
    plt.xticks(x, summary["experiment_name"], rotation=45, ha="right")
    plt.ylabel("Mean drop in predicted-class probability")
    plt.title("Fusion Modality Contribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _load_model(exp: Experiment, info: Any, device: torch.device) -> nn.Module:
    model = build_ablation_model(exp.model_name, exp.input_mode, info).to(device)
    checkpoint_path = OUTPUT_DIR / f"{exp.experiment_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _target_image_module(model: nn.Module, model_name: str) -> nn.Module | None:
    image_model = getattr(model, "image_model", model)
    if model_name == "resnet50" and hasattr(image_model, "layer4"):
        return image_model.layer4[-1]
    if model_name == "cnn" and hasattr(image_model, "layer4"):
        return image_model.layer4[-1]
    if model_name == "transformer" and hasattr(image_model, "patch_embed"):
        return image_model.patch_embed
    return None


def _predict_proba(model: nn.Module, batch: dict[str, torch.Tensor], input_mode: str) -> torch.Tensor:
    logits = forward_by_mode(model, batch, input_mode)
    return torch.softmax(logits, dim=1)


def _grad_cam(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    input_mode: str,
    model_name: str,
    target_class: int,
) -> np.ndarray | None:
    target_module = _target_image_module(model, model_name)
    if target_module is None or "image" not in batch:
        return None

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: nn.Module, _grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0])

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)
    try:
        local = {k: v.detach().clone() for k, v in batch.items()}
        local["image"] = local["image"].requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = forward_by_mode(model, local, input_mode)
        score = logits[:, target_class].sum()
        score.backward()
        if not activations or not gradients:
            return None
        act = activations[-1].detach()
        grad = gradients[-1].detach()
        if act.ndim != 4 or grad.ndim != 4:
            return None
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=local["image"].shape[-2:], mode="bilinear", align_corners=False)
        return _normalize_map(cam[0, 0].cpu().numpy())
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        model.zero_grad(set_to_none=True)


def _mask_image_grid(image: torch.Tensor, mask: np.ndarray, grid_size: int = 7) -> torch.Tensor:
    _, height, width = image.shape
    masked = image.detach().clone()
    baseline = torch.zeros_like(masked)
    index = 0
    for gy in range(grid_size):
        y0 = int(round(gy * height / grid_size))
        y1 = int(round((gy + 1) * height / grid_size))
        for gx in range(grid_size):
            x0 = int(round(gx * width / grid_size))
            x1 = int(round((gx + 1) * width / grid_size))
            if mask[index] < 0.5:
                masked[:, y0:y1, x0:x1] = baseline[:, y0:y1, x0:x1]
            index += 1
    return masked


def _ridge_coefficients(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-8
    y_mean = y.mean()
    xs = (x - x_mean) / x_std
    ys = y - y_mean
    if weights is not None:
        w = np.sqrt(weights).reshape(-1, 1)
        xs = xs * w
        ys = ys * w[:, 0]
    eye = np.eye(xs.shape[1], dtype=np.float64)
    coef = np.linalg.pinv(xs.T @ xs + alpha * eye) @ xs.T @ ys
    return coef / x_std.reshape(-1)


def _lime_image(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    input_mode: str,
    target_class: int,
    device: torch.device,
    grid_size: int = 7,
    n_masks: int = 48,
    batch_size: int = 16,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_segments = grid_size * grid_size
    masks = rng.binomial(1, 0.6, size=(n_masks, n_segments)).astype(np.float32)
    masks[0, :] = 1.0
    masks[1, :] = 0.0
    image = batch["image"][0].detach().cpu()
    scores: list[float] = []
    for start in range(0, len(masks), batch_size):
        chunk = masks[start : start + batch_size]
        images = torch.stack([_mask_image_grid(image, mask, grid_size=grid_size) for mask in chunk]).to(device)
        local = {k: v.repeat(len(chunk), *([1] * (v.ndim - 1))).to(device) for k, v in batch.items() if k != "image"}
        local["image"] = images
        with torch.no_grad():
            probs = _predict_proba(model, local, input_mode)[:, target_class].detach().cpu().numpy()
        scores.extend(probs.tolist())
    distances = np.sqrt(((1.0 - masks) ** 2).mean(axis=1))
    weights = np.exp(-(distances**2) / 0.25)
    coefs = _ridge_coefficients(masks, np.asarray(scores), alpha=1e-2, weights=weights)
    heatmap = np.zeros(tuple(image.shape[-2:]), dtype=np.float32)
    index = 0
    for gy in range(grid_size):
        y0 = int(round(gy * heatmap.shape[0] / grid_size))
        y1 = int(round((gy + 1) * heatmap.shape[0] / grid_size))
        for gx in range(grid_size):
            x0 = int(round(gx * heatmap.shape[1] / grid_size))
            x1 = int(round((gx + 1) * heatmap.shape[1] / grid_size))
            heatmap[y0:y1, x0:x1] = coefs[index]
            index += 1
    return _normalize_map(heatmap)


def _metadata_forward_scores(
    model: nn.Module,
    input_mode: str,
    metadata_values: torch.Tensor,
    fixed_batch: dict[str, torch.Tensor],
    target_class: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, metadata_values.shape[0], batch_size):
        meta = metadata_values[start : start + batch_size].to(device)
        local = {"metadata_tensor": meta}
        if input_mode != "metadata_only":
            local["image"] = fixed_batch["image"].repeat(meta.shape[0], 1, 1, 1).to(device)
        with torch.no_grad():
            probs = _predict_proba(model, local, input_mode)[:, target_class].detach().cpu().numpy()
        scores.append(probs)
    return np.concatenate(scores)


def _kernelshap_metadata(
    model: nn.Module,
    input_mode: str,
    sample_meta: torch.Tensor,
    baseline_meta: torch.Tensor,
    fixed_batch: dict[str, torch.Tensor],
    target_class: int,
    device: torch.device,
    n_masks: int = 80,
    seed: int = 17,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = sample_meta.numel()
    masks = rng.binomial(1, 0.5, size=(n_masks, n_features)).astype(np.float32)
    masks[0, :] = 0.0
    masks[1, :] = 1.0
    masks = np.unique(masks, axis=0)
    mask_tensor = torch.tensor(masks, dtype=torch.float32)
    sample = sample_meta.detach().cpu().view(1, -1)
    baseline = baseline_meta.detach().cpu().view(1, -1)
    perturbed = baseline + mask_tensor * (sample - baseline)
    scores = _metadata_forward_scores(model, input_mode, perturbed, fixed_batch, target_class, device)
    z = masks.sum(axis=1)
    kernel = (n_features - 1.0) / (np.maximum(z, 1.0) * np.maximum(n_features - z, 1.0))
    kernel[(z == 0) | (z == n_features)] = 100.0
    return _ridge_coefficients(masks, scores, alpha=1e-2, weights=kernel).astype(np.float32)


def _lime_metadata(
    model: nn.Module,
    input_mode: str,
    sample_meta: torch.Tensor,
    metadata_std: torch.Tensor,
    fixed_batch: dict[str, torch.Tensor],
    target_class: int,
    device: torch.device,
    n_samples: int = 96,
    seed: int = 23,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample = sample_meta.detach().cpu().view(1, -1)
    std = torch.clamp(metadata_std.detach().cpu().view(1, -1), min=1e-3)
    noise = torch.tensor(rng.normal(0.0, 0.35, size=(n_samples, sample.shape[1])), dtype=torch.float32) * std
    perturbed = sample + noise
    perturbed[0] = sample[0]
    scores = _metadata_forward_scores(model, input_mode, perturbed, fixed_batch, target_class, device)
    distances = torch.sqrt(((perturbed - sample) ** 2 / (std**2)).mean(dim=1)).numpy()
    weights = np.exp(-(distances**2) / 1.0)
    return _ridge_coefficients(perturbed.numpy(), scores, alpha=1e-2, weights=weights).astype(np.float32)


def _modality_contribution(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    input_mode: str,
    baseline_meta: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> dict[str, float]:
    if input_mode not in {"early_fusion", "intermediate_fusion", "late_fusion"}:
        return {}
    with torch.no_grad():
        original = _predict_proba(model, batch, input_mode)[0, target_class].item()
        image_masked = {k: v.detach().clone() for k, v in batch.items()}
        image_masked["image"] = torch.zeros_like(image_masked["image"])
        image_removed = _predict_proba(model, image_masked, input_mode)[0, target_class].item()
        metadata_masked = {k: v.detach().clone() for k, v in batch.items()}
        metadata_masked["metadata_tensor"] = baseline_meta.view(1, -1).to(device)
        metadata_removed = _predict_proba(model, metadata_masked, input_mode)[0, target_class].item()
    return {
        "original_prob": original,
        "image_removed_prob": image_removed,
        "metadata_removed_prob": metadata_removed,
        "image_delta": original - image_removed,
        "metadata_delta": original - metadata_removed,
    }


def _select_balanced_samples(test_df: pd.DataFrame, max_per_class: int = 1) -> pd.DataFrame:
    samples = []
    for label, group in test_df.groupby("label_raw", sort=True):
        samples.append(group.head(max_per_class))
    return pd.concat(samples, ignore_index=True)


def _sample_batch(row: pd.Series, info: Any, input_mode: str, device: torch.device) -> dict[str, torch.Tensor]:
    dataset = FusionTensorDataset(pd.DataFrame([row]), info, input_mode)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    return move_batch(batch, device, input_mode)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_posthoc_xai(
    *,
    max_samples_per_class: int = 1,
    image_lime_masks: int = 48,
    metadata_shap_masks: int = 80,
    metadata_lime_samples: int = 96,
    device_name: str = "cuda",
) -> pd.DataFrame:
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    RESULT_XAI_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    info = get_data_info()
    split_dfs = make_split_dataframes(seed=42)
    test_samples = _select_balanced_samples(split_dfs["test"], max_per_class=max_samples_per_class)
    train_meta = torch.tensor(split_dfs["train"][info.feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    baseline_meta = train_meta.mean(dim=0).to(device)
    metadata_std = train_meta.std(dim=0).to(device)

    all_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for model_name, input_mode, experiment_name in tqdm(ABLATIONS, desc="XAI experiments"):
        exp = Experiment(model_name, input_mode, experiment_name)
        out_dir = RESULT_XAI_DIR / exp.experiment_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gradcam").mkdir(exist_ok=True)
        (out_dir / "lime_image").mkdir(exist_ok=True)
        model = _load_model(exp, info, device)
        target_module = _target_image_module(model, exp.model_name) if exp.uses_image else None

        exp_rows: list[dict[str, Any]] = []
        exp_metadata_shap: list[np.ndarray] = []
        exp_metadata_lime: list[np.ndarray] = []

        for sample_idx, row in tqdm(list(test_samples.iterrows()), desc=exp.experiment_name, leave=False):
            batch = _sample_batch(row, info, exp.input_mode, device)
            with torch.no_grad():
                probs = _predict_proba(model, batch, exp.input_mode)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_label = info.class_names[pred_idx]
            true_label = str(row["label_raw"])
            pred_prob = float(probs[0, pred_idx].item())
            sample_name = f"{sample_idx:02d}_{_safe_name(true_label)}_pred_{_safe_name(pred_label)}"

            row_record: dict[str, Any] = {
                "experiment_name": exp.experiment_name,
                "model_name": exp.model_name,
                "input_mode": exp.input_mode,
                "sample_index": int(sample_idx),
                "img_id": str(row["img_id"]),
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": pred_prob,
            }

            if exp.uses_image and target_module is not None:
                cam = _grad_cam(model, batch, exp.input_mode, exp.model_name, pred_idx)
                if cam is not None:
                    cam_path = out_dir / "gradcam" / f"{sample_name}_gradcam.png"
                    npy_path = out_dir / "gradcam" / f"{sample_name}_gradcam.npy"
                    _plot_overlay(batch["image"][0], cam, "Grad-CAM", cam_path, cmap="magma")
                    np.save(npy_path, cam)
                    row_record["gradcam_path"] = str(cam_path)
                    row_record["gradcam_npy_path"] = str(npy_path)

                lime_map = _lime_image(
                    model,
                    batch,
                    exp.input_mode,
                    pred_idx,
                    device,
                    n_masks=image_lime_masks,
                    seed=1000 + sample_idx,
                )
                lime_path = out_dir / "lime_image" / f"{sample_name}_lime_image.png"
                lime_npy = out_dir / "lime_image" / f"{sample_name}_lime_image.npy"
                _plot_overlay(batch["image"][0], lime_map, "LIME image", lime_path, cmap="viridis")
                np.save(lime_npy, lime_map)
                row_record["lime_image_path"] = str(lime_path)
                row_record["lime_image_npy_path"] = str(lime_npy)

            if exp.uses_metadata:
                shap_values = _kernelshap_metadata(
                    model,
                    exp.input_mode,
                    batch["metadata_tensor"][0],
                    baseline_meta,
                    batch,
                    pred_idx,
                    device,
                    n_masks=metadata_shap_masks,
                    seed=2000 + sample_idx,
                )
                lime_values = _lime_metadata(
                    model,
                    exp.input_mode,
                    batch["metadata_tensor"][0],
                    metadata_std,
                    batch,
                    pred_idx,
                    device,
                    n_samples=metadata_lime_samples,
                    seed=3000 + sample_idx,
                )
                exp_metadata_shap.append(shap_values)
                exp_metadata_lime.append(lime_values)
                for method, values in [("kernelshap", shap_values), ("lime_metadata", lime_values)]:
                    for feature, value in zip(info.feature_cols, values):
                        metadata_rows.append(
                            {
                                "experiment_name": exp.experiment_name,
                                "model_name": exp.model_name,
                                "input_mode": exp.input_mode,
                                "sample_index": int(sample_idx),
                                "true_label": true_label,
                                "pred_label": pred_label,
                                "method": method,
                                "feature": feature,
                                "attribution": float(value),
                                "abs_attribution": float(abs(value)),
                            }
                        )

            if exp.is_fusion:
                contribution = _modality_contribution(model, batch, exp.input_mode, baseline_meta, pred_idx, device)
                contribution.update(row_record)
                modality_rows.append(contribution)
                row_record.update({f"modality_{k}": v for k, v in contribution.items() if isinstance(v, float)})

            exp_rows.append(row_record)
            all_rows.append(row_record)

        pd.DataFrame(exp_rows).to_csv(out_dir / "xai_sample_index.csv", index=False)

        if exp_metadata_shap:
            shap_arr = np.vstack(exp_metadata_shap)
            lime_arr = np.vstack(exp_metadata_lime)
            for method, arr in [("kernelshap", shap_arr), ("lime_metadata", lime_arr)]:
                summary = pd.DataFrame(
                    {
                        "feature": info.feature_cols,
                        "mean_attribution": arr.mean(axis=0),
                        "mean_abs_attribution": np.abs(arr).mean(axis=0),
                    }
                ).sort_values("mean_abs_attribution", ascending=False)
                summary.to_csv(out_dir / f"{method}_metadata_summary.csv", index=False)
                _plot_bar(
                    summary,
                    "feature",
                    "mean_attribution",
                    f"{exp.experiment_name} {method} metadata attribution",
                    out_dir / f"{method}_metadata_summary.png",
                    top_k=20,
                )

        _save_json(
            out_dir / "xai_run_config.json",
            {
                "experiment_name": exp.experiment_name,
                "model_name": exp.model_name,
                "input_mode": exp.input_mode,
                "max_samples_per_class": max_samples_per_class,
                "image_lime_masks": image_lime_masks,
                "metadata_shap_masks": metadata_shap_masks,
                "metadata_lime_samples": metadata_lime_samples,
                "notes": {
                    "gradcam": "Gradient-weighted activation map on the final spatial image module.",
                    "kernelshap": "Dependency-free KernelSHAP-style local surrogate on metadata features.",
                    "lime": "Dependency-free LIME-style local surrogate for image patches and metadata perturbations.",
                    "modality_delta": "Drop in predicted-class probability when image or metadata is replaced by a baseline.",
                },
            },
        )

    sample_df = pd.DataFrame(all_rows)
    sample_df.to_csv(RESULT_XAI_DIR / "xai_all_sample_index.csv", index=False)

    metadata_df = pd.DataFrame(metadata_rows)
    if not metadata_df.empty:
        metadata_df.to_csv(RESULT_XAI_DIR / "metadata_attributions_all_models.csv", index=False)
        top_meta = (
            metadata_df.groupby(["experiment_name", "method", "feature"], as_index=False)["abs_attribution"]
            .mean()
            .sort_values(["experiment_name", "method", "abs_attribution"], ascending=[True, True, False])
        )
        top_meta.to_csv(COMPARISON_DIR / "top_metadata_features_by_model.csv", index=False)

    modality_df = pd.DataFrame(modality_rows)
    if not modality_df.empty:
        modality_df.to_csv(COMPARISON_DIR / "modality_contribution_by_sample.csv", index=False)
        modality_summary = (
            modality_df.groupby(["experiment_name", "model_name", "input_mode"], as_index=False)[
                ["image_delta", "metadata_delta", "original_prob"]
            ]
            .mean()
            .sort_values("experiment_name")
        )
        modality_summary.to_csv(COMPARISON_DIR / "modality_contribution_summary.csv", index=False)
        _plot_modality_contribution(modality_df, COMPARISON_DIR / "modality_contribution_summary.png")

    pair_rows = []
    pairings = [
        ("cnn_img_only", "transformer_img_only", "image-only CNN vs Transformer"),
        ("cnn_meta_only", "transformer_meta_only", "metadata-only CNN MLP vs Transformer MLP"),
        ("cnn_early_fusion", "transformer_early_fusion", "early fusion CNN vs Transformer"),
        ("cnn_inter_fusion", "transformer_inter_fusion", "intermediate fusion CNN vs Transformer"),
        ("cnn_late_fusion", "transformer_late_fusion", "late fusion CNN vs Transformer"),
        ("cnn_img_only", "cnn_meta_only", "CNN image vs metadata"),
        ("transformer_img_only", "transformer_meta_only", "Transformer image vs metadata"),
        ("cnn_inter_fusion", "cnn_late_fusion", "CNN intermediate vs late fusion"),
        ("transformer_inter_fusion", "transformer_late_fusion", "Transformer intermediate vs late fusion"),
    ]
    for left, right, label in pairings:
        left_result = REPO_ROOT / "Results_Logs" / left / "result_summary.csv"
        right_result = REPO_ROOT / "Results_Logs" / right / "result_summary.csv"
        if left_result.exists() and right_result.exists():
            ldf = pd.read_csv(left_result).iloc[0]
            rdf = pd.read_csv(right_result).iloc[0]
            pair_rows.append(
                {
                    "comparison": label,
                    "left_experiment": left,
                    "right_experiment": right,
                    "left_test_acc": ldf["test_acc"],
                    "right_test_acc": rdf["test_acc"],
                    "delta_test_acc_left_minus_right": ldf["test_acc"] - rdf["test_acc"],
                    "left_test_bacc": ldf["test_bacc"],
                    "right_test_bacc": rdf["test_bacc"],
                    "delta_test_bacc_left_minus_right": ldf["test_bacc"] - rdf["test_bacc"],
                    "left_test_auc": ldf["test_auc"],
                    "right_test_auc": rdf["test_auc"],
                    "delta_test_auc_left_minus_right": ldf["test_auc"] - rdf["test_auc"],
                    "left_test_f1_macro": ldf["test_f1_macro"],
                    "right_test_f1_macro": rdf["test_f1_macro"],
                    "delta_test_f1_left_minus_right": ldf["test_f1_macro"] - rdf["test_f1_macro"],
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(COMPARISON_DIR / "pairwise_performance_comparison.csv", index=False)

    if not metadata_df.empty:
        pivot = (
            metadata_df[metadata_df["method"] == "kernelshap"]
            .groupby(["experiment_name", "feature"])["abs_attribution"]
            .mean()
            .unstack(fill_value=0.0)
        )
        similarity = pd.DataFrame(
            cosine_similarity(pivot.values),
            index=pivot.index,
            columns=pivot.index,
        )
        similarity.to_csv(COMPARISON_DIR / "metadata_kernelshap_similarity.csv")
        plt.figure(figsize=(9, 7))
        plt.imshow(similarity.values, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(label="Cosine similarity")
        plt.xticks(np.arange(len(similarity.columns)), similarity.columns, rotation=90)
        plt.yticks(np.arange(len(similarity.index)), similarity.index)
        plt.title("Metadata KernelSHAP Attribution Similarity")
        plt.tight_layout()
        plt.savefig(COMPARISON_DIR / "metadata_kernelshap_similarity.png", dpi=160)
        plt.close()

    _save_json(
        COMPARISON_DIR / "xai_manifest.json",
        {
            "experiments": [exp for _, _, exp in ABLATIONS],
            "sample_count": int(len(test_samples)),
            "outputs": {
                "all_sample_index": str(RESULT_XAI_DIR / "xai_all_sample_index.csv"),
                "metadata_attributions": str(RESULT_XAI_DIR / "metadata_attributions_all_models.csv"),
                "modality_summary": str(COMPARISON_DIR / "modality_contribution_summary.csv"),
                "pairwise_performance": str(COMPARISON_DIR / "pairwise_performance_comparison.csv"),
                "metadata_similarity": str(COMPARISON_DIR / "metadata_kernelshap_similarity.csv"),
            },
        },
    )
    return pair_df


def main() -> None:
    pair_df = run_posthoc_xai()
    print("Post-hoc XAI complete.")
    print(f"Results root: {RESULT_XAI_DIR}")
    print(f"Comparison root: {COMPARISON_DIR}")
    if not pair_df.empty:
        print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
