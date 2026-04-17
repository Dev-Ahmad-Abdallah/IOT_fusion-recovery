from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/zeus/content/IOT_fusion")
TRAINING = REPO_ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "code") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "code"))

from v2_models import ModelSpec, TwoExpertMoE, build_model, forward_model
from v2_pipeline import AblationDataset, CLASS_NAMES, Experiment, make_data_bundle
from training.SOTA.sota_models import ExpertSpec, SOTAMultiExpertFusion

OUTPUTS = REPO_ROOT / "outputs" / "ablations_v2"
NEW = REPO_ROOT / "NEW"
IMAGE_MODELS = ["cnn", "transformer", "unet", "efficientnet", "conformer", "cvt"]
TABULAR_MODELS = ["mlp", "tabtransformer"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def exp_from_dict(data: dict) -> Experiment:
    return Experiment(
        str(data["dataset_source"]),
        str(data["variant"]),
        str(data["experiment_name"]),
        data.get("image_model"),
        data.get("tabular_model"),
        str(data["mode"]),
        str(data.get("scope", "")),
    )


def checkpoint_path(exp: Experiment) -> Path:
    return OUTPUTS / exp.dataset_source / exp.variant / exp.experiment_name / "best.pth"


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr -= arr.min()
    denom = arr.max() - arr.min()
    if denom > 1e-8:
        arr /= denom
    return np.clip(arr, 0.0, 1.0)


def image_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * STD + MEAN
    return np.clip(arr, 0.0, 1.0)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def save_heatmap(path: Path, heatmap: np.ndarray, title: str, cmap: str = "jet", signed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    if signed:
        denom = float(np.max(np.abs(heatmap))) or 1.0
        data = np.clip(heatmap / denom, -1, 1)
        im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1)
    else:
        data = normalize_map(heatmap)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def targeted_overlay(image: np.ndarray, heatmap: np.ndarray, cmap: str = "jet", alpha: float = 0.58, q: float = 0.72) -> np.ndarray:
    heat = normalize_map(heatmap)
    threshold = float(np.quantile(heat, q))
    mask = np.clip((heat - threshold) / max(1.0 - threshold, 1e-6), 0, 1)
    for _ in range(2):
        pad = np.pad(mask, 1, mode="edge")
        mask = (pad[1:-1, 1:-1] + pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 5.0
    color = plt.get_cmap(cmap)(heat)[..., :3]
    blend = alpha * mask[..., None]
    return np.clip(image * (1.0 - blend) + color * blend, 0.0, 1.0)


def patch_grid_from_scores(scores: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid_y, grid_x = scores.shape
    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    for gy in range(grid_y):
        y0, y1 = int(round(gy * h / grid_y)), int(round((gy + 1) * h / grid_y))
        for gx in range(grid_x):
            x0, x1 = int(round(gx * w / grid_x)), int(round((gx + 1) * w / grid_x))
            out[y0:y1, x0:x1] = scores[gy, gx]
    return out


def occlude_patch(image: torch.Tensor, gy: int, gx: int, grid: int = 8) -> torch.Tensor:
    masked = image.detach().clone()
    _, h, w = masked.shape
    y0, y1 = int(round(gy * h / grid)), int(round((gy + 1) * h / grid))
    x0, x1 = int(round(gx * w / grid)), int(round((gx + 1) * w / grid))
    masked[:, y0:y1, x0:x1] = 0.0
    return masked


def call_model(model: nn.Module, exp: Experiment, image: torch.Tensor | None, metadata: torch.Tensor | None) -> torch.Tensor:
    if exp.experiment_name == "sota_best":
        assert image is not None and metadata is not None
        return model(image, metadata)
    batch: dict[str, torch.Tensor] = {}
    if image is not None:
        batch["image"] = image
    if metadata is not None:
        batch["metadata"] = metadata
    mode = exp.mode if exp.mode != "moe" else "late_fusion"
    return forward_model(model, batch, mode)


def build_regular_model(exp: Experiment, metadata_dim: int, device: torch.device) -> nn.Module:
    model = build_model(ModelSpec(exp.experiment_name, exp.image_model, exp.tabular_model, exp.mode), metadata_dim, len(CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint_path(exp), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_moe_model(exp: Experiment, metadata_dim: int, state_dict: dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    # Try candidate expert architectures until the saved MoE state dict fits exactly.
    for image_name in IMAGE_MODELS:
        for tab_name in TABULAR_MODELS:
            try:
                image = build_model(ModelSpec(f"{image_name}_img_only", image_name, None, "image_only"), metadata_dim, len(CLASS_NAMES)).to(device)
                tab = build_model(ModelSpec(f"{tab_name}_tab_only", None, tab_name, "tabular_only"), metadata_dim, len(CLASS_NAMES)).to(device)
                model = TwoExpertMoE(image, tab, metadata_dim).to(device)
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                return model
            except Exception:
                continue
    raise RuntimeError(f"Could not infer MoE experts for {exp.dataset_source}/{exp.variant}/{exp.experiment_name}")


def build_sota_model(exp: Experiment, ckpt: dict, metadata_dim: int, device: torch.device) -> nn.Module:
    experts = []
    for expert_data in ckpt.get("experts", []):
        expert_exp = exp_from_dict(expert_data)
        expert_ckpt = torch.load(checkpoint_path(expert_exp), map_location=device)
        if expert_exp.mode == "moe":
            expert_model = build_moe_model(expert_exp, metadata_dim, expert_ckpt["model_state_dict"], device)
        else:
            expert_model = build_regular_model(expert_exp, metadata_dim, device)
        for p in expert_model.parameters():
            p.requires_grad = False
        expert_model.eval()
        experts.append((ExpertSpec(expert_exp.experiment_name, expert_exp.mode), expert_model))
    model = SOTAMultiExpertFusion(experts, metadata_dim=metadata_dim, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def build_any_model(exp: Experiment, ckpt: dict, metadata_dim: int, device: torch.device) -> nn.Module:
    if exp.experiment_name == "sota_best":
        return build_sota_model(exp, ckpt, metadata_dim, device)
    if exp.mode == "moe":
        return build_moe_model(exp, metadata_dim, ckpt["model_state_dict"], device)
    return build_regular_model(exp, metadata_dim, device)


def data_mode(exp: Experiment) -> str:
    if exp.mode == "tabular_only":
        return "tabular_only"
    if exp.mode == "image_only":
        return "image_only"
    return "fusion"


def get_sample(exp: Experiment):
    bundle = make_data_bundle(exp.dataset_source, exp.variant)
    ds = AblationDataset(bundle.splits["test"], bundle, data_mode(exp), train=False)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
    return bundle, batch


def last_conv_module(model: nn.Module) -> nn.Module | None:
    found = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            found = module
    return found


def gradcam_or_gradient(model: nn.Module, exp: Experiment, image: torch.Tensor, metadata: torch.Tensor | None, target: int) -> np.ndarray:
    image = image.detach().clone().requires_grad_(True)
    metadata_local = metadata.detach().clone() if metadata is not None else None
    target_module = last_conv_module(model)
    activations, gradients = [], []
    handles = []
    if target_module is not None:
        handles.append(target_module.register_forward_hook(lambda _m, _i, out: activations.append(out)))
        handles.append(target_module.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0])))
    try:
        model.zero_grad(set_to_none=True)
        logits = call_model(model, exp, image, metadata_local)
        logits[:, target].sum().backward()
        if activations and gradients and activations[-1].ndim == 4:
            act = activations[-1].detach()
            grad = gradients[-1].detach()
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
            return normalize_map(cam[0, 0].cpu().numpy())
        grad_map = image.grad.detach().abs().amax(dim=1)[0].cpu().numpy()
        return normalize_map(grad_map)
    finally:
        for h in handles:
            h.remove()


def image_shap_and_lime(model: nn.Module, exp: Experiment, image: torch.Tensor, metadata: torch.Tensor | None, target: int, grid: int = 8):
    with torch.no_grad():
        base_prob = torch.softmax(call_model(model, exp, image, metadata), dim=1)[0, target].item()
    shap_scores = np.zeros((grid, grid), dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            masked = occlude_patch(image[0].cpu(), gy, gx, grid=grid).unsqueeze(0).to(image.device)
            with torch.no_grad():
                prob = torch.softmax(call_model(model, exp, masked, metadata), dim=1)[0, target].item()
            shap_scores[gy, gx] = base_prob - prob
    shap_map = patch_grid_from_scores(shap_scores, tuple(image.shape[-2:]))
    # LIME-style: strongest positive patch regions from the same local perturbation surface.
    lime_scores = normalize_map(np.maximum(shap_scores, 0))
    lime_map = patch_grid_from_scores(lime_scores, tuple(image.shape[-2:]))
    return shap_map, lime_map, shap_scores


def metadata_shap_lime(model: nn.Module, exp: Experiment, metadata: torch.Tensor, image: torch.Tensor | None, target: int, feature_cols: list[str], out_dir: Path):
    with torch.no_grad():
        base_prob = torch.softmax(call_model(model, exp, image, metadata), dim=1)[0, target].item()
    vals = metadata.detach().clone()
    shap_vals = []
    for idx in range(vals.shape[1]):
        masked = vals.clone()
        masked[:, idx] = 0.0
        with torch.no_grad():
            prob = torch.softmax(call_model(model, exp, image, masked), dim=1)[0, target].item()
        shap_vals.append(base_prob - prob)
    shap = pd.DataFrame({"feature": feature_cols, "value": shap_vals})
    shap["abs_value"] = shap["value"].abs()
    shap = shap.sort_values("abs_value", ascending=False)
    shap.to_csv(out_dir / "metadata_shap.csv", index=False)

    # LIME-style local linear proxy around the sample.
    rng = np.random.default_rng(42)
    n = min(96, max(48, vals.shape[1] + 8))
    noise = torch.tensor(rng.normal(0, 0.35, size=(n, vals.shape[1])), dtype=torch.float32, device=vals.device)
    samples = vals.repeat(n, 1) + noise
    samples[0] = vals[0]
    scores = []
    for start in range(0, n, 32):
        chunk = samples[start : start + 32]
        img_chunk = image.repeat(len(chunk), 1, 1, 1) if image is not None else None
        with torch.no_grad():
            probs = torch.softmax(call_model(model, exp, img_chunk, chunk), dim=1)[:, target]
        scores.extend(probs.detach().cpu().numpy().tolist())
    x = samples.detach().cpu().numpy()
    y = np.asarray(scores, dtype=np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    coef = np.linalg.pinv(x.T @ x + 1e-2 * np.eye(x.shape[1])) @ x.T @ (y - y.mean())
    lime = pd.DataFrame({"feature": feature_cols, "value": coef})
    lime["abs_value"] = lime["value"].abs()
    lime = lime.sort_values("abs_value", ascending=False)
    lime.to_csv(out_dir / "metadata_lime.csv", index=False)
    plot_metadata_bar(shap, out_dir / "metadata_shap.png", "Metadata SHAP")
    plot_metadata_bar(lime, out_dir / "metadata_lime.png", "Metadata LIME")


def plot_metadata_bar(df: pd.DataFrame, path: Path, title: str, top_k: int = 12) -> None:
    top = df.head(top_k).iloc[::-1]
    colors = np.where(top["value"].to_numpy() >= 0, "#f28e2b", "#4e79a7")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(range(len(top)), top["value"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(v)[-42:] for v in top["feature"]], fontsize=8)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def process_checkpoint(path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    exp = exp_from_dict(ckpt["experiment"])
    out_dir = NEW / exp.dataset_source / exp.variant / exp.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle, batch = get_sample(exp)
    metadata_dim = len(bundle.feature_cols)
    model = build_any_model(exp, ckpt, metadata_dim, device)
    image = batch.get("image")
    metadata = batch.get("metadata")
    if image is not None:
        image = image.to(device)
    if metadata is not None:
        metadata = metadata.to(device)

    with torch.no_grad():
        logits = call_model(model, exp, image, metadata)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    target = int(probs.argmax())
    rgb = image_to_rgb(image[0]) if image is not None else None
    manifest = {
        "checkpoint": str(path),
        "experiment": asdict(exp),
        "predicted_class": CLASS_NAMES[target],
        "predicted_probability": float(probs[target]),
        "class_probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)},
    }

    if image is not None and rgb is not None:
        save_rgb(out_dir / "original_image.png", rgb)
        gradcam = gradcam_or_gradient(model, exp, image, metadata, target)
        save_heatmap(out_dir / "gradcam_heatmap.png", gradcam, "Grad-CAM Heatmap", cmap="jet")
        save_rgb(out_dir / "gradcam_overlay.png", targeted_overlay(rgb, gradcam, cmap="jet", alpha=0.58, q=0.72))

        shap_map, lime_map, _scores = image_shap_and_lime(model, exp, image, metadata, target, grid=8)
        save_heatmap(out_dir / "shap_heatmap.png", shap_map, "Image SHAP Heatmap", cmap="coolwarm", signed=True)
        save_rgb(out_dir / "shap_overlay.png", targeted_overlay(rgb, np.abs(shap_map), cmap="coolwarm", alpha=0.50, q=0.62))
        save_heatmap(out_dir / "lime_heatmap.png", lime_map, "Image LIME Heatmap", cmap="viridis")
        save_rgb(out_dir / "lime_overlay.png", targeted_overlay(rgb, lime_map, cmap="viridis", alpha=0.52, q=0.70))

    if metadata is not None:
        metadata_shap_lime(model, exp, metadata, image, target, bundle.feature_cols, out_dir)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ok", "out_dir": str(out_dir), **manifest}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NEW.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in OUTPUTS.rglob("best.pth") if "_focal_trials" not in p.parts)
    rows = []
    for idx, path in enumerate(paths, 1):
        print(f"[{idx}/{len(paths)}] {path.relative_to(REPO_ROOT)}", flush=True)
        try:
            rows.append(process_checkpoint(path, device))
        except Exception as exc:
            rows.append({"status": "error", "checkpoint": str(path), "error": repr(exc)})
            print(f"  ERROR {exc!r}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(NEW / "new_xai_manifest.csv", index=False)
    # Validate requested outputs.
    checks = []
    for _, row in df.iterrows():
        out = Path(str(row.get("out_dir", "")))
        if not out.exists():
            continue
        check = {"out_dir": str(out)}
        for name in ["gradcam_heatmap.png", "gradcam_overlay.png", "shap_heatmap.png", "lime_heatmap.png", "metadata_shap.png", "metadata_lime.png"]:
            p = out / name
            check[name] = p.exists() and p.stat().st_size > 1000
        checks.append(check)
    pd.DataFrame(checks).to_csv(NEW / "new_xai_validation.csv", index=False)
    print(df["status"].value_counts(dropna=False).to_string(), flush=True)
    print("wrote", NEW, flush=True)


if __name__ == "__main__":
    main()
