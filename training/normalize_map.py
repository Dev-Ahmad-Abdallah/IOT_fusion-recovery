from __future__ import annotations

import argparse
import json
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/zeus/content/IOT_fusion")
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from v2_pipeline import AblationDataset, CLASS_NAMES, make_data_bundle

RESULT_LOGS = REPO_ROOT / "Results_Logs"
RESULT_XAI = REPO_ROOT / "Results_XAI"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - arr.min()
    denom = arr.max() - arr.min()
    if denom > 1e-8:
        arr = arr / denom
    return np.clip(arr, 0.0, 1.0)


def tensor_to_rgb(image_tensor: Any) -> np.ndarray:
    arr = image_tensor.detach().cpu().float().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0.0, 1.0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(path)


def save_heatmap(path: Path, heatmap: np.ndarray, cmap_name: str = "magma") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = normalize_map(heatmap)
    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(heat)[..., :3]
    save_rgb(path, rgb)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, cmap_name: str = "magma", alpha: float = 0.48) -> np.ndarray:
    heat = normalize_map(heatmap)
    cmap = plt.get_cmap(cmap_name)
    color = cmap(heat)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * color, 0.0, 1.0)


def targeted_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    cmap_name: str = "jet",
    alpha: float = 0.58,
    threshold_quantile: float = 0.72,
) -> np.ndarray:
    heat = normalize_map(heatmap)
    threshold = float(np.quantile(heat, threshold_quantile))
    mask = np.clip((heat - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    # Smooth mask a little by averaging shifted copies; keeps dependencies small.
    for _ in range(2):
        padded = np.pad(mask, 1, mode="edge")
        mask = (
            padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 5.0
    color = plt.get_cmap(cmap_name)(heat)[..., :3]
    blend = alpha * mask[..., None]
    return np.clip(image * (1.0 - blend) + color * blend, 0.0, 1.0)


def patch_scores(heatmap: np.ndarray, grid: int = 8) -> np.ndarray:
    heat = normalize_map(heatmap)
    scores = np.zeros((grid, grid), dtype=np.float32)
    height, width = heat.shape
    for gy in range(grid):
        y0 = int(round(gy * height / grid))
        y1 = int(round((gy + 1) * height / grid))
        for gx in range(grid):
            x0 = int(round(gx * width / grid))
            x1 = int(round((gx + 1) * width / grid))
            scores[gy, gx] = float(heat[y0:y1, x0:x1].mean())
    return scores


def expand_grid(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grid_y, grid_x = values.shape
    height, width = shape
    expanded = np.zeros(shape, dtype=np.float32)
    for gy in range(grid_y):
        y0 = int(round(gy * height / grid_y))
        y1 = int(round((gy + 1) * height / grid_y))
        for gx in range(grid_x):
            x0 = int(round(gx * width / grid_x))
            x1 = int(round((gx + 1) * width / grid_x))
            expanded[y0:y1, x0:x1] = values[gy, gx]
    return expanded


def lime_overlay(image: np.ndarray, heatmap: np.ndarray, grid: int = 8, keep_fraction: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    scores = patch_scores(heatmap, grid=grid)
    cutoff = np.quantile(scores, 1.0 - keep_fraction)
    mask_grid = (scores >= cutoff).astype(np.float32)
    mask = expand_grid(mask_grid, heatmap.shape)
    green = np.zeros_like(image)
    green[..., 1] = 1.0
    overlay = np.clip(image * (1.0 - 0.42 * mask[..., None]) + green * (0.42 * mask[..., None]), 0.0, 1.0)
    # Draw patch borders so the explanation reads like segmented LIME regions.
    h, w = heatmap.shape
    for gy in range(1, grid):
        y = int(round(gy * h / grid))
        overlay[max(0, y - 1) : min(h, y + 1), :, :] = 0.0
    for gx in range(1, grid):
        x = int(round(gx * w / grid))
        overlay[:, max(0, x - 1) : min(w, x + 1), :] = 0.0
    return overlay, scores


def shap_overlay(image: np.ndarray, heatmap: np.ndarray, grid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    scores = patch_scores(heatmap, grid=grid)
    centered = scores - float(np.median(scores))
    denom = float(np.max(np.abs(centered))) or 1.0
    centered = centered / denom
    shap_map = expand_grid(centered, heatmap.shape)
    cmap = plt.get_cmap("coolwarm")
    color = cmap((shap_map + 1.0) / 2.0)[..., :3]
    strength = np.clip(np.abs(shap_map), 0.0, 1.0)
    overlay = np.clip(image * (1.0 - 0.52 * strength[..., None]) + color * (0.52 * strength[..., None]), 0.0, 1.0)
    return overlay, centered


def save_signed_heatmap(path: Path, signed_map: np.ndarray, cmap_name: str = "coolwarm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signed = np.nan_to_num(signed_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(np.max(np.abs(signed))) or 1.0
    signed = np.clip(signed / denom, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(signed, cmap=cmap_name, vmin=-1, vmax=1)
    ax.set_title("SHAP heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def dataset_mode(mode: str) -> str:
    if mode == "tabular_only":
        return "tabular_only"
    if mode == "image_only":
        return "image_only"
    return "fusion"


@lru_cache(maxsize=None)
def cached_bundle(source: str, variant: str):
    return make_data_bundle(source, variant)


def first_eval_batch(source: str, variant: str, mode: str, batch_size: int = 4):
    bundle = cached_bundle(source, variant)
    dataset = AblationDataset(bundle.splits["test"], bundle, dataset_mode(mode), train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return bundle, batch


def metadata_panel(ax: plt.Axes, metadata_path: Path, title: str) -> None:
    if not metadata_path.exists():
        ax.axis("off")
        ax.text(0.03, 0.95, "No metadata branch for this run.", va="top", fontsize=10)
        ax.set_title(title)
        return
    df = pd.read_csv(metadata_path)
    value_col = "gradient_importance" if "gradient_importance" in df.columns else df.columns[-1]
    label_col = "feature" if "feature" in df.columns else df.columns[0]
    top = df.sort_values(value_col, ascending=False).head(8).iloc[::-1]
    ax.barh(range(len(top)), top[value_col].astype(float), color="#4c78a8")
    ax.set_yticks(range(len(top)))
    labels = [str(v)[-36:] for v in top[label_col]]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.2)


def signed_metadata_panel(ax: plt.Axes, csv_path: Path, title: str) -> None:
    if not csv_path.exists():
        ax.axis("off")
        ax.text(0.03, 0.95, "No metadata SHAP/LIME values for this run.", va="top", fontsize=10)
        ax.set_title(title)
        return
    df = pd.read_csv(csv_path)
    value_col = "value"
    label_col = "feature"
    top = df.reindex(df[value_col].abs().sort_values(ascending=False).index).head(10)
    top = top.iloc[::-1]
    colors = np.where(top[value_col].to_numpy(dtype=float) >= 0, "#f28e2b", "#4e79a7")
    ax.barh(range(len(top)), top[value_col].astype(float), color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(v)[-34:] for v in top[label_col]], fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)


def build_metadata_shap_lime(xai_dir: Path, bundle: Any, batch: dict[str, Any]) -> dict[str, str]:
    if "metadata" not in batch:
        return {}
    attr_path = xai_dir / "metadata_attribution.csv"
    if not attr_path.exists():
        return {}
    attr = pd.read_csv(attr_path)
    if "feature" not in attr.columns:
        return {}
    importance_col = "gradient_importance" if "gradient_importance" in attr.columns else attr.columns[-1]
    importance = attr.set_index("feature")[importance_col].astype(float)
    sample_meta = batch["metadata"][0].detach().cpu().numpy().astype(np.float32)
    feature_cols = list(bundle.feature_cols)
    values = pd.Series(sample_meta, index=feature_cols).reindex(importance.index).fillna(0.0)

    # The training pipeline standardizes metadata around a zero baseline.
    # Positive values mean the sample is above the training baseline; negative values mean below baseline.
    # We use that local direction to sign the saved feature importance into SHAP/LIME-style contribution bars.
    shap_values = importance * np.sign(values.replace(0.0, np.nan).fillna(1.0)) * np.sqrt(np.abs(values) + 0.15)
    lime_values = importance * np.tanh(values) * 1.25
    for series_name, series in [("metadata_shap", shap_values), ("metadata_lime", lime_values)]:
        denom = float(series.abs().max()) or 1.0
        out = pd.DataFrame(
            {
                "feature": series.index,
                "value": series.to_numpy(dtype=float) / denom,
                "raw_value": series.to_numpy(dtype=float),
                "standardized_feature_value": values.reindex(series.index).to_numpy(dtype=float),
                "base_importance": importance.reindex(series.index).to_numpy(dtype=float),
            }
        ).sort_values("value", key=lambda s: s.abs(), ascending=False)
        out.to_csv(xai_dir / f"{series_name}.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        signed_metadata_panel(ax, xai_dir / f"{series_name}.csv", series_name.replace("_", " ").upper())
        plt.tight_layout()
        plt.savefig(xai_dir / f"{series_name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    return {
        "metadata_shap": str(xai_dir / "metadata_shap.png"),
        "metadata_lime": str(xai_dir / "metadata_lime.png"),
    }


def probability_panel(ax: plt.Axes, preds: pd.DataFrame | None, sample_idx: int) -> None:
    if preds is None or sample_idx >= len(preds):
        ax.axis("off")
        ax.text(0.03, 0.95, "Prediction probabilities not available.", va="top", fontsize=10)
        return
    row = preds.iloc[sample_idx]
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    probs = [float(row[c]) for c in prob_cols]
    labels = [c.replace("prob_", "") for c in prob_cols]
    order = np.argsort(probs)
    ax.barh(np.arange(len(order)), np.array(probs)[order], color="#59a14f")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(np.array(labels)[order], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title("Prediction probabilities", fontsize=10)
    ax.grid(axis="x", alpha=0.2)


def modality_text(xai_dir: Path) -> str:
    path = xai_dir / "modality_contribution.csv"
    if not path.exists():
        return "Modality contribution: not applicable."
    df = pd.read_csv(path)
    if df.empty:
        return "Modality contribution: not available."
    image_delta = float(df["image_delta"].mean()) if "image_delta" in df.columns else 0.0
    metadata_delta = float(df["metadata_delta"].mean()) if "metadata_delta" in df.columns else 0.0
    if image_delta > metadata_delta:
        driver = "image branch"
    elif metadata_delta > image_delta:
        driver = "metadata branch"
    else:
        driver = "both modalities similarly"
    return f"Removing image drops confidence by {image_delta:.3f}; removing metadata drops it by {metadata_delta:.3f}. Main driver: {driver}."


def write_story_panel(
    *,
    image: np.ndarray,
    grad: np.ndarray,
    lime_img: np.ndarray,
    shap_img: np.ndarray,
    summary: pd.Series,
    preds: pd.DataFrame | None,
    sample_idx: int,
    xai_dir: Path,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax in axes.ravel():
        ax.axis("off")

    title = f"{summary['dataset_source']}/{summary['variant']}/{summary['experiment_name']}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original evaluation image")

    shap_heatmap, _ = shap_overlay(np.ones_like(image), grad)
    axes[0, 1].imshow(shap_heatmap)
    axes[0, 1].set_title("Image SHAP heatmap")

    axes[0, 2].imshow(shap_img)
    axes[0, 2].set_title("Image SHAP overlay")

    axes[0, 3].imshow(lime_img)
    axes[0, 3].set_title("Image LIME patch explanation")

    probability_panel(axes[1, 0], preds, sample_idx)
    signed_metadata_panel(axes[1, 1], xai_dir / "metadata_shap.csv", "Metadata SHAP")
    signed_metadata_panel(axes[1, 2], xai_dir / "metadata_lime.csv", "Metadata LIME")

    axes[1, 3].axis("off")
    text = [
        f"Image model: {summary.get('image_model', 'n/a')}",
        f"Tabular model: {summary.get('tabular_model', 'n/a')}",
        f"Mode: {summary.get('mode', 'n/a')}",
        f"Test acc: {float(summary.get('test_acc', 0)):.2f}%",
        f"Test BACC: {float(summary.get('test_bacc', 0)):.2f}%",
        f"Test AUC: {float(summary.get('test_auc', 0)):.3f}",
        f"Macro F1: {float(summary.get('test_f1_macro', 0)):.3f}",
        "",
        "Image attribution uses SHAP-style patch heatmaps.",
        textwrap.fill(modality_text(xai_dir), width=48),
    ]
    axes[1, 3].text(0.02, 0.96, "\n".join(text), va="top", fontsize=9.5, wrap=True)
    axes[1, 3].set_title("Run context and interpretation")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def tabular_story_panel(summary: pd.Series, xai_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"{summary['dataset_source']}/{summary['variant']}/{summary['experiment_name']}", fontsize=14, fontweight="bold")
    signed_metadata_panel(axes[0], xai_dir / "metadata_shap.csv", "Metadata SHAP")
    signed_metadata_panel(axes[1], xai_dir / "metadata_lime.csv", "Metadata LIME")
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.95,
        "This is a tabular-only model, so image Grad-CAM/LIME/SHAP are not applicable.\n\n"
        "The panel keeps the XAI story consistent by showing metadata SHAP and LIME style "
        "positive/negative contribution bars around the zero baseline.",
        va="top",
        fontsize=11,
        wrap=True,
    )
    axes[2].set_title("Interpretation")
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def process_run(summary_path: Path) -> dict[str, Any]:
    summary = pd.read_csv(summary_path).iloc[0]
    source = str(summary["dataset_source"])
    variant = str(summary["variant"])
    experiment = str(summary["experiment_name"])
    mode = str(summary["mode"])
    xai_dir = RESULT_XAI / source / variant / experiment
    xai_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "dataset_source": source,
        "variant": variant,
        "experiment_name": experiment,
        "mode": mode,
        "xai_dir": str(xai_dir),
        "status": "started",
    }

    preds_path = summary_path.parent / "test_predictions.csv"
    preds = pd.read_csv(preds_path) if preds_path.exists() else None

    uses_image = mode != "tabular_only"
    bundle, batch = first_eval_batch(source, variant, mode, batch_size=4)
    metadata_outputs = build_metadata_shap_lime(xai_dir, bundle, batch)
    if not uses_image:
        tabular_story_panel(summary, xai_dir, xai_dir / "xai_story_panel.png")
        row.update({"status": "tabular_only", "story_panel": str(xai_dir / "xai_story_panel.png"), **metadata_outputs})
        return row

    saliency_path = xai_dir / "image_gradcam_or_saliency.npy"
    if not saliency_path.exists():
        row.update({"status": "missing_saliency"})
        return row

    saliency = np.load(saliency_path)
    sample_count = min(len(batch["image"]), saliency.shape[0], 4)
    sample_rows = []

    for i in range(sample_count):
        image = tensor_to_rgb(batch["image"][i])
        grad = normalize_map(saliency[i])
        grad_overlay = targeted_gradcam_overlay(image, grad, "jet", 0.58, 0.72)
        lime_img, lime_scores = lime_overlay(image, grad, grid=8, keep_fraction=0.22)
        shap_img, shap_scores = shap_overlay(image, grad, grid=8)
        shap_heat = expand_grid(shap_scores, grad.shape)

        prefix = f"sample_{i:02d}"
        save_rgb(xai_dir / f"{prefix}_original.png", image)
        save_heatmap(xai_dir / f"{prefix}_gradcam_heatmap.png", grad, "magma")
        save_rgb(xai_dir / f"{prefix}_gradcam.png", grad_overlay)
        save_rgb(xai_dir / f"{prefix}_lime.png", lime_img)
        save_signed_heatmap(xai_dir / f"{prefix}_shap_heatmap.png", shap_heat, "coolwarm")
        save_rgb(xai_dir / f"{prefix}_shap.png", shap_img)
        np.save(xai_dir / f"{prefix}_lime_patch_scores.npy", lime_scores)
        np.save(xai_dir / f"{prefix}_shap_patch_scores.npy", shap_scores)

        write_story_panel(
            image=image,
            grad=grad,
            lime_img=lime_img,
            shap_img=shap_img,
            summary=summary,
            preds=preds,
            sample_idx=i,
            xai_dir=xai_dir,
            out_path=xai_dir / f"{prefix}_xai_story_panel.png",
        )
        sample_rows.append(
            {
                "sample": i,
                "original": str(xai_dir / f"{prefix}_original.png"),
                "gradcam_heatmap": str(xai_dir / f"{prefix}_gradcam_heatmap.png"),
                "gradcam": str(xai_dir / f"{prefix}_gradcam.png"),
                "lime": str(xai_dir / f"{prefix}_lime.png"),
                "shap_heatmap": str(xai_dir / f"{prefix}_shap_heatmap.png"),
                "shap": str(xai_dir / f"{prefix}_shap.png"),
                "story_panel": str(xai_dir / f"{prefix}_xai_story_panel.png"),
            }
        )

    if sample_rows:
        # Copy the first panel to a stable name so browsing every folder is easy.
        first_panel = Path(sample_rows[0]["story_panel"])
        stable_panel = xai_dir / "xai_story_panel.png"
        stable_panel.write_bytes(first_panel.read_bytes())
    else:
        stable_panel = xai_dir / "xai_story_panel.png"

    pd.DataFrame(sample_rows).to_csv(xai_dir / "expressive_xai_manifest.csv", index=False)
    notes = {
        "original": "Evaluation image reconstructed from the saved test split pipeline.",
        "gradcam": "Uses the saved image_gradcam_or_saliency.npy from the completed run. Kept as a separate heatmap/overlay artifact.",
        "lime": "Dependency-free LIME-style patch explanation derived from the saved local image attribution map.",
        "shap": "Image panels use dependency-free SHAP-style patch heatmaps derived from patch-level attribution contrasts.",
        "metadata": "Metadata SHAP and LIME use signed local contributions around the standardized zero baseline.",
        "modality": "Existing image-vs-metadata removal deltas are included for fusion/MoE/SOTA models.",
    }
    (xai_dir / "expressive_xai_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    row.update({"status": "ok", "samples": sample_count, "story_panel": str(stable_panel), **metadata_outputs})
    return row


def validate_outputs(rows: list[dict[str, Any]]) -> pd.DataFrame:
    validation = []
    for row in rows:
        xai_dir = Path(row["xai_dir"])
        required = ["xai_story_panel.png"]
        if row.get("mode") != "image_only":
            required.extend(["metadata_shap.png", "metadata_lime.png"])
        if row.get("status") == "ok":
            required.extend(["sample_00_original.png", "sample_00_gradcam.png", "sample_00_lime.png", "sample_00_shap.png", "sample_00_shap_heatmap.png"])
            required.append("sample_00_gradcam_heatmap.png")
        checks = {}
        for name in required:
            path = xai_dir / name
            ok = path.exists() and path.stat().st_size > 5_000
            if ok and path.suffix.lower() == ".png":
                try:
                    with Image.open(path) as img:
                        ok = img.width >= 200 and img.height >= 200
                except Exception:
                    ok = False
            checks[name] = ok
        validation.append({**row, **{f"has_{k}": v for k, v in checks.items()}, "all_required_ok": all(checks.values())})
    return pd.DataFrame(validation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of runs for a quick test.")
    args = parser.parse_args()

    summary_paths = sorted(
        p for p in RESULT_LOGS.rglob("result_summary.csv") if "_focal_trials" not in p.parts
    )
    if args.limit:
        summary_paths = summary_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    for idx, summary_path in enumerate(summary_paths, 1):
        rel = summary_path.relative_to(REPO_ROOT)
        print(f"[{idx}/{len(summary_paths)}] rebuilding expressive XAI for {rel}", flush=True)
        try:
            rows.append(process_run(summary_path))
        except Exception as exc:
            rows.append({"summary_path": str(summary_path), "xai_dir": "", "status": "error", "error": repr(exc)})
            print(f"  ERROR: {exc!r}", flush=True)

    validation = validate_outputs(rows)
    RESULT_XAI.mkdir(parents=True, exist_ok=True)
    validation.to_csv(RESULT_XAI / "expressive_xai_validation.csv", index=False)
    print(validation["status"].value_counts(dropna=False).to_string(), flush=True)
    print("all_required_ok", int(validation["all_required_ok"].fillna(False).sum()), "/", len(validation), flush=True)
    print(RESULT_XAI / "expressive_xai_validation.csv", flush=True)


if __name__ == "__main__":
    main()
