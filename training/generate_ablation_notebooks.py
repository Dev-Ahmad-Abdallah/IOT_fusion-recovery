from __future__ import annotations

import nbformat as nbf

from ablation_utils import ABLATIONS, REPO_ROOT


NOTEBOOK_DIR = REPO_ROOT / "training"


def title_for(model_name: str, input_mode: str) -> str:
    return f"{model_name.upper()} {input_mode.replace('_', ' ').title()} Ablation"


def make_notebook(model_name: str, input_mode: str, experiment_name: str) -> nbf.NotebookNode:
    epochs = 60 if model_name == "cnn" else 25
    batch_size = 64 if model_name in {"resnet50", "cnn"} else 256
    lr = 3e-4 if model_name == "cnn" else (1e-4 if model_name == "resnet50" else 5e-4)
    model_note = (
        "This notebook uses torchvision's pretrained ResNet-50 (`IMAGENET1K_V2`) with an adapted classification head. "
        if model_name == "resnet50"
        else ""
    )
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# {title_for(model_name, input_mode)}\n\n"
            f"Experiment: `{experiment_name}`\n\n"
            f"{model_note}"
            "This notebook runs full GPU training on a stratified 80/10/10 train/validation/test split "
            "built from the extracted processed tabular data and image tensors. "
            "Image runs use tensor-safe train augmentation and class-weighted loss. "
            "CNN runs use tempered class-weighted loss, stronger paper-style augmentation, and early stopping to reduce overfitting; balanced sampling can be enabled manually, but is off by default because it over-corrected on this dataset. "
            "Fusion modes combine the tabular metadata vector with the image tensor. "
            "It prints per-epoch train/validation loss, accuracy, macro AUC, and macro F1 with tqdm progress bars. "
            "Epoch logs are written under `Results_Logs/<experiment>/`, figures under `Results_Figures/<experiment>/`, and XAI artifacts under `Results_XAI/<experiment>/`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, sys\n"
            "REPO_ROOT = Path('/home/zeus/content/IOT_fusion')\n"
            "os.chdir(REPO_ROOT)\n"
            "sys.path.insert(0, str(REPO_ROOT / 'training'))\n"
            "from ablation_utils import get_data_info, run_training\n\n"
            "info = get_data_info()\n"
            "print('classes:', info.class_to_idx)\n"
            "print('metadata_dim:', info.metadata_dim)\n"
        ),
        nbf.v4.new_markdown_cell("## Full GPU Training\nAdjust `epochs` and `batch_size` if needed."),
        nbf.v4.new_code_cell(
            f"full_result = run_training(\n"
            f"    model_name='{model_name}',\n"
            f"    input_mode='{input_mode}',\n"
            f"    experiment_name='{experiment_name}',\n"
            f"    epochs={epochs},\n"
            f"    batch_size={batch_size},\n"
            f"    lr={lr},\n"
            f"    num_workers=2,\n"
            f"    device_name='cuda',\n"
            f"    wandb_enabled=True,\n"
            f"    xai_enabled=True,\n"
            f"    augment_train=True,\n"
            f"    class_weighted_loss=True,\n"
            f"    class_weight_power=0.5,\n"
            f"    class_balanced_sampler=False,\n"
            f"    early_stopping_patience=12,\n"
            f"    min_epochs=15,\n"
            f")\n"
            f"full_result\n"
        ),
        nbf.v4.new_markdown_cell("## Training Outputs"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            f"output_dir = Path('outputs/ablations')\n"
            f"logs_dir = Path('Results_Logs') / '{experiment_name}'\n"
            f"figures_dir = Path('Results_Figures') / '{experiment_name}'\n"
            f"xai_dir = Path('Results_XAI') / '{experiment_name}'\n"
            f"print('metrics:', logs_dir / 'epoch_metrics.csv')\n"
            f"print('epoch log:', logs_dir / 'epoch_log.txt')\n"
            f"print('result:', logs_dir / 'result_summary.csv')\n"
            f"print('checkpoint:', output_dir / '{experiment_name}_best.pth')\n"
            f"print('classification report:', logs_dir / 'classification_report.csv')\n"
            f"print('confusion matrix CSV:', logs_dir / 'confusion_matrix.csv')\n"
            f"print('loss curve:', figures_dir / 'loss_curve.png')\n"
            f"print('roc curve:', figures_dir / 'roc_curve.png')\n"
            f"sorted(str(p) for p in list(logs_dir.glob('*')) + list(figures_dir.glob('*')) + list(xai_dir.glob('*')))\n"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, input_mode, experiment_name in ABLATIONS:
        path = NOTEBOOK_DIR / f"{experiment_name}.ipynb"
        nbf.write(make_notebook(model_name, input_mode, experiment_name), path)
        print(path)


if __name__ == "__main__":
    main()
