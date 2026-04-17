from __future__ import annotations

import nbformat as nbf

from ablation_utils import REPO_ROOT


def main() -> None:
    notebook_path = REPO_ROOT / "training" / "xai_all_ablation_analysis.ipynb"
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Post-hoc XAI for All Ablations\n\n"
            "This notebook loads the best checkpoint from every completed ablation and runs dependency-free "
            "Grad-CAM, KernelSHAP-style metadata attribution, and LIME-style image/metadata explanations. "
            "Outputs are written under `Results_XAI/<experiment>/` and cross-model comparisons under "
            "`Results_XAI/Comparisons/`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os, sys\n"
            "REPO_ROOT = Path('/home/zeus/content/IOT_fusion')\n"
            "os.chdir(REPO_ROOT)\n"
            "sys.path.insert(0, str(REPO_ROOT / 'training'))\n"
            "from xai_posthoc_analysis import run_posthoc_xai, RESULT_XAI_DIR, COMPARISON_DIR\n"
            "print('XAI root:', RESULT_XAI_DIR)\n"
            "print('Comparison root:', COMPARISON_DIR)\n"
        ),
        nbf.v4.new_markdown_cell("## Run XAI"),
        nbf.v4.new_code_cell(
            "pairwise_comparison = run_posthoc_xai(\n"
            "    max_samples_per_class=1,\n"
            "    image_lime_masks=48,\n"
            "    metadata_shap_masks=80,\n"
            "    metadata_lime_samples=96,\n"
            "    device_name='cuda',\n"
            ")\n"
            "pairwise_comparison\n"
        ),
        nbf.v4.new_markdown_cell("## Output Index"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "for path in sorted(Path('Results_XAI').glob('*')):\n"
            "    if path.is_dir():\n"
            "        print(path)\n"
            "        for child in sorted(path.glob('*'))[:12]:\n"
            "            print('  ', child.name)\n"
        "\n"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, notebook_path)
    print(notebook_path)


if __name__ == "__main__":
    main()
