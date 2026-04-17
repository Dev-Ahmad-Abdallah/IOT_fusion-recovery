# IOT Fusion Ablation Training

This folder contains full GPU training notebooks and utilities for the processed IOT Fusion dataset. Runs use a stratified 80/10/10 train/validation/test split built from the extracted processed CSVs and tensor files.

## Environment

Use the Lightning `cloudspace` Python environment:

```bash
/system/conda/miniconda3/envs/cloudspace/bin/python
```

W&B settings are loaded from:

```bash
/home/zeus/content/IOT_fusion/W&B.env
```

## Notebooks

Core requested ablations:

- `resnet50_img_only.ipynb` - pretrained torchvision ResNet-50 image-only baseline with train augmentation and class-weighted loss
- `cnn_img_only.ipynb` - scratch-trained ResNet-50-style CNN from `code/CNN/CNN.py`; this is not ImageNet-pretrained
- `cnn_meta_only.ipynb`
- `cnn_inter_fusion.ipynb`
- `cnn_late_fusion.ipynb`
- `transformer_img_only.ipynb`
- `transformer_meta_only.ipynb`
- `transformer_inter_fusion.ipynb`
- `transformer_late_fusion.ipynb`

Extra early-fusion ablations:

- `cnn_early_fusion.ipynb`
- `transformer_early_fusion.ipynb`

Fusion notebooks combine the tabular processed metadata vector with the image tensor. Early fusion applies metadata-conditioned image modulation before the image model. Intermediate fusion combines image logits/features with metadata features in a fusion head. Late fusion averages image and metadata logits.

The scratch CNN image branch uses residual bottleneck blocks with squeeze-excitation, stochastic depth, classifier dropout, label smoothing, tempered class-weighted loss, AdamW weight decay, tensor-safe augmentation, and early stopping. Balanced sampling is available from the CLI with `--balanced_sampler`, but it is off by default because it over-corrected toward minority classes during testing.

## Manual Run Steps

1. Open one notebook in `training/`.
2. Select the `cloudspace` Python kernel if Jupyter asks.
3. Run the setup cell.
4. Run the full GPU training cell.
5. Watch the tqdm progress bars and per-epoch printout for train/validation loss, accuracy, macro AUC, and macro F1.
6. Review W&B and the local outputs.

## Terminal Run

Run one ablation from the terminal:

```bash
cd /home/zeus/content/IOT_fusion
/system/conda/miniconda3/envs/cloudspace/bin/python training/run_one_ablation.py --experiment_name cnn_img_only --epochs 60 --batch_size 64 --lr 3e-4
```

Use `--class_weight_power 1.0` for full inverse-frequency class weights, `--class_weight_power 0.5` for the default tempered weights, or `--no_class_weights` to train without class weights.

Pretrained ResNet-50 image-only run:

```bash
/system/conda/miniconda3/envs/cloudspace/bin/python training/run_one_ablation.py --experiment_name resnet50_img_only --epochs 25 --batch_size 64 --lr 1e-4
```

## Outputs

- Per-run logs folder: `Results_Logs/<experiment>/`
- Per-epoch text log: `Results_Logs/<experiment>/epoch_log.txt`
- Per-epoch metrics: `Results_Logs/<experiment>/epoch_metrics.csv`
- Final summary: `Results_Logs/<experiment>/result_summary.csv`
- Split summary: `Results_Logs/<experiment>/split_summary.csv`
- Best checkpoint: `outputs/ablations/<experiment>_best.pth`
- Classification report: `Results_Logs/<experiment>/classification_report.csv`
- Confusion matrix CSV: `Results_Logs/<experiment>/confusion_matrix.csv`
- Test predictions/probabilities: `Results_Logs/<experiment>/test_predictions.csv`
- Figures folder: `Results_Figures/<experiment>/`
- Loss curve: `Results_Figures/<experiment>/loss_curve.png`
- Epoch metric curves: `Results_Figures/<experiment>/epoch_metrics.png`
- ROC curves: `Results_Figures/<experiment>/roc_curve.png`
- Confusion matrix plot: `Results_Figures/<experiment>/confusion_matrix.png`
- XAI folder: `Results_XAI/<experiment>/`
- XAI image saliency arrays: `Results_XAI/<experiment>/<experiment>_image_saliency.npy`
- XAI metadata importance CSVs: `Results_XAI/<experiment>/<experiment>_metadata_xai_top20.csv`

Compatibility copies are also mirrored under `outputs/ablations/`.

## Post-hoc XAI

Run all post-hoc explanations after the ablation checkpoints exist:

```bash
cd /home/zeus/content/IOT_fusion
/system/conda/miniconda3/envs/cloudspace/bin/jupyter nbconvert --to notebook --execute training/xai_all_ablation_analysis.ipynb --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
```

This writes Grad-CAM image explanations, LIME-style image explanations, KernelSHAP-style metadata explanations, LIME-style metadata explanations, fusion modality contribution reports, and pairwise model comparisons under `Results_XAI/`.
