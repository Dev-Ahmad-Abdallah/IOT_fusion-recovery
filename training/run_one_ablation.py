from __future__ import annotations

import argparse

from ablation_utils import parse_ablation_name, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_xai", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--min_epochs", type=int, default=15)
    args = parser.parse_args()

    model_name, input_mode, experiment_name = parse_ablation_name(args.experiment_name)
    lr = args.lr
    if lr is None:
        lr = 3e-4 if model_name == "cnn" else (1e-4 if model_name == "resnet50" else 5e-4)
    run_training(
        model_name,
        input_mode,
        experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        num_workers=args.num_workers,
        device_name=args.device,
        wandb_enabled=not args.no_wandb,
        xai_enabled=not args.no_xai,
        augment_train=not args.no_augment,
        class_weighted_loss=not args.no_class_weights,
        class_weight_power=args.class_weight_power,
        class_balanced_sampler=args.balanced_sampler,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
    )


if __name__ == "__main__":
    main()
