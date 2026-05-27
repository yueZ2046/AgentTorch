"""Command-line entry point for Shenzhen urban vitality prediction."""

import argparse

from .train import save_predictions, train_model


def main():
    parser = argparse.ArgumentParser(description="Train Shenzhen urban vitality predictor.")
    parser.add_argument("--data-dir", default="data_shenzhen")
    parser.add_argument("--output", default="outputs/shenzhen_vitality_predictions.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    runner, dataset, metrics, _ = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
    )
    output_path = save_predictions(runner, dataset, args.output)
    print(f"blocks={dataset.num_blocks} features={dataset.num_features} output={output_path}")
    print(
        "train_mae={:.3f} train_rmse={:.3f} validation_mae={:.3f} validation_rmse={:.3f}".format(
            metrics["train"]["mae"],
            metrics["train"]["rmse"],
            metrics["validation"]["mae"],
            metrics["validation"]["rmse"],
        )
    )


if __name__ == "__main__":
    main()
