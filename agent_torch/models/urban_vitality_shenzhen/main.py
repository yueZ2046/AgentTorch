"""Command-line entry point for Shenzhen urban vitality prediction."""

import argparse
import json

from .train import (
    diagnose_errors,
    run_baselines,
    save_predictions,
    train_model,
    train_multi_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Train Shenzhen urban vitality predictor.")
    parser.add_argument("--data-dir", default="data_shenzhen")
    parser.add_argument("--output", default="outputs/shenzhen_vitality_predictions.csv")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-cosine-lr", action="store_true",
                        help="Disable cosine LR schedule (use constant LR).")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Run multi-seed validation over these seeds (e.g. --seeds 42 123 456). "
             "When set, --seed is ignored and no CSV output is written.",
    )
    # Phase 1: spatial split
    parser.add_argument(
        "--split-strategy", choices=["random", "district"], default="random",
        help="Validation split strategy. 'district' holds out one admin district entirely.",
    )
    parser.add_argument(
        "--holdout-district", default=None,
        help="District name to hold out when --split-strategy=district. "
             "Options: 福田区 南山区 罗湖区 宝安区 龙岗区 龙华区 光明区 盐田区 坪山区 大鹏新区",
    )
    # Phase 1: baseline comparison
    parser.add_argument(
        "--baselines", action="store_true",
        help="Run Ridge / GBT / MLP baselines after agent model training.",
    )
    # Phase 1: error diagnosis
    parser.add_argument(
        "--diagnose", type=int, default=0, metavar="N",
        help="Print top-N highest-error validation blocks after training (0 = off).",
    )
    args = parser.parse_args()

    cosine_lr = not args.no_cosine_lr

    if args.seeds:
        result = train_multi_seed(
            seeds=args.seeds,
            data_dir=args.data_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            validation_fraction=args.validation_fraction,
            device=args.device,
            cosine_lr=cosine_lr,
            split_strategy=args.split_strategy,
            holdout_district=args.holdout_district,
        )
        print("\n=== Multi-Seed Results ===")
        print(json.dumps(result["summary"], indent=2))
        return

    runner, dataset, metrics, history = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        cosine_lr=cosine_lr,
        split_strategy=args.split_strategy,
        holdout_district=args.holdout_district,
    )

    v = metrics["validation"]
    t = metrics["train"]
    split_desc = (
        f"district={args.holdout_district}" if args.split_strategy == "district"
        else f"random_seed={args.seed}"
    )
    print(f"\nblocks={dataset.num_blocks}  features={dataset.num_features}  "
          f"split={split_desc}  val_n={dataset.validation_mask.sum().item()}")
    print("train:  mae={mae:.0f}  rmse={rmse:.0f}  corr={corr:.3f}".format(**t))
    print(
        "val:    mae={mae:.0f}  rmse={rmse:.0f}  corr={corr:.3f}  "
        "median_ae={median_ae:.0f}".format(**v)
    )
    if v.get("mae_by_tier"):
        tier_str = "  ".join(f"{k}={vv:.0f}" for k, vv in v["mae_by_tier"].items())
        print(f"val by tier:  {tier_str}")
    print(f"final_loss={metrics['training_loss']:.4f}")

    if args.baselines:
        print("\n=== Baseline Comparison (val MAE, raw LBS scale) ===")
        baseline_results = run_baselines(dataset)
        rows = [("model", "val_mae", "note")]
        rows += [(name, f"{r['val_mae']:.0f}", r.get("note", ""))
                 for name, r in baseline_results.items()]
        rows.append(("agent_torch", f"{v['mae']:.0f}", "agent-based differentiable sim"))
        col_w = [max(len(r[i]) for r in rows) for i in range(3)]
        for row in rows:
            print("  ".join(s.ljust(w) for s, w in zip(row, col_w)))

    if args.diagnose > 0:
        print(f"\n=== Top-{args.diagnose} Highest-Error Validation Blocks ===")
        df = diagnose_errors(runner, dataset, top_n=args.diagnose)
        print(df.to_string(index=False))

    output_path = save_predictions(runner, dataset, args.output)
    print(f"\npredictions → {output_path}")


if __name__ == "__main__":
    main()
