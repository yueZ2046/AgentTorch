"""Training and export utilities for the Shenzhen urban vitality model."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import create_runner


def _masked_metrics(prediction, target, mask) -> Dict:
    """Compute MAE, RMSE and per-tier breakdown for a split mask."""
    if not bool(mask.any()):
        return {"mae": float("nan"), "rmse": float("nan"),
                "median_ae": float("nan"), "corr": float("nan"),
                "mae_by_tier": {}}
    pred_m = prediction[mask]    # (N_split, 48)
    tgt_m  = target[mask]

    # Flatten over time slots for aggregate metrics
    error = pred_m - tgt_m
    abs_err = error.abs()

    # Correlation between predicted and observed (flattened)
    p_flat = pred_m.reshape(-1)
    t_flat = tgt_m.reshape(-1)
    if p_flat.std() > 1e-6 and t_flat.std() > 1e-6:
        corr = float(torch.corrcoef(torch.stack([p_flat, t_flat]))[0, 1].item())
    else:
        corr = float("nan")

    # Per-vitality-tier MAE: divide observed by mean into quartile tiers
    mean_obs = tgt_m.mean(dim=1)  # (N_split,) — average vitality per block
    q1, q2, q3 = mean_obs.quantile(torch.tensor([0.25, 0.5, 0.75])).tolist()
    tier_mae = {}
    for tier_label, tier_mask in [
        ("low",    mean_obs <= q1),
        ("medium", (mean_obs > q1) & (mean_obs <= q2)),
        ("high",   (mean_obs > q2) & (mean_obs <= q3)),
        ("top",    mean_obs > q3),
    ]:
        if tier_mask.any():
            tier_mae[tier_label] = float(abs_err[tier_mask].mean().item())
        else:
            tier_mae[tier_label] = float("nan")

    # Per-time-slot MAE (averaged over blocks in split)
    slot_mae = abs_err.mean(dim=0).tolist()  # list of 48 floats

    return {
        "mae":        float(abs_err.mean().item()),
        "rmse":       float(error.square().mean().sqrt().item()),
        "median_ae":  float(abs_err.median().item()),
        "corr":       corr,
        "mae_by_tier": tier_mae,
        "slot_mae":    slot_mae,
    }


def run_baselines(dataset) -> Dict:
    """Train Ridge, GBT, and MLP baselines; return per-model validation MAE.

    Each baseline receives the same normalized feature matrix as the agent model
    and predicts all 48 time-slot vitality values.  Results are on raw (un-logged)
    vitality scale so they are directly comparable to the agent model's MAE.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor

    X_tr = dataset.features[dataset.train_mask].numpy()
    X_va = dataset.features[dataset.validation_mask].numpy()
    y_tr = dataset.vitality[dataset.train_mask].numpy()      # raw LBS counts
    y_va = dataset.vitality[dataset.validation_mask].numpy()

    results: Dict = {}

    naive_pred = y_tr.mean(axis=0, keepdims=True)
    results["naive_mean"] = {
        "val_mae": float(np.abs(y_va - naive_pred).mean()),
        "note": "training-set mean per time slot",
    }

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    results["ridge"] = {
        "val_mae": float(np.abs(y_va - ridge.predict(X_va)).mean()),
        "note": "Ridge(alpha=1)",
    }

    gbt = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=200, max_depth=4, random_state=42),
        n_jobs=-1,
    )
    gbt.fit(X_tr, y_tr)
    results["gradient_boosting"] = {
        "val_mae": float(np.abs(y_va - gbt.predict(X_va)).mean()),
        "note": "HistGBT(max_iter=200, depth=4) × 48 slots",
    }

    mlp = MLPRegressor(hidden_layer_sizes=(128,), max_iter=500, random_state=42,
                       early_stopping=True, n_iter_no_change=20)
    mlp.fit(X_tr, y_tr)
    results["mlp_no_agent"] = {
        "val_mae": float(np.abs(y_va - mlp.predict(X_va)).mean()),
        "note": "MLP(128) — same features, no agent aggregation",
    }

    return results


def diagnose_errors(runner, dataset, top_n: int = 20) -> pd.DataFrame:
    """Return top_n highest-error validation blocks with spatial diagnostics."""
    prediction = runner.state["environment"]["predicted_vitality"].detach().cpu().numpy()
    observed   = runner.state["environment"]["observed_vitality"].detach().cpu().numpy()

    val_idx   = dataset.validation_mask.numpy()
    block_ids = dataset.block_ids[dataset.validation_mask].numpy()
    pred_val  = prediction[val_idx]
    obs_val   = observed[val_idx]

    per_block_mae  = np.abs(pred_val - obs_val).mean(axis=1)
    mean_obs       = obs_val.mean(axis=1)
    bias           = (pred_val - obs_val).mean(axis=1)   # + = over-predict

    df = pd.DataFrame({
        "block_id":     block_ids,
        "mean_observed": mean_obs.round(1),
        "mae":          per_block_mae.round(1),
        "bias":         bias.round(1),
        "mae_pct":      (per_block_mae / (mean_obs + 1e-6) * 100).round(1),
    })

    if dataset.districts is not None:
        df["district"] = dataset.districts[val_idx]

    return df.nlargest(top_n, "mae").reset_index(drop=True)


def train_model(
    data_dir="data_shenzhen",
    epochs: int = 400,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
    cosine_lr: bool = False,
    early_stop_patience: int = 60,
    split_strategy: str = "random",
    holdout_district=None,
):
    """Fit the agent-based vitality model through an AgentTorch runner.

    Args:
        cosine_lr: If True, apply CosineAnnealingLR. Default False: constant LR
                   converges better within the standard 400-epoch budget.
        early_stop_patience: Stop training if val MAE has not improved for this
                             many epochs. Set to 0 to disable.
    """
    torch.manual_seed(seed)
    runner, dataset = create_runner(
        data_dir=data_dir,
        hidden_dim=hidden_dim,
        validation_fraction=validation_fraction,
        seed=seed,
        device=device,
        split_strategy=split_strategy,
        holdout_district=holdout_district,
    )
    runner.to(runner.initializer.device)
    dev = runner.initializer.device
    train_mask = dataset.train_mask.to(dev)
    move_policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    param_groups = [
        {"params": move_policy.home_logits, "lr": learning_rate * 5},
        {"params": move_policy.attract_net.parameters(), "lr": learning_rate,
         "weight_decay": 0.1},
        # scale_net (no bias) adds per-block temporal deviation on top of log_scale.
        # weight_decay=0.1 keeps block deviations small; 2× LR matches log_scale pace.
        {"params": move_policy.scale_net.parameters(), "lr": learning_rate * 2,
         "weight_decay": 0.1},
    ]
    if move_policy.spatial_attn is not None:
        param_groups.append(
            {"params": move_policy.spatial_attn.parameters(), "lr": learning_rate,
             "weight_decay": 1e-4}
        )
    aggregate = runner.initializer.transition_function["0"]["aggregate_vitality"]
    param_groups.append(
        {"params": aggregate.log_scale, "lr": learning_rate * 2}
    )
    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        if cosine_lr else None
    )
    # Huber loss on standardised log-scale: behaves like MAE for large residuals
    # and MSE near zero, directly aligning training with the evaluation metric.
    loss_fn = nn.HuberLoss(delta=1.0)

    history = []
    best_val_mae = float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0
    val_mask_dev = dataset.validation_mask.to(dev)

    for epoch in range(epochs):
        runner.reset_state()
        optimizer.zero_grad()
        runner.step(1)
        predicted = runner.state["environment"]["predicted_vitality_scaled"]
        observed  = runner.state["environment"]["observed_vitality_scaled"]
        loss = loss_fn(predicted[train_mask], observed[train_mask])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        history.append(float(loss.detach().cpu().item()))

        # Early stopping: track validation MAE every epoch (cheap — no grad)
        if early_stop_patience > 0 and dataset.validation_mask.any():
            with torch.no_grad():
                pred_raw = runner.state["environment"]["predicted_vitality"].detach().cpu()
                obs_raw  = runner.state["environment"]["observed_vitality"].detach().cpu()
            current_val_mae = float(
                (pred_raw[dataset.validation_mask] - obs_raw[dataset.validation_mask])
                .abs().mean().item()
            )
            if current_val_mae < best_val_mae - 1.0:
                best_val_mae = current_val_mae
                # Save lightweight param snapshot (not full runner state)
                import copy
                best_state = copy.deepcopy({
                    k: v.detach().clone()
                    for k, v in runner.named_parameters()
                })
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break

    # Restore best weights if early stopping fired
    if best_state is not None:
        for name, param in runner.named_parameters():
            if name in best_state:
                param.data.copy_(best_state[name])

    runner.reset_state()
    with torch.no_grad():
        runner.step(1)
    prediction = runner.state["environment"]["predicted_vitality"].detach().cpu()
    observed   = runner.state["environment"]["observed_vitality"].detach().cpu()
    metrics = {
        "training_loss": history[-1] if history else float("nan"),
        "num_features":  dataset.num_features,
        "num_blocks":    dataset.num_blocks,
        "train":      _masked_metrics(prediction, observed, dataset.train_mask),
        "validation": _masked_metrics(prediction, observed, dataset.validation_mask),
    }
    return runner, dataset, metrics, history


def train_multi_seed(
    seeds: List[int] = (42, 123, 456),
    data_dir: str = "data_shenzhen",
    epochs: int = 400,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    validation_fraction: float = 0.2,
    device: str = "auto",
    cosine_lr: bool = False,
    early_stop_patience: int = 60,
    split_strategy: str = "random",
    holdout_district=None,
) -> Dict:
    """Run train_model for multiple seeds and report aggregate statistics.

    Returns a dict with per-seed metrics and summary statistics
    (mean ± std) over seeds for val MAE and val RMSE.
    """
    per_seed: List[Dict] = []
    for seed in seeds:
        print(f"\n[seed={seed}]")
        _, _, metrics, _ = train_model(
            data_dir=data_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            validation_fraction=validation_fraction,
            seed=seed,
            device=device,
            cosine_lr=cosine_lr,
            early_stop_patience=early_stop_patience,
            split_strategy=split_strategy,
            holdout_district=holdout_district,
        )
        per_seed.append({"seed": seed, **metrics})
        print(
            f"  val MAE={metrics['validation']['mae']:.0f}  "
            f"val RMSE={metrics['validation']['rmse']:.0f}  "
            f"corr={metrics['validation']['corr']:.3f}"
        )

    val_maes  = [r["validation"]["mae"]  for r in per_seed if not np.isnan(r["validation"]["mae"])]
    val_rmses = [r["validation"]["rmse"] for r in per_seed if not np.isnan(r["validation"]["rmse"])]
    val_corrs = [r["validation"]["corr"] for r in per_seed if not np.isnan(r["validation"]["corr"])]

    summary = {
        "val_mae_mean":  float(np.mean(val_maes)),
        "val_mae_std":   float(np.std(val_maes)),
        "val_mae_best":  float(np.min(val_maes)),
        "val_mae_worst": float(np.max(val_maes)),
        "val_rmse_mean": float(np.mean(val_rmses)),
        "val_corr_mean": float(np.mean(val_corrs)),
    }
    print(
        f"\n[multi-seed summary] "
        f"val MAE = {summary['val_mae_mean']:.0f} ± {summary['val_mae_std']:.0f}  "
        f"(best={summary['val_mae_best']:.0f}, worst={summary['val_mae_worst']:.0f})  "
        f"corr={summary['val_corr_mean']:.3f}"
    )
    return {"per_seed": per_seed, "summary": summary}


def prediction_frame(runner, dataset):
    predicted = runner.state["environment"]["predicted_vitality"].detach().cpu().numpy()
    observed  = runner.state["environment"]["observed_vitality"].detach().cpu().numpy()
    result = pd.DataFrame({"Block_ID": dataset.block_ids.numpy()})
    result["split"] = dataset.validation_mask.numpy().astype("int8")
    result["observed_weekday_vitality"]  = observed[:, :24].mean(axis=1)
    result["predicted_weekday_vitality"] = predicted[:, :24].mean(axis=1)
    result["observed_weekend_vitality"]  = observed[:, 24:].mean(axis=1)
    result["predicted_weekend_vitality"] = predicted[:, 24:].mean(axis=1)
    for index, name in enumerate(dataset.target_names):
        result[f"predicted_{name}"] = predicted[:, index]
    return result


def save_predictions(runner, dataset, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(runner, dataset).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
