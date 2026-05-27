"""Training and export utilities for the Shenzhen urban vitality model."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from . import create_runner


def _masked_metrics(prediction, target, mask):
    if not bool(mask.any()):
        return {"mae": float("nan"), "rmse": float("nan")}
    error = prediction[mask] - target[mask]
    return {
        "mae":  float(error.abs().mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
    }


def train_model(
    data_dir="data_shenzhen",
    epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_dim: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    device: str = "auto",
):
    """Fit the agent-based vitality model through an AgentTorch runner."""
    torch.manual_seed(seed)
    runner, dataset = create_runner(
        data_dir=data_dir,
        hidden_dim=hidden_dim,
        validation_fraction=validation_fraction,
        seed=seed,
        device=device,
    )
    runner.to(runner.initializer.device)
    dev = runner.initializer.device
    train_mask = dataset.train_mask.to(dev)
    move_policy = runner.initializer.policy_function["0"]["residents"]["move_policy"]
    param_groups = [
        {"params": move_policy.home_logits, "lr": learning_rate * 5},
        # Strong regularization on attract_net: global softmax creates zero-sum
        # competition — spatial_attn handles local context instead.
        {"params": move_policy.attract_net.parameters(), "lr": learning_rate,
         "weight_decay": 0.1},
    ]
    if move_policy.spatial_attn is not None:
        param_groups.append(
            {"params": move_policy.spatial_attn.parameters(), "lr": learning_rate,
             "weight_decay": 1e-4}
        )
    aggregate = runner.initializer.transition_function["0"]["aggregate_vitality"]
    param_groups.append(
        # log_scale bridges population-unit predictions to LBS sampling scale;
        # give it 2× LR since it adjusts a simple multiplicative offset.
        {"params": aggregate.log_scale, "lr": learning_rate * 2}
    )
    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        runner.reset_state()
        optimizer.zero_grad()
        runner.step(1)
        predicted = runner.state["environment"]["predicted_vitality_scaled"]
        observed  = runner.state["environment"]["observed_vitality_scaled"]
        loss = loss_fn(predicted[train_mask], observed[train_mask])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu().item()))

    runner.reset_state()
    with torch.no_grad():
        runner.step(1)
    prediction = runner.state["environment"]["predicted_vitality"].detach().cpu()
    observed   = runner.state["environment"]["observed_vitality"].detach().cpu()
    metrics = {
        "training_loss": history[-1] if history else float("nan"),
        "train":      _masked_metrics(prediction, observed, dataset.train_mask),
        "validation": _masked_metrics(prediction, observed, dataset.validation_mask),
    }
    return runner, dataset, metrics, history


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
