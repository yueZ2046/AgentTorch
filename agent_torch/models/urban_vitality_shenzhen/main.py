"""深圳城市活力预测模型的命令行入口。

用法示例：

    # 基础训练（随机划分 20% 验证集）
    python -m agent_torch.models.urban_vitality_shenzhen

    # 指定超参数
    python -m agent_torch.models.urban_vitality_shenzhen \\
        --epochs 600 --hidden-dim 256 --learning-rate 5e-4

    # 与基线模型对比
    python -m agent_torch.models.urban_vitality_shenzhen --baselines

    # 多随机种子验证（报告均值±标准差）
    python -m agent_torch.models.urban_vitality_shenzhen --seeds 42 123 456 789

    # 行政区留出验证（把南山区作为验证集）
    python -m agent_torch.models.urban_vitality_shenzhen \\
        --split-strategy district --holdout-district 南山区

    # 对所有行政区做留出扫描
    python -m agent_torch.models.urban_vitality_shenzhen --district-sweep

    # 特征重要性分析
    python -m agent_torch.models.urban_vitality_shenzhen --explain-groups

    # 方案比选（需要提前准备 JSON 方案文件）
    python -m agent_torch.models.urban_vitality_shenzhen \\
        --scenario-file my_plan.json --scenario-output outputs/scenario_result

    # 查看可用的特征名称和数值范围（用于编写方案 JSON）
    python -m agent_torch.models.urban_vitality_shenzhen --list-features
"""

import argparse
import json

from .train import (
    diagnose_errors,
    explain_feature_groups,
    run_baselines,
    save_predictions,
    train_district_sweep,
    train_model,
    train_multi_seed,
)
from .scenario import ScenarioPlan, list_scheme_features, run_scenario_plan


def main():
    """命令行入口：解析参数并按选定模式运行训练、评估或方案比选。"""
    parser = argparse.ArgumentParser(description="训练深圳城市活力预测模型。")

    # ── 数据与输出路径 ──────────────────────────────────────────────────────
    parser.add_argument("--data-dir", default="data_shenzhen",
                        help="数据目录路径（包含 街坊_数据连接.csv 等文件）")
    parser.add_argument("--output", default="outputs/shenzhen_vitality_predictions.csv",
                        help="预测结果 CSV 的输出路径")

    # ── 训练超参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=400,
                        help="最大训练轮数（早停触发时实际轮数更少）")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam 基础学习率（各参数组有独立倍率）")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="attract_net 隐藏层维度")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="随机划分时验证集占比（0.2 = 20%）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（控制数据划分和参数初始化）")
    parser.add_argument("--device", default="auto",
                        help='"auto" 自动选 GPU/CPU，或手动指定 "cpu" / "cuda"')
    parser.add_argument("--no-cosine-lr", action="store_true",
                        help="禁用余弦退火学习率调度（默认使用恒定 LR，收敛更稳定）")

    # ── 多随机种子验证 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="多种子验证的种子列表（如 --seeds 42 123 456），"
             "启用时 --seed 被忽略，不输出 CSV 预测结果。",
    )

    # ── 验证集划分策略 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--split-strategy", choices=["random", "district"], default="random",
        help="验证集划分方式：random=随机，district=行政区留出（空间泛化测试）",
    )
    parser.add_argument(
        "--holdout-district", default=None,
        help="--split-strategy=district 时指定留出的行政区名称，"
             "可选：福田区 南山区 罗湖区 宝安区 龙岗区 龙华区 光明区 盐田区 坪山区 大鹏新区",
    )

    # ── 基线对比 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--baselines", action="store_true",
        help="训练结束后运行 Ridge / GBT / MLP 基线模型并打印对比表",
    )

    # ── 误差诊断 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--diagnose", type=int, default=0, metavar="N",
        help="打印验证集中误差最大的前 N 个街坊的详细信息（0 = 关闭）",
    )
    parser.add_argument(
        "--explain-groups", action="store_true",
        help="运行特征组消融实验：评估各类特征（建筑/POI/OD/人口）对验证 MAE 的贡献",
    )

    # ── 行政区扫描 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--district-sweep", action="store_true",
        help="对深圳 10 个行政区逐一做留出验证（Leave-One-District-Out），完成后退出",
    )
    parser.add_argument(
        "--district-sweep-output", default="outputs/district_sweep.csv", metavar="CSV",
        help="--district-sweep 结果的 CSV 输出路径",
    )

    # ── 方案比选（Phase 4）──────────────────────────────────────────────────
    parser.add_argument(
        "--scenario-file", default=None, metavar="JSON",
        help="方案比选 JSON 文件路径，训练完成后运行方案对比分析",
    )
    parser.add_argument(
        "--scenario-output", default="outputs/scenario", metavar="DIR",
        help="方案比选结果的输出目录（默认：outputs/scenario）",
    )
    parser.add_argument(
        "--no-od-feedback", action="store_true",
        help="禁用方案比选中的 OD 反馈（冻结 OD 特征在当前值，不根据 POI 变化预测新 OD）",
    )

    # ── 特征列表 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--list-features", action="store_true",
        help="打印可在方案 JSON 中使用的建筑/POI 特征名称和城市全局分位数范围，然后退出",
    )

    args = parser.parse_args()

    # ── --list-features：打印特征范围后直接退出 ────────────────────────────
    if args.list_features:
        from .data import load_shenzhen_vitality_data
        ds = load_shenzhen_vitality_data(args.data_dir)
        list_scheme_features(ds)
        return

    cosine_lr = not args.no_cosine_lr

    # ── --district-sweep：对所有行政区做留出扫描后退出 ─────────────────────
    if args.district_sweep:
        df = train_district_sweep(
            data_dir=args.data_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            device=args.device,
            cosine_lr=cosine_lr,
        )
        print("\n=== 行政区留出验证结果 ===")
        print(df.to_string(index=False))
        from pathlib import Path
        out = Path(args.district_sweep_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"结果已保存 → {out}")
        return

    # ── --seeds：多种子验证后退出（不保存预测 CSV）──────────────────────────
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
        print("\n=== 多种子验证汇总 ===")
        print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
        return

    # ── 默认路径：单次训练 ────────────────────────────────────────────────
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

    # ── 打印训练结果摘要 ──────────────────────────────────────────────────
    v = metrics["validation"]
    t = metrics["train"]
    split_desc = (
        f"district={args.holdout_district}" if args.split_strategy == "district"
        else f"random_seed={args.seed}"
    )
    print(f"\n街坊数={dataset.num_blocks}  特征数={dataset.num_features}  "
          f"划分={split_desc}  验证集街坊数={dataset.validation_mask.sum().item()}")
    print("训练集:  mae={mae:.0f}  rmse={rmse:.0f}  corr={corr:.3f}".format(**t))
    print(
        "验证集:  mae={mae:.0f}  rmse={rmse:.0f}  corr={corr:.3f}  "
        "中位绝对误差={median_ae:.0f}".format(**v)
    )

    # 打印排名指标（需要 scipy 计算 Spearman / Kendall）
    rank = v.get("rank", {})
    topk = rank.get("topk_hit_rate", {})
    if rank:
        print(
            "排名:   spearman={spearman:.3f}  kendall={kendall:.3f}  "
            "pairwise={pairwise_accuracy:.3f}  前20%命中={top20pct:.3f}".format(
                spearman=rank.get("spearman", float("nan")),
                kendall=rank.get("kendall", float("nan")),
                pairwise_accuracy=rank.get("pairwise_accuracy", float("nan")),
                top20pct=topk.get("top20pct", float("nan")),
            )
        )

    # 按活力分层的 MAE（low/medium/high/top 各占 25%）
    if v.get("mae_by_tier"):
        tier_str = "  ".join(f"{k}={vv:.0f}" for k, vv in v["mae_by_tier"].items())
        print(f"分层验证MAE:  {tier_str}")
    print(f"最终训练损失={metrics['training_loss']:.4f}")

    # ── --baselines：打印基线对比表 ────────────────────────────────────────
    if args.baselines:
        print("\n=== 基线模型对比（验证集 MAE，原始 LBS 人口数尺度）===")
        baseline_results = run_baselines(dataset)
        rows = [("模型", "val_mae", "备注")]
        rows += [(name, f"{r['val_mae']:.0f}", r.get("note", ""))
                 for name, r in baseline_results.items()]
        rows.append(("agent_torch", f"{v['mae']:.0f}", "可微分 Agent 仿真"))
        col_w = [max(len(r[i]) for r in rows) for i in range(3)]
        for row in rows:
            print("  ".join(s.ljust(w) for s, w in zip(row, col_w)))

    # ── --diagnose N：打印误差最大的 N 个街坊 ─────────────────────────────
    if args.diagnose > 0:
        print(f"\n=== 验证集误差最大的前 {args.diagnose} 个街坊 ===")
        df = diagnose_errors(runner, dataset, top_n=args.diagnose)
        print(df.to_string(index=False))

    # ── --explain-groups：特征组消融实验 ──────────────────────────────────
    if args.explain_groups:
        print("\n=== 特征组消融（验证集 MAE 敏感性）===")
        df = explain_feature_groups(runner, dataset)
        print(df.to_string(index=False))

    # ── --scenario-file：方案比选 ─────────────────────────────────────────
    if args.scenario_file:
        print(f"\n=== Phase 4 方案比选 ===")
        plan = ScenarioPlan.from_json(args.scenario_file)
        result = run_scenario_plan(
            runner, dataset, plan,
            data_dir=args.data_dir,
            od_feedback=not args.no_od_feedback,
        )
        result.print_report(dataset)
        result.to_csv(args.scenario_output)

    # ── 保存预测 CSV（默认最后一步）──────────────────────────────────────
    output_path = save_predictions(runner, dataset, args.output)
    print(f"\n预测结果已保存 → {output_path}")


if __name__ == "__main__":
    main()
