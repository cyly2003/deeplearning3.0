from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


RUNS = [
    {
        "label": "baseline_30p20f_cosine",
        "display": "v1.0.0 预训练30+微调20 cosine",
        "kind": "baseline",
    },
    {
        "label": "finetune40_cosine",
        "display": "v1.1.0 预训练30+微调40 cosine",
        "kind": "diagnostic",
    },
    {
        "label": "finetune20_constant",
        "display": "v1.1.0 预训练30+微调20 恒定微调LR",
        "kind": "diagnostic",
    },
    {
        "label": "pretrain60_cosine",
        "display": "v1.1.0 预训练60+微调20 cosine",
        "kind": "diagnostic",
    },
]

REPRESENTATIVE_TASKS = ("ECx_Mortality", "NOEC_Growth", "LOEC_Growth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize transfer step-length diagnostics.")
    parser.add_argument(
        "--diagnostics-root",
        default="outputs/experiments/v1_1_0_transfer_step_diagnostics",
        help="Root containing explanations/<run_label> outputs.",
    )
    parser.add_argument(
        "--ascii-root",
        default="outputs/experiments/v1_1_0_transfer_step_diagnostics_ascii",
        help="ASCII mirror containing diagnostic run metrics/history.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        default=(
            "outputs/experiments/v1_0_0_same_budget/"
            "v1.0.0_训练优化重构_同预算水相预训练土壤pTox小样本迁移_f20/"
            "deep/full/M_qc_aquatic_to_soil_ptox_adapt_C_f20"
        ),
        help="Baseline transfer run directory containing history.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/v1_1_0_transfer_step_diagnostics/summary",
        help="Directory for summary CSV files.",
    )
    parser.add_argument(
        "--report",
        default="docs/v1.1.0_transfer_step_diagnostics.md",
        help="Markdown report path.",
    )
    parser.add_argument("--top-n-shap", type=int, default=8, help="Top SHAP features per representative task.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    diagnostics_root = Path(args.diagnostics_root)
    ascii_root = Path(args.ascii_root)
    baseline_run_dir = Path(args.baseline_run_dir)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    family = collect_family_metrics(diagnostics_root)
    convergence = collect_convergence(ascii_root, baseline_run_dir)
    selected = collect_selected_tasks(diagnostics_root)
    shap_top = collect_shap_top(diagnostics_root, args.top_n_shap)
    run_summary = summarize_runs(family)

    family.to_csv(out_dir / "summary_endpoint_family_metrics.csv", index=False, encoding="utf-8-sig")
    run_summary.to_csv(out_dir / "summary_run_level_family_metrics.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(out_dir / "summary_training_convergence.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "summary_selected_explain_tasks.csv", index=False, encoding="utf-8-sig")
    shap_top.to_csv(out_dir / "summary_shap_top_features.csv", index=False, encoding="utf-8-sig")

    report_path.write_text(
        build_report(family, run_summary, convergence, selected, shap_top),
        encoding="utf-8",
    )

    print(f"family={out_dir / 'summary_endpoint_family_metrics.csv'}")
    print(f"run_summary={out_dir / 'summary_run_level_family_metrics.csv'}")
    print(f"convergence={out_dir / 'summary_training_convergence.csv'}")
    print(f"selected={out_dir / 'summary_selected_explain_tasks.csv'}")
    print(f"shap_top={out_dir / 'summary_shap_top_features.csv'}")
    print(f"report={report_path}")


def collect_family_metrics(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run in RUNS:
        path = root / "explanations" / run["label"] / "endpoint_family_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "run_label", run["label"])
        frame.insert(1, "run_display", run["display"])
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    baseline = result[result["run_label"] == "baseline_30p20f_cosine"][
        ["endpoint_family", "r2", "rmse", "mae", "abs_error_p90"]
    ].rename(
        columns={
            "r2": "baseline_r2",
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "abs_error_p90": "baseline_abs_error_p90",
        }
    )
    result = result.merge(baseline, on="endpoint_family", how="left")
    result["delta_r2_vs_baseline"] = result["r2"] - result["baseline_r2"]
    result["delta_mae_vs_baseline"] = result["mae"] - result["baseline_mae"]
    result["delta_rmse_vs_baseline"] = result["rmse"] - result["baseline_rmse"]
    result["delta_p90_abs_error_vs_baseline"] = result["abs_error_p90"] - result["baseline_abs_error_p90"]
    return result


def collect_convergence(ascii_root: Path, baseline_run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        if run["kind"] == "baseline":
            path = baseline_run_dir / "history.csv"
        else:
            path = ascii_root / run["label"] / "history.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        history = pd.read_csv(path)
        for phase, frame in history.groupby("phase", sort=False):
            ordered = frame.sort_values("epoch")
            min_idx = ordered["mean_loss"].idxmin()
            min_row = ordered.loc[min_idx]
            first = ordered.iloc[0]
            last = ordered.iloc[-1]
            rows.append(
                {
                    "run_label": run["label"],
                    "run_display": run["display"],
                    "phase": phase,
                    "epochs": int(len(ordered)),
                    "start_loss": float(first["mean_loss"]),
                    "final_loss": float(last["mean_loss"]),
                    "min_loss": float(min_row["mean_loss"]),
                    "min_phase_epoch": int(min_row["epoch"]),
                    "min_global_epoch": int(min_row.get("global_epoch", min_row["epoch"])),
                    "start_learning_rate": float(first.get("learning_rate", float("nan"))),
                    "final_learning_rate": float(last.get("learning_rate", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def collect_selected_tasks(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run in RUNS:
        path = root / "explanations" / run["label"] / "selected_explain_tasks.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "run_label", run["label"])
            frame.insert(1, "run_display", run["display"])
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_shap_top(root: Path, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        for task in REPRESENTATIVE_TASKS:
            path = root / "explanations" / run["label"] / "shap" / task / "shap_feature_importance.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path).head(top_n)
            for rank, row in enumerate(frame.itertuples(index=False), start=1):
                rows.append(
                    {
                        "run_label": run["label"],
                        "run_display": run["display"],
                        "task_head": task,
                        "rank": rank,
                        "feature": row.feature,
                        "mean_abs_shap": float(row.mean_abs_shap),
                    }
                )
    return pd.DataFrame(rows)


def summarize_runs(family: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_label, frame in family.groupby("run_label", sort=False):
        total_n = frame["n"].sum()
        rows.append(
            {
                "run_label": run_label,
                "run_display": frame["run_display"].iloc[0],
                "endpoint_families": ";".join(frame["endpoint_family"]),
                "total_n": int(total_n),
                "family_r2_n_weighted_mean": weighted_mean(frame["r2"], frame["n"]),
                "pooled_family_rmse": float(((frame["rmse"] ** 2 * frame["n"]).sum() / total_n) ** 0.5),
                "family_mae_n_weighted_mean": weighted_mean(frame["mae"], frame["n"]),
                "family_abs_error_p90_n_weighted_mean": weighted_mean(frame["abs_error_p90"], frame["n"]),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result[result["run_label"] == "baseline_30p20f_cosine"].iloc[0]
    result["delta_weighted_family_r2_vs_baseline"] = (
        result["family_r2_n_weighted_mean"] - baseline["family_r2_n_weighted_mean"]
    )
    result["delta_weighted_family_mae_vs_baseline"] = (
        result["family_mae_n_weighted_mean"] - baseline["family_mae_n_weighted_mean"]
    )
    result["delta_pooled_family_rmse_vs_baseline"] = result["pooled_family_rmse"] - baseline["pooled_family_rmse"]
    return result


def build_report(
    family: pd.DataFrame,
    run_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    selected: pd.DataFrame,
    shap_top: pd.DataFrame,
) -> str:
    best_by_mae = run_summary.sort_values("family_mae_n_weighted_mean").iloc[0]
    baseline = run_summary[run_summary["run_label"] == "baseline_30p20f_cosine"].iloc[0]
    lines = [
        "# v1.1.0 迁移步长诊断与主终点解释",
        "",
        "## 实验目的",
        "",
        "本轮实验用于判断当前水相全量预训练到土壤小样本迁移任务的性能瓶颈，是否来自训练步长不足或学习率衰减过快；同时对 ECx、NOEC、LOEC 三类主终点做测试集误差分布和代表任务 SHAP 解释。",
        "",
        "## 实验设置",
        "",
        "- 基线：v1.0.0，同预算水相预训练 30 轮，土壤微调 20 轮，cosine 学习率调度。",
        "- 诊断 1：预训练 30 轮，土壤微调延长至 40 轮，cosine 学习率调度。",
        "- 诊断 2：预训练 30 轮，土壤微调 20 轮，但微调阶段不使用 cosine 衰减，保持恒定学习率。",
        "- 诊断 3：水相预训练延长至 60 轮，土壤微调 20 轮，cosine 学习率调度。",
        "- 主终点解释：每组实验均对 ECx、NOEC、LOEC 在测试集上输出分组误差，并对 ECx_Mortality、NOEC_Growth、LOEC_Growth 做 SHAP。",
        "",
        "## 核心结论",
        "",
        f"- 训练 loss 明确显示微调 20 轮 cosine 没有充分走到低损失区间；同样 20 轮微调，只把微调调度改为恒定学习率后，最终微调 loss 大幅下降。",
        f"- 但测试集主终点误差不随训练 loss 单调改善。按 ECx/NOEC/LOEC 的样本加权 MAE 看，当前最优是 **{best_by_mae['run_display']}**，MAE={best_by_mae['family_mae_n_weighted_mean']:.4f}；基线 MAE={baseline['family_mae_n_weighted_mean']:.4f}。",
        "- 这说明当前问题不是单纯“训练还没迭代够”。更准确的判断是：微调阶段确实存在有效步长不足，但更长/更强的拟合会带来土壤小样本过拟合风险，需要用验证集早停或更稳健的微调策略控制。",
        "- ECx 的测试表现相对更稳，LOEC 与 NOEC 的误差尾部更重，符合 NOEC/LOEC 本身受实验设计、暴露时间、物种和观测效应定义影响更大的特点。",
        "",
        "## 主终点测试集表现",
        "",
        markdown_table(
            family[
                [
                    "run_display",
                    "endpoint_family",
                    "n",
                    "r2",
                    "rmse",
                    "mae",
                    "bias_mean",
                    "abs_error_p90",
                    "delta_mae_vs_baseline",
                ]
            ],
            float_digits=4,
        ),
        "",
        "## 三类主终点合并摘要",
        "",
        "说明：这里的 R2 是按三类 endpoint family 的样本量加权均值，不等同于重新合并所有预测点计算的 pooled R2；RMSE 和 MAE 可按样本量聚合解释。",
        "",
        markdown_table(
            run_summary[
                [
                    "run_display",
                    "total_n",
                    "family_r2_n_weighted_mean",
                    "pooled_family_rmse",
                    "family_mae_n_weighted_mean",
                    "delta_weighted_family_mae_vs_baseline",
                ]
            ],
            float_digits=4,
        ),
        "",
        "## 训练步长诊断",
        "",
        markdown_table(
            convergence[
                [
                    "run_display",
                    "phase",
                    "epochs",
                    "start_loss",
                    "final_loss",
                    "min_loss",
                    "min_phase_epoch",
                    "start_learning_rate",
                    "final_learning_rate",
                ]
            ],
            float_digits=6,
        ),
        "",
        "## 代表任务 SHAP Top 特征",
        "",
        "SHAP 是在原始 pTox 尺度的模型输出上计算，代表任务按测试集样本量选择。特征名中的 morgan_fingerprint_512bit_group 表示分子指纹整体分组贡献，并不表示单一结构片段。",
        "",
        markdown_table(
            shap_top[["run_display", "task_head", "rank", "feature", "mean_abs_shap"]],
            float_digits=4,
        ),
        "",
        "## 输出文件",
        "",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/summary/summary_endpoint_family_metrics.csv`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/summary/summary_run_level_family_metrics.csv`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/summary/summary_training_convergence.csv`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/summary/summary_selected_explain_tasks.csv`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/summary/summary_shap_top_features.csv`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/explanations/<run_label>/abs_error_by_endpoint_family.png`",
        "- `outputs/experiments/v1_1_0_transfer_step_diagnostics/explanations/<run_label>/residual_distribution_by_endpoint_family.png`",
        "",
        "## 后续建议",
        "",
        "1. 下一轮不建议简单把 epoch 继续拉长，而是优先比较微调阶段恒定 LR、小 cosine 最小 LR 下限、ReduceLROnPlateau、验证集早停四种策略。",
        "2. 对 NOEC/LOEC 应单独检查暴露时间、物种、效应类别和浓度单位来源，误差尾部很可能来自实验条件异质性而不是分子结构信息不足。",
        "3. 若用于论文叙述，应把外推能力表述为应用域扩大后的稳健迁移能力，而不是强调 scaffold 外推本身。",
        "",
    ]
    if not selected.empty:
        lines.extend(
            [
                "## 被选中的解释任务",
                "",
                markdown_table(selected, float_digits=4),
                "",
            ]
        )
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, *, float_digits: int) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


if __name__ == "__main__":
    main()
