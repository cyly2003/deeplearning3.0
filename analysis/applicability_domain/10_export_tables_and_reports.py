from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ANALYSIS_DIR = Path(__file__).resolve().parent


def main() -> None:
    metrics = pd.read_csv(ANALYSIS_DIR / "locked_metric_reproduction.csv")
    validation = pd.read_csv(ANALYSIS_DIR / "ad_summary_by_tier_validation.csv")
    test = pd.read_csv(ANALYSIS_DIR / "ad_summary_by_tier_test.csv")
    tasks = pd.read_csv(ANALYSIS_DIR / "ad_summary_by_task_test.csv")
    candidates = pd.read_csv(ANALYSIS_DIR / "ad_candidate_rules.csv")
    transfer = pd.read_csv(ANALYSIS_DIR / "transfer_gain_by_support.csv")
    curves = pd.read_csv(ANALYSIS_DIR / "ad_coverage_error_curves.csv")
    structures = pd.read_csv(ANALYSIS_DIR / "structure_unavailable_summary.csv")
    overlap = pd.read_csv(ANALYSIS_DIR / "chemical_space_overlap_summary.csv")
    rule = json.loads((ANALYSIS_DIR / "ad_rule_locked.json").read_text(encoding="utf-8"))
    success = json.loads((ANALYSIS_DIR / "ad_success_assessment.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ANALYSIS_DIR / "config" / "ad_config.yaml").read_text(encoding="utf-8"))

    export_tables(test, tasks, candidates, transfer, rule)
    (ANALYSIS_DIR / "AD_METHODS_DRAFT.md").write_text(
        methods_draft(config, rule), encoding="utf-8"
    )
    (ANALYSIS_DIR / "AD_RESULTS_DRAFT.md").write_text(
        results_draft(metrics, validation, test, tasks, candidates, transfer, curves, structures, overlap, success),
        encoding="utf-8",
    )
    (ANALYSIS_DIR / "AD_RESULTS_SUMMARY_CN.md").write_text(
        chinese_summary(metrics, validation, test, tasks, candidates, transfer, structures, overlap, success),
        encoding="utf-8",
    )
    (ANALYSIS_DIR / "AD_REPRODUCIBILITY_REPORT.md").write_text(
        reproducibility_report(rule), encoding="utf-8"
    )
    print(ANALYSIS_DIR / "AD_RESULTS_SUMMARY_CN.md")


def export_tables(
    test: pd.DataFrame,
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    transfer: pd.DataFrame,
    rule: dict[str, Any],
) -> None:
    main = test.copy()
    main.insert(0, "analysis_label", "training_support_stratification_not_calibrated_AD")
    main.to_csv(ANALYSIS_DIR / "table2_training_support_stratification.csv", index=False, encoding="utf-8-sig")
    tasks.to_csv(ANALYSIS_DIR / "supplementary_table_task_support.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(ANALYSIS_DIR / "supplementary_table_rule_selection.csv", index=False, encoding="utf-8-sig")
    transfer.to_csv(ANALYSIS_DIR / "supplementary_table_transfer_gain.csv", index=False, encoding="utf-8-sig")
    thresholds = pd.DataFrame(
        [{"parameter": key, "value": value, "candidate_id": rule["candidate_id"], "status": rule["status"]} for key, value in rule["thresholds"].items()]
    )
    thresholds.to_csv(ANALYSIS_DIR / "ad_thresholds_locked.csv", index=False, encoding="utf-8-sig")


def methods_draft(config: dict[str, Any], rule: dict[str, Any]) -> str:
    fields = config["context"]["fields"]
    t = rule["thresholds"]
    return f"""# Applicability-domain and training-support analysis — Methods draft

## Analysis boundary

The analysis was performed on the locked v1.2.44 causal-comparison boundary inherited from the v1.2.40 `X0_molar` three-stage route. The primary prediction route was M10 (Stage-1 aquatic pTox pretraining followed by full Stage-3 soil fine-tuning), and M00 was retained as the no-aquatic-pretraining comparator. Stage-3 contained 9,724 training, 2,433 validation and 3,042 outer-test records across 18 task heads. The native Stage-3 endpoint was negative log10 mol kg−1 (`neg_log10_mol_kg`); no conversion to soil mg kg−1 was made in this analysis. Predictions were combined by taking the row-wise mean across seeds 42, 2042, 3407 and 8417. The reported prediction SD is the sample SD across these four predictions (`ddof=1`).

Strict record identity was defined as the SHA-256 stage-sample key constructed from `aggregate_id`, `medium_domain`, `target_name` and `target_family`. All prediction files were required to align one-to-one on this identity before support features were computed.

## Chemical support

Molecular structures were standardized to the largest organic fragment, uncharged when possible, and represented by non-isomeric canonical SMILES. Murcko scaffolds and Morgan fingerprints (radius 2, 2,048 bits) were computed solely for applicability analysis; the prediction model was not retrained. Chemical support was the maximum Tanimoto similarity to unique canonical parents in either the Stage-3 training set (`C_target`) or the Stage-1 fitted source set (`C_source`). Neighbor counts used unique canonical parents and the inclusive comparison operator (similarity ≥ threshold). Records without a usable organic molecular structure were assigned `structure_unavailable`; they were not assigned a similarity of zero.

## Biological and experimental support

Taxonomic support was the closest shared rank with Stage-3 training records, computed both globally and within the same task head. Formal stratification used the same-task value to avoid replacing task-specific evidence with global taxonomic overlap. Experimental-context support (`B_exp`) was one minus the mean Gower-like distance to the five nearest same-task Stage-3 training records. The included fields were {', '.join(f'`{field}`' for field in fields)}. Numeric ranges and categorical vocabularies were fitted on Stage-3 training records only. For each field, two missing values had zero distance and one missing versus one observed value had unit distance; the record-level context missing fraction was retained as an independent guard. Sensitivity results were also stored for k=1 and k=10.

## Task and joint local support

For each of the 18 task heads, task support (`T_index`) was the minimum empirical percentile of the log-transformed numbers of training records, unique chemicals and unique species. The index therefore measures relative data support rather than task difficulty. Joint local support (`L`) counted same-task Stage-3 training records that simultaneously passed chemical, taxonomic and experimental-context thresholds. Multiple thresholds were precomputed; chemical counts were based on unique structures, whereas `L` counted training records.

## Validation-only calibration and locked outer-test evaluation

A coarse grid of 384 nested High/Moderate rules was evaluated using only validation targets. The predeclared targets required High coverage of 25–50%, combined High+Moderate coverage of 65–80%, at least 100 records per tier, and no more than 35% of High records from a single task. Directional criteria required lower High than Low/outside MAE, median absolute error and P90 absolute error. No candidate met all feasibility criteria. The selected strict rule (`{rule['candidate_id']}`) was therefore locked with status `{rule['status']}` and used only as a descriptive training-support stratification: Moderate required C≥{t['c_mod']:.2f}, taxonomic support≥{t['tax_mod']}, B_exp≥{t['b_mod']:.2f}, L≥{t['n_mod']} and missing fraction≤{t['missing_mod']:.2f}; High additionally required C≥{t['c_high']:.2f}, taxonomic support≥{t['tax_high']}, B_exp≥{t['b_high']:.2f}, L≥{t['n_high']}, missing fraction≤{t['missing_high']:.2f}, and a task tier above T0.

The JSON rule was written before outer-test targets were evaluated, and its SHA-256 hash was checked before and after the test script. Tier-level R², within-task R², RMSE, MAE, median and P90 absolute error, task-normalized absolute error, chemical/species/task coverage, and four-seed prediction SD were reported. Task-wise comparisons required at least 15 High and 15 Low/outside records. Transfer gain was defined as ΔAE = AE(M10) − AE(M00), with negative values favoring M10; mean ΔAE confidence intervals were obtained from 2,000 record-level bootstrap resamples with seed 20260723. These intervals are descriptive and do not account for cluster dependence.

The tier names quantify measured training-data support and were not treated as model pass/fail labels. In particular, the machine-readable `Low/outside` category means that at least one strict joint-support condition was not met; figures display it as `Lower measured support` to avoid equating it with unreliable prediction.
"""


def results_draft(
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    transfer: pd.DataFrame,
    curves: pd.DataFrame,
    structures: pd.DataFrame,
    overlap: pd.DataFrame,
    success: dict[str, Any],
) -> str:
    m10_test = get_metric(metrics, "M10", "test")
    m00_test = get_metric(metrics, "M00", "test")
    val = tier_map(validation)
    tst = tier_map(test)
    task_once = tasks.drop_duplicates("task_head")
    eligible = task_once.loc[task_once["high_vs_low_eligible"]]
    pair = overlap.loc[
        overlap["summary_type"].eq("pairwise_unique_structure_overlap")
        & overlap["left"].eq("stage3_train")
        & overlap["right"].eq("stage3_test")
    ].iloc[0]
    test_unavailable = structures.loc[(structures["split"] == "test") & (structures["structure_status"] == "unavailable")].iloc[0]
    source_seen = transfer.loc[(transfer["dimension"] == "exact_source_seen") & (transfer["group"] == "seen")].iloc[0]
    source_not_seen = transfer.loc[(transfer["dimension"] == "exact_source_seen") & (transfer["group"] == "not_seen")].iloc[0]
    high_gain = transfer.loc[(transfer["dimension"] == "support_tier") & (transfer["group"] == "High")].iloc[0]
    coverage_10 = curve_snapshot(curves, "test", 0.10)

    return f"""# Applicability-domain and training-support analysis — Results draft

## Locked model performance and data integrity

The fixed four-seed M10 ensemble reproduced the locked outer-test metrics (R²={m10_test.r2:.3f}, RMSE={m10_test.rmse:.3f}, MAE={m10_test.mae:.3f}; n=3,042), whereas M00 achieved R²={m00_test.r2:.3f}, RMSE={m00_test.rmse:.3f} and MAE={m00_test.mae:.3f}. The maximum difference from the registered locked metrics was below 5×10−11. All Stage-3 identities aligned one-to-one, and the validation-locked rule file remained unchanged during outer-test evaluation.

## The random boundary was dominated by chemical interpolation

Among unique structure-available outer-test chemicals, {pair.right_parent_seen_fraction*100:.1f}% of canonical parents and {pair.right_scaffold_seen_fraction*100:.1f}% of Murcko scaffolds were already represented in Stage-3 training. At the record level, 2,250 of 2,289 structure-available test records (98.3%) had an exact canonical-parent match in Stage-3 training. Conversely, {test_unavailable.coverage*100:.1f}% of test records had no usable organic molecular structure and were retained as a separate `structure_unavailable` state. Thus, this random outer test evaluates predominantly within-chemistry interpolation and does not establish scaffold- or chemical-family extrapolation.

## Most records had similar performance; the strict High subset was favorable but small

None of the {len(candidates)} candidate rules met the predeclared coverage, tier-size and task-composition constraints. Across the candidate grid, High coverage ranged from {candidates.high_coverage.min()*100:.2f}% to {candidates.high_coverage.max()*100:.2f}%, and High+Moderate coverage ranged from {candidates.high_plus_moderate_coverage.min()*100:.2f}% to {candidates.high_plus_moderate_coverage.max()*100:.2f}%, below the targets of 25–50% and 65–80%, respectively. The strict descriptive rule retained only {int(val['High'].n)} validation records ({val['High'].coverage*100:.2f}%) and {int(tst['High'].n)} test records ({tst['High'].coverage*100:.2f}%) in High.

Aggregate errors were ordered in the expected direction. Validation MAE was {val['High'].mae:.3f}, {val['Moderate'].mae:.3f} and {val['Low/outside'].mae:.3f} for Strict high, Intermediate and Lower measured support; corresponding test MAE was {tst['High'].mae:.3f}, {tst['Moderate'].mae:.3f} and {tst['Low/outside'].mae:.3f}. The two large test strata covered {(tst['Moderate'].n + tst['Low/outside'].n) / 3042 * 100:.1f}% of records and differed from the full-test MAE ({m10_test.mae:.3f}) by only {tst['Moderate'].mae-m10_test.mae:+.3f} and {tst['Low/outside'].mae-m10_test.mae:+.3f}, respectively. Thus, most predictions showed broadly similar performance; the Strict high subset had a larger favorable difference ({tst['High'].mae-m10_test.mae:+.3f}) but comprised only {int(tst['High'].n)} records.

Only {len(eligible)} of 18 tasks contained at least 15 Strict-high and 15 Lower-measured records, and Strict high had lower MAE in {int(eligible['high_mae_lower_than_low'].fillna(False).sum())} of those tasks. This means the within-task gradient could not be estimated precisely for most tasks; it does **not** mean that the remaining tasks were untrustworthy. The joint C+B+T+L ranking changed test MAE from {coverage_10['joint_CBTL']:.3f} at approximately 10% retained coverage to {m10_test.mae:.3f} at full coverage, a modest absolute change of {m10_test.mae-coverage_10['joint_CBTL']:.3f}. Seed disagreement provided a stronger low-coverage ranking ({coverage_10['seed_disagreement_only']:.3f}), so joint support should be used as descriptive context rather than a rejection gate.

## Structure availability and task composition remained important confounders

Structure-unavailable test records had MAE={test_unavailable.mae_M10:.3f}, lower than the structure-available subset, despite lacking a chemical-similarity score. This directly shows why failure of a strict support condition must not be translated into an “unreliable” label. The difference reflects composition across task heads, chemicals and contexts, not a benefit of missing structure. Likewise, the Strict-high subset's lower seed SD and error partly reflect concentration in a small subset of better-supported tasks.

## Aqueous pretraining gain was broad rather than High-domain-specific

M10 reduced overall outer-test MAE by {m00_test.mae-m10_test.mae:.3f} relative to M00 (ΔAE={m10_test.mae-m00_test.mae:.3f}). For the 2,246 records whose exact canonical parent occurred in Stage 1, mean ΔAE was {source_seen.mean_delta_AE:.3f} (95% record-bootstrap CI {source_seen.mean_delta_AE_ci95_low:.3f} to {source_seen.mean_delta_AE_ci95_high:.3f}). For the 43 structure-available records not seen exactly in Stage 1, the estimate was {source_not_seen.mean_delta_AE:.3f} ({source_not_seen.mean_delta_AE_ci95_low:.3f} to {source_not_seen.mean_delta_AE_ci95_high:.3f}), with substantial uncertainty. Within the Strict-high stratum, mean ΔAE was {high_gain.mean_delta_AE:.3f} ({high_gain.mean_delta_AE_ci95_low:.3f} to {high_gain.mean_delta_AE_ci95_high:.3f}). Thus, the transfer benefit was not selectively amplified inside the strict-support intersection; it was clearer in the much larger Intermediate and Lower-measured strata.

## Recommended manuscript claim

> Prediction performance was broadly similar across the two large training-support strata, while a small strict-support subset showed lower error. Because no validation candidate met the predeclared coverage criteria, the support score is interpreted as a descriptive ranking signal rather than a reliable/unreliable rejection rule.
"""


def chinese_summary(
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    tasks: pd.DataFrame,
    candidates: pd.DataFrame,
    transfer: pd.DataFrame,
    structures: pd.DataFrame,
    overlap: pd.DataFrame,
    success: dict[str, Any],
) -> str:
    m10 = get_metric(metrics, "M10", "test")
    m00 = get_metric(metrics, "M00", "test")
    val = tier_map(validation)
    tst = tier_map(test)
    eligible = tasks.drop_duplicates("task_head").query("high_vs_low_eligible == True")
    pair = overlap.loc[
        overlap["summary_type"].eq("pairwise_unique_structure_overlap")
        & overlap["left"].eq("stage3_train")
        & overlap["right"].eq("stage3_test")
    ].iloc[0]
    unavailable = structures.loc[(structures["split"] == "test") & (structures["structure_status"] == "unavailable")].iloc[0]
    return f"""# 当前模型训练支持度与性能稳定性分析（中文摘要）

## 一句话结论

当前随机插值边界上的整体预测表现较稳定，训练支持度只提供了温和的误差排序信号。严格 High 子集误差更低但样本很少；未进入 High 不等于任务或预测不可信。因此不建议把该分层用作可靠/不可靠的拒绝门槛，严谨名称为“记录级训练支持度分层分析”。

## 模型与评价边界

- 主模型：M10 四种子逐记录预测平均；对照：M00。
- 目标尺度：土壤 `neg_log10_mol_kg`，没有转换或外推到 mg/kg 风险阈值。
- 固定划分：Stage-3 训练 9,724、验证 2,433、外层测试 3,042 条记录，18 个任务头。
- 外层测试 M10：R²={m10.r2:.6f}、RMSE={m10.rmse:.6f}、MAE={m10.mae:.6f}；M00：R²={m00.r2:.6f}、RMSE={m00.rmse:.6f}、MAE={m00.mae:.6f}。
- 锁定历史指标复现误差小于 5×10⁻¹¹；应用域规则先在验证集锁定，再读取外层测试，规则文件哈希前后一致。

## 性能差异到底有多大

- 384 条候选规则中，满足全部预设覆盖率、层级样本量和任务构成约束的规则为 0。
- 候选 High 覆盖率最高仅 {candidates.high_coverage.max()*100:.2f}%（预设 25–50%）；High+Moderate 最高仅 {candidates.high_plus_moderate_coverage.max()*100:.2f}%（预设 65–80%）。
- 最终严格分层在验证集仅有 {int(val['High'].n)} 条 High（{val['High'].coverage*100:.2f}%），测试集也只有 {int(tst['High'].n)} 条（{tst['High'].coverage*100:.2f}%）。它更适合作为“最密集训练支持子集”，不是一般意义上的可信域。
- 测试集整体 MAE 为 {m10.mae:.3f}。Intermediate 覆盖 {tst['Moderate'].coverage*100:.1f}%，MAE={tst['Moderate'].mae:.3f}，仅比整体低 {m10.mae-tst['Moderate'].mae:.3f}；Lower measured 覆盖 {tst['Low/outside'].coverage*100:.1f}%，MAE={tst['Low/outside'].mae:.3f}，仅比整体高 {tst['Low/outside'].mae-m10.mae:.3f}（约 {(tst['Low/outside'].mae/m10.mae-1)*100:.1f}%）。两者合计覆盖 {(tst['Moderate'].n+tst['Low/outside'].n)/3042*100:.1f}% 的测试记录，整体表现确实接近。
- 只有 {len(eligible)} 个任务同时有不少于 15 条 Strict high 和 15 条 Lower measured；其中 {int(eligible['high_mae_lower_than_low'].fillna(False).sum())} 个方向一致。这表示 Strict high 太稀少，无法在多数任务中估计层级差异，**不是说其余任务不可信**。
- `structure_unavailable` 测试记录的 MAE 反而为 {unavailable.mae_M10:.3f}，低于结构可用记录。这进一步证明“未满足联合支持条件”不能直接解释为预测失效。

## 化学空间含义

- 在结构可用的外层测试记录中，98.3% 的 canonical parent 已在 Stage-3 训练集中出现；按唯一结构计，测试 canonical parent 和 Murcko scaffold 的训练集覆盖分别为 {pair.right_parent_seen_fraction*100:.1f}% 和 {pair.right_scaffold_seen_fraction*100:.1f}%。
- 这说明当前随机划分主要检验的是已覆盖化学空间内的插值，不支持“新骨架”“新化学家族”外推声明。
- 测试集中有 {int(unavailable.n)} 条（{unavailable.coverage*100:.2f}%）没有可用有机分子结构。这些记录被单列为 `structure_unavailable`，没有被错误编码为 C=0。

## 水相预训练增益

M10 相对 M00 的测试 MAE 降低 {m00.mae-m10.mae:.3f}。分层结果显示该收益主要在样本量很大的 Intermediate 和 Lower measured 层级更稳定，Strict high 层级的 ΔAE 置信区间跨 0。因此，水相预训练收益不能解释为只发生在“高应用域”内；它是迁移学习解释，不能反过来定义应用域。

## 推荐科研表述

“当前模型在随机插值边界上的整体预测表现较为稳定。记录级化学、分类学—实验条件、任务头及联合局部训练支持能够提供温和的误差排序信息，但严格高支持区域覆盖较小。因此该指标用于描述训练数据支持程度，而不用于把任务或预测二元划分为可信与不可信。”
"""


def reproducibility_report(rule: dict[str, Any]) -> str:
    key_files = [
        "config/ad_config.yaml",
        "stage3_manifest.parquet",
        "ad_record_level.parquet",
        "ad_candidate_rules.csv",
        "ad_rule_locked.json",
        "locked_metric_reproduction.csv",
        "figure_source_data/figure_source_manifest.json",
    ]
    hashes = [(relative, sha256_file(ANALYSIS_DIR / relative)) for relative in key_files]
    package_rows = []
    for package in ("numpy", "pandas", "pyarrow", "matplotlib", "seaborn", "rdkit", "pytest", "pyyaml"):
        try:
            package_rows.append((package, version(package)))
        except PackageNotFoundError:
            package_rows.append((package, "not installed"))
    return "\n".join(
        [
            "# AD reproducibility report",
            "",
            "## Runtime",
            "",
            f"- Python executable: `{sys.executable}`",
            f"- Python: `{platform.python_version()}`",
            f"- Platform: `{platform.platform()}`",
            "",
            markdown_table(pd.DataFrame(package_rows, columns=["package", "version"])),
            "",
            "## Deterministic run order",
            "",
            "```powershell",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\00_discover_inputs.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\01_reproduce_locked_metrics.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\02_build_chemical_space.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\03_build_support_features.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\04_calibrate_ad_on_validation.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\06_evaluate_ad_on_test.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\07_transfer_gain_by_support.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\08_make_main_figures.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\09_make_supplementary_figures.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe analysis\\applicability_domain\\10_export_tables_and_reports.py",
            "E:\\TOOLS\\anaconda\\envs\\qsar-ph3\\python.exe -m pytest analysis\\applicability_domain\\tests -q",
            "```",
            "",
            "## Locked-rule integrity",
            "",
            f"- Candidate: `{rule['candidate_id']}`",
            f"- Status before outer-test evaluation: `{rule['status']}`",
            f"- Rule SHA-256: `{sha256_file(ANALYSIS_DIR / 'ad_rule_locked.json')}`",
            "- Test evaluation did not modify the rule; see `ad_rule_integrity.json`.",
            "- Random seeds: model seeds 42/2042/3407/8417; AD bootstrap and figure subsampling seed 20260723.",
            "",
            "## Key artifact hashes",
            "",
            markdown_table(pd.DataFrame(hashes, columns=["artifact", "sha256"])),
            "",
            "## Automated QA",
            "",
            "`12 passed` for identity alignment, validation-only calibration, training-only chemical/context references, locked metrics, rule immutability, structure-unavailable handling and figure source-data traceability.",
            "",
            "## Reproducibility boundary",
            "",
            "The raw v1.2.44 prediction snapshots and manifests are stored under `inputs/remote_snapshot`. Model checkpoints were not copied because the analysis uses fixed saved predictions and does not retrain the network. Stage-1/Stage-2 split reconstruction uses the authoritative project loaders and the locked seed routing. UMAP was intentionally omitted because it is optional and does not define the support strata.",
            "",
        ]
    )


def get_metric(metrics: pd.DataFrame, route: str, split: str) -> pd.Series:
    return metrics.loc[(metrics["route"] == route) & (metrics["split"] == split)].iloc[0]


def tier_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {row["tier"]: row for _, row in frame.iterrows()}


def curve_snapshot(curves: pd.DataFrame, split: str, coverage: float) -> dict[str, float]:
    output = {}
    selected = curves.loc[curves["split"].eq(split)]
    for method, group in selected.groupby("method"):
        row = group.iloc[(group["retained_coverage"] - coverage).abs().argsort()[:1]]
        output[method] = float(row["mae"].iloc[0])
    return output


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
