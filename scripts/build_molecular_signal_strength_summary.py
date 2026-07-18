from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUT_DIR = Path("实验汇总") / "分子信号强度探索_20260707"
METRIC_COLUMNS = ("n", "r2", "rmse", "mae", "huber_loss", "task_count")
CURRENT_ABLATIONS = (
    "full",
    "no_context",
    "no_species_lifestage",
    "no_descriptors",
    "no_fingerprint",
    "no_molecular_size_descriptors",
    "no_molecular_residual",
    "descriptors_only",
    "no_source_weighting",
    "no_toxicity_binning",
    "no_censored_loss",
)


@dataclass(frozen=True)
class SourceSpec:
    version: str
    path: Path
    include_status: str
    reason: str
    dataset_boundary: str
    split_policy: str
    seeds: str
    ensemble: str
    interpretation_boundary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build molecular signal strength summary package.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = build_source_manifest()
    metrics = build_metrics_table()
    metrics = add_deltas(metrics)
    coverage = build_coverage_matrix(metrics)
    gaps = build_rerun_gap_matrix(coverage)

    source_manifest.to_csv(out_dir / "source_manifest.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out_dir / "molecular_signal_overall_metrics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(out_dir / "ablation_coverage_matrix.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(out_dir / "rerun_gap_matrix.csv", index=False, encoding="utf-8-sig")
    write_readme(out_dir, metrics, gaps)
    print(f"[molecular-signal] wrote {out_dir}")


def build_source_manifest() -> pd.DataFrame:
    specs = [
        SourceSpec(
            "v1.2.15",
            Path("outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary"),
            "included",
            "current random full baseline",
            "all-current aquatic-to-soil pTox random split",
            "random_8_2;random_5fold",
            "42;1042;2042;3042;4042",
            "5_seed_ensemble plus single-seed rows",
            "random interpolation baseline; not chemical-family extrapolation",
        ),
        SourceSpec(
            "v1.2.21",
            Path("outputs/experiments/v1_2_21_random_split_ablation_remote_summary"),
            "included",
            "current random targeted ablation",
            "all-current aquatic-to-soil pTox random split",
            "random_8_2;random_5fold",
            "42 for 5fold; multiple seeds for random_8_2 where present",
            "single-seed/fold targeted ablation",
            "random interpolation ablation; contains metal/inorganic chemicals",
        ),
        SourceSpec(
            "v1.2.22",
            Path("outputs/experiments/v1_2_22_no_metal_random_split_remote_summary"),
            "included",
            "no-metal/inorganic random full baseline",
            "no-metal/inorganic organic-descriptor applicability boundary",
            "random_8_2;random_5fold",
            "42;1042;2042;3042;4042",
            "5_seed_ensemble plus single-seed rows",
            "organic-representable random interpolation baseline",
        ),
        SourceSpec(
            "v1.2.24",
            Path("outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary"),
            "included",
            "current scaffold-cluster chemical-family full baseline",
            "no-metal/inorganic scaffold/cluster boundary",
            "scaffold_cluster_8_2;scaffold_cluster_5fold",
            "42;1042;2042;3042;4042",
            "5_seed_ensemble plus single-seed rows",
            "current chemical-family extrapolation evidence",
        ),
        SourceSpec(
            "v1.2.26",
            Path("outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary"),
            "included",
            "no-metal/inorganic seed2042 targeted ablation",
            "no-metal/inorganic organic-descriptor applicability boundary",
            "random_8_2;random_5fold",
            "2042",
            "single_seed",
            "random interpolation sensitivity; not scaffold extrapolation",
        ),
        SourceSpec(
            "v1.2.31",
            Path("outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary"),
            "included",
            "molecular size descriptor sensitivity",
            "no-metal/inorganic random plus scaffold-cluster priority matrix",
            "random_8_2;scaffold_cluster_8_2",
            "2042",
            "single_seed",
            "priority sensitivity on molecular weight/size coupling",
        ),
        SourceSpec(
            "v1.2.18",
            Path("outputs/experiments/v1_2_18_mainline_ablation_remote_summary"),
            "excluded_historical",
            "user excluded fixed chemical holdout/CAS-number project from future evidence",
            "legacy CAS-number fixed chemical holdout",
            "fixed_chemical_holdout",
            "42;1042;2042;3042;4042",
            "historical_multi_seed",
            "do not use for current conclusions or coverage",
        ),
    ]
    return pd.DataFrame([spec.__dict__ | {"metric_fields": ";".join(METRIC_COLUMNS)} for spec in specs])


def build_metrics_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(load_full_ensemble_rows("v1.2.15", Path("outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary")))
    rows.extend(load_full_ensemble_rows("v1.2.22", Path("outputs/experiments/v1_2_22_no_metal_random_split_remote_summary")))
    rows.extend(load_full_ensemble_rows("v1.2.24", Path("outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary")))
    rows.extend(load_run_summary_seed_rows("v1.2.15", Path("outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary"), "42"))
    rows.extend(load_run_summary_seed_rows("v1.2.22", Path("outputs/experiments/v1_2_22_no_metal_random_split_remote_summary"), "2042"))
    rows.extend(load_focus_ablation_rows("v1.2.21", Path("outputs/experiments/v1_2_21_random_split_ablation_remote_summary"), "42"))
    rows.extend(load_focus_ablation_rows("v1.2.26", Path("outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary"), "2042"))
    rows.extend(load_v1_2_31_rows(Path("outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary")))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    order = [
        "source_version",
        "source_path",
        "dataset_boundary",
        "split_policy",
        "fold",
        "seed_set",
        "ensemble_type",
        "ablation",
        "prediction_split_part",
        "n",
        "r2",
        "rmse",
        "mae",
        "huber_loss",
        "task_count",
        "baseline_source",
        "delta_mae",
        "delta_r2",
        "interpretation_boundary",
    ]
    for column in order:
        if column not in frame:
            frame[column] = ""
    return frame[order]


def load_full_ensemble_rows(version: str, summary_dir: Path) -> list[dict[str, Any]]:
    path = summary_dir / "split_policy_ensemble_combined_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append(metric_row(
            source_version=version,
            source_path=path,
            dataset_boundary=dataset_boundary(version),
            split_policy=str(row["split_policy"]),
            fold=str(row.get("folds_combined", "")),
            seed_set=str(row.get("seeds", "")),
            ensemble_type="5_seed_ensemble",
            ablation="full",
            prediction_split_part="test",
            row=row,
            interpretation_boundary=interpretation_boundary(version),
        ))
    return rows


def load_run_summary_seed_rows(version: str, summary_dir: Path, seed: str) -> list[dict[str, Any]]:
    path = summary_dir / "split_policy_run_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["run"].astype(str).str.contains(f"seed{seed}")]
    rows = []
    for split_policy, group in frame.groupby("split_policy"):
        combined = combine_metric_group(group)
        rows.append(metric_row(
            source_version=version,
            source_path=path,
            dataset_boundary=dataset_boundary(version),
            split_policy=str(split_policy),
            fold="holdout" if "8_2" in str(split_policy) else "fold1-5",
            seed_set=seed,
            ensemble_type="single_seed_baseline",
            ablation="full",
            prediction_split_part="test",
            row=combined,
            interpretation_boundary=interpretation_boundary(version),
        ))
    return rows


def load_focus_ablation_rows(version: str, summary_dir: Path, seed: str) -> list[dict[str, Any]]:
    path = summary_dir / "focus_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["prediction_split_part"].astype(str).str.lower() == "test"].copy()
    frame = frame[frame["run"].astype(str).str.contains(f"seed{seed}")]
    frame["ablation"] = frame["run"].map(parse_ablation)
    frame["split_policy"] = frame["run"].map(parse_split_policy)
    rows = []
    for (ablation, split_policy), group in frame.groupby(["ablation", "split_policy"]):
        combined = combine_metric_group(group)
        rows.append(metric_row(
            source_version=version,
            source_path=path,
            dataset_boundary=dataset_boundary(version),
            split_policy=str(split_policy),
            fold="holdout" if "8_2" in str(split_policy) else "fold1-5",
            seed_set=seed,
            ensemble_type="single_seed_ablation",
            ablation=str(ablation),
            prediction_split_part="test",
            row=combined,
            interpretation_boundary=interpretation_boundary(version),
        ))
    return rows


def load_v1_2_31_rows(summary_dir: Path) -> list[dict[str, Any]]:
    path = summary_dir / "focus_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["prediction_split_part"].astype(str).str.lower() == "test"].copy()
    frame["ablation"] = frame["run"].map(parse_ablation)
    frame["split_policy"] = frame["run"].map(parse_split_policy)
    rows = []
    for _, row in frame.iterrows():
        rows.append(metric_row(
            source_version="v1.2.31",
            source_path=path,
            dataset_boundary=dataset_boundary("v1.2.31"),
            split_policy=str(row["split_policy"]),
            fold="holdout",
            seed_set="2042",
            ensemble_type="single_seed_sensitivity",
            ablation=str(row["ablation"]),
            prediction_split_part="test",
            row=row,
            interpretation_boundary=interpretation_boundary("v1.2.31"),
        ))
    return rows


def metric_row(
    *,
    source_version: str,
    source_path: Path,
    dataset_boundary: str,
    split_policy: str,
    fold: str,
    seed_set: str,
    ensemble_type: str,
    ablation: str,
    prediction_split_part: str,
    row: Any,
    interpretation_boundary: str,
) -> dict[str, Any]:
    return {
        "source_version": source_version,
        "source_path": str(source_path),
        "dataset_boundary": dataset_boundary,
        "split_policy": split_policy,
        "fold": fold,
        "seed_set": seed_set,
        "ensemble_type": ensemble_type,
        "ablation": ablation,
        "prediction_split_part": prediction_split_part,
        "n": numeric_value(row, "n"),
        "r2": numeric_value(row, "r2"),
        "rmse": numeric_value(row, "rmse"),
        "mae": numeric_value(row, "mae"),
        "huber_loss": numeric_value(row, "huber_loss"),
        "task_count": numeric_value(row, "task_count"),
        "baseline_source": "",
        "delta_mae": "",
        "delta_r2": "",
        "interpretation_boundary": interpretation_boundary,
    }


def combine_metric_group(group: pd.DataFrame) -> dict[str, float]:
    n = pd.to_numeric(group["n"], errors="coerce").fillna(0.0)
    total = float(n.sum())
    if total <= 0:
        total = float(len(group))
        n = pd.Series([1.0] * len(group), index=group.index)
    return {
        "n": total,
        "r2": weighted_average(group["r2"], n),
        "rmse": math.sqrt(float((n * pd.to_numeric(group["rmse"], errors="coerce").fillna(0.0) ** 2).sum()) / total),
        "mae": weighted_average(group["mae"], n),
        "huber_loss": weighted_average(group["huber_loss"], n),
        "task_count": weighted_average(group["task_count"], n),
    }


def weighted_average(values: Any, weights: Any) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float((numeric * weights).sum() / max(float(weights.sum()), 1.0))


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    frame = metrics.copy()
    frame["baseline_source"] = frame["baseline_source"].astype("object")
    frame["delta_mae"] = pd.to_numeric(frame["delta_mae"], errors="coerce")
    frame["delta_r2"] = pd.to_numeric(frame["delta_r2"], errors="coerce")
    baseline_lookup = {
        (row.source_version, row.split_policy, row.seed_set): row
        for row in frame.itertuples()
        if row.ablation == "full" and row.ensemble_type != "5_seed_ensemble"
    }
    for idx, row in frame.iterrows():
        if row["ablation"] == "full":
            continue
        baseline = baseline_lookup.get((row["source_version"], row["split_policy"], row["seed_set"]))
        if baseline is None and row["source_version"] == "v1.2.21":
            baseline = baseline_lookup.get(("v1.2.15", row["split_policy"], row["seed_set"]))
        if baseline is None and row["source_version"] in {"v1.2.26"}:
            baseline = baseline_lookup.get(("v1.2.22", row["split_policy"], row["seed_set"]))
        if baseline is None and row["source_version"] == "v1.2.31":
            baseline = baseline_lookup.get(("v1.2.31", row["split_policy"], row["seed_set"]))
        if baseline is None:
            continue
        frame.at[idx, "baseline_source"] = f"{baseline.source_version}|{baseline.split_policy}|seed{baseline.seed_set}|full"
        frame.at[idx, "delta_mae"] = float(row["mae"]) - float(baseline.mae)
        frame.at[idx, "delta_r2"] = float(row["r2"]) - float(baseline.r2)
    return frame


def build_coverage_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ablation in CURRENT_ABLATIONS:
        subset = metrics[metrics["ablation"] == ablation] if not metrics.empty else pd.DataFrame()
        current_status = "available_current_boundary" if not subset.empty else "missing_current_boundary"
        recommended = "none"
        if ablation in {"no_descriptors", "no_fingerprint", "descriptors_only"} and subset.empty:
            recommended = "run seed2042 on no-metal random_8_2 and scaffold_cluster_8_2"
        elif ablation in {"full", "no_context", "no_species_lifestage", "no_molecular_residual", "no_molecular_size_descriptors"}:
            recommended = "use existing current-boundary evidence"
        rows.append({
            "ablation": ablation,
            "random_full": has_split(subset, "random_8_2"),
            "no_metal_random": any(subset["source_version"].isin(["v1.2.22", "v1.2.26", "v1.2.31"])) if not subset.empty else False,
            "scaffold_cluster": has_split(subset, "scaffold_cluster_8_2"),
            "seed2042_available": any(subset["seed_set"].astype(str).str.contains("2042")) if not subset.empty else False,
            "multi_seed_available": any(subset["ensemble_type"].astype(str).str.contains("5_seed")) if not subset.empty else False,
            "current_status": current_status,
            "recommended_action": recommended,
        })
    return pd.DataFrame(rows)


def build_rerun_gap_matrix(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priority = 1
    for ablation in ("no_descriptors", "no_fingerprint", "descriptors_only"):
        status = coverage.loc[coverage["ablation"] == ablation, "current_status"]
        if status.empty or status.iloc[0] != "missing_current_boundary":
            continue
        for split_policy in ("random_8_2", "scaffold_cluster_8_2"):
            rows.append({
                "priority": priority,
                "ablation": ablation,
                "split_policy": split_policy,
                "seed": 2042,
                "baseline_to_compare": "v1.2.31 full seed2042 for matching split when available; otherwise v1.2.22/v1.2.24 seed2042 full",
                "reason": "current boundary lacks this molecular-input ablation after excluding v1.2.18/CAS-number holdout",
                "expected_decision_use": "separate molecular descriptor/fingerprint contribution from species/context embedding dominance",
            })
            priority += 1
    rows.append({
        "priority": priority,
        "ablation": "graph_only_no_descriptors_no_fingerprint",
        "split_policy": "random_8_2;scaffold_cluster_8_2",
        "seed": 2042,
        "baseline_to_compare": "same split full seed2042",
        "reason": "requested second-stage branch; run only after graph encoder training path is enabled",
        "expected_decision_use": "test molecular graph signal without RDKit descriptors or Morgan fingerprint while retaining context embeddings",
    })
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, metrics: pd.DataFrame, gaps: pd.DataFrame) -> None:
    best_lines = []
    for _, row in metrics[(metrics["source_version"] == "v1.2.31") & (metrics["prediction_split_part"] == "test")].iterrows():
        delta = row["delta_mae"]
        delta_text = "baseline" if pd.isna(delta) else f"{float(delta):+.4f}"
        best_lines.append(
            f"- {row['split_policy']} `{row['ablation']}`: MAE={float(row['mae']):.4f}, "
            f"R2={float(row['r2']):.4f}, delta_MAE={delta_text}"
        )
    gap_lines = [
        f"- P{int(row.priority)} `{row.ablation}` on `{row.split_policy}` seed `{row.seed}`"
        for row in gaps.itertuples()
    ]
    text = "\n".join([
        "# 分子信号强度探索 20260707",
        "",
        "## 边界",
        "",
        "- 当前汇总排除 `v1.2.18` 固定化学留出/CAS-number 项目；它只保留在 `source_manifest.csv` 的 `excluded_historical` 行中。",
        "- 当前结论只使用随机主线、去金属/无机随机、结构骨架聚类外推和 v1.2.31 分子大小敏感性结果。",
        "- 最佳性能/最终模型可使用 ensemble；机制消融和诊断消融按 seed `2042` 规划补跑。",
        "",
        "## 文件",
        "",
        "- `source_manifest.csv`: 来源、纳入状态、边界和解释范围。",
        "- `molecular_signal_overall_metrics.csv`: 当前边界内 full/ablation 指标和可计算 delta。",
        "- `ablation_coverage_matrix.csv`: 当前边界下消融覆盖情况。",
        "- `rerun_gap_matrix.csv`: 推荐补跑矩阵。",
        "- 旁支设计说明：`docs/molecular_signal_strength_padel_graph_branch_design_20260707.md`。",
        "",
        "## v1.2.31 分子大小敏感性",
        "",
        *(best_lines or ["- 未找到 v1.2.31 指标。"]),
        "",
        "## 推荐补跑",
        "",
        *(gap_lines or ["- 当前无优先补跑缺口。"]),
        "",
        "## 科研解释要点",
        "",
        "- `no_context` 和 `no_species_lifestage` 已有当前边界证据，说明物种/上下文 embedding 在随机插值中贡献很大。",
        "- `no_molecular_size_descriptors` 在 random 8:2 几乎不损失、在 scaffold-cluster 8:2 小幅损失，不支持模型主要依赖分子量/大小耦合获得性能。",
        "- `no_descriptors`、`no_fingerprint`、`descriptors_only` 在排除 v1.2.18 后缺少当前边界证据，应优先用 seed2042 补齐。",
        "",
    ])
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_ablation(run: str) -> str:
    text = str(run)
    if "no_molecular_size_descriptors" in text:
        return "no_molecular_size_descriptors"
    match = re.search(r"ablation_([^_]+(?:_[^_]+)*?)_seed", text)
    if match:
        return match.group(1)
    match = re.search(r"strategy_([^_]+(?:_[^_]+)*?)_seed", text)
    if match:
        return match.group(1)
    if "_full_" in text or text.endswith("_full"):
        return "full"
    return text


def parse_split_policy(run: str) -> str:
    text = str(run).lower()
    if "scaffold_cluster_8_2" in text or "scaffold8_2" in text:
        return "scaffold_cluster_8_2"
    if "scaffold_cluster_5fold" in text or "scaffold5fold" in text:
        return "scaffold_cluster_5fold"
    if "random_8_2" in text or "random8_2" in text:
        return "random_8_2"
    if "random_5fold" in text or "random5fold" in text:
        return "random_5fold"
    return "unknown"


def dataset_boundary(version: str) -> str:
    return {
        "v1.2.15": "all-current random split",
        "v1.2.21": "all-current random split targeted ablation",
        "v1.2.22": "no-metal/inorganic random split",
        "v1.2.24": "no-metal/inorganic scaffold-cluster split",
        "v1.2.26": "no-metal/inorganic random split targeted ablation",
        "v1.2.31": "no-metal/inorganic molecular-size sensitivity",
    }.get(version, "")


def interpretation_boundary(version: str) -> str:
    return {
        "v1.2.15": "random interpolation baseline",
        "v1.2.21": "random interpolation targeted ablation",
        "v1.2.22": "organic-representable random interpolation baseline",
        "v1.2.24": "chemical-family extrapolation baseline",
        "v1.2.26": "single-seed no-metal random interpolation sensitivity",
        "v1.2.31": "single-seed molecular-size sensitivity on random and scaffold holdout",
    }.get(version, "")


def numeric_value(row: Any, key: str) -> float:
    if isinstance(row, dict):
        value = row.get(key, "")
    else:
        value = row[key] if key in row else ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def has_split(frame: pd.DataFrame, split_policy: str) -> bool:
    return not frame.empty and any(frame["split_policy"].astype(str) == split_policy)


if __name__ == "__main__":
    main()
