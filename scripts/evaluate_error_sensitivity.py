from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager


DEFAULT_ROWS = "outputs/experiments/v1_2_26_label_conflict_audit/conflict_enriched_prediction_rows.csv"
DEFAULT_GROUPS = "outputs/experiments/v1_2_26_label_conflict_audit/group_conflict_summary.csv"
DEFAULT_OUT_DIR = "outputs/experiments/v1_2_27_no_training_sensitivity"
DEFAULT_STYLE = "style_journal_clean_v1.yaml"
DEFAULT_GROUP_COLUMNS = ("chemical_label", "species_label", "task_head", "target_basis")
RISK_COLUMNS = (
    "group_label_conflict",
    "source_conflict",
    "repeated_high_error",
    "in_domain_high_error",
    "multi_reference",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate no-training error sensitivity by recalculating metrics after "
            "excluding label/source conflict groups from existing predictions."
        )
    )
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--groups", default=DEFAULT_GROUPS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--group-columns", default=",".join(DEFAULT_GROUP_COLUMNS))
    parser.add_argument("--min-dimension-n", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows_path = Path(args.rows)
    groups_path = Path(args.groups)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    style = load_style(Path(args.style))
    apply_matplotlib_style(style)

    group_columns = tuple(column.strip() for column in str(args.group_columns).split(",") if column.strip())
    rows = prepare_rows(pd.read_csv(rows_path, low_memory=False))
    groups = prepare_groups(pd.read_csv(groups_path, low_memory=False), group_columns=group_columns)
    data = attach_group_flags(rows, groups, group_columns=group_columns)

    scenarios = build_scenarios(data)
    outputs: dict[str, str] = {}

    overall = summarize_scenarios(data, scenarios)
    overall = add_overall_deltas(overall)
    path = out_dir / "sensitivity_overall.csv"
    overall.to_csv(path, index=False, encoding="utf-8-sig")
    outputs["sensitivity_overall"] = str(path)

    dimension_specs = {
        "sensitivity_by_split_policy": ["split_policy"],
        "sensitivity_by_task": ["task_head"],
        "sensitivity_by_task_family": ["task_family"],
        "sensitivity_by_ad_tier": ["ad_tier"],
        "sensitivity_by_chemical_class": ["chemical_class_l1", "chemical_class_l2"],
        "sensitivity_by_taxon": ["taxon_group_l1"],
    }
    for name, columns in dimension_specs.items():
        available = [column for column in columns if column in data.columns]
        if len(available) != len(columns):
            continue
        table = summarize_by_dimension(data, scenarios, dimensions=columns, min_n=int(args.min_dimension_n))
        if table.empty:
            continue
        dim_path = out_dir / f"{name}.csv"
        table.to_csv(dim_path, index=False, encoding="utf-8-sig")
        outputs[name] = str(dim_path)

    removed = summarize_removed_groups(groups)
    removed_path = out_dir / "removed_group_summary.csv"
    removed.to_csv(removed_path, index=False, encoding="utf-8-sig")
    outputs["removed_group_summary"] = str(removed_path)

    outputs.update(plot_outputs(overall, data, scenarios, figures_dir=figures_dir, style=style, top_n=int(args.top_n), min_n=int(args.min_dimension_n)))

    manifest = {
        "rows": str(rows_path),
        "groups": str(groups_path),
        "out_dir": str(out_dir),
        "group_columns": list(group_columns),
        "prediction_rows": int(len(data)),
        "group_rows": int(len(groups)),
        "scenarios": [
            {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "prospective_qc_filter": bool(scenario["prospective_qc_filter"]),
            }
            for scenario in scenarios
        ],
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    outputs["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": int(len(data)), "outputs": outputs}, ensure_ascii=False))


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ("y_true", "y_pred", "abs_error", "residual"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "residual" not in data.columns:
        data["residual"] = data["y_true"] - data["y_pred"]
    if "abs_error" not in data.columns:
        data["abs_error"] = data["residual"].abs()
    if "is_high_error" in data.columns:
        data["is_high_error"] = data["is_high_error"].map(parse_bool)
    else:
        threshold = float(data["abs_error"].quantile(0.90))
        data["is_high_error"] = data["abs_error"] >= threshold
    for column in DEFAULT_GROUP_COLUMNS + (
        "split_policy",
        "task_head",
        "task_family",
        "ad_tier",
        "chemical_class_l1",
        "chemical_class_l2",
        "taxon_group_l1",
    ):
        if column not in data.columns:
            data[column] = "unknown"
        data[column] = data[column].map(clean_text).replace("", "unknown")
    return data


def prepare_groups(frame: pd.DataFrame, *, group_columns: tuple[str, ...]) -> pd.DataFrame:
    data = frame.copy()
    for column in group_columns:
        if column not in data.columns:
            data[column] = "unknown"
        data[column] = data[column].map(clean_text).replace("", "unknown")
    for column in RISK_COLUMNS:
        if column not in data.columns:
            data[column] = False
        data[column] = data[column].map(parse_bool)
    if "conflict_risk_score" not in data.columns:
        data["conflict_risk_score"] = data.loc[:, list(RISK_COLUMNS)].sum(axis=1)
    data["conflict_risk_score"] = pd.to_numeric(data["conflict_risk_score"], errors="coerce").fillna(0).astype(int)
    for column in ("prediction_n", "mae", "source_target_range", "y_true_range"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def attach_group_flags(frame: pd.DataFrame, groups: pd.DataFrame, *, group_columns: tuple[str, ...]) -> pd.DataFrame:
    selected_columns = [
        *group_columns,
        *RISK_COLUMNS,
        "conflict_risk_score",
        "risk_reasons",
        "source_target_range",
        "y_true_range",
    ]
    selected = groups.loc[:, [column for column in selected_columns if column in groups.columns]].drop_duplicates(list(group_columns))
    data = frame.merge(selected, on=list(group_columns), how="left")
    for column in RISK_COLUMNS:
        if column not in data.columns:
            data[column] = False
        data[column] = data[column].map(parse_bool)
    data["conflict_risk_score"] = pd.to_numeric(data.get("conflict_risk_score", 0), errors="coerce").fillna(0).astype(int)
    data["risk_reasons"] = data.get("risk_reasons", "").map(clean_text) if "risk_reasons" in data.columns else ""
    return data


def build_scenarios(frame: pd.DataFrame) -> list[dict[str, Any]]:
    label_or_source = frame["group_label_conflict"] | frame["source_conflict"]
    any_risk = frame["conflict_risk_score"] > 0
    ad_a = frame["ad_tier"].map(clean_text).eq("AD-A")
    return [
        {
            "name": "baseline_all",
            "description": "All audited prediction rows.",
            "mask": pd.Series(True, index=frame.index),
            "prospective_qc_filter": True,
        },
        {
            "name": "exclude_group_label_conflict",
            "description": "Exclude groups whose observed y_true range is at least 2 log units.",
            "mask": ~frame["group_label_conflict"],
            "prospective_qc_filter": True,
        },
        {
            "name": "exclude_source_conflict",
            "description": "Exclude groups flagged by source-level target/reference conflict.",
            "mask": ~frame["source_conflict"],
            "prospective_qc_filter": True,
        },
        {
            "name": "exclude_label_or_source_conflict",
            "description": "Exclude groups with either group-level label conflict or source conflict.",
            "mask": ~label_or_source,
            "prospective_qc_filter": True,
        },
        {
            "name": "exclude_all_five_risk_flags",
            "description": "Exclude only the strongest groups that trigger all five conflict risk flags.",
            "mask": frame["conflict_risk_score"] < len(RISK_COLUMNS),
            "prospective_qc_filter": False,
        },
        {
            "name": "exclude_any_conflict_risk",
            "description": "Keep only groups with no conflict risk flag.",
            "mask": ~any_risk,
            "prospective_qc_filter": False,
        },
        {
            "name": "ad_a_only",
            "description": "Keep only AD-A predictions to isolate in-domain behavior.",
            "mask": ad_a,
            "prospective_qc_filter": True,
        },
        {
            "name": "ad_a_exclude_label_or_source_conflict",
            "description": "Keep AD-A predictions and exclude label/source conflict groups.",
            "mask": ad_a & ~label_or_source,
            "prospective_qc_filter": True,
        },
    ]


def summarize_scenarios(frame: pd.DataFrame, scenarios: list[dict[str, Any]]) -> pd.DataFrame:
    baseline_n = int(len(frame))
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        sub = frame[scenario["mask"]].copy()
        row = metric_row(sub, baseline_n=baseline_n)
        row.update(
            {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "prospective_qc_filter": bool(scenario["prospective_qc_filter"]),
                "removed_n": int(baseline_n - len(sub)),
                "removed_fraction": float(1.0 - len(sub) / baseline_n) if baseline_n else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_dimension(
    frame: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    *,
    dimensions: list[str],
    min_n: int,
) -> pd.DataFrame:
    baseline_counts = frame.groupby(dimensions, dropna=False).size().rename("baseline_n").reset_index()
    baseline_counts = baseline_counts[baseline_counts["baseline_n"] >= min_n].copy()
    if baseline_counts.empty:
        return pd.DataFrame()
    allowed = baseline_counts.loc[:, dimensions].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        sub = frame[scenario["mask"]].merge(allowed, on=dimensions, how="inner")
        for key, part in sub.groupby(dimensions, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            base_n = int(baseline_counts.merge(pd.DataFrame([{column: value for column, value in zip(dimensions, key)}]), on=dimensions, how="inner")["baseline_n"].iloc[0])
            row = metric_row(part, baseline_n=base_n)
            row.update({column: value for column, value in zip(dimensions, key)})
            row["scenario"] = scenario["name"]
            row["description"] = scenario["description"]
            row["prospective_qc_filter"] = bool(scenario["prospective_qc_filter"])
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = result[result["scenario"].eq("baseline_all")].loc[:, [*dimensions, "mae", "rmse", "r2", "n"]]
    baseline = baseline.rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse", "r2": "baseline_r2", "n": "baseline_n"})
    result = result.merge(baseline, on=dimensions, how="left")
    result["delta_mae_vs_baseline"] = result["mae"] - result["baseline_mae"]
    result["delta_rmse_vs_baseline"] = result["rmse"] - result["baseline_rmse"]
    result["retained_fraction_of_dimension"] = result["n"] / result["baseline_n"]
    return result.sort_values(["scenario", "delta_mae_vs_baseline", "baseline_n"], ascending=[True, True, False])


def add_overall_deltas(overall: pd.DataFrame) -> pd.DataFrame:
    result = overall.copy()
    baseline = result[result["scenario"].eq("baseline_all")].iloc[0]
    result["delta_mae_vs_baseline"] = result["mae"] - float(baseline["mae"])
    result["delta_rmse_vs_baseline"] = result["rmse"] - float(baseline["rmse"])
    result["delta_r2_vs_baseline"] = result["r2"] - float(baseline["r2"])
    return result


def metric_row(frame: pd.DataFrame, *, baseline_n: int) -> dict[str, Any]:
    y_true = pd.to_numeric(frame.get("y_true", pd.Series(dtype=float)), errors="coerce")
    y_pred = pd.to_numeric(frame.get("y_pred", pd.Series(dtype=float)), errors="coerce")
    residual = y_true - y_pred
    abs_error = residual.abs()
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = residual[valid]
    abs_error = abs_error[valid]
    n = int(valid.sum())
    high_error = frame.loc[valid.index[valid], "is_high_error"] if "is_high_error" in frame.columns and n else pd.Series(dtype=bool)
    high_error = high_error.map(parse_bool) if len(high_error) else high_error
    return {
        "n": n,
        "retained_fraction": float(n / baseline_n) if baseline_n else math.nan,
        "mae": float(abs_error.mean()) if n else math.nan,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))) if n else math.nan,
        "r2": r2_score(y_true, y_pred),
        "median_abs_error": float(abs_error.median()) if n else math.nan,
        "p90_abs_error": float(abs_error.quantile(0.90)) if n else math.nan,
        "max_abs_error": float(abs_error.max()) if n else math.nan,
        "mean_signed_error": float(residual.mean()) if n else math.nan,
        "high_error_n": int(high_error.sum()) if len(high_error) else 0,
        "high_error_fraction": float(high_error.mean()) if len(high_error) else math.nan,
        "y_true_mean": float(y_true.mean()) if n else math.nan,
        "y_true_std": float(y_true.std(ddof=0)) if n else math.nan,
    }


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 2:
        return math.nan
    denominator = float(np.square(y_true - y_true.mean()).sum())
    if denominator <= 0:
        return math.nan
    numerator = float(np.square(y_true - y_pred).sum())
    return 1.0 - numerator / denominator


def summarize_removed_groups(groups: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("exclude_group_label_conflict", groups["group_label_conflict"]),
        ("exclude_source_conflict", groups["source_conflict"]),
        ("exclude_label_or_source_conflict", groups["group_label_conflict"] | groups["source_conflict"]),
        ("exclude_all_five_risk_flags", groups["conflict_risk_score"] >= len(RISK_COLUMNS)),
        ("exclude_any_conflict_risk", groups["conflict_risk_score"] > 0),
    ]
    rows: list[dict[str, Any]] = []
    total_groups = len(groups)
    total_predictions = float(pd.to_numeric(groups.get("prediction_n", pd.Series(dtype=float)), errors="coerce").sum())
    for scenario, mask in specs:
        sub = groups[mask].copy()
        removed_predictions = float(pd.to_numeric(sub.get("prediction_n", pd.Series(dtype=float)), errors="coerce").sum()) if not sub.empty else 0.0
        rows.append(
            {
                "scenario": scenario,
                "removed_group_n": int(len(sub)),
                "removed_group_fraction": float(len(sub) / total_groups) if total_groups else math.nan,
                "removed_prediction_n": int(removed_predictions),
                "removed_prediction_fraction": float(removed_predictions / total_predictions) if total_predictions else math.nan,
                "removed_group_weighted_mae": weighted_mean(sub.get("mae", pd.Series(dtype=float)), sub.get("prediction_n", pd.Series(dtype=float))) if not sub.empty else math.nan,
                "removed_max_source_target_range": float(pd.to_numeric(sub.get("source_target_range", pd.Series(dtype=float)), errors="coerce").max()) if not sub.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_outputs(
    overall: pd.DataFrame,
    frame: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    *,
    figures_dir: Path,
    style: dict[str, Any],
    top_n: int,
    min_n: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    outputs.update(plot_overall_delta(overall, figures_dir / "overall_delta_mae", style=style))
    outputs.update(plot_mae_vs_coverage(overall, figures_dir / "mae_vs_retained_fraction", style=style))
    task = summarize_by_dimension(frame, scenarios, dimensions=["task_head"], min_n=min_n)
    outputs.update(plot_task_sensitivity(task, figures_dir / "task_sensitivity_label_or_source", style=style, top_n=top_n))
    manifest = pd.DataFrame([{"figure": key, "path": value} for key, value in sorted(outputs.items())])
    path = figures_dir / "figure_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    outputs["figure_manifest"] = str(path)
    return outputs


def plot_overall_delta(overall: pd.DataFrame, out_base: Path, *, style: dict[str, Any]) -> dict[str, str]:
    data = overall[~overall["scenario"].eq("baseline_all")].copy()
    data = data.sort_values("delta_mae_vs_baseline", ascending=True)
    fig_height = max(3.8, 0.35 * len(data) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    colors = ["#009E73" if value < 0 else "#D55E00" for value in data["delta_mae_vs_baseline"]]
    bars = ax.barh(data["scenario"], data["delta_mae_vs_baseline"], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("MAE change vs baseline")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    style_axes(ax, style)
    for bar, retained in zip(bars, data["retained_fraction"]):
        x = bar.get_width()
        if x < 0:
            ax.text(-0.002, bar.get_y() + bar.get_height() / 2, f"{retained * 100:.1f}% kept", va="center", ha="right", fontsize=7)
        else:
            ax.text(x + 0.002, bar.get_y() + bar.get_height() / 2, f"{retained * 100:.1f}% kept", va="center", ha="left", fontsize=7)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def plot_mae_vs_coverage(overall: pd.DataFrame, out_base: Path, *, style: dict[str, Any]) -> dict[str, str]:
    data = overall.copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = ["#D55E00" if not value else "#0072B2" for value in data["prospective_qc_filter"]]
    ax.scatter(data["retained_fraction"] * 100.0, data["mae"], s=36, color=colors)
    for _, row in data.iterrows():
        ax.text(float(row["retained_fraction"]) * 100.0 + 0.25, float(row["mae"]), shorten(row["scenario"], 28), fontsize=6.5, va="center")
    ax.set_xlabel("Retained prediction rows (%)")
    ax.set_ylabel("MAE")
    ax.grid(True)
    style_axes(ax, style)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def plot_task_sensitivity(task: pd.DataFrame, out_base: Path, *, style: dict[str, Any], top_n: int) -> dict[str, str]:
    if task.empty:
        return {}
    data = task[task["scenario"].eq("exclude_label_or_source_conflict")].copy()
    data = data[data["n"] >= 10].copy()
    if data.empty:
        return {}
    data = data.reindex(data["delta_mae_vs_baseline"].abs().sort_values(ascending=False).index).head(top_n)
    data = data.sort_values("delta_mae_vs_baseline", ascending=True)
    fig_height = max(4.0, 0.27 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    colors = ["#009E73" if value < 0 else "#D55E00" for value in data["delta_mae_vs_baseline"]]
    ax.barh(data["task_head"], data["delta_mae_vs_baseline"], color=colors)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Task MAE change after excluding label/source conflict")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    style_axes(ax, style)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def parse_bool(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce")
    mask = numeric.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return math.nan
    return float((numeric[mask] * weight[mask]).sum() / weight[mask].sum())


def shorten(value: Any, max_length: int) -> str:
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "..."


def load_style(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def apply_matplotlib_style(style: dict[str, Any]) -> None:
    rc = style.get("matplotlib_rc", {})
    if isinstance(rc, dict):
        plt.rcParams.update(rc)
    fonts = style.get("fonts", {})
    english = fonts.get("english", {}) if isinstance(fonts, dict) else {}
    regular = english.get("regular")
    if regular:
        font_path = Path(regular)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = english.get("family_name", "Arial")
    sizes = style.get("font_sizes_pt", {})
    if sizes:
        plt.rcParams["axes.labelsize"] = float(sizes.get("axis_label", 8.0))
        plt.rcParams["xtick.labelsize"] = float(sizes.get("tick_label", 7.0))
        plt.rcParams["ytick.labelsize"] = float(sizes.get("tick_label", 7.0))
        plt.rcParams["legend.fontsize"] = float(sizes.get("legend", 7.0))


def style_axes(ax: Any, style: dict[str, Any]) -> None:
    axes = style.get("axes", {})
    ticks = style.get("ticks", {})
    grid = style.get("grid", {})
    for spine in ax.spines.values():
        spine.set_color(axes.get("spine_color", "#333333"))
        spine.set_linewidth(float(axes.get("spine_linewidth", 0.75)))
    ax.tick_params(
        direction=ticks.get("direction", "out"),
        length=float(ticks.get("major_size", 3.0)),
        width=float(ticks.get("major_width", 0.65)),
        colors=ticks.get("color", "#333333"),
        pad=float(ticks.get("pad", 2.5)),
    )
    ax.grid(
        bool(grid.get("show", True)),
        color=grid.get("color", "#DDE3EA"),
        linewidth=float(grid.get("linewidth", 0.45)),
        linestyle=grid.get("linestyle", "--"),
        alpha=float(grid.get("alpha", 0.72)),
        zorder=int(grid.get("zorder", 0)),
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


if __name__ == "__main__":
    main()
