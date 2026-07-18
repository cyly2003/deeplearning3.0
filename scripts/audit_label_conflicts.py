from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager


DEFAULT_AUDITED_ROWS = "outputs/experiments/v1_2_25_prediction_error_audit/audited_prediction_rows.csv"
DEFAULT_DB = "outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
DEFAULT_SOURCE_TABLE = "aggregated_task_records_aquatic_soil_ptox_qc"
DEFAULT_OUT_DIR = "outputs/experiments/v1_2_26_label_conflict_audit"
DEFAULT_STYLE = "style_journal_clean_v1.yaml"
DEFAULT_GROUP_COLUMNS = ("chemical_label", "species_label", "task_head", "target_basis")
SOURCE_METADATA_COLUMNS = (
    "aggregate_id",
    "target_value_median",
    "target_value_mean",
    "target_value_std",
    "target_value_count",
    "target_value_min",
    "target_value_max",
    "cross_reference_conflict_flag",
    "cross_reference_range",
    "source_reference_count",
    "source_test_count",
    "publication_year_latest",
    "publication_years",
    "reference_numbers",
    "test_ids",
    "result_ids",
    "duration_bin_h",
    "duration_bin_rule",
    "medium_conflict_flag",
    "mean_reference_weight",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit label/source conflicts behind high prediction errors by grouping "
            "chemical-species-task-target combinations."
        )
    )
    parser.add_argument("--audited-rows", default=DEFAULT_AUDITED_ROWS)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--group-columns", default=",".join(DEFAULT_GROUP_COLUMNS))
    parser.add_argument("--group-range-threshold", type=float, default=2.0)
    parser.add_argument("--source-range-threshold", type=float, default=1.0)
    parser.add_argument("--min-high-error-n", type=int, default=2)
    parser.add_argument("--min-high-error-fraction", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--skip-db-metadata", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audited_path = Path(args.audited_rows)
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    style = load_style(Path(args.style))
    apply_matplotlib_style(style)

    rows = pd.read_csv(audited_path, low_memory=False)
    if not args.skip_db_metadata and db_path.exists():
        rows = add_source_metadata(rows, db_path=db_path, source_table=str(args.source_table))
    rows = prepare_rows(rows)
    group_columns = tuple(column.strip() for column in str(args.group_columns).split(",") if column.strip())
    groups = summarize_conflict_groups(
        rows,
        group_columns=group_columns,
        group_range_threshold=float(args.group_range_threshold),
        source_range_threshold=float(args.source_range_threshold),
        min_high_error_n=int(args.min_high_error_n),
        min_high_error_fraction=float(args.min_high_error_fraction),
    )

    outputs: dict[str, str] = {}
    enriched_path = out_dir / "conflict_enriched_prediction_rows.csv"
    rows.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    outputs["conflict_enriched_prediction_rows"] = str(enriched_path)

    summary_path = out_dir / "group_conflict_summary.csv"
    groups.to_csv(summary_path, index=False, encoding="utf-8-sig")
    outputs["group_conflict_summary"] = str(summary_path)

    high_risk = groups[groups["conflict_risk_score"] > 0].copy()
    high_risk_path = out_dir / "high_risk_conflict_groups.csv"
    high_risk.to_csv(high_risk_path, index=False, encoding="utf-8-sig")
    outputs["high_risk_conflict_groups"] = str(high_risk_path)

    outputs.update(write_dimension_summaries(groups, out_dir=out_dir))
    outputs.update(write_case_tables(rows, groups, out_dir=out_dir, top_n=int(args.top_n)))
    outputs.update(plot_outputs(groups, figures_dir=figures_dir, style=style, top_n=int(args.top_n)))

    manifest = {
        "audited_rows": str(audited_path),
        "db": str(db_path),
        "source_table": str(args.source_table),
        "out_dir": str(out_dir),
        "group_columns": list(group_columns),
        "rows": int(len(rows)),
        "groups": int(len(groups)),
        "high_risk_groups": int(len(high_risk)),
        "group_range_threshold": float(args.group_range_threshold),
        "source_range_threshold": float(args.source_range_threshold),
        "min_high_error_n": int(args.min_high_error_n),
        "min_high_error_fraction": float(args.min_high_error_fraction),
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    outputs["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "groups": int(len(groups)), "high_risk_groups": int(len(high_risk)), "outputs": outputs}, ensure_ascii=False))


def add_source_metadata(frame: pd.DataFrame, *, db_path: Path, source_table: str) -> pd.DataFrame:
    if "aggregate_id" not in frame.columns:
        return frame
    with sqlite3.connect(db_path) as conn:
        available = table_columns(conn, source_table)
        selected = [column for column in SOURCE_METADATA_COLUMNS if column in available]
        if "aggregate_id" not in selected:
            return frame
        quoted = ", ".join(quote_identifier(column) for column in selected)
        metadata = pd.read_sql_query(f"SELECT {quoted} FROM {quote_identifier(source_table)}", conn)
    metadata = metadata.drop_duplicates("aggregate_id")
    left = frame.copy()
    left["aggregate_id"] = left["aggregate_id"].map(clean_text)
    metadata["aggregate_id"] = metadata["aggregate_id"].map(clean_text)
    overlapping = [column for column in metadata.columns if column in left.columns and column != "aggregate_id"]
    if overlapping:
        metadata = metadata.rename(columns={column: f"{column}_source" for column in overlapping})
    merged = left.merge(metadata, on="aggregate_id", how="left")
    for column in overlapping:
        source_column = f"{column}_source"
        merged[column] = merged[column].where(has_value(merged[column]), merged[source_column])
    return merged.drop(columns=[column for column in merged.columns if column.endswith("_source")])


def prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in (
        "y_true",
        "y_pred",
        "abs_error",
        "residual",
        "target_value_min",
        "target_value_max",
        "target_value_count",
        "target_value_std",
        "cross_reference_range",
        "source_reference_count",
        "source_test_count",
        "duration_bin_h",
    ):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "residual" not in data.columns:
        data["residual"] = pd.to_numeric(data["y_true"], errors="coerce") - pd.to_numeric(data["y_pred"], errors="coerce")
    if "abs_error" not in data.columns:
        data["abs_error"] = data["residual"].abs()
    data["source_target_min"] = data["target_value_min"] if "target_value_min" in data.columns else data["y_true"]
    data["source_target_max"] = data["target_value_max"] if "target_value_max" in data.columns else data["y_true"]
    data["source_target_min"] = data["source_target_min"].where(data["source_target_min"].notna(), data["y_true"])
    data["source_target_max"] = data["source_target_max"].where(data["source_target_max"].notna(), data["y_true"])
    if "is_high_error" in data.columns:
        data["is_high_error"] = data["is_high_error"].map(parse_bool)
    else:
        threshold = float(data["abs_error"].quantile(0.90))
        data["is_high_error"] = data["abs_error"] >= threshold
    for column in DEFAULT_GROUP_COLUMNS + (
        "split_policy",
        "fold",
        "ad_tier",
        "chemical_class_l1",
        "chemical_class_l2",
        "taxon_group_l1",
        "quality_label",
        "conversion_path",
        "value_quality",
    ):
        if column not in data.columns:
            data[column] = "unknown"
        data[column] = data[column].map(clean_text).replace("", "unknown")
    return data


def summarize_conflict_groups(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    group_range_threshold: float,
    source_range_threshold: float,
    min_high_error_n: int,
    min_high_error_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in frame.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {column: value for column, value in zip(group_columns, key)}
        y_true = pd.to_numeric(sub["y_true"], errors="coerce")
        y_pred = pd.to_numeric(sub["y_pred"], errors="coerce")
        abs_error = pd.to_numeric(sub["abs_error"], errors="coerce")
        residual = pd.to_numeric(sub["residual"], errors="coerce")
        source_min = pd.to_numeric(sub["source_target_min"], errors="coerce")
        source_max = pd.to_numeric(sub["source_target_max"], errors="coerce")
        source_range = float(source_max.max() - source_min.min()) if source_min.notna().any() and source_max.notna().any() else math.nan
        y_range = float(y_true.max() - y_true.min()) if y_true.notna().any() else math.nan
        high_error_n = int(sub["is_high_error"].fillna(False).astype(bool).sum())
        prediction_n = int(len(sub))
        high_error_fraction = high_error_n / prediction_n if prediction_n else math.nan
        max_cross_range = max_numeric(sub.get("cross_reference_range"))
        any_cross_flag = any_bool(sub.get("cross_reference_conflict_flag"))
        max_target_value_count = max_numeric(sub.get("target_value_count"))
        repeated_raw_values = bool(math.isfinite(max_target_value_count) and max_target_value_count > 1)
        source_conflict = bool(
            any_cross_flag
            or (math.isfinite(max_cross_range) and max_cross_range >= source_range_threshold)
            or (math.isfinite(source_range) and source_range >= source_range_threshold and repeated_raw_values)
        )
        group_label_conflict = bool(math.isfinite(y_range) and y_range >= group_range_threshold)
        repeated_high_error = bool(high_error_n >= min_high_error_n or high_error_fraction >= min_high_error_fraction)
        ad_a_fraction = fraction_equal(sub.get("ad_tier"), "AD-A")
        in_domain_high_error = bool(ad_a_fraction >= 0.8 and high_error_n > 0)
        unique_reference_count = unique_token_count(sub.get("reference_numbers"))
        unique_test_count = unique_token_count(sub.get("test_ids"))
        multi_reference = bool(unique_reference_count > 1 or max_numeric(sub.get("source_reference_count")) > 1)
        risk_flags = {
            "group_label_conflict": group_label_conflict,
            "source_conflict": source_conflict,
            "repeated_high_error": repeated_high_error,
            "in_domain_high_error": in_domain_high_error,
            "multi_reference": multi_reference,
        }
        row.update(
            {
                "prediction_n": prediction_n,
                "unique_aggregate_n": nunique_nonempty(sub.get("aggregate_id")),
                "split_policies": join_unique(sub.get("split_policy")),
                "folds": join_unique(sub.get("fold")),
                "ad_tiers": join_unique(sub.get("ad_tier")),
                "ad_a_fraction": ad_a_fraction,
                "chemical_class_l1": mode_text(sub.get("chemical_class_l1")),
                "chemical_class_l2": mode_text(sub.get("chemical_class_l2")),
                "taxon_group_l1": mode_text(sub.get("taxon_group_l1")),
                "mae": float(abs_error.mean()),
                "median_abs_error": float(abs_error.median()),
                "p90_abs_error": float(abs_error.quantile(0.90)),
                "max_abs_error": float(abs_error.max()),
                "mean_signed_error": float(residual.mean()),
                "high_error_n": high_error_n,
                "high_error_fraction": high_error_fraction,
                "y_true_min": float(y_true.min()),
                "y_true_max": float(y_true.max()),
                "y_true_range": y_range,
                "y_pred_min": float(y_pred.min()),
                "y_pred_max": float(y_pred.max()),
                "source_target_min": float(source_min.min()) if source_min.notna().any() else math.nan,
                "source_target_max": float(source_max.max()) if source_max.notna().any() else math.nan,
                "source_target_range": source_range,
                "max_cross_reference_range": max_cross_range,
                "any_cross_reference_conflict_flag": any_cross_flag,
                "target_value_count_sum": sum_numeric(sub.get("target_value_count")),
                "max_target_value_count": max_target_value_count,
                "source_reference_count_max": max_numeric(sub.get("source_reference_count")),
                "source_test_count_max": max_numeric(sub.get("source_test_count")),
                "unique_reference_count": unique_reference_count,
                "unique_test_count": unique_test_count,
                "duration_bin_count": nunique_numeric(sub.get("duration_bin_h")),
                "duration_bins": join_unique(sub.get("duration_bin_h")),
                "effect_level_count": nunique_nonempty(sub.get("effect_level_x")),
                "effect_levels": join_unique(sub.get("effect_level_x")),
                "quality_labels": join_unique(sub.get("quality_label"), limit=8),
                "conversion_paths": join_unique(sub.get("conversion_path"), limit=8),
                **risk_flags,
                "conflict_risk_score": int(sum(1 for value in risk_flags.values() if value)),
            }
        )
        row["risk_reasons"] = ";".join(name for name, enabled in risk_flags.items() if enabled)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["conflict_risk_score", "high_error_fraction", "source_target_range", "mae", "prediction_n"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def write_dimension_summaries(groups: pd.DataFrame, *, out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    specs = {
        "task_conflict_summary": ["task_head"],
        "chemical_conflict_summary": ["chemical_label", "chemical_class_l1", "chemical_class_l2"],
        "species_conflict_summary": ["species_label", "taxon_group_l1"],
        "taxon_conflict_summary": ["taxon_group_l1"],
        "chemical_class_conflict_summary": ["chemical_class_l1", "chemical_class_l2"],
    }
    for name, keys in specs.items():
        if not all(key in groups.columns for key in keys):
            continue
        summary = summarize_group_dimensions(groups, keys)
        path = out_dir / f"{name}.csv"
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = str(path)
    flag_summary = summarize_flags(groups)
    flag_path = out_dir / "conflict_flag_summary.csv"
    flag_summary.to_csv(flag_path, index=False, encoding="utf-8-sig")
    outputs["conflict_flag_summary"] = str(flag_path)
    return outputs


def summarize_group_dimensions(groups: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in groups.groupby(keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        high_risk = sub["conflict_risk_score"] > 0
        row: dict[str, Any] = {column: value for column, value in zip(keys, key)}
        row.update(
            {
                "group_n": int(len(sub)),
                "prediction_n": int(pd.to_numeric(sub["prediction_n"], errors="coerce").sum()),
                "high_risk_group_n": int(high_risk.sum()),
                "high_risk_group_fraction": float(high_risk.mean()) if len(sub) else math.nan,
                "mean_group_mae": weighted_mean(sub["mae"], sub["prediction_n"]),
                "max_group_mae": float(pd.to_numeric(sub["mae"], errors="coerce").max()),
                "mean_high_error_fraction": weighted_mean(sub["high_error_fraction"], sub["prediction_n"]),
                "max_source_target_range": float(pd.to_numeric(sub["source_target_range"], errors="coerce").max()),
                "max_y_true_range": float(pd.to_numeric(sub["y_true_range"], errors="coerce").max()),
                "max_conflict_risk_score": int(pd.to_numeric(sub["conflict_risk_score"], errors="coerce").max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["max_conflict_risk_score", "mean_high_error_fraction", "max_source_target_range", "prediction_n"],
        ascending=[False, False, False, False],
    )


def summarize_flags(groups: pd.DataFrame) -> pd.DataFrame:
    flag_columns = [
        "group_label_conflict",
        "source_conflict",
        "repeated_high_error",
        "in_domain_high_error",
        "multi_reference",
    ]
    rows: list[dict[str, Any]] = []
    for flag in flag_columns:
        if flag not in groups.columns:
            continue
        sub = groups[groups[flag].astype(bool)].copy()
        rows.append(
            {
                "flag": flag,
                "group_n": int(len(sub)),
                "prediction_n": int(pd.to_numeric(sub.get("prediction_n", pd.Series(dtype=float)), errors="coerce").sum()) if not sub.empty else 0,
                "mean_group_mae": weighted_mean(sub["mae"], sub["prediction_n"]) if not sub.empty else math.nan,
                "median_source_target_range": float(pd.to_numeric(sub.get("source_target_range", pd.Series(dtype=float)), errors="coerce").median()) if not sub.empty else math.nan,
                "max_source_target_range": float(pd.to_numeric(sub.get("source_target_range", pd.Series(dtype=float)), errors="coerce").max()) if not sub.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_case_tables(frame: pd.DataFrame, groups: pd.DataFrame, *, out_dir: Path, top_n: int) -> dict[str, str]:
    outputs: dict[str, str] = {}
    key_columns = [column for column in DEFAULT_GROUP_COLUMNS if column in frame.columns and column in groups.columns]
    high_risk_keys = groups[groups["conflict_risk_score"] > 0].loc[:, key_columns].drop_duplicates()
    flagged_rows = frame.merge(high_risk_keys, on=key_columns, how="inner") if key_columns and not high_risk_keys.empty else pd.DataFrame()
    case_columns = [
        "split_policy",
        "fold",
        "aggregate_id",
        "task_head",
        "chemical_label",
        "species_label",
        "target_basis",
        "effect_level_x",
        "duration_bin_h",
        "y_true",
        "y_pred",
        "abs_error",
        "is_high_error",
        "ad_tier",
        "target_value_min",
        "target_value_max",
        "target_value_count",
        "cross_reference_range",
        "cross_reference_conflict_flag",
        "source_reference_count",
        "source_test_count",
        "reference_numbers",
        "test_ids",
        "result_ids",
    ]
    if not flagged_rows.empty:
        path = out_dir / "high_risk_conflict_prediction_rows.csv"
        flagged_rows.sort_values("abs_error", ascending=False).loc[:, [c for c in case_columns if c in flagged_rows.columns]].head(top_n * 10).to_csv(path, index=False, encoding="utf-8-sig")
        outputs["high_risk_conflict_prediction_rows"] = str(path)
    top_groups = groups.head(top_n)
    path = out_dir / "top_conflict_groups_for_review.csv"
    review_columns = [
        *key_columns,
        "prediction_n",
        "unique_aggregate_n",
        "mae",
        "high_error_n",
        "high_error_fraction",
        "y_true_range",
        "source_target_range",
        "max_cross_reference_range",
        "target_value_count_sum",
        "source_reference_count_max",
        "unique_reference_count",
        "duration_bin_count",
        "effect_level_count",
        "ad_a_fraction",
        "risk_reasons",
        "quality_labels",
        "conversion_paths",
    ]
    top_groups.loc[:, [column for column in review_columns if column in top_groups.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    outputs["top_conflict_groups_for_review"] = str(path)
    return outputs


def plot_outputs(groups: pd.DataFrame, *, figures_dir: Path, style: dict[str, Any], top_n: int) -> dict[str, str]:
    outputs: dict[str, str] = {}
    outputs.update(plot_top_conflict_groups(groups.head(top_n), figures_dir / "top_conflict_groups", style=style))
    task_summary = summarize_group_dimensions(groups, ["task_head"]) if "task_head" in groups.columns else pd.DataFrame()
    outputs.update(plot_task_conflict_summary(task_summary.head(top_n), figures_dir / "task_conflict_summary", style=style))
    flag_summary = summarize_flags(groups)
    outputs.update(plot_flag_summary(flag_summary, figures_dir / "conflict_flag_summary", style=style))
    manifest = pd.DataFrame([{"figure": key, "path": value} for key, value in sorted(outputs.items())])
    path = figures_dir / "figure_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    outputs["figure_manifest"] = str(path)
    return outputs


def plot_top_conflict_groups(groups: pd.DataFrame, out_base: Path, *, style: dict[str, Any]) -> dict[str, str]:
    if groups.empty:
        return {}
    data = groups.sort_values(["conflict_risk_score", "source_target_range", "mae"], ascending=True).copy()
    data["label"] = data.apply(group_label, axis=1).map(lambda value: shorten(value, 64))
    fig_height = max(4.0, 0.24 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    color_values = pd.to_numeric(data["conflict_risk_score"], errors="coerce").fillna(0)
    bars = ax.barh(data["label"], pd.to_numeric(data["source_target_range"], errors="coerce"), color=plt.cm.inferno(np.clip(color_values / max(color_values.max(), 1), 0, 1)))
    ax.set_xlabel("Source/observed pTox range within group")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    style_axes(ax, style)
    for bar, mae in zip(bars, data["mae"]):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"MAE={float(mae):.2f}", va="center", fontsize=7)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def plot_task_conflict_summary(summary: pd.DataFrame, out_base: Path, *, style: dict[str, Any]) -> dict[str, str]:
    if summary.empty:
        return {}
    data = summary.sort_values("high_risk_group_fraction", ascending=True).copy()
    fig_height = max(3.8, 0.22 * len(data) + 1.3)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    ax.barh(data["task_head"].astype(str), data["high_risk_group_fraction"] * 100.0, color="#0072B2")
    ax.set_xlabel("High-risk conflict groups (%)")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    style_axes(ax, style)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def plot_flag_summary(summary: pd.DataFrame, out_base: Path, *, style: dict[str, Any]) -> dict[str, str]:
    if summary.empty:
        return {}
    data = summary.sort_values("group_n", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(data["flag"], data["group_n"], color="#009E73")
    ax.set_xlabel("Group count")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    style_axes(ax, style)
    save_figure(fig, out_base)
    return {f"figure_{out_base.name}": str(out_base.with_suffix(".png"))}


def group_label(row: pd.Series) -> str:
    return f"{row.get('task_head', '')} | {row.get('chemical_label', '')} | {row.get('species_label', '')}"


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")}


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def parse_bool(value: Any) -> bool:
    text = clean_text(value).lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def any_bool(values: pd.Series | None) -> bool:
    if values is None:
        return False
    return bool(values.map(parse_bool).any())


def fraction_equal(values: pd.Series | None, target: str) -> float:
    if values is None or len(values) == 0:
        return math.nan
    return float(values.map(clean_text).eq(target).mean())


def max_numeric(values: pd.Series | None) -> float:
    if values is None:
        return math.nan
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max()) if numeric.notna().any() else math.nan


def sum_numeric(values: pd.Series | None) -> float:
    if values is None:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.sum()) if numeric.notna().any() else 0.0


def nunique_numeric(values: pd.Series | None) -> int:
    if values is None:
        return 0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return int(numeric.nunique())


def nunique_nonempty(values: pd.Series | None) -> int:
    if values is None:
        return 0
    cleaned = values.map(clean_text)
    return int(cleaned[cleaned.ne("")].nunique())


def mode_text(values: pd.Series | None) -> str:
    if values is None:
        return "unknown"
    cleaned = values.map(clean_text)
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return "unknown"
    return str(cleaned.value_counts().index[0])


def join_unique(values: pd.Series | None, *, limit: int = 12) -> str:
    if values is None:
        return ""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    extra = max(len(seen) - len(output), 0)
    return ";".join(output) + (f";...(+{extra})" if extra else "")


def unique_token_count(values: pd.Series | None) -> int:
    if values is None:
        return 0
    tokens: set[str] = set()
    for value in values:
        for token in parse_tokens(value):
            tokens.add(token)
    return len(tokens)


def parse_tokens(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [clean_text(item) for item in parsed if clean_text(item)]
    return [token for token in re.split(r"[;,|]\s*", text) if token]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    weight = pd.to_numeric(weights, errors="coerce")
    mask = numeric.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return math.nan
    return float((numeric[mask] * weight[mask]).sum() / weight[mask].sum())


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


def has_value(series: pd.Series) -> pd.Series:
    return series.map(clean_text).ne("")


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
