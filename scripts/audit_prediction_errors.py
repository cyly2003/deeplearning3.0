from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager


DEFAULT_PREDICTION_ROWS = (
    "outputs/experiments/v1_2_23_random_mainline_cst_ad/"
    "random_mainline_cst_ad_prediction_rows.csv"
)
DEFAULT_DB = "outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"
DEFAULT_SOURCE_TABLE = "aggregated_task_records_aquatic_soil_ptox_qc"
DEFAULT_CHEMICAL_METADATA_TABLE = "target_records"
DEFAULT_OUT_DIR = "outputs/experiments/v1_2_25_prediction_error_audit"
DEFAULT_STYLE = "style_journal_clean_v1.yaml"
METADATA_COLUMNS = (
    "aggregate_id",
    "value_quality",
    "unit_conversion_source",
    "unit_conversion_confidence",
    "unit_conversion_note",
    "active_ingredient_basis",
    "acid_equivalent_basis",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "measurement",
    "tox_value_source",
    "target_status",
    "excluded_reason",
    "task_excluded_reason",
)
CHEMICAL_METADATA_COLUMNS = (
    "cas_number",
    "dtxsid",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "chemical_class_source",
    "chemical_class_confidence",
    "chemical_class_review_status",
)
TIER_ORDER = ["AD-A", "AD-B", "AD-C", "AD-D", "unknown"]
TIER_COLORS = {
    "AD-A": "#009E73",
    "AD-B": "#0072B2",
    "AD-C": "#E69F00",
    "AD-D": "#D55E00",
    "unknown": "#777777",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit prediction errors by task, chemical, species, AD tier, and "
            "data-quality labels. Writes CSV cross-tabs and publication-style figures."
        )
    )
    parser.add_argument("--prediction-rows", default=DEFAULT_PREDICTION_ROWS)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--chemical-metadata-table", default=DEFAULT_CHEMICAL_METADATA_TABLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--style", default=DEFAULT_STYLE)
    parser.add_argument("--split-part", default="test")
    parser.add_argument(
        "--task-prefixes",
        default="",
        help="Optional comma-separated task prefixes to keep, e.g. ECx,NOEC,LOEC. Empty keeps all tasks.",
    )
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--high-error-absolute", type=float, default=1.0)
    parser.add_argument("--min-task-n", type=int, default=5)
    parser.add_argument("--min-chemical-n", type=int, default=3)
    parser.add_argument("--min-species-n", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--skip-db-metadata", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_path = Path(args.prediction_rows)
    db_path = Path(args.db) if args.db else None
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    style = _load_style(Path(args.style))
    _apply_matplotlib_style(style)

    rows = load_prediction_rows(
        prediction_path,
        split_part=args.split_part,
        task_prefixes=parse_prefixes(args.task_prefixes),
    )
    if not args.skip_db_metadata and db_path is not None and db_path.exists():
        rows = add_db_metadata(rows, db_path=db_path, source_table=str(args.source_table))
        rows = add_chemical_metadata(rows, db_path=db_path, table=str(args.chemical_metadata_table))
    rows = prepare_audit_rows(
        rows,
        high_error_quantile=float(args.high_error_quantile),
        high_error_absolute=float(args.high_error_absolute),
    )

    audited_path = out_dir / "audited_prediction_rows.csv"
    rows.to_csv(audited_path, index=False, encoding="utf-8-sig")

    outputs: dict[str, str] = {"audited_prediction_rows": str(audited_path)}
    outputs.update(
        write_summary_tables(
            rows,
            out_dir=out_dir,
            min_task_n=int(args.min_task_n),
            min_chemical_n=int(args.min_chemical_n),
            min_species_n=int(args.min_species_n),
            top_n=int(args.top_n),
        )
    )
    outputs.update(plot_audit_figures(rows, figures_dir=figures_dir, style=style, top_n=int(args.top_n)))

    manifest = {
        "prediction_rows": str(prediction_path),
        "db": str(db_path) if db_path else None,
        "source_table": str(args.source_table),
        "chemical_metadata_table": str(args.chemical_metadata_table),
        "out_dir": str(out_dir),
        "split_part": str(args.split_part),
        "task_prefixes": parse_prefixes(args.task_prefixes),
        "rows": int(len(rows)),
        "high_error_quantile": float(args.high_error_quantile),
        "high_error_absolute": float(args.high_error_absolute),
        "high_error_threshold_used": float(rows["high_error_threshold_used"].iloc[0]) if len(rows) else math.nan,
        "metadata_join": {
            "enabled": bool(not args.skip_db_metadata and db_path is not None and db_path.exists()),
            "columns_requested": list(METADATA_COLUMNS),
            "columns_present": list(
                dict.fromkeys(
                    column
                    for column in (*METADATA_COLUMNS, *CHEMICAL_METADATA_COLUMNS)
                    if column in rows.columns or f"{column}_meta" in rows.columns
                )
            ),
        },
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    outputs["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "rows": int(len(rows)), "outputs": outputs}, ensure_ascii=False))


def load_prediction_rows(path: Path, *, split_part: str, task_prefixes: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    required = {"y_true", "y_pred"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns in {path}: {missing}")
    if "split_part" in frame.columns and split_part:
        frame = frame[normalize_text(frame["split_part"]).eq(str(split_part).strip().lower())].copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame = frame[frame["y_true"].notna() & frame["y_pred"].notna()].copy()
    if task_prefixes and "task_head" in frame.columns:
        task = frame["task_head"].astype(str)
        frame = frame[task.apply(lambda value: value.startswith(task_prefixes))].copy()
    if frame.empty:
        raise ValueError(f"No usable prediction rows after filtering: {path}")
    return frame.reset_index(drop=True)


def add_db_metadata(frame: pd.DataFrame, *, db_path: Path, source_table: str) -> pd.DataFrame:
    if "aggregate_id" not in frame.columns:
        return frame
    with sqlite3.connect(db_path) as conn:
        available = table_columns(conn, source_table)
        selected = [column for column in METADATA_COLUMNS if column in available]
        if "aggregate_id" not in selected:
            return frame
        quoted = ", ".join(quote_identifier(column) for column in selected)
        metadata = pd.read_sql_query(
            f"SELECT {quoted} FROM {quote_identifier(source_table)}",
            conn,
        )
    metadata = metadata.drop_duplicates("aggregate_id")
    left = frame.copy()
    left["aggregate_id"] = left["aggregate_id"].astype(str)
    metadata["aggregate_id"] = metadata["aggregate_id"].astype(str)
    overlapping = [column for column in metadata.columns if column in left.columns and column != "aggregate_id"]
    if overlapping:
        metadata = metadata.rename(columns={column: f"{column}_meta" for column in overlapping})
    merged = left.merge(metadata, on="aggregate_id", how="left")
    for column in overlapping:
        meta_column = f"{column}_meta"
        merged[column] = merged[column].where(has_value(merged[column]), merged[meta_column])
    return merged


def add_chemical_metadata(frame: pd.DataFrame, *, db_path: Path, table: str) -> pd.DataFrame:
    key_columns = [column for column in ("cas_number", "dtxsid") if column in frame.columns]
    if not key_columns:
        return frame
    with sqlite3.connect(db_path) as conn:
        available = table_columns(conn, table)
        selected = [column for column in CHEMICAL_METADATA_COLUMNS if column in available]
        if not any(column in selected for column in ("cas_number", "dtxsid")):
            return frame
        quoted = ", ".join(quote_identifier(column) for column in selected)
        metadata = pd.read_sql_query(f"SELECT {quoted} FROM {quote_identifier(table)}", conn)
    if metadata.empty:
        return frame
    merged = frame.copy()
    for key_column in key_columns:
        if key_column not in metadata.columns:
            continue
        lookup = build_metadata_lookup(metadata, key_column)
        if lookup.empty:
            continue
        merged[key_column] = merged[key_column].map(clean_text)
        merged = merged.merge(lookup, on=key_column, how="left", suffixes=("", "_chem"))
        for column in lookup.columns:
            if column == key_column:
                continue
            chem_column = f"{column}_chem"
            if column not in merged.columns:
                merged[column] = merged[chem_column]
            elif chem_column in merged.columns:
                merged[column] = merged[column].where(~needs_fill(merged[column]), merged[chem_column])
        chem_columns = [column for column in merged.columns if column.endswith("_chem")]
        merged = merged.drop(columns=chem_columns)
    return merged


def build_metadata_lookup(metadata: pd.DataFrame, key_column: str) -> pd.DataFrame:
    data = metadata.copy()
    data[key_column] = data[key_column].map(clean_text)
    data = data[data[key_column].ne("")].copy()
    if data.empty:
        return pd.DataFrame()
    value_columns = [column for column in CHEMICAL_METADATA_COLUMNS if column in data.columns and column != key_column]
    rows: list[dict[str, Any]] = []
    for key, sub in data.groupby(key_column, sort=False):
        row: dict[str, Any] = {key_column: key}
        for column in value_columns:
            row[column] = most_common_nonempty(sub[column])
        rows.append(row)
    return pd.DataFrame(rows)


def most_common_nonempty(values: pd.Series) -> str:
    cleaned = values.map(clean_text)
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts(sort=True)
    return str(counts.index[0])


def prepare_audit_rows(frame: pd.DataFrame, *, high_error_quantile: float, high_error_absolute: float) -> pd.DataFrame:
    data = frame.copy()
    data["residual"] = data["y_true"] - data["y_pred"]
    data["abs_error"] = data["residual"].abs()
    data["task_head"] = coalesce_text(data, ("task_head", "target_name"), default="unknown_task")
    data["task_family"] = coalesce_text(data, ("task_family", "target_family", "effect_family"), default="")
    data["task_family"] = data["task_family"].where(
        data["task_family"].ne(""),
        data["task_head"].astype(str).str.split("_", n=1).str[0],
    )
    data["chemical_label"] = chemical_label(data)
    data["species_label"] = coalesce_text(data, ("latin_name", "species_number"), default="unknown_species")
    data["ad_tier"] = coalesce_text(data, ("cst_ad_tier", "ad_tier"), default="unknown")
    data["ad_tier"] = data["ad_tier"].where(data["ad_tier"].isin(TIER_ORDER), "unknown")
    data["quality_label"] = build_quality_labels(data)
    threshold = high_error_threshold(data["abs_error"], quantile=high_error_quantile, absolute=high_error_absolute)
    data["high_error_threshold_used"] = threshold
    data["is_high_error"] = data["abs_error"] >= threshold
    data["error_direction"] = np.where(data["residual"] > 0, "underprediction", "overprediction")
    data.loc[data["residual"].abs() < 1e-12, "error_direction"] = "near_exact"
    return data


def write_summary_tables(
    frame: pd.DataFrame,
    *,
    out_dir: Path,
    min_task_n: int,
    min_chemical_n: int,
    min_species_n: int,
    top_n: int,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    table_specs = {
        "overall_error_summary": group_summary(frame, []),
        "high_mae_tasks": group_summary(frame, ["task_family", "task_head"], min_n=min_task_n)
        .sort_values(["mae", "p90_abs_error", "n"], ascending=[False, False, False])
        .head(top_n),
        "high_mae_chemicals": group_summary(
            frame,
            ["chemical_label", "cas_number", "dtxsid", "chemical_class_l1", "chemical_class_l2"],
            min_n=min_chemical_n,
        )
        .sort_values(["mae", "high_error_fraction", "n"], ascending=[False, False, False])
        .head(top_n),
        "high_mae_species": group_summary(
            frame,
            ["species_label", "family", "genus", "taxon_group_l1", "taxon_group_l2"],
            min_n=min_species_n,
        )
        .sort_values(["mae", "high_error_fraction", "n"], ascending=[False, False, False])
        .head(top_n),
        "ad_tier_summary": group_summary(frame, ["ad_tier"]).sort_values("ad_tier"),
        "task_family_summary": group_summary(frame, ["task_family"]).sort_values("mae", ascending=False),
        "data_quality_label_summary": group_summary(frame, ["quality_label"]).sort_values("mae", ascending=False),
        "error_direction_summary": group_summary(frame, ["error_direction"]).sort_values("n", ascending=False),
    }
    for name, table in table_specs.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = str(path)

    dimension_summary = summarize_quality_dimensions(frame)
    dimension_path = out_dir / "quality_dimension_summary.csv"
    dimension_summary.to_csv(dimension_path, index=False, encoding="utf-8-sig")
    outputs["quality_dimension_summary"] = str(dimension_path)

    cross_specs = {
        "task_by_ad_tier_mae": metric_pivot(frame, index="task_head", columns="ad_tier", value="abs_error", metric="mean"),
        "task_by_ad_tier_n": metric_pivot(frame, index="task_head", columns="ad_tier", value="abs_error", metric="size"),
        "task_by_quality_label_mae": metric_pivot(frame, index="task_head", columns="quality_label", value="abs_error", metric="mean"),
        "task_by_quality_label_n": metric_pivot(frame, index="task_head", columns="quality_label", value="abs_error", metric="size"),
        "ad_tier_by_quality_label_mae": metric_pivot(frame, index="ad_tier", columns="quality_label", value="abs_error", metric="mean"),
        "task_family_by_quality_label_mae": metric_pivot(frame, index="task_family", columns="quality_label", value="abs_error", metric="mean"),
        "task_family_by_ad_tier_mae": metric_pivot(frame, index="task_family", columns="ad_tier", value="abs_error", metric="mean"),
    }
    for name, table in cross_specs.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, encoding="utf-8-sig")
        outputs[name] = str(path)

    failures = select_failure_cases(frame, limit=max(top_n * 4, 100))
    failures_path = out_dir / "high_error_cases.csv"
    failures.to_csv(failures_path, index=False, encoding="utf-8-sig")
    outputs["high_error_cases"] = str(failures_path)
    return outputs


def group_summary(frame: pd.DataFrame, keys: list[str], *, min_n: int = 1) -> pd.DataFrame:
    data = frame.copy()
    for key in keys:
        if key not in data.columns:
            data[key] = "unknown"
        data[key] = data[key].astype("string").fillna("unknown").replace({"": "unknown"})
    groups: list[tuple[Any, pd.DataFrame]]
    if keys:
        groups = list(data.groupby(keys, dropna=False, sort=True))
    else:
        groups = [((), data)]
    rows: list[dict[str, Any]] = []
    for key_value, sub in groups:
        if len(sub) < min_n:
            continue
        if not isinstance(key_value, tuple):
            key_value = (key_value,)
        row = {key: value for key, value in zip(keys, key_value)}
        row.update(metrics_for_frame(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = pd.to_numeric(frame["y_true"], errors="coerce")
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce")
    residual = y_true - y_pred
    abs_error = residual.abs()
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = residual[valid]
    abs_error = abs_error[valid]
    n = int(len(abs_error))
    if n == 0:
        return {
            "n": 0,
            "r2": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "median_abs_error": math.nan,
            "p90_abs_error": math.nan,
            "max_abs_error": math.nan,
            "mean_signed_error": math.nan,
            "high_error_n": 0,
            "high_error_fraction": math.nan,
            "task_count": 0,
            "chemical_count": 0,
            "species_count": 0,
            "y_true_min": math.nan,
            "y_true_max": math.nan,
        }
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - float(y_true.mean())) ** 2).sum())
    high_error = frame.loc[valid, "is_high_error"] if "is_high_error" in frame.columns else pd.Series(False, index=abs_error.index)
    return {
        "n": n,
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / n),
        "mae": float(abs_error.mean()),
        "median_abs_error": float(abs_error.median()),
        "p90_abs_error": float(abs_error.quantile(0.90)),
        "max_abs_error": float(abs_error.max()),
        "mean_signed_error": float(residual.mean()),
        "high_error_n": int(high_error.fillna(False).astype(bool).sum()),
        "high_error_fraction": float(high_error.fillna(False).astype(bool).mean()),
        "task_count": nunique_nonempty(frame.get("task_head")),
        "chemical_count": nunique_nonempty(frame.get("chemical_label")),
        "species_count": nunique_nonempty(frame.get("species_label")),
        "y_true_min": float(y_true.min()),
        "y_true_max": float(y_true.max()),
    }


def summarize_quality_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        "quality_label",
        "value_quality",
        "unit_conversion_confidence",
        "toxicity_bin_status",
        "toxicity_bin_label",
        "toxicity_bin_boundary_flag",
        "target_basis",
        "conversion_path",
        "chemical_class_l1",
        "chemical_class_l2",
        "active_ingredient_basis",
        "acid_equivalent_basis",
    ]
    rows: list[pd.DataFrame] = []
    for dimension in dimensions:
        if dimension not in frame.columns:
            continue
        summary = group_summary(frame.rename(columns={dimension: "quality_value"}), ["quality_value"])
        if summary.empty:
            continue
        summary.insert(0, "quality_dimension", dimension)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metric_pivot(frame: pd.DataFrame, *, index: str, columns: str, value: str, metric: str) -> pd.DataFrame:
    data = frame.copy()
    for column in (index, columns):
        if column not in data.columns:
            data[column] = "unknown"
        data[column] = data[column].astype("string").fillna("unknown").replace({"": "unknown"})
    values = pd.to_numeric(data[value], errors="coerce")
    data[value] = values
    return data.pivot_table(index=index, columns=columns, values=value, aggfunc=metric, dropna=False)


def select_failure_cases(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    columns = [
        "split_policy",
        "fold",
        "sample_id",
        "aggregate_id",
        "task_family",
        "task_head",
        "chemical_label",
        "cas_number",
        "dtxsid",
        "species_label",
        "family",
        "genus",
        "y_true",
        "y_pred",
        "residual",
        "abs_error",
        "error_direction",
        "ad_tier",
        "cst_chemical_tier",
        "cst_species_tier",
        "cst_task_tier",
        "quality_label",
        "value_quality",
        "unit_conversion_confidence",
        "toxicity_bin_status",
        "toxicity_bin_boundary_flag",
        "target_basis",
        "conversion_path",
        "chemical_class_l1",
        "chemical_class_l2",
    ]
    present = [column for column in columns if column in frame.columns]
    return frame.sort_values("abs_error", ascending=False).loc[:, present].head(limit).reset_index(drop=True)


def plot_audit_figures(frame: pd.DataFrame, *, figures_dir: Path, style: dict[str, Any], top_n: int) -> dict[str, str]:
    outputs: dict[str, str] = {}
    task = group_summary(frame, ["task_head"], min_n=5).sort_values("mae", ascending=False).head(top_n)
    chemical = group_summary(frame, ["chemical_label"], min_n=3).sort_values("mae", ascending=False).head(min(top_n, 20))
    species = group_summary(frame, ["species_label"], min_n=3).sort_values("mae", ascending=False).head(min(top_n, 20))
    ad = group_summary(frame, ["ad_tier"]).sort_values("ad_tier")
    quality = group_summary(frame, ["quality_label"]).sort_values("mae", ascending=False).head(min(top_n, 20))

    outputs.update(plot_horizontal_bar(task, label_column="task_head", out_base=figures_dir / "top_task_mae", style=style))
    outputs.update(plot_horizontal_bar(chemical, label_column="chemical_label", out_base=figures_dir / "top_chemical_mae", style=style))
    outputs.update(plot_horizontal_bar(species, label_column="species_label", out_base=figures_dir / "top_species_mae", style=style))
    outputs.update(plot_vertical_tier_bars(ad, out_base=figures_dir / "ad_tier_error_profile", style=style))
    outputs.update(plot_horizontal_bar(quality, label_column="quality_label", out_base=figures_dir / "data_quality_label_mae", style=style))
    outputs.update(
        plot_heatmap(
            metric_pivot(frame, index="task_head", columns="ad_tier", value="abs_error", metric="mean"),
            out_base=figures_dir / "task_by_ad_tier_mae_heatmap",
            style=style,
            top_n=top_n,
            ylabel="Task head",
            xlabel="CST-AD tier",
            colorbar_label="MAE on pTox log10 scale",
        )
    )
    outputs.update(
        plot_heatmap(
            metric_pivot(frame, index="task_head", columns="quality_label", value="abs_error", metric="mean"),
            out_base=figures_dir / "task_by_quality_label_mae_heatmap",
            style=style,
            top_n=min(top_n, 18),
            ylabel="Task head",
            xlabel="Data-quality label",
            colorbar_label="MAE on pTox log10 scale",
        )
    )
    manifest = pd.DataFrame(
        [{"figure": name, "path": path} for name, path in sorted(outputs.items())]
    )
    manifest_path = figures_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    outputs["figure_manifest"] = str(manifest_path)
    return outputs


def plot_horizontal_bar(frame: pd.DataFrame, *, label_column: str, out_base: Path, style: dict[str, Any]) -> dict[str, str]:
    if frame.empty or label_column not in frame.columns:
        return {}
    data = frame.sort_values("mae", ascending=True).copy()
    data[label_column] = data[label_column].astype(str).map(lambda value: shorten_label(value, 58))
    fig_height = max(3.2, 0.22 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    colors = data["high_error_fraction"].fillna(0.0).to_numpy(dtype=float)
    bars = ax.barh(data[label_column], data["mae"], color=plt.cm.viridis(np.clip(colors, 0, 1)))
    ax.set_xlabel("MAE on pTox log10 scale")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    _style_axes(ax, style)
    for bar, n in zip(bars, data["n"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(n)}",
            va="center",
            ha="left",
            fontsize=float(style.get("font_sizes_pt", {}).get("annotation", 7.0)),
        )
    save_figure(fig, out_base)
    return {out_base.name: str(out_base.with_suffix(".png"))}


def plot_vertical_tier_bars(frame: pd.DataFrame, *, out_base: Path, style: dict[str, Any]) -> dict[str, str]:
    if frame.empty:
        return {}
    data = frame.copy()
    data["ad_tier"] = pd.Categorical(data["ad_tier"], categories=TIER_ORDER, ordered=True)
    data = data.sort_values("ad_tier")
    fig, axes = plt.subplots(1, 2, figsize=_figsize_inches(style, "double_column_wide", default=(7.0, 4.1)))
    labels = data["ad_tier"].astype(str).tolist()
    colors = [TIER_COLORS.get(label, "#777777") for label in labels]
    axes[0].bar(labels, data["mae"], color=colors)
    axes[0].set_ylabel("MAE on pTox log10 scale")
    axes[0].set_xlabel("CST-AD tier")
    axes[1].bar(labels, data["high_error_fraction"] * 100.0, color=colors)
    axes[1].set_ylabel("High-error rows (%)")
    axes[1].set_xlabel("CST-AD tier")
    for ax in axes:
        ax.grid(True, axis="y")
        _style_axes(ax, style)
    fig.tight_layout()
    save_figure(fig, out_base)
    return {out_base.name: str(out_base.with_suffix(".png"))}


def plot_heatmap(
    table: pd.DataFrame,
    *,
    out_base: Path,
    style: dict[str, Any],
    top_n: int,
    ylabel: str,
    xlabel: str,
    colorbar_label: str,
) -> dict[str, str]:
    if table.empty:
        return {}
    row_scores = table.mean(axis=1, skipna=True).sort_values(ascending=False)
    data = table.loc[row_scores.head(top_n).index].copy()
    if data.shape[1] > 14:
        col_scores = data.mean(axis=0, skipna=True).sort_values(ascending=False)
        data = data.loc[:, col_scores.head(14).index]
    if data.empty:
        return {}
    values = data.to_numpy(dtype=float)
    fig_width = max(5.5, 0.34 * data.shape[1] + 3.0)
    fig_height = max(4.0, 0.24 * data.shape[0] + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="viridis", aspect="auto")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([shorten_label(value, 38) for value in data.index.astype(str)])
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([shorten_label(value, 32) for value in data.columns.astype(str)], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    cbar = fig.colorbar(image, ax=ax, pad=0.012)
    cbar.set_label(colorbar_label)
    _style_axes(ax, style)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, out_base)
    return {out_base.name: str(out_base.with_suffix(".png"))}


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


def build_quality_labels(frame: pd.DataFrame) -> pd.Series:
    labels: list[str] = []
    for _, row in frame.iterrows():
        parts: list[str] = []
        value_quality = clean_text(row_value(row, "value_quality"))
        if value_quality:
            parts.append(f"value={value_quality}")
        unit_conf = clean_text(row_value(row, "unit_conversion_confidence"))
        unit_source = clean_text(row_value(row, "unit_conversion_source"))
        if unit_conf:
            parts.append(f"unit={unit_conf}")
        elif unit_source:
            parts.append(f"unit_source={unit_source}")
        bin_status = clean_text(row_value(row, "toxicity_bin_status"))
        if bin_status:
            parts.append(f"bin={bin_status}")
        boundary = clean_text(row_value(row, "toxicity_bin_boundary_flag"))
        if boundary in {"1", "1.0", "true", "yes"}:
            parts.append("bin_boundary")
        class_l2 = clean_text(row_value(row, "chemical_class_l2"))
        class_l1 = clean_text(row_value(row, "chemical_class_l1"))
        if class_l2 and class_l2 not in {"unknown", "unclassified"}:
            parts.append(f"chem={class_l2}")
        elif class_l1 and class_l1 not in {"unknown", "unclassified"}:
            parts.append(f"chem={class_l1}")
        if clean_text(row_value(row, "active_ingredient_basis")) in {"1", "1.0", "true", "yes"}:
            parts.append("active_ingredient_basis")
        if clean_text(row_value(row, "acid_equivalent_basis")) in {"1", "1.0", "true", "yes"}:
            parts.append("acid_equivalent_basis")
        conversion = clean_text(row_value(row, "conversion_path"))
        if not unit_conf and conversion:
            parts.append(f"conversion={conversion}")
        labels.append("|".join(parts) if parts else "quality_unknown")
    return pd.Series(labels, index=frame.index, dtype="object")


def chemical_label(frame: pd.DataFrame) -> pd.Series:
    name = coalesce_text(frame, ("chemical_name", "cas_number", "dtxsid", "smiles"), default="unknown_chemical")
    cas = coalesce_text(frame, ("cas_number",), default="")
    return name.where(cas.eq(""), name + " [" + cas + "]")


def coalesce_text(frame: pd.DataFrame, columns: tuple[str, ...], *, default: str) -> pd.Series:
    output = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].map(clean_text)
        output = output.where(output.ne(""), values)
    return output.where(output.ne(""), default)


def high_error_threshold(values: pd.Series, *, quantile: float, absolute: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float(absolute)
    q = float(numeric.quantile(float(quantile)))
    return max(float(absolute), q)


def parse_prefixes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")}


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def has_value(series: pd.Series) -> pd.Series:
    return series.map(clean_text).ne("")


def needs_fill(series: pd.Series) -> pd.Series:
    normalized = normalize_text(series)
    return normalized.isin({"", "unknown", "unclassified", "none"})


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower().replace({"<na>": "", "nan": "", "none": ""})


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


def row_value(row: pd.Series, column: str) -> Any:
    if column in row.index:
        return row[column]
    meta = f"{column}_meta"
    if meta in row.index:
        return row[meta]
    return ""


def nunique_nonempty(series: pd.Series | None) -> int:
    if series is None:
        return 0
    values = series.map(clean_text)
    return int(values[values.ne("")].nunique())


def shorten_label(value: Any, max_length: int) -> str:
    text = str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "..."


def _load_style(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _apply_matplotlib_style(style: dict[str, Any]) -> None:
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
    font_sizes = style.get("font_sizes_pt", {})
    if font_sizes:
        plt.rcParams["axes.labelsize"] = float(font_sizes.get("axis_label", 8.0))
        plt.rcParams["xtick.labelsize"] = float(font_sizes.get("tick_label", 7.0))
        plt.rcParams["ytick.labelsize"] = float(font_sizes.get("tick_label", 7.0))
        plt.rcParams["legend.fontsize"] = float(font_sizes.get("legend", 7.0))


def _style_axes(ax: Any, style: dict[str, Any]) -> None:
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


def _figsize_inches(style: dict[str, Any], key: str, *, default: tuple[float, float]) -> tuple[float, float]:
    sizes = style.get("figure_sizes_mm", {})
    if key not in sizes:
        return default
    width_mm, height_mm = sizes[key]
    return float(width_mm) / 25.4, float(height_mm) / 25.4


if __name__ == "__main__":
    main()
