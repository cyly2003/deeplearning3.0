from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.config import load_config
from qsar_tl.evaluation.splits import table_exists
from qsar_tl.training.baseline import add_duration_nonlinear_features
from qsar_tl.training.deep_experiment import (
    CATEGORICAL_COLUMNS,
    CONTEXT_NUMERIC_COLUMNS,
    MOLECULAR_DESCRIPTOR_NAMES,
)


FINGERPRINT_SUMMARY_COLUMNS = (
    "morgan_active_bits",
    "morgan_bit_density",
)
DEFAULT_MIN_TASK_ROWS = 200
DEFAULT_MAX_TASKS_PER_TABLE = 0
DEFAULT_HIST_BINS = 36


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    db_path = Path(args.db or config.get("data", {}).get("modeling_tables_db", "outputs/derived/modeling_dataset.sqlite"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_tables = args.source_table or discover_source_tables(db_path, config)
    if not source_tables:
        raise ValueError("No source tables were provided or discovered.")

    molecular_cache = load_molecular_cache(Path(args.molecular_cache or molecular_cache_path(config)))
    manifest_rows: list[dict[str, Any]] = []
    numeric_summary_frames: list[pd.DataFrame] = []
    fingerprint_summary_frames: list[pd.DataFrame] = []
    category_summary_frames: list[pd.DataFrame] = []

    with sqlite3.connect(db_path) as conn:
        for source_table in source_tables:
            if not table_exists(conn, source_table):
                print(f"[skip] missing table: {source_table}", flush=True)
                continue
            print(f"[table] {source_table}", flush=True)
            frame = pd.read_sql_query(f'SELECT * FROM "{source_table}"', conn)
            if frame.empty:
                print(f"[skip] empty table: {source_table}", flush=True)
                continue
            frame = add_duration_nonlinear_features(frame)
            frame = attach_molecular_features(frame, molecular_cache)
            task_column = "task_head" if "task_head" in frame.columns else None
            if task_column is None:
                frame["task_head"] = "default"
                task_column = "task_head"

            tasks = select_tasks(frame, task_column, args.min_task_rows, args.max_tasks_per_table)
            numeric_columns = select_numeric_columns(frame)
            categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in frame.columns]
            print(
                f"[table] rows={len(frame)} tasks={len(tasks)} numeric_features={len(numeric_columns)} "
                f"categorical_features={len(categorical_columns)}",
                flush=True,
            )

            table_numeric = summarize_numeric(frame, source_table, task_column, tasks, numeric_columns)
            table_fingerprint = summarize_numeric(
                frame,
                source_table,
                task_column,
                tasks,
                [column for column in FINGERPRINT_SUMMARY_COLUMNS if column in frame.columns],
            )
            if not table_fingerprint.empty:
                table_fingerprint["feature_role"] = "fingerprint_distribution_summary_not_direct_model_field"
            table_categories = summarize_categories(
                frame,
                source_table,
                task_column,
                tasks,
                categorical_columns,
                top_n=args.top_categories,
            )
            numeric_summary_frames.append(table_numeric)
            fingerprint_summary_frames.append(table_fingerprint)
            category_summary_frames.append(table_categories)

            table_dir = out_dir / sanitize_filename(source_table)
            table_dir.mkdir(parents=True, exist_ok=True)
            write_table_outputs(
                frame=frame,
                source_table=source_table,
                task_column=task_column,
                tasks=tasks,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                numeric_summary=table_numeric,
                category_summary=table_categories,
                table_dir=table_dir,
                hist_bins=args.hist_bins,
                max_category_panels=args.max_category_panels,
                manifest_rows=manifest_rows,
            )

    if numeric_summary_frames:
        numeric_all = pd.concat(numeric_summary_frames, ignore_index=True)
    else:
        numeric_all = pd.DataFrame()
    if category_summary_frames:
        category_all = pd.concat(category_summary_frames, ignore_index=True)
    else:
        category_all = pd.DataFrame()
    if fingerprint_summary_frames:
        fingerprint_all = pd.concat(fingerprint_summary_frames, ignore_index=True)
    else:
        fingerprint_all = pd.DataFrame()

    numeric_csv = out_dir / "numeric_feature_distribution_summary.csv"
    fingerprint_csv = out_dir / "fingerprint_distribution_summary.csv"
    category_csv = out_dir / "categorical_feature_frequency_summary.csv"
    manifest_csv = out_dir / "figure_manifest.csv"
    numeric_all.to_csv(numeric_csv, index=False, encoding="utf-8-sig")
    fingerprint_all.to_csv(fingerprint_csv, index=False, encoding="utf-8-sig")
    category_all.to_csv(category_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    write_model_variable_list(out_dir, numeric_all, category_all)
    write_markdown_report(out_dir, db_path, source_tables, numeric_all, category_all, manifest_rows)
    print(f"[done] {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile distributions and skewness of modeling features by source table and task head."
    )
    parser.add_argument("--config", default="configs/experiment.remote.easyai.yaml")
    parser.add_argument("--db", default=None)
    parser.add_argument("--source-table", action="append", default=[])
    parser.add_argument("--molecular-cache", default=None)
    parser.add_argument("--out-dir", default="outputs/audits/feature_distributions")
    parser.add_argument("--min-task-rows", type=int, default=DEFAULT_MIN_TASK_ROWS)
    parser.add_argument("--max-tasks-per-table", type=int, default=DEFAULT_MAX_TASKS_PER_TABLE)
    parser.add_argument("--hist-bins", type=int, default=DEFAULT_HIST_BINS)
    parser.add_argument("--top-categories", type=int, default=12)
    parser.add_argument("--max-category-panels", type=int, default=9)
    return parser.parse_args()


def discover_source_tables(db_path: Path, config: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    medium_units = config.get("targets", {}).get("medium_units", {})
    for medium_cfg in medium_units.values():
        if not isinstance(medium_cfg, dict):
            continue
        for key in ("recommended_source_table", "ptox_reference_table", "mg_kg_reference_table"):
            value = medium_cfg.get(key)
            if value:
                candidates.append(str(value))
    candidates.append("aggregated_task_records_aquatic_soil_ptox_qc")
    candidates.append("aggregated_task_records_qc")
    result: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for table in candidates:
            if table not in result and table_exists(conn, table):
                result.append(table)
    return result


def molecular_cache_path(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {})
    value = experiment.get("molecular_feature_cache")
    return str(value or "outputs/features/molecular_features_rdkit_morgan512.jsonl")


def load_molecular_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] molecular cache missing: {path}", flush=True)
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row.get("smiles", "")).strip()
            if not smiles:
                continue
            values: dict[str, Any] = {"smiles": smiles}
            descriptor_names = list(row.get("descriptor_names") or MOLECULAR_DESCRIPTOR_NAMES)
            descriptors = list(row.get("descriptors") or [])
            for idx, name in enumerate(descriptor_names):
                values[str(name)] = float(descriptors[idx]) if idx < len(descriptors) else np.nan
            fingerprint = row.get("fingerprint") or []
            if fingerprint:
                active_bits = float(np.sum(np.asarray(fingerprint, dtype=float) > 0))
                bit_density = active_bits / float(len(fingerprint))
            else:
                active_bits = np.nan
                bit_density = np.nan
            values["morgan_active_bits"] = active_bits
            values["morgan_bit_density"] = bit_density
            rows.append(values)
    cache = pd.DataFrame(rows).drop_duplicates(subset=["smiles"], keep="first")
    print(f"[cache] molecules={len(cache)} path={path}", flush=True)
    return cache


def attach_molecular_features(frame: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (*MOLECULAR_DESCRIPTOR_NAMES, *FINGERPRINT_SUMMARY_COLUMNS):
        if column not in result.columns:
            result[column] = np.nan
    if "smiles" not in result.columns or cache.empty:
        return result
    cache_columns = [
        column
        for column in (*MOLECULAR_DESCRIPTOR_NAMES, *FINGERPRINT_SUMMARY_COLUMNS)
        if column in cache.columns
    ]
    lookup = cache[["smiles", *cache_columns]].copy()
    lookup["smiles_key"] = lookup["smiles"].astype("string").fillna("").str.strip()
    lookup = lookup.drop(columns=["smiles"])
    keys = pd.DataFrame(
        {
            "_row_id": np.arange(len(result), dtype=int),
            "smiles_key": result["smiles"].astype("string").fillna("").str.strip().to_numpy(),
        }
    )
    merged = keys.merge(lookup, on="smiles_key", how="left").sort_values("_row_id")
    for column in cache_columns:
        result[column] = merged[column].to_numpy()
    return result


def select_tasks(
    frame: pd.DataFrame,
    task_column: str,
    min_rows: int,
    max_tasks: int,
) -> list[str]:
    counts = frame[task_column].astype("string").fillna("default").value_counts()
    counts = counts[counts >= int(min_rows)]
    if max_tasks and max_tasks > 0:
        counts = counts.head(max_tasks)
    return [str(task) for task in counts.index]


def select_numeric_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [*MOLECULAR_DESCRIPTOR_NAMES, *CONTEXT_NUMERIC_COLUMNS]
    selected: list[str] = []
    for column in candidates:
        if column not in frame.columns or column in selected:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            selected.append(column)
    return selected


def summarize_numeric(
    frame: pd.DataFrame,
    source_table: str,
    task_column: str,
    tasks: Iterable[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_values = frame[task_column].astype("string").fillna("default")
    for task in tasks:
        task_frame = frame[task_values == task]
        for column in numeric_columns:
            values = pd.to_numeric(task_frame[column], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(numeric_stats(source_table, task, column, len(task_frame), values, finite))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["skew_flag"] = result["skew"].abs() >= 1.0
        result["strong_skew_flag"] = result["skew"].abs() >= 2.0
        result["zero_inflated_flag"] = result["zero_rate"] >= 0.5
    return result


def numeric_stats(
    source_table: str,
    task: str,
    feature: str,
    task_rows: int,
    values: pd.Series,
    finite: pd.Series,
) -> dict[str, Any]:
    if finite.empty:
        return {
            "source_table": source_table,
            "task_head": task,
            "feature": feature,
            "task_rows": int(task_rows),
            "n": 0,
            "missing_rate": 1.0,
            "zero_rate": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "right_tail_ratio": np.nan,
        }
    quantiles = finite.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = float(quantiles.loc[0.75] - quantiles.loc[0.25])
    right_tail_ratio = np.nan
    if iqr > 0:
        right_tail_ratio = float((quantiles.loc[0.95] - quantiles.loc[0.5]) / iqr)
    return {
        "source_table": source_table,
        "task_head": task,
        "feature": feature,
        "task_rows": int(task_rows),
        "n": int(finite.shape[0]),
        "missing_rate": float(values.isna().mean()),
        "zero_rate": float((finite == 0).mean()),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.shape[0] > 1 else 0.0,
        "skew": float(finite.skew()) if finite.shape[0] > 2 else 0.0,
        "min": float(finite.min()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(finite.max()),
        "iqr": iqr,
        "right_tail_ratio": right_tail_ratio,
    }


def summarize_categories(
    frame: pd.DataFrame,
    source_table: str,
    task_column: str,
    tasks: Iterable[str],
    categorical_columns: list[str],
    *,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_values = frame[task_column].astype("string").fillna("default")
    for task in tasks:
        task_frame = frame[task_values == task]
        for column in categorical_columns:
            values = normalize_category(task_frame[column])
            counts = values.value_counts(dropna=False).head(top_n)
            total = max(int(values.shape[0]), 1)
            for rank, (category, count) in enumerate(counts.items(), start=1):
                rows.append(
                    {
                        "source_table": source_table,
                        "task_head": task,
                        "feature": column,
                        "rank": rank,
                        "category": str(category),
                        "count": int(count),
                        "fraction": float(count / total),
                        "task_rows": int(total),
                    }
                )
    return pd.DataFrame(rows)


def write_table_outputs(
    *,
    frame: pd.DataFrame,
    source_table: str,
    task_column: str,
    tasks: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    table_dir: Path,
    hist_bins: int,
    max_category_panels: int,
    manifest_rows: list[dict[str, Any]],
) -> None:
    task_values = frame[task_column].astype("string").fillna("default")
    for task in tasks:
        task_frame = frame[task_values == task]
        task_slug = sanitize_filename(task)
        numeric_path = table_dir / f"{task_slug}_numeric_distributions.png"
        plot_numeric_grid(
            task_frame,
            task,
            source_table,
            numeric_columns,
            numeric_summary[numeric_summary["task_head"] == task],
            numeric_path,
            hist_bins,
        )
        manifest_rows.append(
            {
                "source_table": source_table,
                "task_head": task,
                "figure_type": "numeric_distribution",
                "path": str(numeric_path.as_posix()),
            }
        )
        if categorical_columns:
            category_path = table_dir / f"{task_slug}_categorical_top_counts.png"
            plot_category_grid(
                category_summary[category_summary["task_head"] == task],
                task,
                source_table,
                category_path,
                max_panels=max_category_panels,
            )
            manifest_rows.append(
                {
                    "source_table": source_table,
                    "task_head": task,
                    "figure_type": "categorical_top_counts",
                    "path": str(category_path.as_posix()),
                }
            )


def plot_numeric_grid(
    frame: pd.DataFrame,
    task: str,
    source_table: str,
    numeric_columns: list[str],
    summary: pd.DataFrame,
    out_path: Path,
    hist_bins: int,
) -> None:
    columns = [column for column in numeric_columns if column in frame.columns]
    if not columns:
        return
    ncols = 4
    nrows = int(math.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    summary_by_feature = summary.set_index("feature") if not summary.empty else pd.DataFrame()
    for axis, column in zip(axes_flat, columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            axis.text(0.5, 0.5, "all missing", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.hist(finite, bins=hist_bins, color="#4C78A8", edgecolor="white", alpha=0.88)
        median = float(finite.median())
        axis.axvline(median, color="#D55E00", linewidth=1.2)
        row = summary_by_feature.loc[column] if column in summary_by_feature.index else None
        skew = float(row["skew"]) if row is not None else float(finite.skew())
        miss = float(row["missing_rate"]) if row is not None else float(values.isna().mean())
        title = f"{column}\nskew={skew:.2f}, missing={miss:.1%}"
        if abs(skew) >= 2:
            title += " *"
        axis.set_title(title, fontsize=8)
        axis.set_ylabel("Count", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
    for axis in axes_flat[len(columns) :]:
        axis.set_axis_off()
    fig.suptitle(wrap_title(f"{source_table} | {task} | numeric feature distributions"), fontsize=11)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_category_grid(
    summary: pd.DataFrame,
    task: str,
    source_table: str,
    out_path: Path,
    *,
    max_panels: int,
) -> None:
    if summary.empty:
        return
    features = list(summary["feature"].drop_duplicates())[:max_panels]
    ncols = 3
    nrows = int(math.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, feature in zip(axes_flat, features):
        sub = summary[summary["feature"] == feature].sort_values("rank", ascending=False)
        labels = [truncate_label(str(value), 24) for value in sub["category"]]
        axis.barh(labels, sub["fraction"], color="#59A14F")
        axis.set_title(feature, fontsize=8)
        axis.set_xlabel("Fraction", fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        axis.set_xlim(0, max(0.05, min(1.0, float(sub["fraction"].max()) * 1.15)))
    for axis in axes_flat[len(features) :]:
        axis.set_axis_off()
    fig.suptitle(wrap_title(f"{source_table} | {task} | categorical top counts"), fontsize=11)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def write_markdown_report(
    out_dir: Path,
    db_path: Path,
    source_tables: list[str],
    numeric_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    manifest_rows: list[dict[str, Any]],
) -> None:
    report_path = out_dir / "README.md"
    lines = [
        "# Modeling Feature Distribution Audit",
        "",
        f"- Database: `{db_path}`",
        f"- Source tables: {', '.join(f'`{table}`' for table in source_tables)}",
        f"- Numeric summary: `numeric_feature_distribution_summary.csv`",
        f"- Fingerprint summary: `fingerprint_distribution_summary.csv`",
        f"- Categorical summary: `categorical_feature_frequency_summary.csv`",
        f"- Actual model variable list: `model_variable_list.csv`",
        f"- Figure manifest: `figure_manifest.csv`",
        "",
        "## Strongest Numeric Skew Signals",
        "",
    ]
    if numeric_summary.empty:
        lines.append("No numeric feature summaries were generated.")
    else:
        top = (
            numeric_summary[numeric_summary["n"] > 2]
            .assign(abs_skew=lambda data: data["skew"].abs())
            .sort_values(["abs_skew", "missing_rate"], ascending=[False, False])
            .head(30)
        )
        lines.extend(markdown_table(top[
            [
                "source_table",
                "task_head",
                "feature",
                "task_rows",
                "missing_rate",
                "zero_rate",
                "skew",
                "median",
                "p95",
                "max",
            ]
        ]))
    lines.extend(["", "## Figure Index", ""])
    for row in manifest_rows:
        if row["figure_type"] != "numeric_distribution":
            continue
        lines.append(
            f"- `{row['source_table']}` / `{row['task_head']}`: "
            f"[numeric]({Path(row['path']).name if Path(row['path']).parent == out_dir else row['path']})"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(
        "- Numeric skew uses pandas sample skewness. `|skew| >= 1` is flagged as skewed; "
        "`|skew| >= 2` is flagged as strongly skewed in the CSV."
    )
    lines.append(
        "- Morgan fingerprints enter the deep model as 512 binary bits. They are not included in the numeric "
        "feature panels; `fingerprint_distribution_summary.csv` summarizes active bit count and density only."
    )
    lines.append(
        "- Categorical fields are summarized by top category fractions because skewness is not defined for labels."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_variable_list(out_dir: Path, numeric_summary: pd.DataFrame, category_summary: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for feature in sorted(numeric_summary["feature"].dropna().unique().tolist()) if not numeric_summary.empty else []:
        rows.append({"feature": feature, "feature_group": "numeric_molecular_or_context", "model_input": "molecular_numeric"})
    for feature in sorted(category_summary["feature"].dropna().unique().tolist()) if not category_summary.empty else []:
        rows.append({"feature": feature, "feature_group": "categorical_context", "model_input": "categorical_embedding"})
    rows.append(
        {
            "feature": "morgan_fingerprint_512_bits",
            "feature_group": "molecular_fingerprint",
            "model_input": "fingerprint",
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "model_variable_list.csv", index=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    return [header, separator, *body]


def normalize_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().replace("", "<missing>")


def sanitize_filename(value: str) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    result = "".join(allowed).strip("_")
    return result[:120] or "item"


def truncate_label(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def wrap_title(value: str) -> str:
    return "\n".join(textwrap.wrap(value, width=95))


if __name__ == "__main__":
    main()
