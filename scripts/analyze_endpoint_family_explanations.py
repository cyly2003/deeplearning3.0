from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.evaluation.metrics import regression_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze ECx/NOEC/LOEC errors and run representative SHAP explanations.")
    parser.add_argument("--run-dir", required=True, help="Deep run directory containing predictions.csv and best_model.pt.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--split-part", default="test")
    parser.add_argument("--families", nargs="+", default=["ECx", "NOEC", "LOEC"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-explain-tasks-per-family", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--shap-rows", type=int, default=32)
    parser.add_argument("--shap-max-evals", type=int, default=1200)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--continue-on-explain-error", action="store_true")
    parser.add_argument("--style", default="style_journal_clean_v1.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "endpoint_family_explanations")
    out_dir.mkdir(parents=True, exist_ok=True)
    style = _load_style(Path(args.style))
    _apply_style(style)

    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions["endpoint_family"] = predictions["task_head"].astype(str).str.split("_", n=1).str[0]
    predictions["residual"] = pd.to_numeric(predictions["residual"], errors="coerce")
    predictions["abs_error"] = pd.to_numeric(predictions["abs_error"], errors="coerce")
    predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
    predictions["y_pred"] = pd.to_numeric(predictions["y_pred"], errors="coerce")

    selected = predictions[
        predictions["split_part"].astype(str).eq(args.split_part)
        & predictions["endpoint_family"].isin(args.families)
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split_part={args.split_part!r}, families={args.families!r}.")

    family_metrics = _metrics_by(selected, ["endpoint_family"])
    task_metrics = _metrics_by(selected, ["endpoint_family", "task_head"])
    family_metrics.to_csv(out_dir / "endpoint_family_metrics.csv", index=False, encoding="utf-8-sig")
    task_metrics.to_csv(out_dir / "endpoint_task_metrics.csv", index=False, encoding="utf-8-sig")
    _plot_abs_error_box(selected, style, out_dir / "abs_error_by_endpoint_family")
    _plot_residual_hist(selected, style, out_dir / "residual_distribution_by_endpoint_family")

    explain_tasks = _select_explain_tasks(
        selected,
        families=args.families,
        max_per_family=args.max_explain_tasks_per_family,
    )
    explain_tasks.to_csv(out_dir / "selected_explain_tasks.csv", index=False, encoding="utf-8-sig")

    explain_rows: list[dict[str, Any]] = []
    if not args.skip_shap:
        for row in explain_tasks.to_dict("records"):
            task_head = str(row["task_head"])
            task_out = out_dir / "shap" / task_head
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "explain_deep_model.py"),
                "--config",
                args.config,
                "--run-dir",
                str(run_dir),
                "--db",
                args.db,
                "--source-table",
                args.source_table,
                "--task-head",
                task_head,
                "--split-part",
                args.split_part,
                "--max-rows",
                str(args.max_rows),
                "--shap-rows",
                str(args.shap_rows),
                "--shap-max-evals",
                str(args.shap_max_evals),
                "--out-dir",
                str(task_out),
                "--style",
                args.style,
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
            explain_rows.append(
                {
                    "endpoint_family": row["endpoint_family"],
                    "task_head": task_head,
                    "n": int(row["n"]),
                    "out_dir": str(task_out),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            if result.returncode != 0 and not args.continue_on_explain_error:
                (out_dir / "explain_failures.json").write_text(
                    json.dumps(explain_rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(f"Explanation failed for {task_head}: {result.stderr}")

    (out_dir / "explain_runs.json").write_text(
        json.dumps(explain_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"family_metrics={out_dir / 'endpoint_family_metrics.csv'}")
    print(f"task_metrics={out_dir / 'endpoint_task_metrics.csv'}")
    print(f"selected_tasks={out_dir / 'selected_explain_tasks.csv'}")
    print(f"figures={out_dir}")


def _metrics_by(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        finite = group[np.isfinite(group["y_true"]) & np.isfinite(group["y_pred"])].copy()
        metrics = regression_metrics(finite["y_true"].tolist(), finite["y_pred"].tolist()) if not finite.empty else {}
        row = {column: value for column, value in zip(columns, key)}
        row.update(
            {
                "n": int(len(finite)),
                "r2": _finite_or_blank(metrics.get("r2")),
                "rmse": _finite_or_blank(metrics.get("rmse")),
                "mae": _finite_or_blank(metrics.get("mae")),
                "bias_mean": _finite_or_blank(float(finite["residual"].mean()) if not finite.empty else math.nan),
                "abs_error_median": _finite_or_blank(float(finite["abs_error"].median()) if not finite.empty else math.nan),
                "abs_error_p90": _finite_or_blank(float(finite["abs_error"].quantile(0.90)) if not finite.empty else math.nan),
                "abs_error_p95": _finite_or_blank(float(finite["abs_error"].quantile(0.95)) if not finite.empty else math.nan),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values([*columns])


def _select_explain_tasks(frame: pd.DataFrame, *, families: list[str], max_per_family: int) -> pd.DataFrame:
    rows = []
    for family in families:
        subset = frame[frame["endpoint_family"].eq(family)]
        counts = subset["task_head"].value_counts()
        for task_head, n in counts.head(max(1, max_per_family)).items():
            rows.append({"endpoint_family": family, "task_head": task_head, "n": int(n)})
    return pd.DataFrame(rows)


def _plot_abs_error_box(frame: pd.DataFrame, style: dict[str, Any], out_base: Path) -> None:
    families = list(dict.fromkeys(frame["endpoint_family"].tolist()))
    data = [frame.loc[frame["endpoint_family"].eq(family), "abs_error"].dropna().to_numpy() for family in families]
    fig, ax = plt.subplots(figsize=_figsize(style, "single_column_wide", (3.35, 2.8)))
    ax.boxplot(data, tick_labels=families, showfliers=False)
    ax.set_xlabel("Endpoint family")
    ax.set_ylabel("Absolute error")
    _style_axes(ax)
    _save(fig, out_base)


def _plot_residual_hist(frame: pd.DataFrame, style: dict[str, Any], out_base: Path) -> None:
    families = list(dict.fromkeys(frame["endpoint_family"].tolist()))
    fig, axes = plt.subplots(len(families), 1, figsize=_figsize(style, "single_column_tall", (3.35, 1.7 * len(families))), sharex=True)
    if len(families) == 1:
        axes = [axes]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for idx, (ax, family) in enumerate(zip(axes, families)):
        residual = frame.loc[frame["endpoint_family"].eq(family), "residual"].dropna().to_numpy()
        if residual.size:
            ax.hist(residual, bins=40, color=colors[idx % len(colors)], alpha=0.82)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_ylabel(family)
        _style_axes(ax)
    axes[-1].set_xlabel("Residual (observed - predicted)")
    _save(fig, out_base)


def _finite_or_blank(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _load_style(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _apply_style(style: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": _preferred_font(style),
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "savefig.dpi": int(style.get("dpi", 300)),
        }
    )


def _preferred_font(style: dict[str, Any]) -> str:
    candidates = [
        (style.get("fonts", {}).get("english", {}) or {}).get("family_name"),
        style.get("font", {}).get("english"),
        "Arial",
        "DejaVu Sans",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            font_manager.findfont(str(candidate), fallback_to_default=False)
            return str(candidate)
        except Exception:
            continue
    return "DejaVu Sans"


def _figsize(style: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = style.get("figure_sizes", {}).get(key)
    if not value:
        return default
    return (float(value[0]), float(value[1]))


def _style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)


def _save(fig: Any, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
