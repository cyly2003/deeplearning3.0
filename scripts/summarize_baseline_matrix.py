from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import font_manager


METRIC_COLUMNS = ("r2", "rmse", "mae", "huber_loss")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize traditional ML baseline matrix metrics.")
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Baseline root containing <split>/<model>_metrics.csv. Repeat to combine roots.",
    )
    parser.add_argument("--style", default="style_journal_clean_v1.yaml", help="Plot style YAML")
    parser.add_argument("--out-dir", default=None, help="Output directory, default: <root>")
    parser.add_argument(
        "--deep-full",
        default=None,
        help="Optional deep full summary_test_metrics.csv for baseline-vs-deep comparison.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(root) for root in args.root]
    out_dir = Path(args.out_dir) if args.out_dir else roots[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    style = _load_style(Path(args.style))
    _apply_matplotlib_style(style)

    summary = collect_baseline_metrics(roots)
    if summary.empty:
        raise ValueError(f"No *_metrics.csv files found under {', '.join(str(root) for root in roots)}")

    test = summary[summary["split_part"] == "test"].copy()
    family_model = summarize_by_family_model(test)
    split_model = summarize_by_split_model(test)

    summary_path = out_dir / "summary_metrics.csv"
    test_path = out_dir / "summary_test_metrics.csv"
    split_model_path = out_dir / "summary_split_model_metrics.csv"
    family_model_path = out_dir / "summary_family_model_metrics.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    split_model.to_csv(split_model_path, index=False, encoding="utf-8-sig")
    family_model.to_csv(family_model_path, index=False, encoding="utf-8-sig")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_family_metric(family_model, "r2_mean", "Mean test R2 across task heads", style, figures_dir / "baseline_mean_test_r2")
    plot_family_metric(
        family_model,
        "huber_loss_mean",
        "Mean test Huber loss across task heads",
        style,
        figures_dir / "baseline_mean_test_huber",
    )

    print(f"summary={summary_path}")
    print(f"test={test_path}")
    print(f"split_model={split_model_path}")
    print(f"family_model={family_model_path}")

    if args.deep_full:
        compare = compare_with_deep_full(test, Path(args.deep_full))
        compare_path = out_dir / "summary_baseline_delta_vs_deep_full.csv"
        compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
        family_compare = summarize_baseline_deep_delta(compare)
        family_compare_path = out_dir / "summary_family_model_delta_vs_deep_full.csv"
        family_compare.to_csv(family_compare_path, index=False, encoding="utf-8-sig")
        plot_family_metric(
            family_compare,
            "delta_r2_vs_deep_full_mean",
            "Mean test R2 change vs deep full",
            style,
            figures_dir / "baseline_delta_r2_vs_deep_full",
            add_zero_line=True,
        )
        print(f"delta={compare_path}")
        print(f"family_delta={family_compare_path}")
    print(f"figures={figures_dir}")


def collect_baseline_metrics(roots: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for root in roots:
        for metrics_path in sorted(root.glob("*/*_metrics.csv")):
            split_name = metrics_path.parent.name
            model = metrics_path.stem.removesuffix("_metrics")
            metrics = pd.read_csv(metrics_path)
            metrics["source_root"] = str(root)
            metrics["split_name"] = split_name
            metrics["split_family"] = split_family(split_name)
            metrics["model"] = model
            front = ["source_root", "split_name", "split_family", "model"]
            metrics = metrics[front + [c for c in metrics.columns if c not in set(front)]]
            rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_by_split_model(test: pd.DataFrame) -> pd.DataFrame:
    return (
        test.groupby(["split_family", "split_name", "model"], as_index=False)
        .apply(_aggregate_metrics)
        .reset_index(drop=True)
        .sort_values(["split_family", "split_name", "r2_mean"], ascending=[True, True, False])
    )


def summarize_by_family_model(test: pd.DataFrame) -> pd.DataFrame:
    return (
        test.groupby(["split_family", "model"], as_index=False)
        .apply(_aggregate_metrics)
        .reset_index(drop=True)
        .sort_values(["split_family", "r2_mean"], ascending=[True, False])
    )


def compare_with_deep_full(test: pd.DataFrame, deep_full_path: Path) -> pd.DataFrame:
    deep = pd.read_csv(deep_full_path)
    if "ablation" in deep.columns:
        deep = deep[deep["ablation"] == "full"].copy()
    deep = deep[deep["split_part"] == "test"].copy()
    keep = ["split_name", "split_part", "task_head", *METRIC_COLUMNS]
    deep = deep[keep].rename(columns={metric: f"deep_full_{metric}" for metric in METRIC_COLUMNS})
    merged = test.merge(deep, on=["split_name", "split_part", "task_head"], how="left")
    for metric in METRIC_COLUMNS:
        merged[f"delta_{metric}_vs_deep_full"] = merged[metric] - merged[f"deep_full_{metric}"]
    return merged


def summarize_baseline_deep_delta(compare: pd.DataFrame) -> pd.DataFrame:
    delta_columns = [f"delta_{metric}_vs_deep_full" for metric in METRIC_COLUMNS]
    valid = compare.dropna(subset=delta_columns).copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby(["split_family", "model"], as_index=False)
    rows: list[dict[str, Any]] = []
    for (family, model), frame in grouped:
        row: dict[str, Any] = {
            "split_family": family,
            "model": model,
            "task_rows": int(len(frame)),
            "total_n": int(frame["n"].sum()),
        }
        for column in delta_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_n_weighted_mean"] = _weighted_mean(frame[column], frame["n"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_family", "delta_r2_vs_deep_full_mean"], ascending=[True, False])


def _aggregate_metrics(frame: pd.DataFrame) -> pd.Series:
    result: dict[str, Any] = {
        "task_rows": int(len(frame)),
        "total_n": int(frame["n"].sum()),
        "task_heads": ";".join(sorted(frame["task_head"].unique())),
    }
    for metric in METRIC_COLUMNS:
        result[f"{metric}_mean"] = float(frame[metric].mean())
        result[f"{metric}_n_weighted_mean"] = _weighted_mean(frame[metric], frame["n"])
    return pd.Series(result)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def split_family(split_name: str) -> str:
    match = re.search(r"(?:^|_)([A-F])_(?:random|chemical)", split_name)
    if match:
        return match.group(1)
    return split_name.split("_", 1)[0]


def plot_family_metric(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    style: dict[str, Any],
    out_base: Path,
    *,
    add_zero_line: bool = False,
) -> None:
    if frame.empty or value_column not in frame.columns:
        return
    families = list(dict.fromkeys(frame["split_family"]))
    models = list(dict.fromkeys(frame["model"]))
    values = {
        (row.split_family, row.model): float(getattr(row, value_column))
        for row in frame.itertuples(index=False)
    }
    width = 0.8 / max(len(models), 1)
    x_positions = list(range(len(families)))
    fig, ax = plt.subplots(figsize=_figsize_inches(style, "double_column_wide", default=(7.0, 4.1)))
    colors = _model_colors(style, models)
    for idx, model in enumerate(models):
        offsets = [x + (idx - (len(models) - 1) / 2) * width for x in x_positions]
        heights = [values.get((family, model), float("nan")) for family in families]
        ax.bar(offsets, heights, width=width * 0.92, label=model, color=colors[idx])
    if add_zero_line:
        zero = style.get("zero_lines", {})
        ax.axhline(
            0.0,
            color=zero.get("color", "#333333"),
            linewidth=float(zero.get("linewidth", 0.8)),
            linestyle=zero.get("linestyle", "-"),
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Split strategy family")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(families)
    ax.legend(ncols=min(3, len(models)))
    ax.grid(True, axis="y")
    _style_axes(ax, style)
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


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
        else:
            print(f"[warn] configured font not found, using matplotlib default: {font_path}")
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


def _model_colors(style: dict[str, Any], models: list[str]) -> list[str]:
    palette = style.get("colors", {}).get("okabe_ito", {})
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#555555"]
    preferred = [
        palette.get("blue", fallback[0]),
        palette.get("vermillion", fallback[1]),
        palette.get("bluish_green", fallback[2]),
        palette.get("reddish_purple", fallback[3]),
        palette.get("orange", fallback[4]),
        palette.get("sky_blue", fallback[5]),
        palette.get("black", fallback[6]),
    ]
    return [preferred[idx] if idx < len(preferred) else fallback[idx % len(fallback)] for idx, _ in enumerate(models)]


if __name__ == "__main__":
    main()
