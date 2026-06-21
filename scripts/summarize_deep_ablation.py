from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import font_manager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize deep ablation experiment metrics.")
    parser.add_argument("--root", required=True, help="Experiment root containing deep/<ablation>/<split>/metrics.csv")
    parser.add_argument("--style", default="style_journal_clean_v1.yaml", help="Plot style YAML")
    parser.add_argument("--out-dir", default=None, help="Output directory, default: <root>")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    style = _load_style(Path(args.style))
    _apply_matplotlib_style(style)

    summary = collect_metrics(root)
    if summary.empty:
        raise ValueError(f"No metrics.csv files found under {root}")

    summary_path = out_dir / "summary_metrics.csv"
    test_path = out_dir / "summary_test_metrics.csv"
    delta_path = out_dir / "summary_ablation_delta_vs_full.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    test = summary[summary["split_part"] == "test"].copy()
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    delta = compute_delta_vs_full(summary)
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")
    convergence = collect_histories(root)
    convergence_path = out_dir / "summary_training_convergence.csv"
    convergence.to_csv(convergence_path, index=False, encoding="utf-8-sig")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_test_r2(test, style, figures_dir / "mean_test_r2_by_ablation")
    plot_delta_test_r2(delta[delta["split_part"] == "test"].copy(), style, figures_dir / "delta_test_r2_vs_full")
    plot_training_curves(convergence, style, figures_dir)

    print(f"summary={summary_path}")
    print(f"test={test_path}")
    print(f"delta={delta_path}")
    print(f"convergence={convergence_path}")
    print(f"figures={figures_dir}")


def collect_metrics(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(root.glob("deep/*/*/metrics.csv")):
        split_name = metrics_path.parent.name
        ablation = metrics_path.parent.parent.name
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "split_name", split_name)
        metrics.insert(0, "ablation", ablation)
        manifest_path = metrics_path.with_name("manifest.json")
        manifest = _read_json(manifest_path)
        metrics["encoder_source"] = manifest.get("encoder_source", "")
        metrics["epochs"] = manifest.get("epochs", "")
        metrics["learning_rate"] = manifest.get("learning_rate", "")
        metrics["batch_size"] = manifest.get("batch_size", "")
        metrics["rows"] = manifest.get("rows", "")
        metrics["train_rows"] = manifest.get("train_rows", "")
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_delta_vs_full(summary: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["split_name", "split_part", "task_head"]
    full = summary[summary["ablation"] == "full"][
        key_columns + ["r2", "rmse", "mae", "huber_loss"]
    ].rename(
        columns={
            "r2": "full_r2",
            "rmse": "full_rmse",
            "mae": "full_mae",
            "huber_loss": "full_huber_loss",
        }
    )
    merged = summary.merge(full, on=key_columns, how="left")
    merged["delta_r2_vs_full"] = merged["r2"] - merged["full_r2"]
    merged["delta_rmse_vs_full"] = merged["rmse"] - merged["full_rmse"]
    merged["delta_mae_vs_full"] = merged["mae"] - merged["full_mae"]
    merged["delta_huber_vs_full"] = merged["huber_loss"] - merged["full_huber_loss"]
    return merged


def collect_histories(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for history_path in sorted(root.glob("deep/*/*/history.csv")):
        ablation = history_path.parent.parent.name
        split_name = history_path.parent.name
        history = pd.read_csv(history_path)
        if history.empty:
            continue
        for row in history.itertuples(index=False):
            rows.append(
                {
                    "split_name": split_name,
                    "ablation": ablation,
                    "epoch": int(row.epoch),
                    "mean_loss": float(row.mean_loss),
                    "samples": int(row.samples),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["last10_mean_loss"] = (
        frame.sort_values("epoch")
        .groupby(["split_name", "ablation"])["mean_loss"]
        .transform(lambda values: values.tail(min(10, len(values))).mean())
    )
    return frame


def plot_mean_test_r2(test: pd.DataFrame, style: dict[str, Any], out_base: Path) -> None:
    grouped = (
        test.groupby(["split_name", "ablation"], as_index=False)
        .agg(mean_r2=("r2", "mean"), total_n=("n", "sum"))
        .sort_values(["split_name", "ablation"])
    )
    _plot_grouped_bars(
        grouped,
        value_column="mean_r2",
        ylabel="Mean test R2 across task heads",
        title=None,
        style=style,
        out_base=out_base,
    )


def plot_delta_test_r2(delta: pd.DataFrame, style: dict[str, Any], out_base: Path) -> None:
    grouped = (
        delta[delta["ablation"] != "full"]
        .groupby(["split_name", "ablation"], as_index=False)
        .agg(delta_r2=("delta_r2_vs_full", "mean"), total_n=("n", "sum"))
        .sort_values(["split_name", "ablation"])
    )
    _plot_grouped_bars(
        grouped,
        value_column="delta_r2",
        ylabel="Mean test R2 change vs full",
        title=None,
        style=style,
        out_base=out_base,
        add_zero_line=True,
    )


def plot_training_curves(convergence: pd.DataFrame, style: dict[str, Any], figures_dir: Path) -> None:
    if convergence.empty:
        return
    for split_name, frame in convergence.groupby("split_name"):
        fig_size = _figsize_inches(style, "double_column_wide", default=(7.0, 4.1))
        fig, ax = plt.subplots(figsize=fig_size)
        ablations = list(dict.fromkeys(frame["ablation"]))
        colors = _ablation_colors(style, ablations)
        for idx, ablation in enumerate(ablations):
            series = frame[frame["ablation"] == ablation].sort_values("epoch")
            ax.plot(
                series["epoch"],
                series["mean_loss"],
                label=ablation,
                color=colors[idx],
                linewidth=float(style.get("lines", {}).get("main_linewidth", 1.25)),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Huber loss")
        ax.legend(ncols=min(3, len(ablations)))
        ax.grid(True, axis="both")
        _style_axes(ax, style)
        fig.tight_layout()
        safe_name = split_name.replace("/", "_").replace("\\", "_")
        out_base = figures_dir / f"training_loss_{safe_name}"
        for suffix in (".png", ".svg"):
            fig.savefig(out_base.with_suffix(suffix))
        plt.close(fig)
def _plot_grouped_bars(
    frame: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str | None,
    style: dict[str, Any],
    out_base: Path,
    add_zero_line: bool = False,
) -> None:
    if frame.empty:
        return
    split_names = list(dict.fromkeys(frame["split_name"]))
    ablations = list(dict.fromkeys(frame["ablation"]))
    values = {
        (row.split_name, row.ablation): float(getattr(row, value_column))
        for row in frame.itertuples(index=False)
    }
    width = 0.8 / max(len(ablations), 1)
    x_positions = list(range(len(split_names)))
    fig_size = _figsize_inches(style, "double_column_wide", default=(7.0, 4.1))
    fig, ax = plt.subplots(figsize=fig_size)
    colors = _ablation_colors(style, ablations)
    for idx, ablation in enumerate(ablations):
        offsets = [x + (idx - (len(ablations) - 1) / 2) * width for x in x_positions]
        heights = [values.get((split_name, ablation), float("nan")) for split_name in split_names]
        ax.bar(offsets, heights, width=width * 0.92, label=ablation, color=colors[idx])
    if add_zero_line:
        zero = style.get("zero_lines", {})
        ax.axhline(
            0.0,
            color=zero.get("color", "#333333"),
            linewidth=float(zero.get("linewidth", 0.8)),
            linestyle=zero.get("linestyle", "-"),
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_names, rotation=18, ha="right")
    if title:
        ax.set_title(title)
    ax.legend(ncols=min(3, len(ablations)))
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


def _ablation_colors(style: dict[str, Any], ablations: list[str]) -> list[str]:
    palette = style.get("colors", {}).get("model_lines", {})
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555", "#E69F00"]
    preferred = [
        palette.get("main_model", fallback[0]),
        palette.get("ablation_1", fallback[1]),
        palette.get("ablation_2", fallback[2]),
        palette.get("ablation_3", fallback[3]),
        palette.get("baseline", fallback[4]),
    ]
    colors = []
    for idx, _ in enumerate(ablations):
        colors.append(preferred[idx] if idx < len(preferred) else fallback[idx % len(fallback)])
    return colors


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
