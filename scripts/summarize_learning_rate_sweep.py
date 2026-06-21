from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import font_manager


METRIC_COLUMNS = ("r2", "rmse", "mae", "huber_loss")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize deep learning-rate sweep results.")
    parser.add_argument("--root", required=True, help="Sweep root containing lr_*/deep/<ablation>/<split>/metrics.csv")
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
        raise ValueError(f"No learning-rate metrics found under {root}")
    test = summary[summary["split_part"] == "test"].copy()
    family_lr = summarize_family_learning_rate(test)
    split_lr = summarize_split_learning_rate(test)
    convergence = collect_histories(root)

    summary.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out_dir / "summary_test_metrics.csv", index=False, encoding="utf-8-sig")
    split_lr.to_csv(out_dir / "summary_split_learning_rate_metrics.csv", index=False, encoding="utf-8-sig")
    family_lr.to_csv(out_dir / "summary_family_learning_rate_metrics.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(out_dir / "summary_training_convergence.csv", index=False, encoding="utf-8-sig")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_by_lr(family_lr, "r2_mean", "Mean test R2 across task heads", style, figures_dir / "lr_mean_test_r2")
    plot_metric_by_lr(
        family_lr,
        "huber_loss_mean",
        "Mean test Huber loss across task heads",
        style,
        figures_dir / "lr_mean_test_huber",
    )
    plot_training_curves(convergence, style, figures_dir)

    print(f"summary={out_dir / 'summary_metrics.csv'}")
    print(f"test={out_dir / 'summary_test_metrics.csv'}")
    print(f"split_lr={out_dir / 'summary_split_learning_rate_metrics.csv'}")
    print(f"family_lr={out_dir / 'summary_family_learning_rate_metrics.csv'}")
    print(f"convergence={out_dir / 'summary_training_convergence.csv'}")
    print(f"figures={figures_dir}")


def collect_metrics(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(root.glob("lr_*/deep/*/*/metrics.csv")):
        lr_label = metrics_path.parts[-5]
        ablation = metrics_path.parent.parent.name
        split_name = metrics_path.parent.name
        manifest = _read_json(metrics_path.with_name("manifest.json"))
        learning_rate = manifest.get("learning_rate", _learning_rate_from_label(lr_label))
        metrics = pd.read_csv(metrics_path)
        metrics["lr_label"] = lr_label
        metrics["learning_rate"] = learning_rate
        metrics["split_family"] = split_name.split("_", 1)[0]
        metrics["ablation"] = ablation
        metrics["split_name"] = split_name
        metrics["epochs"] = manifest.get("epochs", "")
        metrics["epochs_ran"] = manifest.get("epochs_ran", "")
        metrics["best_epoch"] = manifest.get("best_epoch", "")
        metrics["encoder_source"] = manifest.get("encoder_source", "")
        ordered = [
            "lr_label",
            "learning_rate",
            "split_family",
            "ablation",
            "split_name",
            *[column for column in metrics.columns if column not in {"lr_label", "learning_rate", "split_family", "ablation", "split_name"}],
        ]
        rows.append(metrics[ordered])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_histories(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for history_path in sorted(root.glob("lr_*/deep/*/*/history.csv")):
        lr_label = history_path.parts[-5]
        manifest = _read_json(history_path.with_name("manifest.json"))
        learning_rate = manifest.get("learning_rate", _learning_rate_from_label(lr_label))
        ablation = history_path.parent.parent.name
        split_name = history_path.parent.name
        history = pd.read_csv(history_path)
        for row in history.itertuples(index=False):
            rows.append(
                {
                    "lr_label": lr_label,
                    "learning_rate": learning_rate,
                    "split_family": split_name.split("_", 1)[0],
                    "ablation": ablation,
                    "split_name": split_name,
                    "epoch": int(row.epoch),
                    "mean_loss": float(row.mean_loss),
                    "samples": int(row.samples),
                }
            )
    return pd.DataFrame(rows)


def summarize_split_learning_rate(test: pd.DataFrame) -> pd.DataFrame:
    return (
        test.groupby(["split_family", "split_name", "learning_rate"], as_index=False)
        .apply(_aggregate_metrics)
        .reset_index(drop=True)
        .sort_values(["split_family", "split_name", "learning_rate"])
    )


def summarize_family_learning_rate(test: pd.DataFrame) -> pd.DataFrame:
    return (
        test.groupby(["split_family", "learning_rate"], as_index=False)
        .apply(_aggregate_metrics)
        .reset_index(drop=True)
        .sort_values(["split_family", "learning_rate"])
    )


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


def plot_metric_by_lr(frame: pd.DataFrame, value_column: str, ylabel: str, style: dict[str, Any], out_base: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=_figsize_inches(style, "double_column_wide", default=(7.0, 4.1)))
    families = list(dict.fromkeys(frame["split_family"]))
    colors = _family_colors(style, families)
    for idx, family in enumerate(families):
        series = frame[frame["split_family"] == family].sort_values("learning_rate")
        ax.plot(
            series["learning_rate"],
            series[value_column],
            marker="o",
            label=family,
            color=colors[idx],
            linewidth=float(style.get("lines", {}).get("main_linewidth", 1.25)),
            markersize=float(style.get("lines", {}).get("marker_size", 3.0)),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(ylabel)
    ax.legend(ncols=min(3, len(families)))
    _style_axes(ax, style)
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


def plot_training_curves(convergence: pd.DataFrame, style: dict[str, Any], figures_dir: Path) -> None:
    if convergence.empty:
        return
    for split_name, frame in convergence.groupby("split_name"):
        fig, ax = plt.subplots(figsize=_figsize_inches(style, "double_column_wide", default=(7.0, 4.1)))
        rates = sorted(frame["learning_rate"].unique())
        colors = _rate_colors(style, rates)
        for idx, rate in enumerate(rates):
            series = frame[frame["learning_rate"] == rate].sort_values("epoch")
            ax.plot(
                series["epoch"],
                series["mean_loss"],
                label=f"{rate:g}",
                color=colors[idx],
                linewidth=float(style.get("lines", {}).get("main_linewidth", 1.25)),
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Huber loss")
        ax.legend(title="LR", ncols=min(4, len(rates)))
        _style_axes(ax, style)
        fig.tight_layout()
        safe_name = split_name.replace("/", "_").replace("\\", "_")
        for suffix in (".png", ".svg"):
            fig.savefig((figures_dir / f"lr_training_loss_{safe_name}").with_suffix(suffix))
        plt.close(fig)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _learning_rate_from_label(label: str) -> float | str:
    if label.startswith("lr_"):
        try:
            return float(label.removeprefix("lr_").replace("p", "."))
        except ValueError:
            return label
    return label


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


def _family_colors(style: dict[str, Any], families: list[str]) -> list[str]:
    palette = style.get("colors", {}).get("okabe_ito", {})
    fallback = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    preferred = [
        palette.get("blue", fallback[0]),
        palette.get("vermillion", fallback[1]),
        palette.get("bluish_green", fallback[2]),
        palette.get("reddish_purple", fallback[3]),
        palette.get("orange", fallback[4]),
        palette.get("sky_blue", fallback[5]),
    ]
    return [preferred[idx] if idx < len(preferred) else fallback[idx % len(fallback)] for idx, _ in enumerate(families)]


def _rate_colors(style: dict[str, Any], rates: list[float]) -> list[str]:
    return _family_colors(style, [str(rate) for rate in rates])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
