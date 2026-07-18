from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


RANDOM_MAINLINE_HISTORIES = [
    (
        "Random 8:2",
        Path(
            "outputs/experiments/v1_2_22_no_metal_random_split_remote/"
            "v1.2.22_no_metal_random8_2_seed2042_cebin_lw0025_censored_w0p01/"
            "deep/full/M_v2_aquatic_to_soil_ptox_no_metal_adapt_B_random_8_2_f100/history.csv"
        ),
    ),
    (
        "Random 5-fold fold 3",
        Path(
            "outputs/experiments/v1_2_22_no_metal_random_split_remote/"
            "v1.2.22_no_metal_random5fold_fold3_seed2042_cebin_lw0025_censored_w0p01/"
            "deep/full/M_v2_aquatic_to_soil_ptox_no_metal_adapt_E_random_5fold_fold3_f100/history.csv"
        ),
    ),
]

SCAFFOLD_MAINLINE_HISTORY = (
    "Scaffold/cluster 8:2",
    Path(
        "outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote/"
        "v1.2.24_no_metal_scaffold8_2_seed2042_cebin_lw0025_censored_w0p01/"
        "deep/full/M_v2_aquatic_to_soil_ptox_no_metal_adapt_G_scaffold_cluster_8_2_f100/history.csv"
    ),
)


@dataclass(frozen=True)
class HistorySource:
    label: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Huber task-loss convergence from mainline deep-training history.csv files."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/figures/mainline_huber_loss_20260714"),
        help="Directory for PNG/SVG/source-data exports.",
    )
    parser.add_argument(
        "--include-scaffold",
        action="store_true",
        help="Also include the locally synced v1.2.24 scaffold/cluster 8:2 history.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    parser.add_argument(
        "--style",
        choices=["paper", "cute"],
        default="paper",
        help="Figure style. 'cute' exports a compact white-card version without a title.",
    )
    args = parser.parse_args()

    sources = [HistorySource(label, path) for label, path in RANDOM_MAINLINE_HISTORIES]
    if args.include_scaffold:
        label, path = SCAFFOLD_MAINLINE_HISTORY
        sources.append(HistorySource(label, path))

    history = load_histories(sources)
    summary = summarize_history(history)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    history.to_csv(args.out_dir / "mainline_huber_loss_source_data.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "mainline_huber_loss_summary.csv", index=False, encoding="utf-8-sig")

    configure_matplotlib()
    if args.style == "cute":
        plot_cute_loss(summary, history, args.out_dir / "mainline_huber_loss_cute", dpi=args.dpi)
    else:
        plot_loss(summary, history, args.out_dir / "mainline_huber_loss_decline", dpi=args.dpi)
    write_readme(args.out_dir, sources, history, summary)


def load_histories(sources: list[HistorySource]) -> pd.DataFrame:
    required = {"phase", "epoch", "global_epoch", "mean_task_loss", "validation_task_loss"}
    frames: list[pd.DataFrame] = []
    missing_files: list[Path] = []

    for source in sources:
        if not source.path.exists():
            missing_files.append(source.path)
            continue
        frame = pd.read_csv(source.path)
        missing_columns = required.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{source.path} is missing required columns: {sorted(missing_columns)}")

        slim = frame.copy()
        slim["run_label"] = source.label
        slim["source_history"] = source.path.as_posix()
        slim["global_epoch"] = pd.to_numeric(slim["global_epoch"], errors="coerce")
        slim["epoch"] = pd.to_numeric(slim["epoch"], errors="coerce")
        slim["train_huber_task_loss"] = pd.to_numeric(slim["mean_task_loss"], errors="coerce")
        slim["validation_huber_task_loss"] = pd.to_numeric(slim["validation_task_loss"], errors="coerce")
        frames.append(
            slim[
                [
                    "run_label",
                    "source_history",
                    "phase",
                    "epoch",
                    "global_epoch",
                    "train_huber_task_loss",
                    "validation_huber_task_loss",
                    "samples",
                    "validation_samples",
                    "best_epoch",
                    "learning_rate",
                ]
            ]
        )

    if missing_files:
        joined = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing history files:\n{joined}")
    if not frames:
        raise FileNotFoundError("No history files were loaded.")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["global_epoch", "train_huber_task_loss", "validation_huber_task_loss"])
    data["global_epoch"] = data["global_epoch"].astype(int)
    data["phase_epoch"] = data["epoch"].astype(int)
    max_pretrain_epoch = int(data.loc[data["phase"].eq("pretrain"), "phase_epoch"].max())
    data["plot_epoch"] = np.where(
        data["phase"].eq("pretrain"),
        data["phase_epoch"],
        max_pretrain_epoch + data["phase_epoch"],
    ).astype(int)
    return data.sort_values(["run_label", "plot_epoch"]).reset_index(drop=True)


def summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (plot_epoch, phase), sub in history.groupby(["plot_epoch", "phase"], sort=True):
        row = {
            "plot_epoch": int(plot_epoch),
            "phase": str(phase),
            "phase_epoch": int(sub["phase_epoch"].median()),
            "min_global_epoch": int(sub["global_epoch"].min()),
            "max_global_epoch": int(sub["global_epoch"].max()),
            "n_histories": int(sub["run_label"].nunique()),
        }
        for source_col, prefix in [
            ("train_huber_task_loss", "train"),
            ("validation_huber_task_loss", "validation"),
        ]:
            values = pd.to_numeric(sub[source_col], errors="coerce").dropna()
            row[f"{prefix}_mean"] = float(values.mean())
            row[f"{prefix}_min"] = float(values.min())
            row[f"{prefix}_max"] = float(values.max())
            row[f"{prefix}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("plot_epoch").reset_index(drop=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def plot_loss(summary: pd.DataFrame, history: pd.DataFrame, stem: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.85))

    phase_bounds = get_phase_bounds(summary)
    if "pretrain" in phase_bounds:
        ax.axvspan(*phase_bounds["pretrain"], color="#EEF2F6", alpha=0.75, linewidth=0)
    if "finetune" in phase_bounds:
        ax.axvspan(*phase_bounds["finetune"], color="#F7F0E8", alpha=0.72, linewidth=0)

    train_color = "#2563A8"
    val_color = "#C45A1C"
    for phase_index, phase in enumerate(["pretrain", "finetune"]):
        sub = summary[summary["phase"].eq(phase)].sort_values("plot_epoch")
        if sub.empty:
            continue
        x = sub["plot_epoch"].to_numpy(dtype=float)
        train_mean = sub["train_mean"].to_numpy(dtype=float)
        train_min = sub["train_min"].to_numpy(dtype=float)
        train_max = sub["train_max"].to_numpy(dtype=float)
        val_mean = sub["validation_mean"].to_numpy(dtype=float)
        val_min = sub["validation_min"].to_numpy(dtype=float)
        val_max = sub["validation_max"].to_numpy(dtype=float)

        ax.fill_between(x, train_min, train_max, color=train_color, alpha=0.13, linewidth=0)
        ax.fill_between(x, val_min, val_max, color=val_color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            train_mean,
            color=train_color,
            linewidth=1.9,
            label="Training Huber task loss" if phase_index == 0 else None,
        )
        ax.plot(
            x,
            val_mean,
            color=val_color,
            linewidth=1.9,
            linestyle=(0, (4, 2)),
            label="Validation Huber task loss" if phase_index == 0 else None,
        )

    finetune_epochs = summary.loc[summary["phase"].eq("finetune"), "plot_epoch"]
    if not finetune_epochs.empty:
        boundary = float(finetune_epochs.min()) - 0.5
        ax.axvline(boundary, color="#4B5563", linewidth=0.9, linestyle=(0, (2, 2)))

    annotate_phase_labels(ax, phase_bounds, summary)
    ax.set_title("Mainline Huber Loss Decline")
    ax.set_xlabel("Training epoch (phase-aligned)")
    ax.set_ylabel("Huber task loss (objective scale)")
    ax.grid(axis="y", color="#D1D5DB", alpha=0.55, linewidth=0.6)
    ax.legend(loc="upper right", frameon=False)

    n_runs = history["run_label"].nunique()
    note = (
        f"Mean with min-max band across locally synced random-mainline histories (n={n_runs}); "
        "finetune epochs are aligned after pretrain."
    )
    fig.text(0.125, 0.015, note, ha="left", va="bottom", fontsize=6.6, color="#4B5563")
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_cute_loss(summary: pd.DataFrame, history: pd.DataFrame, stem: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(2.45, 3.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    card = FancyBboxPatch(
        (0.055, 0.055),
        0.89,
        0.89,
        boxstyle="round,pad=0.012,rounding_size=0.05",
        linewidth=1.25,
        edgecolor="#F2BA57",
        facecolor="white",
        transform=fig.transFigure,
        zorder=-10,
    )
    fig.patches.append(card)

    line_color = "#EF3B58"
    axis_color = "#111827"

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x0, y0 = 0.16, 0.16
    x1, y1 = 0.89, 0.78
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": axis_color, "mutation_scale": 8.5},
    )
    ax.annotate(
        "",
        xy=(x0, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": axis_color, "mutation_scale": 8.5},
    )

    data = summary.sort_values("plot_epoch")
    x_raw = data["plot_epoch"].to_numpy(dtype=float)
    y_raw = smooth_for_cute(data["validation_mean"].to_numpy(dtype=float))
    x = 0.23 + 0.58 * (x_raw - float(x_raw.min())) / (float(x_raw.max()) - float(x_raw.min()))
    y = 0.23 + 0.46 * (y_raw - float(y_raw.min())) / (float(y_raw.max()) - float(y_raw.min()))
    ax.plot(x, y, color=line_color, linewidth=2.25, solid_capstyle="round", transform=ax.transAxes)

    ax.text((x0 + x1) / 2, 0.055, "epoch", color=axis_color, fontsize=7.6, ha="center", va="center")
    ax.text(
        0.055,
        (y0 + y1) / 2,
        "Huber loss",
        color=axis_color,
        fontsize=7.6,
        rotation=90,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    fig.subplots_adjust(left=0.18, right=0.89, bottom=0.16, top=0.88)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def smooth_for_cute(values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values
    return pd.Series(values).rolling(window=3, center=True, min_periods=1).mean().to_numpy(dtype=float)


def get_phase_bounds(summary: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for phase, sub in summary.groupby("phase", sort=False):
        epochs = sub["plot_epoch"].to_numpy(dtype=float)
        bounds[str(phase)] = (float(np.nanmin(epochs)) - 0.5, float(np.nanmax(epochs)) + 0.5)
    return bounds


def annotate_phase_labels(
    ax: plt.Axes,
    phase_bounds: dict[str, tuple[float, float]],
    summary: pd.DataFrame,
) -> None:
    ymax = max(float(summary["train_max"].max()), float(summary["validation_max"].max()))
    for phase, label in [("pretrain", "Pretrain"), ("finetune", "Finetune")]:
        if phase not in phase_bounds:
            continue
        low, high = phase_bounds[phase]
        ax.text((low + high) / 2, ymax * 1.01, label, ha="center", va="bottom", fontsize=7.5, color="#374151")


def write_readme(
    out_dir: Path,
    sources: list[HistorySource],
    history: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    source_lines = "\n".join(f"- {source.label}: `{source.path.as_posix()}`" for source in sources)
    final = summary.iloc[-1]
    content = f"""# Mainline Huber Loss Figure

This folder contains a publication-style Huber task-loss decline plot for the current random-mainline boundary.

## Files

- `mainline_huber_loss_decline.png`: high-resolution raster export.
- `mainline_huber_loss_decline.svg`: editable vector export.
- `mainline_huber_loss_cute.png`: compact white-card export when `--style cute` is used.
- `mainline_huber_loss_cute.svg`: editable compact white-card export when `--style cute` is used.
- `mainline_huber_loss_source_data.csv`: long-format epoch-level source data.
- `mainline_huber_loss_summary.csv`: epoch-level mean/min/max summary used for plotting.

## Source Histories

{source_lines}

## Notes

- The plotted loss is `mean_task_loss` and `validation_task_loss`, i.e. the Huber task-loss component of the training objective.
- The figure does not plot `mean_loss`, because that total objective also includes toxicity-bin cross-entropy and censored-loss terms.
- Final epoch summary: training mean={final['train_mean']:.6g}, validation mean={final['validation_mean']:.6g}, histories={int(history['run_label'].nunique())}.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
