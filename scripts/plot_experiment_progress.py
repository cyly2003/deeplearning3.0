"""Plot publication-ready summaries of the v1.2 transfer-learning experiments.

The script reads compact CSV summaries that are already tracked in the project
outputs and writes PNG/SVG figures for progress review and reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "figures" / "experiment_progress_20260625"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_progress_metrics(out_dir: Path) -> None:
    rows = [
        {
            "track": "f20 low-soil transfer",
            "experiment": "v1.2.1",
            "label": "Corrected\nbaseline",
            "order": 1,
            "mae": 1.2311,
            "r2": 0.1700,
            "marker": "o",
        },
        {
            "track": "f20 low-soil transfer",
            "experiment": "v1.2.2",
            "label": "Source\nTanimoto",
            "order": 2,
            "mae": 1.1634,
            "r2": 0.2785,
            "marker": "o",
        },
        {
            "track": "f20 low-soil transfer",
            "experiment": "v1.2.6",
            "label": "Authority\nCE-bin",
            "order": 3,
            "mae": 1.1478,
            "r2": 0.2978,
            "marker": "o",
        },
        {
            "track": "f20 low-soil transfer",
            "experiment": "v1.2.8",
            "label": "Censored\nloss",
            "order": 4,
            "mae": 1.1412,
            "r2": 0.2944,
            "marker": "o",
        },
        {
            "track": "f100 transfer",
            "experiment": "v1.2.6",
            "label": "Authority\nCE-bin",
            "order": 5,
            "mae": 0.9768,
            "r2": 0.4978,
            "marker": "s",
        },
        {
            "track": "f100 transfer",
            "experiment": "v1.2.8",
            "label": "Censored\nloss",
            "order": 6,
            "mae": 0.9692,
            "r2": 0.5072,
            "marker": "s",
        },
        {
            "track": "f100 transfer",
            "experiment": "v1.2.11",
            "label": "3-seed\nmean",
            "order": 7,
            "mae": 0.9935,
            "r2": 0.4822,
            "marker": "s",
        },
        {
            "track": "f100 transfer",
            "experiment": "v1.2.12",
            "label": "5-seed\nmean",
            "order": 8,
            "mae": 0.9883,
            "r2": 0.4875,
            "marker": "s",
        },
        {
            "track": "f100 transfer",
            "experiment": "v1.2.12",
            "label": "5-seed\nensemble",
            "order": 9,
            "mae": 0.9277,
            "r2": 0.5388,
            "marker": "D",
        },
    ]
    frame = pd.DataFrame(rows)
    colors = {
        "f20 low-soil transfer": OKABE_ITO["blue"],
        "f100 transfer": OKABE_ITO["orange"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharex=True)
    for track, group in frame.groupby("track", sort=False):
        axes[0].plot(
            group["order"],
            group["mae"],
            color=colors[track],
            lw=1.8,
            label=track,
        )
        axes[1].plot(
            group["order"],
            group["r2"],
            color=colors[track],
            lw=1.8,
            label=track,
        )
        for _, row in group.iterrows():
            axes[0].scatter(row["order"], row["mae"], color=colors[track], marker=row["marker"], s=42)
            axes[1].scatter(row["order"], row["r2"], color=colors[track], marker=row["marker"], s=42)
            offset = 9 if row["track"].startswith("f20") else -15
            for ax, metric in zip(axes, ["mae", "r2"]):
                ax.annotate(
                    row["label"].replace("\n", " "),
                    (row["order"], row[metric]),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="#333333",
                )

    for ax, metric, title in zip(axes, ["mae", "r2"], ["MAE progression", "R2 progression"]):
        ax.set_title(title)
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)
        ax.set_xticks(frame["order"])
        ax.set_xticklabels(frame["experiment"], rotation=45, ha="right")
        ax.set_xlabel("Experiment stage")
        if metric == "mae":
            ax.set_ylabel("MAE (log10 toxicity unit; lower is better)")
        else:
            ax.set_ylabel("R2 (higher is better)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Representative progress of soil pTox transfer experiments", y=1.03)
    save_figure(fig, out_dir, "fig1_experiment_progress_metrics")


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def readable_run_name(run: str) -> str:
    mapping = {
        "anchor_tanimoto_a1_5seed_ensemble": "Anchor\nTanimoto",
        "tanimoto_proxy_a0p5_5seed_ensemble": "Tanimoto +\nproxy",
        "proxydist_a0p5_5seed_ensemble": "Proxy\ndistance",
        "anchor_tanimoto_a1_val0_5seed_ensemble": "Val0\nanchor",
    }
    return mapping.get(run, run)


def plot_final_candidates(out_dir: Path) -> None:
    frame = read_csv(
        "outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/"
        "seed_mean_ensemble_focus_summary.csv"
    )
    frame = frame[frame["tier"].eq("all_test")].copy()
    frame["label"] = frame["run"].map(readable_run_name)
    metrics = [
        ("mae", "MAE", "lower is better"),
        ("r2", "R2", "higher is better"),
        ("huber_loss", "Huber loss", "lower is better"),
    ]
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        bars = ax.bar(frame["label"], frame[metric], color=colors, width=0.72)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=8)
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    fig.suptitle("Final 5-seed ensemble candidates on the fixed f100 test set", y=1.04)
    save_figure(fig, out_dir, "fig2_final_candidate_comparison")


def plot_ad_gate(out_dir: Path) -> None:
    frame = read_csv(
        "outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/"
        "seed_mean_ensemble_ad_gate_summary.csv"
    )
    frame = frame[frame["run"].eq("anchor_tanimoto_a1_5seed_ensemble")].copy()
    selected = [
        ("all_test", "All test"),
        ("overall_in_domain", "Overall\nin-domain"),
        ("species_seen_train", "Species\nseen"),
        ("species_task_family_seen_train", "Species+family\nseen"),
        ("species_extrapolation_only", "Species\nextrap."),
        ("species_task_family_unseen", "Species+family\nunseen"),
    ]
    labels = {key: label for key, label in selected}
    frame = frame[frame["tier"].isin(labels)].copy()
    frame["label"] = frame["tier"].map(labels)
    frame["order"] = frame["tier"].map({key: i for i, (key, _) in enumerate(selected)})
    frame = frame.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    y = np.arange(len(frame))
    for ax, metric, color, xlabel in [
        (axes[0], "mae", OKABE_ITO["blue"], "MAE (log10 toxicity unit)"),
        (axes[1], "r2", OKABE_ITO["orange"], "R2"),
    ]:
        bars = ax.barh(y, frame[metric], color=color, height=0.66)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["label"])
        ax.grid(axis="x", color="#DDDDDD", lw=0.7)
        for bar, (_, row) in zip(bars, frame.iterrows()):
            value = bar.get_width()
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.2f} (n={int(row['n'])})",
                ha="left",
                va="center",
                fontsize=7,
            )
    axes[0].invert_yaxis()
    fig.suptitle("Applicability-domain stratification of the current anchor ensemble", y=1.04)
    save_figure(fig, out_dir, "fig3_ad_gate_profile")


def plot_endpoint_family(out_dir: Path) -> None:
    frame = read_csv(
        "outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/"
        "seed_mean_ensemble_family_summary.csv"
    )
    run_order = [
        "anchor_tanimoto_a1_5seed_ensemble",
        "tanimoto_proxy_a0p5_5seed_ensemble",
        "proxydist_a0p5_5seed_ensemble",
    ]
    family_order = ["ECx", "LOEC", "NOEC"]
    frame = frame[frame["run"].isin(run_order) & frame["task_family"].isin(family_order)].copy()
    frame["run_label"] = frame["run"].map(readable_run_name)
    colors = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
    x = np.arange(len(family_order))
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True)
    for i, run in enumerate(run_order):
        subset = frame[frame["run"].eq(run)].set_index("task_family").loc[family_order]
        offset = (i - 1) * width
        axes[0].bar(x + offset, subset["mae"], width, color=colors[i], label=readable_run_name(run).replace("\n", " "))
        axes[1].bar(x + offset, subset["r2"], width, color=colors[i], label=readable_run_name(run).replace("\n", " "))
    for ax, ylabel in zip(axes, ["MAE (log10 toxicity unit)", "R2"]):
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(family_order)
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.24)
    fig.suptitle("Endpoint-family behavior of final f100 candidates", y=1.04)
    save_figure(fig, out_dir, "fig4_endpoint_family_profile")


def plot_validation_policy(out_dir: Path) -> None:
    frame = read_csv(
        "outputs/experiments/v1_2_13_anchor_validation_policy_remote_summary/"
        "validation_policy_comparison.csv"
    )
    frame["label"] = frame["policy"].map(
        {
            "val0p2_control_existing_v1_2_12": "Keep 20%\nfinetune val",
            "val0_new_v1_2_13": "Use all\nfinetune rows",
        }
    )
    metrics = [
        ("mae", "MAE", "lower is better"),
        ("r2", "R2", "higher is better"),
        ("huber_loss", "Huber loss", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    colors = [OKABE_ITO["blue"], OKABE_ITO["red"]]
    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        bars = ax.bar(frame["label"], frame[metric], color=colors, width=0.68)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=8)
        ax.grid(axis="y", color="#DDDDDD", lw=0.7)
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    fig.suptitle("Validation-policy sensitivity on the same fixed f100 test set", y=1.04)
    save_figure(fig, out_dir, "fig5_validation_policy_check")


def write_index(out_dir: Path) -> None:
    lines = [
        "# Experiment Progress Visualization",
        "",
        "Generated by `scripts/plot_experiment_progress.py`.",
        "",
        "## Figures",
        "",
        "1. `fig1_experiment_progress_metrics`: representative MAE/R2 progression across v1.2 stages.",
        "2. `fig2_final_candidate_comparison`: final 5-seed ensemble candidate comparison.",
        "3. `fig3_ad_gate_profile`: applicability-domain stratification for the current anchor ensemble.",
        "4. `fig4_endpoint_family_profile`: ECx/LOEC/NOEC behavior of final candidates.",
        "5. `fig5_validation_policy_check`: v1.2.13 validation-policy sensitivity check.",
        "",
        "## Interpretation Notes",
        "",
        "- Do not compare f20 and f100 points as the same split; the progress plot separates the tracks.",
        "- The v1.2.12 ensemble improvement reflects seed averaging on the same fixed test set, not a changed test set.",
        "- AD and endpoint-family plots should be reported with the aggregate metrics to avoid hiding extrapolation risks.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    configure_matplotlib()
    plot_progress_metrics(out_dir)
    plot_final_candidates(out_dir)
    plot_ad_gate(out_dir)
    plot_endpoint_family(out_dir)
    plot_validation_policy(out_dir)
    write_index(out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
