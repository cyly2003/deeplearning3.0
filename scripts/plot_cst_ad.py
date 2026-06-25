from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIER_ORDER = ["AD-A", "AD-B", "AD-C", "AD-D"]
TIER_COLORS = {
    "AD-A": "#1b9e77",
    "AD-B": "#7570b3",
    "AD-C": "#d95f02",
    "AD-D": "#b2182b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CST-AD figures from cst_ad_prediction_rows.csv.")
    parser.add_argument("--cst-ad-rows", required=True)
    parser.add_argument("--tier-summary", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="cst_ad")
    args = parser.parse_args()

    rows = pd.read_csv(args.cst_ad_rows)
    summary = pd.read_csv(args.tier_summary) if args.tier_summary else summarize_for_plot(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_chemical_species_space(rows, out_dir, args.prefix)
    plot_chemical_species_space_by_task_family(rows, out_dir, args.prefix)
    plot_individual_task_family_spaces(rows, out_dir, args.prefix)
    plot_tier_performance(summary, out_dir, args.prefix)
    plot_tier_performance_by_task_family(rows, out_dir, args.prefix)
    plot_observed_vs_predicted(rows, out_dir, args.prefix)
    plot_uncertainty(rows, out_dir, args.prefix)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def plot_chemical_species_space(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    data = frame.copy()
    data["plot_chemical_score"] = pd.to_numeric(data["cst_chemical_score"], errors="coerce").round(4)
    data["plot_species_score"] = pd.to_numeric(data["cst_species_score"], errors="coerce").round(4)
    data["plot_abs_error"] = pd.to_numeric(data["abs_error"], errors="coerce")
    grouped = (
        data.dropna(subset=["plot_chemical_score", "plot_species_score"])
        .groupby(["plot_chemical_score", "plot_species_score"], as_index=False)
        .agg(n=("plot_abs_error", "size"), mean_abs_error=("plot_abs_error", "mean"))
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sizes = 26.0 + 18.0 * np.sqrt(grouped["n"].to_numpy(dtype=float))
    scatter = ax.scatter(
        grouped["plot_chemical_score"],
        grouped["plot_species_score"],
        c=grouped["mean_abs_error"],
        s=sizes,
        cmap="viridis",
        alpha=0.78,
        linewidths=0.35,
        edgecolors="white",
    )
    add_threshold_lines(ax, data)
    ax.set_xlabel("Chemical similarity to training set (max Tanimoto)")
    ax.set_ylabel("Species taxonomy similarity to training set")
    ax.set_title("CST-AD chemical-species space (all endpoint families)")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Mean absolute error on log10 toxicity scale")
    legend_sizes = [10, 100, 500]
    handles = [
        ax.scatter([], [], s=26.0 + 18.0 * np.sqrt(value), color="#777777", alpha=0.45, linewidths=0)
        for value in legend_sizes
    ]
    ax.legend(handles, [str(value) for value in legend_sizes], title="Samples", frameon=False, loc="lower left")
    ax.grid(alpha=0.18, linewidth=0.6)
    save(fig, out_dir / f"{prefix}_chemical_species_space")


def plot_chemical_species_space_by_task_family(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if "task_family" not in frame.columns:
        return
    data = prepare_chemical_species_plot_data(frame)
    families = sorted(str(value) for value in data["task_family"].dropna().unique())
    if not families:
        return
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(4.8 * len(families) + 0.8, 4.7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    color_max = float(data["mean_abs_error"].quantile(0.98)) if not data.empty else 1.0
    mappable = None
    for ax, family in zip(axes.ravel(), families):
        sub = data[data["task_family"].astype(str).eq(family)]
        if sub.empty:
            ax.set_visible(False)
            continue
        sizes = 26.0 + 18.0 * np.sqrt(sub["n"].to_numpy(dtype=float))
        mappable = ax.scatter(
            sub["plot_chemical_score"],
            sub["plot_species_score"],
            c=sub["mean_abs_error"].clip(upper=color_max),
            s=sizes,
            cmap="viridis",
            vmin=0,
            vmax=color_max,
            alpha=0.80,
            linewidths=0.35,
            edgecolors="white",
        )
        add_threshold_lines(ax, frame)
        ax.set_title(f"{family} (n={int(sub['n'].sum())})")
        ax.grid(alpha=0.18, linewidth=0.6)
    axes[0, 0].set_ylabel("Species taxonomy similarity to training set")
    for ax in axes.ravel():
        ax.set_xlabel("Chemical similarity to training set")
    if mappable is not None:
        fig.subplots_adjust(left=0.065, right=0.90, bottom=0.14, top=0.82, wspace=0.08)
        cbar_ax = fig.add_axes([0.925, 0.20, 0.015, 0.56])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Mean absolute error on log10 toxicity scale")
    fig.suptitle("CST-AD chemical-species space by endpoint family", y=1.02)
    save(fig, out_dir / f"{prefix}_chemical_species_space_by_task_family", tight=False)


def plot_individual_task_family_spaces(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if "task_family" not in frame.columns:
        return
    for family in sorted(str(value) for value in frame["task_family"].dropna().unique()):
        sub = frame[frame["task_family"].astype(str).eq(family)]
        if sub.empty:
            continue
        plot_single_task_family_space(sub, out_dir, f"{prefix}_chemical_species_space_{sanitize(family)}", family)


def plot_single_task_family_space(frame: pd.DataFrame, out_dir: Path, stem: str, family: str) -> None:
    grouped = prepare_chemical_species_plot_data(frame)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sizes = 26.0 + 18.0 * np.sqrt(grouped["n"].to_numpy(dtype=float))
    scatter = ax.scatter(
        grouped["plot_chemical_score"],
        grouped["plot_species_score"],
        c=grouped["mean_abs_error"],
        s=sizes,
        cmap="viridis",
        alpha=0.80,
        linewidths=0.35,
        edgecolors="white",
    )
    add_threshold_lines(ax, frame)
    ax.set_xlabel("Chemical similarity to training set (max Tanimoto)")
    ax.set_ylabel("Species taxonomy similarity to training set")
    ax.set_title(f"CST-AD chemical-species space: {family}")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Mean absolute error on log10 toxicity scale")
    add_sample_size_legend(ax)
    ax.grid(alpha=0.18, linewidth=0.6)
    save(fig, out_dir / stem)


def prepare_chemical_species_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["plot_chemical_score"] = pd.to_numeric(data["cst_chemical_score"], errors="coerce").round(4)
    data["plot_species_score"] = pd.to_numeric(data["cst_species_score"], errors="coerce").round(4)
    data["plot_abs_error"] = pd.to_numeric(data["abs_error"], errors="coerce")
    group_columns = ["plot_chemical_score", "plot_species_score"]
    if "task_family" in data.columns:
        group_columns.append("task_family")
    return (
        data.dropna(subset=["plot_chemical_score", "plot_species_score"])
        .groupby(group_columns, as_index=False)
        .agg(n=("plot_abs_error", "size"), mean_abs_error=("plot_abs_error", "mean"))
    )


def add_sample_size_legend(ax: plt.Axes) -> None:
    legend_sizes = [10, 100, 500]
    handles = [
        ax.scatter([], [], s=26.0 + 18.0 * np.sqrt(value), color="#777777", alpha=0.45, linewidths=0)
        for value in legend_sizes
    ]
    ax.legend(handles, [str(value) for value in legend_sizes], title="Samples", frameon=False, loc="lower left")


def add_threshold_lines(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if "cst_chemical_threshold" in frame.columns:
        value = pd.to_numeric(frame["cst_chemical_threshold"], errors="coerce").dropna()
        if not value.empty:
            ax.axvline(value.iloc[0], color="#444444", linestyle="--", linewidth=1.0)
    if "cst_species_threshold" in frame.columns:
        value = pd.to_numeric(frame["cst_species_threshold"], errors="coerce").dropna()
        if not value.empty:
            ax.axhline(value.iloc[0], color="#444444", linestyle="--", linewidth=1.0)


def plot_tier_performance(summary: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    data = summary.copy()
    data["cst_ad_tier"] = pd.Categorical(data["cst_ad_tier"], categories=TIER_ORDER, ordered=True)
    data = data.sort_values("cst_ad_tier")
    colors = [TIER_COLORS.get(str(tier), "#666666") for tier in data["cst_ad_tier"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    axes[0].bar(data["cst_ad_tier"].astype(str), pd.to_numeric(data["mae"], errors="coerce"), color=colors)
    axes[0].set_ylabel("MAE on log10 toxicity scale")
    axes[0].set_xlabel("CST-AD tier")
    axes[0].set_title("Prediction error by AD tier")
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[1].bar(
        data["cst_ad_tier"].astype(str),
        pd.to_numeric(data["coverage_fraction"], errors="coerce") * 100.0,
        color=colors,
    )
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_xlabel("CST-AD tier")
    axes[1].set_title("Test-set coverage")
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.6)
    save(fig, out_dir / f"{prefix}_tier_performance")


def plot_tier_performance_by_task_family(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if "task_family" not in frame.columns:
        return
    rows = []
    for (family, tier), sub in frame.groupby(["task_family", "cst_ad_tier"], dropna=False, sort=True):
        residual = pd.to_numeric(sub["y_true"], errors="coerce") - pd.to_numeric(sub["y_pred"], errors="coerce")
        family_total = int((frame["task_family"].astype(str) == str(family)).sum())
        rows.append(
            {
                "task_family": str(family),
                "cst_ad_tier": str(tier),
                "mae": float(residual.abs().mean()),
                "coverage_fraction": len(sub) / family_total if family_total else np.nan,
            }
        )
    data = pd.DataFrame(rows)
    if data.empty:
        return
    families = sorted(data["task_family"].unique())
    x = np.arange(len(TIER_ORDER))
    width = 0.78 / max(len(families), 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    for index, family in enumerate(families):
        sub = data[data["task_family"].eq(family)].set_index("cst_ad_tier").reindex(TIER_ORDER)
        offset = (index - (len(families) - 1) / 2) * width
        axes[0].bar(x + offset, sub["mae"], width=width, label=family)
        axes[1].bar(x + offset, sub["coverage_fraction"] * 100.0, width=width, label=family)
    axes[0].set_ylabel("MAE on log10 toxicity scale")
    axes[0].set_title("Prediction error by endpoint family")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_title("Coverage by endpoint family")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(TIER_ORDER)
        ax.set_xlabel("CST-AD tier")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8, title="Endpoint")
    save(fig, out_dir / f"{prefix}_tier_performance_by_task_family")


def plot_observed_vs_predicted(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    data = frame.copy()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for tier in TIER_ORDER:
        sub = data[data["cst_ad_tier"].astype(str).eq(tier)]
        if sub.empty:
            continue
        ax.scatter(
            pd.to_numeric(sub["y_true"], errors="coerce"),
            pd.to_numeric(sub["y_pred"], errors="coerce"),
            s=16,
            alpha=0.65,
            color=TIER_COLORS.get(tier, "#666666"),
            label=tier,
            linewidths=0,
        )
    values = pd.concat(
        [
            pd.to_numeric(data["y_true"], errors="coerce"),
            pd.to_numeric(data["y_pred"], errors="coerce"),
        ]
    ).dropna()
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.04
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#333333", linewidth=1.0)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Observed toxicity (pTox, log10 scale)")
    ax.set_ylabel("Predicted toxicity (pTox, log10 scale)")
    ax.set_title("Observed vs predicted by CST-AD tier")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    save(fig, out_dir / f"{prefix}_observed_vs_predicted")


def plot_uncertainty(frame: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if "cst_ensemble_sd" not in frame.columns or pd.to_numeric(frame["cst_ensemble_sd"], errors="coerce").dropna().empty:
        return
    data = frame.copy()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for tier in TIER_ORDER:
        sub = data[data["cst_ad_tier"].astype(str).eq(tier)]
        if sub.empty:
            continue
        ax.scatter(
            pd.to_numeric(sub["cst_ensemble_sd"], errors="coerce"),
            pd.to_numeric(sub["abs_error"], errors="coerce"),
            s=16,
            alpha=0.65,
            color=TIER_COLORS.get(tier, "#666666"),
            label=tier,
            linewidths=0,
        )
    if "cst_uncertainty_threshold" in data.columns:
        threshold = pd.to_numeric(data["cst_uncertainty_threshold"], errors="coerce").dropna()
        if not threshold.empty:
            ax.axvline(threshold.iloc[0], color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("5-seed ensemble prediction SD")
    ax.set_ylabel("Absolute error on log10 toxicity scale")
    ax.set_title("Uncertainty-error relationship")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    save(fig, out_dir / f"{prefix}_uncertainty_error")


def summarize_for_plot(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for tier, sub in frame.groupby("cst_ad_tier", sort=True):
        residual = pd.to_numeric(sub["y_true"], errors="coerce") - pd.to_numeric(sub["y_pred"], errors="coerce")
        rows.append(
            {
                "cst_ad_tier": tier,
                "coverage_fraction": len(sub) / total if total else np.nan,
                "mae": residual.abs().mean(),
            }
        )
    return pd.DataFrame(rows)


def save(fig: plt.Figure, stem: Path, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)).strip("_").lower()


if __name__ == "__main__":
    main()
