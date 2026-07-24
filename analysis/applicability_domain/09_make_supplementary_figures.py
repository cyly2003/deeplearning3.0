from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ANALYSIS_DIR / "figures"
SOURCE_DIR = ANALYSIS_DIR / "figure_source_data"
PALETTE = {
    "pale_green": "#dfead0",
    "teal": "#75c3c8",
    "blue": "#3f86b9",
    "slate": "#738988",
    "gold": "#f1c84b",
    "low_gray": "#b8bec3",
    "ink": "#263238",
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    records = pd.read_parquet(ANALYSIS_DIR / "ad_record_level.parquet")
    test = records.loc[records["analysis_split"].eq("test")].copy()
    validation = records.loc[records["analysis_split"].eq("validation")].copy()
    transfer = pd.read_csv(ANALYSIS_DIR / "transfer_gain_by_support.csv")
    task = pd.read_csv(ANALYSIS_DIR / "ad_summary_by_task_test.csv")
    curves = pd.read_csv(ANALYSIS_DIR / "ad_coverage_error_curves.csv")

    width = 183 / 25.4
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(width, width * 1.18),
        gridspec_kw={"wspace": 0.34, "hspace": 0.52},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    tax_source = taxonomy_panel(ax_a, validation, test)
    k_source = context_k_panel(ax_b, validation, test)
    task_source = task_performance_panel(ax_c, task)
    transfer_source = transfer_gain_panel(ax_d, transfer)
    structure_source = structure_panel(ax_e, validation, test)
    uncertainty_source = uncertainty_panel(ax_f, test)

    base = FIGURE_DIR / "fig_supplementary_support_diagnostics"
    save_figure(fig, base)
    plt.close(fig)
    coverage_base, coverage_source = coverage_curve_figure(curves)

    sources = {
        "taxonomy": tax_source,
        "context_k": k_source,
        "task_performance": task_source,
        "transfer_gain": transfer_source,
        "structure_status": structure_source,
        "seed_disagreement": uncertainty_source,
    }
    paths = []
    for name, data in sources.items():
        path = SOURCE_DIR / f"supp_{name}.csv"
        data.to_csv(path, index=False)
        paths.append(str(path))
    manifest_path = SOURCE_DIR / "figure_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["fig_supplementary_support_diagnostics"] = {
        "conclusion": "Per-task performance remains broadly distributed around the overall model error, while measured support is a descriptive density signal rather than a task reliability label.",
        "outputs": [str(base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")],
        "source_data": paths,
    }
    coverage_path = SOURCE_DIR / "supp_coverage_error_curves.csv"
    coverage_source.to_csv(coverage_path, index=False)
    manifest["fig_supplementary_coverage_curves"] = {
        "conclusion": "Validation and outer-test coverage curves show that training support provides a modest ranking signal; seed disagreement is retained as a separate model-uncertainty diagnostic.",
        "outputs": [str(coverage_base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")],
        "source_data": [str(coverage_path)],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(base.with_suffix(".png"))
    print(coverage_base.with_suffix(".png"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.7,
            "text.color": PALETTE["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def taxonomy_panel(ax, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["none", "class", "order", "family", "genus", "species"]
    for split, frame in (("Validation", validation), ("Outer test", test)):
        counts = frame["tax_support_same_task"].fillna("none").value_counts()
        total = len(frame)
        for category in order:
            rows.append({"split": split, "taxonomy_support": category, "n": int(counts.get(category, 0)), "fraction": counts.get(category, 0) / total})
    source = pd.DataFrame(rows)
    bottom = np.zeros(2)
    colors = [PALETTE["low_gray"], PALETTE["pale_green"], PALETTE["teal"], PALETTE["blue"], PALETTE["slate"], PALETTE["gold"]]
    for category, color in zip(order, colors):
        values = source.loc[source["taxonomy_support"].eq(category)].set_index("split").reindex(["Validation", "Outer test"])["fraction"].to_numpy() * 100
        ax.bar([0, 1], values, bottom=bottom, color=color, width=0.62, label=category.title())
        bottom += values
    ax.set_xticks([0, 1], ["Validation", "Outer test"])
    ax.set_ylabel("Records (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.00), columnspacing=0.7, handlelength=1.1)
    grid_y(ax)
    panel_label(ax, "a", "Same-task taxonomic support")
    return source


def context_k_panel(ax, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in (("Validation", validation), ("Outer test", test)):
        for k in (1, 5, 10):
            similarity = 1.0 - frame[f"context_top{k}_mean_distance"].to_numpy(float)
            rows.extend({"split": split, "k": k, "context_similarity": value} for value in similarity)
    source = pd.DataFrame(rows)
    positions = [0.85, 1.15, 1.85, 2.15, 2.85, 3.15]
    data = []
    colors = []
    for k in (1, 5, 10):
        for split, color in (("Validation", PALETTE["teal"]), ("Outer test", PALETTE["blue"])):
            data.append(source.loc[(source["k"] == k) & (source["split"] == split), "context_similarity"].to_numpy())
            colors.append(color)
    box = ax.boxplot(data, positions=positions, widths=0.24, patch_artist=True, showfliers=False, medianprops={"color": PALETTE["ink"], "linewidth": 0.9})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.74)
    ax.set_xticks([1, 2, 3], ["k=1", "k=5", "k=10"])
    ax.set_ylabel("Bio-context similarity (1 − distance)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(
        [mpl.patches.Patch(color=PALETTE["teal"]), mpl.patches.Patch(color=PALETTE["blue"])],
        ["Validation", "Outer test"],
        frameon=False,
        loc="lower left",
    )
    grid_y(ax)
    panel_label(ax, "b", "Context-neighbor sensitivity")
    return source


def task_performance_panel(ax, task: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_head, group in task.groupby("task_head", sort=False):
        valid = group.loc[group["n"].gt(0) & group["mae"].notna()].copy()
        total_n = int(valid["n"].sum())
        overall_mae = float(np.average(valid["mae"], weights=valid["n"]))
        broad_coverage = float(valid.loc[valid["tier"].isin(["High", "Moderate"]), "n"].sum() / total_n)
        rows.append(
            {
                "task_head": task_head,
                "task_n": total_n,
                "task_mae": overall_mae,
                "broad_support_coverage": broad_coverage,
                "T_tier": group["T_tier"].iloc[0],
            }
        )
    source = pd.DataFrame(rows).sort_values("task_mae", ascending=True).reset_index(drop=True)
    overall = float(np.average(source["task_mae"], weights=source["task_n"]))
    source["broad_support_class"] = pd.cut(
        source["broad_support_coverage"],
        bins=[-np.inf, 0.35, 0.50, np.inf],
        labels=["<35%", "35–50%", ">50%"],
    ).astype("string")
    y = np.arange(len(source))
    class_colors = {"<35%": PALETTE["pale_green"], "35–50%": PALETTE["teal"], ">50%": PALETTE["blue"]}
    for category, color in class_colors.items():
        selected = source.loc[source["broad_support_class"].eq(category)]
        selected_y = selected.index.to_numpy()
        ax.scatter(
            selected["task_mae"],
            selected_y,
            color=color,
            s=18 + 52 * np.sqrt(selected["task_n"] / source["task_n"].max()),
            edgecolor=PALETTE["ink"],
            linewidth=0.45,
            label=category,
            zorder=3,
        )
    ax.axvline(overall, color=PALETTE["ink"], linewidth=0.75, linestyle="--")
    ax.set_yticks(y, [short_task_label(value) for value in source["task_head"]])
    ax.set_xlabel("Full-task test MAE")
    ax.set_xlim(0.24, 0.75)
    ax.grid(axis="x", color="#dfe6ea", linewidth=0.5)
    ax.legend(
        title="High + intermediate",
        frameon=False,
        loc="lower right",
        fontsize=4.8,
        title_fontsize=5.0,
        handletextpad=0.3,
        labelspacing=0.25,
    )
    ax.text(0.02, 0.98, f"dashed: overall MAE={overall:.3f}\npoint size scales with task n", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=PALETTE["slate"])
    clean_axes(ax)
    panel_label(ax, "c", "Task performance without pass/fail labels")
    return source


def transfer_gain_panel(ax, transfer: pd.DataFrame) -> pd.DataFrame:
    source = transfer.loc[transfer["dimension"].isin(["support_tier", "exact_source_seen"])].copy()
    order = [
        ("support_tier", "High", "Strict high"),
        ("support_tier", "Moderate", "Intermediate"),
        ("support_tier", "Low/outside", "Lower measured"),
        ("exact_source_seen", "seen", "Source exact seen"),
        ("exact_source_seen", "not_seen", "Source not seen"),
        ("exact_source_seen", "structure_unavailable", "Structure unavailable"),
    ]
    selected_rows = []
    for dimension, group, label in order:
        row = source.loc[(source["dimension"] == dimension) & (source["group"] == group)].iloc[0].copy()
        row["display_label"] = label
        selected_rows.append(row)
    plot = pd.DataFrame(selected_rows)
    y = np.arange(len(plot))
    mean = plot["mean_delta_AE"].to_numpy(float)
    low = plot["mean_delta_AE_ci95_low"].to_numpy(float)
    high = plot["mean_delta_AE_ci95_high"].to_numpy(float)
    colors = [PALETTE["gold"], PALETTE["teal"], PALETTE["blue"], PALETTE["slate"], PALETTE["pale_green"], PALETTE["low_gray"]]
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="none", ecolor=PALETTE["ink"], elinewidth=0.8, capsize=2)
    ax.scatter(mean, y, c=colors, s=26, edgecolor=PALETTE["ink"], linewidth=0.45, zorder=3)
    ax.axvline(0, color=PALETTE["ink"], linewidth=0.7, linestyle="--")
    ax.set_yticks(y, [f"{label}\n(n={n:,})" for label, n in zip(plot["display_label"], plot["n"])])
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean ΔAE (M10 − M00); negative favors M10")
    ax.grid(axis="x", color="#dfe6ea", linewidth=0.5)
    clean_axes(ax)
    panel_label(ax, "d", "Aqueous-pretraining gain")
    return source


def structure_panel(ax, validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, frame in (("Validation", validation), ("Outer test", test)):
        for status, label in (("ok", "Structure available"), ("unavailable", "Structure unavailable")):
            selected = frame.loc[frame["structure_status"].eq(status)]
            rows.append({"split": split, "structure": label, "n": len(selected), "coverage": len(selected) / len(frame), "mae": selected["AE_M10"].mean(), "median_ae": selected["AE_M10"].median()})
    source = pd.DataFrame(rows)
    positions = np.arange(2)
    width = 0.34
    for offset, split, color in ((-width / 2, "Validation", PALETTE["teal"]), (width / 2, "Outer test", PALETTE["blue"])):
        selected = source.loc[source["split"].eq(split)].set_index("structure").reindex(["Structure available", "Structure unavailable"])
        bars = ax.bar(positions + offset, selected["mae"], width, color=color, label=split)
        for bar, n in zip(bars, selected["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"n={int(n):,}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_xticks(positions, ["Available", "Unavailable"])
    ax.set_ylabel("Mean absolute error")
    ax.set_ylim(0, max(0.8, source["mae"].max() + 0.16))
    ax.legend(frameon=False)
    grid_y(ax)
    panel_label(ax, "e", "Structure-unavailable audit")
    return source


def uncertainty_panel(ax, test: pd.DataFrame) -> pd.DataFrame:
    order = ["High", "Moderate", "Low/outside"]
    display_labels = ["Strict high", "Intermediate", "Lower measured"]
    source = test[["record_id", "ad_tier", "M10_prediction_sd", "AE_M10"]].copy()
    data = [source.loc[source["ad_tier"].eq(tier), "M10_prediction_sd"].to_numpy() for tier in order]
    violin = ax.violinplot(data, positions=np.arange(3), showextrema=False, widths=0.75)
    for body, color in zip(violin["bodies"], [PALETTE["gold"], PALETTE["teal"], PALETTE["blue"]]):
        body.set_facecolor(color)
        body.set_edgecolor(PALETTE["ink"])
        body.set_alpha(0.72)
        body.set_linewidth(0.5)
    ax.boxplot(data, positions=np.arange(3), widths=0.18, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "alpha": 0.75}, medianprops={"color": PALETTE["ink"]})
    ax.set_xticks(np.arange(3), display_labels)
    ax.set_ylabel("Four-seed prediction SD")
    ax.set_ylim(bottom=0)
    grid_y(ax)
    panel_label(ax, "f", "Seed disagreement by support")
    return source


def coverage_curve_figure(curves: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    colors = {
        "chemical_only": PALETTE["blue"],
        "bio_context_only": PALETTE["teal"],
        "joint_CBTL": PALETTE["gold"],
        "seed_disagreement_only": PALETTE["slate"],
    }
    labels = {
        "chemical_only": "Chemical only",
        "bio_context_only": "Bio-context only",
        "joint_CBTL": "Joint C+B+T+L",
        "seed_disagreement_only": "Seed disagreement",
    }
    source = curves.loc[curves["method"].isin(colors)].copy()
    width = 183 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.42), gridspec_kw={"wspace": 0.27})
    fig.subplots_adjust(bottom=0.23)
    for ax, split, label, title in (
        (axes[0], "validation", "a", "Validation-only ranking"),
        (axes[1], "test", "b", "Locked outer-test ranking"),
    ):
        split_rows = source.loc[source["split"].eq(split)]
        full_row = split_rows.iloc[(split_rows["retained_coverage"] - 1.0).abs().argsort()[:1]]
        full_mae = float(full_row["mae"].iloc[0])
        ax.axhspan(-0.05, 0.05, color=PALETTE["pale_green"], alpha=0.32, zorder=0)
        ax.axhline(0, color=PALETTE["ink"], linewidth=0.75, zorder=1)
        for method, color in colors.items():
            selected = split_rows.loc[split_rows["method"].eq(method)].sort_values("retained_coverage")
            ax.plot(
                selected["retained_coverage"] * 100,
                selected["mae"] - full_mae,
                color=color,
                linewidth=1.35,
                marker="o",
                markersize=2.0,
                label=labels[method],
            )
        ax.set_xlim(8, 102)
        ax.set_ylim(-0.21, 0.06)
        ax.set_xlabel("Retained coverage (%)")
        ax.set_ylabel("MAE difference from full split" if split == "validation" else "")
        ax.grid(axis="y", color="#dfe6ea", linewidth=0.5)
        ax.text(
            0.98,
            0.04,
            f"full-split MAE = {full_mae:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.4,
            color=PALETTE["slate"],
        )
        clean_axes(ax)
        panel_label(ax, label, title)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        handlelength=1.8,
        columnspacing=0.9,
    )
    base = FIGURE_DIR / "fig_supplementary_coverage_curves"
    save_figure(fig, base)
    plt.close(fig)
    return base, source


def short_task_label(value: str) -> str:
    endpoint, _, rest = value.partition("__")
    basis = "mol/kg" if "mol_kg" in rest else rest.replace("solid_", "")
    return f"{endpoint} | {basis}"


def save_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def panel_label(ax, label: str, title: str) -> None:
    ax.text(-0.15, 1.08, f"({label})", transform=ax.transAxes, fontweight="bold", fontsize=8.5)
    ax.text(0.02, 1.08, title, transform=ax.transAxes, fontweight="bold", fontsize=7.2)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grid_y(ax) -> None:
    ax.grid(axis="y", color="#dfe6ea", linewidth=0.5)
    clean_axes(ax)


if __name__ == "__main__":
    main()
