from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ANALYSIS_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ANALYSIS_DIR / "figures"
SOURCE_DIR = ANALYSIS_DIR / "figure_source_data"

PALETTE = {
    "pale_green": "#dfead0",
    "teal": "#75c3c8",
    "blue": "#3f86b9",
    "slate": "#738988",
    "gold": "#f1c84b",
    "threshold_gray": "#cfd6dc",
    "reliable_green": "#90ad98",
    "low_gray": "#b8bec3",
    "ink": "#263238",
}
SUPPORT_CMAP = LinearSegmentedColormap.from_list(
    "support_reference_style",
    [
        PALETTE["pale_green"],
        PALETTE["teal"],
        PALETTE["blue"],
        PALETTE["slate"],
        PALETTE["gold"],
    ],
)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    records = pd.read_parquet(ANALYSIS_DIR / "ad_record_level.parquet")
    test = records.loc[records["analysis_split"].eq("test")].copy()
    curves = pd.read_csv(ANALYSIS_DIR / "ad_coverage_error_curves.csv")
    rule = json.loads((ANALYSIS_DIR / "ad_rule_locked.json").read_text(encoding="utf-8"))
    success = json.loads((ANALYSIS_DIR / "ad_success_assessment.json").read_text(encoding="utf-8"))

    source_manifest = make_figure5(test, curves, rule, success)
    source_manifest.update(make_chemical_space_figure(records))
    (SOURCE_DIR / "figure_source_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(FIGURE_DIR)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def make_figure5(
    test: pd.DataFrame,
    curves: pd.DataFrame,
    rule: dict,
    success: dict,
) -> dict[str, dict]:
    width = 183 / 25.4
    fig = plt.figure(figsize=(width, width * 0.91), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        bottom=0.095,
        top=0.975,
        wspace=0.28,
        hspace=0.35,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a_support_projection(ax_a, test, rule)

    heat_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.12)
    heat_axes = [fig.add_subplot(heat_gs[0, index]) for index in range(3)]
    heat_source = panel_b_heatmaps(fig, heat_axes, test)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_c_coverage_error(ax_c, curves)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_d_error_distribution(ax_d, test)

    fig.text(
        0.50,
        0.018,
        "Support strata describe training-data density, not pass/fail model reliability",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=PALETTE["slate"],
    )

    base = FIGURE_DIR / "fig5_training_support_stratification"
    save_publication_figure(fig, base)
    plt.close(fig)

    panel_a = test.loc[test["structure_status"].eq("ok")].copy()
    if len(panel_a) > 1200:
        panel_a = panel_a.sample(1200, random_state=20260723)
    panel_a_columns = [
        "record_id",
        "model_head",
        "C_target",
        "B_exp",
        "B_bottleneck",
        "T_index",
        rule["local_high_column"],
        "ad_tier",
        "AE_M10",
    ]
    panel_a["point_size_T"] = 10.0 + 52.0 * panel_a["T_index"].astype(float)
    panel_a_columns.append("point_size_T")
    panel_a[panel_a_columns].to_csv(SOURCE_DIR / "fig5_panel_a_support_points.csv", index=False)
    heat_source.to_csv(SOURCE_DIR / "fig5_panel_b_heatmap_cells.csv", index=False)
    curves.to_csv(SOURCE_DIR / "fig5_panel_c_coverage_error_curves.csv", index=False)
    test[["record_id", "model_head", "ad_tier", "AE_M10", "NAE_M10"]].to_csv(
        SOURCE_DIR / "fig5_panel_d_error_distribution.csv", index=False
    )
    return {
        "fig5_training_support_stratification": {
            "conclusion": "Joint chemical, bio-context and task support provides continuous, modest reliability enrichment on the fixed random interpolation boundary rather than a binary reliability threshold.",
            "final_label": success["final_label"],
            "outputs": [str(base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")],
            "source_data": [
                str(SOURCE_DIR / "fig5_panel_a_support_points.csv"),
                str(SOURCE_DIR / "fig5_panel_b_heatmap_cells.csv"),
                str(SOURCE_DIR / "fig5_panel_c_coverage_error_curves.csv"),
                str(SOURCE_DIR / "fig5_panel_d_error_distribution.csv"),
            ],
        }
    }


def panel_a_support_projection(ax, frame: pd.DataFrame, rule: dict) -> None:
    t = rule["thresholds"]
    local = rule["local_high_column"]
    selected = frame.loc[frame["structure_status"].eq("ok")].copy()
    if len(selected) > 1200:
        selected = selected.sample(1200, random_state=20260723)
    selected = selected.sort_values([local, "T_index"], ascending=False)
    color = np.log10(selected[local].to_numpy(float) + 1.0)
    point_size = 10.0 + 52.0 * selected["T_index"].to_numpy(float)

    ax.add_patch(
        Rectangle(
            (t["c_high"], t["b_high"]),
            1.02 - t["c_high"],
            1.02 - t["b_high"],
            facecolor=PALETTE["pale_green"],
            edgecolor="none",
            alpha=0.28,
            zorder=0,
        )
    )
    ax.axvline(t["c_high"], color=PALETTE["threshold_gray"], linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(t["b_high"], color=PALETTE["threshold_gray"], linestyle="--", linewidth=0.8, zorder=1)
    scatter = ax.scatter(
        selected["C_target"],
        selected["B_bottleneck"],
        c=color,
        cmap=SUPPORT_CMAP,
        s=point_size,
        alpha=0.52,
        linewidths=0.25,
        edgecolors="white",
        rasterized=True,
        zorder=2,
    )
    exact_parent_pct = float((frame.loc[frame["structure_status"].eq("ok"), "C_target"] >= 1.0 - 1e-12).mean() * 100)
    ax.text(
        0.03,
        0.05,
        f"{exact_parent_pct:.1f}% exact-parent supported\namong structure-available records",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["slate"],
    )
    ax.text(
        0.72,
        0.835,
        "strict-support\nreference corner",
        fontsize=5.4,
        color=PALETTE["slate"],
    )

    ax.set_xlim(0.20, 1.02)
    ax.set_ylim(0.20, 1.02)
    ax.set_xlabel("Chemical support, C")
    ax.set_ylabel("Bio-context support, B")
    ax.set_xticks([0.25, 0.50, 0.70, 1.00])
    ax.set_yticks([0.25, 0.50, 0.80, 1.00])
    ax.grid(color="#dfe6ea", linewidth=0.45)
    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=np.sqrt(10.0 + 52.0 * value),
            markerfacecolor="white",
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.45,
            label=f"{value:.1f}",
        )
        for value in (0.2, 0.6, 0.9)
    ]
    ax.legend(
        handles=size_handles,
        title="Task support, T",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.00, 0.93),
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.6,
        fontsize=5.2,
        title_fontsize=5.4,
    )
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.040, pad=0.025, shrink=0.84)
    colorbar.set_label(r"log$_{10}$(local records + 1)", fontsize=5.6)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    clean_axes(ax)
    panel_label(ax, "a", "Chemical–bio-context support projection")


def panel_b_heatmaps(fig, axes: list, frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame["structure_status"].eq("ok")].copy()
    c_bins = [-np.inf, 0.70, 1.00, np.inf]
    c_labels = ["<0.70", "0.70–<1.00", "=1.00"]
    b_bins = [-np.inf, 0.80, 0.95, np.inf]
    b_labels = ["<0.80", "0.80–<0.95", "≥0.95"]
    work["C_bin"] = pd.cut(work["C_target"], c_bins, labels=c_labels, right=False)
    work["B_bin"] = pd.cut(work["B_bottleneck"], b_bins, labels=b_labels, right=False)
    work["T_group"] = np.select(
        [work["T_tier"].eq("T0"), work["T_tier"].isin(["T1", "T2"])],
        ["Low task support", "Mid task support"],
        default="High task support",
    )
    grouped = (
        work.groupby(["T_group", "B_bin", "C_bin"], observed=False)
        .agg(n=("record_id", "size"), median_AE=("AE_M10", "median"), mae=("AE_M10", "mean"))
        .reset_index()
    )
    order = ["Low task support", "Mid task support", "High task support"]
    image = None
    for index, (ax, group) in enumerate(zip(axes, order)):
        selected = grouped.loc[grouped["T_group"].eq(group)]
        median = selected.pivot(index="B_bin", columns="C_bin", values="median_AE").reindex(index=b_labels, columns=c_labels)
        count = selected.pivot(index="B_bin", columns="C_bin", values="n").reindex(index=b_labels, columns=c_labels).fillna(0)
        matrix = median.to_numpy(float)
        masked = np.ma.masked_where(count.to_numpy(float) < 15, matrix)
        image = ax.imshow(masked, origin="lower", cmap=SUPPORT_CMAP, vmin=0.20, vmax=0.65, aspect="auto")
        for row in range(3):
            for col in range(3):
                n = int(count.iloc[row, col])
                if 0 < n < 15:
                    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="#f0f2f3", edgecolor="#c2c9cd", hatch="///", linewidth=0.35))
                elif n == 0:
                    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor="white", edgecolor="#e2e7ea", linewidth=0.35))
                if n > 0:
                    ax.text(col, row, compact_count(n), ha="center", va="center", fontsize=5.0, color=PALETTE["ink"])
        ax.set_title(group.replace(" task support", ""), pad=3)
        ax.set_xticks(range(3), c_labels, rotation=55, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(b_labels if index == 0 else [])
        ax.tick_params(length=0, pad=1.5)
        ax.set_xlabel("C")
        if index == 0:
            ax.set_ylabel("B")
        for spine in ax.spines.values():
            spine.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.02, shrink=0.76)
        cbar.set_label("Test median absolute error", fontsize=6.0)
        cbar.ax.tick_params(labelsize=5.5, length=2)
    axes[0].text(-0.42, 1.13, "(b)", transform=axes[0].transAxes, fontweight="bold", fontsize=8.5)
    axes[0].text(-0.16, 1.13, "Observed support cells and conditional error", transform=axes[0].transAxes, fontweight="bold", fontsize=7.2)
    axes[1].text(
        0.50,
        -0.36,
        "color: median AE  •  number: records  •  hatched: n<15",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=5.1,
        color=PALETTE["slate"],
    )
    return grouped


def panel_c_coverage_error(ax, curves: pd.DataFrame) -> None:
    colors = {
        "chemical_only": PALETTE["blue"],
        "bio_context_only": PALETTE["teal"],
        "joint_CBTL": PALETTE["gold"],
    }
    labels = {
        "chemical_only": "Chemical only",
        "bio_context_only": "Bio-context only",
        "joint_CBTL": "Joint C+B+T+L",
    }
    split_rows = curves.loc[curves["split"].eq("test")]
    full_row = split_rows.iloc[(split_rows["retained_coverage"] - 1.0).abs().argsort()[:1]]
    full_mae = float(full_row["mae"].iloc[0])
    ax.axhspan(-0.05, 0.05, color=PALETTE["pale_green"], alpha=0.32, zorder=0)
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.75, zorder=1)
    for method in colors:
        selected = curves.loc[(curves["method"] == method) & (curves["split"] == "test")].sort_values("retained_coverage")
        ax.plot(
            selected["retained_coverage"] * 100,
            selected["mae"] - full_mae,
            color=colors[method],
            linewidth=1.55,
            marker="o",
            markersize=2.2,
            label=labels[method],
        )
    ax.set_xlabel("Retained coverage (%)")
    ax.set_ylabel("MAE difference from full test set")
    ax.set_xlim(8, 102)
    ax.set_ylim(-0.09, 0.035)
    ax.grid(axis="y", color="#dfe6ea", linewidth=0.55)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        handlelength=1.8,
        columnspacing=0.8,
    )
    ax.text(
        0.02,
        0.035,
        f"full-test MAE = {full_mae:.3f}\nshading: within ±0.05",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=PALETTE["slate"],
    )
    clean_axes(ax)
    panel_label(ax, "c", "Coverage changes error only modestly")


def panel_d_error_distribution(ax, frame: pd.DataFrame) -> None:
    order = ["High", "Moderate", "Low/outside"]
    display_labels = ["Strict high", "Intermediate", "Lower measured"]
    colors = [PALETTE["gold"], PALETTE["teal"], PALETTE["slate"]]
    values = [frame.loc[frame["ad_tier"].eq(tier), "AE_M10"].to_numpy(float) for tier in order]
    overall_mae = float(frame["AE_M10"].mean())
    ax.axhline(overall_mae, color=PALETTE["ink"], linewidth=0.8, linestyle="--", zorder=0)
    violin = ax.violinplot(values, positions=np.arange(3), widths=0.72, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(PALETTE["ink"])
        body.set_linewidth(0.45)
        body.set_alpha(0.72)
    box = ax.boxplot(
        values,
        positions=np.arange(3),
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["ink"], "linewidth": 1.0},
        whiskerprops={"color": PALETTE["ink"], "linewidth": 0.6},
        capprops={"color": PALETTE["ink"], "linewidth": 0.6},
        boxprops={"edgecolor": PALETTE["ink"], "linewidth": 0.55},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.72)
    rng = np.random.default_rng(20260723)
    for index, (tier, group, color) in enumerate(zip(order, values, colors)):
        sample = group if len(group) <= 180 else rng.choice(group, size=180, replace=False)
        jitter = rng.normal(index, 0.055, size=len(sample))
        ax.scatter(jitter, sample, s=3, color=color, alpha=0.28, linewidths=0, rasterized=True)
        mae = float(np.mean(group))
        p90 = float(np.quantile(group, 0.90))
        delta = mae - overall_mae
        ax.text(index, 2.18, f"n={len(group):,}\nMAE={mae:.3f}\nΔ={delta:+.3f}", ha="center", va="top", fontsize=5.6)
    ax.set_xticks(range(3), display_labels)
    ax.set_ylabel("Absolute prediction error")
    ax.set_ylim(0, 2.32)
    ax.grid(axis="y", color="#dfe6ea", linewidth=0.55)
    ax.text(
        0.98,
        overall_mae + 0.035,
        f"full-test MAE = {overall_mae:.3f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=PALETTE["slate"],
    )
    clean_axes(ax)
    panel_label(ax, "d", "Higher support improves reliability without strict separation")


def compact_count(value: int) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}k"
    return str(value)


def make_chemical_space_figure(records: pd.DataFrame) -> dict[str, dict]:
    stages = pd.read_csv(ANALYSIS_DIR / "chemical_space_stage_summary.csv")
    overlap = pd.read_csv(ANALYSIS_DIR / "chemical_space_overlap_summary.csv")
    test = records.loc[records["analysis_split"].eq("test")].copy()
    validation = records.loc[records["analysis_split"].eq("validation")].copy()

    width = 183 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width, width * 0.70), gridspec_kw={"wspace": 0.30, "hspace": 0.40})
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    labels = ["Stage 1\naquatic fit", "Stage 2\nbridge fit", "Stage 3\nsoil train", "Validation", "Outer test"]
    x = np.arange(len(stages))
    ax_a.bar(x, stages["n_records"], color=PALETTE["blue"], alpha=0.82, label="All records")
    ax_a.bar(x, stages["structure_valid_rows"], color=PALETTE["teal"], alpha=0.95, label="Structure available")
    ax_a.set_yscale("log")
    ax_a.set_xticks(x, labels, rotation=25, ha="right")
    ax_a.set_ylabel("Records (log scale)")
    ax_a.legend(frameon=False)
    clean_axes(ax_a)
    panel_label(ax_a, "a", "Cross-stage record inventory")

    pair = overlap.loc[
        overlap["summary_type"].eq("pairwise_unique_structure_overlap")
        & overlap["left"].eq("stage3_train")
        & overlap["right"].isin(["stage3_validation", "stage3_test"])
    ].copy()
    positions = np.arange(len(pair))
    width_bar = 0.34
    ax_b.bar(positions - width_bar / 2, pair["right_parent_seen_fraction"] * 100, width_bar, color=PALETTE["blue"], label="Canonical parent")
    ax_b.bar(positions + width_bar / 2, pair["right_scaffold_seen_fraction"] * 100, width_bar, color=PALETTE["gold"], label="Murcko scaffold")
    ax_b.set_xticks(positions, ["Validation", "Outer test"])
    ax_b.set_ylabel("Unique structures seen in Stage 3 train (%)")
    ax_b.set_ylim(0, 100)
    ax_b.legend(frameon=False)
    ax_b.grid(axis="y", color="#dfe6ea", linewidth=0.55)
    clean_axes(ax_b)
    panel_label(ax_b, "b", "Target-domain structural overlap")

    for frame, label, color, style in (
        (validation, "Validation → target train", PALETTE["teal"], "--"),
        (test, "Test → target train", PALETTE["blue"], "-"),
        (validation, "Validation → source fit", PALETTE["pale_green"], "--"),
        (test, "Test → source fit", PALETTE["gold"], "-"),
    ):
        column = "C_source" if "source" in label else "C_target"
        values = np.sort(frame.loc[frame["structure_status"].eq("ok"), column].dropna().to_numpy(float))
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax_c.plot(values, ecdf * 100, color=color, linestyle=style, linewidth=1.35, label=label)
    ax_c.set_xlabel("Maximum Tanimoto similarity")
    ax_c.set_ylabel("Cumulative records (%)")
    ax_c.set_xlim(0, 1.01)
    ax_c.grid(color="#dfe6ea", linewidth=0.5)
    ax_c.legend(frameon=False, loc="upper left")
    clean_axes(ax_c)
    panel_label(ax_c, "c", "Chemical-support distributions")

    categories = ["Exact parent", "Scaffold", "Structure unavailable"]
    val_values = [validation["exact_parent_seen_target"].fillna(False).mean(), validation["scaffold_seen_target"].fillna(False).mean(), validation["structure_status"].ne("ok").mean()]
    test_values = [test["exact_parent_seen_target"].fillna(False).mean(), test["scaffold_seen_target"].fillna(False).mean(), test["structure_status"].ne("ok").mean()]
    positions = np.arange(3)
    ax_d.bar(positions - width_bar / 2, np.array(val_values) * 100, width_bar, color=PALETTE["teal"], label="Validation")
    ax_d.bar(positions + width_bar / 2, np.array(test_values) * 100, width_bar, color=PALETTE["blue"], label="Outer test")
    ax_d.set_xticks(positions, categories, rotation=18, ha="right")
    ax_d.set_ylabel("Record fraction (%)")
    ax_d.set_ylim(0, 105)
    ax_d.legend(frameon=False)
    ax_d.grid(axis="y", color="#dfe6ea", linewidth=0.55)
    clean_axes(ax_d)
    panel_label(ax_d, "d", "Interpolation-dominated boundary")

    base = FIGURE_DIR / "fig_chemical_space_overview"
    save_publication_figure(fig, base)
    plt.close(fig)

    stages.to_csv(SOURCE_DIR / "chemical_space_stage_summary.csv", index=False)
    pair.to_csv(SOURCE_DIR / "chemical_space_stage3_overlap.csv", index=False)
    records[["record_id", "analysis_split", "structure_status", "C_target", "C_source", "exact_parent_seen_target", "scaffold_seen_target"]].to_csv(
        SOURCE_DIR / "chemical_space_record_support.csv", index=False
    )
    return {
        "fig_chemical_space_overview": {
            "conclusion": "The random Stage-3 boundary is chemically interpolation dominated, while about one quarter of records lack usable molecular structures and require a separate support state.",
            "outputs": [str(base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")],
            "source_data": [
                str(SOURCE_DIR / "chemical_space_stage_summary.csv"),
                str(SOURCE_DIR / "chemical_space_stage3_overlap.csv"),
                str(SOURCE_DIR / "chemical_space_record_support.csv"),
            ],
        }
    }


def save_publication_figure(fig, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def panel_label(ax, label: str, title: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(-0.10, 1.07, f"({label})", transform=ax.transAxes, fontweight="bold", fontsize=8.5)
    text_method(0.02, 1.07, title, transform=ax.transAxes, fontweight="bold", fontsize=7.2)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    main()
