from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_ROWS = Path("实验汇总/08_随机主线CST应用域/random_mainline_cst_ad_prediction_rows.csv")
DEFAULT_OUT_DIR = Path("outputs/paper_figures/fig_model_results_integrated_20260709/application_domain")
DEFAULT_PREFIX = "application_domain_cst_3d_starry_night"
DEFAULT_PALETTE = Path(
    "outputs/paper_figures/fig_model_results_integrated_20260709/performance/figure_palette_starry_night_v1.yaml"
)

CHEMICAL_THRESHOLD = 0.50
CHEMICAL_HIGH_THRESHOLD = 0.70
SPECIES_RAW_THRESHOLD = 0.80
SPECIES_TRUST_THRESHOLD = 0.85
TASK_TRUST_THRESHOLD = 0.70
TASK_MEDIUM_THRESHOLD = 0.35

SPECIES_COORD = {
    "S0": 1.00,  # species-task-family seen
    "S1": 0.85,  # species seen
    "S2": 0.62,  # genus/family seen
    "S3": 0.45,  # order seen or taxonomy similarity threshold reached
    "S4": 0.15,  # low taxonomy coverage
}
TASK_COORD = {
    "T0": 1.00,  # chemical-species-task exact seen
    "T1": 0.70,  # species-task head/family seen
    "T2": 0.35,  # task head/family only
    "T3": 0.05,  # task unseen
}

STARRY_DEFAULTS = {
    "midnight_navy": "#182D4F",
    "ultramarine": "#264A8A",
    "cobalt_blue": "#2F6DB3",
    "swirl_cyan": "#6AAED6",
    "star_yellow": "#F2C94C",
    "moon_gold": "#D8A528",
    "warm_orange": "#D8892B",
    "cypress_green": "#315B4B",
    "lavender_shadow": "#7C6FA6",
    "night_gray": "#7B8491",
    "ivory": "#F7F1D0",
    "paper": "#FFFFFF",
    "grid": "#DDE3EA",
    "charcoal": "#20262B",
}

BACKGROUND = STARRY_DEFAULTS["paper"]
PANEL = STARRY_DEFAULTS["paper"]
PANE = "#F5F8FB"
GRID = STARRY_DEFAULTS["grid"]
TEXT = STARRY_DEFAULTS["charcoal"]
MUTED = STARRY_DEFAULTS["night_gray"]
TRUST = STARRY_DEFAULTS["cypress_green"]
THRESHOLD = STARRY_DEFAULTS["charcoal"]

AD_COLORS = {
    "AD-A": STARRY_DEFAULTS["cobalt_blue"],
    "AD-B": STARRY_DEFAULTS["lavender_shadow"],
    "AD-C": STARRY_DEFAULTS["moon_gold"],
    "AD-D": STARRY_DEFAULTS["warm_orange"],
}
CHEM_COLORS = {
    "C0": STARRY_DEFAULTS["cobalt_blue"],
    "C1": STARRY_DEFAULTS["star_yellow"],
    "C2": STARRY_DEFAULTS["warm_orange"],
}
SPECIES_COLORS = {
    "S0": STARRY_DEFAULTS["cobalt_blue"],
    "S1": STARRY_DEFAULTS["swirl_cyan"],
    "S2": STARRY_DEFAULTS["lavender_shadow"],
    "S3": STARRY_DEFAULTS["moon_gold"],
    "S4": STARRY_DEFAULTS["warm_orange"],
}
TASK_COLORS = {
    "T0": STARRY_DEFAULTS["cypress_green"],
    "T1": STARRY_DEFAULTS["cobalt_blue"],
    "T2": STARRY_DEFAULTS["moon_gold"],
    "T3": STARRY_DEFAULTS["warm_orange"],
}

ROLE_FIELDS = {
    "chemical": "chemical_axis",
    "taxonomic": "species_axis",
    "task": "task_axis",
}

ROLE_RELIABLE_MIN = {
    "chemical": CHEMICAL_THRESHOLD,
    "taxonomic": SPECIES_TRUST_THRESHOLD,
    "task": TASK_TRUST_THRESHOLD,
}

ROLE_LABELS = {
    "chemical": "Chemical support C",
    "taxonomic": "Taxonomic support S",
    "task": "Task-head support T",
}

ROLE_SYMBOLS = {
    "chemical": "C",
    "taxonomic": "S",
    "task": "T",
}

ROLE_TICKS = {
    "chemical": ([0.0, CHEMICAL_THRESHOLD, CHEMICAL_HIGH_THRESHOLD, 1.0], ["0", "0.50", "0.70", "1.00"]),
    "taxonomic": ([0.15, 0.45, 0.62, SPECIES_TRUST_THRESHOLD, 1.0], ["S4", "S3", "S2", "S1", "S0"]),
    "task": ([0.05, TASK_MEDIUM_THRESHOLD, TASK_TRUST_THRESHOLD, 1.0], ["T3", "T2", "T1", "T0"]),
}

PANEL_SPECS = {
    "overview": {
        "roles": ("chemical", "taxonomic", "task"),
        "legend_title": "CST support surface",
        "emphasized_role": "task",
    },
    "species": {
        "roles": ("chemical", "task", "taxonomic"),
        "legend_title": "Taxonomic support surface",
        "emphasized_role": "taxonomic",
    },
    "chemical": {
        "roles": ("taxonomic", "task", "chemical"),
        "legend_title": "Chemical support surface",
        "emphasized_role": "chemical",
    },
    "task": {
        "roles": ("chemical", "taxonomic", "task"),
        "legend_title": "Task-head support surface",
        "emphasized_role": "task",
    },
}

FIG_TITLE_SIZE = 11.2
PANEL_TITLE_SIZE = 10.0
AXIS_LABEL_SIZE = 7.8
TICK_LABEL_SIZE = 7.0
COLORBAR_LABEL_SIZE = 7.4
COLORBAR_TICK_SIZE = 6.9
LEGEND_TEXT_SIZE = 7.2
PEAK_LABEL_SIZE = 6.4
PEAK_MARKER_SIZE = 82


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a starry 3D CST application-domain overview and axis-focused panels."
    )
    parser.add_argument("--rows", default=str(DEFAULT_ROWS), help="CST-AD prediction rows CSV.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for figures.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output filename prefix.")
    parser.add_argument("--palette", default=str(DEFAULT_PALETTE), help="Starry Night palette YAML.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG export DPI.")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = load_starry_palette(Path(args.palette))
    apply_palette(palette)
    configure_matplotlib()
    rows = pd.read_csv(rows_path)
    plot_data = prepare_plot_data(rows)

    outputs: list[str] = []
    outputs.extend(plot_composite(plot_data, out_dir=out_dir, prefix=args.prefix, dpi=args.dpi))
    for mode, stem, title in [
        ("overview", "a_overview", "(a) Overview | C+S+T"),
        ("species", "b_taxonomy_layer", "(b) Taxonomic layer"),
        ("chemical", "c_chemical_layer", "(c) Chemical layer"),
        ("task", "d_task_head_layer", "(d) Task-head layer"),
    ]:
        outputs.extend(
            plot_single_panel(
                plot_data,
                mode=mode,
                title=title,
                out_base=out_dir / f"{args.prefix}_{stem}",
                dpi=args.dpi,
            )
        )

    plot_csv = out_dir / f"{args.prefix}_plot_data.csv"
    plot_data.to_csv(plot_csv, index=False, encoding="utf-8-sig")
    outputs.append(str(plot_csv))

    caption_doc = write_caption_doc(out_dir, args.prefix, rows_path, plot_data)
    outputs.append(str(caption_doc))

    manifest_path = out_dir / f"{args.prefix}_manifest.json"
    manifest = {
        "source_rows": str(rows_path),
        "source_n": int(len(rows)),
        "aggregated_points_n": int(len(plot_data)),
        "coordinate_note": "3D coordinates are tier-aligned support coordinates, not new model scores.",
        "palette": str(Path(args.palette)),
        "thresholds": {
            "chemical_threshold": CHEMICAL_THRESHOLD,
            "chemical_high_threshold": CHEMICAL_HIGH_THRESHOLD,
            "species_raw_threshold": SPECIES_RAW_THRESHOLD,
            "species_trusted_tier_boundary": SPECIES_TRUST_THRESHOLD,
            "task_trusted_tier_boundary": TASK_TRUST_THRESHOLD,
            "task_medium_boundary": TASK_MEDIUM_THRESHOLD,
        },
        "coordinate_mapping": {
            "chemical": "raw cst_chemical_score, with Williams-domain C1 lifted to >=0.55 for tier-aligned plotting",
            "species": SPECIES_COORD,
            "task": TASK_COORD,
        },
        "surface_note": "Surface geometry uses sample-count-weighted inverse-distance interpolation from aggregated CST-AD support cells; it is a visualization summary, not a new model output.",
        "surface_color_note": "Surface color maps local log10(sample_n + 1) density with robust 5th-85th percentile normalization of the interpolated density surface; the shared horizontal colorbar provides numeric log-density ticks.",
        "trusted_application_domain": "C0/C1 + S0/S1 + T0/T1, shown as translucent corner color patches rather than a wireframe cuboid.",
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(str(manifest_path))

    print(json.dumps({"outputs": outputs}, ensure_ascii=False, indent=2))


def load_starry_palette(path: Path) -> dict[str, str]:
    colors = dict(STARRY_DEFAULTS)
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        colors.update(payload.get("colors", {}) or {})
    return colors


def apply_palette(colors: dict[str, str]) -> None:
    global BACKGROUND, PANEL, PANE, GRID, TEXT, MUTED, TRUST, THRESHOLD
    BACKGROUND = colors["paper"]
    PANEL = colors["paper"]
    PANE = "#F5F8FB"
    GRID = colors["grid"]
    TEXT = colors["charcoal"]
    MUTED = colors["night_gray"]
    TRUST = colors["cypress_green"]
    THRESHOLD = colors["charcoal"]

    AD_COLORS.update(
        {
            "AD-A": colors["cobalt_blue"],
            "AD-B": colors["lavender_shadow"],
            "AD-C": colors["moon_gold"],
            "AD-D": colors["warm_orange"],
        }
    )
    CHEM_COLORS.update(
        {
            "C0": colors["cobalt_blue"],
            "C1": colors["star_yellow"],
            "C2": colors["warm_orange"],
        }
    )
    SPECIES_COLORS.update(
        {
            "S0": colors["cobalt_blue"],
            "S1": colors["swirl_cyan"],
            "S2": colors["lavender_shadow"],
            "S3": colors["moon_gold"],
            "S4": colors["warm_orange"],
        }
    )
    TASK_COLORS.update(
        {
            "T0": colors["cypress_green"],
            "T1": colors["cobalt_blue"],
            "T2": colors["moon_gold"],
            "T3": colors["warm_orange"],
        }
    )


def configure_matplotlib() -> None:
    font_path = first_existing(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
        ]
    )
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BACKGROUND,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 0.85,
        }
    )


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def prepare_plot_data(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    data["chemical_axis"] = chemical_support_coordinate(data)
    data["species_axis"] = data["cst_species_tier"].astype(str).map(SPECIES_COORD).fillna(0.05)
    data["task_axis"] = data["cst_task_tier"].astype(str).map(TASK_COORD).fillna(0.05)
    data["abs_error"] = pd.to_numeric(data["abs_error"], errors="coerce")

    group_columns = [
        "split_policy",
        "task_family",
        "cst_ad_tier",
        "cst_chemical_tier",
        "cst_species_tier",
        "cst_task_tier",
        "chemical_axis",
        "species_axis",
        "task_axis",
    ]
    grouped = (
        data.dropna(subset=["chemical_axis", "species_axis", "task_axis"])
        .assign(chemical_axis=lambda frame: frame["chemical_axis"].round(3))
        .groupby(group_columns, dropna=False, as_index=False)
        .agg(
            n=("sample_id", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_task_head_n=("cst_train_n_task_head", "median"),
        )
    )

    rng = np.random.default_rng(20260710)
    grouped["x_plot"] = np.clip(grouped["chemical_axis"] + rng.uniform(-0.008, 0.008, len(grouped)), 0.0, 1.0)
    grouped["y_plot"] = np.clip(grouped["species_axis"] + rng.uniform(-0.018, 0.018, len(grouped)), 0.0, 1.0)
    grouped["z_plot"] = np.clip(grouped["task_axis"] + rng.uniform(-0.018, 0.018, len(grouped)), 0.0, 1.0)
    sqrt_n = np.sqrt(grouped["n"].astype(float))
    grouped["marker_size"] = 24.0 + 90.0 * sqrt_n / max(float(sqrt_n.max()), 1.0)
    return grouped.sort_values(["cst_ad_tier", "cst_chemical_tier", "cst_species_tier", "cst_task_tier"]).reset_index(drop=True)


def chemical_support_coordinate(data: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(data["cst_chemical_score"], errors="coerce").clip(0.0, 1.0)
    tier = data["cst_chemical_tier"].astype(str)
    coord = raw.copy()
    coord = coord.mask(tier.eq("C0"), np.maximum(coord, CHEMICAL_HIGH_THRESHOLD))
    coord = coord.mask(tier.eq("C1"), np.maximum(coord, 0.55))
    coord = coord.mask(tier.eq("C2"), np.minimum(coord, 0.30))
    return coord.fillna(0.0)


def plot_composite(plot_data: pd.DataFrame, *, out_dir: Path, prefix: str, dpi: int) -> list[str]:
    fig = plt.figure(figsize=mm_to_inch(178, 160), facecolor=BACKGROUND)
    density_norm = sample_density_norm(plot_data)
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]
    specs = [
        ("overview", "(a) Overview | C+S+T"),
        ("species", "(b) Taxonomic layer"),
        ("chemical", "(c) Chemical layer"),
        ("task", "(d) Task-head layer"),
    ]
    for ax, (mode, title) in zip(axes, specs):
        plot_panel(ax, plot_data, mode=mode, title=title, density_norm=density_norm)

    fig.text(
        0.5,
        0.975,
        "CST application domain across chemical, taxonomic, and task-head support",
        ha="center",
        va="center",
        color=TEXT,
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
    )
    add_figure_legend(
        fig,
        density_norm=density_norm,
        cbar_rect=(0.17, 0.047, 0.47, 0.017),
        legend_anchor=(0.790, 0.051),
        fontsize=LEGEND_TEXT_SIZE,
        labelsize=COLORBAR_LABEL_SIZE,
    )
    fig.subplots_adjust(left=0.020, right=0.980, top=0.895, bottom=0.138, wspace=0.00, hspace=0.18)
    return save_figure(fig, out_dir / prefix, dpi=dpi)


def plot_single_panel(
    plot_data: pd.DataFrame,
    *,
    mode: str,
    title: str,
    out_base: Path,
    dpi: int,
) -> list[str]:
    fig = plt.figure(figsize=mm_to_inch(88, 78), facecolor=BACKGROUND)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    density_norm = sample_density_norm(plot_data)
    plot_panel(ax, plot_data, mode=mode, title=title, density_norm=density_norm)
    add_figure_legend(
        fig,
        density_norm=density_norm,
        cbar_rect=(0.15, 0.055, 0.50, 0.024),
        legend_anchor=(0.805, 0.062),
        fontsize=6.2,
        labelsize=6.0,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.185)
    return save_figure(fig, out_base, dpi=dpi, tight=True)


def mm_to_inch(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / 25.4, height_mm / 25.4


def plot_panel(ax: plt.Axes, plot_data: pd.DataFrame, *, mode: str, title: str, density_norm: Normalize) -> None:
    spec = PANEL_SPECS[mode]
    style_3d_axis(ax, spec)
    add_threshold_planes(ax, spec=spec, focus=mode)
    add_reliable_domain_blocks(ax, spec=spec, include_hatching=False)

    x_role, y_role, z_role = spec["roles"]
    x, y, z, density = interpolated_surface(
        plot_data,
        x_field=ROLE_FIELDS[x_role],
        y_field=ROLE_FIELDS[y_role],
        z_field=ROLE_FIELDS[z_role],
    )
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=density_cmap()(density_norm(density)),
        rstride=1,
        cstride=1,
        linewidth=0.55,
        edgecolor=matplotlib.colors.to_rgba(BACKGROUND, 0.92),
        antialiased=True,
        shade=False,
        alpha=0.96,
    )
    add_peak_marker(ax, x, y, z, z_role)
    add_reliable_domain_hatching(ax, spec=spec)

    ax.text2D(
        0.02,
        0.965,
        title,
        transform=ax.transAxes,
        color=TEXT,
        fontsize=PANEL_TITLE_SIZE,
        fontweight="bold",
    )


def interpolated_surface(
    data: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    z_field: str,
    grid_n: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_obs = pd.to_numeric(data[x_field], errors="coerce").to_numpy(dtype=float)
    y_obs = pd.to_numeric(data[y_field], errors="coerce").to_numpy(dtype=float)
    z_obs = pd.to_numeric(data[z_field], errors="coerce").to_numpy(dtype=float)
    n_raw = pd.to_numeric(data["n"], errors="coerce").fillna(1).to_numpy(dtype=float)
    keep = np.isfinite(x_obs) & np.isfinite(y_obs) & np.isfinite(z_obs)
    x_obs = x_obs[keep]
    y_obs = y_obs[keep]
    z_obs = z_obs[keep]
    n_obs = np.maximum(n_raw[keep], 1.0)
    support_weights = np.sqrt(n_obs)
    log_n_obs = np.log10(n_obs + 1.0)

    x_grid = np.linspace(0.0, 1.0, grid_n)
    y_grid = np.linspace(0.0, 1.0, grid_n)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    z_mesh = np.zeros_like(x_mesh, dtype=float)
    density_mesh = np.zeros_like(x_mesh, dtype=float)
    for index in np.ndindex(x_mesh.shape):
        dx = x_obs - x_mesh[index]
        dy = y_obs - y_mesh[index]
        dist2 = dx * dx + dy * dy
        support_kernel = support_weights / np.power(dist2 + 0.006, 1.15)
        density_kernel = 1.0 / np.power(dist2 + 0.006, 1.10)
        z_mesh[index] = float(np.sum(support_kernel * z_obs) / np.sum(support_kernel)) if np.sum(support_kernel) else np.nan
        density_mesh[index] = (
            float(np.sum(density_kernel * log_n_obs) / np.sum(density_kernel)) if np.sum(density_kernel) else np.nan
        )
    return x_mesh, y_mesh, np.clip(z_mesh, 0.0, 1.0), density_mesh


def surface_cmap(mode: str) -> LinearSegmentedColormap:
    if mode == "chemical":
        colors = [STARRY_DEFAULTS["ivory"], CHEM_COLORS["C1"], CHEM_COLORS["C0"]]
    elif mode == "species":
        colors = [STARRY_DEFAULTS["ivory"], SPECIES_COLORS["S1"], SPECIES_COLORS["S0"]]
    elif mode == "task":
        colors = [STARRY_DEFAULTS["ivory"], TASK_COLORS["T2"], TASK_COLORS["T0"]]
    else:
        colors = [STARRY_DEFAULTS["ivory"], STARRY_DEFAULTS["swirl_cyan"], STARRY_DEFAULTS["cobalt_blue"]]
    return LinearSegmentedColormap.from_list(f"{mode}_starry_surface", colors)


def density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "starry_sample_density",
        [
            "#EAF2D8",
            "#8FCBD3",
            STARRY_DEFAULTS["cobalt_blue"],
            STARRY_DEFAULTS["star_yellow"],
        ],
        N=256,
    )


def sample_density_norm(data: pd.DataFrame) -> Normalize:
    density_values: list[float] = []
    for spec in PANEL_SPECS.values():
        x_role, y_role, z_role = spec["roles"]
        _, _, _, density = interpolated_surface(
            data,
            x_field=ROLE_FIELDS[x_role],
            y_field=ROLE_FIELDS[y_role],
            z_field=ROLE_FIELDS[z_role],
        )
        density_values.extend(np.asarray(density, dtype=float).ravel().tolist())
    values = np.asarray(density_values, dtype=float)
    values = values[np.isfinite(values)]
    vmin = float(np.nanquantile(values, 0.05)) if len(values) else 0.0
    vmax = float(np.nanquantile(values, 0.85)) if len(values) else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin + 1.0, 1.0)
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def add_peak_marker(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, z_role: str) -> None:
    idx = np.unravel_index(np.nanargmax(z), z.shape)
    ax.scatter(
        [x[idx]],
        [y[idx]],
        [z[idx]],
        marker="*",
        s=PEAK_MARKER_SIZE,
        c=STARRY_DEFAULTS["star_yellow"],
        edgecolors=TEXT,
        linewidths=0.45,
        depthshade=False,
        zorder=20,
    )
    ax.text(
        float(x[idx]),
        float(y[idx]),
        min(float(z[idx]) + 0.05, 1.02),
        f"max {ROLE_SYMBOLS[z_role]}={z[idx]:.2f}",
        color=TEXT,
        fontsize=PEAK_LABEL_SIZE,
        fontweight="bold",
    )


def style_3d_axis(ax: plt.Axes, spec: dict[str, Any]) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_zlim(0.0, 1.02)
    ax.view_init(elev=23, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    x_role, y_role, z_role = spec["roles"]
    ax.set_xlabel(ROLE_LABELS[x_role], color=TEXT, labelpad=2)
    ax.set_ylabel(ROLE_LABELS[y_role], color=TEXT, labelpad=2)
    ax.set_zlabel(ROLE_LABELS[z_role], color=TEXT, labelpad=2)
    x_ticks, x_labels = ROLE_TICKS[x_role]
    y_ticks, y_labels = ROLE_TICKS[y_role]
    z_ticks, z_labels = ROLE_TICKS[z_role]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, color=MUTED)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color=MUTED)
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=TICK_LABEL_SIZE, pad=1)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.zaxis.label.set_size(AXIS_LABEL_SIZE)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(matplotlib.colors.to_rgba(PANE, 0.72))
        axis.pane.set_edgecolor(matplotlib.colors.to_rgba(GRID, 0.95))
        axis._axinfo["grid"]["color"] = matplotlib.colors.to_rgba(GRID, 0.92)
        axis._axinfo["grid"]["linewidth"] = 0.48
        axis._axinfo["tick"]["color"] = matplotlib.colors.to_rgba(MUTED, 0.95)
        axis._axinfo["axisline"]["color"] = matplotlib.colors.to_rgba(MUTED, 0.95)


def add_threshold_planes(ax: plt.Axes, *, spec: dict[str, Any], focus: str) -> None:
    weak_alpha = 0.045
    strong_alpha = 0.145
    role_thresholds = {
        "chemical": [
            (CHEMICAL_THRESHOLD, CHEM_COLORS["C1"], "C=0.50"),
            (CHEMICAL_HIGH_THRESHOLD, AD_COLORS["AD-A"], "C=0.70"),
        ],
        "taxonomic": [(SPECIES_TRUST_THRESHOLD, SPECIES_COLORS["S2"], "S0/S1")],
        "task": [(TASK_TRUST_THRESHOLD, TASK_COLORS["T0"], "T0/T1")],
    }
    for role in spec["roles"]:
        for value, color, _label in role_thresholds.get(role, []):
            alpha = strong_alpha if focus in {"overview", spec.get("emphasized_role")} else weak_alpha
            add_role_plane(ax, spec=spec, role=role, value=value, color=color, alpha=alpha)


def add_role_plane(ax: plt.Axes, *, spec: dict[str, Any], role: str, value: float, color: str, alpha: float) -> None:
    axis_index = spec["roles"].index(role)
    a, b = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    coords = [None, None, None]
    coords[axis_index] = np.full_like(a, value)
    remaining = [idx for idx in range(3) if idx != axis_index]
    coords[remaining[0]] = a
    coords[remaining[1]] = b
    ax.plot_surface(coords[0], coords[1], coords[2], color=color, alpha=alpha, linewidth=0, shade=False)


def add_reliable_domain_blocks(ax: plt.Axes, *, spec: dict[str, Any], include_hatching: bool = True) -> None:
    role_bounds = {role: (ROLE_RELIABLE_MIN[role], 1.0) for role in ("chemical", "taxonomic", "task")}
    axis_bounds = [role_bounds[role] for role in spec["roles"]]

    x_min, y_min, z_min = (bounds[0] for bounds in axis_bounds)
    x_max = y_max = z_max = 1.0
    eps = 0.006
    faces = [
        # Trusted-domain projection on the base pane.
        [(x_min, y_min, eps), (x_max, y_min, eps), (x_max, y_max, eps), (x_min, y_max, eps)],
        # Threshold-start patches on the three positive support faces.
        [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min)],
        [(x_min, y_min, z_min), (x_min, y_max, z_min), (x_min, y_max, z_max), (x_min, y_min, z_max)],
        [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_min, z_max), (x_min, y_min, z_max)],
    ]
    facecolors = [
        matplotlib.colors.to_rgba(TRUST, 0.13),
        matplotlib.colors.to_rgba(TRUST, 0.20),
        matplotlib.colors.to_rgba(STARRY_DEFAULTS["swirl_cyan"], 0.16),
        matplotlib.colors.to_rgba(STARRY_DEFAULTS["moon_gold"], 0.14),
    ]
    blocks = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolor=matplotlib.colors.to_rgba(TRUST, 0.45),
        linewidth=0.35,
    )
    ax.add_collection3d(blocks)
    if include_hatching:
        add_reliable_domain_hatching(ax, spec=spec)


def add_reliable_domain_hatching(ax: plt.Axes, *, spec: dict[str, Any]) -> None:
    role_bounds = {role: (ROLE_RELIABLE_MIN[role], 1.0) for role in ("chemical", "taxonomic", "task")}
    axis_bounds = [role_bounds[role] for role in spec["roles"]]
    x_min, y_min, z_min = (bounds[0] for bounds in axis_bounds)
    z = z_min + 0.010
    for offset in np.linspace(y_min - 1.0, 1.0 - x_min, 8):
        x_low = max(x_min, y_min - offset)
        x_high = min(1.0, 1.0 - offset)
        if x_high <= x_low:
            continue
        xs = np.array([x_low, x_high])
        ys = xs + offset
        ax.plot(xs, ys, np.full_like(xs, z), color=TRUST, alpha=0.62, linewidth=0.45)


def add_figure_legend(
    fig: plt.Figure,
    *,
    density_norm: Normalize,
    cbar_rect: tuple[float, float, float, float],
    legend_anchor: tuple[float, float],
    fontsize: float,
    labelsize: float,
) -> None:
    cax = fig.add_axes(cbar_rect)
    scalar = matplotlib.cm.ScalarMappable(norm=density_norm, cmap=density_cmap())
    scalar.set_array([])
    ticks = np.linspace(float(density_norm.vmin), float(density_norm.vmax), 4)
    cbar = fig.colorbar(scalar, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.set_label("Surface color: log10(local sample n + 1)", color=TEXT, fontsize=labelsize, fontweight="bold")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, colors=MUTED, length=2.2, width=0.65, pad=1.8)
    cbar.ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    cbar.outline.set_edgecolor(matplotlib.colors.to_rgba(GRID, 0.95))
    cbar.outline.set_linewidth(0.65)

    threshold_handle = Patch(facecolor=matplotlib.colors.to_rgba(THRESHOLD, 0.16), edgecolor="none")
    trust_handle = Patch(facecolor=matplotlib.colors.to_rgba(TRUST, 0.26), edgecolor=TRUST, linewidth=0.9)
    legend = fig.legend(
        [threshold_handle, trust_handle],
        ["Threshold planes", "Reliable AD"],
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        fontsize=fontsize,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=1.05,
        borderpad=0.35,
    )
    legend.get_frame().set_facecolor(matplotlib.colors.to_rgba(BACKGROUND, 0.92))
    legend.get_frame().set_edgecolor(matplotlib.colors.to_rgba(GRID, 0.95))
    for text in legend.get_texts():
        text.set_color(TEXT)


def write_caption_doc(out_dir: Path, prefix: str, rows_path: Path, plot_data: pd.DataFrame) -> Path:
    doc_path = out_dir / f"{prefix}_中文图注与解读.md"
    total_n = int(plot_data["n"].sum())
    reliable_mask = (
        (plot_data["chemical_axis"] >= CHEMICAL_THRESHOLD)
        & (plot_data["species_axis"] >= SPECIES_TRUST_THRESHOLD)
        & (plot_data["task_axis"] >= TASK_TRUST_THRESHOLD)
    )
    reliable_n = int(plot_data.loc[reliable_mask, "n"].sum())
    reliable_pct = 100.0 * reliable_n / total_n if total_n else 0.0
    tier_summary = (
        plot_data.groupby("cst_ad_tier", dropna=False)["n"]
        .sum()
        .sort_index()
        .reset_index()
        .rename(columns={"cst_ad_tier": "tier"})
    )
    tier_lines = "\n".join(
        f"| {row.tier} | {int(row.n):,} | {100.0 * float(row.n) / total_n:.1f}% |"
        for row in tier_summary.itertuples(index=False)
    )

    content = f"""# CST 三轴应用域图中文图注与解读

## 图件与数据

- 图件目录：`{out_dir}`
- 主图：`{prefix}.png` / `{prefix}.svg`
- 分览图：`{prefix}_a_overview`、`{prefix}_b_taxonomy_layer`、`{prefix}_c_chemical_layer`、`{prefix}_d_task_head_layer`
- 数据来源：`{rows_path}`
- 绘图样本数：{total_n:,} 条预测记录；聚合后的 CST 支持单元数：{len(plot_data):,} 个。
- 可信应用域覆盖：{reliable_n:,} 条预测记录，占 {reliable_pct:.1f}%。

## 阈值与标注说明

本图的三个坐标轴分别表示化学支持度 C、物种分类学支持度 S 和任务头支持度 T。坐标为 CST-AD 的分层支持坐标，用于展示模型应用域结构，不是新增模型输出。

- 化学信息层 C：以 `cst_chemical_score` 为核心，`C = 0.50` 表示进入化学相似性支持范围，`C = 0.70` 表示较高化学支持。
- 物种分类学层 S：`S0/S1` 分界对应 `S >= {SPECIES_TRUST_THRESHOLD:.2f}`，表示物种或物种-任务族层面已有较强训练支持。
- 任务头层 T：`T0/T1` 分界对应 `T >= {TASK_TRUST_THRESHOLD:.2f}`，表示任务头或任务族已有较强训练支持。
- 可信应用域范围定义为 `C >= {CHEMICAL_THRESHOLD:.2f}`、`S >= {SPECIES_TRUST_THRESHOLD:.2f}`、`T >= {TASK_TRUST_THRESHOLD:.2f}` 同时满足的区域。图中用半透明色块和斜纹角区表示该范围，避免使用完整绿色线框长方体造成遮挡。
- 曲面的三维形态为按样本量加权的聚合支持单元插值结果，用于增强论文图的连续可读性；它不改变原始 CST-AD 判定结果。
- 曲面颜色映射局部样本密度，采用 `log10(n + 1)` 尺度，并基于四个面板的插值密度曲面使用 5%–85% 分位数稳健归一化，以避免极端高样本单元压缩色差。页面底端的统一横向色条给出对应的数值刻度；颜色由浅绿、青色、钴蓝逐渐过渡到金色，表示局部样本密度由低到高。该颜色只用于观察样本分布，不代表预测误差或模型性能。

## CST-AD 层级分布

| 应用域层级 | 预测记录数 | 占比 |
|---|---:|---:|
{tier_lines}

## 图注

**图 (a) 总览图：CST 三轴完整应用域。**
该图同时展示化学支持度 C、物种分类学支持度 S 和任务头支持度 T 的三维分布。半透明阈值面标出关键应用域边界，色块斜纹角区表示三类支持同时达到可信阈值的范围。曲面越靠近高 C、高 S、高 T 区域，说明对应预测样本更接近训练数据覆盖充分的插值应用域；曲面颜色越接近深蓝或金色，表示该区间聚合样本越集中。

**图 (b) 分类学信息分览：Taxonomic layer。**
该图将分类学支持度 S 作为主要响应轴，相当于在化学支持度和任务头支持度背景下观察物种分类学覆盖的变化。高 S 区域表示测试物种或近缘分类单元在训练集中已有较强支撑；低 S 区域提示模型可能进入分类学外推，解释毒性预测时应降低置信度。颜色映射可用于判断高或低分类学支持区间是否由足够样本支撑。

**图 (c) 化学信息分览：Chemical layer。**
该图将化学支持度 C 作为主要响应轴，展示不同物种-任务组合下化学相似性覆盖的变化。高 C 区域代表待预测化合物更接近训练集化学空间；低 C 区域说明化学结构外推风险增加，需要结合相似化合物、结构片段和适用域审计结果共同解释。颜色较深或偏金的区间说明该化学支持区间中预测记录更多。

**图 (d) 任务头分览：Task-head layer。**
该图将任务头支持度 T 作为主要响应轴，展示化学-分类学组合下任务标签或任务族覆盖的变化。高 T 区域表明训练集中已有相近任务背景，预测更偏向插值；低 T 区域提示任务层面的迁移或外推更强，适合在论文中作为模型适用边界和不确定性来源讨论。颜色映射帮助区分任务头覆盖区域是由大量样本支持，还是仅由少量样本形成。

## 结果解读建议

本图应作为模型完整应用域的结构性总览，而不是单独的性能评估图。论文正文中建议将其用于说明：模型可信预测主要位于化学相似性、分类学覆盖和任务头覆盖同时较高的区域；当任一轴低于阈值时，预测仍可作为筛查结果使用，但应标注为外推或低支持区域，并避免过度解释单个毒性数值。
"""
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int, tight: bool = False) -> list[str]:
    png_path = out_base.with_suffix(".png")
    svg_path = out_base.with_suffix(".svg")
    bbox_inches = "tight" if tight else None
    pad_inches = 0.12 if tight else 0.0
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(svg_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close(fig)
    return [str(png_path), str(svg_path)]


if __name__ == "__main__":
    main()
