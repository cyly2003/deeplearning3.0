from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.traditional_comparison import (
    BOUNDARY_SHA256,
    CONTRACT_DIRS,
    MODEL_DIRS,
    MODEL_LABELS,
    MODEL_ORDER,
    EXPECTED_COUNTS,
    canonical_sha256,
    load_deep_reference_ensemble,
    regression_metrics,
)
DEFAULT_ROOT = Path("三阶段版") / "与传统机器学习算法对比"
DEFAULT_SNAPSHOT = Path("analysis/applicability_domain/stage3_discovery_context_snapshot.parquet")
DEFAULT_DEEP_ENSEMBLE = Path("analysis/applicability_domain/stage3_predictions_ensemble.parquet")
BOOTSTRAP_SEED = 20_260_720
BOOTSTRAP_REPLICATES = 20_000
CONTRACT_DEEP_REFERENCE = {
    "molecule_only": "M11U",
    "matched_full": "M00",
}
CONTRACT_EVIDENCE_ROLE = {
    "molecule_only": "overall_framework_advantage",
    "matched_full": "architecture_and_training_system_advantage",
}
CONTRACT_FIGURE_LABEL = {
    "molecule_only": "Molecular-only traditional ML vs M11U",
    "matched_full": "Matched-input traditional ML vs M00",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize v1.2.53 species-endpoint traditional baselines against "
            "deep references on the same fixed Stage-3 test records."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--deep-ensemble", type=Path, default=DEFAULT_DEEP_ENSEMBLE)
    parser.add_argument(
        "--m11u-root",
        type=Path,
        default=DEFAULT_ROOT / "00_共同协议与审计" / "深度模型参考预测" / "M11U",
    )
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-paired-rows",
        type=int,
        default=1_135,
        help="Fail-closed expected coverage; use 194 only for the locked two-subtask smoke test.",
    )
    return parser


def within_group_r2(frame: pd.DataFrame, group_column: str) -> float:
    truth = frame["y_true"].to_numpy(float)
    pred = frame["y_pred"].to_numpy(float)
    numerator = float(np.square(truth - pred).sum())
    denominator = 0.0
    for _, group in frame.groupby(group_column, sort=False):
        values = group["y_true"].to_numpy(float)
        denominator += float(np.square(values - values.mean()).sum())
    return float(1.0 - numerator / denominator) if denominator > 0 else float("nan")


def metric_record(
    frame: pd.DataFrame,
    *,
    contract: str,
    model: str,
    model_label: str,
    model_family: str,
    evidence_role: str,
    reference_route: str,
    full_test_rows: int,
) -> dict[str, Any]:
    values = regression_metrics(frame["y_true"], frame["y_pred"])
    return {
        "contract": contract,
        "contract_label": CONTRACT_DIRS[contract],
        "evidence_role": evidence_role,
        "reference_route": reference_route,
        "model": model,
        "model_label": model_label,
        "model_family": model_family,
        "n": int(len(frame)),
        "full_fixed_test_n": int(full_test_rows),
        "coverage_fraction": float(len(frame) / full_test_rows),
        **values,
        "within_model_head_r2": within_group_r2(frame, "model_head"),
        "within_species_endpoint_r2": within_group_r2(frame, "species_endpoint"),
    }


def load_traditional_predictions(
    root: Path,
) -> tuple[dict[tuple[str, str], pd.DataFrame], set[str]]:
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    common_ids: set[str] | None = None
    for contract in CONTRACT_DIRS:
        for model in MODEL_ORDER:
            model_dir = root / CONTRACT_DIRS[contract] / MODEL_DIRS[model]
            complete = model_dir / "完成标记.json"
            path = model_dir / "预测值_逐行四种子集成.parquet"
            if not complete.exists() or not path.exists():
                raise FileNotFoundError(
                    f"Incomplete formal result for {contract}/{model}: {model_dir}"
                )
            frame = pd.read_parquet(path)
            frame = frame.loc[frame["analysis_split"].astype(str).eq("test")].copy()
            required = {
                "stable_record_id",
                "aggregate_id",
                "model_head",
                "latin_name",
                "species_endpoint",
                "y_true",
                "y_pred",
            }
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{path} lacks required columns: {missing}")
            if frame["stable_record_id"].astype(str).duplicated().any():
                raise ValueError(f"Duplicate test stable IDs in {path}")
            frame["stable_record_id"] = frame["stable_record_id"].astype(str)
            ids = set(frame["stable_record_id"])
            common_ids = ids if common_ids is None else common_ids
            if ids != common_ids:
                raise ValueError(
                    f"Traditional cells do not share one eligible test subset: "
                    f"{contract}/{model} has {len(ids)}, expected {len(common_ids)}"
                )
            predictions[(contract, model)] = frame
    assert common_ids is not None
    return predictions, common_ids


def attach_test_metadata(
    frame: pd.DataFrame,
    *,
    snapshot_test: pd.DataFrame,
) -> pd.DataFrame:
    metadata = snapshot_test[
        ["stable_record_id", "latin_name", "model_head", "aggregate_id", "y_true"]
    ].copy()
    metadata["stable_record_id"] = metadata["stable_record_id"].astype(str)
    metadata["species_endpoint"] = (
        metadata["latin_name"].astype(str).str.strip()
        + "||"
        + metadata["model_head"].astype(str)
    )
    selected = frame.copy()
    selected["stable_record_id"] = selected["stable_record_id"].astype(str)
    merged = selected.merge(
        metadata,
        on="stable_record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_locked"),
    )
    for column in ("aggregate_id", "model_head", "y_true"):
        locked = f"{column}_locked"
        if locked not in merged:
            continue
        if column == "y_true":
            aligned = np.allclose(
                merged[column].to_numpy(float),
                merged[locked].to_numpy(float),
                rtol=1e-12,
                atol=1e-10,
            )
        else:
            aligned = (
                merged[column].astype(str).to_numpy()
                == merged[locked].astype(str).to_numpy()
            ).all()
        if not aligned:
            raise ValueError(f"Deep reference {column} differs from locked snapshot.")
        merged = merged.drop(columns=[locked])
    return merged


def load_deep_predictions(
    *,
    m11u_root: Path,
    deep_ensemble_path: Path,
    snapshot_test: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    m11u_paths = [
        m11u_root / f"seed{seed}" / "predictions.csv"
        for seed in (42, 2042, 3407, 8417)
    ]
    output: dict[str, pd.DataFrame] = {}
    for route in ("M00", "M10", "M11U"):
        frame = load_deep_reference_ensemble(
            route=route,
            deep_ensemble_path=deep_ensemble_path,
            m11u_prediction_paths=m11u_paths if route == "M11U" else (),
        )
        frame = frame.loc[frame["analysis_split"].astype(str).eq("test")].copy()
        if len(frame) != EXPECTED_COUNTS["test"]:
            raise ValueError(f"{route} test rows changed: {len(frame)}")
        if frame["stable_record_id"].astype(str).duplicated().any():
            raise ValueError(f"{route} contains duplicate test stable IDs.")
        output[route] = attach_test_metadata(frame, snapshot_test=snapshot_test)
    reference_ids = set(snapshot_test["stable_record_id"].astype(str))
    for route, frame in output.items():
        if set(frame["stable_record_id"].astype(str)) != reference_ids:
            raise ValueError(f"{route} does not cover the fixed 3,042-row test boundary.")
    return output


def align_prediction(frame: pd.DataFrame, ordered_ids: Sequence[str]) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["stable_record_id"] = keyed["stable_record_id"].astype(str)
    keyed = keyed.set_index("stable_record_id")
    if not keyed.index.is_unique:
        raise ValueError("Prediction frame contains duplicate stable IDs.")
    missing = sorted(set(ordered_ids) - set(keyed.index))
    if missing:
        raise ValueError(f"Prediction frame misses paired IDs: {missing[:8]}")
    return keyed.loc[list(ordered_ids)].reset_index()


def paired_species_endpoint_stratified_bootstrap(
    model_frames: Mapping[str, pd.DataFrame],
    *,
    contrasts: Sequence[tuple[str, str, str, str, str]],
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if replicates < 1_000:
        raise ValueError("At least 1,000 bootstrap replicates are required.")
    ordered_ids = sorted(
        set.intersection(
            *(set(frame["stable_record_id"].astype(str)) for frame in model_frames.values())
        )
    )
    if not ordered_ids:
        raise ValueError("No paired prediction IDs are available.")
    aligned = {
        name: align_prediction(frame, ordered_ids)
        for name, frame in model_frames.items()
    }
    first = next(iter(aligned.values()))
    truth = first["y_true"].to_numpy(float)
    if "species_endpoint" not in first.columns:
        raise ValueError("Stratified bootstrap requires species_endpoint metadata.")
    strata = first["species_endpoint"].astype(str).to_numpy()
    for name, frame in aligned.items():
        if not np.allclose(
            frame["y_true"].to_numpy(float),
            truth,
            rtol=1e-12,
            atol=1e-10,
        ):
            raise ValueError(f"Truth values differ for paired model {name}.")
        if not np.array_equal(frame["species_endpoint"].astype(str).to_numpy(), strata):
            raise ValueError(f"Species-endpoint routing differs for paired model {name}.")
    stratum_indices = [
        np.flatnonzero(strata == stratum)
        for stratum in sorted(set(strata))
    ]
    if any(len(indices) < 1 for indices in stratum_indices):
        raise ValueError("Empty species-endpoint stratum encountered.")
    predictions = {
        name: frame["y_pred"].to_numpy(float)
        for name, frame in aligned.items()
    }
    observed = {
        name: regression_metrics(truth, prediction)
        for name, prediction in predictions.items()
    }
    distributions = {
        label: {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in ("r2", "rmse", "mae")
        }
        for label, *_ in contrasts
    }
    model_distributions = {
        name: {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in ("r2", "rmse", "mae")
        }
        for name in aligned
    }
    rng = np.random.default_rng(seed)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        weights = np.zeros((size, len(ordered_ids)), dtype=np.float64)
        for indices in stratum_indices:
            probability = np.full(
                len(indices),
                1.0 / len(indices),
                dtype=np.float64,
            )
            weights[:, indices] = rng.multinomial(
                len(indices),
                probability,
                size=size,
            )
        sum_weight = weights.sum(axis=1)
        sum_y = weights @ truth
        sum_y2 = weights @ np.square(truth)
        sst = sum_y2 - np.square(sum_y) / sum_weight
        chunk_metrics: dict[str, dict[str, np.ndarray]] = {}
        for name, prediction in predictions.items():
            residual = truth - prediction
            sae = weights @ np.abs(residual)
            sse = weights @ np.square(residual)
            chunk_metrics[name] = {
                "r2": 1.0 - sse / sst,
                "rmse": np.sqrt(sse / sum_weight),
                "mae": sae / sum_weight,
            }
        end = start + size
        for name in aligned:
            for metric in ("r2", "rmse", "mae"):
                model_distributions[name][metric][start:end] = chunk_metrics[name][metric]
        for label, deep, traditional, _, _ in contrasts:
            for metric in ("r2", "rmse", "mae"):
                distributions[label][metric][start:end] = (
                    chunk_metrics[deep][metric]
                    - chunk_metrics[traditional][metric]
                )
    rows: list[dict[str, Any]] = []
    quantiles: dict[str, Any] = {}
    for label, deep, traditional, contract, evidence_role in contrasts:
        item: dict[str, Any] = {
            "contrast": label,
            "contract": contract,
            "evidence_role": evidence_role,
            "deep_model": deep,
            "traditional_model": traditional,
            "n": int(len(ordered_ids)),
            "species_endpoint_strata": int(len(stratum_indices)),
            "replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "resampling_unit": (
                "paired rows resampled with replacement within each locked "
                "latin_name x model_head stratum; stratum sample sizes fixed"
            ),
            "delta_definition": "metric(deep_model)-metric(traditional_model)",
        }
        quantiles[label] = {}
        for metric in ("r2", "rmse", "mae"):
            values = distributions[label][metric]
            q = np.quantile(values, [0.025, 0.5, 0.975])
            point = observed[deep][metric] - observed[traditional][metric]
            nonpositive = (float(np.count_nonzero(values <= 0.0)) + 1.0) / (
                len(values) + 1.0
            )
            nonnegative = (float(np.count_nonzero(values >= 0.0)) + 1.0) / (
                len(values) + 1.0
            )
            two_sided_p = min(1.0, 2.0 * min(nonpositive, nonnegative))
            item[f"delta_{metric}"] = float(point)
            item[f"delta_{metric}_ci95_low"] = float(q[0])
            item[f"delta_{metric}_bootstrap_median"] = float(q[1])
            item[f"delta_{metric}_ci95_high"] = float(q[2])
            item[f"p_{metric}_two_sided_bootstrap"] = float(two_sided_p)
            if metric == "r2":
                item[f"probability_deep_better_{metric}"] = float(np.mean(values > 0.0))
                item[f"deep_superior_95_{metric}"] = bool(q[0] > 0.0)
            else:
                item[f"probability_deep_better_{metric}"] = float(np.mean(values < 0.0))
                item[f"deep_superior_95_{metric}"] = bool(q[2] < 0.0)
            quantiles[label][metric] = {
                "q025": float(q[0]),
                "q50": float(q[1]),
                "q975": float(q[2]),
            }
        rows.append(item)
    model_interval_rows: list[dict[str, Any]] = []
    for name in sorted(aligned):
        item: dict[str, Any] = {
            "model_key": name,
            "n": int(len(ordered_ids)),
            "species_endpoint_strata": int(len(stratum_indices)),
            "replicates": int(replicates),
            "bootstrap_seed": int(seed),
        }
        for metric in ("r2", "rmse", "mae"):
            q = np.quantile(model_distributions[name][metric], [0.025, 0.5, 0.975])
            item[metric] = float(observed[name][metric])
            item[f"{metric}_ci95_low"] = float(q[0])
            item[f"{metric}_bootstrap_median"] = float(q[1])
            item[f"{metric}_ci95_high"] = float(q[2])
        model_interval_rows.append(item)
    return pd.DataFrame(rows), quantiles, pd.DataFrame(model_interval_rows)


def add_holm_significance(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for metric in ("r2", "rmse", "mae"):
        source = f"p_{metric}_two_sided_bootstrap"
        values = output[source].to_numpy(float)
        order = np.argsort(values)
        adjusted_sorted = np.empty(len(values), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            candidate = min(1.0, (len(values) - rank) * values[index])
            running = max(running, candidate)
            adjusted_sorted[rank] = running
        adjusted = np.empty(len(values), dtype=float)
        adjusted[order] = adjusted_sorted
        output[f"p_{metric}_holm"] = adjusted
        favorable = (
            output[f"delta_{metric}"].to_numpy(float) > 0.0
            if metric == "r2"
            else output[f"delta_{metric}"].to_numpy(float) < 0.0
        )
        symbols = []
        for is_favorable, p_value in zip(favorable, adjusted):
            if not is_favorable or p_value >= 0.05:
                symbols.append("")
            elif p_value < 0.001:
                symbols.append("***")
            elif p_value < 0.01:
                symbols.append("**")
            else:
                symbols.append("*")
        output[f"deep_better_significance_{metric}"] = symbols
    return output


def build_species_endpoint_deltas(
    traditional: Mapping[tuple[str, str], pd.DataFrame],
    deep: Mapping[str, pd.DataFrame],
    paired_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contract in CONTRACT_DIRS:
        reference = CONTRACT_DEEP_REFERENCE[contract]
        deep_subset = deep[reference].loc[
            deep[reference]["stable_record_id"].astype(str).isin(paired_ids)
        ]
        deep_by_id = deep_subset.set_index("stable_record_id")
        if not deep_by_id.index.is_unique:
            raise ValueError(f"{reference} contains duplicate stable IDs.")
        for model in MODEL_ORDER:
            frame = traditional[(contract, model)].copy()
            frame["stable_record_id"] = frame["stable_record_id"].astype(str)
            joined = frame.merge(
                deep_by_id[["y_pred"]].rename(columns={"y_pred": "deep_y_pred"}),
                left_on="stable_record_id",
                right_index=True,
                how="left",
                validate="one_to_one",
            )
            for subtask, group in joined.groupby("species_endpoint", sort=True):
                traditional_metrics = regression_metrics(group["y_true"], group["y_pred"])
                deep_metrics = regression_metrics(group["y_true"], group["deep_y_pred"])
                rows.append(
                    {
                        "contract": contract,
                        "evidence_role": CONTRACT_EVIDENCE_ROLE[contract],
                        "deep_model": reference,
                        "traditional_model": model,
                        "traditional_model_label": MODEL_LABELS[model],
                        "species_endpoint": subtask,
                        "latin_name": str(group["latin_name"].iloc[0]),
                        "model_head": str(group["model_head"].iloc[0]),
                        "n": int(len(group)),
                        "delta_r2_deep_minus_traditional": float(
                            deep_metrics["r2"] - traditional_metrics["r2"]
                        ),
                        "delta_rmse_deep_minus_traditional": float(
                            deep_metrics["rmse"] - traditional_metrics["rmse"]
                        ),
                        "delta_mae_deep_minus_traditional": float(
                            deep_metrics["mae"] - traditional_metrics["mae"]
                        ),
                        "deep_r2": float(deep_metrics["r2"]),
                        "traditional_r2": float(traditional_metrics["r2"]),
                        "deep_rmse": float(deep_metrics["rmse"]),
                        "traditional_rmse": float(traditional_metrics["rmse"]),
                        "deep_mae": float(deep_metrics["mae"]),
                        "traditional_mae": float(traditional_metrics["mae"]),
                    }
                )
    return pd.DataFrame(rows)


def save_figure_bundle(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def make_performance_figure(
    metrics: pd.DataFrame,
    model_intervals: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    colors = {
        "random_forest": "#999999",
        "xgboost": "#56B4E9",
        "lightgbm": "#009E73",
        "pls": "#E69F00",
        "knn": "#CC79A7",
        "deep": "#0072B2",
    }
    intervals = model_intervals.copy()
    split_keys = intervals["model_key"].str.split("::", n=1, expand=True)
    intervals["contract"] = split_keys[0]
    intervals["model"] = split_keys[1]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    panel_labels = ("a", "b", "c", "d")
    for column, contract in enumerate(("molecule_only", "matched_full")):
        frame = metrics.loc[metrics["contract"].eq(contract)].copy().merge(
            intervals.drop(columns=["n"]),
            on=["contract", "model"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_bootstrap"),
        )
        order = list(MODEL_ORDER) + [CONTRACT_DEEP_REFERENCE[contract]]
        frame["order"] = frame["model"].map({name: idx for idx, name in enumerate(order)})
        frame = frame.sort_values("order")
        x = np.arange(len(frame))
        for row, metric in enumerate(("r2", "mae")):
            ax = axes[row, column]
            values = frame[metric].to_numpy(float)
            low = frame[f"{metric}_ci95_low"].to_numpy(float)
            high = frame[f"{metric}_ci95_high"].to_numpy(float)
            bar_colors = [
                colors["deep"] if family == "deep" else colors[model]
                for family, model in zip(frame["model_family"], frame["model"])
            ]
            bars = ax.bar(
                x,
                values,
                width=0.68,
                color=bar_colors,
                edgecolor="#333333",
                linewidth=0.5,
                yerr=np.vstack([values - low, high - values]),
                error_kw={
                    "ecolor": "#222222",
                    "elinewidth": 0.8,
                    "capsize": 2.0,
                    "capthick": 0.8,
                },
                zorder=2,
            )
            for bar, family in zip(bars, frame["model_family"]):
                if family == "deep":
                    bar.set_hatch("//")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    "RF" if name == "random_forest" else MODEL_LABELS.get(name, name)
                    for name in frame["model"]
                ],
                rotation=35,
                ha="right",
            )
            ax.set_ylabel(
                "Test R²"
                if metric == "r2"
                else r"Test MAE [log$_{10}$(mol kg$^{-1}$)]"
            )
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            if metric == "mae":
                ax.set_ylim(bottom=0.0)
            else:
                ax.set_ylim(bottom=min(0.0, float(np.nanmin(low)) * 1.05))
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.16,
                1.06,
                panel_labels[row * 2 + column],
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
        axes[0, column].set_title(CONTRACT_FIGURE_LABEL[contract])
    fig.suptitle(
        "Traditional models are fitted separately for each species–endpoint task\n"
        f"Paired fixed-test coverage: n={int(metrics['n'].min()):,} "
        f"({metrics['coverage_fraction'].min() * 100:.1f}% of 3,042)",
        fontsize=9,
    )
    source = output_dir / "Figure_1_source_data.csv"
    intervals.to_csv(
        output_dir / "Figure_1_bootstrap_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(source, index=False, encoding="utf-8-sig")
    save_figure_bundle(fig, output_dir / "Figure_1_two_level_performance")
    plt.close(fig)


def make_bootstrap_figure(bootstrap: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    frame = bootstrap.copy()
    contract_order = {"molecule_only": 0, "matched_full": 1}
    model_order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    frame["contract_order"] = frame["contract"].map(contract_order)
    frame["model_order"] = frame["traditional_model"].map(model_order)
    frame = frame.sort_values(["contract_order", "model_order"]).reset_index(drop=True)
    labels = [
        (
            "Molecular | " if contract == "molecule_only" else "Matched | "
        )
        + ("RF" if model == "random_forest" else MODEL_LABELS[model])
        for contract, model in zip(frame["contract"], frame["traditional_model"])
    ]
    y = np.arange(len(frame))[::-1]
    colors = [
        "#D55E00" if contract == "molecule_only" else "#009E73"
        for contract in frame["contract"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), constrained_layout=True)
    definitions = (
        (
            "r2",
            "ΔR² (deep − traditional)",
            "Deep model better →",
        ),
        (
            "mae",
            r"ΔMAE (deep − traditional) [log$_{10}$(mol kg$^{-1}$)]",
            "← Deep model better",
        ),
    )
    for panel, (metric, xlabel, direction) in enumerate(definitions):
        ax = axes[panel]
        point = frame[f"delta_{metric}"].to_numpy(float)
        low = frame[f"delta_{metric}_ci95_low"].to_numpy(float)
        high = frame[f"delta_{metric}_ci95_high"].to_numpy(float)
        for index, color in enumerate(colors):
            ax.errorbar(
                point[index],
                y[index],
                xerr=np.asarray(
                    [
                        [point[index] - low[index]],
                        [high[index] - point[index]],
                    ]
                ),
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=2.5,
                zorder=2,
            )
        ax.scatter(point, y, c=colors, s=26, edgecolors="white", linewidths=0.5, zorder=3)
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if panel == 0 else [])
        ax.set_xlabel(xlabel)
        ax.set_title(("a  Paired R² difference" if panel == 0 else "b  Paired MAE difference"))
        ax.grid(axis="x", color="#E2E2E2", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.text(
            0.98 if metric == "r2" else 0.02,
            0.99,
            direction,
            transform=ax.transAxes,
            ha="right" if metric == "r2" else "left",
            va="top",
            color="#555555",
            fontsize=7,
        )
    fig.suptitle(
        "Paired species–endpoint-stratified bootstrap "
        f"({int(frame['replicates'].iloc[0]):,} replicates; 95% percentile CI)",
        fontsize=9,
    )
    frame.to_csv(output_dir / "Figure_2_source_data.csv", index=False, encoding="utf-8-sig")
    save_figure_bundle(fig, output_dir / "Figure_2_paired_cluster_bootstrap")
    plt.close(fig)


def make_integrated_figure(
    metrics: pd.DataFrame,
    model_intervals: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    palette = {
        "random_forest": "#999999",
        "xgboost": "#56B4E9",
        "lightgbm": "#009E73",
        "pls": "#E69F00",
        "knn": "#CC79A7",
        "deep": "#0072B2",
    }
    intervals = model_intervals.copy()
    split_keys = intervals["model_key"].str.split("::", n=1, expand=True)
    intervals["contract"] = split_keys[0]
    intervals["model"] = split_keys[1]
    performance = metrics.merge(
        intervals.drop(columns=["n"]),
        on=["contract", "model"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_bootstrap"),
    )
    order_map = {name: index for index, name in enumerate(MODEL_ORDER)}
    performance["contract_order"] = performance["contract"].map(
        {"molecule_only": 0, "matched_full": 1}
    )
    performance["model_order"] = [
        5 if family == "deep" else order_map[model]
        for family, model in zip(performance["model_family"], performance["model"])
    ]
    performance = performance.sort_values(
        ["contract_order", "model_order"]
    ).reset_index(drop=True)
    positions = np.asarray(list(range(6)) + list(range(7, 13)), dtype=float)
    labels = [
        "RF" if model == "random_forest" else MODEL_LABELS.get(model, model)
        for model in performance["model"]
    ]
    colors = [
        palette["deep"] if family == "deep" else palette[model]
        for family, model in zip(performance["model_family"], performance["model"])
    ]
    significance = bootstrap.set_index(["contract", "traditional_model"])
    forest = bootstrap.copy()
    forest["contract_order"] = forest["contract"].map(
        {"molecule_only": 0, "matched_full": 1}
    )
    forest["model_order"] = forest["traditional_model"].map(order_map)
    forest = forest.sort_values(["contract_order", "model_order"]).reset_index(drop=True)
    heatmap_models = list(MODEL_ORDER)
    heatmap_contracts = ["molecule_only", "matched_full"]
    heatmap_labels = [
        "RF" if model == "random_forest" else MODEL_LABELS[model]
        for model in heatmap_models
    ]

    fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.95, 1.35))
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    panel_labels = ("a", "b", "c", "d")

    for panel, metric in enumerate(("r2", "mae")):
        ax = axes[0, panel]
        values = performance[metric].to_numpy(float)
        low = performance[f"{metric}_ci95_low"].to_numpy(float)
        high = performance[f"{metric}_ci95_high"].to_numpy(float)
        bars = ax.bar(
            positions,
            values,
            width=0.72,
            color=colors,
            edgecolor="#333333",
            linewidth=0.45,
            yerr=np.vstack([values - low, high - values]),
            error_kw={
                "ecolor": "#222222",
                "elinewidth": 0.7,
                "capsize": 1.8,
                "capthick": 0.7,
            },
            zorder=2,
        )
        for index, (bar, row) in enumerate(zip(bars, performance.itertuples(index=False))):
            if row.model_family == "deep":
                bar.set_hatch("//")
                continue
            symbol = significance.loc[
                (row.contract, row.model),
                f"deep_better_significance_{metric}",
            ]
            if symbol:
                margin = max(0.01, 0.025 * (float(np.nanmax(high)) - float(np.nanmin(low))))
                ax.text(
                    positions[index],
                    high[index] + margin,
                    symbol,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
        ax.axvline(6.0, color="#BFBFBF", linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(
            "Test R²"
            if metric == "r2"
            else r"Test MAE [log$_{10}$(mol kg$^{-1}$)]"
        )
        ax.set_title(
            "Overall performance (higher is better)"
            if metric == "r2"
            else "Overall error (lower is better)"
        )
        ax.grid(axis="y", color="#DEDEDE", linewidth=0.55, zorder=0)
        if metric == "mae":
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=min(0.0, float(np.nanmin(low)) * 1.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.13,
            1.08,
            panel_labels[panel],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )

    heatmap_definitions = (
        (
            "r2",
            "R² gain (deep − traditional)",
            "deep_better_significance_r2",
            1.0,
        ),
        (
            "mae",
            "MAE reduction (traditional − deep)",
            "deep_better_significance_mae",
            -1.0,
        ),
    )
    for offset, (metric, title, significance_column, direction) in enumerate(
        heatmap_definitions
    ):
        ax = axes[1, offset]
        matrix = np.empty((len(heatmap_models), len(heatmap_contracts)), dtype=float)
        symbols: list[list[str]] = [
            [""] * len(heatmap_contracts) for _ in heatmap_models
        ]
        for row_index, model in enumerate(heatmap_models):
            for column_index, contract in enumerate(heatmap_contracts):
                item = forest.loc[
                    forest["traditional_model"].eq(model)
                    & forest["contract"].eq(contract)
                ].iloc[0]
                matrix[row_index, column_index] = (
                    direction * float(item[f"delta_{metric}"])
                )
                symbols[row_index][column_index] = str(item[significance_column])
        extent = float(np.nanmax(np.abs(matrix)))
        extent = max(extent, 1.0e-6)
        image = ax.imshow(
            matrix,
            cmap="BrBG",
            vmin=-extent,
            vmax=extent,
            aspect="auto",
        )
        ax.set_xticks(np.arange(len(heatmap_contracts)))
        ax.set_xticklabels(
            ["Overall framework\nM11U vs ML", "Matched inputs\nM00 vs ML"]
        )
        ax.set_yticks(np.arange(len(heatmap_models)))
        ax.set_yticklabels(heatmap_labels)
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                symbol = symbols[row_index][column_index]
                color = "white" if abs(value) > 0.55 * extent else "#202020"
                annotation = f"{value:+.3f}"
                if symbol:
                    annotation += f"\n{symbol}"
                ax.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                    fontweight="bold" if symbol else "normal",
                )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
        colorbar.ax.tick_params(labelsize=5.5, length=2)
        colorbar.set_label("Positive = deep model better", fontsize=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.text(
            -0.13,
            1.06,
            panel_labels[2 + offset],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "Species–endpoint-specific traditional ML versus the deep transfer framework\n"
        f"Paired fixed-test coverage: n={int(metrics['n'].min()):,}; "
        f"stratified bootstrap={int(bootstrap['replicates'].iloc[0]):,}",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        -0.005,
        "* Holm-adjusted paired bootstrap p<0.05; ** p<0.01; *** p<0.001. "
        "No symbol is shown otherwise.",
        ha="center",
        va="top",
        fontsize=6.2,
        color="#444444",
    )
    performance.to_csv(
        output_dir / "Figure_1_metrics_and_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    forest.to_csv(
        output_dir / "Figure_1_paired_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_figure_bundle(fig, output_dir / "Figure_1_integrated_traditional_comparison")
    plt.close(fig)


def write_figure_qa_notes(
    *,
    output_dir: Path,
    eligible_subtasks: int,
    paired_rows: int,
    coverage_fraction: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    text = f"""# 图表统计与 QA 说明 / Figure statistics and QA notes

## 中文

- 核心结论：在同一固定测试子集上，分别检验“仅分子传统模型 vs M11U”的整体框架差异，以及“相同分子+上下文传统模型 vs M00”的架构与训练系统差异。
- 图型：一张四面板组合图；上排为带误差线的分组柱图，下排为配对优势热图。
- 数据边界：v1.2.44 固定随机划分；训练/验证/测试为 9,724/2,433/3,042 行。
- 传统模型建模单位：每个 `latin_name × model_head` 独立模型；无物种 embedding、无跨物种参数共享。
- `n` 定义：满足训练≥35、验证≥10、测试≥10 的 {eligible_subtasks} 个物种—终点子任务所覆盖的 {paired_rows:,} 条测试记录，占完整测试集 {coverage_fraction * 100:.2f}%。
- 生物学重复：不适用；记录来自聚合后的生态毒理试验数据。
- 技术重复：RF/XGBoost/LightGBM 为 4 个模型随机种子（42、2042、3407、8417）逐行集成；PLS/KNN 为确定性重复，四次预测相同。
- 中心统计量：Figure 1 为测试集逐行集成预测计算的点估计。
- 区间：在每个 `latin_name × model_head` 层内保持原测试样本量不变并配对重采样，Figure 1 为 {bootstrap_replicates:,} 次分层 bootstrap 的 95% 百分位区间，bootstrap seed={bootstrap_seed}。
- 指标：R²、RMSE、MAE 均在原生 `neg_log10(mol/kg)` 尺度计算；差值定义为深度模型减传统模型。
- 多重比较：对每个指标的 10 个预先定义对比使用 Holm 校正；只有本线路方向更优且校正后 `p<0.05` 的单元格显示星号。
- 基线定义：Figure 1 左层为 5 种仅分子传统模型；右层为 5 种输入一致传统模型。
- 源数据：`Figure_1_metrics_and_intervals.csv`、`Figure_1_paired_differences.csv`。
- 导出：Python/Matplotlib 单一后端；PNG 与 TIFF 为 600 dpi，另有可编辑文字的 SVG 和 PDF。
- 审稿边界：这是随机划分插值比较，不证明结构外推、新物种零样本泛化或单一网络层的纯因果优势。

## English

- Core conclusion: two controlled questions are tested on the same fixed-test subset: overall-framework performance (molecular-only traditional models versus M11U) and architecture/training-system performance under matched molecular-plus-context inputs (traditional models versus M00).
- Archetype: one four-panel figure with grouped quantitative bars and paired advantage heatmaps.
- Data boundary: locked v1.2.44 random split; 9,724/2,433/3,042 train/validation/test records.
- Traditional modeling unit: one independent model per `latin_name × model_head`; no species embedding and no cross-species parameter sharing.
- Definition of `n`: {paired_rows:,} test records from {eligible_subtasks} species–endpoint tasks satisfying train≥35, validation≥10, and test≥10; coverage={coverage_fraction * 100:.2f}%.
- Biological replicates: not applicable; records are aggregated ecotoxicology study observations.
- Technical replicates: four model seeds (42, 2042, 3407, and 8417) for RF/XGBoost/LightGBM; PLS/KNN are deterministic repeats.
- Center statistic: Figure 1 shows point estimates from per-record ensemble predictions.
- Interval: rows are resampled in pairs within each locked `latin_name × model_head` stratum while preserving its test size; Figure 1 shows 95% percentile intervals from {bootstrap_replicates:,} stratified bootstrap replicates (seed={bootstrap_seed}).
- Metrics: R², RMSE, and MAE on native `neg_log10(mol/kg)` values; deltas are defined as deep minus traditional.
- Multiple comparisons: Holm correction is applied across the 10 prespecified contrasts within each metric; a star is shown only when the deep route is favorable and the adjusted p-value is below 0.05.
- Baselines: five molecular-only traditional models in the overall-framework layer and five matched-input traditional models in the architecture/training-system layer.
- Source data: `Figure_1_metrics_and_intervals.csv` and `Figure_1_paired_differences.csv`.
- Export: Python/Matplotlib only; 600-dpi PNG/TIFF plus SVG/PDF with editable text.
- Reviewer boundary: random-split interpolation only; no claim of structural extrapolation, zero-shot new-species generalization, or a pure causal effect of one network layer.
"""
    (output_dir / "图表统计与QA说明_中英文.md").write_text(text, encoding="utf-8")


def write_layer_outputs(
    *,
    root: Path,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    subtask_deltas: pd.DataFrame,
    deep_full_metrics: pd.DataFrame,
    figure_dir: Path,
) -> None:
    molecule_compare = root / "仅分子信息" / "06_与完整三阶段框架比较"
    molecule_stats = root / "仅分子信息" / "07_统一图表与统计检验"
    matched_compare = root / "输入一致" / "06_与M00深度模型比较_架构证据"
    matched_supplement = root / "输入一致" / "07_与M10_M11U比较_完整框架证据"
    matched_stats = root / "输入一致" / "08_统一图表与统计检验"
    for path in (
        molecule_compare,
        molecule_stats,
        matched_compare,
        matched_supplement,
        matched_stats,
    ):
        path.mkdir(parents=True, exist_ok=True)
    metrics.loc[metrics["contract"].eq("molecule_only")].to_csv(
        molecule_compare / "同覆盖测试集指标.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap.loc[bootstrap["contract"].eq("molecule_only")].to_csv(
        molecule_stats / "配对物种终点分层Bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    subtask_deltas.loc[subtask_deltas["contract"].eq("molecule_only")].to_csv(
        molecule_stats / "逐物种终点配对差值.csv", index=False, encoding="utf-8-sig"
    )
    metrics.loc[metrics["contract"].eq("matched_full")].to_csv(
        matched_compare / "同覆盖测试集指标.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap.loc[bootstrap["contract"].eq("matched_full")].to_csv(
        matched_stats / "配对物种终点分层Bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    subtask_deltas.loc[subtask_deltas["contract"].eq("matched_full")].to_csv(
        matched_stats / "逐物种终点配对差值.csv", index=False, encoding="utf-8-sig"
    )
    deep_full_metrics.to_csv(
        matched_supplement / "M00_M10_M11U完整3042行参考指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for source in figure_dir.glob("Figure_*"):
        if source.is_file():
            shutil.copy2(source, molecule_stats / source.name)
            shutil.copy2(source, matched_stats / source.name)


def main() -> None:
    args = build_parser().parse_args()
    summary_dir = args.root / "统一汇总"
    figure_dir = summary_dir / "图表"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    snapshot = pd.read_parquet(args.snapshot)
    snapshot_test = snapshot.loc[snapshot["analysis_split"].astype(str).eq("test")].copy()
    if len(snapshot_test) != EXPECTED_COUNTS["test"]:
        raise ValueError(f"Locked test count changed: {len(snapshot_test)}")
    snapshot_test["stable_record_id"] = snapshot_test["stable_record_id"].astype(str)
    traditional, paired_ids = load_traditional_predictions(args.root)
    eligible_count = int(
        next(iter(traditional.values()))["species_endpoint"].nunique()
    )
    deep = load_deep_predictions(
        m11u_root=args.m11u_root,
        deep_ensemble_path=args.deep_ensemble,
        snapshot_test=snapshot_test,
    )
    if len(paired_ids) != int(args.expected_paired_rows):
        raise ValueError(
            "Locked species-endpoint support coverage changed: "
            f"{len(paired_ids)} != {args.expected_paired_rows}"
        )

    metric_rows: list[dict[str, Any]] = []
    model_frames: dict[str, pd.DataFrame] = {}
    contrasts: list[tuple[str, str, str, str, str]] = []
    for contract in CONTRACT_DIRS:
        reference = CONTRACT_DEEP_REFERENCE[contract]
        deep_subset = deep[reference].loc[
            deep[reference]["stable_record_id"].astype(str).isin(paired_ids)
        ].copy()
        deep_key = f"{contract}::{reference}"
        model_frames[deep_key] = deep_subset
        metric_rows.append(
            metric_record(
                deep_subset,
                contract=contract,
                model=reference,
                model_label=reference,
                model_family="deep",
                evidence_role=CONTRACT_EVIDENCE_ROLE[contract],
                reference_route=reference,
                full_test_rows=EXPECTED_COUNTS["test"],
            )
        )
        for model in MODEL_ORDER:
            frame = traditional[(contract, model)]
            traditional_key = f"{contract}::{model}"
            model_frames[traditional_key] = frame
            metric_rows.append(
                metric_record(
                    frame,
                    contract=contract,
                    model=model,
                    model_label=MODEL_LABELS[model],
                    model_family="traditional",
                    evidence_role=CONTRACT_EVIDENCE_ROLE[contract],
                    reference_route=reference,
                    full_test_rows=EXPECTED_COUNTS["test"],
                )
            )
            contrasts.append(
                (
                    f"{reference}_minus_{model}_{contract}",
                    deep_key,
                    traditional_key,
                    contract,
                    CONTRACT_EVIDENCE_ROLE[contract],
                )
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        summary_dir / "同覆盖测试集模型指标.csv", index=False, encoding="utf-8-sig"
    )

    deep_full_rows = []
    for route, frame in deep.items():
        values = regression_metrics(frame["y_true"], frame["y_pred"])
        deep_full_rows.append(
            {
                "route": route,
                "n": int(len(frame)),
                **values,
                "within_model_head_r2": within_group_r2(frame, "model_head"),
                "within_species_endpoint_r2": within_group_r2(frame, "species_endpoint"),
            }
        )
    deep_full_metrics = pd.DataFrame(deep_full_rows)
    deep_full_metrics.to_csv(
        summary_dir / "深度模型完整3042行参考指标.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bootstrap, bootstrap_quantiles, model_intervals = (
        paired_species_endpoint_stratified_bootstrap(
            model_frames,
            contrasts=contrasts,
            replicates=args.replicates,
            seed=args.bootstrap_seed,
        )
    )
    bootstrap = add_holm_significance(bootstrap)
    model_intervals.to_csv(
        summary_dir / "模型指标物种终点分层Bootstrap区间.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap["deep_model"] = bootstrap["deep_model"].str.split("::").str[-1]
    bootstrap["traditional_model"] = (
        bootstrap["traditional_model"].str.split("::").str[-1]
    )
    bootstrap.to_csv(
        summary_dir / "配对物种终点分层Bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (summary_dir / "Bootstrap分位数.json").write_text(
        json.dumps(bootstrap_quantiles, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    subtask_deltas = build_species_endpoint_deltas(traditional, deep, paired_ids)
    subtask_deltas.to_csv(
        summary_dir / "逐物种终点配对差值.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_integrated_figure(metrics, model_intervals, bootstrap, figure_dir)
    write_figure_qa_notes(
        output_dir=figure_dir,
        eligible_subtasks=eligible_count,
        paired_rows=len(paired_ids),
        coverage_fraction=len(paired_ids) / EXPECTED_COUNTS["test"],
        bootstrap_replicates=args.replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_layer_outputs(
        root=args.root,
        metrics=metrics,
        bootstrap=bootstrap,
        subtask_deltas=subtask_deltas,
        deep_full_metrics=deep_full_metrics,
        figure_dir=figure_dir,
    )
    summary = {
        "schema": "v1_2_53_traditional_ml_fixed_boundary_summary_v1",
        "status": "complete",
        "boundary_sha256": BOUNDARY_SHA256,
        "traditional_modeling_unit": "independent latin_name x model_head",
        "eligible_species_endpoint_models": eligible_count,
        "paired_test_rows": int(len(paired_ids)),
        "full_fixed_test_rows": EXPECTED_COUNTS["test"],
        "coverage_fraction": float(len(paired_ids) / EXPECTED_COUNTS["test"]),
        "bootstrap_replicates": int(args.replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_resampling_unit": (
            "paired rows within locked latin_name x model_head strata; "
            "stratum test sample sizes preserved"
        ),
        "multiple_comparison_correction": "Holm within each metric across 10 contrasts",
        "primary_contrasts": {
            "overall_framework": "molecule_only traditional ML vs M11U",
            "architecture_and_training_system": "matched_full traditional ML vs M00",
        },
        "claim_boundary": (
            f"random-split interpolation on {eligible_count} supported "
            "species-endpoint subtasks; "
            "not structural extrapolation or zero-shot cross-species prediction"
        ),
        "files": {
            "metrics": "同覆盖测试集模型指标.csv",
            "deep_full_reference": "深度模型完整3042行参考指标.csv",
            "model_metric_intervals": "模型指标物种终点分层Bootstrap区间.csv",
            "paired_bootstrap": "配对物种终点分层Bootstrap.csv",
            "species_endpoint_deltas": "逐物种终点配对差值.csv",
            "figures": "图表",
        },
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
