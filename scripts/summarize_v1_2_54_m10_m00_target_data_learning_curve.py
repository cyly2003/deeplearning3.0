from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v1_2_44_second_layer_splits import canonical_sha256, frame_record_id
from scripts.build_v1_2_54_m10_m00_target_data_learning_curve_splits import (
    FRACTIONS,
    MODEL_SEEDS,
    SPLIT_TEMPLATE,
)


ROUTES = ("M10", "M00")
ALL_FRACTIONS = (*FRACTIONS, 100)
LEGACY_SPLITS = {
    "M10": "M_v1_2_44_M10_仅水相预训练_固定评价边界",
    "M00": "M_v1_2_44_M00_仅Stage3从头训练_固定评价边界",
}
LEGACY_RUN_NAMES = {
    "M10": "v1.2.44_M10_水相预训练后直接迁移_种子{seed}",
    "M00": "v1.2.44_M00_从头训练_固定评价边界_种子{seed}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the paired v1.2.54 M10/M00 target-data learning curve."
    )
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_724)
    return parser.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "r2": float("nan") if sst <= 0 else 1.0 - sse / sst,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def within_task_r2(
    y_true: np.ndarray, y_pred: np.ndarray, tasks: np.ndarray
) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = 0.0
    for task in np.unique(tasks):
        values = y_true[tasks == task]
        sst += float(np.sum((values - np.mean(values)) ** 2))
    return float("nan") if sst <= 0 else 1.0 - sse / sst


def run_dir(
    *,
    matrix_root: Path,
    legacy_root: Path,
    route: str,
    fraction: int,
    seed: int,
) -> Path:
    if fraction == 100:
        return (
            legacy_root
            / LEGACY_RUN_NAMES[route].format(seed=seed)
            / "deep"
            / "full"
            / LEGACY_SPLITS[route]
        )
    split_name = SPLIT_TEMPLATE.format(route=route, fraction=fraction)
    return (
        matrix_root
        / f"v1.2.54_{route}_F{fraction}_种子{seed}"
        / "deep"
        / "full"
        / split_name
    )


def load_test_predictions(path: Path) -> pd.DataFrame:
    prediction_path = path / "predictions.csv"
    manifest_path = path / "manifest.json"
    if not prediction_path.is_file() or prediction_path.stat().st_size <= 0:
        raise ValueError(f"Missing predictions: {prediction_path}")
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        raise ValueError(f"Missing manifest: {manifest_path}")
    frame = pd.read_csv(prediction_path, low_memory=False)
    frame = frame.loc[frame["split_part"].astype(str) == "test"].copy()
    if len(frame) != 3_042:
        raise ValueError(f"Expected 3042 test rows in {prediction_path}, found {len(frame)}")
    frame["record_id"] = [
        frame_record_id(row) for row in frame.to_dict(orient="records")
    ]
    if frame["record_id"].duplicated().any():
        raise ValueError(f"Duplicate test identities: {prediction_path}")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise")
    frame["task_head"] = frame["task_head"].astype(str)
    return frame.sort_values("record_id").reset_index(drop=True)


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return (
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    )


def task_stratified_indices(
    tasks: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    sampled = []
    for task in np.unique(tasks):
        indices = np.flatnonzero(tasks == task)
        sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled)


def paired_bootstrap(
    *,
    y_true: np.ndarray,
    tasks: np.ndarray,
    m10_pred: np.ndarray,
    m00_pred: np.ndarray,
    m10_100_pred: np.ndarray,
    m00_100_pred: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    delta_mae: list[float] = []
    delta_rmse: list[float] = []
    delta_r2: list[float] = []
    benefit_mae: list[float] = []
    excess_benefit_vs_100: list[float] = []
    for _ in range(replicates):
        index = task_stratified_indices(tasks, rng=rng)
        truth = y_true[index]
        m10 = metrics(truth, m10_pred[index])
        m00 = metrics(truth, m00_pred[index])
        m10_100 = metrics(truth, m10_100_pred[index])
        m00_100 = metrics(truth, m00_100_pred[index])
        delta_mae.append(m10["mae"] - m00["mae"])
        delta_rmse.append(m10["rmse"] - m00["rmse"])
        delta_r2.append(m10["r2"] - m00["r2"])
        benefit = m00["mae"] - m10["mae"]
        benefit_100 = m00_100["mae"] - m10_100["mae"]
        benefit_mae.append(benefit)
        excess_benefit_vs_100.append(benefit - benefit_100)

    output: dict[str, Any] = {}
    for name, values in (
        ("delta_mae_m10_minus_m00", delta_mae),
        ("delta_rmse_m10_minus_m00", delta_rmse),
        ("delta_r2_m10_minus_m00", delta_r2),
        ("mae_benefit_m00_minus_m10", benefit_mae),
        ("excess_mae_benefit_vs_100", excess_benefit_vs_100),
    ):
        low, high = percentile_interval(values)
        output[f"{name}_ci95_low"] = low
        output[f"{name}_ci95_high"] = high
    return output


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    reference_ids: list[str] | None = None
    reference_truth: np.ndarray | None = None
    reference_tasks: np.ndarray | None = None
    cell_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    for fraction in ALL_FRACTIONS:
        for route in ROUTES:
            for seed in MODEL_SEEDS:
                path = run_dir(
                    matrix_root=args.matrix_root,
                    legacy_root=args.legacy_root,
                    route=route,
                    fraction=fraction,
                    seed=seed,
                )
                frame = load_test_predictions(path)
                record_ids = frame["record_id"].tolist()
                truth = frame["y_true"].to_numpy(dtype=float)
                tasks = frame["task_head"].to_numpy(dtype=str)
                if reference_ids is None:
                    reference_ids = record_ids
                    reference_truth = truth
                    reference_tasks = tasks
                else:
                    if record_ids != reference_ids:
                        raise ValueError(
                            f"Test identities differ for {route} F{fraction} seed{seed}"
                        )
                    if not np.array_equal(tasks, reference_tasks):
                        raise ValueError(
                            f"Test task labels differ for {route} F{fraction} seed{seed}"
                        )
                    if not np.allclose(truth, reference_truth, rtol=0, atol=0):
                        raise ValueError(
                            f"Test y_true differs for {route} F{fraction} seed{seed}"
                        )
                frames[(route, fraction, seed)] = frame
                score = metrics(truth, frame["y_pred"].to_numpy(dtype=float))
                cell_rows.append(
                    {
                        "route": route,
                        "fraction_percent": fraction,
                        "seed": seed,
                        "n_test": len(frame),
                        **score,
                        "within_task_r2": within_task_r2(
                            truth,
                            frame["y_pred"].to_numpy(dtype=float),
                            tasks,
                        ),
                        "run_dir": str(path),
                    }
                )
                for task in sorted(frame["task_head"].unique()):
                    task_frame = frame.loc[frame["task_head"] == task]
                    task_score = metrics(
                        task_frame["y_true"].to_numpy(dtype=float),
                        task_frame["y_pred"].to_numpy(dtype=float),
                    )
                    per_task_rows.append(
                        {
                            "route": route,
                            "fraction_percent": fraction,
                            "seed": seed,
                            "task_head": task,
                            "n": len(task_frame),
                            **task_score,
                        }
                    )

    assert reference_ids is not None
    assert reference_truth is not None
    assert reference_tasks is not None
    test_hash = canonical_sha256(reference_ids)

    ensemble_predictions: dict[tuple[str, int], np.ndarray] = {}
    ensemble_rows: list[dict[str, Any]] = []
    for fraction in ALL_FRACTIONS:
        for route in ROUTES:
            predictions = np.column_stack(
                [
                    frames[(route, fraction, seed)]["y_pred"].to_numpy(dtype=float)
                    for seed in MODEL_SEEDS
                ]
            )
            ensemble = np.mean(predictions, axis=1)
            ensemble_predictions[(route, fraction)] = ensemble
            score = metrics(reference_truth, ensemble)
            ensemble_rows.append(
                {
                    "route": route,
                    "fraction_percent": fraction,
                    "seed_count": len(MODEL_SEEDS),
                    "n_test": len(reference_truth),
                    **score,
                    "within_task_r2": within_task_r2(
                        reference_truth, ensemble, reference_tasks
                    ),
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    full_benefit = (
        metrics(reference_truth, ensemble_predictions[("M00", 100)])["mae"]
        - metrics(reference_truth, ensemble_predictions[("M10", 100)])["mae"]
    )
    for fraction in ALL_FRACTIONS:
        m10 = metrics(reference_truth, ensemble_predictions[("M10", fraction)])
        m00 = metrics(reference_truth, ensemble_predictions[("M00", fraction)])
        row = {
            "fraction_percent": fraction,
            "n_test": len(reference_truth),
            "delta_r2_m10_minus_m00": m10["r2"] - m00["r2"],
            "delta_rmse_m10_minus_m00": m10["rmse"] - m00["rmse"],
            "delta_mae_m10_minus_m00": m10["mae"] - m00["mae"],
            "mae_benefit_m00_minus_m10": m00["mae"] - m10["mae"],
            "excess_mae_benefit_vs_100": (m00["mae"] - m10["mae"]) - full_benefit,
        }
        row.update(
            paired_bootstrap(
                y_true=reference_truth,
                tasks=reference_tasks,
                m10_pred=ensemble_predictions[("M10", fraction)],
                m00_pred=ensemble_predictions[("M00", fraction)],
                m10_100_pred=ensemble_predictions[("M10", 100)],
                m00_100_pred=ensemble_predictions[("M00", 100)],
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + fraction,
            )
        )
        contrast_rows.append(row)

    cell_frame = pd.DataFrame(cell_rows)
    cell_frame.to_csv(args.output_dir / "single_seed_metrics.csv", index=False)
    pd.DataFrame(per_task_rows).to_csv(
        args.output_dir / "single_seed_per_task_metrics.csv", index=False
    )
    ensemble_frame = pd.DataFrame(ensemble_rows)
    ensemble_frame.to_csv(args.output_dir / "four_seed_ensemble_metrics.csv", index=False)
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame.to_csv(args.output_dir / "paired_route_contrasts.csv", index=False)

    seed_summary = (
        cell_frame.groupby(["route", "fraction_percent"])[
            ["r2", "rmse", "mae", "within_task_r2"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    seed_summary.columns = [
        "_".join(str(value) for value in column if str(value))
        for column in seed_summary.columns
    ]
    seed_summary.to_csv(args.output_dir / "single_seed_mean_sd.csv", index=False)

    low_data = contrast_frame.loc[
        contrast_frame["fraction_percent"].isin([10, 25, 50])
    ]
    if (
        low_data["excess_mae_benefit_vs_100_ci95_low"].astype(float) > 0
    ).any():
        conclusion = (
            "At least one 10%-50% target-data level shows a larger M10 MAE benefit "
            "than the full-data benefit with a task-stratified paired 95% CI above zero; "
            "this supports enhanced target-data efficiency within the fixed random 8:2 boundary."
        )
        conclusion_level = "enhanced_data_efficiency_supported"
    elif (
        low_data["mae_benefit_m00_minus_m10_ci95_low"].astype(float) > 0
    ).any():
        conclusion = (
            "Aquatic pretraining provides a supported low-data MAE benefit at one or more "
            "10%-50% levels, but its excess benefit over the full-data contrast is uncertain; "
            "describe this as a robust low-data transfer benefit, not enhanced efficiency."
        )
        conclusion_level = "low_data_transfer_benefit_supported"
    else:
        conclusion = (
            "The paired curve does not establish a statistically supported low-data MAE "
            "benefit; retain aquatic pretraining only as an incremental full-data optimization "
            "if the 100% contrast remains supported."
        )
        conclusion_level = "enhanced_data_efficiency_not_supported"

    summary = {
        "schema": "v1_2_54_m10_m00_learning_curve_summary_v1",
        "status": "complete",
        "evaluation_boundary": (
            "locked v1.2.44 row-random 8:2 interpolation test; no scaffold or reference grouping"
        ),
        "routes": list(ROUTES),
        "fractions_percent": list(ALL_FRACTIONS),
        "model_seeds": list(MODEL_SEEDS),
        "new_training_cells": 32,
        "reused_100_percent_cells": 8,
        "n_test": len(reference_truth),
        "test_record_id_sha256": test_hash,
        "bootstrap": {
            "method": "task-stratified row-level paired bootstrap on the same test identities",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "reference_or_scaffold_grouping_used": False,
        },
        "primary_direction": {
            "delta_mae_m10_minus_m00": "negative favors M10",
            "delta_rmse_m10_minus_m00": "negative favors M10",
            "delta_r2_m10_minus_m00": "positive favors M10",
            "mae_benefit_m00_minus_m10": "positive favors M10",
            "excess_mae_benefit_vs_100": (
                "positive means aquatic-pretraining MAE benefit is larger than at 100%"
            ),
        },
        "conclusion_level": conclusion_level,
        "conclusion": conclusion,
        "outputs": {
            "single_seed_metrics": "single_seed_metrics.csv",
            "single_seed_mean_sd": "single_seed_mean_sd.csv",
            "four_seed_ensemble_metrics": "four_seed_ensemble_metrics.csv",
            "paired_route_contrasts": "paired_route_contrasts.csv",
            "single_seed_per_task_metrics": "single_seed_per_task_metrics.csv",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = [
        "# v1.2.54 M10–M00 target-data learning curve",
        "",
        f"- Status: {summary['status']}",
        f"- Boundary: {summary['evaluation_boundary']}",
        f"- Test rows: {summary['n_test']}",
        f"- Bootstrap: {args.bootstrap_replicates} task-stratified paired replicates",
        f"- Conclusion: {conclusion}",
        "",
        "## Four-seed ensemble metrics",
        "",
        "```csv",
        ensemble_frame.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired M10–M00 contrasts",
        "",
        "```csv",
        contrast_frame.to_csv(index=False).strip(),
        "```",
        "",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
