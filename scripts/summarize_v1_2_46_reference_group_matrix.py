from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v1_2_44_second_layer_splits import canonical_sha256
from scripts.summarize_v1_2_40_paired_mass_molar import write_csv
from scripts.summarize_v1_2_44_second_layer_matrix import (
    DEFAULT_SEEDS,
    PARTS,
    assert_aligned_prediction_sets,
    build_ensembles,
    build_prediction_sets,
    build_seed_metrics,
    discover_runs,
    metric_row,
    require_complete_runs,
    require_task_boundary,
    summarize_mean_sd,
)


CELLS = ("M00", "M10", "M11U")
CONTRASTS = (
    ("M10_minus_M00", "M10", "M00", "aquatic_pretraining"),
    ("M11U_minus_M10", "M11U", "M10", "soil_ptox_increment"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the completed v1.2.46 reference-group matrix."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/experiments/v1_2_46_reference_group_matrix"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    output_dir = args.output_dir or args.root / "统一汇总_中文"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.root, allowed=set(CELLS))
    run_map = require_complete_runs(runs, cells=CELLS, seeds=seeds)
    prediction_sets = build_prediction_sets(run_map)
    assert_aligned_prediction_sets(
        prediction_sets, cells=CELLS, seeds=seeds, parts=PARTS
    )
    tasks = require_task_boundary(
        prediction_sets, cells=CELLS, seeds=seeds, expected_count=18
    )
    seed_metrics = build_seed_metrics(prediction_sets, cells=CELLS, seeds=seeds)
    mean_sd = summarize_mean_sd(seed_metrics)
    ensembles = build_ensembles(prediction_sets, cells=CELLS, seeds=seeds)
    ensemble_metrics = [
        metric_row(cell, "test", ensembles[cell], aggregation="prediction_ensemble", seeds=seeds)
        for cell in CELLS
    ]
    by_cell = {str(row["cell"]): row for row in ensemble_metrics}
    contrasts = []
    for label, new, control, role in CONTRASTS:
        new_row = by_cell[new]
        control_row = by_cell[control]
        contrasts.append(
            {
                "contrast": label,
                "scientific_role": role,
                "new_model": new,
                "control_model": control,
                "n": new_row["n"],
                "delta_native_molkg_r2": (
                    new_row["common_molkg_r2"] - control_row["common_molkg_r2"]
                ),
                "delta_native_molkg_rmse": (
                    new_row["common_molkg_rmse"] - control_row["common_molkg_rmse"]
                ),
                "delta_native_molkg_mae": (
                    new_row["common_molkg_mae"] - control_row["common_molkg_mae"]
                ),
                "delta_common_mgkg_r2": (
                    new_row["common_mgkg_r2"] - control_row["common_mgkg_r2"]
                ),
                "delta_common_mgkg_mae": (
                    new_row["common_mgkg_mae"] - control_row["common_mgkg_mae"]
                ),
            }
        )

    task_rows = []
    for cell in CELLS:
        for task in tasks:
            selected = [row for row in ensembles[cell] if row["task_head"] == task]
            item = metric_row(
                cell,
                "test",
                selected,
                aggregation="prediction_ensemble_task",
                seeds=seeds,
            )
            item["task_head"] = task
            item["r2_support"] = "supported" if len(selected) >= 30 else "insufficient_n"
            task_rows.append(item)

    outputs = {
        "seed_metrics": output_dir / "01_逐种子整体指标.csv",
        "mean_sd": output_dir / "02_四种子均值与标准差.csv",
        "ensemble_metrics": output_dir / "03_四种子逐行集成主指标.csv",
        "contrasts": output_dir / "04_核心配对差异.csv",
        "task_metrics": output_dir / "05_18任务集成指标.csv",
        "summary": output_dir / "汇总说明.json",
    }
    write_csv(outputs["seed_metrics"], seed_metrics)
    write_csv(outputs["mean_sd"], mean_sd)
    write_csv(outputs["ensemble_metrics"], ensemble_metrics)
    write_csv(outputs["contrasts"], contrasts)
    write_csv(outputs["task_metrics"], task_rows)
    summary = {
        "schema": "v1_2_46_reference_group_summary_v1",
        "status": "complete",
        "root": str(args.root),
        "cells": list(CELLS),
        "seeds": list(seeds),
        "task_count": len(tasks),
        "tasks": list(tasks),
        "primary_boundary": "reference-group external holdout",
        "primary_scale": "neg_log10_mol_kg",
        "selection_uses_test": False,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
