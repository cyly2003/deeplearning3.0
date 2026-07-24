from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v1_2_44_second_layer_splits import canonical_sha256
from scripts.summarize_v1_2_40_paired_mass_molar import metrics, stable_hash, write_csv
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
from scripts.summarize_v1_2_45_reference_cluster_bootstrap import (
    EQUIVALENCE_MARGIN,
    paired_cluster_bootstrap,
)
from scripts.validate_v1_2_47_scaffold_family_run import (
    SCHEMA as SPLIT_SCHEMA,
    SOURCE_TABLE,
    load_json,
    validate_split_summary,
)


CELLS = ("M00", "M10", "M11U")
MODEL_ZH = {
    "M00": "SF-M00 结构族外推从头训练",
    "M10": "SF-M10 结构过滤水相预训练",
    "M11U": "SF-M11U 结构过滤完整三阶段",
}
CONTRASTS = (
    ("M10_minus_M00", "M10", "M00", "molkg", "aquatic_pretraining"),
    ("M11U_minus_M10", "M11U", "M10", "molkg", "soil_ptox_increment"),
)
DEFAULT_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_721


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the completed v1.2.47 scaffold-family matrix and run "
            "paired structure-component bootstrap."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/experiments/v1_2_47_scaffold_family_matrix"),
    )
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 1_000:
        raise ValueError("At least 1,000 paired bootstrap replicates are required.")
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("Exactly four unique model seeds are required.")

    split_summary = load_json(args.split_summary)
    validate_split_summary(
        split_summary,
        source_table=SOURCE_TABLE,
    )
    if split_summary.get("schema") != SPLIT_SCHEMA:
        raise ValueError("Unexpected scaffold-family split-summary schema.")

    output_dir = args.output_dir or args.root / "统一汇总_中文"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.root, allowed=set(CELLS))
    run_map = require_complete_runs(runs, cells=CELLS, seeds=seeds)
    prediction_sets = build_prediction_sets(run_map)
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=CELLS,
        seeds=seeds,
        parts=PARTS,
    )
    tasks = require_task_boundary(
        prediction_sets,
        cells=CELLS,
        seeds=seeds,
        expected_count=18,
    )

    seed_metrics = build_seed_metrics(prediction_sets, cells=CELLS, seeds=seeds)
    for row in seed_metrics:
        row["model_zh"] = MODEL_ZH[str(row["cell"])]
    mean_sd = summarize_mean_sd(seed_metrics)
    ensembles = build_ensembles(prediction_sets, cells=CELLS, seeds=seeds)
    ensemble_metrics = []
    for cell in CELLS:
        row = metric_row(
            cell,
            "test",
            ensembles[cell],
            aggregation="prediction_ensemble",
            seeds=seeds,
        )
        row["model_zh"] = MODEL_ZH[cell]
        ensemble_metrics.append(row)

    by_cell = {str(row["cell"]): row for row in ensemble_metrics}
    contrasts = []
    for label, new, control, _, role in CONTRASTS:
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

    test_ids = {str(row["aggregate_id"]) for row in ensembles[CELLS[0]]}
    component_by_aggregate = load_test_component_mapping(split_summary, test_ids=test_ids)
    task_rows = build_task_rows(
        ensembles,
        tasks=tasks,
        seeds=seeds,
        component_by_aggregate=component_by_aggregate,
    )
    bootstrap_rows, bootstrap_quantiles = paired_cluster_bootstrap(
        ensembles,
        group_by_aggregate=component_by_aggregate,
        contrasts=CONTRASTS,
        replicates=int(args.replicates),
        seed=int(args.bootstrap_seed),
        equivalence_margin=EQUIVALENCE_MARGIN,
    )

    outputs = {
        "seed_metrics": output_dir / "01_逐种子整体指标.csv",
        "mean_sd": output_dir / "02_四种子均值与标准差.csv",
        "ensemble_metrics": output_dir / "03_四种子逐行集成主指标.csv",
        "contrasts": output_dir / "04_核心配对差异.csv",
        "task_metrics": output_dir / "05_18任务结构组支持与集成指标.csv",
        "bootstrap": output_dir / "06_结构组配对bootstrap.csv",
        "bootstrap_quantiles": output_dir / "07_bootstrap分布分位数.json",
        "summary": output_dir / "汇总说明.json",
    }
    write_csv(outputs["seed_metrics"], seed_metrics)
    write_csv(outputs["mean_sd"], mean_sd)
    write_csv(outputs["ensemble_metrics"], ensemble_metrics)
    write_csv(outputs["contrasts"], contrasts)
    write_csv(outputs["task_metrics"], task_rows)
    write_csv(outputs["bootstrap"], bootstrap_rows)
    outputs["bootstrap_quantiles"].write_text(
        json.dumps(bootstrap_quantiles, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "schema": "v1_2_47_scaffold_family_summary_v1",
        "status": "complete",
        "root": str(args.root),
        "split_contract_sha256": split_summary["contract_sha256"],
        "cells": list(CELLS),
        "seeds": list(seeds),
        "task_count": len(tasks),
        "tasks": list(tasks),
        "test_rows": len(test_ids),
        "test_structure_components": len(set(component_by_aggregate.values())),
        "primary_boundary": "scaffold/similarity-family external holdout",
        "primary_scale": "neg_log10_mol_kg",
        "selection_uses_test": False,
        "test_read_after_all_cells_validated": True,
        "bootstrap": {
            "replicates": int(args.replicates),
            "seed": int(args.bootstrap_seed),
            "cluster_unit": "heldout_structure_component",
            "equivalence_margin_mae": EQUIVALENCE_MARGIN,
            "equivalence_ci": "90_percentile_CI_inside_plus_minus_0.01",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def load_test_component_mapping(
    summary: Mapping[str, Any], *, test_ids: set[str]
) -> dict[str, str]:
    target = summary.get("target_pool", {})
    raw = target.get("component_by_aggregate")
    if not isinstance(raw, Mapping):
        raise ValueError("Split summary is missing target_pool.component_by_aggregate.")
    mapping = {
        str(aggregate_id): str(component)
        for aggregate_id, component in raw.items()
        if str(aggregate_id) in test_ids
    }
    if set(mapping) != set(test_ids):
        missing = sorted(test_ids - set(mapping))[:5]
        extra = sorted(set(mapping) - test_ids)[:5]
        raise ValueError(
            "Test prediction identities and structure-component mapping differ: "
            f"missing={missing}, extra={extra}"
        )
    if any(not component for component in mapping.values()):
        raise ValueError("Test structure-component mapping contains empty values.")
    return mapping


def build_task_rows(
    ensembles: Mapping[str, list[dict[str, Any]]],
    *,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    component_by_aggregate: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        for task in tasks:
            selected = [row for row in ensembles[cell] if row["task_head"] == task]
            component_count = len(
                {
                    component_by_aggregate[str(row["aggregate_id"])]
                    for row in selected
                }
            )
            supported = len(selected) >= 30 and component_count >= 5
            if supported:
                item = metric_row(
                    cell,
                    "test",
                    selected,
                    aggregation="prediction_ensemble_task",
                    seeds=seeds,
                )
            else:
                item = low_support_task_metric_row(cell, selected, seeds=seeds)
            item["model_zh"] = MODEL_ZH[cell]
            item["task_head"] = task
            item["structure_component_count"] = component_count
            item["r2_support"] = (
                "supported_n30_components5"
                if supported
                else "low_support_report_mae_rmse_only"
            )
            if not supported:
                for field in (
                    "common_molkg_r2",
                    "common_mgkg_r2",
                    "within_r2_molkg",
                    "within_r2_mgkg",
                ):
                    item[field] = ""
            rows.append(item)
    return rows


def low_support_task_metric_row(
    cell: str, rows: list[dict[str, Any]], *, seeds: tuple[int, ...]
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Low-support task is empty for {cell}.")
    molar = metrics(rows, truth="y_molkg", prediction="pred_molkg")
    mass = metrics(rows, truth="y_mgkg", prediction="pred_mgkg")
    return {
        "cell": cell,
        "evaluation_part": "test",
        "aggregation": "prediction_ensemble_task",
        "seed": None,
        "seed_count": len(seeds),
        "seeds": ",".join(str(seed) for seed in seeds),
        "n": molar["n"],
        "common_molkg_r2": "",
        "common_molkg_rmse": molar["rmse"],
        "common_molkg_mae": molar["mae"],
        "common_mgkg_r2": "",
        "common_mgkg_rmse": mass["rmse"],
        "common_mgkg_mae": mass["mae"],
        "within_r2_molkg": "",
        "within_r2_mgkg": "",
        "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in rows),
        "result_ids_sha256": stable_hash(
            result_id for row in rows for result_id in row["result_ids"]
        ),
    }


if __name__ == "__main__":
    main()
