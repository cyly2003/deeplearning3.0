from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.baseline import load_split_frame
from scripts.build_v1_2_44_second_layer_splits import canonical_sha256, parse_result_ids
from scripts.build_v1_2_46_reference_group_splits import (
    PARENT_SPLIT,
    SOURCE_TABLE,
    build_reference_components,
    parse_string_set,
)
from scripts.summarize_v1_2_40_paired_mass_molar import write_csv
from scripts.summarize_v1_2_44_second_layer_matrix import (
    B2_TRAINED_CELLS,
    DEFAULT_SEEDS,
    PARTS,
    REQUIRED_CELLS,
    assert_aligned_prediction_sets,
    audit_run_manifests,
    build_ensembles,
    build_prediction_sets,
    discover_runs,
    require_complete_runs,
)


BOOTSTRAP_SEED = 20_260_720
DEFAULT_REPLICATES = 20_000
EQUIVALENCE_MARGIN = 0.01
CONTRASTS = (
    ("M10_minus_M00", "M10", "M00", "molkg", "aquatic_pretraining"),
    ("M01_minus_M00", "M01", "M00", "molkg", "soil_ptox_only"),
    ("M11U_minus_M10", "M11U", "M10", "molkg", "soil_ptox_increment"),
    ("M11U_minus_M11_v140", "M11U", "M11_v140", "molkg", "reproduction"),
    ("ScaleMol_minus_ScaleMass", "M11_v140", "ScaleMass", "mgkg", "target_scale"),
    ("M11U_minus_M11F", "M11U", "M11F", "molkg", "full_target_adaptation"),
    ("Full_minus_MoleculeOnly", "M11U", "B2M", "molkg", "context_complement"),
    ("Full_minus_ContextOnly", "M11U", "B2C", "molkg", "molecule_complement"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired reference-cluster bootstrap and configuration/data-boundary "
            "audits for the completed v1.2.44 matrix."
        )
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote"),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("outputs/experiments/第二层核心因果实验矩阵_v1_2_44"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/v1_2_45_reference_cluster_bootstrap"),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"),
    )
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--split-name", default=PARENT_SPLIT)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicates < 1_000:
        raise ValueError("At least 1,000 bootstrap replicates are required.")
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("Exactly four unique model seeds are required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.legacy_root, allowed={"M11_v140", "ScaleMass"})
    runs.extend(
        discover_runs(
            args.matrix_root,
            allowed={"M00", "M10", "M01", "M11F", "M11U", "B2C", "B2M"},
        )
    )
    run_map = require_complete_runs(runs, cells=REQUIRED_CELLS, seeds=seeds)
    manifest_audit = audit_run_manifests(run_map, seeds=seeds)
    prediction_sets = build_prediction_sets(run_map)
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=("M00", "M10", "M01", "M11F", "M11U", "M11_v140"),
        seeds=seeds,
        parts=PARTS,
    )
    assert_aligned_prediction_sets(
        prediction_sets, cells=B2_TRAINED_CELLS, seeds=seeds, parts=PARTS
    )
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=("M11_v140", "ScaleMass"),
        seeds=seeds,
        parts=PARTS,
    )
    ensembles = build_ensembles(prediction_sets, cells=REQUIRED_CELLS, seeds=seeds)

    split_frame = load_split_frame(
        args.db,
        split_name=args.split_name,
        source_table=args.source_table,
        allow_mixed_target_dimensions=True,
    )
    boundary_audit = audit_data_boundary(split_frame)
    target_test = split_frame.loc[
        (split_frame["split_part"].astype(str) == "test")
        & (split_frame["target_name"].astype(str) == "neg_log10_mol_kg")
        & (split_frame["target_family"].astype(str) == "solid_neglog_mol_kg")
    ].copy()
    reference_by_aggregate = {
        str(row["aggregate_id"]): row["reference_numbers"]
        for _, row in target_test.iterrows()
    }
    reference_records = [
        {
            "record_id": aggregate_id,
            "reference_numbers": references,
        }
        for aggregate_id, references in reference_by_aggregate.items()
    ]
    components = build_reference_components(reference_records)

    bootstrap_rows, distributions = paired_cluster_bootstrap(
        ensembles,
        group_by_aggregate=components["group_by_record"],
        contrasts=CONTRASTS,
        replicates=args.replicates,
        seed=args.bootstrap_seed,
        equivalence_margin=EQUIVALENCE_MARGIN,
    )
    seed_rows = build_seed_paired_deltas(prediction_sets, seeds=seeds, contrasts=CONTRASTS)
    configuration_rows = build_configuration_audit(run_map, seeds=seeds)

    outputs = {
        "bootstrap": args.output_dir / "paired_reference_cluster_bootstrap.csv",
        "seed_deltas": args.output_dir / "paired_seed_deltas.csv",
        "configuration": args.output_dir / "effective_configuration_audit.csv",
        "boundary": args.output_dir / "data_independence_audit.json",
        "distribution": args.output_dir / "bootstrap_distribution_quantiles.json",
        "summary": args.output_dir / "summary.json",
    }
    write_csv(outputs["bootstrap"], bootstrap_rows)
    write_csv(outputs["seed_deltas"], seed_rows)
    write_csv(outputs["configuration"], configuration_rows)
    outputs["boundary"].write_text(
        json.dumps(boundary_audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs["distribution"].write_text(
        json.dumps(distributions, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "v1_2_45_reference_cluster_bootstrap_v1",
        "status": "complete",
        "replicates": int(args.replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "cluster_unit": "connected_component_of_test_reference_numbers",
        "test_rows": len(next(iter(ensembles.values()))),
        "test_reference_components": int(components["component_count"]),
        "test_reference_ids": int(components["reference_count"]),
        "equivalence_margin_log10": EQUIVALENCE_MARGIN,
        "primary_metric": "paired_delta_mae",
        "ci_policy": {
            "uncertainty": "95_percentile_cluster_bootstrap",
            "equivalence": "90_percentile_CI_inside_plus_minus_0.01",
            "conservative_equivalence": "95_percentile_CI_inside_plus_minus_0.01",
        },
        "manifest_audit": manifest_audit,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def audit_data_boundary(frame: Any) -> dict[str, Any]:
    parts = ("train", "finetune", "finetune_mgkg", "valid", "test")
    identities: dict[str, dict[str, set[str]]] = {}
    for part in parts:
        selected = frame.loc[frame["split_part"].astype(str) == part]
        identities[part] = {
            "aggregate_id": {str(value) for value in selected["aggregate_id"].tolist()},
            "result_id": {
                value
                for raw in selected["result_ids"].tolist()
                for value in parse_result_ids(raw)
            },
            "test_id": {
                value
                for raw in selected["test_ids"].tolist()
                for value in parse_string_set(raw, field="test_ids")
            },
            "reference_number": {
                value
                for raw in selected["reference_numbers"].tolist()
                for value in parse_string_set(raw, field="reference_numbers")
            },
        }
    comparisons = (
        ("stage2_vs_stage3_test", "finetune", "test"),
        ("stage3_train_vs_valid", "finetune_mgkg", "valid"),
        ("stage3_train_vs_test", "finetune_mgkg", "test"),
        ("stage3_valid_vs_test", "valid", "test"),
    )
    overlap = {}
    for label, left, right in comparisons:
        overlap[label] = {
            field: len(identities[left][field] & identities[right][field])
            for field in identities[left]
        }
    exact_failures = {
        label: values
        for label, values in overlap.items()
        if values["aggregate_id"] or values["result_id"]
    }
    if exact_failures:
        raise ValueError(f"Exact aggregate/result reuse detected: {exact_failures}")
    return {
        "schema": "v1_2_45_data_independence_audit_v1",
        "counts": {
            part: {
                field: len(values) for field, values in identities[part].items()
            }
            for part in parts
        },
        "overlap": overlap,
        "interpretation": (
            "aggregate_id/result_id overlap is the fail-closed exact-source gate; "
            "test_id/reference overlap is reported as the random-split boundary and is not "
            "automatically labelled leakage"
        ),
    }


def paired_cluster_bootstrap(
    ensembles: Mapping[str, list[dict[str, Any]]],
    *,
    group_by_aggregate: Mapping[str, str],
    contrasts: Iterable[tuple[str, str, str, str, str]],
    replicates: int,
    seed: int,
    equivalence_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contrast_list = list(contrasts)
    cells = sorted({item for _, new, control, _, _ in contrast_list for item in (new, control)})
    ordered_ids = sorted(group_by_aggregate)
    for cell in cells:
        observed = {str(row["aggregate_id"]) for row in ensembles[cell]}
        if observed != set(ordered_ids):
            raise ValueError(
                f"Reference metadata and ensemble rows differ for {cell}: "
                f"metadata={len(ordered_ids)}, predictions={len(observed)}"
            )
    rows_by_cell = {
        cell: {str(row["aggregate_id"]): row for row in ensembles[cell]} for cell in cells
    }
    groups = sorted(set(group_by_aggregate.values()))
    group_index = {group: index for index, group in enumerate(groups)}
    codes = np.asarray(
        [group_index[group_by_aggregate[aggregate_id]] for aggregate_id in ordered_ids],
        dtype=np.int64,
    )
    predictions = {
        (cell, scale): np.asarray(
            [float(rows_by_cell[cell][identity][f"pred_{scale}"]) for identity in ordered_ids],
            dtype=np.float64,
        )
        for cell in cells
        for scale in ("molkg", "mgkg")
    }
    truths = {}
    reference_cell = cells[0]
    for scale in ("molkg", "mgkg"):
        truth = np.asarray(
            [
                float(rows_by_cell[reference_cell][identity][f"y_{scale}"])
                for identity in ordered_ids
            ],
            dtype=np.float64,
        )
        for cell in cells[1:]:
            candidate = np.asarray(
                [float(rows_by_cell[cell][identity][f"y_{scale}"]) for identity in ordered_ids],
                dtype=np.float64,
            )
            if not np.allclose(candidate, truth, rtol=1e-12, atol=1e-10):
                raise ValueError(f"Truth values are not aligned across models on {scale}.")
        truths[scale] = truth

    observed_metrics = {
        (cell, scale): array_metrics(truths[scale], predictions[(cell, scale)])
        for cell in cells
        for scale in ("molkg", "mgkg")
    }
    rng = np.random.default_rng(seed)
    distributions: dict[str, dict[str, list[float]]] = {
        label: {"delta_r2": [], "delta_rmse": [], "delta_mae": []}
        for label, *_ in contrast_list
    }
    probabilities = np.full(len(groups), 1.0 / len(groups), dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        group_weights = rng.multinomial(len(groups), probabilities, size=size)
        row_weights = group_weights[:, codes].astype(np.float64, copy=False)
        chunk_metrics: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for scale in ("molkg", "mgkg"):
            truth = truths[scale]
            sum_weight = row_weights.sum(axis=1)
            sum_y = row_weights @ truth
            sum_y2 = row_weights @ np.square(truth)
            sst = sum_y2 - np.square(sum_y) / sum_weight
            for cell in cells:
                residual = truth - predictions[(cell, scale)]
                sae = row_weights @ np.abs(residual)
                sse = row_weights @ np.square(residual)
                chunk_metrics[(cell, scale)] = {
                    "mae": sae / sum_weight,
                    "rmse": np.sqrt(sse / sum_weight),
                    "r2": 1.0 - sse / sst,
                }
        for label, new, control, scale, _ in contrast_list:
            for metric in ("r2", "rmse", "mae"):
                delta = (
                    chunk_metrics[(new, scale)][metric]
                    - chunk_metrics[(control, scale)][metric]
                )
                distributions[label][f"delta_{metric}"].extend(delta.tolist())

    output = []
    quantile_output: dict[str, Any] = {}
    for label, new, control, scale, scientific_role in contrast_list:
        item: dict[str, Any] = {
            "contrast": label,
            "scientific_role": scientific_role,
            "new_model": new,
            "control_model": control,
            "scale": f"neg_log10_{scale}",
            "n": len(ordered_ids),
            "reference_components": len(groups),
            "replicates": replicates,
            "bootstrap_seed": seed,
            "delta_definition": "metric(new_model)-metric(control_model)",
            "equivalence_margin_mae": equivalence_margin,
        }
        quantile_output[label] = {}
        for metric in ("r2", "rmse", "mae"):
            point = observed_metrics[(new, scale)][metric] - observed_metrics[(control, scale)][metric]
            values = np.asarray(distributions[label][f"delta_{metric}"], dtype=np.float64)
            q = np.quantile(values, [0.025, 0.05, 0.5, 0.95, 0.975])
            item[f"delta_{metric}"] = float(point)
            item[f"delta_{metric}_ci95_low"] = float(q[0])
            item[f"delta_{metric}_ci90_low"] = float(q[1])
            item[f"delta_{metric}_bootstrap_median"] = float(q[2])
            item[f"delta_{metric}_ci90_high"] = float(q[3])
            item[f"delta_{metric}_ci95_high"] = float(q[4])
            quantile_output[label][metric] = {
                "q025": float(q[0]),
                "q05": float(q[1]),
                "q50": float(q[2]),
                "q95": float(q[3]),
                "q975": float(q[4]),
            }
        item["mae_equivalent_90"] = bool(
            item["delta_mae_ci90_low"] > -equivalence_margin
            and item["delta_mae_ci90_high"] < equivalence_margin
        )
        item["mae_equivalent_95_conservative"] = bool(
            item["delta_mae_ci95_low"] > -equivalence_margin
            and item["delta_mae_ci95_high"] < equivalence_margin
        )
        item["new_model_mae_superior_95"] = bool(item["delta_mae_ci95_high"] < 0)
        output.append(item)
    return output, quantile_output


def build_seed_paired_deltas(
    prediction_sets: Mapping[tuple[str, int, str], list[dict[str, Any]]],
    *,
    seeds: Iterable[int],
    contrasts: Iterable[tuple[str, str, str, str, str]],
) -> list[dict[str, Any]]:
    output = []
    for label, new, control, scale, role in contrasts:
        for seed in seeds:
            new_rows = {
                str(row["aggregate_id"]): row
                for row in prediction_sets[(new, int(seed), "test")]
            }
            control_rows = {
                str(row["aggregate_id"]): row
                for row in prediction_sets[(control, int(seed), "test")]
            }
            if set(new_rows) != set(control_rows):
                raise ValueError(f"Seed-level paired rows differ for {label}, seed={seed}.")
            ids = sorted(new_rows)
            truth = np.asarray([float(new_rows[i][f"y_{scale}"]) for i in ids])
            new_pred = np.asarray([float(new_rows[i][f"pred_{scale}"]) for i in ids])
            control_pred = np.asarray([float(control_rows[i][f"pred_{scale}"]) for i in ids])
            new_metric = array_metrics(truth, new_pred)
            control_metric = array_metrics(truth, control_pred)
            output.append(
                {
                    "contrast": label,
                    "scientific_role": role,
                    "seed": int(seed),
                    "scale": f"neg_log10_{scale}",
                    "n": len(ids),
                    "delta_r2": new_metric["r2"] - control_metric["r2"],
                    "delta_rmse": new_metric["rmse"] - control_metric["rmse"],
                    "delta_mae": new_metric["mae"] - control_metric["mae"],
                }
            )
    return output


def build_configuration_audit(
    run_map: Mapping[tuple[str, int], Mapping[str, Any]], *, seeds: Iterable[int]
) -> list[dict[str, Any]]:
    output = []
    for (cell, seed), run in sorted(run_map.items()):
        manifest = run["manifest"]
        stage3 = manifest.get("finetune_mgkg", {})
        source_weight = manifest.get("source_weighting", {})
        checkpoint = stage3.get("stage3_init_checkpoint", {})
        freeze = str(stage3.get("freeze", ""))
        trunk_lr = stage3.get("trunk_learning_rate")
        unified_lr = stage3.get("learning_rate")
        output.append(
            {
                "cell": cell,
                "seed": int(seed),
                "split_name": manifest.get("split_name"),
                "stage1_epochs": manifest.get("epochs"),
                "stage2_epochs": manifest.get("finetune_epochs"),
                "stage3_epochs": manifest.get("finetune_mgkg_epochs"),
                "stage3_freeze": freeze,
                "stage3_learning_rate": unified_lr,
                "stage3_trunk_learning_rate": trunk_lr,
                "stage3_all_parameters_trainable": freeze == "none" and trunk_lr in (None, 0, 0.0),
                "stage3_learning_rate_interpretation": (
                    "unified_lr_for_all_trainable_parameters"
                    if freeze == "none" and trunk_lr in (None, 0, 0.0)
                    else "separate_or_frozen_policy"
                ),
                "head_routing": manifest.get("head_routing"),
                "use_medium_adapter": manifest.get("ablation_features", {}).get(
                    "use_medium_adapter"
                ),
                "source_weighting_method": source_weight.get("method"),
                "source_weighting_applied": source_weight.get("applied"),
                "stage3_checkpoint_loaded": checkpoint.get("loaded"),
                "stage3_checkpoint_exported": checkpoint.get("exported"),
                "stage1_stage2_skipped": checkpoint.get("stage1_stage2_skipped"),
                "run_dir": run["run_dir"],
            }
        )
    if {int(row["seed"]) for row in output} != {int(seed) for seed in seeds}:
        raise ValueError("Configuration audit did not cover all expected seeds.")
    return output


def array_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    sse = float(np.square(residual).sum())
    sst = float(np.square(y_true - y_true.mean()).sum())
    return {
        "r2": 1.0 - sse / sst,
        "rmse": math.sqrt(float(np.square(residual).mean())),
        "mae": float(np.abs(residual).mean()),
    }


if __name__ == "__main__":
    main()
