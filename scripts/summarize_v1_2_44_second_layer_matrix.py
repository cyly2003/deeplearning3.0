from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_v1_2_40_paired_mass_molar import (
    metrics,
    parse_result_ids,
    pearson,
    stable_hash,
    unique_by_aggregate,
    write_csv,
)


DEFAULT_SEEDS = (42, 2042, 3407, 8417)
PARTS = ("target_train", "validation", "test")

# Canonical IDs are deliberately short and ASCII so downstream tables remain
# machine-friendly. Discovery also accepts the required Chinese run folders.
CELL_PATTERNS: dict[str, tuple[str, ...]] = {
    "M00": (
        r"(?:^|[/_.-])M00(?:_|-)",
        r"M00_从头训练_固定评价边界_种子\d+",
    ),
    "M11_v140": (r"(?:^|[/_.-])X0_molar_seed\d+",),
    "ScaleMass": (r"(?:^|[/_.-])X0_mass_seed\d+",),
    "M10": (r"(?:^|[/_.-])M10(?:_|-)", r"M10_仅水相预训练_种子\d+"),
    "M01": (r"(?:^|[/_.-])M01(?:_|-)", r"M01_仅土壤pTox适配_种子\d+"),
    "M11F": (r"(?:^|[/_.-])M11F(?:_|-)", r"M11F_冻结主干_种子\d+"),
    "M11U": (
        r"(?:^|[/_.-])M11U(?:_|-|复现)",
        r"M11U复现_全参数微调_种子\d+",
    ),
    "B2C": (
        r"(?:^|[/_.-])B2C(?:_|-)",
        r"B2C_仅上下文输入_种子\d+",
        r"B2_仅上下文输入_种子\d+",
    ),
    "B2M": (
        r"(?:^|[/_.-])B2M(?:_|-)",
        r"B2M_仅分子输入_种子\d+",
        r"B2_仅分子输入_种子\d+",
    ),
}

CELL_SCALE = {
    "M00": "molar",
    "M11_v140": "molar",
    "ScaleMass": "mass",
    "M10": "molar",
    "M01": "molar",
    "M11F": "molar",
    "M11U": "molar",
    "B2C": "molar",
    "B2M": "molar",
}

CELL_ZH = {
    "M00": "M00 从头训练",
    "M11_v140": "M11U v1.2.40主线",
    "ScaleMass": "Scale-Mass",
    "M10": "M10 仅水相预训练",
    "M01": "M01 仅土壤pTox适配",
    "M11F": "M11F 冻结主干",
    "M11U": "M11U复现 全参数微调",
    "B2C": "Context-only 仅上下文输入",
    "B2M": "Molecule-only 仅分子输入",
    "MeanBaseline": "Mean baseline 训练集任务均值",
    "MWOnly": "MW-only 分子量与任务均值",
}

REQUIRED_CELLS = tuple(CELL_SCALE)
B1_CELLS = ("M00", "M10", "M01", "M11F", "M11U")
B2_TRAINED_CELLS = ("B2C", "B2M", "M11U")
B1_CONTRASTS = (
    ("M10", "M11U", "M11U−M10 土壤pTox适配贡献"),
    ("M01", "M11U", "M11U−M01 水相预训练贡献"),
    ("M00", "M11U", "M11U−M00 完整迁移总贡献"),
    ("M11F", "M11U", "M11U−M11F 解冻主干贡献"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="严格汇总v1.2.44第二层核心因果实验矩阵。"
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("outputs/experiments/v1_2_40_paired_mass_molar_matrix_remote"),
        help="v1.2.40 X0_molar/X0_mass原始结果根目录；旧D_molar不进入严格B1矩阵。",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("outputs/experiments/第二层核心因果实验矩阵_v1_2_44"),
        help="v1.2.44中文实验矩阵根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="中文汇总目录；默认位于matrix-root/统一汇总_中文。",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--expected-task-count", type=int, default=18)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(int(seed) for seed in args.seeds)
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError(f"Exactly four unique seeds are required; received {seeds}")
    output_dir = args.output_dir or args.matrix_root / "统一汇总_中文"
    output_dir.mkdir(parents=True, exist_ok=True)

    # The historical D_molar baseline used an internal validation split with a
    # different seed and exported train+validation as one `train` block.  It is
    # therefore not a protocol-compatible M00 for the fixed 9724/2433/3042 B1
    # boundary and is deliberately excluded instead of being relabelled.
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

    # The causal matrix must share the complete Stage-3 train/validation/test
    # boundary, not merely the test count. v1.2.40 M11 is included as a replay
    # reference, while B3 checks the paired mass/molar target identities.
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=("M00", "M11_v140", "M10", "M01", "M11F", "M11U"),
        seeds=seeds,
        parts=PARTS,
    )
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=B2_TRAINED_CELLS,
        seeds=seeds,
        parts=PARTS,
    )
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=("M11_v140", "ScaleMass"),
        seeds=seeds,
        parts=PARTS,
    )

    tasks = require_task_boundary(
        prediction_sets,
        cells=B1_CELLS,
        seeds=seeds,
        expected_count=args.expected_task_count,
    )

    seed_metrics = build_seed_metrics(prediction_sets, cells=REQUIRED_CELLS, seeds=seeds)
    mean_sd = summarize_mean_sd(seed_metrics)
    ensembles = build_ensembles(prediction_sets, cells=REQUIRED_CELLS, seeds=seeds)
    ensemble_metrics = [
        metric_row(cell, "test", rows, aggregation="prediction_ensemble", seeds=seeds)
        for cell, rows in sorted(ensembles.items())
    ]

    b1_task_seed = build_b1_task_seed_metrics(prediction_sets, seeds=seeds, tasks=tasks)
    b1_task_ensemble = build_b1_task_ensemble_metrics(ensembles, tasks=tasks, seeds=seeds)
    b1_heatmap = build_b1_heatmap_data(
        b1_task_seed,
        b1_task_ensemble,
        seeds=seeds,
        tasks=tasks,
    )
    b2_rows, b2_mean_sd = build_b2_comparison(
        prediction_sets,
        ensembles,
        seeds=seeds,
        disclosures=manifest_audit["b2_disclosures"],
    )
    b3 = build_b3_analysis(prediction_sets, ensembles, seeds=seeds, tasks=tasks)
    replay_rows = build_replay_contrast(seed_metrics, ensemble_metrics, seeds=seeds)

    outputs: dict[str, Path] = {
        "seed_metrics": output_dir / "01_逐种子整体指标.csv",
        "mean_sd": output_dir / "02_四种子均值与标准差.csv",
        "ensemble_metrics": output_dir / "03_四种子逐行集成主指标.csv",
        "b1_task_seed": output_dir / "04_B1_18任务逐种子指标.csv",
        "b1_task_ensemble": output_dir / "05_B1_18任务四种子集成指标.csv",
        "b1_heatmap": output_dir / "06_B1_18任务迁移增益热图数据.csv",
        "b2": output_dir / "07_B2_任务均值与输入信号对照.csv",
        "b2_mean_sd": output_dir / "07b_B2_四种子均值与标准差.csv",
        "b3_metrics": output_dir / "08_B3_尺度对照指标.csv",
        "b3_row_error": output_dir / "09_B3_逐行绝对误差与分子量.csv",
        "b3_mw_bins": output_dir / "10_B3_MW分层增益.csv",
        "b3_task_span": output_dir / "11_B3_任务MW跨度与增益.csv",
        "b3_relation": output_dir / "12_B3_误差分子量相关性.csv",
        "replay": output_dir / "13_M11U复现与v1_2_40主线对照.csv",
    }
    write_csv(outputs["seed_metrics"], seed_metrics)
    write_csv(outputs["mean_sd"], mean_sd)
    write_csv(outputs["ensemble_metrics"], ensemble_metrics)
    write_csv(outputs["b1_task_seed"], b1_task_seed)
    write_csv(outputs["b1_task_ensemble"], b1_task_ensemble)
    write_csv(outputs["b1_heatmap"], b1_heatmap)
    write_csv(outputs["b2"], b2_rows)
    write_csv(outputs["b2_mean_sd"], b2_mean_sd)
    write_csv(outputs["b3_metrics"], b3["metrics"])
    write_csv(outputs["b3_row_error"], b3["row_error"])
    write_csv(outputs["b3_mw_bins"], b3["mw_bins"])
    write_csv(outputs["b3_task_span"], b3["task_span"])
    write_csv(outputs["b3_relation"], b3["relation"])
    write_csv(outputs["replay"], replay_rows)

    figure_base = output_dir / "B1_18任务迁移增益热图"
    plot_b1_heatmap(b1_heatmap, tasks=tasks, output_base=figure_base)

    summary = {
        "schema": "v1_2_44_second_layer_summary_v1",
        "status": "complete",
        "legacy_root": str(args.legacy_root),
        "matrix_root": str(args.matrix_root),
        "output_dir": str(output_dir),
        "cells": list(REQUIRED_CELLS),
        "seeds": list(seeds),
        "task_count": len(tasks),
        "tasks": list(tasks),
        "selection_uses_test": False,
        "primary_reporting": "four_seed_rowwise_prediction_ensemble_on_test",
        "within_r2_definition": (
            "1-sum((y-yhat)^2)/sum((y-mean_y_within_evaluation_task)^2)"
        ),
        "mean_baseline_fit_boundary": "target_train_only_per_seed_and_task",
        "delta_mae_sign": "positive means M11U has lower MAE than the comparator",
        "figure_contract": {
            "core_conclusion": "Identify which of the 18 tasks benefit from each transfer component.",
            "archetype": "quantitative_grid",
            "hero_panel": "B1_18任务迁移增益热图",
            "source_data": str(outputs["b1_heatmap"]),
            "exports": [str(figure_base.with_suffix(".png")), str(figure_base.with_suffix(".svg"))],
            "review_risks": [
                "row_identity_mismatch",
                "task_boundary_drift",
                "delta_mae_sign_confusion",
                "small_task_support",
            ],
        },
        "alignment": build_alignment_audit(prediction_sets, seeds=seeds),
        "manifest_audit": manifest_audit,
        "b3_task_span_delta_mae_logmw_span_pearson": b3["task_span_pearson"],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    summary_path = output_dir / "汇总说明.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def discover_runs(root: Path, *, allowed: set[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"Missing run root: {root}")
    discovered: list[dict[str, Any]] = []
    for manifest_path in root.glob("**/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Match only the run-relative path. Parent folders (including pytest
        # temp paths or a user's workspace name) may themselves contain labels
        # such as M01 and must not influence cell identity.
        path_text = "/".join(manifest_path.relative_to(root).parts)
        cell = identify_cell(path_text, allowed=allowed)
        if cell is None:
            # Restrict fallback matching to explicit run labels. Searching the
            # entire manifest can falsely match a referenced checkpoint path.
            label_text = "\n".join(
                str(manifest.get(key, ""))
                for key in ("run_name", "run_name_zh", "experiment_name", "version")
            )
            cell = identify_cell(label_text, allowed=allowed)
        if cell is None:
            continue
        predictions_path = manifest_path.parent / "predictions.csv"
        if not predictions_path.is_file() or predictions_path.stat().st_size <= 0:
            raise ValueError(f"Missing predictions.csv for identified cell {cell}: {manifest_path.parent}")
        path_seed = seed_from_text(path_text)
        manifest_seed = optional_int(manifest.get("seed"))
        seed = manifest_seed if manifest_seed is not None else path_seed
        if seed is None:
            raise ValueError(f"Cannot determine seed for {manifest_path}")
        if path_seed is not None and manifest_seed is not None and path_seed != manifest_seed:
            raise ValueError(
                f"Seed mismatch between path and manifest for {manifest_path}: "
                f"path={path_seed}, manifest={manifest_seed}"
            )
        predictions = stream_final_predictions(predictions_path, scale=CELL_SCALE[cell])
        discovered.append(
            {
                "cell": cell,
                "seed": int(seed),
                "manifest": manifest,
                "normalized_predictions": predictions,
                "run_dir": str(manifest_path.parent),
            }
        )
    return discovered


def stream_final_predictions(path: Path, *, scale: str) -> list[dict[str, Any]]:
    """Stream one large predictions.csv and retain only final soil-target rows."""
    expected_target = {
        "molar": ("neg_log10_mol_kg", "solid_neglog_mol_kg"),
        "mass": ("neg_log10_mg_kg", "solid_neglog_mg_kg"),
    }[scale]
    required = {
        "aggregate_id",
        "split_part",
        "task_head",
        "target_name",
        "target_family",
        "medium_domain",
        "y_true",
        "y_pred",
        "molecular_weight_g_mol_used",
    }
    retained: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"Prediction schema missing columns {missing_columns}: {path}")
        for raw in reader:
            if (
                str(raw.get("target_name", "")).strip() != expected_target[0]
                or str(raw.get("target_family", "")).strip() != expected_target[1]
                or str(raw.get("medium_domain", "")).strip().lower() != "soil"
            ):
                continue
            part_raw = str(raw.get("split_part", "")).strip().lower()
            if part_raw in {"train", "finetune_mgkg"}:
                part = "target_train"
            elif part_raw in {"validation", "valid", "finetune_mgkg_validation"}:
                part = "validation"
            elif part_raw == "test":
                part = "test"
            else:
                continue
            aggregate_id = str(raw.get("aggregate_id", "")).strip()
            task = str(raw.get("base_task_head") or raw.get("task_head") or "").strip()
            if not aggregate_id or not task:
                raise ValueError(f"Final soil row lacks aggregate_id/task_head in {path}")
            mw = require_finite_float(raw.get("molecular_weight_g_mol_used"), "molecular_weight")
            if mw <= 0:
                raise ValueError(f"Non-positive molecular weight for aggregate_id={aggregate_id} in {path}")
            y_native = require_finite_float(raw.get("y_true"), "y_true")
            pred_native = require_finite_float(raw.get("y_pred"), "y_pred")
            offset = math.log10(1000.0 * mw)
            if scale == "molar":
                y_molkg, pred_molkg = y_native, pred_native
                y_mgkg, pred_mgkg = y_native - offset, pred_native - offset
            else:
                y_mgkg, pred_mgkg = y_native, pred_native
                y_molkg, pred_molkg = y_native + offset, pred_native + offset
            retained.append(
                {
                    "aggregate_id": aggregate_id,
                    "task_head": task,
                    "evaluation_part": part,
                    "original_split_part": str(raw.get("original_split_part", "")).strip().lower(),
                    "y_native": y_native,
                    "pred_native": pred_native,
                    "y_mgkg": y_mgkg,
                    "pred_mgkg": pred_mgkg,
                    "y_molkg": y_molkg,
                    "pred_molkg": pred_molkg,
                    "molecular_weight": mw,
                    "log10_mw": math.log10(mw),
                    "multifragment": "." in str(raw.get("smiles", "") or ""),
                    "result_ids": parse_result_ids(raw.get("result_ids")),
                }
            )
    if not retained:
        raise ValueError(
            f"No final soil {expected_target[0]}/{expected_target[1]} rows found in {path}"
        )
    return retained


def require_finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return parsed


def identify_cell(text: str, *, allowed: set[str] | None = None) -> str | None:
    matches = []
    for cell, patterns in CELL_PATTERNS.items():
        if allowed is not None and cell not in allowed:
            continue
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matches.append(cell)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous cell identity {matches}")
    return matches[0] if matches else None


def seed_from_text(text: str) -> int | None:
    matches = re.findall(r"(?:seed|种子)[_-]?(\d+)", text, flags=re.IGNORECASE)
    unique = {int(value) for value in matches}
    if len(unique) > 1:
        raise ValueError(f"Ambiguous seeds in path: {sorted(unique)}")
    return next(iter(unique)) if unique else None


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def require_complete_runs(
    runs: list[dict[str, Any]], *, cells: Iterable[str], seeds: Iterable[int]
) -> dict[tuple[str, int], dict[str, Any]]:
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        key = (str(run["cell"]), int(run["seed"]))
        if key in unique:
            raise ValueError(f"Duplicate run for cell/seed={key}")
        unique[key] = run
    expected = {(cell, int(seed)) for cell in cells for seed in seeds}
    missing = sorted(expected - set(unique))
    extra = sorted(set(unique) - expected)
    if missing or extra:
        raise ValueError(f"Run matrix is incomplete: missing={missing}, extra={extra}")
    return unique


def audit_run_manifests(
    run_map: dict[tuple[str, int], dict[str, Any]], *, seeds: Iterable[int]
) -> dict[str, Any]:
    expected_ablation = {cell: "full" for cell in REQUIRED_CELLS}
    expected_ablation.update({"B2C": "no_molecular_input", "B2M": "no_context"})
    per_cell: dict[str, dict[str, Any]] = {}
    for cell in REQUIRED_CELLS:
        observations = []
        for seed in seeds:
            manifest = run_map[(cell, int(seed))]["manifest"]
            ablation = str(manifest.get("ablation", ""))
            features = manifest.get("ablation_features")
            if not isinstance(features, dict):
                raise ValueError(f"Manifest lacks ablation_features for {cell}, seed={seed}")
            if ablation != expected_ablation[cell]:
                raise ValueError(
                    f"Unexpected ablation for {cell}, seed={seed}: "
                    f"observed={ablation!r}, expected={expected_ablation[cell]!r}"
                )
            if features.get("use_medium_adapter") is not False:
                raise ValueError(f"Medium adapter must be disabled for {cell}, seed={seed}")
            head_routing = str(manifest.get("head_routing", ""))
            source_weighting = manifest.get("source_weighting")
            if not isinstance(source_weighting, dict):
                raise ValueError(f"Manifest lacks source_weighting audit for {cell}, seed={seed}")
            observation = {
                "ablation": ablation,
                "head_routing": head_routing,
                "source_weighting_method": str(source_weighting.get("method", "")),
                "source_weighting_applied": bool(source_weighting.get("applied", False)),
                "use_descriptors": features.get("use_descriptors"),
                "use_fingerprint": features.get("use_fingerprint"),
                "use_molecular_graph": features.get("use_molecular_graph"),
                "use_context_numeric": features.get("use_context_numeric"),
                "use_duration_features": features.get("use_duration_features"),
                "use_species_lifestage": features.get("use_species_lifestage"),
                "use_other_categorical_context": features.get("use_other_categorical_context"),
                "use_medium_adapter": features.get("use_medium_adapter"),
            }
            observations.append(observation)
        canonical = observations[0]
        if any(item != canonical for item in observations[1:]):
            raise ValueError(f"Manifest protocol differs across seeds for {cell}")
        per_cell[cell] = canonical

    context = per_cell["B2C"]
    if not (
        context["head_routing"] == "task_target"
        and context["use_descriptors"] is False
        and context["use_fingerprint"] is False
        and context["use_molecular_graph"] is False
        and context["source_weighting_method"] == "tanimoto_to_finetune"
        and context["source_weighting_applied"] is True
    ):
        raise ValueError(
            "B2 context-only protocol must remove molecular model inputs while retaining "
            "applied tanimoto_to_finetune source weights and task_target routing"
        )
    molecule = per_cell["B2M"]
    if not (
        molecule["head_routing"] == "task_target"
        and molecule["use_descriptors"] is True
        and molecule["use_fingerprint"] is True
        and molecule["use_context_numeric"] is False
        and molecule["use_duration_features"] is False
        and molecule["use_species_lifestage"] is False
        and molecule["use_other_categorical_context"] is False
        and molecule["source_weighting_method"] == "tanimoto_to_finetune"
        and molecule["source_weighting_applied"] is True
    ):
        raise ValueError(
            "B2 molecule-only protocol must remove experimental context, retain molecular "
            "descriptors/fingerprint, applied tanimoto weights, and task_target routing"
        )
    full = per_cell["M11U"]
    if not (
        full["head_routing"] == "task_target"
        and full["use_descriptors"] is True
        and full["use_fingerprint"] is True
        and full["use_context_numeric"] is True
        and full["source_weighting_method"] == "tanimoto_to_finetune"
        and full["source_weighting_applied"] is True
    ):
        raise ValueError("M11U Full manifest does not match the expected full-input protocol")

    disclosures = {
        "MeanBaseline": {
            "model_input": "task identity only; train-only task mean",
            "training_source_weights": "none",
            "task_routing_disclosure": "task identity defines the train-only mean",
            "indirect_molecular_information": "none",
        },
        "B2C": {
            "model_input": "no molecular descriptors/fingerprint/graph",
            "training_source_weights": context["source_weighting_method"],
            "task_routing_disclosure": context["head_routing"],
            "indirect_molecular_information": (
                "yes; source weights use Morgan Tanimoto similarity to finetune chemicals"
            ),
        },
        "B2M": {
            "model_input": "molecular descriptors and fingerprint; no experimental context",
            "training_source_weights": molecule["source_weighting_method"],
            "task_routing_disclosure": (
                "task_target output heads remain; task identity is retained in routing"
            ),
            "indirect_molecular_information": "not applicable; molecular input is explicit",
        },
        "M11U": {
            "model_input": "molecular structure plus species and experimental context",
            "training_source_weights": full["source_weighting_method"],
            "task_routing_disclosure": full["head_routing"],
            "indirect_molecular_information": "molecular information is explicit",
        },
    }
    return {
        "all_cells_ablation_and_no_medium_adapter_verified": True,
        "per_cell": per_cell,
        "b2_disclosures": disclosures,
    }


def build_prediction_sets(
    run_map: dict[tuple[str, int], dict[str, Any]]
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    output: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for (cell, seed), run in sorted(run_map.items()):
        enriched = list(run["normalized_predictions"])
        for part in PARTS:
            selected = [row for row in enriched if row["evaluation_part"] == part]
            if not selected:
                raise ValueError(f"Missing {part} predictions for cell={cell}, seed={seed}")
            unique_by_aggregate(selected)
            output[(cell, seed, part)] = selected
    return output


def assert_aligned_prediction_sets(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    cells: Iterable[str],
    seeds: Iterable[int],
    parts: Iterable[str],
) -> None:
    for part in parts:
        keys = [(cell, int(seed), part) for seed in seeds for cell in cells]
        missing = [key for key in keys if key not in prediction_sets]
        if missing:
            raise ValueError(f"Missing prediction sets for alignment: {missing}")
        reference = unique_by_aggregate(prediction_sets[keys[0]])
        for key in keys[1:]:
            candidate = unique_by_aggregate(prediction_sets[key])
            if set(candidate) != set(reference):
                missing_ids = sorted(set(reference) - set(candidate))[:5]
                extra_ids = sorted(set(candidate) - set(reference))[:5]
                raise ValueError(
                    f"Row identity mismatch for {key}: missing={missing_ids}, extra={extra_ids}"
                )
            for aggregate_id, expected in reference.items():
                actual = candidate[aggregate_id]
                if actual["task_head"] != expected["task_head"]:
                    raise ValueError(f"Task mismatch for {key}, aggregate_id={aggregate_id}")
                if tuple(actual["result_ids"]) != tuple(expected["result_ids"]):
                    raise ValueError(f"Result-ID mismatch for {key}, aggregate_id={aggregate_id}")
                for field in ("y_mgkg", "y_molkg", "molecular_weight"):
                    if not math.isclose(
                        float(actual[field]),
                        float(expected[field]),
                        rel_tol=1e-12,
                        abs_tol=1e-10,
                    ):
                        raise ValueError(
                            f"Truth/MW mismatch for {key}, aggregate_id={aggregate_id}, field={field}"
                        )


def require_task_boundary(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    cells: Iterable[str],
    seeds: Iterable[int],
    expected_count: int,
) -> tuple[str, ...]:
    reference: set[str] | None = None
    for cell in cells:
        for seed in seeds:
            tasks = {row["task_head"] for row in prediction_sets[(cell, int(seed), "test")]}
            if reference is None:
                reference = tasks
            elif tasks != reference:
                raise ValueError(f"Task boundary mismatch for cell={cell}, seed={seed}")
    ordered = tuple(sorted(reference or (), key=task_sort_key))
    if len(ordered) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} test tasks, observed {len(ordered)}: {ordered}"
        )
    return ordered


def task_sort_key(task: str) -> tuple[int, str]:
    prefix = task.split("_", 1)[0]
    order = {"ECx": 0, "ICx": 1, "LOEC": 2, "NOEC": 3}
    return order.get(prefix, 9), task


def metric_row(
    cell: str,
    part: str,
    rows: list[dict[str, Any]],
    *,
    aggregation: str,
    seed: int | None = None,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    molar = metrics(rows, truth="y_molkg", prediction="pred_molkg")
    mass = metrics(rows, truth="y_mgkg", prediction="pred_mgkg")
    return {
        "cell": cell,
        "model_zh": CELL_ZH.get(cell, cell),
        "evaluation_part": part,
        "aggregation": aggregation,
        "seed": seed,
        "seed_count": 1 if seed is not None else len(tuple(seeds or ())),
        "seeds": "" if seeds is None else ",".join(str(value) for value in seeds),
        "n": molar["n"],
        "common_molkg_r2": molar["r2"],
        "common_molkg_rmse": molar["rmse"],
        "common_molkg_mae": molar["mae"],
        "common_mgkg_r2": mass["r2"],
        "common_mgkg_rmse": mass["rmse"],
        "common_mgkg_mae": mass["mae"],
        "within_r2_molkg": within_r2(rows, truth="y_molkg", prediction="pred_molkg"),
        "within_r2_mgkg": within_r2(rows, truth="y_mgkg", prediction="pred_mgkg"),
        "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in rows),
        "result_ids_sha256": stable_hash(
            result_id for row in rows for result_id in row["result_ids"]
        ),
    }


def within_r2(rows: list[dict[str, Any]], *, truth: str, prediction: str) -> float:
    if not rows:
        raise ValueError("Within-task R2 requires non-empty evaluation rows")
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_head"])].append(float(row[truth]))
    task_means = {task: statistics.fmean(values) for task, values in by_task.items()}
    ss_res = sum((float(row[truth]) - float(row[prediction])) ** 2 for row in rows)
    ss_within = sum(
        (float(row[truth]) - task_means[str(row["task_head"])]) ** 2 for row in rows
    )
    if ss_within <= 0:
        raise ValueError("Within-task R2 denominator is zero")
    return 1.0 - ss_res / ss_within


def build_seed_metrics(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    cells: Iterable[str],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    output = []
    for cell in cells:
        for seed in seeds:
            for part in ("validation", "test"):
                output.append(
                    metric_row(
                        cell,
                        part,
                        prediction_sets[(cell, int(seed), part)],
                        aggregation="single_seed",
                        seed=int(seed),
                    )
                )
    return output


def summarize_mean_sd(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cell"]), str(row["evaluation_part"]))].append(row)
    fields = (
        "common_molkg_r2",
        "common_molkg_rmse",
        "common_molkg_mae",
        "common_mgkg_r2",
        "common_mgkg_rmse",
        "common_mgkg_mae",
        "within_r2_molkg",
        "within_r2_mgkg",
    )
    output = []
    for (cell, part), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "cell": cell,
            "model_zh": CELL_ZH.get(cell, cell),
            "evaluation_part": part,
            "seed_count": len(group),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            "n": group[0]["n"],
        }
        if any(row["n"] != item["n"] for row in group):
            raise ValueError(f"Evaluation n differs across seeds for {cell}, {part}")
        for field in fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def build_ensembles(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    cells: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, list[dict[str, Any]]]:
    seed_list = tuple(int(seed) for seed in seeds)
    output: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        indexed = [unique_by_aggregate(prediction_sets[(cell, seed, "test")]) for seed in seed_list]
        identities = set(indexed[0])
        if any(set(rows) != identities for rows in indexed[1:]):
            raise ValueError(f"Ensemble identities differ across seeds for {cell}")
        ensemble = []
        for aggregate_id in sorted(identities):
            source = [rows[aggregate_id] for rows in indexed]
            reference = dict(source[0])
            for row in source[1:]:
                if (
                    row["task_head"] != reference["task_head"]
                    or abs(float(row["y_mgkg"]) - float(reference["y_mgkg"])) > 1e-10
                    or abs(float(row["y_molkg"]) - float(reference["y_molkg"])) > 1e-10
                ):
                    raise ValueError(f"Ensemble truth/task differs for {cell}, id={aggregate_id}")
            reference["pred_mgkg"] = statistics.fmean(float(row["pred_mgkg"]) for row in source)
            reference["pred_molkg"] = statistics.fmean(float(row["pred_molkg"]) for row in source)
            ensemble.append(reference)
        output[cell] = ensemble
    return output


def build_b1_task_seed_metrics(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    seeds: Iterable[int],
    tasks: Iterable[str],
) -> list[dict[str, Any]]:
    output = []
    for cell in B1_CELLS:
        for seed in seeds:
            rows = prediction_sets[(cell, int(seed), "test")]
            for task in tasks:
                selected = [row for row in rows if row["task_head"] == task]
                if not selected:
                    raise ValueError(f"Missing B1 task {task} for {cell}, seed={seed}")
                item = metric_row(
                    cell, "test", selected, aggregation="single_seed_task", seed=int(seed)
                )
                item["task_head"] = task
                output.append(item)
    return output


def build_b1_task_ensemble_metrics(
    ensembles: dict[str, list[dict[str, Any]]],
    *,
    tasks: Iterable[str],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    output = []
    for cell in B1_CELLS:
        for task in tasks:
            selected = [row for row in ensembles[cell] if row["task_head"] == task]
            if not selected:
                raise ValueError(f"Missing ensemble B1 task {task} for {cell}")
            item = metric_row(
                cell,
                "test",
                selected,
                aggregation="prediction_ensemble_task",
                seeds=seeds,
            )
            item["task_head"] = task
            output.append(item)
    return output


def build_b1_heatmap_data(
    task_seed_rows: list[dict[str, Any]],
    task_ensemble_rows: list[dict[str, Any]],
    *,
    seeds: Iterable[int],
    tasks: Iterable[str],
) -> list[dict[str, Any]]:
    seed_lookup = {
        (row["cell"], int(row["seed"]), row["task_head"]): row for row in task_seed_rows
    }
    ensemble_lookup = {(row["cell"], row["task_head"]): row for row in task_ensemble_rows}
    output = []
    for task in tasks:
        for comparator, full, label in B1_CONTRASTS:
            per_seed = [
                float(seed_lookup[(comparator, int(seed), task)]["common_molkg_mae"])
                - float(seed_lookup[(full, int(seed), task)]["common_molkg_mae"])
                for seed in seeds
            ]
            comparator_ensemble = ensemble_lookup[(comparator, task)]
            full_ensemble = ensemble_lookup[(full, task)]
            if comparator_ensemble["n"] != full_ensemble["n"]:
                raise ValueError(f"B1 task n mismatch for {label}, task={task}")
            output.append(
                {
                    "task_head": task,
                    "contrast": label,
                    "comparator": comparator,
                    "full_model": full,
                    "n_test": full_ensemble["n"],
                    "delta_mae_ensemble": float(comparator_ensemble["common_molkg_mae"])
                    - float(full_ensemble["common_molkg_mae"]),
                    "delta_mae_seed_mean": statistics.fmean(per_seed),
                    "delta_mae_seed_sd": statistics.stdev(per_seed),
                    "seeds_improved": sum(value > 0 for value in per_seed),
                    "seed_count": len(per_seed),
                    "sign_definition": "positive=comparator_MAE−M11U_MAE; transfer beneficial",
                }
            )
    return output


def build_b2_comparison(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    ensembles: dict[str, list[dict[str, Any]]],
    *,
    seeds: Iterable[int],
    disclosures: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        train = prediction_sets[("M11U", int(seed), "target_train")]
        test = prediction_sets[("M11U", int(seed), "test")]
        assert_train_test_disjoint(train, test)
        baseline = task_mean_predictions(train, test)
        baseline_by_seed[int(seed)] = baseline
        output.append(
            {
                **metric_row(
                    "MeanBaseline", "test", baseline, aggregation="single_seed", seed=int(seed)
                ),
                "baseline_fit_part": "target_train_only",
                **disclosures["MeanBaseline"],
            }
        )
        for cell in B2_TRAINED_CELLS:
            output.append(
                {
                    **metric_row(
                        cell,
                        "test",
                        prediction_sets[(cell, int(seed), "test")],
                        aggregation="single_seed",
                        seed=int(seed),
                    ),
                    "baseline_fit_part": "not_applicable",
                    **disclosures[cell],
                }
            )
    baseline_ensemble = average_prediction_rows(
        [baseline_by_seed[int(seed)] for seed in seeds], label="B2 mean baseline"
    )
    output.append(
        {
            **metric_row(
                "MeanBaseline",
                "test",
                baseline_ensemble,
                aggregation="prediction_ensemble",
                seeds=seeds,
            ),
            "baseline_fit_part": "target_train_only",
            **disclosures["MeanBaseline"],
        }
    )
    for cell in B2_TRAINED_CELLS:
        output.append(
            {
                **metric_row(
                    cell,
                    "test",
                    ensembles[cell],
                    aggregation="prediction_ensemble",
                    seeds=seeds,
                ),
                "baseline_fit_part": "not_applicable",
                **disclosures[cell],
            }
        )
    single_seed = [row for row in output if row["aggregation"] == "single_seed"]
    return output, summarize_mean_sd(single_seed)


def assert_train_test_disjoint(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> None:
    overlap = {row["aggregate_id"] for row in train} & {row["aggregate_id"] for row in test}
    if overlap:
        raise ValueError(f"Train/test aggregate overlap detected: {sorted(overlap)[:5]}")


def task_mean_predictions(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_task_molar: dict[str, list[float]] = defaultdict(list)
    by_task_mass: dict[str, list[float]] = defaultdict(list)
    for row in train:
        by_task_molar[row["task_head"]].append(float(row["y_molkg"]))
        by_task_mass[row["task_head"]].append(float(row["y_mgkg"]))
    means_molar = {task: statistics.fmean(values) for task, values in by_task_molar.items()}
    means_mass = {task: statistics.fmean(values) for task, values in by_task_mass.items()}
    missing = sorted({row["task_head"] for row in test} - set(means_molar))
    if missing:
        raise ValueError(f"Mean baseline lacks train-only task means for {missing}")
    output = []
    for row in test:
        item = dict(row)
        item["pred_molkg"] = means_molar[row["task_head"]]
        item["pred_mgkg"] = means_mass[row["task_head"]]
        output.append(item)
    return output


def build_b3_analysis(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    ensembles: dict[str, list[dict[str, Any]]],
    *,
    seeds: Iterable[int],
    tasks: Iterable[str],
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    mw_predictions: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        for cell in ("ScaleMass", "M11_v140"):
            metric_rows.append(
                metric_row(
                    cell,
                    "test",
                    prediction_sets[(cell, int(seed), "test")],
                    aggregation="single_seed",
                    seed=int(seed),
                )
            )
        train = prediction_sets[("M11_v140", int(seed), "target_train")]
        test = prediction_sets[("M11_v140", int(seed), "test")]
        assert_train_test_disjoint(train, test)
        molar_model = fit_task_mw_baseline(train, target="y_molkg")
        mass_model = fit_task_mw_baseline(train, target="y_mgkg")
        predicted = apply_task_mw_baseline(test, molar_model=molar_model, mass_model=mass_model)
        mw_predictions[int(seed)] = predicted
        metric_rows.append(
            metric_row(
                "MWOnly", "test", predicted, aggregation="single_seed", seed=int(seed)
            )
        )
    for cell in ("ScaleMass", "M11_v140"):
        metric_rows.append(
            metric_row(
                cell,
                "test",
                ensembles[cell],
                aggregation="prediction_ensemble",
                seeds=seeds,
            )
        )
    mw_ensemble = average_prediction_rows(
        [mw_predictions[int(seed)] for seed in seeds], label="B3 MW-only"
    )
    metric_rows.append(
        metric_row(
            "MWOnly",
            "test",
            mw_ensemble,
            aggregation="prediction_ensemble",
            seeds=seeds,
        )
    )

    row_error: list[dict[str, Any]] = []
    for seed in seeds:
        row_error.extend(
            build_delta_ae_rows(
                prediction_sets[("ScaleMass", int(seed), "test")],
                prediction_sets[("M11_v140", int(seed), "test")],
                aggregation="single_seed",
                seed=int(seed),
            )
        )
    ensemble_delta = build_delta_ae_rows(
        ensembles["ScaleMass"],
        ensembles["M11_v140"],
        aggregation="prediction_ensemble",
        seed=None,
    )
    row_error.extend(ensemble_delta)
    relation = []
    for seed in seeds:
        selected = [row for row in row_error if row["seed"] == int(seed)]
        relation.append(delta_ae_relation(selected, aggregation="single_seed", seed=int(seed)))
    relation.append(delta_ae_relation(ensemble_delta, aggregation="prediction_ensemble", seed=None))

    mw_bins = build_mw_bin_summary(ensemble_delta)
    task_span = build_task_mw_span_summary(ensemble_delta, tasks=tasks)
    span_r = pearson(
        [float(row["log10_mw_span"]) for row in task_span],
        [float(row["delta_mae_mass_minus_molar"]) for row in task_span],
    )
    if span_r is None:
        raise ValueError("Insufficient task-level MW span variation for B3 correlation")
    relation.append(
        {
            "analysis_level": "task",
            "aggregation": "prediction_ensemble",
            "seed": None,
            "n": len(task_span),
            "pearson_delta_ae_log10_mw": None,
            "linear_slope": None,
            "linear_intercept": None,
            "linear_r2": None,
            "pearson_task_delta_mae_log10_mw_span": span_r,
        }
    )
    return {
        "metrics": metric_rows,
        "row_error": row_error,
        "mw_bins": mw_bins,
        "task_span": task_span,
        "relation": relation,
        "task_span_pearson": span_r,
    }


def fit_task_mw_baseline(
    rows: list[dict[str, Any]], *, target: str
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_head"]].append(row)
    sparse = sorted(task for task, values in by_task.items() if len(values) < 2)
    if sparse:
        raise ValueError(f"MW-only baseline has fewer than two training rows for tasks: {sparse}")
    x_means = {
        task: statistics.fmean(float(row["log10_mw"]) for row in values)
        for task, values in by_task.items()
    }
    y_means = {
        task: statistics.fmean(float(row[target]) for row in values)
        for task, values in by_task.items()
    }
    denominator = sum(
        (float(row["log10_mw"]) - x_means[task]) ** 2
        for task, values in by_task.items()
        for row in values
    )
    if denominator <= 0:
        raise ValueError("MW-only baseline lacks within-task logMW variation")
    numerator = sum(
        (float(row["log10_mw"]) - x_means[task])
        * (float(row[target]) - y_means[task])
        for task, values in by_task.items()
        for row in values
    )
    slope = numerator / denominator
    intercepts = {task: y_means[task] - slope * x_means[task] for task in by_task}
    return {"target": target, "slope": slope, "intercepts": intercepts}


def apply_task_mw_baseline(
    rows: list[dict[str, Any]],
    *,
    molar_model: dict[str, Any],
    mass_model: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks = {row["task_head"] for row in rows}
    missing_molar = sorted(tasks - set(molar_model["intercepts"]))
    missing_mass = sorted(tasks - set(mass_model["intercepts"]))
    if missing_molar or missing_mass:
        raise ValueError(
            f"MW-only baseline missing evaluation tasks: molar={missing_molar}, mass={missing_mass}"
        )
    output = []
    for row in rows:
        item = dict(row)
        task = row["task_head"]
        x = float(row["log10_mw"])
        item["pred_molkg"] = float(molar_model["intercepts"][task]) + float(
            molar_model["slope"]
        ) * x
        item["pred_mgkg"] = float(mass_model["intercepts"][task]) + float(
            mass_model["slope"]
        ) * x
        output.append(item)
    return output


def average_prediction_rows(
    sets: list[list[dict[str, Any]]], *, label: str
) -> list[dict[str, Any]]:
    if not sets:
        raise ValueError(f"No prediction sets for {label}")
    indexed = [unique_by_aggregate(rows) for rows in sets]
    identities = set(indexed[0])
    if any(set(rows) != identities for rows in indexed[1:]):
        raise ValueError(f"Prediction identities differ for {label}")
    output = []
    for aggregate_id in sorted(identities):
        source = [rows[aggregate_id] for rows in indexed]
        item = dict(source[0])
        item["pred_molkg"] = statistics.fmean(float(row["pred_molkg"]) for row in source)
        item["pred_mgkg"] = statistics.fmean(float(row["pred_mgkg"]) for row in source)
        output.append(item)
    return output


def build_delta_ae_rows(
    mass_rows: list[dict[str, Any]],
    molar_rows: list[dict[str, Any]],
    *,
    aggregation: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    mass = unique_by_aggregate(mass_rows)
    molar = unique_by_aggregate(molar_rows)
    if set(mass) != set(molar):
        raise ValueError("Scale-Mass and Scale-Mol identities differ for delta-AE analysis")
    output = []
    for aggregate_id in sorted(mass):
        left, right = mass[aggregate_id], molar[aggregate_id]
        if abs(float(left["y_mgkg"]) - float(right["y_mgkg"])) > 1e-10:
            raise ValueError(f"Scale truth mismatch for aggregate_id={aggregate_id}")
        ae_mass = abs(float(left["pred_mgkg"]) - float(left["y_mgkg"]))
        ae_molar = abs(float(right["pred_mgkg"]) - float(right["y_mgkg"]))
        output.append(
            {
                "aggregation": aggregation,
                "seed": seed,
                "aggregate_id": aggregate_id,
                "task_head": left["task_head"],
                "molecular_weight_g_mol": left["molecular_weight"],
                "log10_mw": left["log10_mw"],
                "ae_scale_mass_common_mgkg": ae_mass,
                "ae_scale_mol_common_mgkg": ae_molar,
                "delta_ae_mass_minus_molar": ae_mass - ae_molar,
                "sign_definition": "positive=Scale-Mol lower absolute error",
            }
        )
    return output


def delta_ae_relation(
    rows: list[dict[str, Any]], *, aggregation: str, seed: int | None
) -> dict[str, Any]:
    if len(rows) < 3:
        raise ValueError("B3 delta-AE/logMW relation requires at least three rows")
    x = [float(row["log10_mw"]) for row in rows]
    y = [float(row["delta_ae_mass_minus_molar"]) for row in rows]
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("B3 delta-AE/logMW relation lacks logMW variation")
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    r = pearson(x, y)
    if r is None:
        raise ValueError("B3 delta-AE/logMW correlation is undefined")
    return {
        "analysis_level": "row",
        "aggregation": aggregation,
        "seed": seed,
        "n": len(rows),
        "pearson_delta_ae_log10_mw": r,
        "linear_slope": slope,
        "linear_intercept": intercept,
        "linear_r2": r * r,
        "pearson_task_delta_mae_log10_mw_span": None,
    }


def build_mw_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = sorted(float(row["log10_mw"]) for row in rows)
    if len(values) < 4:
        raise ValueError("B3 MW quartiles require at least four rows")
    cuts = [quantile(values, fraction) for fraction in (0.25, 0.5, 0.75)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        x = float(row["log10_mw"])
        index = sum(x > cut for cut in cuts) + 1
        grouped[f"Q{index}"] .append(row)
    if set(grouped) != {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError(f"B3 MW quartiles are not all populated: {sorted(grouped)}")
    output = []
    for name in ("Q1", "Q2", "Q3", "Q4"):
        selected = grouped[name]
        output.append(
            {
                "mw_quartile": name,
                "n": len(selected),
                "log10_mw_min": min(float(row["log10_mw"]) for row in selected),
                "log10_mw_max": max(float(row["log10_mw"]) for row in selected),
                "mae_scale_mass_common_mgkg": statistics.fmean(
                    float(row["ae_scale_mass_common_mgkg"]) for row in selected
                ),
                "mae_scale_mol_common_mgkg": statistics.fmean(
                    float(row["ae_scale_mol_common_mgkg"]) for row in selected
                ),
                "delta_mae_mass_minus_molar": statistics.fmean(
                    float(row["delta_ae_mass_minus_molar"]) for row in selected
                ),
            }
        )
    return output


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_task_mw_span_summary(
    rows: list[dict[str, Any]], *, tasks: Iterable[str]
) -> list[dict[str, Any]]:
    output = []
    for task in tasks:
        selected = [row for row in rows if row["task_head"] == task]
        if len(selected) < 2:
            raise ValueError(f"B3 task MW-span analysis lacks support for {task}")
        log_mw = [float(row["log10_mw"]) for row in selected]
        output.append(
            {
                "task_head": task,
                "n": len(selected),
                "log10_mw_min": min(log_mw),
                "log10_mw_max": max(log_mw),
                "log10_mw_span": max(log_mw) - min(log_mw),
                "mae_scale_mass_common_mgkg": statistics.fmean(
                    float(row["ae_scale_mass_common_mgkg"]) for row in selected
                ),
                "mae_scale_mol_common_mgkg": statistics.fmean(
                    float(row["ae_scale_mol_common_mgkg"]) for row in selected
                ),
                "delta_mae_mass_minus_molar": statistics.fmean(
                    float(row["delta_ae_mass_minus_molar"]) for row in selected
                ),
            }
        )
    return output


def build_replay_contrast(
    seed_metrics: list[dict[str, Any]],
    ensemble_metrics: list[dict[str, Any]],
    *,
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    seed_lookup = {
        (row["cell"], row["evaluation_part"], row["seed"]): row for row in seed_metrics
    }
    output = []
    for seed in seeds:
        old = seed_lookup[("M11_v140", "test", int(seed))]
        new = seed_lookup[("M11U", "test", int(seed))]
        output.append(replay_delta_row(old, new, aggregation="single_seed", seed=int(seed)))
    ensemble_lookup = {row["cell"]: row for row in ensemble_metrics}
    output.append(
        replay_delta_row(
            ensemble_lookup["M11_v140"],
            ensemble_lookup["M11U"],
            aggregation="prediction_ensemble",
            seed=None,
        )
    )
    return output


def replay_delta_row(
    old: dict[str, Any], new: dict[str, Any], *, aggregation: str, seed: int | None
) -> dict[str, Any]:
    return {
        "aggregation": aggregation,
        "seed": seed,
        "n": old["n"],
        "delta_r2_new_minus_v140_molkg": float(new["common_molkg_r2"])
        - float(old["common_molkg_r2"]),
        "delta_rmse_new_minus_v140_molkg": float(new["common_molkg_rmse"])
        - float(old["common_molkg_rmse"]),
        "delta_mae_new_minus_v140_molkg": float(new["common_molkg_mae"])
        - float(old["common_molkg_mae"]),
        "aggregate_id_sha256": old["aggregate_id_sha256"],
    }


def build_alignment_audit(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    output = []
    for part in PARTS:
        reference = prediction_sets[("M11U", int(tuple(seeds)[0]), part)]
        output.append(
            {
                "part": part,
                "n": len(reference),
                "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in reference),
                "result_ids_sha256": stable_hash(
                    result_id for row in reference for result_id in row["result_ids"]
                ),
            }
        )
    return output


def plot_b1_heatmap(
    rows: list[dict[str, Any]], *, tasks: Iterable[str], output_base: Path
) -> None:
    try:
        import matplotlib as mpl
        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("matplotlib is required for PNG/SVG heatmap export") from exc

    task_list = list(tasks)
    contrast_list = [label for _, _, label in B1_CONTRASTS]
    lookup = {(row["task_head"], row["contrast"]): row for row in rows}
    matrix = [
        [float(lookup[(task, contrast)]["delta_mae_ensemble"]) for contrast in contrast_list]
        for task in task_list
    ]
    limit = max(abs(value) for row in matrix for value in row)
    if limit <= 0:
        limit = 1e-6
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "Droid Sans Fallback",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 8.4), constrained_layout=True)
    image = ax.imshow(matrix, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
    short_labels = [
        "土壤pTox适配",
        "水相预训练",
        "完整迁移",
        "解冻主干",
    ]
    ax.set_xticks(range(len(short_labels)), labels=short_labels)
    ax.set_yticks(range(len(task_list)), labels=task_list)
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("因果比较（正值表示M11U降低MAE）")
    ax.set_ylabel("土壤毒性任务")
    ax.set_title("18个任务的迁移增益（四种子逐行预测集成）", loc="left", weight="bold")
    threshold = limit * 0.55
    for row_index, task in enumerate(task_list):
        for column_index, contrast in enumerate(contrast_list):
            value = float(lookup[(task, contrast)]["delta_mae_ensemble"])
            ax.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) >= threshold else "#222222",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("ΔMAE = MAE比较模型 − MAE(M11U)")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
