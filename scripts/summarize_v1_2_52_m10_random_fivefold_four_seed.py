from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsar_tl.training.baseline import stage_sample_record_id
from scripts.build_v1_2_44_second_layer_splits import canonical_sha256
from scripts.summarize_v1_2_44_second_layer_matrix import (
    metric_row,
    stream_final_predictions,
)


SEEDS = (42, 2042, 3407, 8417)
FOLDS = (1, 2, 3, 4, 5)
SPLIT_PREFIX = "M_v1_2_51_M10_随机五折_折"
V51_ROOT = Path("outputs/experiments/v1_2_51_m10_random_fivefold")
V52_ROOT = Path("outputs/experiments/v1_2_52_m10_random_fivefold_remaining_seeds")
SPLIT_AUDIT = Path("outputs/audits/v1_2_51_m10_random_fivefold/split_contract.json")
OUTPUT_DIR = V52_ROOT / "summary_four_seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the v1.2.51 seed3407 and v1.2.52 remaining-seed M10 "
            "random-fivefold predictions into four-seed OOF summaries."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--v51-root", type=Path, default=V51_ROOT)
    parser.add_argument("--v52-root", type=Path, default=V52_ROOT)
    parser.add_argument("--split-audit", type=Path, default=SPLIT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_record_id(row: dict[str, Any]) -> str:
    return stage_sample_record_id(
        row["aggregate_id"],
        "soil",
        "neg_log10_mol_kg",
        "solid_neglog_mol_kg",
    )


def run_dir_for(v51_root: Path, v52_root: Path, *, seed: int, fold: int) -> Path:
    split_name = f"{SPLIT_PREFIX}{fold}"
    if seed == 3407:
        return (
            v51_root
            / f"v1.2.51_M10_Full_随机五折_折{fold}_种子{seed}"
            / "deep"
            / "full"
            / split_name
        )
    return (
        v52_root
        / f"v1.2.52_M10_Full_随机五折_折{fold}_种子{seed}"
        / "deep"
        / "full"
        / split_name
    )


def validate_manifest(path: Path, *, seed: int, fold: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "seed": seed,
        "split_name": f"{SPLIT_PREFIX}{fold}",
        "ablation": "full",
        "epochs": 30,
        "finetune_epochs": 0,
        "finetune_mgkg_epochs": 30,
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    weighting = manifest.get("source_weighting")
    if not isinstance(weighting, dict) or (
        str(weighting.get("method", "")).lower() != "none"
        or bool(weighting.get("applied", False))
    ):
        mismatches["source_weighting"] = {
            "observed": weighting,
            "expected": {"method": "none", "applied": False},
        }
    if mismatches:
        raise ValueError(f"Manifest mismatch seed={seed}, fold={fold}: {mismatches}")
    return {
        "seed": seed,
        "fold": fold,
        "split_name": expected["split_name"],
        "run_version": str(manifest.get("run_version", "")),
        "stage1_epochs": 30,
        "stage2_epochs": 0,
        "stage3_epochs": 30,
        "ablation": "full",
        "source_weighting": "none",
        "manifest": str(path),
    }


def assert_same_identity(reference: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    if (
        reference["task_head"] != actual["task_head"]
        or tuple(reference["result_ids"]) != tuple(actual["result_ids"])
    ):
        raise ValueError(f"Task/result identity mismatch for {label}")
    for field in ("y_molkg", "y_mgkg", "molecular_weight"):
        if not math.isclose(
            float(reference[field]),
            float(actual[field]),
            rel_tol=1e-12,
            abs_tol=1e-10,
        ):
            raise ValueError(f"Truth mismatch for {label}, field={field}")


def ensemble_rows(
    rows_by_seed: dict[int, list[dict[str, Any]]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    indexed = {
        seed: {target_record_id(row): row for row in rows}
        for seed, rows in rows_by_seed.items()
    }
    identities = set(next(iter(indexed.values())))
    for seed, rows in indexed.items():
        if set(rows) != identities:
            raise ValueError(f"Seed identity mismatch for {label}, seed={seed}")
    output: list[dict[str, Any]] = []
    for record_id in sorted(identities):
        source = [indexed[seed][record_id] for seed in SEEDS]
        reference = dict(source[0])
        for actual in source[1:]:
            assert_same_identity(reference, actual, f"{label}, record={record_id}")
        reference["pred_molkg"] = statistics.fmean(
            float(row["pred_molkg"]) for row in source
        )
        reference["pred_mgkg"] = statistics.fmean(
            float(row["pred_mgkg"]) for row in source
        )
        output.append(reference)
    return output


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    v51_root = resolve(repo_root, args.v51_root)
    v52_root = resolve(repo_root, args.v52_root)
    audit_path = resolve(repo_root, args.split_audit)
    output_dir = resolve(repo_root, args.output_dir)
    split_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        split_audit.get("status") != "persisted_and_validated"
        or bool(split_audit.get("is_scaffold_split", True))
    ):
        raise ValueError("Expected the persisted non-scaffold v1.2.51 split contract")
    expected_target_rows = int(split_audit["target_pool_rows"])
    expected_target_hash = str(split_audit["target_pool_record_id_sha256"])
    expected_fold_counts = {
        int(row["fold"]): int(row["target_test_rows"])
        for row in split_audit["folds"]
    }
    expected_fold_hashes = {
        int(row["fold"]): str(row["target_test_record_id_sha256"])
        for row in split_audit["folds"]
    }

    predictions: dict[tuple[int, int], list[dict[str, Any]]] = {}
    manifest_audit: list[dict[str, Any]] = []
    for seed in SEEDS:
        seen_for_seed: set[str] = set()
        for fold in FOLDS:
            run_dir = run_dir_for(v51_root, v52_root, seed=seed, fold=fold)
            pred_path = run_dir / "predictions.csv"
            manifest_path = run_dir / "manifest.json"
            model_path = run_dir / "best_model.pt"
            for path in (pred_path, manifest_path, model_path):
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"Missing completed artifact: {path}")
            manifest_audit.append(validate_manifest(manifest_path, seed=seed, fold=fold))
            rows = [
                row
                for row in stream_final_predictions(pred_path, scale="molar")
                if row["evaluation_part"] == "test"
            ]
            ids = [target_record_id(row) for row in rows]
            if len(rows) != expected_fold_counts[fold]:
                raise ValueError(
                    f"Count mismatch seed={seed}, fold={fold}: "
                    f"{len(rows)} != {expected_fold_counts[fold]}"
                )
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate identities seed={seed}, fold={fold}")
            observed_fold_hash = canonical_sha256(sorted(ids))
            if observed_fold_hash != expected_fold_hashes[fold]:
                raise ValueError(f"Fold identity hash mismatch seed={seed}, fold={fold}")
            if seen_for_seed & set(ids):
                raise ValueError(f"OOF overlap seed={seed}, fold={fold}")
            seen_for_seed.update(ids)
            for row in rows:
                row["outer_fold"] = fold
            predictions[(seed, fold)] = rows
        if len(seen_for_seed) != expected_target_rows:
            raise ValueError(f"OOF row count mismatch for seed={seed}")
        if canonical_sha256(sorted(seen_for_seed)) != expected_target_hash:
            raise ValueError(f"OOF target hash mismatch for seed={seed}")

    seed_oof_metrics: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = [row for fold in FOLDS for row in predictions[(seed, fold)]]
        seed_oof_metrics.append(
            metric_row(
                f"M10-R5F-seed{seed}",
                "test",
                rows,
                aggregation="single_seed_fivefold_oof",
                seed=seed,
            )
        )

    fold_ensemble_metrics: list[dict[str, Any]] = []
    fold_ensembles: dict[int, list[dict[str, Any]]] = {}
    for fold in FOLDS:
        ensemble = ensemble_rows(
            {seed: predictions[(seed, fold)] for seed in SEEDS},
            label=f"fold={fold}",
        )
        fold_ensembles[fold] = ensemble
        metric = metric_row(
            f"M10-R5F-four-seed-fold{fold}",
            "test",
            ensemble,
            aggregation="four_seed_prediction_ensemble_fold",
            seeds=SEEDS,
        )
        metric["outer_fold"] = fold
        fold_ensemble_metrics.append(metric)

    oof_ensemble = [row for fold in FOLDS for row in fold_ensembles[fold]]
    if len(oof_ensemble) != expected_target_rows:
        raise ValueError("Four-seed OOF ensemble count mismatch")
    four_seed_oof = metric_row(
        "M10-R5F-four-seed-OOF",
        "test",
        oof_ensemble,
        aggregation="four_seed_prediction_ensemble_then_fivefold_oof",
        seeds=SEEDS,
    )
    four_seed_oof["outer_fold_count"] = len(FOLDS)
    four_seed_oof["split_policy"] = (
        "task-stratified row-random fivefold; no scaffold grouping"
    )

    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in oof_ensemble:
        task_groups[str(row["task_head"])].append(row)
    task_metrics: list[dict[str, Any]] = []
    for task, rows in sorted(task_groups.items()):
        metric = metric_row(
            "M10-R5F-four-seed-OOF",
            "test",
            rows,
            aggregation="four_seed_prediction_ensemble_oof_task",
            seeds=SEEDS,
        )
        metric["task_head"] = task
        task_metrics.append(metric)

    seed_summary: dict[str, Any] = {
        "cell": "M10-R5F-single-seed-OOF",
        "seed_count": len(SEEDS),
        "seeds": ",".join(str(seed) for seed in SEEDS),
        "n_per_seed": expected_target_rows,
    }
    for field in (
        "common_molkg_r2",
        "common_molkg_rmse",
        "common_molkg_mae",
        "within_r2_molkg",
    ):
        values = [float(row[field]) for row in seed_oof_metrics]
        seed_summary[f"{field}_mean"] = statistics.fmean(values)
        seed_summary[f"{field}_sd"] = statistics.stdev(values)
        seed_summary[f"{field}_min"] = min(values)
        seed_summary[f"{field}_max"] = max(values)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = output_dir / "m10_random_fivefold_single_seed_oof_metrics.csv"
    seed_summary_path = output_dir / "m10_random_fivefold_single_seed_mean_sd.csv"
    fold_path = output_dir / "m10_random_fivefold_four_seed_fold_metrics.csv"
    ensemble_path = output_dir / "m10_random_fivefold_four_seed_oof_metrics.csv"
    task_path = output_dir / "m10_random_fivefold_four_seed_task_metrics.csv"
    manifest_path = output_dir / "m10_random_fivefold_four_seed_manifest_audit.csv"
    write_csv(seed_path, seed_oof_metrics)
    write_csv(seed_summary_path, [seed_summary])
    write_csv(fold_path, fold_ensemble_metrics)
    write_csv(ensemble_path, [four_seed_oof])
    write_csv(task_path, task_metrics)
    write_csv(manifest_path, manifest_audit)

    summary = {
        "schema": "v1_2_52_m10_random_fivefold_four_seed_summary_v1",
        "status": "complete",
        "evaluation_boundary": "task-stratified row-random fivefold OOF",
        "is_scaffold_split": False,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "training_cells": len(SEEDS) * len(FOLDS),
        "oof_rows": expected_target_rows,
        "oof_exact_once_per_seed_verified": True,
        "four_seed_aggregation": (
            "average predictions across four seeds for each row within its outer fold, "
            "then concatenate the five exact-once OOF folds"
        ),
        "four_seed_oof_metrics": four_seed_oof,
        "single_seed_oof_mean_sd": seed_summary,
        "outputs": {
            "single_seed_oof_metrics": str(seed_path.relative_to(repo_root)),
            "single_seed_mean_sd": str(seed_summary_path.relative_to(repo_root)),
            "four_seed_fold_metrics": str(fold_path.relative_to(repo_root)),
            "four_seed_oof_metrics": str(ensemble_path.relative_to(repo_root)),
            "four_seed_task_metrics": str(task_path.relative_to(repo_root)),
            "manifest_audit": str(manifest_path.relative_to(repo_root)),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
