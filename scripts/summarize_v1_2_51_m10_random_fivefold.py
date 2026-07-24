from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
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


FOLDS = (1, 2, 3, 4, 5)
SEED = 3407
SPLIT_PREFIX = "M_v1_2_51_M10_随机五折_折"
RUN_ROOT = Path("outputs/experiments/v1_2_51_m10_random_fivefold")
SPLIT_AUDIT = Path("outputs/audits/v1_2_51_m10_random_fivefold/split_contract.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the fixed-seed M10 row-random fivefold OOF predictions."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--split-audit", type=Path, default=SPLIT_AUDIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "summary",
    )
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


def audit_manifest(path: Path, *, fold: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_split = f"{SPLIT_PREFIX}{fold}"
    checks = {
        "seed": (int(manifest.get("seed", -1)), SEED),
        "split_name": (str(manifest.get("split_name", "")), expected_split),
        "ablation": (str(manifest.get("ablation", "")), "full"),
        "stage1_epochs": (int(manifest.get("epochs", -1)), 30),
        "stage2_epochs": (int(manifest.get("finetune_epochs", -1)), 0),
        "stage3_epochs": (int(manifest.get("finetune_mgkg_epochs", -1)), 30),
    }
    mismatches = {
        key: {"observed": observed, "expected": expected}
        for key, (observed, expected) in checks.items()
        if observed != expected
    }
    weighting = manifest.get("source_weighting")
    if not isinstance(weighting, dict):
        mismatches["source_weighting"] = {"observed": weighting, "expected": "audit dict"}
    elif (
        str(weighting.get("method", "")).lower() != "none"
        or bool(weighting.get("applied", False))
    ):
        mismatches["source_weighting"] = {
            "observed": weighting,
            "expected": {"method": "none", "applied": False},
        }
    if mismatches:
        raise ValueError(f"Manifest contract mismatch for fold {fold}: {mismatches}")
    return {
        "fold": fold,
        "seed": SEED,
        "split_name": expected_split,
        "ablation": "full",
        "stage1_epochs": 30,
        "stage2_epochs": 0,
        "stage3_epochs": 30,
        "source_weighting": "none",
        "manifest": str(path),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_root = resolve(repo_root, args.run_root)
    audit_path = resolve(repo_root, args.split_audit)
    output_dir = resolve(repo_root, args.output_dir)
    split_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if split_audit.get("status") != "persisted_and_validated":
        raise ValueError("Fivefold split contract is not persisted_and_validated")
    if bool(split_audit.get("is_scaffold_split", True)):
        raise ValueError("v1.2.51 summary requires a non-scaffold row-random split")

    expected_fold_counts = {
        int(row["fold"]): int(row["target_test_rows"])
        for row in split_audit["folds"]
    }
    expected_fold_hashes = {
        int(row["fold"]): str(row["target_test_record_id_sha256"])
        for row in split_audit["folds"]
    }

    fold_metrics: list[dict[str, Any]] = []
    manifest_audit: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {}
    oof_rows: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for fold in FOLDS:
        split_name = f"{SPLIT_PREFIX}{fold}"
        run_dir = (
            run_root
            / f"v1.2.51_M10_Full_随机五折_折{fold}_种子{SEED}"
            / "deep"
            / "full"
            / split_name
        )
        pred_path = run_dir / "predictions.csv"
        manifest_path = run_dir / "manifest.json"
        model_path = run_dir / "best_model.pt"
        for path in (pred_path, manifest_path, model_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"Missing completed fold artifact: {path}")
        manifest_audit.append(audit_manifest(manifest_path, fold=fold))
        rows = [
            row
            for row in stream_final_predictions(pred_path, scale="molar")
            if row["evaluation_part"] == "test"
        ]
        record_ids = [target_record_id(row) for row in rows]
        if len(rows) != expected_fold_counts[fold]:
            raise ValueError(
                f"Fold {fold} test count mismatch: {len(rows)} != {expected_fold_counts[fold]}"
            )
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"Fold {fold} contains duplicate target identities")
        observed_hash = canonical_sha256(sorted(record_ids))
        if observed_hash != expected_fold_hashes[fold]:
            raise ValueError(
                f"Fold {fold} test identity hash mismatch: "
                f"{observed_hash} != {expected_fold_hashes[fold]}"
            )
        overlap = seen_record_ids & set(record_ids)
        if overlap:
            raise ValueError(f"OOF test overlap detected for fold {fold}: {len(overlap)} rows")
        seen_record_ids.update(record_ids)
        for row in rows:
            row["outer_fold"] = fold
            task_rows.setdefault(str(row["task_head"]), []).append(row)
        oof_rows.extend(rows)
        metric = metric_row(
            f"M10-R5F-fold{fold}",
            "test",
            rows,
            aggregation="single_outer_fold",
            seed=SEED,
        )
        metric["outer_fold"] = fold
        fold_metrics.append(metric)

    expected_target_rows = int(split_audit["target_pool_rows"])
    expected_target_hash = str(split_audit["target_pool_record_id_sha256"])
    if len(oof_rows) != expected_target_rows:
        raise ValueError(f"OOF count mismatch: {len(oof_rows)} != {expected_target_rows}")
    observed_target_hash = canonical_sha256(sorted(seen_record_ids))
    if observed_target_hash != expected_target_hash:
        raise ValueError(
            f"OOF target identity mismatch: {observed_target_hash} != {expected_target_hash}"
        )

    pooled = metric_row(
        "M10-R5F-OOF",
        "test",
        oof_rows,
        aggregation="fivefold_oof_concatenation",
        seed=SEED,
    )
    pooled["outer_fold_count"] = len(FOLDS)
    pooled["split_policy"] = "task-stratified row-random fivefold; no scaffold grouping"

    task_metrics: list[dict[str, Any]] = []
    for task, rows in sorted(task_rows.items()):
        metric = metric_row(
            "M10-R5F-OOF",
            "test",
            rows,
            aggregation="fivefold_oof_task",
            seed=SEED,
        )
        metric["task_head"] = task
        task_metrics.append(metric)

    mean_sd: dict[str, Any] = {
        "cell": "M10-R5F",
        "fold_count": len(FOLDS),
        "seed": SEED,
    }
    for field in (
        "common_molkg_r2",
        "common_molkg_rmse",
        "common_molkg_mae",
        "within_r2_molkg",
    ):
        values = [float(row[field]) for row in fold_metrics]
        mean_sd[f"{field}_mean"] = statistics.fmean(values)
        mean_sd[f"{field}_sd"] = statistics.stdev(values)
        mean_sd[f"{field}_min"] = min(values)
        mean_sd[f"{field}_max"] = max(values)

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "m10_random_fivefold_fold_metrics.csv"
    pooled_path = output_dir / "m10_random_fivefold_oof_pooled_metrics.csv"
    task_path = output_dir / "m10_random_fivefold_oof_task_metrics.csv"
    mean_sd_path = output_dir / "m10_random_fivefold_fold_mean_sd.csv"
    manifest_path = output_dir / "m10_random_fivefold_manifest_audit.csv"
    write_csv(fold_path, fold_metrics)
    write_csv(pooled_path, [pooled])
    write_csv(task_path, task_metrics)
    write_csv(mean_sd_path, [mean_sd])
    write_csv(manifest_path, manifest_audit)

    summary = {
        "schema": "v1_2_51_m10_random_fivefold_summary_v1",
        "status": "complete",
        "evaluation_boundary": "task-stratified row-random fivefold OOF",
        "is_scaffold_split": False,
        "target": "neg_log10_mol_kg; final soil; 18 task heads",
        "seed": SEED,
        "fold_count": len(FOLDS),
        "oof_rows": len(oof_rows),
        "oof_exact_once_verified": True,
        "oof_target_record_id_sha256": observed_target_hash,
        "pooled_metrics": pooled,
        "fold_mean_sd": mean_sd,
        "outputs": {
            "fold_metrics": str(fold_path.relative_to(repo_root)),
            "pooled_metrics": str(pooled_path.relative_to(repo_root)),
            "task_metrics": str(task_path.relative_to(repo_root)),
            "fold_mean_sd": str(mean_sd_path.relative_to(repo_root)),
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
