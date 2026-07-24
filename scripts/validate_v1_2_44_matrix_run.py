from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v1_2_44_second_layer_splits import (
    SOURCE_TABLE,
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
)


PREDICTION_PARTS = ("finetune_mgkg", "finetune_mgkg_validation", "test")
CELL_SPECS: dict[str, dict[str, Any]] = {
    "M00": {"route": "M00", "ablation": "full", "epochs": (0, 0, 30), "min_total": 200, "min_train": 100, "min_eval": 30, "freeze": "none"},
    "M10": {"route": "M10", "ablation": "full", "epochs": (30, 0, 30), "min_total": 0, "min_train": 0, "min_eval": 0, "freeze": "none"},
    "M01": {"route": "M01", "ablation": "full", "epochs": (20, 0, 30), "min_total": 0, "min_train": 0, "min_eval": 0, "freeze": "none"},
    "M11F": {"route": "M11", "ablation": "full", "epochs": (30, 20, 30), "min_total": 200, "min_train": 100, "min_eval": 30, "freeze": "heads_only"},
    "M11U": {"route": "M11", "ablation": "full", "epochs": (30, 20, 30), "min_total": 200, "min_train": 100, "min_eval": 30, "freeze": "none"},
    "B2_CONTEXT": {"route": "M11", "ablation": "no_molecular_input", "epochs": (30, 20, 30), "min_total": 200, "min_train": 100, "min_eval": 30, "freeze": "none"},
    "B2_MOLECULE": {"route": "M11", "ablation": "no_context", "epochs": (30, 20, 30), "min_total": 200, "min_train": 100, "min_eval": 30, "freeze": "none"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed validator for one v1.2.44 matrix run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cell", required=True, choices=sorted(CELL_SPECS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--stage2-cache", type=Path)
    parser.add_argument("--paired-run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_run(
        args.run_dir,
        cell=args.cell,
        seed=args.seed,
        split_summary=args.split_summary,
        source_table=args.source_table,
        stage2_cache=args.stage2_cache,
        paired_run_dir=args.paired_run_dir,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def validate_run(
    run_dir: Path,
    *,
    cell: str,
    seed: int,
    split_summary: Path,
    source_table: str = SOURCE_TABLE,
    stage2_cache: Path | None = None,
    paired_run_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    spec = CELL_SPECS[cell]
    summary = load_json(split_summary)
    validate_split_summary(summary)
    route = summary["routes"][spec["route"]]
    manifest = load_json(run_dir / "manifest.json")
    for name in ("predictions.csv", "history.csv", "best_model.pt", "preprocessing.json"):
        require_nonempty(run_dir / name)

    expected_epochs = (
        tuple(1 if value > 0 else 0 for value in spec["epochs"])
        if smoke
        else spec["epochs"]
    )
    checks = {
        "manifest_seed": (manifest.get("seed"), int(seed)),
        "manifest_split": (manifest.get("split_name"), route["split_name"]),
        "source_table": (manifest.get("data_source", {}).get("source_table"), source_table),
        "head_routing": (manifest.get("head_routing"), "task_target"),
        "mixed_targets": (manifest.get("allow_mixed_target_dimensions"), True),
        "epochs": (manifest.get("epochs"), expected_epochs[0]),
        "finetune_epochs": (manifest.get("finetune_epochs"), expected_epochs[1]),
        "stage3_epochs": (manifest.get("finetune_mgkg_epochs"), expected_epochs[2]),
        "ablation": (manifest.get("ablation"), spec["ablation"]),
        "stage2_freeze": (manifest.get("finetune", {}).get("freeze"), "none"),
        "stage3_freeze": (manifest.get("finetune_mgkg", {}).get("freeze"), spec["freeze"]),
        "stage3_validation_source": (manifest.get("finetune_mgkg", {}).get("validation_source"), "valid"),
        "stage3_validation_rows": (
            manifest.get("finetune_mgkg", {}).get("validation_rows"),
            summary["stage3_boundary"]["validation_rows"],
        ),
        "task_filter_min_total": (manifest.get("task_filter", {}).get("min_total"), spec["min_total"]),
        "task_filter_min_train": (manifest.get("task_filter", {}).get("min_train"), spec["min_train"]),
        "task_filter_min_eval": (manifest.get("task_filter", {}).get("min_eval"), spec["min_eval"]),
        "prediction_parts": (
            sorted(manifest.get("prediction_output_split_parts", [])),
            sorted(PREDICTION_PARTS),
        ),
        "medium_adapter": (manifest.get("ablation_features", {}).get("use_medium_adapter"), False),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")
    validate_ablation_features(manifest, spec["ablation"])
    validate_stage_protocol(manifest, cell=cell, summary=summary, smoke=smoke)

    predictions = read_csv(run_dir / "predictions.csv")
    actual_parts = sorted({str(row.get("split_part", "")) for row in predictions})
    if actual_parts != sorted(PREDICTION_PARTS):
        raise ValueError(f"prediction split parts mismatch: {actual_parts}")
    part_audit: dict[str, Any] = {}
    for part in PREDICTION_PARTS:
        rows = [row for row in predictions if row.get("split_part") == part]
        records = [evaluation_identity_record(row, split_part=part) for row in rows]
        expected = summary["stage3_boundary"]["evaluation_identity"][part]
        observed = {
            "rows": len(records),
            "sha256": canonical_sha256(sorted(records, key=canonical_json)),
        }
        if observed != expected:
            raise ValueError(
                f"prediction identity mismatch for {part}: observed={observed}, expected={expected}"
            )
        part_audit[part] = observed

    if cell in {"M10", "M01"}:
        exception_key = (
            "m10_task_filter_exception" if cell == "M10" else "m01_task_filter_exception"
        )
        expected_seed = 42 if cell == "M10" else 17_073
        exception = summary.get(exception_key, {})
        if (
            exception.get("task_filter_min_total") != 0
            or exception.get("task_filter_min_train") != 0
            or exception.get("task_filter_min_eval") != 0
            or exception.get("validation_seed") != expected_seed
        ):
            raise ValueError(f"{cell}-only task-filter/validation exception is absent from split audit.")
        if manifest.get("validation_seed") != expected_seed:
            raise ValueError(f"{cell} Stage-1 validation_seed must be {expected_seed}.")
        if manifest.get("actual_train_rows") != exception.get("train_rows"):
            raise ValueError(f"{cell} actual Stage-1 training row count differs from audited boundary.")
        if manifest.get("validation_rows") != exception.get("validation_rows"):
            raise ValueError(f"{cell} Stage-1 validation row count differs from audited boundary.")

    if cell == "M01":
        exception = summary.get("m01_task_filter_exception", {})
        if exception.get("route") not in {None, "M01_only"}:
            raise ValueError("M01 audit route label is invalid.")

    validate_checkpoint_pair(
        manifest,
        cell=cell,
        stage2_cache=stage2_cache,
        paired_run_dir=paired_run_dir,
    )
    return {
        "status": "ok",
        "cell": cell,
        "seed": int(seed),
        "split_name": route["split_name"],
        "prediction_identity": part_audit,
    }


def validate_split_summary(summary: Mapping[str, Any]) -> None:
    claimed = str(summary.get("contract_sha256", ""))
    payload = dict(summary)
    payload.pop("contract_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("split summary contract hash is invalid.")
    if summary.get("matrix_version") != "v1.2.44" or summary.get("source_table") != SOURCE_TABLE:
        raise ValueError("split summary is not the pinned v1.2.44 source contract.")
    stage3 = summary.get("stage3_boundary", {})
    if stage3.get("train_rows") != 9_724 or stage3.get("validation_rows") != 2_433 or stage3.get("test_rows") != 3_042:
        raise ValueError(f"v1.2.40 effective Stage-3 boundary changed: {stage3}")
    task_counts = stage3.get("task_counts", [])
    if len(task_counts) != 18:
        raise ValueError(f"Expected exactly 18 effective Stage-3 tasks, found {len(task_counts)}.")
    hashes = {route.get("stage3_boundary_sha256") for route in summary.get("routes", {}).values()}
    if hashes != {stage3.get("stage3_boundary_sha256")}:
        raise ValueError("Route Stage-3 hashes are not identical.")


def validate_ablation_features(manifest: Mapping[str, Any], ablation: str) -> None:
    features = manifest.get("ablation_features", {})
    expected = {
        "full": (True, True, True, True, True),
        "no_molecular_input": (False, False, True, True, True),
        "no_context": (True, True, False, False, False),
    }[ablation]
    observed = (
        features.get("use_descriptors"),
        features.get("use_fingerprint"),
        features.get("use_context_numeric"),
        features.get("use_species_lifestage"),
        features.get("use_other_categorical_context"),
    )
    if observed != expected:
        raise ValueError(f"ablation feature contract mismatch: observed={observed}, expected={expected}")


def validate_stage_protocol(
    manifest: Mapping[str, Any], *, cell: str, summary: Mapping[str, Any], smoke: bool
) -> None:
    finetune = manifest.get("finetune", {})
    stage3 = manifest.get("finetune_mgkg", {})
    if float(stage3.get("learning_rate", -1)) != 0.0005:
        raise ValueError("Stage-3 head learning rate must be 5e-4.")
    if stage3.get("trunk_learning_rate") is not None:
        raise ValueError("v1.2.40 replay requires unified LR (manifest trunk_learning_rate=None).")
    if float(stage3.get("soil_ptox_replay_fraction_requested", -1)) != 0.0:
        raise ValueError("Stage-3 replay fraction must be zero.")
    if cell not in {"M10", "M01"} and float(finetune.get("learning_rate", -1)) != 0.0001:
        raise ValueError("Stage-2 learning rate must be 1e-4.")
    if cell == "M10" and manifest.get("finetune_rows") != 0:
        raise ValueError("M10 unexpectedly contains soil-pTox Stage-2 rows.")
    if cell == "M01":
        if manifest.get("finetune_rows") != 0:
            raise ValueError("M01 must train soil pTox from random initialization, not use finetune rows.")
        if float(manifest.get("learning_rate", -1)) != 0.0001:
            raise ValueError("M01 soil-pTox learning rate must match Stage 2 at 1e-4.")


def validate_checkpoint_pair(
    manifest: Mapping[str, Any],
    *,
    cell: str,
    stage2_cache: Path | None,
    paired_run_dir: Path | None,
) -> None:
    audit = manifest.get("finetune_mgkg", {}).get("stage3_init_checkpoint", {})
    if cell == "M11F":
        if stage2_cache is None or not stage2_cache.is_file():
            raise ValueError("M11F must export a non-empty same-seed Stage-2 checkpoint.")
        if not audit.get("exported") or audit.get("loaded") or audit.get("stage1_stage2_skipped"):
            raise ValueError(f"M11F checkpoint export audit is invalid: {audit}")
    elif cell == "M11U":
        if stage2_cache is None or not stage2_cache.is_file() or paired_run_dir is None:
            raise ValueError("M11U requires the M11F same-seed checkpoint and paired run directory.")
        frozen = load_json(paired_run_dir / "manifest.json")
        frozen_audit = frozen.get("finetune_mgkg", {}).get("stage3_init_checkpoint", {})
        if not audit.get("loaded") or not audit.get("stage1_stage2_skipped"):
            raise ValueError(f"M11U did not skip Stage1/2 through the checkpoint: {audit}")
        if audit.get("contract_sha256") != frozen_audit.get("contract_sha256"):
            raise ValueError("M11F and M11U Stage-2 checkpoint contract hashes differ.")
    elif audit.get("loaded") or audit.get("exported") or audit.get("stage1_stage2_skipped"):
        raise ValueError(f"Unexpected Stage-2 checkpoint routing for {cell}: {audit}")


def load_json(path: Path) -> dict[str, Any]:
    require_nonempty(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    require_nonempty(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Required artifact is missing or empty: {path}")


if __name__ == "__main__":
    main()
