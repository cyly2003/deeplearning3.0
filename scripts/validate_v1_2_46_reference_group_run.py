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
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
)
from scripts.build_v1_2_46_reference_group_splits import SOURCE_TABLE


PREDICTION_PARTS = ("finetune_mgkg", "finetune_mgkg_validation", "test")
CELL_SPECS: dict[str, dict[str, Any]] = {
    "M00": {
        "route": "M00",
        "epochs": (0, 0, 30),
        "source_weighting": "none",
    },
    "M10": {
        "route": "M10",
        "epochs": (30, 0, 30),
        "source_weighting": "none",
    },
    "M11U": {
        "route": "M11U",
        "epochs": (30, 20, 30),
        "source_weighting": "tanimoto_to_finetune",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validator for one v1.2.46 reference-group run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cell", required=True, choices=sorted(CELL_SPECS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
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
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def validate_run(
    run_dir: Path,
    *,
    cell: str,
    seed: int,
    split_summary: Path,
    source_table: str = SOURCE_TABLE,
    smoke: bool = False,
) -> dict[str, Any]:
    spec = CELL_SPECS[cell]
    summary = load_json(split_summary)
    validate_split_summary(summary, source_table=source_table)
    route = summary["routes"][spec["route"]]
    manifest = load_json(run_dir / "manifest.json")
    for name in ("predictions.csv", "history.csv", "best_model.pt", "preprocessing.json"):
        require_nonempty(run_dir / name)

    expected_epochs = (
        tuple(1 if value > 0 else 0 for value in spec["epochs"])
        if smoke
        else spec["epochs"]
    )
    stage3 = manifest.get("finetune_mgkg", {})
    checks = {
        "manifest_seed": (manifest.get("seed"), int(seed)),
        "manifest_split": (manifest.get("split_name"), route["split_name"]),
        "source_table": (manifest.get("data_source", {}).get("source_table"), source_table),
        "head_routing": (manifest.get("head_routing"), "task_target"),
        "mixed_targets": (manifest.get("allow_mixed_target_dimensions"), True),
        "epochs": (manifest.get("epochs"), expected_epochs[0]),
        "finetune_epochs": (manifest.get("finetune_epochs"), expected_epochs[1]),
        "stage3_epochs": (manifest.get("finetune_mgkg_epochs"), expected_epochs[2]),
        "ablation": (manifest.get("ablation"), "full"),
        "stage2_freeze": (manifest.get("finetune", {}).get("freeze"), "none"),
        "stage3_freeze": (stage3.get("freeze"), "none"),
        "stage3_validation_source": (stage3.get("validation_source"), "valid"),
        "stage3_validation_rows": (
            stage3.get("validation_rows"),
            summary["target_pool"]["counts"]["valid"],
        ),
        "task_filter_min_total": (manifest.get("task_filter", {}).get("min_total"), 0),
        "task_filter_min_train": (manifest.get("task_filter", {}).get("min_train"), 0),
        "task_filter_min_eval": (manifest.get("task_filter", {}).get("min_eval"), 0),
        "prediction_parts": (
            sorted(manifest.get("prediction_output_split_parts", [])),
            sorted(PREDICTION_PARTS),
        ),
        "medium_adapter": (
            manifest.get("ablation_features", {}).get("use_medium_adapter"),
            False,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")

    validate_full_feature_contract(manifest)
    validate_stage_protocol(manifest, cell=cell, expected_weighting=spec["source_weighting"])

    predictions = read_csv(run_dir / "predictions.csv")
    actual_parts = sorted({str(row.get("split_part", "")) for row in predictions})
    if actual_parts != sorted(PREDICTION_PARTS):
        raise ValueError(f"prediction split parts mismatch: {actual_parts}")
    identity_audit = {}
    expected_identity = summary["target_pool"]["evaluation_identity"]
    for part in PREDICTION_PARTS:
        rows = [row for row in predictions if str(row.get("split_part", "")) == part]
        records = [evaluation_identity_record(row, split_part=part) for row in rows]
        observed = {
            "rows": len(records),
            "sha256": canonical_sha256(sorted(records, key=canonical_json)),
        }
        if observed != expected_identity[part]:
            raise ValueError(
                f"prediction identity mismatch for {part}: "
                f"observed={observed}, expected={expected_identity[part]}"
            )
        identity_audit[part] = observed

    return {
        "status": "ok",
        "cell": cell,
        "seed": int(seed),
        "split_name": route["split_name"],
        "split_contract_sha256": summary["contract_sha256"],
        "prediction_identity": identity_audit,
    }


def validate_split_summary(summary: Mapping[str, Any], *, source_table: str) -> None:
    claimed = str(summary.get("contract_sha256", ""))
    payload = dict(summary)
    payload.pop("contract_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("split summary contract hash is invalid.")
    if (
        summary.get("schema") != "v1_2_46_reference_group_split_v1"
        or summary.get("status") != "locked"
        or summary.get("source_table") != source_table
    ):
        raise ValueError("split summary is not the locked v1.2.46 reference-group contract.")
    target = summary.get("target_pool", {})
    counts = target.get("counts", {})
    if sum(int(counts.get(part, 0)) for part in ("finetune_mgkg", "valid", "test")) != 15_199:
        raise ValueError(f"Locked Stage-3 population changed: {counts}")
    if int(target.get("tasks", 0)) != 18:
        raise ValueError("Reference-group boundary must retain exactly 18 tasks.")
    if any(int(value) != 0 for value in target.get("reference_overlap", {}).values()):
        raise ValueError("Reference-group boundary contains cross-part reference overlap.")
    identity_overlap = target.get("identity_overlap", {})
    for overlap in identity_overlap.values():
        if any(int(value) != 0 for value in overlap.values()):
            raise ValueError("Reference-group boundary contains cross-part identity overlap.")
    minimum = summary.get("partition_policy", {}).get("minimum_support", {})
    if minimum != {
        "train_rows_per_task": 100,
        "validation_rows_per_task": 1,
        "test_rows_per_task": 30,
        "validation_groups_per_task": 1,
        "test_groups_per_task": 5,
    }:
        raise ValueError(f"Reference-group support policy changed: {minimum}")
    if summary.get("partition_policy", {}).get("uses_model_predictions") is not False:
        raise ValueError("Reference-group split selection must be prediction-blind.")
    if set(summary.get("routes", {})) != set(CELL_SPECS):
        raise ValueError("Reference-group route set changed.")


def validate_full_feature_contract(manifest: Mapping[str, Any]) -> None:
    features = manifest.get("ablation_features", {})
    observed = (
        features.get("use_descriptors"),
        features.get("use_fingerprint"),
        features.get("use_context_numeric"),
        features.get("use_species_lifestage"),
        features.get("use_other_categorical_context"),
    )
    if observed != (True, True, True, True, True):
        raise ValueError(f"Full-input feature contract changed: {observed}")


def validate_stage_protocol(
    manifest: Mapping[str, Any], *, cell: str, expected_weighting: str
) -> None:
    stage2 = manifest.get("finetune", {})
    stage3 = manifest.get("finetune_mgkg", {})
    if float(stage3.get("learning_rate", -1)) != 0.0005:
        raise ValueError("Stage-3 unified learning rate must be 5e-4.")
    if stage3.get("trunk_learning_rate") is not None:
        raise ValueError(
            "Stage-3 freeze=none must use one unified optimizer LR; "
            "manifest trunk_learning_rate must resolve to null."
        )
    if float(stage3.get("soil_ptox_replay_fraction_requested", -1)) != 0.0:
        raise ValueError("Stage-3 soil-pTox replay fraction must be zero.")
    if cell == "M11U":
        if float(stage2.get("learning_rate", -1)) != 0.0001:
            raise ValueError("M11U Stage-2 learning rate must be 1e-4.")
        if int(manifest.get("finetune_rows", 0)) <= 0:
            raise ValueError("M11U lost its soil-pTox Stage-2 rows.")
    elif int(manifest.get("finetune_rows", 0)) != 0:
        raise ValueError(f"{cell} unexpectedly contains soil-pTox Stage-2 rows.")
    source_weight = manifest.get("source_weighting", {})
    if str(source_weight.get("method", "")) != expected_weighting:
        raise ValueError(
            f"{cell} source-weighting method changed: {source_weight.get('method')!r}"
        )
    checkpoint = stage3.get("stage3_init_checkpoint", {})
    if any(
        bool(checkpoint.get(key))
        for key in ("loaded", "exported", "stage1_stage2_skipped")
    ):
        raise ValueError("v1.2.46 runs must train their own reference-filtered Stage1/2 state.")


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
