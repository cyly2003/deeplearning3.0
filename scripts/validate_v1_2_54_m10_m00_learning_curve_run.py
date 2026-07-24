from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v1_2_44_second_layer_splits import canonical_sha256, frame_record_id
from scripts.build_v1_2_54_m10_m00_target_data_learning_curve_splits import (
    FRACTIONS,
    MODEL_SEEDS,
    SOURCE_TABLE,
)


PREDICTION_PARTS = (
    "finetune_mgkg",
    "finetune_mgkg_validation",
    "test",
)
ROUTES = ("M10", "M00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed validator for one v1.2.54 paired learning-curve cell."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=ROUTES)
    parser.add_argument("--fraction", required=True, type=int, choices=FRACTIONS)
    parser.add_argument("--seed", required=True, type=int, choices=MODEL_SEEDS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Required JSON is missing or empty: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Required artifact is missing or empty: {path}")


def read_predictions(path: Path) -> list[dict[str, str]]:
    require_nonempty(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file has no data rows: {path}")
    return rows


def expected_fraction(audit: dict[str, Any], fraction: int) -> dict[str, Any]:
    matches = [
        item
        for item in audit.get("fractions", [])
        if int(item.get("fraction_percent", -1)) == fraction
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one audit entry for fraction={fraction}")
    return matches[0]


def validate_contract(audit: dict[str, Any]) -> None:
    claimed = str(audit.get("contract_sha256", ""))
    unsigned = dict(audit)
    unsigned.pop("contract_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("Split-audit contract hash is invalid")
    if audit.get("schema") != "v1_2_54_m10_m00_target_data_learning_curve_split_v1":
        raise ValueError("Unexpected split-audit schema")
    if audit.get("status") != "persisted_and_validated":
        raise ValueError("Learning-curve splits were not persisted and validated")
    if audit.get("routes") != list(ROUTES):
        raise ValueError("Learning-curve route order changed")
    if audit.get("model_seeds") != list(MODEL_SEEDS):
        raise ValueError("Learning-curve model seeds changed")
    if audit.get("new_training_cells") != 32:
        raise ValueError("Learning-curve cell count changed")


def validate_run(
    *,
    run_dir: Path,
    split_audit: Path,
    route: str,
    fraction: int,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    audit = load_json(split_audit)
    validate_contract(audit)
    fraction_audit = expected_fraction(audit, fraction)
    route_audit = fraction_audit["routes"][route]

    for name in (
        "manifest.json",
        "predictions.csv",
        "history.csv",
        "preprocessing.json",
        "best_model.pt",
    ):
        require_nonempty(run_dir / name)
    manifest = load_json(run_dir / "manifest.json")
    target_standardization = manifest.get("target_standardization")
    if isinstance(target_standardization, dict):
        target_standardization_mode = target_standardization.get("mode")
    else:
        target_standardization_mode = target_standardization

    stage1_epochs = 0 if route == "M00" else (1 if smoke else 30)
    stage3_epochs = 1 if smoke else 30
    checks = {
        "seed": (manifest.get("seed"), seed),
        "split_name": (manifest.get("split_name"), route_audit["split_name"]),
        "source_table": (
            manifest.get("data_source", {}).get("source_table"),
            SOURCE_TABLE,
        ),
        "epochs": (manifest.get("epochs"), stage1_epochs),
        "finetune_epochs": (manifest.get("finetune_epochs"), 0),
        "finetune_mgkg_epochs": (
            manifest.get("finetune_mgkg_epochs"),
            stage3_epochs,
        ),
        "ablation": (manifest.get("ablation"), "full"),
        "head_routing": (manifest.get("head_routing"), "task_target"),
        "target_standardization": (
            target_standardization_mode,
            "per_task_target",
        ),
        "mixed_target_dimensions": (
            manifest.get("allow_mixed_target_dimensions"),
            True,
        ),
        "task_filter_min_total": (
            manifest.get("task_filter", {}).get("min_total"),
            0,
        ),
        "task_filter_min_train": (
            manifest.get("task_filter", {}).get("min_train"),
            0,
        ),
        "task_filter_min_eval": (
            manifest.get("task_filter", {}).get("min_eval"),
            0,
        ),
        "prediction_parts": (
            sorted(manifest.get("prediction_output_split_parts", [])),
            sorted(PREDICTION_PARTS),
        ),
        "source_weighting_method": (
            manifest.get("source_weighting", {}).get("method"),
            "none",
        ),
        "source_weighting_applied": (
            manifest.get("source_weighting", {}).get("applied"),
            False,
        ),
        "stage3_rows": (
            manifest.get("finetune_mgkg", {}).get("rows"),
            fraction_audit["target_train_rows"],
        ),
        "stage3_validation_rows": (
            manifest.get("finetune_mgkg", {}).get("validation_rows"),
            fraction_audit["target_validation_rows"],
        ),
        "stage3_validation_source": (
            manifest.get("finetune_mgkg", {}).get("validation_source"),
            "valid",
        ),
        "stage3_freeze": (
            manifest.get("finetune_mgkg", {}).get("freeze"),
            "none",
        ),
        "stage3_learning_rate": (
            manifest.get("finetune_mgkg", {}).get("learning_rate"),
            0.0005,
        ),
        "stage3_trunk_learning_rate": (
            manifest.get("finetune_mgkg", {}).get("trunk_learning_rate"),
            None,
        ),
        "stage3_replay_fraction": (
            manifest.get("finetune_mgkg", {}).get(
                "soil_ptox_replay_fraction_requested"
            ),
            0.0,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(
                f"{label} mismatch: observed={observed!r}, expected={expected!r}"
            )

    features = manifest.get("ablation_features", {})
    full_features = (
        features.get("use_descriptors"),
        features.get("use_fingerprint"),
        features.get("use_context_numeric"),
        features.get("use_species_lifestage"),
        features.get("use_other_categorical_context"),
        features.get("use_medium_adapter"),
    )
    if full_features != (True, True, True, True, True, False):
        raise ValueError(f"Full-input feature contract changed: {full_features}")

    checkpoint = manifest.get("finetune_mgkg", {}).get(
        "stage3_init_checkpoint", {}
    )
    if (
        checkpoint.get("loaded")
        or checkpoint.get("exported")
        or checkpoint.get("stage1_stage2_skipped")
    ):
        raise ValueError("Unexpected cross-cell checkpoint reuse")

    if route == "M10":
        if manifest.get("finetune_rows") != 0:
            raise ValueError("M10 unexpectedly contains Stage-2 rows")
        if (
            int(manifest.get("actual_train_rows", -1))
            + int(manifest.get("validation_rows", -1))
            != int(audit["aquatic_stage1_rows"])
        ):
            raise ValueError("M10 aquatic Stage-1 row accounting changed")
    else:
        if any(
            int(manifest.get(name, -1)) != 0
            for name in ("actual_train_rows", "validation_rows", "finetune_rows")
        ):
            raise ValueError("M00 unexpectedly contains Stage-1/2 rows")

    predictions = read_predictions(run_dir / "predictions.csv")
    actual_parts = sorted({row.get("split_part", "") for row in predictions})
    if actual_parts != sorted(PREDICTION_PARTS):
        raise ValueError(f"Prediction split parts changed: {actual_parts}")
    expected_identity = {
        "finetune_mgkg": (
            fraction_audit["target_train_rows"],
            fraction_audit["target_train_record_id_sha256"],
        ),
        "finetune_mgkg_validation": (
            fraction_audit["target_validation_rows"],
            fraction_audit["target_validation_record_id_sha256"],
        ),
        "test": (
            fraction_audit["target_test_rows"],
            fraction_audit["target_test_record_id_sha256"],
        ),
    }
    prediction_audit: dict[str, Any] = {}
    for part in PREDICTION_PARTS:
        rows = [row for row in predictions if row.get("split_part") == part]
        record_ids = [frame_record_id(row) for row in rows]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError(f"Duplicate prediction identities in {part}")
        observed = (len(record_ids), canonical_sha256(sorted(record_ids)))
        if observed != expected_identity[part]:
            raise ValueError(
                f"Prediction identity mismatch for {part}: "
                f"observed={observed}, expected={expected_identity[part]}"
            )
        prediction_audit[part] = {
            "rows": observed[0],
            "record_id_sha256": observed[1],
        }

    return {
        "status": "ok",
        "route": route,
        "fraction_percent": fraction,
        "seed": seed,
        "split_name": route_audit["split_name"],
        "prediction_identity": prediction_audit,
    }


def main() -> None:
    args = parse_args()
    result = validate_run(
        run_dir=args.run_dir,
        split_audit=args.split_audit,
        route=args.route,
        fraction=args.fraction,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
