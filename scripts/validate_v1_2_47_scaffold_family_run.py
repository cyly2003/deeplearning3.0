from __future__ import annotations

import argparse
import csv
import json
import math
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


SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
SCHEMA = "v1_2_47_scaffold_family_split_v1"
SIMILARITY_THRESHOLD = 0.65
LOW_SUPPORT_TASKS = frozenset({"ECx_Population", "ICx_Growth"})
PREDICTION_PARTS = ("finetune_mgkg", "finetune_mgkg_validation", "test")
CELL_SPECS: dict[str, dict[str, Any]] = {
    "M00": {"epochs": (0, 0, 30), "source_weighting": "none"},
    "M10": {"epochs": (30, 0, 30), "source_weighting": "none"},
    "M11U": {
        "epochs": (30, 20, 30),
        "source_weighting": "tanimoto_to_finetune",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed validator for one v1.2.47 scaffold-family run."
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
    route = summary["routes"][cell]
    manifest = load_json(run_dir / "manifest.json")
    for name in ("predictions.csv", "history.csv", "best_model.pt", "preprocessing.json"):
        require_nonempty(run_dir / name)

    expected_epochs = (
        tuple(1 if value > 0 else 0 for value in spec["epochs"])
        if smoke
        else spec["epochs"]
    )
    stage2 = manifest.get("finetune", {})
    stage3 = manifest.get("finetune_mgkg", {})
    valid_rows = int(summary["target_pool"]["counts"]["valid"])
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
        "stage2_freeze": (stage2.get("freeze"), "none"),
        "stage3_freeze": (stage3.get("freeze"), "none"),
        "stage3_early_stopping": (stage3.get("early_stopping"), True),
        "stage3_validation_source": (stage3.get("validation_source"), "valid"),
        "stage3_validation_rows": (stage3.get("validation_rows"), valid_rows),
        "route_rows": (manifest.get("rows"), int(route["rows"])),
        "stage3_train_rows": (
            manifest.get("finetune_mgkg_train_rows"),
            int(summary["target_pool"]["counts"]["finetune_mgkg"]),
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
    validate_stage_protocol(
        manifest,
        cell=cell,
        expected_weighting=str(spec["source_weighting"]),
    )
    validate_stage3_history(run_dir / "history.csv", expected_validation_rows=valid_rows)

    predictions = read_csv(run_dir / "predictions.csv")
    actual_parts = sorted({str(row.get("split_part", "")) for row in predictions})
    if actual_parts != sorted(PREDICTION_PARTS):
        raise ValueError(f"prediction split parts mismatch: {actual_parts}")
    identity_audit: dict[str, dict[str, Any]] = {}
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
        "test_used_for_selection": False,
    }


def validate_split_summary(summary: Mapping[str, Any], *, source_table: str) -> None:
    claimed = str(summary.get("contract_sha256", ""))
    payload = dict(summary)
    payload.pop("contract_sha256", None)
    if claimed != canonical_sha256(payload):
        raise ValueError("split summary contract hash is invalid.")
    if (
        summary.get("schema") != SCHEMA
        or summary.get("status") != "locked"
        or summary.get("source_table") != source_table
    ):
        raise ValueError("split summary is not the locked v1.2.47 scaffold-family contract.")
    if set(summary.get("routes", {})) != set(CELL_SPECS):
        raise ValueError("Scaffold-family route set must be exactly M00/M10/M11U.")
    if set(summary.get("route_semantics", {})) != set(CELL_SPECS):
        raise ValueError("Scaffold-family route semantics are incomplete.")

    policy = summary.get("partition_policy", {})
    if policy.get("uses_model_predictions") is not False:
        raise ValueError("Scaffold-family split selection must be prediction-blind.")
    structure_policy = summary.get("structure_policy", {})
    threshold = float(
        structure_policy.get(
            "tanimoto_threshold",
            policy.get("tanimoto_threshold", policy.get("similarity_threshold", -1)),
        )
    )
    if not math.isclose(threshold, SIMILARITY_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Scaffold-family Tanimoto threshold changed: {threshold}")

    target = summary.get("target_pool", {})
    counts = target.get("counts", {})
    if set(counts) != {"finetune_mgkg", "valid", "test"}:
        raise ValueError(f"Unexpected Stage-3 split parts: {counts}")
    if any(int(counts[part]) <= 0 for part in counts):
        raise ValueError(f"Stage-3 split contains an empty part: {counts}")
    if int(target.get("tasks", 0)) != 18:
        raise ValueError("Scaffold-family boundary must retain exactly 18 tasks.")
    if int(target.get("original_rows", 15_199)) != 15_199:
        raise ValueError("Locked v1.2.44 Stage-3 candidate population changed.")
    validate_route_counts(summary.get("routes", {}), target_counts=counts)

    require_zero_nested(target.get("identity_overlap", {}), label="target identity overlap")
    require_zero_nested(target.get("structure_overlap", {}), label="target structure overlap")
    require_similarity_below_threshold(
        target.get("max_cross_part_tanimoto", {}),
        label="target cross-part Tanimoto",
    )
    validate_task_support(target.get("task_counts", {}))
    source = summary.get("source_structure_exclusion", {})
    if not source:
        raise ValueError("Missing source-domain structure-exclusion audit.")
    require_zero_nested(
        source.get(
            "structure_overlap",
            {
                "source_train_vs_stage3_valid_test": {
                    "canonical_parent": source.get("canonical_overlap"),
                    "murcko_scaffold": source.get("murcko_scaffold_overlap"),
                }
            },
        ),
        label="source versus held-out structure overlap",
    )
    require_zero_nested(
        source.get("exact_identity_overlap", {}),
        label="source versus held-out exact identity overlap",
    )
    require_similarity_below_threshold(
        source.get("max_tanimoto_to_heldout", {}),
        label="source versus held-out Tanimoto",
    )
    invalid_policy = source.get(
        "missing_or_invalid_structure_policy", structure_policy.get("invalid_policy")
    )
    if invalid_policy not in {"exclude", "exclude_and_report", "exclude_and_audit"}:
        raise ValueError("Source records with missing/invalid structures must be excluded.")

    support_policy = policy.get("minimum_support", {})
    if set(support_policy.get("locked_low_support_tasks", [])) != LOW_SUPPORT_TASKS:
        raise ValueError("Locked low-support task set changed.")
    tiers = support_policy.get("support_tiers", {})
    expected_tiers = {
        "standard": {
            "train_rows": 100,
            "train_components": 5,
            "validation_rows": 1,
            "validation_components": 1,
            "test_rows": 30,
            "test_components": 5,
        },
        "low_support": {
            "train_rows": 20,
            "train_components": 1,
            "validation_rows": 1,
            "validation_components": 1,
            "test_rows": 10,
            "test_components": 1,
        },
    }
    for tier, required in expected_tiers.items():
        observed = tiers.get(tier, {})
        if any(int(observed.get(key, -1)) != value for key, value in required.items()):
            raise ValueError(f"Support-tier policy changed for {tier}: {observed}")
    if policy.get("support_tiers") != tiers:
        raise ValueError("Top-level and minimum-support tier contracts differ.")
    task_tiers = policy.get("task_support_tiers", {})
    task_counts = target.get("task_counts", {})
    if set(task_tiers) != set(task_counts):
        raise ValueError("Task support-tier mapping is incomplete.")
    observed_low_support = {
        str(task) for task, tier in task_tiers.items() if str(tier) == "low_support"
    }
    if observed_low_support != LOW_SUPPORT_TASKS:
        raise ValueError(
            "Observed low-support task set changed: "
            f"{sorted(observed_low_support)}"
        )
    reporting_support = target.get("task_reporting_support", {})
    if set(reporting_support) != set(task_counts):
        raise ValueError("Task-level reporting support map is incomplete.")
    for task, parts in task_counts.items():
        test_support = parts.get("test", {}) if isinstance(parts, Mapping) else {}
        expected_r2 = (
            int(test_support.get("rows", 0)) >= 30
            and int(test_support.get("groups", 0)) >= 5
        )
        if bool(reporting_support[task].get("task_r2_supported")) != expected_r2:
            raise ValueError(f"Task-level R2 reporting flag changed for {task}.")


def validate_route_counts(
    routes: Mapping[str, Any], *, target_counts: Mapping[str, Any]
) -> None:
    target = {part: int(target_counts[part]) for part in ("finetune_mgkg", "valid", "test")}
    for cell in CELL_SPECS:
        counts = routes[cell].get("counts", {})
        observed_target = {part: int(counts.get(part, 0)) for part in target}
        if observed_target != target:
            raise ValueError(f"{cell} Stage-3 target identities/counts changed: {counts}")
        has_train = int(counts.get("train", 0)) > 0
        has_stage2 = int(counts.get("finetune", 0)) > 0
        expected = {
            "M00": (False, False),
            "M10": (True, False),
            "M11U": (True, True),
        }[cell]
        if (has_train, has_stage2) != expected:
            raise ValueError(
                f"{cell} route source-stage identity changed: "
                f"observed={(has_train, has_stage2)}, expected={expected}"
            )


def validate_task_support(task_counts: Mapping[str, Any]) -> None:
    if len(task_counts) != 18:
        raise ValueError(f"Expected support audit for 18 tasks, observed {len(task_counts)}.")
    for task, parts in task_counts.items():
        low_support = str(task) in LOW_SUPPORT_TASKS
        minimum = {
            "finetune_mgkg": {
                "rows": 20 if low_support else 100,
                "groups": 1 if low_support else 5,
            },
            "valid": {"rows": 1, "groups": 1},
            "test": {
                "rows": 10 if low_support else 30,
                "groups": 1 if low_support else 5,
            },
        }
        for part, required in minimum.items():
            observed = parts.get(part, {}) if isinstance(parts, Mapping) else {}
            for field, threshold in required.items():
                if int(observed.get(field, 0)) < threshold:
                    raise ValueError(
                        f"Task support gate failed for {task}/{part}/{field}: "
                        f"observed={observed.get(field)!r}, required={threshold}"
                    )


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
        raise ValueError("Stage-3 full finetuning must use one unified optimizer LR.")
    if float(stage3.get("soil_ptox_replay_fraction_requested", -1)) != 0.0:
        raise ValueError("Stage-3 soil-pTox replay fraction must be zero.")
    if cell == "M11U":
        if float(stage2.get("learning_rate", -1)) != 0.0001:
            raise ValueError("M11U Stage-2 learning rate must be 1e-4.")
        if int(manifest.get("finetune_rows", 0)) <= 0:
            raise ValueError("M11U lost its structure-filtered soil-pTox Stage-2 rows.")
    elif int(manifest.get("finetune_rows", 0)) != 0:
        raise ValueError(f"{cell} unexpectedly contains soil-pTox Stage-2 rows.")
    if cell == "M00" and int(manifest.get("train_rows", 0)) != 0:
        raise ValueError("M00 unexpectedly contains aquatic Stage-1 rows.")
    if cell in {"M10", "M11U"} and int(manifest.get("train_rows", 0)) <= 0:
        raise ValueError(f"{cell} lost its structure-filtered aquatic Stage-1 rows.")
    source_weight = manifest.get("source_weighting", {})
    if str(source_weight.get("method", "")) != expected_weighting:
        raise ValueError(f"{cell} source weighting changed: {source_weight.get('method')!r}")
    expected_applied = cell == "M11U"
    if bool(source_weight.get("applied", False)) is not expected_applied:
        raise ValueError(
            f"{cell} source weighting application changed: "
            f"observed={source_weight.get('applied')!r}, expected={expected_applied}"
        )
    checkpoint = stage3.get("stage3_init_checkpoint", {})
    if any(
        bool(checkpoint.get(key))
        for key in ("loaded", "exported", "stage1_stage2_skipped")
    ):
        raise ValueError("v1.2.47 must train its own structure-filtered Stage1/2 state.")


def validate_stage3_history(path: Path, *, expected_validation_rows: int) -> None:
    rows = read_csv(path)
    stage3_rows = [row for row in rows if str(row.get("phase", "")) == "finetune_mgkg"]
    if not stage3_rows:
        raise ValueError("History contains no finetune_mgkg epochs.")
    for row in stage3_rows:
        raw = str(row.get("validation_samples", "")).strip()
        if not raw:
            raise ValueError("Stage-3 history does not record validation_samples.")
        if int(float(raw)) != int(expected_validation_rows):
            raise ValueError(
                "Stage-3 history validation sample count changed: "
                f"observed={raw}, expected={expected_validation_rows}"
            )


def require_zero_nested(value: Any, *, label: str) -> None:
    numbers = list(iter_numbers(value))
    if not numbers:
        raise ValueError(f"Missing {label} audit values.")
    if any(number != 0 for number in numbers):
        raise ValueError(f"{label} is nonzero: {value}")


def require_similarity_below_threshold(value: Any, *, label: str) -> None:
    numbers = list(iter_numbers(value))
    if not numbers:
        raise ValueError(f"Missing {label} audit values.")
    if any(number >= SIMILARITY_THRESHOLD for number in numbers):
        raise ValueError(f"{label} reaches the locked threshold: {value}")


def iter_numbers(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from iter_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_numbers(item)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield float(value)


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
