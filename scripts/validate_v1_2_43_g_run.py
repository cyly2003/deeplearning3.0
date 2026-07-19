from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v1_2_43_g_splits import (
    assignment_hash_by_part,
    file_sha256,
    parse_result_ids,
    stable_hash,
)


PHASE_SPLITS = {
    "screen": "M_v1_2_43_g_screen",
    "final": "M_v1_2_43_g_final",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed protocol audit for one v1.2.43 G-series run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=["G0", "G1", "G2", "G3"])
    parser.add_argument("--phase", required=True, choices=["screen", "final"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--stage2-cache", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = validate_run(
        args.run_dir,
        candidate=args.candidate,
        phase=args.phase,
        seed=args.seed,
        split_name=args.split_name,
        source_table=args.source_table,
        db_path=args.db,
        split_summary=args.split_summary,
        stage2_cache=args.stage2_cache,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def validate_run(
    run_dir: Path,
    *,
    candidate: str,
    phase: str,
    seed: int,
    split_name: str,
    source_table: str,
    db_path: Path,
    split_summary: Path,
    stage2_cache: Path | None,
    smoke: bool,
) -> dict[str, Any]:
    required = ("manifest.json", "history.csv", "predictions.csv", "best_model.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing required G-series run artifacts: {missing}")
    if split_name != PHASE_SPLITS[phase]:
        raise ValueError(
            f"Validator phase/split mismatch: phase={phase} split={split_name}"
        )
    if not db_path.is_file():
        raise ValueError(f"Modeling database is missing: {db_path}")
    if not split_summary.is_file():
        raise ValueError(f"Split summary is missing: {split_summary}")
    if stage2_cache is not None and not stage2_cache.is_file():
        raise ValueError(f"Missing stage-2 initialization cache: {stage2_cache}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage3 = manifest.get("finetune_mgkg", {})
    target_bin = stage3.get("target_bin_sampling", {})
    hierarchy = stage3.get("hierarchical_head", {})
    checkpoint = stage3.get("stage3_init_checkpoint", {})
    expected_target_bin = candidate in {"G1", "G2"}
    expected_hierarchy = candidate == "G3"
    expected_mse = 0.15 if candidate == "G2" else 0.0
    checks: dict[str, tuple[Any, Any]] = {
        "manifest_seed": (manifest.get("seed"), seed),
        "manifest_split": (manifest.get("split_name"), split_name),
        "manifest_source_table": (
            manifest.get("data_source", {}).get("source_table"),
            source_table,
        ),
        "stage1_epochs": (manifest.get("epochs"), 1 if smoke else 30),
        "stage2_epochs": (manifest.get("finetune_epochs"), 1 if smoke else 20),
        "stage3_epochs": (manifest.get("finetune_mgkg_epochs"), 1 if smoke else 40),
        "stage3_epochs_ran": (
            manifest.get("finetune_mgkg_epochs_ran"),
            1 if smoke else 40,
        ),
        "stage3_validation_seed": (manifest.get("finetune_mgkg_validation_seed"), 17073),
        "stage3_validation_source": (stage3.get("validation_source"), "valid"),
        "stage3_early_stopping": (stage3.get("early_stopping"), False),
        "stage3_freeze": (stage3.get("freeze"), "heads_only"),
        "stage3_head_lr": (stage3.get("head_learning_rate"), 0.0005),
        "stage3_trunk_lr": (stage3.get("trunk_learning_rate"), None),
        "stage3_replay": (stage3.get("soil_ptox_replay_fraction_requested"), 0.0),
        "stage3_mse_weight": (stage3.get("regression_loss", {}).get("mse_weight"), expected_mse),
        "target_bin_enabled": (target_bin.get("enabled"), expected_target_bin),
        "hierarchical_enabled": (hierarchy.get("enabled"), expected_hierarchy),
        "medium_adapter_disabled": (
            manifest.get("ablation_features", {}).get("use_medium_adapter"),
            False,
        ),
    }
    manifest_db = Path(str(manifest.get("data_source", {}).get("modeling_tables_db", "")))
    if manifest_db.resolve() != db_path.resolve():
        raise ValueError(
            f"Manifest database mismatch: actual={manifest_db} expected={db_path}"
        )
    expected_prediction_parts = (
        ["finetune_mgkg_validation"]
        if phase == "screen"
        else ["finetune_mgkg_validation", "test"]
    )
    checks["manifest_prediction_boundary"] = (
        sorted(manifest.get("prediction_output_split_parts", [])),
        sorted(expected_prediction_parts),
    )
    if expected_target_bin:
        checks.update(
            {
                "target_bin_kind": (
                    target_bin.get("kind"),
                    "within_task_equal_width_target_bin_inverse_frequency",
                ),
                "target_bins": (target_bin.get("requested_bins"), 10),
                "target_bin_min_weight": (target_bin.get("min_weight"), 0.5),
                "target_bin_max_weight": (target_bin.get("max_weight"), 2.0),
                "target_bin_train_only": (
                    target_bin.get("count_source"),
                    "finetune_mgkg_training_only",
                ),
            }
        )
    if expected_hierarchy:
        checks.update(
            {
                "hierarchy_family_tau": (hierarchy.get("family_tau"), 128.0),
                "hierarchy_task_tau": (hierarchy.get("task_tau"), 64.0),
                "hierarchy_train_only": (
                    hierarchy.get("count_source"),
                    "finetune_mgkg_training_only",
                ),
            }
        )
    if candidate != "G0":
        checks.update(
            {
                "stage2_cache_loaded": (checkpoint.get("loaded"), True),
                "stage1_stage2_skipped": (checkpoint.get("stage1_stage2_skipped"), True),
            }
        )
    elif stage2_cache is not None:
        if not bool(checkpoint.get("loaded")) and not bool(checkpoint.get("exported")):
            raise ValueError("G0 neither loaded nor exported the declared stage-2 cache.")

    assert_prediction_boundary(
        run_dir / "predictions.csv",
        phase=phase,
        split_summary=split_summary,
    )
    assert_current_split_identity(
        db_path,
        split_name=split_name,
        source_table=source_table,
        split_summary=split_summary,
    )
    assert_checkpoint_contract(
        stage2_cache,
        checkpoint_audit=checkpoint,
        seed=seed,
        split_name=split_name,
        source_table=source_table,
    )

    failures = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if not equal(actual, expected)
    }
    if failures:
        raise ValueError(json.dumps(failures, ensure_ascii=False, sort_keys=True))
    return {
        "status": "ok",
        "candidate": candidate,
        "phase": phase,
        "seed": seed,
        "split_name": split_name,
        "split_summary_sha256": file_sha256(split_summary),
        "run_dir": str(run_dir),
        "stage2_cache": "" if stage2_cache is None else str(stage2_cache),
    }


def assert_prediction_boundary(
    predictions_path: Path,
    *,
    phase: str,
    split_summary: Path,
) -> None:
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Predictions export is empty.")
    expected_parts = (
        {"finetune_mgkg_validation"}
        if phase == "screen"
        else {"finetune_mgkg_validation", "test"}
    )
    actual_parts = {str(row.get("split_part", "")).strip().lower() for row in rows}
    if actual_parts != expected_parts:
        raise ValueError(
            f"Prediction boundary mismatch: expected={sorted(expected_parts)} "
            f"actual={sorted(actual_parts)}"
        )
    for row in rows:
        if str(row.get("target_name", "")).strip() != "neg_log10_mol_kg":
            raise ValueError("Prediction export contains a non-molar target row.")
        if str(row.get("target_family", "")).strip() != "solid_neglog_mol_kg":
            raise ValueError("Prediction export contains a non-molar target family.")
        if str(row.get("medium_domain", "")).strip().lower() != "soil":
            raise ValueError("Prediction export contains a non-soil row.")
    summary = json.loads(split_summary.read_text(encoding="utf-8"))
    validation = [
        row
        for row in rows
        if str(row.get("split_part", "")).strip().lower()
        == "finetune_mgkg_validation"
    ]
    validation_ids = [str(row.get("aggregate_id", "")).strip() for row in validation]
    if any(not identity for identity in validation_ids) or len(validation_ids) != len(
        set(validation_ids)
    ):
        raise ValueError("Validation predictions contain blank or duplicate aggregate identities.")
    validation_result_ids = {
        result_id
        for row in validation
        for result_id in parse_result_ids(row.get("result_ids"))
    }
    if stable_hash(validation_ids) != summary.get("new_validation_aggregate_id_sha256"):
        raise ValueError("Prediction validation aggregate hash differs from the split lock.")
    if stable_hash(validation_result_ids) != summary.get("new_validation_result_ids_sha256"):
        raise ValueError("Prediction validation result_ids hash differs from the split lock.")


def assert_current_split_identity(
    db_path: Path,
    *,
    split_name: str,
    source_table: str,
    split_summary: Path,
) -> None:
    summary = json.loads(split_summary.read_text(encoding="utf-8"))
    if summary.get("built_split") != split_name or summary.get("source_table") != source_table:
        raise ValueError("Split summary does not describe the requested current split.")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT split_name, record_id, aggregate_id, split_part, seed,
                   split_type, source_table, group_key
            FROM split_assignments
            WHERE split_name = ? AND source_table = ?
            ORDER BY rowid
            """,
            (split_name, source_table),
        ).fetchall()
    if not rows:
        raise ValueError(f"Current G split is missing from the database: {split_name}")
    assignments = [
        (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            str(row[5]),
            str(row[6]),
            str(row[7] or ""),
        )
        for row in rows
    ]
    current_hashes = assignment_hash_by_part(assignments)
    expected_hashes = summary.get("split_assignment_sha256_by_part")
    if current_hashes != expected_hashes:
        raise ValueError(
            "Current split assignment hash differs from the immutable split summary."
        )


def assert_checkpoint_contract(
    stage2_cache: Path | None,
    *,
    checkpoint_audit: dict[str, Any],
    seed: int,
    split_name: str,
    source_table: str,
) -> None:
    if stage2_cache is None:
        raise ValueError("A stage-2 cache is required for every G-series run.")
    audit_path = Path(str(checkpoint_audit.get("path", "")))
    if audit_path.resolve() != stage2_cache.resolve():
        raise ValueError(
            f"Manifest checkpoint path mismatch: actual={audit_path} expected={stage2_cache}"
        )
    import torch

    try:
        payload = torch.load(stage2_cache, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = torch.load(stage2_cache, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "qsar_stage3_init_v1":
        raise ValueError("Stage-2 cache has an unsupported format.")
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Stage-2 cache is missing its compatibility contract.")
    from qsar_tl.training.deep_experiment import verify_stage3_init_contract

    contract_sha = verify_stage3_init_contract(contract, label="validator")
    data_identity = contract.get("data_identity", {})
    stage12_protocol = contract.get("stage12_protocol", {})
    expected = {
        "seed": (stage12_protocol.get("seed"), seed),
        "split_name": (data_identity.get("split_name"), split_name),
        "source_table": (data_identity.get("source_table"), source_table),
    }
    mismatches = {
        key: {"actual": actual, "expected": value}
        for key, (actual, value) in expected.items()
        if actual != value
    }
    if contract_sha != str(checkpoint_audit.get("contract_sha256", "")):
        mismatches["contract_sha256"] = {
            "actual": contract_sha,
            "expected": checkpoint_audit.get("contract_sha256"),
        }
    if mismatches:
        raise ValueError(
            "Stage-2 cache contract mismatch: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


if __name__ == "__main__":
    main()
