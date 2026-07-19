from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


TARGET_NAME = "neg_log10_mol_kg"
TARGET_FAMILY = "solid_neglog_mol_kg"
OLD_VALIDATION_PART = "finetune_mgkg_validation"
OLD_TRAIN_PART = "finetune_mgkg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v1.2.43 G-series screen/final splits. The new stage-3 "
            "validation set is sampled only from the former stage-3 training "
            "population; the former validation identities return to training."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--source-table",
        default="aggregated_task_records_ptox_soil_mass_molar_qc",
    )
    parser.add_argument(
        "--parent-split",
        default="M_v1_2_40_ptox_to_soil_molkg_B_random_8_2",
    )
    parser.add_argument(
        "--old-baseline-root",
        required=True,
        type=Path,
        help="Root containing the completed v1.2.40 X0_molar seed-42 run.",
    )
    parser.add_argument("--screen-split", default="M_v1_2_43_g_screen")
    parser.add_argument("--final-split", default="M_v1_2_43_g_final")
    parser.add_argument("--phase", choices=["screen", "final"], required=True)
    parser.add_argument(
        "--winner-lock",
        type=Path,
        default=None,
        help="Required for phase=final; proves validation-only selection was locked first.",
    )
    parser.add_argument("--validation-seed", type=int, default=17073)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/audits/v1_2_43_g_series/split_audit.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/audits/v1_2_43_g_series/split_summary.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_g_splits(
        db_path=args.db,
        source_table=args.source_table,
        parent_split=args.parent_split,
        old_baseline_root=args.old_baseline_root,
        screen_split=args.screen_split,
        final_split=args.final_split,
        phase=args.phase,
        winner_lock=args.winner_lock,
        validation_seed=args.validation_seed,
        quantile_bins=args.quantile_bins,
        audit_csv=args.audit_csv,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_g_splits(
    *,
    db_path: Path,
    source_table: str,
    parent_split: str,
    old_baseline_root: Path,
    screen_split: str,
    final_split: str,
    phase: str,
    winner_lock: Path | None,
    validation_seed: int,
    quantile_bins: int,
    audit_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    if phase not in {"screen", "final"}:
        raise ValueError(f"Unsupported G split phase: {phase}")
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least 2.")
    winner_lock_payload: dict[str, Any] | None = None
    if phase == "final":
        winner_lock_payload = validate_winner_lock_presence(winner_lock)

    prediction_path = discover_old_baseline_predictions(old_baseline_root)
    old_train_ids, old_validation_ids, baseline_scientific_by_id = (
        load_old_stage3_identities(prediction_path)
    )
    if old_train_ids & old_validation_ids:
        raise ValueError("Former stage-3 training and validation identities overlap.")

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_table(conn, source_table)
        _ensure_table(conn, "split_assignments")
        parent_sql = """
            SELECT record_id, aggregate_id, split_part, seed, split_type,
                   source_table, group_key
            FROM split_assignments
            WHERE split_name = ? AND source_table = ?
        """
        # Screening is blind even to outer-test assignment identities/counts.
        # The test predicate is appended only after a winner lock exists.
        if phase == "screen":
            parent_sql += " AND split_part IN ('train', 'finetune', 'finetune_mgkg')"
        else:
            parent_sql += " AND split_part IN ('train', 'finetune', 'finetune_mgkg', 'test')"
        parent_sql += " ORDER BY rowid"
        parent = conn.execute(parent_sql, (parent_split, source_table)).fetchall()
        if not parent:
            raise ValueError(f"Parent split does not exist: {parent_split}")
        parent_counts = Counter(str(row["split_part"]) for row in parent)
        required_parts = ("train", "finetune", "finetune_mgkg") + (
            ("test",) if phase == "final" else ()
        )
        for required in required_parts:
            if parent_counts[required] <= 0:
                raise ValueError(f"Parent split is missing required part {required!r}.")

        stage3_rows = [
            row for row in parent if str(row["split_part"]) == "finetune_mgkg"
        ]
        stage3_ids = {
            str(row["aggregate_id"])
            for row in stage3_rows
        }
        if len(stage3_ids) != len(stage3_rows):
            raise ValueError("Parent stage-3 contains duplicate aggregate assignments.")
        parent_stage3_record_ids = {str(row["record_id"]) for row in stage3_rows}
        if len(parent_stage3_record_ids) != len(stage3_rows):
            raise ValueError("Parent stage-3 contains duplicate record assignments.")
        baseline_ids = old_train_ids | old_validation_ids
        if baseline_ids != stage3_ids:
            missing_parent = sorted(baseline_ids - stage3_ids)
            parent_extras = sorted(stage3_ids - baseline_ids)
            raise ValueError(
                "Old baseline and parent stage-3 aggregate populations differ: "
                f"baseline_only={missing_parent[:5]} parent_only={parent_extras[:5]}"
            )

        metadata = load_target_metadata(conn, source_table, stage3_ids)
        missing_metadata = sorted(stage3_ids - set(metadata))
        if missing_metadata:
            raise ValueError(
                "Parent stage-3 contains rows outside the locked soil mol/kg target contract: "
                f"{missing_metadata[:5]}"
            )
        assert_parent_stage3_contract(stage3_rows, metadata=metadata)
        parent_scientific_by_id = parent_stage3_scientific_identities(
            stage3_rows,
            metadata=metadata,
        )
        baseline_record_ids = {
            str(item["record_id"])
            for item in baseline_scientific_by_id.values()
        }
        if baseline_record_ids != parent_stage3_record_ids:
            raise ValueError(
                "Old baseline and parent stage-3 record populations differ: "
                f"baseline_only={sorted(baseline_record_ids - parent_stage3_record_ids)[:5]} "
                f"parent_only={sorted(parent_stage3_record_ids - baseline_record_ids)[:5]}"
            )
        scientific_mismatches = [
            identity
            for identity in sorted(stage3_ids)
            if baseline_scientific_by_id[identity] != parent_scientific_by_id[identity]
        ]
        if scientific_mismatches:
            raise ValueError(
                "Old baseline and parent stage-3 scientific identities differ: "
                f"{scientific_mismatches[:5]}"
            )

        old_validation_result_ids = {
            result_id
            for identity in old_validation_ids
            for result_id in metadata[identity]["result_ids"]
        }
        component_by_identity = aggregate_result_components(stage3_ids, metadata=metadata)
        eligible_pool = {
            identity
            for identity in old_train_ids
            if len(component_by_identity[identity]) == 1
        }
        excluded_exact_source = old_train_ids - eligible_pool
        if len(eligible_pool) < len(old_validation_ids):
            raise ValueError(
                "Not enough old-training identities remain after exact-source exclusion "
                f"to create validation: pool={len(eligible_pool)} requested={len(old_validation_ids)}"
            )
        validation_n_by_task = Counter(
            f"{metadata[identity]['task_head']}|{metadata[identity]['target_family']}"
            for identity in old_validation_ids
        )
        new_validation_ids, stratum_rows = stratified_quantile_sample(
            eligible_pool,
            metadata=metadata,
            sample_n=len(old_validation_ids),
            sample_n_by_task=dict(validation_n_by_task),
            seed=validation_seed,
            quantile_bins=quantile_bins,
        )
        if len(new_validation_ids) != len(old_validation_ids):
            raise ValueError("New validation size does not match the former validation size.")
        if new_validation_ids & old_validation_ids:
            raise ValueError("New validation reuses a former validation aggregate identity.")
        new_validation_result_ids = {
            result_id
            for identity in new_validation_ids
            for result_id in metadata[identity]["result_ids"]
        }
        result_overlap = new_validation_result_ids & old_validation_result_ids
        if result_overlap:
            raise ValueError(
                "New validation reuses former validation result_ids: "
                f"{sorted(result_overlap)[:5]}"
            )

        raw_stage3_train_ids = stage3_ids - new_validation_ids
        raw_stage3_train_result_ids = {
            result_id
            for identity in raw_stage3_train_ids
            for result_id in metadata[identity]["result_ids"]
        }
        train_valid_aggregate_overlap = raw_stage3_train_ids & new_validation_ids
        train_valid_result_overlap = raw_stage3_train_result_ids & new_validation_result_ids
        if train_valid_aggregate_overlap or train_valid_result_overlap:
            raise ValueError(
                "Stage-3 train/validation exact-source boundary is not disjoint: "
                f"aggregate_overlap={len(train_valid_aggregate_overlap)} "
                f"result_id_overlap={len(train_valid_result_overlap)}"
            )

        expected_effective_population = old_train_ids | old_validation_ids
        new_effective_train_ids = expected_effective_population - new_validation_ids
        if new_effective_train_ids | new_validation_ids != expected_effective_population:
            raise ValueError("New effective stage-3 train/valid does not cover the old population.")
        if new_effective_train_ids & new_validation_ids:
            raise ValueError("New effective stage-3 train/valid aggregate identities overlap.")
        old_train_task_counts = Counter(task_key(metadata[identity]) for identity in old_train_ids)
        new_train_task_counts = Counter(
            task_key(metadata[identity]) for identity in new_effective_train_ids
        )
        new_validation_task_counts = Counter(
            task_key(metadata[identity]) for identity in new_validation_ids
        )
        if new_train_task_counts != old_train_task_counts:
            raise ValueError(
                "Explicit validation changed the task-filter training population: "
                f"old={dict(old_train_task_counts)} new={dict(new_train_task_counts)}"
            )
        if new_validation_task_counts != validation_n_by_task:
            raise ValueError(
                "Explicit validation changed per-task evaluation support: "
                f"expected={dict(validation_n_by_task)} actual={dict(new_validation_task_counts)}"
            )

        existing_screen_validation = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT aggregate_id
                FROM split_assignments
                WHERE split_name = ? AND source_table = ? AND split_part = 'valid'
                """,
                (screen_split, source_table),
            ).fetchall()
        }
        if (
            phase == "final"
            and existing_screen_validation
            and existing_screen_validation != new_validation_ids
        ):
            raise ValueError(
                "The final split's deterministic validation identities differ from "
                "the already-materialized screen split."
            )

        built_phases: list[str] = []
        audit_rows: list[dict[str, Any]] = []
        built_assignment_hashes: dict[str, str] = {}
        if phase == "screen":
            screen_assignments = build_assignments(
                parent,
                split_name=screen_split,
                source_table=source_table,
                validation_seed=validation_seed,
                new_validation_ids=new_validation_ids,
                include_test=False,
            )
            replace_assignments(conn, screen_split, source_table, screen_assignments)
            built_assignment_hashes = assignment_hash_by_part(screen_assignments)
            screen_counts = split_counts(conn, screen_split, source_table)
            if screen_counts.get("test", 0) != 0:
                raise ValueError("The G screen split contains outer-test assignments.")
            built_phases.append("screen")
            audit_rows.append(
                split_audit_row(
                    phase="screen",
                    split_name=screen_split,
                    counts=screen_counts,
                    old_train_n=len(old_train_ids),
                    old_validation_n=len(old_validation_ids),
                    new_validation_ids=new_validation_ids,
                    validation_result_ids=new_validation_result_ids,
                    outer_test_included=False,
                )
            )
        if phase == "final":
            final_assignments = build_assignments(
                parent,
                split_name=final_split,
                source_table=source_table,
                validation_seed=validation_seed,
                new_validation_ids=new_validation_ids,
                include_test=True,
            )
            replace_assignments(conn, final_split, source_table, final_assignments)
            built_assignment_hashes = assignment_hash_by_part(final_assignments)
            final_counts = split_counts(conn, final_split, source_table)
            if final_counts.get("test", 0) != parent_counts["test"]:
                raise ValueError("The G final split does not preserve the parent outer test.")
            built_phases.append("final")
            audit_rows.append(
                split_audit_row(
                    phase="final",
                    split_name=final_split,
                    counts=final_counts,
                    old_train_n=len(old_train_ids),
                    old_validation_n=len(old_validation_ids),
                    new_validation_ids=new_validation_ids,
                    validation_result_ids=new_validation_result_ids,
                    outer_test_included=True,
                )
            )
        built_split_name = screen_split if phase == "screen" else final_split
        assert_split_stage3_exact_coverage(
            conn,
            split_name=built_split_name,
            source_table=source_table,
            expected_record_ids=parent_stage3_record_ids,
            expected_validation_ids=new_validation_ids,
        )
        conn.commit()

    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "schema": "v1_2_43_g_split_v1",
        "db": str(db_path),
        "source_table": source_table,
        "parent_split": parent_split,
        "old_baseline_predictions": str(prediction_path),
        "built_phases": built_phases,
        "built_split": screen_split if phase == "screen" else final_split,
        "screen_split": screen_split,
        "final_split": final_split,
        "validation_seed": validation_seed,
        "quantile_bins": quantile_bins,
        "old_stage3_train_n": len(old_train_ids),
        "old_stage3_validation_n": len(old_validation_ids),
        "new_stage3_validation_n": len(new_validation_ids),
        "new_stage3_training_effective_n": (
            len(old_train_ids) - len(new_validation_ids) + len(old_validation_ids)
        ),
        "excluded_old_train_non_singleton_component_n": len(excluded_exact_source),
        "old_validation_aggregate_id_sha256": stable_hash(old_validation_ids),
        "new_validation_aggregate_id_sha256": stable_hash(new_validation_ids),
        "old_validation_result_ids_sha256": stable_hash(old_validation_result_ids),
        "new_validation_result_ids_sha256": stable_hash(new_validation_result_ids),
        "old_new_validation_aggregate_overlap_n": 0,
        "old_new_validation_result_ids_overlap_n": 0,
        "new_train_validation_aggregate_overlap_n": 0,
        "new_train_validation_result_ids_overlap_n": 0,
        "parent_stage3_raw_n": len(stage3_ids),
        "parent_stage3_aggregate_id_sha256": stable_hash(stage3_ids),
        "parent_stage3_record_id_sha256": stable_hash(parent_stage3_record_ids),
        "parent_stage3_scientific_identity_sha256": canonical_sha256(
            list(parent_scientific_by_id.values())
        ),
        "baseline_stage3_scientific_identity_sha256": canonical_sha256(
            list(baseline_scientific_by_id.values())
        ),
        "effective_stage3_population_sha256": stable_hash(expected_effective_population),
        "new_effective_train_aggregate_id_sha256": stable_hash(new_effective_train_ids),
        "split_assignment_sha256_by_part": built_assignment_hashes,
        "outer_test_assignments_queried": phase == "final",
        "outer_test_counts_reported": phase == "final",
        "winner_lock": "" if winner_lock is None else str(winner_lock),
        "locked_winner": (
            None if winner_lock_payload is None else winner_lock_payload["selected_candidate"]
        ),
        "winner_lock_sha256": (
            "" if winner_lock is None else file_sha256(winner_lock)
        ),
        "strata": stratum_rows,
        "audit_csv": str(audit_csv),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def discover_old_baseline_predictions(root: Path) -> Path:
    matches = [
        path
        for path in root.glob("**/predictions.csv")
        if "_X0_molar_seed42" in "/".join(path.parts)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one v1.2.40 X0_molar seed-42 predictions.csv under "
            f"{root}, found={len(matches)}"
        )
    return matches[0]


def validate_winner_lock_presence(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise ValueError("phase=final requires an existing validation-only winner_lock.json.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "v1_2_43_g_winner_lock_v2":
        raise ValueError("Unsupported or missing G-series winner lock schema.")
    if payload.get("selected_candidate") not in {"G1", "G2", "G3"}:
        raise ValueError("Final split requires a locked non-baseline G-series winner.")
    locked_files = payload.get("locked_files")
    required_locked_files = {
        "screen_seed_metrics",
        "validation_candidate_summary",
        "selected_candidate",
    }
    if not isinstance(locked_files, dict) or set(locked_files) != required_locked_files:
        raise ValueError("Winner lock is missing selection file hashes.")
    for label, item in locked_files.items():
        if not isinstance(item, dict):
            raise ValueError(f"Malformed winner-lock file entry: {label}")
        locked_path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        expected_canonical = str(item.get("canonical_sha256", ""))
        if not locked_path.is_file() or not expected:
            raise ValueError(f"Winner-lock selection file is missing: {locked_path}")
        actual = file_sha256(locked_path)
        if actual != expected:
            raise ValueError(
                f"Winner-lock selection file hash mismatch for {label}: "
                f"expected={expected} actual={actual}"
            )
        if (
            not expected_canonical
            or canonical_file_sha256(locked_path) != expected_canonical
        ):
            raise ValueError(
                f"Winner-lock selection canonical hash mismatch for {label}."
            )
    selected_entry = locked_files.get("selected_candidate")
    if not isinstance(selected_entry, dict):
        raise ValueError("Winner lock is missing selected_candidate.json.")
    selected_payload = json.loads(
        Path(str(selected_entry["path"])).read_text(encoding="utf-8")
    )
    if (
        selected_payload.get("selected_candidate") != payload.get("selected_candidate")
        or selected_payload.get("selection_status") != payload.get("selection_status")
    ):
        raise ValueError("Winner lock differs from selected_candidate.json.")
    canonical_inputs = payload.get("canonical_inputs")
    required_inputs = {
        "fresh_validation_predictions_sha256",
        "validation_metric_rows_sha256",
        "validation_candidate_summary_sha256",
        "selection_sha256",
    }
    if not isinstance(canonical_inputs, dict) or set(canonical_inputs) != required_inputs:
        raise ValueError("Winner lock lacks complete canonical validation-input hashes.")
    if canonical_inputs["selection_sha256"] != canonical_sha256(selected_payload):
        raise ValueError("Winner-lock selection canonical content mismatch.")
    validate_locked_screen_evidence(payload.get("screen_evidence"))
    if (
        canonical_inputs["validation_metric_rows_sha256"]
        != str(locked_files["screen_seed_metrics"].get("canonical_sha256", ""))
        or canonical_inputs["validation_candidate_summary_sha256"]
        != str(
            locked_files["validation_candidate_summary"].get("canonical_sha256", "")
        )
        or canonical_inputs["fresh_validation_predictions_sha256"]
        != payload["screen_evidence"].get("fresh_validation_predictions_sha256")
    ):
        raise ValueError("Winner-lock fresh inputs do not match locked selection evidence.")
    return payload


def validate_locked_screen_evidence(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Winner lock lacks screen split/run artifact evidence.")
    split_entry = value.get("screen_split_summary")
    artifacts = value.get("screen_run_artifacts")
    expected_artifact_hash = str(
        value.get("screen_run_artifacts_canonical_sha256", "")
    )
    fresh_predictions_hash = str(value.get("fresh_validation_predictions_sha256", ""))
    if (
        not isinstance(split_entry, dict)
        or not isinstance(artifacts, list)
        or not artifacts
        or not fresh_predictions_hash
    ):
        raise ValueError("Winner lock has incomplete screen split/run artifact evidence.")
    validate_locked_evidence_file(split_entry, label="screen_split_summary")
    if canonical_sha256(artifacts) != expected_artifact_hash:
        raise ValueError("Winner-lock screen run-artifact canonical hash mismatch.")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("phase") != "screen":
            raise ValueError("Winner lock has malformed screen run-artifact evidence.")
        label = f"{artifact.get('candidate')}:{artifact.get('seed')}"
        validate_locked_evidence_file(artifact.get("manifest"), label=f"manifest:{label}")
        validate_locked_evidence_file(
            artifact.get("predictions"), label=f"predictions:{label}"
        )


def validate_locked_evidence_file(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Malformed winner-lock evidence entry: {label}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    expected_canonical = str(value.get("canonical_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise ValueError(f"Winner-lock evidence file hash mismatch: {label}")
    if not expected_canonical or canonical_file_sha256(path) != expected_canonical:
        raise ValueError(f"Winner-lock evidence canonical hash mismatch: {label}")


def task_key(meta: dict[str, Any]) -> str:
    return f"{meta['task_head']}|{meta['target_family']}"


def aggregate_result_components(
    identities: set[str], *, metadata: dict[str, dict[str, Any]]
) -> dict[str, frozenset[str]]:
    parent = {identity: identity for identity in identities}

    def find(identity: str) -> str:
        root = identity
        while parent[root] != root:
            root = parent[root]
        while parent[identity] != identity:
            next_identity = parent[identity]
            parent[identity] = root
            identity = next_identity
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    first_by_result: dict[str, str] = {}
    for identity in sorted(identities):
        for result_id in metadata[identity]["result_ids"]:
            previous = first_by_result.setdefault(str(result_id), identity)
            union(previous, identity)
    members: dict[str, set[str]] = defaultdict(set)
    for identity in identities:
        members[find(identity)].add(identity)
    return {
        identity: frozenset(members[find(identity)])
        for identity in identities
    }


def assert_parent_stage3_contract(
    stage3_rows: list[sqlite3.Row], *, metadata: dict[str, dict[str, Any]]
) -> None:
    for row in stage3_rows:
        identity = str(row["aggregate_id"])
        meta = metadata[identity]
        expected_record_id = stage_sample_record_id(
            identity,
            "soil",
            TARGET_NAME,
            TARGET_FAMILY,
        )
        if str(row["record_id"]) != expected_record_id:
            raise ValueError(
                f"Parent stage-3 record_id violates the strict soil mol/kg identity: {identity}"
            )
        contract = parse_stage_contract(row["group_key"])
        expected = {
            "medium_domain": "soil",
            "target_name": TARGET_NAME,
            "target_family": TARGET_FAMILY,
        }
        mismatches = {
            key: {"actual": contract.get(key), "expected": value}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Parent stage-3 group contract mismatch for {identity}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        if meta["target_family"] != TARGET_FAMILY:
            raise ValueError(f"Parent stage-3 source target family mismatch: {identity}")


def parent_stage3_scientific_identities(
    stage3_rows: list[sqlite3.Row],
    *,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in stage3_rows:
        identity = str(row["aggregate_id"])
        meta = metadata[identity]
        output[identity] = scientific_identity(
            aggregate_id=identity,
            record_id=str(row["record_id"]),
            result_ids=meta["result_ids"],
            task_head=meta["task_head"],
            target_name=meta["target_name"],
            target_family=meta["target_family"],
            medium_domain=meta["medium_domain"],
            target_value=meta["target_value"],
        )
    return output


def scientific_identity(
    *,
    aggregate_id: Any,
    record_id: Any,
    result_ids: Iterable[Any],
    task_head: Any,
    target_name: Any,
    target_family: Any,
    medium_domain: Any,
    target_value: Any,
) -> dict[str, Any]:
    return {
        "aggregate_id": str(aggregate_id).strip(),
        "record_id": str(record_id).strip(),
        "result_ids": sorted({str(item).strip() for item in result_ids}),
        "task_head": str(task_head).strip(),
        "target_name": str(target_name).strip(),
        "target_family": str(target_family).strip(),
        "medium_domain": str(medium_domain).strip().lower(),
        "target_value_raw": float(target_value),
    }


def stage_sample_record_id(
    aggregate_id: Any,
    medium_domain: Any,
    target_name: Any,
    target_family: Any,
) -> str:
    payload = json.dumps(
        [str(aggregate_id), str(medium_domain), str(target_name), str(target_family)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"stage_sample_v1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def parse_stage_contract(value: Any) -> dict[str, str]:
    text = str(value or "")
    parts = text.split("|")
    if not parts or parts[0] != "stage_contract_v1":
        raise ValueError(f"Parent stage-3 assignment lacks a strict stage contract: {text!r}")
    parsed: dict[str, str] = {}
    for item in parts[1:]:
        key, separator, entry = item.partition("=")
        if not separator or not key or not entry or key in parsed:
            raise ValueError(f"Malformed strict stage contract: {text!r}")
        parsed[key] = entry
    return parsed


def load_old_stage3_identities(
    path: Path,
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    train: set[str] = set()
    validation: set[str] = set()
    scientific_by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            part = str(row.get("split_part", "")).strip().lower()
            if part not in {OLD_TRAIN_PART, OLD_VALIDATION_PART}:
                # Do not access aggregate_id/result_ids for outer-test rows.
                continue
            identity = str(row.get("aggregate_id", "")).strip()
            if not identity:
                raise ValueError("Old baseline stage-3 prediction has an empty aggregate_id.")
            target_name = str(row.get("target_name", "")).strip()
            target_family = str(row.get("target_family", "")).strip()
            medium_domain = str(row.get("medium_domain", "")).strip().lower()
            if (
                target_name != TARGET_NAME
                or target_family != TARGET_FAMILY
                or medium_domain != "soil"
            ):
                raise ValueError(
                    "Old baseline stage-3 prediction violates the locked soil mol/kg target contract: "
                    f"{identity}"
                )
            task_head = str(row.get("task_head", "")).strip()
            if not task_head:
                raise ValueError(
                    "Old baseline stage-3 prediction lacks record/task identity: "
                    f"{identity}"
                )
            # v1.2.40 prediction exports predate the explicit ``record_id``
            # column.  The split-assignment identifier is nevertheless fully
            # deterministic from fields present in every prediction row, so
            # reconstruct it and compare it against the parent split instead
            # of weakening the identity audit.
            exported_record_id = str(row.get("record_id", "")).strip()
            record_id = exported_record_id or stage_sample_record_id(
                identity,
                medium_domain,
                target_name,
                target_family,
            )
            if identity in scientific_by_id:
                raise ValueError(f"Duplicate old baseline stage-3 identity: {identity}")
            try:
                target_value = float(row.get("y_true", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Old baseline stage-3 prediction has invalid y_true: {identity}"
                ) from exc
            result_ids = parse_result_ids(row.get("result_ids"))
            scientific_by_id[identity] = scientific_identity(
                aggregate_id=identity,
                record_id=record_id,
                result_ids=result_ids,
                task_head=task_head,
                target_name=target_name,
                target_family=target_family,
                medium_domain=medium_domain,
                target_value=target_value,
            )
            if part == OLD_TRAIN_PART:
                train.add(identity)
            elif part == OLD_VALIDATION_PART:
                validation.add(identity)
    if not train or not validation:
        raise ValueError(
            "Old baseline predictions are missing stage-3 train/validation identities: "
            f"train={len(train)} validation={len(validation)}"
        )
    return train, validation, scientific_by_id


def load_target_metadata(
    conn: sqlite3.Connection,
    source_table: str,
    identities: set[str],
) -> dict[str, dict[str, Any]]:
    columns = {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{source_table}")').fetchall()
    }
    target_column = (
        "target_value_median" if "target_value_median" in columns else "target_value"
    )
    if target_column not in columns:
        raise ValueError(f"Source table {source_table} has no target value column.")
    if "result_ids" not in columns:
        raise ValueError(
            f"Source table {source_table} lacks required exact-source result_ids."
        )
    rows: list[sqlite3.Row] = []
    ordered_identities = sorted(identities)
    for start in range(0, len(ordered_identities), 800):
        chunk = ordered_identities[start : start + 800]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            conn.execute(
                f'''SELECT aggregate_id, task_head, target_name, target_family,
                           medium_domain, "{target_column}" AS target_value, result_ids
                    FROM "{source_table}"
                    WHERE aggregate_id IN ({placeholders})
                      AND target_name = ? AND target_family = ? AND medium_domain = 'soil'
                    ORDER BY aggregate_id''',
                (*chunk, TARGET_NAME, TARGET_FAMILY),
            ).fetchall()
        )
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row["aggregate_id"])
        if identity not in identities:
            continue
        if identity in metadata:
            raise ValueError(f"Duplicate mol/kg source identity: {identity}")
        metadata[identity] = {
            "task_head": str(row["task_head"] or ""),
            "target_name": str(row["target_name"] or ""),
            "target_family": str(row["target_family"] or ""),
            "medium_domain": str(row["medium_domain"] or "").strip().lower(),
            "target_value": float(row["target_value"]),
            "result_ids": tuple(parse_result_ids(row["result_ids"])),
        }
    return metadata


def stratified_quantile_sample(
    identities: set[str],
    *,
    metadata: dict[str, dict[str, Any]],
    sample_n: int,
    sample_n_by_task: dict[str, int],
    seed: int,
    quantile_bins: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    if not 0 < sample_n < len(identities):
        raise ValueError(
            f"sample_n must be within (0, population): sample_n={sample_n} population={len(identities)}"
        )
    by_task: dict[str, list[str]] = defaultdict(list)
    for identity in identities:
        meta = metadata[identity]
        task = f"{meta['task_head']}|{meta['target_family']}"
        by_task[task].append(identity)
    if sum(int(value) for value in sample_n_by_task.values()) != sample_n:
        raise ValueError("Per-task validation quotas do not sum to sample_n.")
    missing_tasks = sorted(set(sample_n_by_task) - set(by_task))
    if missing_tasks:
        raise ValueError(
            "Former validation tasks are absent from the exact-source-clean old training pool: "
            f"{missing_tasks[:5]}"
        )

    strata: dict[str, list[str]] = defaultdict(list)
    for task, task_ids in sorted(by_task.items()):
        ordered = sorted(
            task_ids,
            key=lambda identity: (float(metadata[identity]["target_value"]), identity),
        )
        for rank, identity in enumerate(ordered):
            quantile = min(quantile_bins - 1, int(rank * quantile_bins / len(ordered)))
            strata[f"{task}|q{quantile:02d}"].append(identity)

    quotas: dict[str, int] = {}
    for task, task_ids in sorted(by_task.items()):
        task_target = int(sample_n_by_task.get(task, 0))
        if not 0 <= task_target < len(task_ids):
            raise ValueError(
                "Invalid per-task validation quota: "
                f"task={task!r} quota={task_target} population={len(task_ids)}"
            )
        task_strata = {
            stratum: stratum_ids
            for stratum, stratum_ids in strata.items()
            if stratum.startswith(f"{task}|q")
        }
        fraction = task_target / len(task_ids)
        remainders: list[tuple[float, str]] = []
        for stratum, stratum_ids in sorted(task_strata.items()):
            exact = len(stratum_ids) * fraction
            quotas[stratum] = int(math.floor(exact))
            remainders.append((exact - quotas[stratum], stratum))
        remaining = task_target - sum(quotas[stratum] for stratum in task_strata)
        for _, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            if quotas[stratum] < len(strata[stratum]):
                quotas[stratum] += 1
                remaining -= 1
        if remaining != 0:
            raise ValueError(
                f"Unable to allocate exact task-stratified validation size for {task}: "
                f"remaining={remaining}"
            )

    selected: set[str] = set()
    audit: list[dict[str, Any]] = []
    for stratum, stratum_ids in sorted(strata.items()):
        ordered = sorted(
            stratum_ids,
            key=lambda identity: stable_token(seed, stratum, identity),
        )
        chosen = ordered[: quotas[stratum]]
        selected.update(chosen)
        audit.append(
            {
                "stratum": stratum,
                "task_validation_quota": sample_n_by_task.get(
                    stratum.rsplit("|q", 1)[0], 0
                ),
                "population_n": len(stratum_ids),
                "validation_n": len(chosen),
            }
        )
    if len(selected) != sample_n:
        raise ValueError(f"Stratified sampling size mismatch: {len(selected)} != {sample_n}")
    return selected, audit


def build_assignments(
    parent: list[sqlite3.Row],
    *,
    split_name: str,
    source_table: str,
    validation_seed: int,
    new_validation_ids: set[str],
    include_test: bool,
) -> list[tuple[str, str, str, str, int, str, str, str]]:
    assignments: list[tuple[str, str, str, str, int, str, str, str]] = []
    for row in parent:
        parent_part = str(row["split_part"])
        if parent_part == "test" and not include_test:
            continue
        identity = str(row["aggregate_id"])
        part = "valid" if parent_part == "finetune_mgkg" and identity in new_validation_ids else parent_part
        group_key = str(row["group_key"] or "")
        if part == "valid":
            group_key = stage_contract(
                stage="v1_2_43_g_stage3_validation",
                medium_domain="soil",
                target_name=TARGET_NAME,
                target_family=TARGET_FAMILY,
            )
        assignments.append(
            (
                split_name,
                str(row["record_id"]),
                identity,
                part,
                validation_seed,
                "v1_2_43_g_fresh_validation",
                source_table,
                group_key,
            )
        )
    return assignments


def replace_assignments(
    conn: sqlite3.Connection,
    split_name: str,
    source_table: str,
    assignments: list[tuple[str, str, str, str, int, str, str, str]],
) -> None:
    conn.execute(
        "DELETE FROM split_assignments WHERE split_name = ? AND source_table = ?",
        (split_name, source_table),
    )
    conn.executemany(
        """
        INSERT INTO split_assignments (
            split_name, record_id, aggregate_id, split_part, seed, split_type,
            source_table, group_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        assignments,
    )
    inserted = conn.execute(
        "SELECT COUNT(*) FROM split_assignments WHERE split_name = ? AND source_table = ?",
        (split_name, source_table),
    ).fetchone()[0]
    if int(inserted) != len(assignments):
        raise ValueError(
            f"G split insert mismatch for {split_name}: expected={len(assignments)} inserted={inserted}"
        )


def assert_split_stage3_exact_coverage(
    conn: sqlite3.Connection,
    *,
    split_name: str,
    source_table: str,
    expected_record_ids: set[str],
    expected_validation_ids: set[str],
) -> None:
    rows = conn.execute(
        """
        SELECT record_id, aggregate_id, split_part
        FROM split_assignments
        WHERE split_name = ? AND source_table = ?
          AND split_part IN ('finetune_mgkg', 'valid')
        """,
        (split_name, source_table),
    ).fetchall()
    actual_record_ids = {str(row["record_id"]) for row in rows}
    if len(rows) != len(actual_record_ids) or actual_record_ids != expected_record_ids:
        raise ValueError(
            "Built G split does not exactly cover the parent stage-3 record population: "
            f"expected={len(expected_record_ids)} actual={len(actual_record_ids)}"
        )
    actual_validation_ids = {
        str(row["aggregate_id"])
        for row in rows
        if str(row["split_part"]) == "valid"
    }
    if actual_validation_ids != expected_validation_ids:
        raise ValueError("Built G split validation identities differ from the locked sample.")


def assignment_hash_by_part(
    assignments: list[tuple[str, str, str, str, int, str, str, str]],
) -> dict[str, str]:
    values: dict[str, list[str]] = defaultdict(list)
    for split_name, record_id, aggregate_id, part, seed, split_type, source_table, group_key in assignments:
        values[part].append(
            json.dumps(
                [
                    split_name,
                    record_id,
                    aggregate_id,
                    part,
                    int(seed),
                    split_type,
                    source_table,
                    group_key,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return {part: stable_hash(tokens) for part, tokens in sorted(values.items())}


def split_counts(conn: sqlite3.Connection, split_name: str, source_table: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT split_part, COUNT(*) AS n
        FROM split_assignments
        WHERE split_name = ? AND source_table = ?
        GROUP BY split_part
        """,
        (split_name, source_table),
    ).fetchall()
    return {str(row["split_part"]): int(row["n"]) for row in rows}


def split_audit_row(
    *,
    phase: str,
    split_name: str,
    counts: dict[str, int],
    old_train_n: int,
    old_validation_n: int,
    new_validation_ids: set[str],
    validation_result_ids: set[str],
    outer_test_included: bool,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "split_name": split_name,
        "stage1_train_raw_n": counts.get("train", 0),
        "stage2_finetune_raw_n": counts.get("finetune", 0),
        "stage3_train_raw_n": counts.get("finetune_mgkg", 0),
        "stage3_valid_raw_n": counts.get("valid", 0),
        "outer_test_raw_n": counts.get("test") if outer_test_included else None,
        "old_stage3_train_effective_n": old_train_n,
        "old_stage3_validation_effective_n": old_validation_n,
        "new_stage3_train_effective_n": old_train_n,
        "new_stage3_validation_effective_n": len(new_validation_ids),
        "outer_test_included": outer_test_included,
        "validation_aggregate_id_sha256": stable_hash(new_validation_ids),
        "validation_result_ids_sha256": stable_hash(validation_result_ids),
    }


def parse_result_ids(value: Any) -> list[str]:
    if value is None:
        raise ValueError("G-series stage-3 result_ids must be present and non-empty.")
    if isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("G-series stage-3 result_ids must be present and non-empty.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("G-series stage-3 result_ids must be valid JSON.") from exc
    if isinstance(parsed, (list, tuple, set)):
        result = sorted({str(item).strip() for item in parsed if str(item).strip()})
        if result:
            return result
        raise ValueError("G-series stage-3 result_ids must contain at least one identity.")
    raise ValueError("G-series stage-3 result_ids must decode to a sequence.")


def stable_token(seed: int, *values: str) -> str:
    payload = "|".join((str(seed), *values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_hash(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_file_sha256(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return canonical_sha256(json.loads(path.read_text(encoding="utf-8-sig")))
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return canonical_sha256(
            [
                {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                }
                for row in rows
            ]
        )
    raise ValueError(f"Unsupported canonical lock file type: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_contract(
    *, stage: str, medium_domain: str, target_name: str, target_family: str
) -> str:
    return "|".join(
        (
            "stage_contract_v1",
            f"stage={stage}",
            f"medium_domain={medium_domain}",
            f"target_name={target_name}",
            f"target_family={target_family}",
        )
    )


def _ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if exists is None:
        raise ValueError(f"Required table does not exist: {table_name}")


if __name__ == "__main__":
    main()
