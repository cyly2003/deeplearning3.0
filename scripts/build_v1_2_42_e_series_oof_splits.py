from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any


TARGET_NAME = "neg_log10_mol_kg"
TARGET_FAMILY = "solid_neglog_mol_kg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired Direct/Transfer OOF folds inside the locked v1.2.40 "
            "stage-3 training population while omitting the outer test set."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--source-table",
        default="aggregated_task_records_ptox_soil_mass_molar_qc",
    )
    parser.add_argument(
        "--parent-transfer-split",
        default="M_v1_2_40_ptox_to_soil_molkg_B_random_8_2",
    )
    parser.add_argument("--direct-prefix", default="M_v1_2_42_e_oof_direct")
    parser.add_argument("--transfer-prefix", default="M_v1_2_42_e_oof_transfer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/audits/v1_2_42_e_series/oof_split_audit.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/audits/v1_2_42_e_series/oof_split_summary.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_oof_splits(
        db_path=args.db,
        source_table=args.source_table,
        parent_transfer_split=args.parent_transfer_split,
        direct_prefix=args.direct_prefix,
        transfer_prefix=args.transfer_prefix,
        folds=args.folds,
        seed=args.seed,
        audit_csv=args.audit_csv,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_oof_splits(
    *,
    db_path: Path,
    source_table: str,
    parent_transfer_split: str,
    direct_prefix: str,
    transfer_prefix: str,
    folds: int,
    seed: int,
    audit_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("OOF construction requires at least two folds.")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_table(conn, source_table)
        _ensure_table(conn, "split_assignments")
        parent = conn.execute(
            """
            SELECT record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
            FROM split_assignments
            WHERE split_name = ? AND source_table = ?
            ORDER BY rowid
            """,
            (parent_transfer_split, source_table),
        ).fetchall()
        if not parent:
            raise ValueError(f"Parent transfer split does not exist: {parent_transfer_split}")
        by_part = Counter(str(row["split_part"]) for row in parent)
        for required in ("train", "finetune", "finetune_mgkg", "test"):
            if by_part[required] <= 0:
                raise ValueError(f"Parent split is missing required part {required!r}.")

        stage3_rows = [row for row in parent if str(row["split_part"]) == "finetune_mgkg"]
        outer_test_ids = {
            str(row["aggregate_id"]) for row in parent if str(row["split_part"]) == "test"
        }
        target_meta = _target_metadata(conn, source_table)
        missing = sorted(
            str(row["aggregate_id"])
            for row in stage3_rows
            if str(row["aggregate_id"]) not in target_meta
        )
        if missing:
            raise ValueError(f"Stage-3 rows are missing paired mol/kg metadata: {missing[:5]}")
        fold_by_id = _stratified_folds(
            stage3_rows,
            target_meta=target_meta,
            folds=folds,
            seed=seed,
        )
        if set(fold_by_id) & outer_test_ids:
            raise ValueError("Outer test identities entered the OOF training population.")
        fold_counts = Counter(fold_by_id.values())
        if set(fold_counts) != set(range(1, folds + 1)):
            raise ValueError(f"One or more OOF folds are empty: {dict(fold_counts)}")

        audit_rows: list[dict[str, Any]] = []
        expected_stage3_ids = {str(row["aggregate_id"]) for row in stage3_rows}
        for fold in range(1, folds + 1):
            direct_name = f"{direct_prefix}_fold{fold}"
            transfer_name = f"{transfer_prefix}_fold{fold}"
            direct_assignments = []
            transfer_assignments = []
            valid_ids = {identity for identity, assigned_fold in fold_by_id.items() if assigned_fold == fold}
            train_ids = expected_stage3_ids - valid_ids
            if not valid_ids or not train_ids or valid_ids & train_ids:
                raise ValueError(f"Invalid OOF train/valid partition for fold {fold}.")

            for row in stage3_rows:
                identity = str(row["aggregate_id"])
                part = "valid" if identity in valid_ids else "train"
                direct_assignments.append(
                    (
                        direct_name,
                        str(row["record_id"]),
                        identity,
                        part,
                        seed,
                        "paired_direct_oof",
                        source_table,
                        f"e_series_direct|fold={fold}|part={part}|target={TARGET_NAME}",
                    )
                )

            for row in parent:
                parent_part = str(row["split_part"])
                if parent_part == "test":
                    continue
                identity = str(row["aggregate_id"])
                if parent_part == "finetune_mgkg":
                    part = "valid" if identity in valid_ids else "finetune_mgkg"
                    group_key = f"e_series_transfer_stage3|fold={fold}|part={part}|target={TARGET_NAME}"
                else:
                    part = parent_part
                    group_key = str(row["group_key"] or "")
                transfer_assignments.append(
                    (
                        transfer_name,
                        str(row["record_id"]),
                        identity,
                        part,
                        seed,
                        "strict_three_stage_molkg_oof",
                        source_table,
                        group_key,
                    )
                )

            _replace_assignments(conn, direct_name, source_table, direct_assignments)
            _replace_assignments(conn, transfer_name, source_table, transfer_assignments)
            direct_counts = _split_counts(conn, direct_name, source_table)
            transfer_counts = _split_counts(conn, transfer_name, source_table)
            if direct_counts.get("valid") != len(valid_ids):
                raise ValueError(f"Direct fold {fold} valid count mismatch.")
            if transfer_counts.get("valid") != len(valid_ids):
                raise ValueError(f"Transfer fold {fold} valid count mismatch.")
            if transfer_counts.get("test", 0) != 0 or direct_counts.get("test", 0) != 0:
                raise ValueError("An outer test assignment was written into an OOF fold.")
            audit_rows.append(
                {
                    "fold": fold,
                    "direct_split": direct_name,
                    "transfer_split": transfer_name,
                    "stage1_train_n": transfer_counts.get("train", 0),
                    "stage2_finetune_n": transfer_counts.get("finetune", 0),
                    "stage3_train_n": transfer_counts.get("finetune_mgkg", 0),
                    "stage3_valid_n": transfer_counts.get("valid", 0),
                    "direct_train_n": direct_counts.get("train", 0),
                    "direct_valid_n": direct_counts.get("valid", 0),
                    "outer_test_assigned_n": 0,
                    "valid_aggregate_id_sha256": _stable_hash(valid_ids),
                }
            )
        conn.commit()

    validation_union = {
        identity
        for identity, assigned_fold in fold_by_id.items()
        if 1 <= assigned_fold <= folds
    }
    if validation_union != expected_stage3_ids:
        raise ValueError("OOF validation folds do not cover every outer-training identity exactly once.")
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "schema": "v1_2_42_e_series_oof_split_v1",
        "db": str(db_path),
        "source_table": source_table,
        "parent_transfer_split": parent_transfer_split,
        "folds": folds,
        "seed": seed,
        "stage1_train_n": by_part["train"],
        "stage2_finetune_n": by_part["finetune"],
        "outer_stage3_train_n": len(stage3_rows),
        "outer_test_n": by_part["test"],
        "outer_test_used_in_oof": False,
        "oof_validation_coverage_n": len(validation_union),
        "oof_validation_coverage_sha256": _stable_hash(validation_union),
        "fold_valid_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        "audit_csv": str(audit_csv),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _target_metadata(conn: sqlite3.Connection, source_table: str) -> dict[str, dict[str, str]]:
    rows = conn.execute(
        f'''SELECT aggregate_id, task_head, target_name, target_family
            FROM "{source_table}"
            WHERE target_name = ? AND target_family = ? AND medium_domain = 'soil'
            ORDER BY aggregate_id''',
        (TARGET_NAME, TARGET_FAMILY),
    ).fetchall()
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        identity = str(row["aggregate_id"])
        if identity in metadata:
            raise ValueError(f"Duplicate paired mol/kg aggregate identity: {identity}")
        metadata[identity] = {
            "task_head": str(row["task_head"] or ""),
            "target_name": str(row["target_name"] or ""),
            "target_family": str(row["target_family"] or ""),
        }
    return metadata


def _stratified_folds(
    stage3_rows: list[sqlite3.Row],
    *,
    target_meta: dict[str, dict[str, str]],
    folds: int,
    seed: int,
) -> dict[str, int]:
    strata: dict[str, list[str]] = defaultdict(list)
    for row in stage3_rows:
        identity = str(row["aggregate_id"])
        meta = target_meta[identity]
        key = f"{meta['task_head']}|{meta['target_family']}"
        strata[key].append(identity)
    assigned: dict[str, int] = {}
    for key, identities in sorted(strata.items()):
        local_seed = seed ^ int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(local_seed)
        shuffled = sorted(identities)
        rng.shuffle(shuffled)
        offset = int(hashlib.sha256(f"offset:{key}".encode("utf-8")).hexdigest()[:8], 16) % folds
        for index, identity in enumerate(shuffled):
            if identity in assigned:
                raise ValueError(f"Stage-3 identity assigned twice: {identity}")
            assigned[identity] = ((index + offset) % folds) + 1
    return assigned


def _replace_assignments(
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
            split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
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
            f"OOF assignment insert mismatch for {split_name}: expected={len(assignments)} inserted={inserted}"
        )


def _split_counts(conn: sqlite3.Connection, split_name: str, source_table: str) -> dict[str, int]:
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


def _ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Required table does not exist: {table_name}")


def _stable_hash(values: Any) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


if __name__ == "__main__":
    main()
