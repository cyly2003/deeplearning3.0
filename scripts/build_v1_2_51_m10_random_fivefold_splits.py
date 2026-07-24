from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsar_tl.training.baseline import load_split_frame
from scripts.build_v1_2_44_second_layer_splits import (
    SOURCE_TABLE,
    assignment_sha256,
    assignment_tuple,
    canonical_sha256,
    frame_record_id,
    read_assignments,
    split_internal_finetune_validation,
)


PARENT_SPLIT = "M_v1_2_44_M10_仅水相预训练_固定评价边界"
FOLD_PREFIX = "M_v1_2_51_M10_随机五折_折"
OUTER_SEED = 42
VALIDATION_SEED = 42
FOLD_COUNT = 5
TARGET_PARENT_PARTS = frozenset({"finetune_mgkg", "valid", "test"})
EXPECTED_PARENT_PART_COUNTS = {
    "train": 245_147,
    "finetune_mgkg": 9_724,
    "valid": 2_433,
    "test": 3_042,
}
EXPECTED_TARGET_ROWS = 15_199
EXPECTED_TARGET_TASKS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build five row-random M10 outer folds from the exact locked v1.2.44 "
            "M10 target pool. Scaffold and chemical grouping are deliberately absent."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--parent-split", default=PARENT_SPLIT)
    parser.add_argument("--fold-prefix", default=FOLD_PREFIX)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist missing folds. Existing identical folds are validated without rewriting.",
    )
    return parser.parse_args()


def clone_assignment(
    parent: dict[str, Any],
    *,
    split_name: str,
    split_part: str,
    group_key: str,
) -> dict[str, Any]:
    row = dict(parent)
    row.update(
        {
            "split_name": split_name,
            "split_part": split_part,
            "seed": OUTER_SEED,
            "split_type": "v1_2_51_m10_row_random_fivefold",
            "source_table": SOURCE_TABLE,
            "group_key": group_key,
        }
    )
    return row


def build_folds(
    *,
    db_path: Path,
    source_table: str,
    parent_split: str,
    fold_prefix: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if source_table != SOURCE_TABLE:
        raise ValueError(f"Expected source table {SOURCE_TABLE!r}; received {source_table!r}")
    if parent_split != PARENT_SPLIT:
        raise ValueError(f"Expected parent split {PARENT_SPLIT!r}; received {parent_split!r}")
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        parent_rows = read_assignments(conn, parent_split, source_table)
    parent_counts = Counter(row["split_part"] for row in parent_rows)
    if dict(parent_counts) != EXPECTED_PARENT_PART_COUNTS:
        raise ValueError(
            "Locked parent M10 counts changed: "
            f"observed={dict(parent_counts)}, expected={EXPECTED_PARENT_PART_COUNTS}"
        )

    frame = load_split_frame(
        db_path,
        split_name=parent_split,
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    frame_by_record: dict[str, dict[str, Any]] = {}
    for _, raw in frame.iterrows():
        record_id = frame_record_id(raw)
        if record_id in frame_by_record:
            raise ValueError(f"Duplicate strict record identity in parent frame: {record_id}")
        frame_by_record[record_id] = raw.to_dict()
    parent_by_record = {row["record_id"]: row for row in parent_rows}
    if set(frame_by_record) != set(parent_by_record):
        raise ValueError("Parent M10 assignment/source strict identities are not one-to-one")

    aquatic_ids = sorted(
        row["record_id"] for row in parent_rows if row["split_part"] == "train"
    )
    target_ids = sorted(
        row["record_id"]
        for row in parent_rows
        if row["split_part"] in TARGET_PARENT_PARTS
    )
    if len(target_ids) != EXPECTED_TARGET_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_ROWS} target rows, observed {len(target_ids)}"
        )

    by_task: dict[str, list[str]] = defaultdict(list)
    for record_id in target_ids:
        row = frame_by_record[record_id]
        contract = (
            str(row.get("medium_domain", "")),
            str(row.get("target_name", "")),
            str(row.get("target_family", "")),
        )
        expected = ("soil", "neg_log10_mol_kg", "solid_neglog_mol_kg")
        if contract != expected:
            raise ValueError(f"Unexpected target contract for {record_id}: {contract}")
        by_task[str(row.get("task_head", ""))].append(record_id)
    if len(by_task) != EXPECTED_TARGET_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_TASKS} target tasks, observed {len(by_task)}"
        )

    rng = np.random.default_rng(OUTER_SEED)
    outer_fold: dict[str, int] = {}
    for task in sorted(by_task):
        identities = sorted(by_task[task])
        rng.shuffle(identities)
        for fold_index, chunk in enumerate(np.array_split(identities, FOLD_COUNT), start=1):
            for record_id in chunk.tolist():
                if record_id in outer_fold:
                    raise ValueError(f"Duplicate outer-fold assignment: {record_id}")
                outer_fold[record_id] = fold_index
    if set(outer_fold) != set(target_ids):
        raise ValueError("Outer-fold assignment does not cover the complete target pool")

    target_index = {record_id: index for index, record_id in enumerate(target_ids)}
    samples = [
        {
            "record_id": record_id,
            "task_head": str(frame_by_record[record_id].get("task_head", "")),
        }
        for record_id in target_ids
    ]
    folds: dict[str, list[dict[str, Any]]] = {}
    fold_audits: list[dict[str, Any]] = []
    test_occurrences: Counter[str] = Counter()
    for fold in range(1, FOLD_COUNT + 1):
        split_name = f"{fold_prefix}{fold}"
        test_ids = {record_id for record_id, assigned in outer_fold.items() if assigned == fold}
        remaining_indices = [
            target_index[record_id] for record_id in target_ids if record_id not in test_ids
        ]
        train_indices, validation_indices, validation_source = (
            split_internal_finetune_validation(
                samples,
                finetune_indices=remaining_indices,
                seed=VALIDATION_SEED,
                validation_fraction=0.2,
            )
        )
        if validation_source != "internal_finetune_fraction":
            raise ValueError(f"Fold {fold} failed to construct target validation")
        train_ids = {target_ids[index] for index in train_indices}
        valid_ids = {target_ids[index] for index in validation_indices}
        if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
            raise ValueError(f"Fold {fold} target train/validation/test identities overlap")
        if train_ids | valid_ids | test_ids != set(target_ids):
            raise ValueError(f"Fold {fold} does not cover the complete target pool")
        test_occurrences.update(test_ids)

        rows: list[dict[str, Any]] = []
        for record_id in aquatic_ids:
            rows.append(
                clone_assignment(
                    parent_by_record[record_id],
                    split_name=split_name,
                    split_part="train",
                    group_key="fixed_aquatic_stage1",
                )
            )
        for record_id in target_ids:
            if record_id in test_ids:
                part = "test"
            elif record_id in valid_ids:
                part = "valid"
            else:
                part = "finetune_mgkg"
            rows.append(
                clone_assignment(
                    parent_by_record[record_id],
                    split_name=split_name,
                    split_part=part,
                    group_key=f"row_random_outer_fold={outer_fold[record_id]}",
                )
            )

        counts = Counter(row["split_part"] for row in rows)
        target_task_counts = []
        for task in sorted(by_task):
            task_ids = set(by_task[task])
            target_task_counts.append(
                {
                    "task_head": task,
                    "finetune_mgkg": len(task_ids & train_ids),
                    "valid": len(task_ids & valid_ids),
                    "test": len(task_ids & test_ids),
                }
            )
        if any(
            min(row["finetune_mgkg"], row["valid"], row["test"]) <= 0
            for row in target_task_counts
        ):
            raise ValueError(f"Fold {fold} leaves a target task unsupported")
        folds[split_name] = rows
        fold_audits.append(
            {
                "fold": fold,
                "split_name": split_name,
                "assignment_rows": len(rows),
                "counts": dict(sorted(counts.items())),
                "target_train_rows": len(train_ids),
                "target_validation_rows": len(valid_ids),
                "target_test_rows": len(test_ids),
                "target_train_record_id_sha256": canonical_sha256(sorted(train_ids)),
                "target_validation_record_id_sha256": canonical_sha256(sorted(valid_ids)),
                "target_test_record_id_sha256": canonical_sha256(sorted(test_ids)),
                "assignment_sha256": assignment_sha256(rows),
                "task_counts": target_task_counts,
            }
        )

    invalid_oof = [
        record_id for record_id in target_ids if test_occurrences[record_id] != 1
    ]
    if invalid_oof:
        raise ValueError(
            f"Fivefold OOF coverage is not exactly once for {len(invalid_oof)} target rows"
        )
    audit = {
        "schema": "v1_2_51_m10_row_random_fivefold_split_v1",
        "status": "built_in_memory",
        "source_table": source_table,
        "parent_split": parent_split,
        "split_policy": "row-random task-stratified fivefold OOF; no scaffold grouping",
        "is_scaffold_split": False,
        "outer_seed": OUTER_SEED,
        "validation_seed": VALIDATION_SEED,
        "fold_count": FOLD_COUNT,
        "parent_assignment_sha256": assignment_sha256(parent_rows),
        "aquatic_stage1_rows": len(aquatic_ids),
        "aquatic_stage1_record_id_sha256": canonical_sha256(aquatic_ids),
        "target_pool_rows": len(target_ids),
        "target_pool_record_id_sha256": canonical_sha256(target_ids),
        "target_task_count": len(by_task),
        "oof_test_coverage": "each target row occurs in exactly one test fold",
        "folds": fold_audits,
    }
    audit["contract_sha256"] = canonical_sha256(audit)
    return folds, audit


def persist_or_validate(
    *,
    db_path: Path,
    source_table: str,
    folds: dict[str, list[dict[str, Any]]],
) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        persisted = {
            split_name: read_assignments(conn, split_name, source_table)
            for split_name in folds
        }
        populated = {name for name, rows in persisted.items() if rows}
        if populated:
            if populated != set(folds):
                raise ValueError(
                    "Partial v1.2.51 fold state exists; refusing to overwrite. "
                    f"populated={sorted(populated)}"
                )
            for split_name, expected in folds.items():
                if assignment_sha256(persisted[split_name]) != assignment_sha256(expected):
                    raise ValueError(
                        f"Existing split differs from deterministic contract: {split_name}"
                    )
            return "already_valid_no_write"

        try:
            conn.execute("BEGIN IMMEDIATE")
            for split_name, rows in folds.items():
                conn.executemany(
                    """
                    INSERT INTO split_assignments (
                        split_name, record_id, aggregate_id, split_part, seed,
                        split_type, source_table, group_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [assignment_tuple(row) for row in rows],
                )
                observed = read_assignments(conn, split_name, source_table)
                if assignment_sha256(observed) != assignment_sha256(rows):
                    raise ValueError(f"Persisted split audit failed: {split_name}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return "written"


def main() -> None:
    args = parse_args()
    folds, audit = build_folds(
        db_path=args.db,
        source_table=args.source_table,
        parent_split=args.parent_split,
        fold_prefix=args.fold_prefix,
    )
    if args.write:
        audit["persistence"] = persist_or_validate(
            db_path=args.db,
            source_table=args.source_table,
            folds=folds,
        )
        audit["status"] = "persisted_and_validated"
    else:
        audit["persistence"] = "dry_run"
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
