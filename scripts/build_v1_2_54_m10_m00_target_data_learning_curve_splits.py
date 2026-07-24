from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsar_tl.training.baseline import load_split_frame
from scripts.build_v1_2_44_second_layer_splits import (
    M00_SPLIT,
    M10_SPLIT,
    SOURCE_TABLE,
    assignment_sha256,
    assignment_tuple,
    canonical_sha256,
    frame_record_id,
    read_assignments,
)


FRACTIONS = (10, 25, 50, 75)
MODEL_SEEDS = (42, 2042, 3407, 8417)
SUBSET_SEED = 20_260_724
EXPECTED_TASKS = 18
EXPECTED_PARENT_ASSIGNMENT_SHA256 = {
    "M10": "1433a4e78c94e55b55a7a1cab10962498e0d5ab26335ec56308f5161dc6f6f34",
    "M00": "0ff581c18803f2ee24b47bc054e7f1c0fca6a18dfd5bc23e380fbeb5121d7360",
}
EXPECTED_PARENT_COUNTS = {
    "M10": {
        "train": 245_147,
        "finetune_mgkg": 9_724,
        "valid": 2_433,
        "test": 3_042,
    },
    "M00": {
        "finetune_mgkg": 9_724,
        "valid": 2_433,
        "test": 3_042,
    },
}
TARGET_PARTS = ("finetune_mgkg", "valid", "test")
ROUTES = ("M10", "M00")
SPLIT_TEMPLATE = "M_v1_2_54_{route}_LC_F{fraction}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched nested target-data learning-curve splits for M10 and "
            "M00 on the locked v1.2.44 outer test boundary."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--m10-parent", default=M10_SPLIT)
    parser.add_argument("--m00-parent", default=M00_SPLIT)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def stable_rank(record_id: str, *, part: str) -> str:
    payload = f"{SUBSET_SEED}|{part}|{record_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_nested(
    ids_by_task: dict[str, list[str]],
    *,
    fraction: int,
    part: str,
) -> set[str]:
    selected: set[str] = set()
    for task in sorted(ids_by_task):
        ranked = sorted(
            ids_by_task[task],
            key=lambda record_id: (stable_rank(record_id, part=part), record_id),
        )
        n_selected = min(
            len(ranked),
            max(1, int(round(len(ranked) * fraction / 100.0))),
        )
        selected.update(ranked[:n_selected])
    return selected


def clone_assignment(
    parent: dict[str, Any],
    *,
    split_name: str,
    split_part: str,
    fraction: int,
    route: str,
) -> dict[str, Any]:
    row = dict(parent)
    row.update(
        {
            "split_name": split_name,
            "split_part": split_part,
            "seed": SUBSET_SEED,
            "split_type": "v1_2_54_target_data_learning_curve",
            "source_table": SOURCE_TABLE,
            "group_key": (
                f"target_data_fraction={fraction};route={route};"
                f"parent_part={parent['split_part']}"
            ),
        }
    )
    return row


def task_map(frame: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    for _, raw in frame.iterrows():
        record_id = frame_record_id(raw)
        if record_id in output:
            raise ValueError(f"Duplicate strict record identity: {record_id}")
        output[record_id] = str(raw.get("task_head", ""))
    return output


def count_by_task(
    record_ids: Iterable[str],
    *,
    record_to_task: dict[str, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter(record_to_task[record_id] for record_id in record_ids)
    return dict(sorted(counts.items()))


def build_splits(
    *,
    db_path: Path,
    source_table: str,
    m10_parent: str,
    m00_parent: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if source_table != SOURCE_TABLE:
        raise ValueError(f"Expected source table {SOURCE_TABLE!r}")
    if m10_parent != M10_SPLIT:
        raise ValueError("v1.2.54 is pinned to the exact v1.2.44 M10 parent")
    if m00_parent != M00_SPLIT:
        raise ValueError("v1.2.54 is pinned to the exact v1.2.44 M00 parent")
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        m10_rows = read_assignments(conn, m10_parent, source_table)
        m00_rows = read_assignments(conn, m00_parent, source_table)
    if (
        dict(Counter(row["split_part"] for row in m10_rows))
        != EXPECTED_PARENT_COUNTS["M10"]
    ):
        raise ValueError("Locked M10 parent counts changed")
    if (
        dict(Counter(row["split_part"] for row in m00_rows))
        != EXPECTED_PARENT_COUNTS["M00"]
    ):
        raise ValueError("Locked M00 parent counts changed")
    parent_hashes = {
        "M10": assignment_sha256(m10_rows),
        "M00": assignment_sha256(m00_rows),
    }
    if parent_hashes != EXPECTED_PARENT_ASSIGNMENT_SHA256:
        raise ValueError(
            "Locked v1.2.44 parent assignment hash changed: "
            f"observed={parent_hashes}"
        )

    m10_frame = load_split_frame(
        db_path,
        split_name=m10_parent,
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    m10_task = task_map(m10_frame)
    m10_by_id = {row["record_id"]: row for row in m10_rows}
    m00_by_id = {row["record_id"]: row for row in m00_rows}
    if set(m10_task) != set(m10_by_id):
        raise ValueError("Locked M10 assignment/source identity join is not one-to-one")

    m10_target_by_part = {
        part: {
            row["record_id"]
            for row in m10_rows
            if row["split_part"] == part
        }
        for part in TARGET_PARTS
    }
    target_ids = set().union(*m10_target_by_part.values())
    m00_target_by_part = {
        part: {
            row["record_id"]
            for row in m00_rows
            if row["split_part"] == part
        }
        for part in TARGET_PARTS
    }
    if m10_target_by_part != m00_target_by_part:
        raise ValueError("M10 and M00 locked target identities are not exactly matched")
    record_to_task = {record_id: m10_task[record_id] for record_id in target_ids}
    if len(set(record_to_task.values())) != EXPECTED_TASKS:
        raise ValueError("Expected exactly 18 target tasks")

    by_part_task: dict[str, dict[str, list[str]]] = {}
    for part in ("finetune_mgkg", "valid"):
        grouped: dict[str, list[str]] = defaultdict(list)
        for record_id in sorted(m10_target_by_part[part]):
            grouped[record_to_task[record_id]].append(record_id)
        by_part_task[part] = dict(grouped)
    test_ids = set(m10_target_by_part["test"])
    aquatic_ids = {
        row["record_id"] for row in m10_rows if row["split_part"] == "train"
    }

    selected_by_fraction: dict[int, dict[str, set[str]]] = {}
    for fraction in FRACTIONS:
        selected_by_fraction[fraction] = {
            part: select_nested(
                by_part_task[part],
                fraction=fraction,
                part=part,
            )
            for part in ("finetune_mgkg", "valid")
        }
    for lower, upper in zip(FRACTIONS, FRACTIONS[1:]):
        for part in ("finetune_mgkg", "valid"):
            if not selected_by_fraction[lower][part] < selected_by_fraction[upper][part]:
                raise ValueError(
                    f"Nested subset contract failed for {part}: {lower}% !< {upper}%"
                )

    splits: dict[str, list[dict[str, Any]]] = {}
    fraction_audits: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        selected_train = selected_by_fraction[fraction]["finetune_mgkg"]
        selected_valid = selected_by_fraction[fraction]["valid"]
        if selected_train & selected_valid or selected_train & test_ids or selected_valid & test_ids:
            raise ValueError(f"Target subsets overlap at fraction={fraction}")
        target_task_counts = []
        for task in sorted(set(record_to_task.values())):
            target_task_counts.append(
                {
                    "task_head": task,
                    "finetune_mgkg": sum(
                        record_to_task[record_id] == task for record_id in selected_train
                    ),
                    "valid": sum(
                        record_to_task[record_id] == task for record_id in selected_valid
                    ),
                    "test": sum(
                        record_to_task[record_id] == task for record_id in test_ids
                    ),
                }
            )
        if any(
            min(row["finetune_mgkg"], row["valid"], row["test"]) <= 0
            for row in target_task_counts
        ):
            raise ValueError(f"A task is unsupported at fraction={fraction}")

        route_audits: dict[str, Any] = {}
        for route in ROUTES:
            split_name = SPLIT_TEMPLATE.format(route=route, fraction=fraction)
            parent_by_id = m10_by_id if route == "M10" else m00_by_id
            rows: list[dict[str, Any]] = []
            if route == "M10":
                for record_id in sorted(aquatic_ids):
                    rows.append(
                        clone_assignment(
                            parent_by_id[record_id],
                            split_name=split_name,
                            split_part="train",
                            fraction=fraction,
                            route=route,
                        )
                    )
            for part, record_ids in (
                ("finetune_mgkg", selected_train),
                ("valid", selected_valid),
                ("test", test_ids),
            ):
                for record_id in sorted(record_ids):
                    rows.append(
                        clone_assignment(
                            parent_by_id[record_id],
                            split_name=split_name,
                            split_part=part,
                            fraction=fraction,
                            route=route,
                        )
                    )
            splits[split_name] = rows
            route_audits[route] = {
                "split_name": split_name,
                "counts": dict(
                    sorted(Counter(row["split_part"] for row in rows).items())
                ),
                "assignment_sha256": assignment_sha256(rows),
            }
        fraction_audits.append(
            {
                "fraction_percent": fraction,
                "target_train_rows": len(selected_train),
                "target_validation_rows": len(selected_valid),
                "target_test_rows": len(test_ids),
                "target_development_rows": len(selected_train) + len(selected_valid),
                "target_train_record_id_sha256": canonical_sha256(sorted(selected_train)),
                "target_validation_record_id_sha256": canonical_sha256(sorted(selected_valid)),
                "target_test_record_id_sha256": canonical_sha256(sorted(test_ids)),
                "target_train_task_counts": count_by_task(
                    selected_train, record_to_task=record_to_task
                ),
                "target_validation_task_counts": count_by_task(
                    selected_valid, record_to_task=record_to_task
                ),
                "task_counts": target_task_counts,
                "routes": route_audits,
            }
        )

    audit = {
        "schema": "v1_2_54_m10_m00_target_data_learning_curve_split_v1",
        "status": "built_in_memory",
        "scientific_question": (
            "Does aquatic pretraining improve target-data efficiency when M10 and "
            "M00 use exactly matched labeled-soil subsets?"
        ),
        "source_table": source_table,
        "parents": {"M10": m10_parent, "M00": m00_parent},
        "routes": list(ROUTES),
        "model_seeds": list(MODEL_SEEDS),
        "new_split_count": len(ROUTES) * len(FRACTIONS),
        "new_training_cells": len(ROUTES) * len(FRACTIONS) * len(MODEL_SEEDS),
        "reused_100_percent_cells": len(ROUTES) * len(MODEL_SEEDS),
        "full_curve_cells_including_reused": (
            len(ROUTES) * (len(FRACTIONS) + 1) * len(MODEL_SEEDS)
        ),
        "fractions_percent": list(FRACTIONS),
        "full_100_percent_reused": True,
        "subset_seed": SUBSET_SEED,
        "subset_policy": (
            "task-stratified deterministic nested record ranking applied separately "
            "to the locked Stage-3 train and validation sets"
        ),
        "reference_grouping_used": False,
        "scaffold_grouping_used": False,
        "matched_pretraining_contrast_estimable": True,
        "causal_pretraining_gain_claim_supported": False,
        "interpretation_boundary": (
            "Within the locked random 8:2 interpolation boundary, matched M10-M00 "
            "differences estimate the incremental effect of aquatic Stage-1 pretraining; "
            "a positive gain claim requires the completed paired results."
        ),
        "outer_test_rows": len(test_ids),
        "outer_test_record_id_sha256": canonical_sha256(sorted(test_ids)),
        "target_task_count": len(set(record_to_task.values())),
        "aquatic_stage1_rows": len(aquatic_ids),
        "aquatic_stage1_record_id_sha256": canonical_sha256(sorted(aquatic_ids)),
        "parent_assignment_sha256": parent_hashes,
        "fractions": fraction_audits,
    }
    return splits, audit


def persist_or_validate(
    *,
    db_path: Path,
    source_table: str,
    splits: dict[str, list[dict[str, Any]]],
) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        persisted = {
            split_name: read_assignments(conn, split_name, source_table)
            for split_name in splits
        }
        populated = {name for name, rows in persisted.items() if rows}
        if populated:
            if populated != set(splits):
                raise ValueError(
                    "Partial v1.2.54 split state exists; refusing to overwrite: "
                    f"{sorted(populated)}"
                )
            for split_name, expected in splits.items():
                if assignment_sha256(persisted[split_name]) != assignment_sha256(expected):
                    raise ValueError(f"Persisted split differs: {split_name}")
            return "already_valid_no_write"
        try:
            conn.execute("BEGIN IMMEDIATE")
            for split_name, rows in splits.items():
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
    splits, audit = build_splits(
        db_path=args.db,
        source_table=args.source_table,
        m10_parent=args.m10_parent,
        m00_parent=args.m00_parent,
    )
    if args.write:
        audit["persistence"] = persist_or_validate(
            db_path=args.db,
            source_table=args.source_table,
            splits=splits,
        )
        audit["status"] = "persisted_and_validated"
    else:
        audit["persistence"] = "dry_run"
    audit["contract_sha256"] = canonical_sha256(audit)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
