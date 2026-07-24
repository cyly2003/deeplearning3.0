from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.baseline import (
    load_split_frame,
    stage_sample_record_id,
    task_skip_reason,
)


SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
PARENT_SPLIT = "M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"
M11_SPLIT = "M_v1_2_44_M11_三阶段_固定评价边界"
M00_SPLIT = "M_v1_2_44_M00_仅Stage3从头训练_固定评价边界"
M10_SPLIT = "M_v1_2_44_M10_仅水相预训练_固定评价边界"
M01_SPLIT = "M_v1_2_44_M01_仅土壤pTox_固定评价边界"
ROUTE_SPLITS = {
    "M11": M11_SPLIT,
    "M00": M00_SPLIT,
    "M10": M10_SPLIT,
    "M01": M01_SPLIT,
}
EXPECTED_PARENT_COUNTS = {
    "train": 280_593,
    "finetune": 12_816,
    "finetune_mgkg": 13_943,
    "test": 3_488,
}
# These digests pin the matrix to the exact remote source and assignments that
# produced the completed v1.2.40 molar predictions.  They deliberately exclude
# database path, mtime, and split-assignment created_at.  The source digest was
# additionally verified against all 272,673 rows exported by the completed
# v1.2.40 seed-42 run (zero value or identity mismatches).
EXPECTED_PARENT_ASSIGNMENT_SHA256 = (
    "e6b6b68919146e52c88692c46c52c22f43881bc8ce0c227c4fee952da74ab052"
)
EXPECTED_PARENT_SOURCE_SHA256 = (
    "44e9d602d32c5b80420a475693d93bae92ff5c1365f6544fbebefead281e5514"
)
TASK_FILTER_MIN_TOTAL = 200
TASK_FILTER_MIN_TRAIN = 100
TASK_FILTER_MIN_EVAL = 30
STAGE3_VALIDATION_FRACTION = 0.2
STAGE3_VALIDATION_SEED = 42
EXPECTED_PARENT_SEED = 42
EXPECTED_STAGE3_COUNTS = {"train": 9_724, "validation": 2_433, "test": 3_042}
EXPECTED_STAGE3_TASKS = 18
VALID_PARENT_PARTS = frozenset(EXPECTED_PARENT_COUNTS)
STAGE3_PARTS = frozenset({"finetune_mgkg", "valid", "test"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v1.2.44 strict second-layer routes from the exact v1.2.40 "
            "molar three-stage source and freeze its Stage-3 train/validation/test boundary."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--parent-split", default=PARENT_SPLIT)
    parser.add_argument("--m11-split", default=M11_SPLIT)
    parser.add_argument("--m00-split", default=M00_SPLIT)
    parser.add_argument("--m10-split", default=M10_SPLIT)
    parser.add_argument("--m01-split", default=M01_SPLIT)
    parser.add_argument("--audit-json", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_second_layer_splits(
        db_path=args.db,
        source_table=args.source_table,
        parent_split=args.parent_split,
        route_splits={
            "M11": args.m11_split,
            "M00": args.m00_split,
            "M10": args.m10_split,
            "M01": args.m01_split,
        },
        audit_json=args.audit_json,
        expected_parent_assignment_sha256=EXPECTED_PARENT_ASSIGNMENT_SHA256,
        expected_parent_source_sha256=EXPECTED_PARENT_SOURCE_SHA256,
        expected_parent_counts=EXPECTED_PARENT_COUNTS,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_second_layer_splits(
    *,
    db_path: Path,
    source_table: str = SOURCE_TABLE,
    parent_split: str = PARENT_SPLIT,
    route_splits: Mapping[str, str] | None = None,
    audit_json: Path | None = None,
    expected_parent_assignment_sha256: str | None = EXPECTED_PARENT_ASSIGNMENT_SHA256,
    expected_parent_source_sha256: str | None = EXPECTED_PARENT_SOURCE_SHA256,
    expected_parent_counts: Mapping[str, int] | None = EXPECTED_PARENT_COUNTS,
    task_filter_min_total: int = TASK_FILTER_MIN_TOTAL,
    task_filter_min_train: int = TASK_FILTER_MIN_TRAIN,
    task_filter_min_eval: int = TASK_FILTER_MIN_EVAL,
    validation_fraction: float = STAGE3_VALIDATION_FRACTION,
    validation_seed: int = STAGE3_VALIDATION_SEED,
) -> dict[str, Any]:
    route_splits = dict(route_splits or ROUTE_SPLITS)
    route_order = ("M11", "M00", "M10", "M01")
    if set(route_splits) != set(route_order):
        raise ValueError("route_splits must define exactly M11, M00, M10, and M01.")
    split_names = [parent_split, *route_splits.values()]
    if any(not str(value).strip() for value in split_names) or len(split_names) != len(set(split_names)):
        raise ValueError("Parent and route split names must be non-empty and distinct.")
    if source_table != SOURCE_TABLE:
        raise ValueError(
            f"v1.2.44 is pinned to source_table={SOURCE_TABLE!r}; received {source_table!r}."
        )
    if parent_split != PARENT_SPLIT:
        raise ValueError(
            f"v1.2.44 is pinned to parent_split={PARENT_SPLIT!r}; received {parent_split!r}."
        )
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        assert_table_exists(conn, "split_assignments")
        assert_table_exists(conn, source_table)
        parent_rows = read_assignments(conn, parent_split, source_table)
        parent_audit = audit_parent_assignments(
            parent_rows,
            expected_counts=expected_parent_counts,
            expected_sha256=expected_parent_assignment_sha256,
        )
        source_snapshot_audit = audit_parent_source_snapshot(
            conn,
            parent_split=parent_split,
            source_table=source_table,
            expected_sha256=expected_parent_source_sha256,
        )

    # load_split_frame is the authoritative strict composite-identity join used
    # by training. Reusing it here makes the frozen validation identities match
    # the completed v1.2.40 run rather than reimplementing its row ordering.
    parent_frame = load_split_frame(
        db_path,
        split_name=parent_split,
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    strict_join_audit = audit_parent_source_frame(parent_frame)
    filtered_frame, task_filter_audit = filter_like_deep_experiment(
        parent_frame,
        min_total=task_filter_min_total,
        min_train=task_filter_min_train,
        min_eval=task_filter_min_eval,
    )
    boundary = freeze_stage3_boundary(
        filtered_frame,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
    )
    fixed_validation_record_ids = set(boundary.pop("validation_record_ids"))
    eligible_stage3_record_ids = set(boundary.pop("eligible_stage3_record_ids"))
    eligible_parent_record_ids = {
        frame_record_id(row) for _, row in filtered_frame.iterrows()
    }
    parent_stage2_boundary = freeze_parent_stage2_boundary(filtered_frame)
    parent_stage1_boundary = freeze_parent_stage1_boundary(filtered_frame)

    route_rows = {
        route: build_route_assignments(
            parent_rows,
            route=route,
            split_name=route_splits[route],
            validation_record_ids=fixed_validation_record_ids,
            eligible_stage3_record_ids=eligible_stage3_record_ids,
            eligible_parent_record_ids=eligible_parent_record_ids,
        )
        for route in route_order
    }
    route_audits = {
        route: audit_route_assignments(rows, route=route)
        for route, rows in route_rows.items()
    }
    stage3_hashes = {audit["stage3_boundary_sha256"] for audit in route_audits.values()}
    if stage3_hashes != {boundary["stage3_boundary_sha256"]}:
        raise ValueError(
            "Derived routes do not preserve the exact frozen Stage-3 boundary: "
            f"route_hashes={sorted(stage3_hashes)}, expected={boundary['stage3_boundary_sha256']}"
        )

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            for route in route_order:
                split_name = route_splits[route]
                conn.execute(
                    "DELETE FROM split_assignments WHERE split_name=? AND source_table=?",
                    (split_name, source_table),
                )
                conn.executemany(
                    """
                    INSERT INTO split_assignments (
                        split_name, record_id, aggregate_id, split_part, seed,
                        split_type, source_table, group_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [assignment_tuple(row) for row in route_rows[route]],
                )
                persisted = read_assignments(conn, split_name, source_table)
                persisted_audit = audit_route_assignments(persisted, route=route)
                if persisted_audit != route_audits[route]:
                    raise ValueError(
                        f"Persisted {route} split differs from the in-memory audited route."
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    m10_frame = load_split_frame(
        db_path,
        split_name=route_splits["M10"],
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    m10_exception_audit = audit_m10_training_boundary(
        m10_frame,
        expected_parent_stage1=parent_stage1_boundary,
    )
    m01_frame = load_split_frame(
        db_path,
        split_name=route_splits["M01"],
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    m01_exception_audit = audit_m01_training_boundary(
        m01_frame,
        expected_parent_stage2=parent_stage2_boundary,
        # The route already contains only task heads admitted by the exact
        # v1.2.40 parent filter.  Reapplying 200/100 after removing aquatic
        # Stage 1 would incorrectly discard smaller soil-pTox Stage-2 heads.
        min_total=0,
        min_train=0,
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "matrix_version": "v1.2.44",
        "status": "strict_routes_built",
        "db": str(db_path.resolve()),
        "source_table": source_table,
        "parent_split": parent_split,
        "parent": parent_audit,
        "source_snapshot": {**source_snapshot_audit, **strict_join_audit},
        "task_filter": {
            "min_total": int(task_filter_min_total),
            "min_train": int(task_filter_min_train),
            "min_eval": int(task_filter_min_eval),
            **task_filter_audit,
        },
        "stage3_boundary": boundary,
        "parent_stage2_boundary": parent_stage2_boundary,
        "parent_stage1_boundary": parent_stage1_boundary,
        "m10_task_filter_exception": m10_exception_audit,
        "m01_task_filter_exception": m01_exception_audit,
        "routes": {
            route: {"split_name": route_splits[route], **route_audits[route]}
            for route in route_order
        },
        "stage_semantics": {
            "M11": "aquatic pTox Stage 1 + soil pTox Stage 2 + soil mol/kg Stage 3",
            "M00": "soil mol/kg Stage 3 from random initialization; no Stage 1/2 rows",
            "M10": "aquatic pTox Stage 1 + soil mol/kg Stage 3; no soil pTox rows",
            "M01": "soil pTox from random initialization + soil mol/kg Stage 3; no aquatic rows",
        },
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    if audit_json is not None:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def read_assignments(
    conn: sqlite3.Connection, split_name: str, source_table: str
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
        FROM split_assignments
        WHERE split_name=? AND source_table=?
        ORDER BY rowid
        """,
        (split_name, source_table),
    ).fetchall()
    return [
        {
            "split_name": split_name,
            "record_id": str(row["record_id"] or ""),
            "aggregate_id": str(row["aggregate_id"] or ""),
            "split_part": str(row["split_part"] or ""),
            "seed": int(row["seed"]),
            "split_type": str(row["split_type"] or ""),
            "source_table": str(row["source_table"] or ""),
            "group_key": str(row["group_key"] or ""),
        }
        for row in rows
    ]


def audit_parent_assignments(
    rows: list[dict[str, Any]],
    *,
    expected_counts: Mapping[str, int] | None,
    expected_sha256: str | None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Pinned v1.2.40 parent split is missing.")
    parts = Counter(row["split_part"] for row in rows)
    if set(parts) != VALID_PARENT_PARTS:
        raise ValueError(f"Unexpected v1.2.40 parent split parts: {dict(parts)}")
    if expected_counts is not None and dict(parts) != dict(expected_counts):
        raise ValueError(
            f"v1.2.40 parent assignment counts changed: observed={dict(parts)}, "
            f"expected={dict(expected_counts)}"
        )
    if any(row["seed"] != EXPECTED_PARENT_SEED for row in rows):
        raise ValueError("v1.2.40 parent assignments must all retain split seed 42.")
    if any(not row["record_id"].startswith("stage_sample_v1:") for row in rows):
        raise ValueError("v1.2.40 parent contains a non-strict stage-sample identity.")
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("v1.2.40 parent contains duplicate stage-sample record identities.")
    digest = assignment_sha256(rows)
    enforce_digest("parent assignment", digest, expected_sha256)
    return {
        "rows": len(rows),
        "counts": dict(sorted(parts.items())),
        "assignment_sha256": digest,
        "seed": EXPECTED_PARENT_SEED,
    }


def audit_parent_source_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("v1.2.40 parent strict join returned no source rows.")
    expected_contracts = {
        "train": ("aquatic", "ptox_mol_l", "aquatic_pTox_mol_L"),
        "finetune": ("soil", "ptox_mol_l", "aquatic_pTox_mol_L"),
        "finetune_mgkg": ("soil", "neg_log10_mol_kg", "solid_neglog_mol_kg"),
        "test": ("soil", "neg_log10_mol_kg", "solid_neglog_mol_kg"),
    }
    for part, expected in expected_contracts.items():
        selected = frame.loc[frame["split_part"].astype(str) == part]
        observed = {
            (
                str(row["medium_domain"]),
                str(row["target_name"]),
                str(row["target_family"]),
            )
            for _, row in selected.iterrows()
        }
        if observed != {expected}:
            raise ValueError(
                f"Parent scientific stage contract changed for {part}: "
                f"observed={sorted(observed)}, expected={expected}"
            )
    for value in frame["result_ids"].tolist():
        parse_result_ids(value)
    return {
        "rows": int(len(frame)),
        "strict_join_assignment_rows": int(
            frame.attrs.get("split_join_audit", {}).get("assignment_rows", len(frame))
        ),
    }


def audit_parent_source_snapshot(
    conn: sqlite3.Connection,
    *,
    parent_split: str,
    source_table: str,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Hash the strict parent/source join without SQLite type-coercion assumptions.

    ``split_assignments.aggregate_id`` is text while the pinned source table may
    expose the same identifier as an integer.  SQLite equality joins can treat
    those storage classes differently across query plans.  Reconstructing the
    canonical stage-sample identity in Python mirrors ``load_split_frame`` and
    makes this audit stable across local and remote environments.
    """

    assignment_rows = conn.execute(
        """
        SELECT record_id, aggregate_id, split_part
        FROM split_assignments
        WHERE split_name=? AND source_table=?
        ORDER BY record_id
        """,
        (parent_split, source_table),
    ).fetchall()
    parent_by_record: dict[str, tuple[str, str]] = {}
    for row in assignment_rows:
        record_id = clean_text(row["record_id"])
        if record_id in parent_by_record:
            raise ValueError(f"Pinned parent contains duplicate record_id: {record_id}")
        parent_by_record[record_id] = (
            clean_text(row["aggregate_id"]),
            clean_text(row["split_part"]),
        )

    record_hashes: dict[str, str] = {}
    query = f'''SELECT
            aggregate_id, task_head, medium_domain, target_name, target_family,
            target_value_median, molecular_weight_g_mol_used, result_ids
        FROM "{source_table}"'''
    for row in conn.execute(query):
        record_id = stage_sample_record_id(
            row["aggregate_id"],
            row["medium_domain"],
            row["target_name"],
            row["target_family"],
        )
        parent = parent_by_record.get(record_id)
        if parent is None:
            continue
        parent_aggregate_id, split_part = parent
        source_aggregate_id = clean_text(row["aggregate_id"])
        if source_aggregate_id != parent_aggregate_id:
            raise ValueError(
                "Pinned parent/source aggregate_id changed for "
                f"{record_id}: assignment={parent_aggregate_id!r}, "
                f"source={source_aggregate_id!r}"
            )
        if record_id in record_hashes:
            raise ValueError(
                "Pinned source table resolves one strict stage-sample identity more than once: "
                f"{record_id}"
            )
        record = {
            "record_id": record_id,
            "aggregate_id": source_aggregate_id,
            "split_part": split_part,
            "task_head": clean_text(row["task_head"]),
            "medium_domain": clean_text(row["medium_domain"]),
            "target_name": clean_text(row["target_name"]),
            "target_family": clean_text(row["target_family"]),
            "target_value_median": clean_number(row["target_value_median"]),
            "molecular_weight_g_mol_used": clean_number(
                row["molecular_weight_g_mol_used"]
            ),
            "result_ids": sorted(parse_result_ids(row["result_ids"])),
        }
        record_hashes[record_id] = hashlib.sha256(
            canonical_json(record).encode("utf-8")
        ).hexdigest()

    matched = len(record_hashes)
    expected_rows = len(parent_by_record)
    if matched != expected_rows:
        missing = sorted(set(parent_by_record) - set(record_hashes))[:5]
        raise ValueError(
            "Pinned parent/source strict join is not one-to-one: "
            f"assignments={expected_rows}, matched={matched}, missing_examples={missing}"
        )
    digest = hashlib.sha256()
    for record_id in sorted(record_hashes):
        digest.update(record_id.encode("utf-8"))
        digest.update(b":")
        digest.update(record_hashes[record_id].encode("ascii"))
        digest.update(b"\n")
    observed = digest.hexdigest()
    enforce_digest("parent source", observed, expected_sha256)
    return {
        "scientific_source_sha256": observed,
        "source_rows_hashed": matched,
        "identity_method": "strict_stage_sample_scan_canonical_float_hex_v3",
    }


def filter_like_deep_experiment(
    frame: pd.DataFrame, *, min_total: int, min_train: int, min_eval: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    routed = apply_task_target_routing(frame)
    kept: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for task_head, task_frame in routed.groupby("task_head", dropna=False):
        task_label = "default" if task_head is None else str(task_head)
        filter_frame = task_frame
        if "finetune_mgkg" in set(task_frame["split_part"].astype(str)):
            filter_frame = task_frame.copy()
            filter_frame["split_part"] = filter_frame["split_part"].replace(
                {"finetune_mgkg": "train"}
            )
        reason = task_skip_reason(
            filter_frame,
            min_total=int(min_total),
            min_train=int(min_train),
            min_eval=int(min_eval),
        )
        if reason is None:
            kept.append(task_frame)
        else:
            skipped[task_label] = reason
    if not kept:
        raise ValueError(f"No task heads survive the pinned task filter: {skipped}")
    filtered = pd.concat(kept, ignore_index=True)
    return filtered, {
        "kept_task_heads": sorted(str(value) for value in filtered["task_head"].unique()),
        "kept_task_count": int(filtered["task_head"].nunique()),
        "skipped_tasks": dict(sorted(skipped.items())),
    }


def freeze_stage3_boundary(
    filtered_frame: pd.DataFrame, *, validation_fraction: float, validation_seed: int
) -> dict[str, Any]:
    samples = [
        {"split_part": str(row["split_part"]), "task_head": str(row["task_head"])}
        for _, row in filtered_frame.iterrows()
    ]
    stage3_indices = [
        index
        for index, sample in enumerate(samples)
        if sample["split_part"] == "finetune_mgkg"
    ]
    stage3_train, stage3_validation, validation_source = split_internal_finetune_validation(
        samples,
        finetune_indices=stage3_indices,
        seed=int(validation_seed),
        validation_fraction=float(validation_fraction),
    )
    if validation_source != "internal_finetune_fraction" or not stage3_validation:
        raise ValueError("Failed to reconstruct the v1.2.40 internal Stage-3 validation boundary.")
    test_indices = [
        index for index, sample in enumerate(samples) if sample["split_part"] == "test"
    ]
    validation_record_ids = [frame_record_id(filtered_frame.iloc[index]) for index in stage3_validation]
    train_record_ids = [frame_record_id(filtered_frame.iloc[index]) for index in stage3_train]
    test_record_ids = [frame_record_id(filtered_frame.iloc[index]) for index in test_indices]
    if len(set(validation_record_ids) | set(train_record_ids) | set(test_record_ids)) != (
        len(validation_record_ids) + len(train_record_ids) + len(test_record_ids)
    ):
        raise ValueError("Frozen Stage-3 train/validation/test identities overlap.")

    evaluation_records = []
    for part, indices in (
        ("finetune_mgkg", stage3_train),
        ("finetune_mgkg_validation", stage3_validation),
        ("test", test_indices),
    ):
        evaluation_records.extend(
            evaluation_identity_record(filtered_frame.iloc[index], split_part=part)
            for index in indices
        )
    boundary_rows = [
        {
            "record_id": record_id,
            "split_part": part,
        }
        for part, record_ids in (
            ("finetune_mgkg", train_record_ids),
            ("valid", validation_record_ids),
            ("test", test_record_ids),
        )
        for record_id in record_ids
    ]
    by_part = {
        part: [record for record in evaluation_records if record["split_part"] == part]
        for part in ("finetune_mgkg", "finetune_mgkg_validation", "test")
    }
    observed_counts = {
        "train": len(train_record_ids),
        "validation": len(validation_record_ids),
        "test": len(test_record_ids),
    }
    if observed_counts != EXPECTED_STAGE3_COUNTS:
        raise ValueError(
            "Pinned v1.2.40 effective Stage-3 counts changed: "
            f"observed={observed_counts}, expected={EXPECTED_STAGE3_COUNTS}"
        )
    task_counts = stage3_task_counts(
        filtered_frame, stage3_train, stage3_validation, test_indices
    )
    if len(task_counts) != EXPECTED_STAGE3_TASKS:
        raise ValueError(
            f"Pinned v1.2.40 Stage-3 task count changed: observed={len(task_counts)}, "
            f"expected={EXPECTED_STAGE3_TASKS}"
        )
    return {
        "source": "reconstructed v1.2.40 internal Stage-3 validation",
        "validation_seed": int(validation_seed),
        "validation_fraction": float(validation_fraction),
        "train_rows": len(train_record_ids),
        "validation_rows": len(validation_record_ids),
        "test_rows": len(test_record_ids),
        "train_record_id_sha256": canonical_sha256(sorted(train_record_ids)),
        "validation_record_id_sha256": canonical_sha256(sorted(validation_record_ids)),
        "test_record_id_sha256": canonical_sha256(sorted(test_record_ids)),
        "stage3_boundary_sha256": canonical_sha256(
            sorted(boundary_rows, key=canonical_json)
        ),
        "evaluation_identity": {
            part: {
                "rows": len(records),
                "sha256": canonical_sha256(sorted(records, key=canonical_json)),
            }
            for part, records in by_part.items()
        },
        "task_counts": task_counts,
        "validation_record_ids": validation_record_ids,
        "eligible_stage3_record_ids": sorted(
            set(train_record_ids) | set(validation_record_ids) | set(test_record_ids)
        ),
    }


def build_route_assignments(
    parent_rows: list[dict[str, Any]],
    *,
    route: str,
    split_name: str,
    validation_record_ids: set[str],
    eligible_stage3_record_ids: set[str],
    eligible_parent_record_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for parent in parent_rows:
        parent_part = parent["split_part"]
        if (
            eligible_parent_record_ids is not None
            and parent["record_id"] not in eligible_parent_record_ids
        ):
            continue
        if (
            parent_part in {"finetune_mgkg", "test"}
            and parent["record_id"] not in eligible_stage3_record_ids
        ):
            continue
        if route == "M00" and parent_part in {"train", "finetune"}:
            continue
        if route == "M10" and parent_part == "finetune":
            continue
        if route == "M01" and parent_part == "train":
            continue
        split_part = parent_part
        if route == "M01" and parent_part == "finetune":
            split_part = "train"
        if parent_part == "finetune_mgkg" and parent["record_id"] in validation_record_ids:
            split_part = "valid"
        row = dict(parent)
        row.update(
            {
                "split_name": split_name,
                "split_part": split_part,
                "split_type": f"v1_2_44_second_layer_strict_{route}",
            }
        )
        output.append(row)
    if not output:
        raise ValueError(f"Derived {route} route is empty.")
    return output


def freeze_parent_stage2_boundary(filtered_frame: pd.DataFrame) -> dict[str, Any]:
    samples = [
        {"split_part": str(row["split_part"]), "task_head": str(row["task_head"])}
        for _, row in filtered_frame.iterrows()
    ]
    indices = [
        index for index, sample in enumerate(samples) if sample["split_part"] == "finetune"
    ]
    train, validation, source = split_internal_finetune_validation(
        samples,
        finetune_indices=indices,
        seed=42,
        validation_fraction=0.2,
    )
    if source != "internal_finetune_fraction" or not train or not validation:
        raise ValueError("Failed to reconstruct the parent soil-pTox Stage-2 boundary.")
    train_ids = sorted(frame_record_id(filtered_frame.iloc[index]) for index in train)
    validation_ids = sorted(
        frame_record_id(filtered_frame.iloc[index]) for index in validation
    )
    return {
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_record_id_sha256": canonical_sha256(train_ids),
        "validation_record_id_sha256": canonical_sha256(validation_ids),
        "rng_seed": 17_073,
    }


def freeze_parent_stage1_boundary(filtered_frame: pd.DataFrame) -> dict[str, Any]:
    samples = [
        {"split_part": str(row["split_part"]), "task_head": str(row["task_head"])}
        for _, row in filtered_frame.iterrows()
    ]
    indices = [
        index for index, sample in enumerate(samples) if sample["split_part"] == "train"
    ]
    train, validation, source = split_internal_training_validation(
        samples,
        train_indices=indices,
        seed=42,
        validation_fraction=0.1,
    )
    if source != "internal_train_fraction" or not train or not validation:
        raise ValueError("Failed to reconstruct the parent aquatic-pTox Stage-1 boundary.")
    train_ids = sorted(frame_record_id(filtered_frame.iloc[index]) for index in train)
    validation_ids = sorted(
        frame_record_id(filtered_frame.iloc[index]) for index in validation
    )
    return {
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_record_id_sha256": canonical_sha256(train_ids),
        "validation_record_id_sha256": canonical_sha256(validation_ids),
        "rng_seed": 42,
    }


def audit_m10_training_boundary(
    frame: pd.DataFrame,
    *,
    expected_parent_stage1: Mapping[str, Any],
) -> dict[str, Any]:
    # The exact v1.2.40 task set is already materialized in this route.  Soil
    # pTox is intentionally absent, so reapplying min_eval=30 would remove all
    # aquatic Stage-1 heads even though their internal validation is valid.
    filtered, audit = filter_like_deep_experiment(
        frame,
        min_total=0,
        min_train=0,
        min_eval=0,
    )
    samples = [
        {"split_part": str(row["split_part"]), "task_head": str(row["task_head"])}
        for _, row in filtered.iterrows()
    ]
    indices = [
        index for index, sample in enumerate(samples) if sample["split_part"] == "train"
    ]
    train, validation, source = split_internal_training_validation(
        samples,
        train_indices=indices,
        seed=42,
        validation_fraction=0.1,
    )
    if source != "internal_train_fraction" or not train or not validation:
        raise ValueError("M10 failed to create its aquatic-pTox internal validation boundary.")
    train_ids = sorted(frame_record_id(filtered.iloc[index]) for index in train)
    validation_ids = sorted(frame_record_id(filtered.iloc[index]) for index in validation)
    observed = {
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_record_id_sha256": canonical_sha256(train_ids),
        "validation_record_id_sha256": canonical_sha256(validation_ids),
    }
    expected = {key: expected_parent_stage1[key] for key in observed}
    if observed != expected:
        raise ValueError(
            "M10 aquatic-pTox train/validation identities differ from the parent M11 "
            f"Stage-1 boundary: observed={observed}, expected={expected}"
        )
    return {
        "route": "M10_only",
        "reason": (
            "soil-pTox eval rows are intentionally absent; the exact v1.2.40-admitted "
            "task set is pre-pinned, so thresholds are disabled to preserve every "
            "M11 Stage-1 identity"
        ),
        "task_filter_min_total": 0,
        "task_filter_min_train": 0,
        "task_filter_min_eval": 0,
        "validation_seed": 42,
        "validation_fraction": 0.1,
        **observed,
        "kept_task_count_including_stage3": audit["kept_task_count"],
    }


def audit_m01_training_boundary(
    frame: pd.DataFrame,
    *,
    expected_parent_stage2: Mapping[str, Any],
    min_total: int,
    min_train: int,
) -> dict[str, Any]:
    filtered, audit = filter_like_deep_experiment(
        frame,
        min_total=min_total,
        min_train=min_train,
        min_eval=0,
    )
    samples = [
        {"split_part": str(row["split_part"]), "task_head": str(row["task_head"])}
        for _, row in filtered.iterrows()
    ]
    indices = [index for index, sample in enumerate(samples) if sample["split_part"] == "train"]
    train, validation, source = split_internal_training_validation(
        samples,
        train_indices=indices,
        seed=17_073,
        validation_fraction=0.2,
    )
    if source != "internal_train_fraction" or not train or not validation:
        raise ValueError("M01 failed to create its soil-pTox internal validation boundary.")
    train_ids = sorted(frame_record_id(filtered.iloc[index]) for index in train)
    validation_ids = sorted(frame_record_id(filtered.iloc[index]) for index in validation)
    observed = {
        "train_rows": len(train_ids),
        "validation_rows": len(validation_ids),
        "train_record_id_sha256": canonical_sha256(train_ids),
        "validation_record_id_sha256": canonical_sha256(validation_ids),
    }
    expected = {
        key: expected_parent_stage2[key]
        for key in observed
    }
    if observed != expected:
        raise ValueError(
            "M01 soil-pTox train/validation identities differ from the parent M11 Stage-2 "
            f"boundary: observed={observed}, expected={expected}"
        )
    return {
        "route": "M01_only",
        "reason": (
            "soil-pTox is remapped to train; the exact v1.2.40-admitted task set is "
            "pre-pinned, so thresholds are disabled to preserve every M11 Stage-2 identity"
        ),
        "task_filter_min_total": int(min_total),
        "task_filter_min_train": int(min_train),
        "task_filter_min_eval": 0,
        "validation_seed": 17_073,
        "validation_fraction": 0.2,
        **observed,
        "kept_task_count_including_stage3": audit["kept_task_count"],
    }


def audit_route_assignments(
    rows: list[dict[str, Any]], *, route: str
) -> dict[str, Any]:
    parts = Counter(row["split_part"] for row in rows)
    expected_parts = {
        "M11": {"train", "finetune", "finetune_mgkg", "valid", "test"},
        "M00": {"finetune_mgkg", "valid", "test"},
        "M10": {"train", "finetune_mgkg", "valid", "test"},
        "M01": {"train", "finetune_mgkg", "valid", "test"},
    }[route]
    if set(parts) != expected_parts:
        raise ValueError(f"{route} route has unexpected split parts: {dict(parts)}")
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{route} route contains duplicate strict record identities.")
    if route == "M10" and any("soil_ptox_stage2" in row["group_key"] for row in rows):
        raise ValueError("M10 unexpectedly retains soil pTox Stage-2 rows.")
    if route == "M00" and any(
        part in {"train", "finetune"} for part in (row["split_part"] for row in rows)
    ):
        raise ValueError("M00 unexpectedly retains Stage-1/2 rows.")
    if route == "M01":
        if any("aquatic_ptox_stage1" in row["group_key"] for row in rows):
            raise ValueError("M01 unexpectedly retains aquatic pTox Stage-1 rows.")
        soil_train = [row for row in rows if row["split_part"] == "train"]
        if not soil_train or any("soil_ptox_stage2" not in row["group_key"] for row in soil_train):
            raise ValueError("M01 train rows must be exclusively soil pTox rows.")
    stage3 = [row for row in rows if row["split_part"] in STAGE3_PARTS]
    stage3_records = [
        {"record_id": row["record_id"], "split_part": row["split_part"]}
        for row in stage3
    ]
    return {
        "rows": len(rows),
        "counts": dict(sorted(parts.items())),
        "assignment_sha256": assignment_sha256(rows),
        "stage3_boundary_sha256": canonical_sha256(
            sorted(stage3_records, key=canonical_json)
        ),
    }


def stage3_task_counts(
    frame: pd.DataFrame,
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in sorted(str(value) for value in frame["task_head"].unique()):
        rows.append(
            {
                "task_head": task,
                "train": sum(str(frame.iloc[index]["task_head"]) == task for index in train_indices),
                "validation": sum(
                    str(frame.iloc[index]["task_head"]) == task for index in validation_indices
                ),
                "test": sum(str(frame.iloc[index]["task_head"]) == task for index in test_indices),
            }
        )
    return [row for row in rows if row["train"] or row["validation"] or row["test"]]


def evaluation_identity_record(
    row: Mapping[str, Any], *, split_part: str
) -> dict[str, Any]:
    # Training routes on the target-qualified model head.  In an in-memory
    # routed frame this is also stored in task_head, whereas predictions.csv
    # preserves the base task in task_head and writes the qualified value to
    # model_head.  Prefer model_head so both representations hash identically.
    model_head = clean_text(row.get("model_head")) or clean_text(row.get("task_head"))
    return {
        "aggregate_id": clean_text(row.get("aggregate_id")),
        "result_ids": sorted(parse_result_ids(row.get("result_ids"))),
        "split_part": split_part,
        "task_head": model_head,
        "base_task_head": clean_text(row.get("base_task_head")),
        "target_name": clean_text(row.get("target_name")),
        "target_family": clean_text(row.get("target_family")),
        "medium_domain": clean_text(row.get("medium_domain")),
    }


def frame_record_id(row: Mapping[str, Any]) -> str:
    return stage_sample_record_id(
        row.get("aggregate_id"),
        row.get("medium_domain"),
        row.get("target_name"),
        row.get("target_family"),
    )


def assignment_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    records = [
        {
            "record_id": clean_text(row.get("record_id")),
            "aggregate_id": clean_text(row.get("aggregate_id")),
            "split_part": clean_text(row.get("split_part")),
            "seed": int(row.get("seed", 0)),
            "split_type": clean_text(row.get("split_type")),
            "source_table": clean_text(row.get("source_table")),
            "group_key": clean_text(row.get("group_key")),
        }
        for row in rows
    ]
    return canonical_sha256(records)


def scientific_source_frame_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "aggregate_id",
        "split_part",
        "task_head",
        "medium_domain",
        "target_name",
        "target_family",
        "target_value_median",
        "molecular_weight_g_mol_used",
        "result_ids",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Pinned source frame is missing hash columns: {missing}")
    selected = frame.loc[:, columns].copy()
    selected["record_id"] = [frame_record_id(row) for _, row in frame.iterrows()]
    selected = selected.sort_values("record_id", kind="mergesort")
    digest = hashlib.sha256()
    for row in selected.itertuples(index=False):
        record = {
            "record_id": clean_text(row.record_id),
            "aggregate_id": clean_text(row.aggregate_id),
            "split_part": clean_text(row.split_part),
            "task_head": clean_text(row.task_head),
            "medium_domain": clean_text(row.medium_domain),
            "target_name": clean_text(row.target_name),
            "target_family": clean_text(row.target_family),
            # Python's hexadecimal float representation is exact and stable
            # across pandas/CSV formatter versions on local and remote hosts.
            "target_value_median": clean_number(row.target_value_median),
            "molecular_weight_g_mol_used": clean_number(
                row.molecular_weight_g_mol_used
            ),
            "result_ids": sorted(parse_result_ids(row.result_ids)),
        }
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_result_ids(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"result_ids is not valid JSON: {value!r}") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"result_ids must be a JSON array: {value!r}")
    result = frozenset(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ValueError(f"result_ids must contain at least one non-empty identity: {value!r}")
    return result


def assignment_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["split_name"],
        row["record_id"],
        row["aggregate_id"],
        row["split_part"],
        row["seed"],
        row["split_type"],
        row["source_table"],
        row["group_key"],
    )


def enforce_digest(label: str, observed: str, expected: str | None) -> None:
    if expected in (None, ""):
        return
    if expected == "TO_BE_FILLED":
        raise ValueError(f"{label} digest constant has not been sealed.")
    if observed != expected:
        raise ValueError(f"Pinned {label} SHA-256 changed: observed={observed}, expected={expected}")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def apply_task_target_routing(frame: pd.DataFrame) -> pd.DataFrame:
    if "task_head" not in frame.columns:
        raise ValueError("Task-target routing requires task_head.")
    routed = frame.copy()
    routed["base_task_head"] = routed["task_head"].map(category_value)
    if "target_family" not in routed.columns:
        raise ValueError("Task-target routing requires target_family.")
    routed["model_head"] = (
        routed["base_task_head"]
        + "__"
        + routed["target_family"].map(category_value)
    )
    routed["task_head"] = routed["model_head"]
    return routed


def split_internal_finetune_validation(
    samples: list[dict[str, Any]],
    *,
    finetune_indices: list[int],
    seed: int,
    validation_fraction: float,
) -> tuple[list[int], list[int], str]:
    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(finetune_indices) < 2:
        return list(finetune_indices), [], ""
    rng = np.random.default_rng(int(seed) + 17_031)
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for index in finetune_indices:
        by_task.setdefault(str(samples[index].get("task_head", "")), []).append(index)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(
            len(shuffled) - 1,
            max(1, int(round(len(shuffled) * fraction))),
        )
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    train = [index for index in finetune_indices if index not in validation_set]
    if not train:
        return list(finetune_indices), [], ""
    return train, sorted(validation), "internal_finetune_fraction" if validation else ""


def split_internal_training_validation(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    seed: int,
    validation_fraction: float,
) -> tuple[list[int], list[int], str]:
    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(train_indices) < 2:
        return list(train_indices), [], ""
    rng = np.random.default_rng(int(seed))
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for index in train_indices:
        by_task.setdefault(str(samples[index].get("task_head", "")), []).append(index)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(
            len(shuffled) - 1,
            max(1, int(round(len(shuffled) * fraction))),
        )
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    train = [index for index in train_indices if index not in validation_set]
    return train, sorted(validation), "internal_train_fraction" if validation else ""


def category_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def clean_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number.hex() if math.isfinite(number) else ""


def assert_table_exists(conn: sqlite3.Connection, table_name: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Required table is missing: {table_name}")


if __name__ == "__main__":
    main()
