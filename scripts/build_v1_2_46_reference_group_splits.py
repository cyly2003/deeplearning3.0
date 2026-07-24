from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.baseline import load_split_frame
from scripts.build_v1_2_44_second_layer_splits import (
    SOURCE_TABLE,
    apply_task_target_routing,
    assignment_sha256,
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
    frame_record_id,
    parse_result_ids,
    read_assignments,
)


PARENT_SPLIT = "M_v1_2_44_M11_三阶段_固定评价边界"
M00_SPLIT = "M_v1_2_46_RG_M00_从头训练_reference_group"
M10_SPLIT = "M_v1_2_46_RG_M10_水相预训练_reference_group"
M11U_SPLIT = "M_v1_2_46_RG_M11U_完整三阶段_reference_group"
ROUTE_SPLITS = {"M00": M00_SPLIT, "M10": M10_SPLIT, "M11U": M11U_SPLIT}

EXPECTED_PARENT_COUNTS = {
    "train": 245_147,
    "finetune": 12_327,
    "finetune_mgkg": 9_724,
    "valid": 2_433,
    "test": 3_042,
}
TARGET_PARTS = frozenset({"finetune_mgkg", "valid", "test"})
SOURCE_PARTS = frozenset({"train", "finetune"})
PART_RATIOS = {"finetune_mgkg": 0.64, "valid": 0.16, "test": 0.20}
BASE_SPLIT_SEED = 20_260_720
DEFAULT_CANDIDATES = 4_096
MIN_TASK_TRAIN_ROWS = 100
MIN_TASK_VALID_ROWS = 1
MIN_TASK_TEST_ROWS = 30
MIN_TASK_VALID_GROUPS = 1
MIN_TASK_TEST_GROUPS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v1.2.46 reference-group M00/M10/M11U matrix from the "
            "locked v1.2.44 M11 route without reading model predictions."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--parent-split", default=PARENT_SPLIT)
    parser.add_argument("--m00-split", default=M00_SPLIT)
    parser.add_argument("--m10-split", default=M10_SPLIT)
    parser.add_argument("--m11u-split", default=M11U_SPLIT)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--base-seed", type=int, default=BASE_SPLIT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    route_splits = {
        "M00": args.m00_split,
        "M10": args.m10_split,
        "M11U": args.m11u_split,
    }
    if args.validate_only:
        result = validate_existing_split_state(
            db_path=args.db,
            source_table=args.source_table,
            audit_json=args.audit_json,
            route_splits=route_splits,
        )
    else:
        result = build_reference_group_splits(
            db_path=args.db,
            source_table=args.source_table,
            parent_split=args.parent_split,
            route_splits=route_splits,
            audit_json=args.audit_json,
            candidate_count=args.candidate_count,
            base_seed=args.base_seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_reference_group_splits(
    *,
    db_path: Path,
    source_table: str = SOURCE_TABLE,
    parent_split: str = PARENT_SPLIT,
    route_splits: Mapping[str, str] | None = None,
    audit_json: Path | None = None,
    candidate_count: int = DEFAULT_CANDIDATES,
    base_seed: int = BASE_SPLIT_SEED,
) -> dict[str, Any]:
    route_splits = dict(route_splits or ROUTE_SPLITS)
    if set(route_splits) != set(ROUTE_SPLITS):
        raise ValueError("route_splits must define exactly M00, M10, and M11U.")
    if len(set(route_splits.values())) != len(route_splits):
        raise ValueError("Reference-group split names must be unique.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        parent_rows = read_assignments(conn, parent_split, source_table)
        parent_counts = Counter(row["split_part"] for row in parent_rows)
        if dict(parent_counts) != EXPECTED_PARENT_COUNTS:
            raise ValueError(
                "Locked v1.2.44 M11 assignment counts changed: "
                f"observed={dict(parent_counts)}, expected={EXPECTED_PARENT_COUNTS}"
            )
        parent_assignment_sha256 = assignment_sha256(parent_rows)

    frame = load_split_frame(
        db_path,
        split_name=parent_split,
        source_table=source_table,
        allow_mixed_target_dimensions=True,
    )
    if len(frame) != len(parent_rows):
        raise ValueError(
            "Locked parent strict join changed row count: "
            f"assignments={len(parent_rows)}, source_rows={len(frame)}"
        )
    frame = frame.copy()
    frame["_record_id"] = [frame_record_id(row) for _, row in frame.iterrows()]
    if frame["_record_id"].duplicated().any():
        raise ValueError("Locked v1.2.44 M11 source has duplicate stage-sample identities.")
    by_record = {str(row["_record_id"]): row for _, row in frame.iterrows()}
    if set(by_record) != {row["record_id"] for row in parent_rows}:
        raise ValueError("Parent assignment and source-frame record identities differ.")

    target = frame.loc[frame["split_part"].astype(str).isin(TARGET_PARTS)].copy()
    source = frame.loc[frame["split_part"].astype(str).isin(SOURCE_PARTS)].copy()
    if len(target) != sum(EXPECTED_PARENT_COUNTS[part] for part in TARGET_PARTS):
        raise ValueError("Reference-group target pool does not cover the locked Stage-3 population.")
    if int(target["task_head"].nunique()) != 18:
        raise ValueError(
            f"Expected 18 locked Stage-3 tasks, observed {target['task_head'].nunique()}."
        )

    target_records = dataframe_records(target)
    components = build_reference_components(target_records)
    assignment, selection = choose_reference_partition(
        target_records,
        group_by_record=components["group_by_record"],
        candidate_count=candidate_count,
        base_seed=base_seed,
    )
    target_audit = audit_target_partition(
        target_records,
        assignment=assignment,
        group_by_record=components["group_by_record"],
        references_by_record=components["references_by_record"],
    )

    heldout_references = set(target_audit["reference_ids"]["valid"]) | set(
        target_audit["reference_ids"]["test"]
    )
    source_records = dataframe_records(source)
    kept_source_ids: set[str] = set()
    excluded_source_ids: set[str] = set()
    source_exclusion_reasons: Counter[str] = Counter()
    for record in source_records:
        refs = parse_string_set(record.get("reference_numbers"), field="reference_numbers")
        if not refs:
            excluded_source_ids.add(record["record_id"])
            source_exclusion_reasons["missing_reference"] += 1
        elif refs & heldout_references:
            excluded_source_ids.add(record["record_id"])
            if refs & set(target_audit["reference_ids"]["test"]):
                source_exclusion_reasons["outer_test_reference"] += 1
            if refs & set(target_audit["reference_ids"]["valid"]):
                source_exclusion_reasons["validation_reference"] += 1
        else:
            kept_source_ids.add(record["record_id"])

    source_by_part = Counter(
        record["split_part"] for record in source_records if record["record_id"] in kept_source_ids
    )
    if not source_by_part.get("train") or not source_by_part.get("finetune"):
        raise ValueError(
            "Reference exclusion removed an entire source stage: " f"remaining={dict(source_by_part)}"
        )

    parent_by_record = {row["record_id"]: row for row in parent_rows}
    route_assignments: dict[str, list[dict[str, Any]]] = {}
    for route, split_name in route_splits.items():
        rows = build_route_assignments(
            parent_by_record=parent_by_record,
            route=route,
            split_name=split_name,
            selected_seed=int(selection["selected_seed"]),
            target_assignment=assignment,
            target_group_by_record=components["group_by_record"],
            kept_source_ids=kept_source_ids,
        )
        route_assignments[route] = rows

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        try:
            for route, split_name in route_splits.items():
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
                    [assignment_tuple(row) for row in route_assignments[route]],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    routed_target = apply_task_target_routing(target)
    target_by_record = {
        str(row["_record_id"]): row for _, row in routed_target.iterrows()
    }
    evaluation_identity: dict[str, dict[str, Any]] = {}
    for part in ("finetune_mgkg", "valid", "test"):
        output_part = "finetune_mgkg_validation" if part == "valid" else part
        records = [
            evaluation_identity_record(target_by_record[record_id], split_part=output_part)
            for record_id, assigned in assignment.items()
            if assigned == part
        ]
        evaluation_identity[output_part] = {
            "rows": len(records),
            "sha256": canonical_sha256(sorted(records, key=canonical_json)),
        }

    route_audits = {}
    for route, rows in route_assignments.items():
        route_audits[route] = {
            "split_name": route_splits[route],
            "counts": dict(sorted(Counter(row["split_part"] for row in rows).items())),
            "rows": len(rows),
            "assignment_sha256": assignment_sha256(rows),
        }

    summary: dict[str, Any] = {
        "schema": "v1_2_46_reference_group_split_v1",
        "status": "locked",
        "source_table": source_table,
        "parent_split": parent_split,
        "parent_assignment_sha256": parent_assignment_sha256,
        "parent_counts": dict(EXPECTED_PARENT_COUNTS),
        "partition_policy": {
            "unit": "connected_component_of_reference_numbers_within_locked_stage3_pool",
            "ratios": dict(PART_RATIOS),
            "candidate_selection": (
                "prediction_blind_minimum_balance_score_over_task_and_within_task_target_quintile"
            ),
            "base_seed": int(base_seed),
            "candidate_count": int(candidate_count),
            "selected_seed": int(selection["selected_seed"]),
            "selection_score": float(selection["score"]),
            "minimum_support": {
                "train_rows_per_task": MIN_TASK_TRAIN_ROWS,
                "validation_rows_per_task": MIN_TASK_VALID_ROWS,
                "test_rows_per_task": MIN_TASK_TEST_ROWS,
                "validation_groups_per_task": MIN_TASK_VALID_GROUPS,
                "test_groups_per_task": MIN_TASK_TEST_GROUPS,
            },
            "uses_model_predictions": False,
        },
        "target_pool": {
            "rows": len(target_records),
            "tasks": int(target["task_head"].nunique()),
            "reference_components": int(components["component_count"]),
            "reference_ids": int(components["reference_count"]),
            "multi_reference_rows": int(components["multi_reference_rows"]),
            "missing_reference_rows": int(components["missing_reference_rows"]),
            "counts": target_audit["counts"],
            "group_counts": target_audit["group_counts"],
            "reference_counts": target_audit["reference_counts"],
            "task_counts": target_audit["task_counts"],
            "identity_overlap": target_audit["identity_overlap"],
            "reference_overlap": target_audit["reference_overlap"],
            "evaluation_identity": evaluation_identity,
        },
        "source_reference_exclusion": {
            "source_rows_before": len(source_records),
            "source_rows_after": len(kept_source_ids),
            "excluded_rows": len(excluded_source_ids),
            "remaining_by_part": dict(sorted(source_by_part.items())),
            "reason_counts": dict(sorted(source_exclusion_reasons.items())),
            "heldout_validation_reference_count": len(
                target_audit["reference_ids"]["valid"]
            ),
            "heldout_test_reference_count": len(target_audit["reference_ids"]["test"]),
            "policy": (
                "exclude every Stage1/Stage2 aggregate containing a validation/test reference; "
                "exclude source aggregates with missing reference identity"
            ),
        },
        "routes": route_audits,
        "route_semantics": {
            "M00": "reference-group soil mol/kg Stage3 from random initialization",
            "M10": "reference-filtered aquatic Stage1 plus reference-group soil mol/kg Stage3",
            "M11U": (
                "reference-filtered aquatic Stage1 plus reference-filtered soil pTox Stage2 "
                "plus reference-group soil mol/kg Stage3"
            ),
        },
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    if audit_json is not None:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def validate_existing_split_state(
    *,
    db_path: Path,
    source_table: str,
    audit_json: Path,
    route_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    route_splits = dict(route_splits or ROUTE_SPLITS)
    summary = json.loads(audit_json.read_text(encoding="utf-8"))
    if summary.get("schema") != "v1_2_46_reference_group_split_v1":
        raise ValueError("Unexpected v1.2.46 split-summary schema.")
    expected_contract = summary.get("contract_sha256")
    observed_contract = canonical_sha256(
        {key: value for key, value in summary.items() if key != "contract_sha256"}
    )
    if expected_contract != observed_contract:
        raise ValueError("v1.2.46 split-summary contract hash changed.")
    if summary.get("source_table") != source_table:
        raise ValueError("v1.2.46 split-summary source table changed.")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        parent_rows = read_assignments(conn, str(summary["parent_split"]), source_table)
        if assignment_sha256(parent_rows) != summary["parent_assignment_sha256"]:
            raise ValueError("Locked v1.2.44 parent assignments changed after v1.2.46 build.")
        validated = {}
        for route, split_name in route_splits.items():
            if split_name != summary["routes"][route]["split_name"]:
                raise ValueError(f"Configured {route} split name differs from the locked summary.")
            rows = read_assignments(conn, split_name, source_table)
            observed = assignment_sha256(rows)
            expected = summary["routes"][route]["assignment_sha256"]
            if observed != expected:
                raise ValueError(f"Persisted {route} assignments changed after lock.")
            validated[route] = {"rows": len(rows), "assignment_sha256": observed}
    overlap = summary["target_pool"]["reference_overlap"]
    if any(int(value) != 0 for value in overlap.values()):
        raise ValueError(f"Locked reference partition contains overlap: {overlap}")
    return {
        "status": "ok",
        "contract_sha256": expected_contract,
        "selected_seed": summary["partition_policy"]["selected_seed"],
        "routes": validated,
    }


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for _, row in frame.iterrows():
        records.append(
            {
                "record_id": str(row["_record_id"]),
                "aggregate_id": str(row["aggregate_id"]),
                "split_part": str(row["split_part"]),
                "task_head": str(row["task_head"]),
                "target_value": float(row["target_value_median"]),
                "reference_numbers": row.get("reference_numbers"),
                "test_ids": row.get("test_ids"),
                "result_ids": row.get("result_ids"),
            }
        )
    return records


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        keep, merge = sorted((root_left, root_right))
        self.parent[merge] = keep


def build_reference_components(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    union = UnionFind()
    references_by_record: dict[str, frozenset[str]] = {}
    multi_reference_rows = 0
    missing_reference_rows = 0
    for row in rows:
        record_id = str(row["record_id"])
        refs = parse_string_set(row.get("reference_numbers"), field="reference_numbers")
        if not refs:
            missing_reference_rows += 1
            continue
        references_by_record[record_id] = refs
        ordered = sorted(refs)
        for ref in ordered:
            union.add(ref)
        for ref in ordered[1:]:
            union.union(ordered[0], ref)
        if len(ordered) > 1:
            multi_reference_rows += 1
    if missing_reference_rows:
        raise ValueError(
            "Reference-group Stage-3 pool contains records without reference_numbers: "
            f"missing_rows={missing_reference_rows}."
        )

    refs_by_root: dict[str, set[str]] = defaultdict(set)
    for ref in union.parent:
        refs_by_root[union.find(ref)].add(ref)
    key_by_root = {
        root: "reference_component_v1:"
        + hashlib.sha256("\n".join(sorted(refs)).encode("utf-8")).hexdigest()
        for root, refs in refs_by_root.items()
    }
    group_by_record: dict[str, str] = {}
    for record_id, refs in references_by_record.items():
        roots = {union.find(ref) for ref in refs}
        if len(roots) != 1:
            raise ValueError(f"Reference component resolution failed for {record_id}.")
        group_by_record[record_id] = key_by_root[next(iter(roots))]
    return {
        "group_by_record": group_by_record,
        "references_by_record": references_by_record,
        "component_count": len(set(group_by_record.values())),
        "reference_count": len(union.parent),
        "multi_reference_rows": multi_reference_rows,
        "missing_reference_rows": missing_reference_rows,
    }


def choose_reference_partition(
    records: list[dict[str, Any]],
    *,
    group_by_record: Mapping[str, str],
    candidate_count: int,
    base_seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    enriched = attach_target_quantile_bins(records)
    groups = sorted(set(group_by_record.values()))
    best: tuple[tuple[int, float, int], dict[str, str], dict[str, Any]] | None = None
    feasible_count = 0
    for offset in range(candidate_count):
        seed = int(base_seed + offset)
        group_part = {group: hash_partition(seed, group) for group in groups}
        assignment = {
            row["record_id"]: group_part[group_by_record[row["record_id"]]]
            for row in enriched
        }
        support = partition_support(
            enriched, assignment=assignment, group_by_record=group_by_record
        )
        violations = count_support_violations(support)
        if violations == 0:
            feasible_count += 1
        score = balance_score(enriched, assignment=assignment)
        key = (violations, score, seed)
        if best is None or key < best[0]:
            best = (
                key,
                assignment,
                {
                    "selected_seed": seed,
                    "score": score,
                    "violations": violations,
                    "support": support,
                },
            )
    if best is None:
        raise ValueError("No reference-group split candidate was evaluated.")
    if best[2]["violations"] != 0:
        raise ValueError(
            "No reference-group candidate satisfied the pre-registered support gate; "
            f"best_seed={best[2]['selected_seed']} support={best[2]['support']}"
        )
    best[2]["feasible_candidate_count"] = feasible_count
    return best[1], best[2]


def attach_target_quantile_bins(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    frame["target_bin"] = "Q1"
    for _, indices in frame.groupby("task_head", sort=True).groups.items():
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: (
                float(frame.at[index, "target_value"]),
                str(frame.at[index, "record_id"]),
            ),
        )
        bins = min(5, len(ordered))
        for position, index in enumerate(ordered):
            bin_index = min(bins - 1, int(position * bins / len(ordered)))
            frame.at[index, "target_bin"] = f"Q{bin_index + 1}"
    return frame.to_dict(orient="records")


def hash_partition(seed: int, group: str) -> str:
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < PART_RATIOS["finetune_mgkg"]:
        return "finetune_mgkg"
    if value < PART_RATIOS["finetune_mgkg"] + PART_RATIOS["valid"]:
        return "valid"
    return "test"


def partition_support(
    records: Iterable[Mapping[str, Any]],
    *,
    assignment: Mapping[str, str],
    group_by_record: Mapping[str, str],
) -> dict[str, Any]:
    rows = list(records)
    counts: Counter[tuple[str, str]] = Counter()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        task = str(row["task_head"])
        part = assignment[str(row["record_id"])]
        counts[(task, part)] += 1
        groups[(task, part)].add(group_by_record[str(row["record_id"])])
    output = {}
    for task in sorted({str(row["task_head"]) for row in rows}):
        output[task] = {
            part: {"rows": counts[(task, part)], "groups": len(groups[(task, part)])}
            for part in PART_RATIOS
        }
    return output


def count_support_violations(support: Mapping[str, Any]) -> int:
    violations = 0
    for task in support.values():
        violations += int(task["finetune_mgkg"]["rows"] < MIN_TASK_TRAIN_ROWS)
        violations += int(task["valid"]["rows"] < MIN_TASK_VALID_ROWS)
        violations += int(task["test"]["rows"] < MIN_TASK_TEST_ROWS)
        violations += int(task["valid"]["groups"] < MIN_TASK_VALID_GROUPS)
        violations += int(task["test"]["groups"] < MIN_TASK_TEST_GROUPS)
    return violations


def balance_score(
    records: Iterable[Mapping[str, Any]], *, assignment: Mapping[str, str]
) -> float:
    rows = list(records)
    score = 0.0
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["task_head"]), str(row["target_bin"]))].append(row)
    for stratum in strata.values():
        total = len(stratum)
        for part, ratio in PART_RATIOS.items():
            observed = sum(assignment[str(row["record_id"])] == part for row in stratum)
            expected = total * ratio
            score += (observed - expected) ** 2 / max(expected, 1.0)
    for part, ratio in PART_RATIOS.items():
        observed = sum(assignment[str(row["record_id"])] == part for row in rows)
        expected = len(rows) * ratio
        score += 5.0 * (observed - expected) ** 2 / max(expected, 1.0)
    return float(score)


def audit_target_partition(
    records: list[dict[str, Any]],
    *,
    assignment: Mapping[str, str],
    group_by_record: Mapping[str, str],
    references_by_record: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    by_part = {part: [] for part in PART_RATIOS}
    for row in records:
        by_part[assignment[row["record_id"]]].append(row)
    aggregate_ids = {
        part: {row["aggregate_id"] for row in rows} for part, rows in by_part.items()
    }
    result_ids = {
        part: {
            result_id
            for row in rows
            for result_id in parse_result_ids(row["result_ids"])
        }
        for part, rows in by_part.items()
    }
    test_ids = {
        part: {
            test_id
            for row in rows
            for test_id in parse_string_set(row["test_ids"], field="test_ids")
        }
        for part, rows in by_part.items()
    }
    references = {
        part: {
            ref
            for row in rows
            for ref in references_by_record[row["record_id"]]
        }
        for part, rows in by_part.items()
    }
    group_sets = {
        part: {group_by_record[row["record_id"]] for row in rows}
        for part, rows in by_part.items()
    }
    pairs = (("finetune_mgkg", "valid"), ("finetune_mgkg", "test"), ("valid", "test"))
    identity_overlap = {}
    reference_overlap = {}
    for left, right in pairs:
        label = f"{left}_vs_{right}"
        identity_overlap[label] = {
            "aggregate_id": len(aggregate_ids[left] & aggregate_ids[right]),
            "result_id": len(result_ids[left] & result_ids[right]),
            "test_id": len(test_ids[left] & test_ids[right]),
            "reference_component": len(group_sets[left] & group_sets[right]),
        }
        reference_overlap[label] = len(references[left] & references[right])
    if any(
        any(int(value) != 0 for value in overlap.values())
        for overlap in identity_overlap.values()
    ) or any(int(value) != 0 for value in reference_overlap.values()):
        raise ValueError(
            "Reference-group target boundary is not disjoint: "
            f"identity={identity_overlap}, reference={reference_overlap}"
        )
    support = partition_support(records, assignment=assignment, group_by_record=group_by_record)
    return {
        "counts": {part: len(rows) for part, rows in by_part.items()},
        "group_counts": {part: len(group_sets[part]) for part in PART_RATIOS},
        "reference_counts": {part: len(references[part]) for part in PART_RATIOS},
        "reference_ids": {part: sorted(references[part]) for part in PART_RATIOS},
        "task_counts": support,
        "identity_overlap": identity_overlap,
        "reference_overlap": reference_overlap,
    }


def build_route_assignments(
    *,
    parent_by_record: Mapping[str, Mapping[str, Any]],
    route: str,
    split_name: str,
    selected_seed: int,
    target_assignment: Mapping[str, str],
    target_group_by_record: Mapping[str, str],
    kept_source_ids: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record_id, parent in parent_by_record.items():
        parent_part = str(parent["split_part"])
        if parent_part in SOURCE_PARTS:
            if record_id not in kept_source_ids:
                continue
            if route == "M00":
                continue
            if route == "M10" and parent_part == "finetune":
                continue
            split_part = parent_part
            group_key = str(parent.get("group_key") or "")
        elif parent_part in TARGET_PARTS:
            if record_id not in target_assignment:
                continue
            split_part = target_assignment[record_id]
            component_digest = target_group_by_record[record_id].split(":", 1)[-1]
            group_key = (
                str(parent.get("group_key") or "")
                + f"|reference_component_sha256={component_digest}"
            )
        else:
            continue
        row = dict(parent)
        row.update(
            {
                "split_name": split_name,
                "split_part": split_part,
                "seed": int(selected_seed),
                "split_type": f"v1_2_46_reference_group_{route}",
                "group_key": group_key,
            }
        )
        output.append(row)
    if not output:
        raise ValueError(f"Derived v1.2.46 route is empty: {route}")
    return output


def parse_string_set(value: Any, *, field: str) -> frozenset[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return frozenset()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return frozenset()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON: {value!r}") from exc
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{field} must contain a JSON array: {value!r}")
    return frozenset(str(item).strip() for item in value if str(item).strip())


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


if __name__ == "__main__":
    main()
