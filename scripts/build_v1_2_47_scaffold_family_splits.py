from __future__ import annotations

import argparse
import csv
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
from scripts.build_scaffold_cluster_splits import normalize_structure
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
M00_SPLIT = "M_v1_2_47_SF_M00_从头训练_scaffold_family"
M10_SPLIT = "M_v1_2_47_SF_M10_水相预训练_scaffold_family"
M11U_SPLIT = "M_v1_2_47_SF_M11U_完整三阶段_scaffold_family"
ROUTE_SPLITS = {"M00": M00_SPLIT, "M10": M10_SPLIT, "M11U": M11U_SPLIT}

EXPECTED_PARENT_COUNTS = {
    "train": 245_147,
    "finetune": 12_327,
    "finetune_mgkg": 9_724,
    "valid": 2_433,
    "test": 3_042,
}
EXPECTED_TARGET_ROWS = 15_199
EXPECTED_TARGET_TASKS = 18
TARGET_PARTS = frozenset({"finetune_mgkg", "valid", "test"})
SOURCE_PARTS = frozenset({"train", "finetune"})
PART_RATIOS = {"finetune_mgkg": 0.64, "valid": 0.16, "test": 0.20}
PART_RATIO_RANGES = {
    "finetune_mgkg": (0.60, 0.68),
    "valid": (0.12, 0.20),
    "test": (0.17, 0.23),
}

BASE_SPLIT_SEED = 20_260_721
DEFAULT_CANDIDATES = 4_096
CONSTRAINT_REPAIR_CANDIDATES = 64
CONSTRAINT_REPAIR_MAX_MOVES = 32
TANIMOTO_THRESHOLD = 0.65
MORGAN_RADIUS = 2
MORGAN_BITS = 2_048

MIN_TASK_TRAIN_ROWS = 100
MIN_TASK_TRAIN_COMPONENTS = 5
MIN_TASK_VALID_ROWS = 1
MIN_TASK_VALID_COMPONENTS = 1
PREFERRED_TASK_VALID_ROWS = 20
PREFERRED_TASK_VALID_COMPONENTS = 3
MIN_TASK_TEST_ROWS = 30
MIN_TASK_TEST_COMPONENTS = 5

STANDARD_SUPPORT_TOTAL_ROWS = 131
LOW_SUPPORT_TASKS = frozenset({"ECx_Population", "ICx_Growth"})
LOW_SUPPORT_MIN_TASK_TRAIN_ROWS = 20
LOW_SUPPORT_MIN_TASK_TRAIN_COMPONENTS = 1
LOW_SUPPORT_MIN_TASK_VALID_ROWS = 1
LOW_SUPPORT_MIN_TASK_VALID_COMPONENTS = 1
LOW_SUPPORT_MIN_TASK_TEST_ROWS = 10
LOW_SUPPORT_MIN_TASK_TEST_COMPONENTS = 1
R2_SUPPORTED_MIN_TEST_ROWS = 30
R2_SUPPORTED_MIN_TEST_COMPONENTS = 5

SCHEMA = "v1_2_47_scaffold_family_split_v1"
EXCLUDED_TARGET_CSV = "v1_2_47_target_structure_exclusions.csv"
EXCLUDED_SOURCE_CSV = "v1_2_47_source_structure_exclusions.csv"
COMPONENT_CSV = "v1_2_47_target_structure_components.csv"
TASK_SUPPORT_CSV = "v1_2_47_target_task_support.csv"
OVERLAP_CSV = "v1_2_47_structure_overlap_audit.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the prediction-blind v1.2.47 scaffold/similarity-family "
            "M00/M10/M11U matrix from the locked v1.2.44 M11 route."
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
        result = build_scaffold_family_splits(
            db_path=args.db,
            source_table=args.source_table,
            parent_split=args.parent_split,
            route_splits=route_splits,
            audit_json=args.audit_json,
            candidate_count=args.candidate_count,
            base_seed=args.base_seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def build_scaffold_family_splits(
    *,
    db_path: Path,
    source_table: str = SOURCE_TABLE,
    parent_split: str = PARENT_SPLIT,
    route_splits: Mapping[str, str] | None = None,
    audit_json: Path | None,
    candidate_count: int = DEFAULT_CANDIDATES,
    base_seed: int = BASE_SPLIT_SEED,
) -> dict[str, Any]:
    route_splits = validate_route_splits(route_splits)
    if candidate_count < DEFAULT_CANDIDATES:
        raise ValueError(
            f"v1.2.47 requires at least {DEFAULT_CANDIDATES} prediction-blind candidates."
        )
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if audit_json is None:
        raise ValueError("v1.2.47 requires --audit-json so exclusions cannot be silent.")
    audit_dir = audit_json.parent
    audit_dir.mkdir(parents=True, exist_ok=True)

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
    ).copy()
    if len(frame) != len(parent_rows):
        raise ValueError(
            "Locked v1.2.44 strict join changed row count: "
            f"assignments={len(parent_rows)}, source_rows={len(frame)}"
        )
    frame["_record_id"] = [frame_record_id(row) for _, row in frame.iterrows()]
    if frame["_record_id"].duplicated().any():
        raise ValueError("Locked parent contains duplicate stage-sample identities.")

    target_frame = frame.loc[frame["split_part"].astype(str).isin(TARGET_PARTS)].copy()
    source_frame = frame.loc[frame["split_part"].astype(str).isin(SOURCE_PARTS)].copy()
    if len(target_frame) != EXPECTED_TARGET_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_ROWS} locked Stage-3 rows, observed {len(target_frame)}."
        )
    if int(target_frame["task_head"].nunique()) != EXPECTED_TARGET_TASKS:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_TASKS} locked Stage-3 tasks, "
            f"observed {target_frame['task_head'].nunique()}."
        )

    target_records = dataframe_records(target_frame)
    source_records = dataframe_records(source_frame)
    target_structures = normalize_record_structures(target_records, stage="stage3")
    write_exclusion_csv(audit_dir / EXCLUDED_TARGET_CSV, target_structures["excluded"])

    components = build_target_structure_components(
        target_structures["retained"],
        tanimoto_threshold=TANIMOTO_THRESHOLD,
        morgan_radius=MORGAN_RADIUS,
        morgan_bits=MORGAN_BITS,
    )
    feasibility = target_feasibility(
        target_records,
        retained_records=target_structures["retained"],
        group_by_record=components["group_by_record"],
    )
    write_component_csv(
        audit_dir / COMPONENT_CSV,
        target_structures["retained"],
        components["group_by_record"],
    )
    if not feasibility["feasible"]:
        summary = infeasible_summary(
            source_table=source_table,
            parent_split=parent_split,
            parent_assignment_sha256=parent_assignment_sha256,
            target_records=target_records,
            target_structures=target_structures,
            components=components,
            feasibility=feasibility,
            base_seed=base_seed,
            candidate_count=candidate_count,
            audit_dir=audit_dir,
        )
        write_locked_json(audit_json, summary)
        raise ValueError(
            "v1.2.47 structural target pool cannot satisfy the pre-registered 18-task "
            f"support gate after invalid structures are excluded: {feasibility['failures']}"
        )

    assignment, selection = choose_structure_partition(
        target_structures["retained"],
        group_by_record=components["group_by_record"],
        candidate_count=candidate_count,
        base_seed=base_seed,
        expected_tasks=sorted(feasibility["tasks"]),
        support_tiers=feasibility["support_tiers"],
    )
    target_audit = audit_target_partition(
        target_structures["retained"],
        assignment=assignment,
        group_by_record=components["group_by_record"],
        fingerprints_by_canonical=components["fingerprints_by_canonical"],
        tanimoto_threshold=TANIMOTO_THRESHOLD,
        expected_tasks=sorted(feasibility["tasks"]),
        support_tiers=feasibility["support_tiers"],
    )

    heldout_canonical, heldout_scaffolds = heldout_structure_sets(
        target_structures["retained"], assignment=assignment
    )
    heldout_fingerprints = [
        components["fingerprints_by_canonical"][canonical]
        for canonical in sorted(heldout_canonical)
    ]
    source_structures = normalize_record_structures(source_records, stage="source")
    kept_source_ids, source_excluded = filter_source_structures(
        source_structures["retained"],
        invalid_exclusions=source_structures["excluded"],
        heldout_canonical=heldout_canonical,
        heldout_scaffolds=heldout_scaffolds,
        heldout_fingerprints=heldout_fingerprints,
        tanimoto_threshold=TANIMOTO_THRESHOLD,
        morgan_radius=MORGAN_RADIUS,
        morgan_bits=MORGAN_BITS,
    )
    write_exclusion_csv(audit_dir / EXCLUDED_SOURCE_CSV, source_excluded)
    source_audit = audit_source_isolation(
        source_structures["retained"],
        kept_source_ids=kept_source_ids,
        excluded_records=source_excluded,
        target_records=target_structures["retained"],
        assignment=assignment,
        heldout_canonical=heldout_canonical,
        heldout_scaffolds=heldout_scaffolds,
        heldout_fingerprints=heldout_fingerprints,
        tanimoto_threshold=TANIMOTO_THRESHOLD,
        morgan_radius=MORGAN_RADIUS,
        morgan_bits=MORGAN_BITS,
    )

    parent_by_record = {row["record_id"]: row for row in parent_rows}
    route_assignments = {
        route: build_route_assignments(
            parent_by_record=parent_by_record,
            route=route,
            split_name=route_splits[route],
            selected_seed=int(selection["selected_seed"]),
            target_assignment=assignment,
            target_group_by_record=components["group_by_record"],
            kept_source_ids=kept_source_ids,
        )
        for route in ROUTE_SPLITS
    }
    persist_route_assignments(
        db_path,
        source_table=source_table,
        route_splits=route_splits,
        route_assignments=route_assignments,
    )

    routed_target = apply_task_target_routing(target_frame)
    routed_by_record = {
        str(row["_record_id"]): row for _, row in routed_target.iterrows()
    }
    evaluation_identity: dict[str, dict[str, Any]] = {}
    for part in PART_RATIOS:
        output_part = "finetune_mgkg_validation" if part == "valid" else part
        identities = [
            evaluation_identity_record(routed_by_record[record_id], split_part=output_part)
            for record_id, assigned in assignment.items()
            if assigned == part
        ]
        evaluation_identity[output_part] = {
            "rows": len(identities),
            "sha256": canonical_sha256(sorted(identities, key=canonical_json)),
        }

    route_audits = {
        route: {
            "split_name": route_splits[route],
            "rows": len(rows),
            "counts": dict(sorted(Counter(row["split_part"] for row in rows).items())),
            "assignment_sha256": assignment_sha256(rows),
        }
        for route, rows in route_assignments.items()
    }
    write_task_support_csv(
        audit_dir / TASK_SUPPORT_CSV,
        target_audit["task_counts"],
        reporting_support=target_audit["task_reporting_support"],
    )
    write_overlap_csv(
        audit_dir / OVERLAP_CSV,
        target_audit=target_audit,
        source_audit=source_audit,
    )

    audit_files = audit_file_manifest(
        audit_dir,
        (
            EXCLUDED_TARGET_CSV,
            EXCLUDED_SOURCE_CSV,
            COMPONENT_CSV,
            TASK_SUPPORT_CSV,
            OVERLAP_CSV,
        ),
    )
    component_by_record = dict(sorted(components["group_by_record"].items()))
    component_by_aggregate = {
        str(row["aggregate_id"]): components["group_by_record"][str(row["record_id"])]
        for row in target_structures["retained"]
    }
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "locked",
        "source_table": source_table,
        "parent_split": parent_split,
        "parent_assignment_sha256": parent_assignment_sha256,
        "parent_counts": dict(EXPECTED_PARENT_COUNTS),
        "structure_policy": structure_policy(),
        "partition_policy": {
            "ratios": dict(PART_RATIOS),
            "ratio_ranges": {key: list(value) for key, value in PART_RATIO_RANGES.items()},
            "candidate_count": int(candidate_count),
            "base_seed": int(base_seed),
            "selected_seed": int(selection["selected_seed"]),
            "selection_score": float(selection["score"]),
            "feasible_candidate_count": int(selection["feasible_candidate_count"]),
            "constraint_repair_candidates_evaluated": int(
                selection["constraint_repair_candidates_evaluated"]
            ),
            "constraint_repair_feasible_candidate_count": int(
                selection["constraint_repair_feasible_candidate_count"]
            ),
            "selected_via_constraint_repair": bool(
                selection["selected_via_constraint_repair"]
            ),
            "constraint_repair_moves": selection["constraint_repair_moves"],
            "validation_support_warning_count": int(
                selection["validation_support_warning_count"]
            ),
            "uses_model_predictions": False,
            "selection_inputs": (
                "row ratios, per-task rows, per-task structure-component counts, "
                "and within-task target quintiles"
            ),
            "constraint_repair": (
                "if no raw hash candidate is feasible, greedily relocate complete structure "
                "components among the best 64 candidates using support deficits, ratio gates, "
                "identity gates, and prediction-blind balance score only"
            ),
            "minimum_support": support_policy(),
            "support_tiers": support_policy()["support_tiers"],
            "task_support_tiers": feasibility["support_tiers"],
        },
        "target_pool": {
            "original_rows": len(target_records),
            "retained_rows": len(target_structures["retained"]),
            "excluded_rows": len(target_structures["excluded"]),
            "tasks": len(feasibility["tasks"]),
            "canonical_chemicals": int(components["canonical_count"]),
            "structure_components": int(components["component_count"]),
            "acyclic_chemicals": int(components["acyclic_canonical_count"]),
            "similarity_edges_ge_threshold": int(components["similarity_edge_count"]),
            "structure_component_sha256": canonical_sha256(component_by_record),
            "assignment_sha256": canonical_sha256(
                sorted(assignment.items(), key=lambda item: item[0])
            ),
            "counts": target_audit["counts"],
            "group_counts": target_audit["group_counts"],
            "task_counts": target_audit["task_counts"],
            "task_reporting_support": target_audit["task_reporting_support"],
            "identity_overlap": target_audit["identity_overlap"],
            "structure_overlap": target_audit["structure_overlap"],
            "max_cross_part_tanimoto": target_audit["max_cross_part_tanimoto"],
            "evaluation_identity": evaluation_identity,
            "component_by_record": component_by_record,
            "component_by_aggregate": dict(sorted(component_by_aggregate.items())),
        },
        "source_structure_exclusion": source_audit,
        "routes": route_audits,
        "route_semantics": {
            "M00": "strict scaffold-family Stage3 from random initialization",
            "M10": "heldout-structure-filtered aquatic Stage1 plus strict Stage3",
            "M11U": (
                "heldout-structure-filtered aquatic Stage1 and soil-pTox Stage2 "
                "plus strict Stage3"
            ),
        },
        "audit_files": audit_files,
    }
    write_locked_json(audit_json, summary)
    return summary


def validate_route_splits(
    route_splits: Mapping[str, str] | None,
) -> dict[str, str]:
    output = dict(route_splits or ROUTE_SPLITS)
    if set(output) != set(ROUTE_SPLITS):
        raise ValueError("route_splits must define exactly M00, M10, and M11U.")
    if len(set(output.values())) != len(output) or any(not value.strip() for value in output.values()):
        raise ValueError("v1.2.47 route split names must be non-empty and unique.")
    return output


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "record_id": str(row["_record_id"]),
                "aggregate_id": str(row["aggregate_id"]),
                "split_part": str(row["split_part"]),
                "task_head": str(row["task_head"]),
                "target_value": float(row["target_value_median"]),
                "smiles": clean_text(row.get("smiles")),
                "cas_number": clean_text(row.get("cas_number")),
                "dtxsid": clean_text(row.get("dtxsid")),
                "chemical_name": clean_text(row.get("chemical_name")),
                "test_ids": row.get("test_ids"),
                "result_ids": row.get("result_ids"),
            }
        )
    return records


def normalize_record_structures(
    records: Iterable[Mapping[str, Any]], *, stage: str
) -> dict[str, list[dict[str, Any]]]:
    rows = [dict(row) for row in records]
    normalized_by_smiles: dict[str, dict[str, str]] = {}
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        smiles = clean_text(row.get("smiles"))
        if smiles not in normalized_by_smiles:
            normalized_by_smiles[smiles] = normalize_structure(smiles)
        normalized = normalized_by_smiles[smiles]
        output = {
            **row,
            "stage": stage,
            "canonical_smiles": normalized["canonical_smiles"],
            "scaffold_smiles": normalized["scaffold_smiles"],
            "structure_status": normalized["structure_status"],
            "exclusion_reason": normalized["parse_error"],
        }
        if normalized["structure_status"] != "ok" or not normalized["canonical_smiles"]:
            excluded.append(output)
        else:
            retained.append(output)
    return {"retained": retained, "excluded": excluded}


class UnionFind:
    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent = {str(item): str(item) for item in items}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep


def build_target_structure_components(
    records: Iterable[Mapping[str, Any]],
    *,
    tanimoto_threshold: float = TANIMOTO_THRESHOLD,
    morgan_radius: int = MORGAN_RADIUS,
    morgan_bits: int = MORGAN_BITS,
) -> dict[str, Any]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("No structurally valid Stage-3 records remain.")
    canonical_meta: dict[str, str] = {}
    for row in rows:
        canonical = str(row["canonical_smiles"])
        scaffold = str(row.get("scaffold_smiles") or "")
        previous = canonical_meta.setdefault(canonical, scaffold)
        if previous != scaffold:
            raise ValueError(
                f"Canonical parent has inconsistent Murcko scaffold: {canonical!r}."
            )
    canonicals = sorted(canonical_meta)
    generator = GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
    fingerprints: dict[str, Any] = {}
    for canonical in canonicals:
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"Normalized canonical SMILES cannot be reparsed: {canonical!r}")
        fingerprints[canonical] = generator.GetFingerprint(mol)

    union = UnionFind(canonicals)
    by_scaffold: dict[str, list[str]] = defaultdict(list)
    for canonical, scaffold in canonical_meta.items():
        if scaffold:
            by_scaffold[scaffold].append(canonical)
    for members in by_scaffold.values():
        for canonical in members[1:]:
            union.union(members[0], canonical)

    similarity_edges = 0
    for index, canonical in enumerate(canonicals):
        previous = canonicals[:index]
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[canonical], [fingerprints[item] for item in previous]
        )
        for other, similarity in zip(previous, similarities):
            if float(similarity) >= tanimoto_threshold:
                union.union(canonical, other)
                similarity_edges += 1

    members_by_root: dict[str, set[str]] = defaultdict(set)
    for canonical in canonicals:
        members_by_root[union.find(canonical)].add(canonical)
    component_by_root = {
        root: "structure_component_v1:"
        + hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()
        for root, members in members_by_root.items()
    }
    component_by_canonical = {
        canonical: component_by_root[union.find(canonical)] for canonical in canonicals
    }
    group_by_record = {
        str(row["record_id"]): component_by_canonical[str(row["canonical_smiles"])]
        for row in rows
    }
    return {
        "group_by_record": group_by_record,
        "component_by_canonical": component_by_canonical,
        "fingerprints_by_canonical": fingerprints,
        "component_count": len(set(component_by_canonical.values())),
        "canonical_count": len(canonicals),
        "acyclic_canonical_count": sum(not scaffold for scaffold in canonical_meta.values()),
        "similarity_edge_count": similarity_edges,
    }


def target_feasibility(
    original_records: Iterable[Mapping[str, Any]],
    *,
    retained_records: Iterable[Mapping[str, Any]],
    group_by_record: Mapping[str, str],
) -> dict[str, Any]:
    original = list(original_records)
    retained = list(retained_records)
    tasks = sorted({str(row["task_head"]) for row in original})
    rows_by_task = Counter(str(row["task_head"]) for row in retained)
    groups_by_task: dict[str, set[str]] = defaultdict(set)
    for row in retained:
        groups_by_task[str(row["task_head"])].add(group_by_record[str(row["record_id"])])
    support_tiers = {
        task: (
            "standard"
            if rows_by_task[task] >= STANDARD_SUPPORT_TOTAL_ROWS
            else "low_support"
        )
        for task in tasks
    }
    support = {
        task: {
            "original_rows": sum(str(row["task_head"]) == task for row in original),
            "retained_rows": rows_by_task[task],
            "structure_components": len(groups_by_task[task]),
            "support_tier": support_tiers[task],
        }
        for task in tasks
    }
    failures = {
        task: values
        for task, values in support.items()
        if values["retained_rows"]
        < support_tier_policy(values["support_tier"])["minimum_total_rows"]
        or values["structure_components"]
        < support_tier_policy(values["support_tier"])["minimum_total_components"]
    }
    observed_low_support_tasks = sorted(
        task for task, tier in support_tiers.items() if tier == "low_support"
    )
    configuration_failures: list[str] = []
    if len(tasks) != EXPECTED_TARGET_TASKS:
        configuration_failures.append(
            f"expected_{EXPECTED_TARGET_TASKS}_tasks_observed_{len(tasks)}"
        )
    if len(tasks) == EXPECTED_TARGET_TASKS and set(observed_low_support_tasks) != LOW_SUPPORT_TASKS:
        configuration_failures.append(
            "low_support_task_set_changed:"
            f"observed={observed_low_support_tasks},expected={sorted(LOW_SUPPORT_TASKS)}"
        )
    return {
        "feasible": not failures and not configuration_failures,
        "tasks": tasks,
        "task_support": support,
        "support_tiers": support_tiers,
        "observed_low_support_tasks": observed_low_support_tasks,
        "support_tier_policy": support_policy()["support_tiers"],
        "failures": failures,
        "configuration_failures": configuration_failures,
    }


def choose_structure_partition(
    records: list[dict[str, Any]],
    *,
    group_by_record: Mapping[str, str],
    candidate_count: int,
    base_seed: int,
    expected_tasks: Iterable[str] | None = None,
    support_tiers: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")
    enriched = attach_target_quantile_bins(records)
    groups = sorted(set(group_by_record.values()))
    tasks = sorted(expected_tasks or {str(row["task_head"]) for row in records})
    tiers = resolve_support_tiers(records, tasks=tasks, support_tiers=support_tiers)
    best: tuple[tuple[int, int, float, int], dict[str, str], dict[str, Any]] | None = None
    repair_pool: list[
        tuple[tuple[int, int, float, int], dict[str, str], dict[str, Any]]
    ] = []
    feasible_count = 0
    for offset in range(candidate_count):
        seed = int(base_seed + offset)
        group_part = {group: hash_partition(seed, group) for group in groups}
        assignment = {
            str(row["record_id"]): group_part[group_by_record[str(row["record_id"])]]
            for row in enriched
        }
        support = partition_support(
            enriched,
            assignment=assignment,
            group_by_record=group_by_record,
            expected_tasks=tasks,
        )
        violations = count_support_violations(support, support_tiers=tiers)
        ratio_violations, ratios = ratio_gate(enriched, assignment=assignment)
        violations += ratio_violations
        identity_violations = count_identity_partition_violations(
            enriched, assignment=assignment
        )
        violations += identity_violations
        warnings = validation_support_warning_count(support)
        if violations == 0:
            feasible_count += 1
        score = balance_score(
            enriched,
            assignment=assignment,
            group_by_record=group_by_record,
            expected_tasks=tasks,
        )
        key = (violations, warnings, score, seed)
        if best is None or key < best[0]:
            best = (
                key,
                assignment,
                {
                    "selected_seed": seed,
                    "score": score,
                    "violations": violations,
                    "validation_support_warning_count": warnings,
                    "identity_partition_violations": identity_violations,
                    "support": support,
                    "support_tiers": tiers,
                    "ratios": ratios,
                    "selected_via_constraint_repair": False,
                    "constraint_repair_moves": [],
                },
            )
        repair_pool.append(
            (
                key,
                group_part,
                {
                    "selected_seed": seed,
                    "score": score,
                    "violations": violations,
                    "validation_support_warning_count": warnings,
                    "identity_partition_violations": identity_violations,
                    "support": support,
                    "support_tiers": tiers,
                    "ratios": ratios,
                },
            )
        )
        repair_pool.sort(key=lambda item: item[0])
        if len(repair_pool) > CONSTRAINT_REPAIR_CANDIDATES:
            repair_pool.pop()
    if best is None:
        raise ValueError("No scaffold-family split candidate was evaluated.")
    repaired_feasible_count = 0
    if best[2]["violations"] != 0:
        repaired_best: tuple[
            tuple[int, int, float, int], dict[str, str], dict[str, Any]
        ] | None = None
        for _, group_part, metadata in repair_pool:
            repaired_assignment, repaired_metadata = repair_structure_partition(
                enriched,
                group_part=group_part,
                group_by_record=group_by_record,
                expected_tasks=tasks,
                support_tiers=tiers,
                selected_seed=int(metadata["selected_seed"]),
            )
            if repaired_metadata["violations"] == 0:
                repaired_feasible_count += 1
            repaired_key = (
                int(repaired_metadata["violations"]),
                int(repaired_metadata["validation_support_warning_count"]),
                float(repaired_metadata["score"]),
                int(repaired_metadata["selected_seed"]),
            )
            if repaired_best is None or repaired_key < repaired_best[0]:
                repaired_best = (repaired_key, repaired_assignment, repaired_metadata)
        if repaired_best is not None and repaired_best[2]["violations"] == 0:
            best = repaired_best
        else:
            raise ValueError(
                "No scaffold-family candidate satisfied the pre-registered support and ratio "
                "gates after prediction-blind structure-group constraint repair; "
                f"best_seed={best[2]['selected_seed']} ratios={best[2]['ratios']} "
                f"support={best[2]['support']}"
            )
    best[2]["feasible_candidate_count"] = feasible_count
    best[2]["constraint_repair_candidates_evaluated"] = len(repair_pool)
    best[2]["constraint_repair_feasible_candidate_count"] = repaired_feasible_count
    return best[1], best[2]


def repair_structure_partition(
    records: list[dict[str, Any]],
    *,
    group_part: Mapping[str, str],
    group_by_record: Mapping[str, str],
    expected_tasks: Iterable[str],
    support_tiers: Mapping[str, str],
    selected_seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Repair a near-feasible hash split using only structure-group support counts.

    The search never uses model predictions. Each move relocates one complete
    structure component, so the chemical-family boundary remains intact.
    """

    prepared = (
        records
        if all("target_bin" in row for row in records)
        else attach_target_quantile_bins(records)
    )
    tasks = sorted(expected_tasks)
    parts = dict(group_part)
    group_task_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for row in prepared:
        group_task_rows[group_by_record[str(row["record_id"])]][
            str(row["task_head"])
        ] += 1

    def evaluate(candidate_parts: Mapping[str, str]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        assignment = {
            str(row["record_id"]): candidate_parts[group_by_record[str(row["record_id"])]]
            for row in prepared
        }
        support = partition_support(
            prepared,
            assignment=assignment,
            group_by_record=group_by_record,
            expected_tasks=tasks,
        )
        support_violations = count_support_violations(
            support, support_tiers=support_tiers
        )
        ratio_violations, ratios = ratio_gate(records, assignment=assignment)
        identity_violations = count_identity_partition_violations(
            prepared, assignment=assignment
        )
        violations = support_violations + ratio_violations + identity_violations
        deficit = support_deficit_score(support, support_tiers=support_tiers)
        warnings = validation_support_warning_count(support)
        score = balance_score(
            prepared,
            assignment=assignment,
            group_by_record=group_by_record,
            expected_tasks=tasks,
        )
        objective = (violations, deficit, warnings, score)
        return objective, {
            "assignment": assignment,
            "support": support,
            "ratios": ratios,
            "identity_partition_violations": identity_violations,
            "violations": violations,
            "validation_support_warning_count": warnings,
            "score": score,
        }

    current_objective, current = evaluate(parts)
    moves: list[dict[str, str]] = []
    for _ in range(CONSTRAINT_REPAIR_MAX_MOVES):
        if current["violations"] == 0:
            break
        deficient: list[tuple[str, str]] = []
        for task in tasks:
            policy = support_tier_policy(support_tiers[task])
            for part, row_key, group_key in (
                ("finetune_mgkg", "train_rows", "train_components"),
                ("valid", "validation_rows", "validation_components"),
                ("test", "test_rows", "test_components"),
            ):
                observed = current["support"][task][part]
                if (
                    observed["rows"] < policy[row_key]
                    or observed["groups"] < policy[group_key]
                ):
                    deficient.append((task, part))

        best_move: tuple[tuple[Any, ...], str, str, dict[str, Any]] | None = None
        for task, destination in deficient:
            for group in sorted(group_task_rows):
                if group_task_rows[group][task] <= 0 or parts[group] == destination:
                    continue
                candidate_parts = dict(parts)
                candidate_parts[group] = destination
                objective, evaluated = evaluate(candidate_parts)
                move_key = (*objective, group, destination)
                if move_key >= (*current_objective, "", ""):
                    continue
                if best_move is None or move_key < best_move[0]:
                    best_move = (move_key, group, destination, evaluated)
        if best_move is None:
            break
        _, group, destination, current = best_move
        source = parts[group]
        parts[group] = destination
        current_objective = best_move[0][:4]
        moves.append(
            {
                "structure_component": group,
                "from": source,
                "to": destination,
            }
        )

    return current["assignment"], {
        "selected_seed": int(selected_seed),
        "score": float(current["score"]),
        "violations": int(current["violations"]),
        "validation_support_warning_count": int(
            current["validation_support_warning_count"]
        ),
        "identity_partition_violations": int(
            current["identity_partition_violations"]
        ),
        "support": current["support"],
        "support_tiers": dict(support_tiers),
        "ratios": current["ratios"],
        "selected_via_constraint_repair": bool(moves),
        "constraint_repair_moves": moves,
    }


def support_deficit_score(
    support: Mapping[str, Any], *, support_tiers: Mapping[str, str]
) -> int:
    deficit = 0
    for task, values in support.items():
        policy = support_tier_policy(support_tiers[task])
        for part, row_key, group_key in (
            ("finetune_mgkg", "train_rows", "train_components"),
            ("valid", "validation_rows", "validation_components"),
            ("test", "test_rows", "test_components"),
        ):
            deficit += max(0, policy[row_key] - int(values[part]["rows"]))
            deficit += max(0, policy[group_key] - int(values[part]["groups"]))
    return deficit


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
            frame.at[index, "target_bin"] = f"Q{min(bins - 1, int(position * bins / len(ordered))) + 1}"
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
    expected_tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    tasks = sorted(expected_tasks or {str(row["task_head"]) for row in rows})
    counts: Counter[tuple[str, str]] = Counter()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        task = str(row["task_head"])
        part = assignment[str(row["record_id"])]
        counts[(task, part)] += 1
        groups[(task, part)].add(group_by_record[str(row["record_id"])])
    return {
        task: {
            part: {"rows": counts[(task, part)], "groups": len(groups[(task, part)])}
            for part in PART_RATIOS
        }
        for task in tasks
    }


def count_support_violations(
    support: Mapping[str, Any], *, support_tiers: Mapping[str, str] | None = None
) -> int:
    violations = 0
    tiers = support_tiers or {task: "standard" for task in support}
    for task_name, task in support.items():
        policy = support_tier_policy(tiers[task_name])
        violations += int(task["finetune_mgkg"]["rows"] < policy["train_rows"])
        violations += int(task["finetune_mgkg"]["groups"] < policy["train_components"])
        violations += int(task["valid"]["rows"] < policy["validation_rows"])
        violations += int(task["valid"]["groups"] < policy["validation_components"])
        violations += int(task["test"]["rows"] < policy["test_rows"])
        violations += int(task["test"]["groups"] < policy["test_components"])
    return violations


def resolve_support_tiers(
    records: Iterable[Mapping[str, Any]],
    *,
    tasks: Iterable[str],
    support_tiers: Mapping[str, str] | None,
) -> dict[str, str]:
    task_names = sorted(tasks)
    if support_tiers is None:
        totals = Counter(str(row["task_head"]) for row in records)
        resolved = {
            task: (
                "standard"
                if totals[task] >= STANDARD_SUPPORT_TOTAL_ROWS
                else "low_support"
            )
            for task in task_names
        }
    else:
        resolved = {task: str(support_tiers[task]) for task in task_names}
    if set(resolved) != set(task_names):
        raise ValueError("Support-tier mapping does not match the target tasks.")
    for task, tier in resolved.items():
        support_tier_policy(tier)
    return resolved


def count_identity_partition_violations(
    records: Iterable[Mapping[str, Any]], *, assignment: Mapping[str, str]
) -> int:
    partitions_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        part = assignment[str(row["record_id"])]
        partitions_by_identity[("aggregate_id", str(row["aggregate_id"]))].add(part)
        for result_id in parse_result_ids(row.get("result_ids")):
            partitions_by_identity[("result_id", result_id)].add(part)
        for test_id in parse_string_set(row.get("test_ids"), field="test_ids"):
            partitions_by_identity[("test_id", test_id)].add(part)
    return sum(len(parts) - 1 for parts in partitions_by_identity.values() if len(parts) > 1)


def validation_support_warning_count(support: Mapping[str, Any]) -> int:
    return sum(
        values["valid"]["rows"] < PREFERRED_TASK_VALID_ROWS
        or values["valid"]["groups"] < PREFERRED_TASK_VALID_COMPONENTS
        for values in support.values()
    )


def ratio_gate(
    records: Iterable[Mapping[str, Any]], *, assignment: Mapping[str, str]
) -> tuple[int, dict[str, float]]:
    rows = list(records)
    ratios = {
        part: sum(assignment[str(row["record_id"])] == part for row in rows) / len(rows)
        for part in PART_RATIOS
    }
    violations = sum(
        not (PART_RATIO_RANGES[part][0] <= value <= PART_RATIO_RANGES[part][1])
        for part, value in ratios.items()
    )
    return violations, ratios


def balance_score(
    records: Iterable[Mapping[str, Any]],
    *,
    assignment: Mapping[str, str],
    group_by_record: Mapping[str, str],
    expected_tasks: Iterable[str] | None = None,
) -> float:
    rows = list(records)
    tasks = sorted(expected_tasks or {str(row["task_head"]) for row in rows})
    score = 0.0
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["task_head"]), str(row["target_bin"]))].append(row)
    for stratum in strata.values():
        for part, ratio in PART_RATIOS.items():
            observed = sum(assignment[str(row["record_id"])] == part for row in stratum)
            expected = len(stratum) * ratio
            score += (observed - expected) ** 2 / max(expected, 1.0)
    for part, ratio in PART_RATIOS.items():
        observed = sum(assignment[str(row["record_id"])] == part for row in rows)
        expected = len(rows) * ratio
        score += 5.0 * (observed - expected) ** 2 / max(expected, 1.0)
    for task in tasks:
        task_rows = [row for row in rows if str(row["task_head"]) == task]
        task_groups = {group_by_record[str(row["record_id"])] for row in task_rows}
        for part, ratio in PART_RATIOS.items():
            observed = len(
                {
                    group_by_record[str(row["record_id"])]
                    for row in task_rows
                    if assignment[str(row["record_id"])] == part
                }
            )
            expected = len(task_groups) * ratio
            score += (observed - expected) ** 2 / max(expected, 1.0)
    return float(score)


def audit_target_partition(
    records: list[dict[str, Any]],
    *,
    assignment: Mapping[str, str],
    group_by_record: Mapping[str, str],
    fingerprints_by_canonical: Mapping[str, Any],
    tanimoto_threshold: float,
    expected_tasks: Iterable[str] | None = None,
    support_tiers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    by_part = {part: [] for part in PART_RATIOS}
    for row in records:
        by_part[assignment[str(row["record_id"])]].append(row)
    canonical = {
        part: {str(row["canonical_smiles"]) for row in rows}
        for part, rows in by_part.items()
    }
    scaffolds = {
        part: {str(row["scaffold_smiles"]) for row in rows if str(row["scaffold_smiles"])}
        for part, rows in by_part.items()
    }
    components = {
        part: {group_by_record[str(row["record_id"])] for row in rows}
        for part, rows in by_part.items()
    }
    aggregate_ids = {
        part: {str(row["aggregate_id"]) for row in rows}
        for part, rows in by_part.items()
    }
    result_ids = {
        part: {item for row in rows for item in parse_result_ids(row.get("result_ids"))}
        for part, rows in by_part.items()
    }
    test_ids = {
        part: {
            item for row in rows for item in parse_string_set(row.get("test_ids"), field="test_ids")
        }
        for part, rows in by_part.items()
    }
    pairs = (("finetune_mgkg", "valid"), ("finetune_mgkg", "test"), ("valid", "test"))
    identity_overlap: dict[str, Any] = {}
    structure_overlap: dict[str, Any] = {}
    max_similarity: dict[str, float] = {}
    for left, right in pairs:
        label = f"{left}_vs_{right}"
        identity_overlap[label] = {
            "aggregate_id": len(aggregate_ids[left] & aggregate_ids[right]),
            "result_id": len(result_ids[left] & result_ids[right]),
            "test_id": len(test_ids[left] & test_ids[right]),
        }
        structure_overlap[label] = {
            "canonical_parent": len(canonical[left] & canonical[right]),
            "murcko_scaffold": len(scaffolds[left] & scaffolds[right]),
            "structure_component": len(components[left] & components[right]),
        }
        max_similarity[label] = max_cross_similarity(
            canonical[left], canonical[right], fingerprints_by_canonical
        )
    if any(any(value for value in values.values()) for values in identity_overlap.values()):
        raise ValueError(f"Stage-3 identity boundary overlaps: {identity_overlap}")
    if any(any(value for value in values.values()) for values in structure_overlap.values()):
        raise ValueError(f"Stage-3 structure boundary overlaps: {structure_overlap}")
    if any(value >= tanimoto_threshold for value in max_similarity.values()):
        raise ValueError(
            "Stage-3 cross-part Morgan similarity violates the locked threshold: "
            f"{max_similarity}"
        )
    support = partition_support(
        records,
        assignment=assignment,
        group_by_record=group_by_record,
        expected_tasks=expected_tasks,
    )
    tiers = resolve_support_tiers(
        records,
        tasks=sorted(support),
        support_tiers=support_tiers,
    )
    violations, ratios = ratio_gate(records, assignment=assignment)
    if count_support_violations(support, support_tiers=tiers) or violations:
        raise ValueError(f"Selected Stage-3 boundary violates locked support/ratio gates: {support}")
    reporting_support = {
        task: {
            "support_tier": tiers[task],
            "low_support": tiers[task] == "low_support",
            "task_r2_supported": (
                values["test"]["rows"] >= R2_SUPPORTED_MIN_TEST_ROWS
                and values["test"]["groups"] >= R2_SUPPORTED_MIN_TEST_COMPONENTS
            ),
            "task_r2_rule": (
                f"test rows >= {R2_SUPPORTED_MIN_TEST_ROWS} and test structure components "
                f">= {R2_SUPPORTED_MIN_TEST_COMPONENTS}"
            ),
        }
        for task, values in support.items()
    }
    return {
        "counts": {part: len(rows) for part, rows in by_part.items()},
        "ratios": ratios,
        "group_counts": {part: len(components[part]) for part in PART_RATIOS},
        "task_counts": support,
        "task_reporting_support": reporting_support,
        "identity_overlap": identity_overlap,
        "structure_overlap": structure_overlap,
        "max_cross_part_tanimoto": max_similarity,
    }


def heldout_structure_sets(
    records: Iterable[Mapping[str, Any]], *, assignment: Mapping[str, str]
) -> tuple[set[str], set[str]]:
    heldout = [
        row
        for row in records
        if assignment[str(row["record_id"])] in {"valid", "test"}
    ]
    return (
        {str(row["canonical_smiles"]) for row in heldout},
        {str(row["scaffold_smiles"]) for row in heldout if str(row["scaffold_smiles"])},
    )


def filter_source_structures(
    records: Iterable[Mapping[str, Any]],
    *,
    invalid_exclusions: Iterable[Mapping[str, Any]],
    heldout_canonical: set[str],
    heldout_scaffolds: set[str],
    heldout_fingerprints: list[Any],
    tanimoto_threshold: float,
    morgan_radius: int,
    morgan_bits: int,
) -> tuple[set[str], list[dict[str, Any]]]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
    decision_by_canonical: dict[str, tuple[bool, str, float]] = {}
    kept: set[str] = set()
    excluded = [dict(row) for row in invalid_exclusions]
    for row_value in records:
        row = dict(row_value)
        canonical = str(row["canonical_smiles"])
        if canonical not in decision_by_canonical:
            scaffold = str(row.get("scaffold_smiles") or "")
            if canonical in heldout_canonical:
                decision_by_canonical[canonical] = (False, "heldout_canonical_parent", 1.0)
            elif scaffold and scaffold in heldout_scaffolds:
                decision_by_canonical[canonical] = (False, "heldout_murcko_scaffold", 1.0)
            else:
                mol = Chem.MolFromSmiles(canonical)
                if mol is None:
                    decision_by_canonical[canonical] = (False, "normalized_reparse_failed", math.nan)
                else:
                    fingerprint = generator.GetFingerprint(mol)
                    max_similarity = (
                        max(DataStructs.BulkTanimotoSimilarity(fingerprint, heldout_fingerprints))
                        if heldout_fingerprints
                        else 0.0
                    )
                    decision_by_canonical[canonical] = (
                        float(max_similarity) < tanimoto_threshold,
                        "heldout_morgan_similarity" if max_similarity >= tanimoto_threshold else "",
                        float(max_similarity),
                    )
        is_kept, reason, similarity = decision_by_canonical[canonical]
        if is_kept:
            kept.add(str(row["record_id"]))
        else:
            row["exclusion_reason"] = reason
            row["max_tanimoto_to_heldout"] = similarity
            excluded.append(row)
    return kept, excluded


def audit_source_isolation(
    source_records: Iterable[Mapping[str, Any]],
    *,
    kept_source_ids: set[str],
    excluded_records: Iterable[Mapping[str, Any]],
    target_records: Iterable[Mapping[str, Any]],
    assignment: Mapping[str, str],
    heldout_canonical: set[str],
    heldout_scaffolds: set[str],
    heldout_fingerprints: list[Any],
    tanimoto_threshold: float,
    morgan_radius: int,
    morgan_bits: int,
) -> dict[str, Any]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    all_source = list(source_records)
    kept = [row for row in all_source if str(row["record_id"]) in kept_source_ids]
    if not kept or {str(row["split_part"]) for row in kept} != SOURCE_PARTS:
        raise ValueError("Structure exclusion removed an entire source training stage.")
    kept_canonical = {str(row["canonical_smiles"]) for row in kept}
    kept_scaffolds = {
        str(row["scaffold_smiles"]) for row in kept if str(row["scaffold_smiles"])
    }
    generator = GetMorganGenerator(radius=morgan_radius, fpSize=morgan_bits)
    maximum = 0.0
    for canonical in sorted(kept_canonical):
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"Kept source canonical cannot be reparsed: {canonical!r}")
        fp = generator.GetFingerprint(mol)
        if heldout_fingerprints:
            maximum = max(
                maximum,
                max(float(value) for value in DataStructs.BulkTanimotoSimilarity(fp, heldout_fingerprints)),
            )
    if kept_canonical & heldout_canonical or kept_scaffolds & heldout_scaffolds:
        raise ValueError("Source training retains a held-out canonical parent or scaffold.")
    if maximum >= tanimoto_threshold:
        raise ValueError(f"Source-to-heldout maximum Tanimoto remains {maximum}.")

    excluded = list(excluded_records)
    heldout_target_by_part = {
        part: [
            row
            for row in target_records
            if assignment[str(row["record_id"])] == part
        ]
        for part in ("valid", "test")
    }
    heldout_target_by_part["valid_test"] = (
        heldout_target_by_part["valid"] + heldout_target_by_part["test"]
    )
    source_aggregates = {str(row["aggregate_id"]) for row in kept}
    source_results = {item for row in kept for item in parse_result_ids(row.get("result_ids"))}
    source_tests = {
        item for row in kept for item in parse_string_set(row.get("test_ids"), field="test_ids")
    }
    exact_overlap: dict[str, dict[str, int]] = {}
    structure_overlap: dict[str, dict[str, int]] = {}
    max_tanimoto_by_boundary: dict[str, float] = {}
    for part, target_rows in heldout_target_by_part.items():
        label = f"source_train_vs_stage3_{part}"
        target_aggregates = {str(row["aggregate_id"]) for row in target_rows}
        target_results = {
            item for row in target_rows for item in parse_result_ids(row.get("result_ids"))
        }
        target_tests = {
            item
            for row in target_rows
            for item in parse_string_set(row.get("test_ids"), field="test_ids")
        }
        target_canonical = {str(row["canonical_smiles"]) for row in target_rows}
        target_scaffolds = {
            str(row["scaffold_smiles"])
            for row in target_rows
            if str(row["scaffold_smiles"])
        }
        target_fingerprints = []
        for canonical in sorted(target_canonical):
            mol = Chem.MolFromSmiles(canonical)
            if mol is None:
                raise ValueError(f"Heldout canonical cannot be reparsed: {canonical!r}")
            target_fingerprints.append(generator.GetFingerprint(mol))
        boundary_maximum = 0.0
        for canonical in sorted(kept_canonical):
            mol = Chem.MolFromSmiles(canonical)
            if mol is None:
                raise ValueError(f"Kept source canonical cannot be reparsed: {canonical!r}")
            fingerprint = generator.GetFingerprint(mol)
            if target_fingerprints:
                boundary_maximum = max(
                    boundary_maximum,
                    max(
                        float(value)
                        for value in DataStructs.BulkTanimotoSimilarity(
                            fingerprint, target_fingerprints
                        )
                    ),
                )
        exact_overlap[label] = {
            "aggregate_id": len(source_aggregates & target_aggregates),
            "result_id": len(source_results & target_results),
            "test_id": len(source_tests & target_tests),
        }
        structure_overlap[label] = {
            "canonical_parent": len(kept_canonical & target_canonical),
            "murcko_scaffold": len(kept_scaffolds & target_scaffolds),
        }
        max_tanimoto_by_boundary[label] = boundary_maximum
    if any(
        any(values.values())
        for values in exact_overlap.values()
    ):
        raise ValueError(f"Source stages retain exact held-out target identities: {exact_overlap}")
    if any(any(values.values()) for values in structure_overlap.values()):
        raise ValueError(f"Source stages retain held-out target structures: {structure_overlap}")
    if any(value >= tanimoto_threshold for value in max_tanimoto_by_boundary.values()):
        raise ValueError(
            "Source stages retain a held-out-similar target structure: "
            f"{max_tanimoto_by_boundary}"
        )
    combined_label = "source_train_vs_stage3_valid_test"
    exclusion_reasons = Counter(
        str(row.get("exclusion_reason") or "unspecified") for row in excluded
    )
    return {
        "missing_or_invalid_structure_policy": "exclude_and_report",
        "source_rows_before_valid_structure_filter": len(all_source) + sum(
            str(row.get("structure_status")) != "ok" for row in excluded
        ),
        "source_rows_after": len(kept),
        "excluded_valid_structure_rows": len(all_source) - len(kept),
        "excluded_rows_total": len(excluded),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "remaining_by_part": dict(sorted(Counter(str(row["split_part"]) for row in kept).items())),
        "heldout_canonical_count": len(heldout_canonical),
        "heldout_scaffold_count": len(heldout_scaffolds),
        "canonical_overlap": structure_overlap[combined_label]["canonical_parent"],
        "murcko_scaffold_overlap": structure_overlap[combined_label]["murcko_scaffold"],
        "structure_overlap": structure_overlap,
        "max_tanimoto_to_heldout": max_tanimoto_by_boundary,
        "max_tanimoto_to_heldout_overall": maximum,
        "exact_identity_overlap": exact_overlap,
        "retained_record_identity_sha256": canonical_sha256(
            sorted(str(row["record_id"]) for row in kept)
        ),
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
    if route not in ROUTE_SPLITS:
        raise ValueError(f"Unknown v1.2.47 route: {route}")
    output: list[dict[str, Any]] = []
    for record_id, parent in parent_by_record.items():
        parent_part = str(parent["split_part"])
        if parent_part in SOURCE_PARTS:
            if record_id not in kept_source_ids or route == "M00":
                continue
            if route == "M10" and parent_part == "finetune":
                continue
            split_part = parent_part
            group_key = str(parent.get("group_key") or "")
        elif parent_part in TARGET_PARTS:
            if record_id not in target_assignment:
                continue
            split_part = target_assignment[record_id]
            digest = target_group_by_record[record_id].split(":", 1)[-1]
            group_key = str(parent.get("group_key") or "") + f"|structure_component_sha256={digest}"
        else:
            continue
        row = dict(parent)
        row.update(
            {
                "split_name": split_name,
                "split_part": split_part,
                "seed": int(selected_seed),
                "split_type": f"v1_2_47_scaffold_family_{route}",
                "group_key": group_key,
            }
        )
        output.append(row)
    if not output:
        raise ValueError(f"Derived v1.2.47 route is empty: {route}")
    return output


def persist_route_assignments(
    db_path: Path,
    *,
    source_table: str,
    route_splits: Mapping[str, str],
    route_assignments: Mapping[str, list[dict[str, Any]]],
) -> None:
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


def validate_existing_split_state(
    *,
    db_path: Path,
    source_table: str,
    audit_json: Path,
    route_splits: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    route_splits = validate_route_splits(route_splits)
    summary = json.loads(audit_json.read_text(encoding="utf-8"))
    if summary.get("schema") != SCHEMA or summary.get("status") != "locked":
        raise ValueError("Unexpected or unlocked v1.2.47 split-summary schema/status.")
    expected_contract = summary.get("contract_sha256")
    observed_contract = canonical_sha256(
        {key: value for key, value in summary.items() if key != "contract_sha256"}
    )
    if expected_contract != observed_contract:
        raise ValueError("v1.2.47 split-summary contract hash changed.")
    if summary.get("source_table") != source_table:
        raise ValueError("v1.2.47 split-summary source table changed.")
    validate_audit_file_manifest(audit_json.parent, summary.get("audit_files", {}))
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        parent_rows = read_assignments(conn, str(summary["parent_split"]), source_table)
        if assignment_sha256(parent_rows) != summary["parent_assignment_sha256"]:
            raise ValueError("Locked v1.2.44 parent assignments changed after v1.2.47 build.")
        validated: dict[str, Any] = {}
        for route, split_name in route_splits.items():
            expected = summary["routes"][route]
            if split_name != expected["split_name"]:
                raise ValueError(f"Configured {route} split name differs from locked summary.")
            rows = read_assignments(conn, split_name, source_table)
            observed = assignment_sha256(rows)
            if observed != expected["assignment_sha256"]:
                raise ValueError(f"Persisted {route} assignments changed after lock.")
            validated[route] = {"rows": len(rows), "assignment_sha256": observed}
    for values in summary["target_pool"]["identity_overlap"].values():
        if any(int(value) != 0 for value in values.values()):
            raise ValueError("Locked v1.2.47 target identity overlap is non-zero.")
    for values in summary["target_pool"]["structure_overlap"].values():
        if any(int(value) != 0 for value in values.values()):
            raise ValueError("Locked v1.2.47 target structure overlap is non-zero.")
    if any(
        float(value) >= TANIMOTO_THRESHOLD
        for value in summary["target_pool"]["max_cross_part_tanimoto"].values()
    ):
        raise ValueError("Locked v1.2.47 target maximum similarity violates threshold.")
    task_tiers = summary["partition_policy"].get("task_support_tiers", {})
    if summary["partition_policy"].get("support_tiers") != support_policy()["support_tiers"]:
        raise ValueError("Locked v1.2.47 support-tier policy changed.")
    task_counts = summary["target_pool"].get("task_counts", {})
    if len(task_tiers) != EXPECTED_TARGET_TASKS or set(task_tiers) != set(task_counts):
        raise ValueError("Locked v1.2.47 task support-tier mapping is incomplete.")
    observed_low_support = {
        task for task, tier in task_tiers.items() if tier == "low_support"
    }
    if observed_low_support != LOW_SUPPORT_TASKS:
        raise ValueError(
            "Locked v1.2.47 low-support task set changed: "
            f"{sorted(observed_low_support)}"
        )
    if count_support_violations(task_counts, support_tiers=task_tiers):
        raise ValueError("Locked v1.2.47 per-task tier support gate is violated.")
    reporting_support = summary["target_pool"].get("task_reporting_support", {})
    for task, counts in task_counts.items():
        expected_r2_supported = (
            int(counts["test"]["rows"]) >= R2_SUPPORTED_MIN_TEST_ROWS
            and int(counts["test"]["groups"]) >= R2_SUPPORTED_MIN_TEST_COMPONENTS
        )
        if bool(reporting_support.get(task, {}).get("task_r2_supported")) != expected_r2_supported:
            raise ValueError(f"Locked v1.2.47 task-R2 support flag changed for {task}.")
    source_maximum = summary["source_structure_exclusion"]["max_tanimoto_to_heldout"]
    if any(float(value) >= TANIMOTO_THRESHOLD for value in source_maximum.values()):
        raise ValueError("Locked v1.2.47 source maximum similarity violates threshold.")
    for values in summary["source_structure_exclusion"]["structure_overlap"].values():
        if any(int(value) != 0 for value in values.values()):
            raise ValueError("Locked v1.2.47 source structure overlap is non-zero.")
    for values in summary["source_structure_exclusion"]["exact_identity_overlap"].values():
        if any(int(value) != 0 for value in values.values()):
            raise ValueError("Locked v1.2.47 source identity overlap is non-zero.")
    return {
        "status": "ok",
        "contract_sha256": expected_contract,
        "selected_seed": summary["partition_policy"]["selected_seed"],
        "routes": validated,
    }


def max_cross_similarity(
    left: Iterable[str], right: Iterable[str], fingerprints: Mapping[str, Any]
) -> float:
    from rdkit import DataStructs

    left_values, right_values = sorted(set(left)), sorted(set(right))
    if not left_values or not right_values:
        return 0.0
    right_fps = [fingerprints[item] for item in right_values]
    maximum = 0.0
    for canonical in left_values:
        maximum = max(
            maximum,
            max(float(value) for value in DataStructs.BulkTanimotoSimilarity(fingerprints[canonical], right_fps)),
        )
    return maximum


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


def structure_policy() -> dict[str, Any]:
    return {
        "normalization": (
            "RDKit parse; largest carbon-containing parent fragment; uncharge when possible; "
            "canonical non-isomeric SMILES"
        ),
        "invalid_policy": "exclude_and_audit",
        "acyclic_policy": "retain; empty Murcko scaffold is not a shared scaffold key",
        "component_rule": (
            "union exact canonical parent, shared non-empty Murcko scaffold, and every direct "
            "Morgan pair with Tanimoto >= threshold"
        ),
        "morgan_radius": MORGAN_RADIUS,
        "morgan_bits": MORGAN_BITS,
        "tanimoto_threshold": TANIMOTO_THRESHOLD,
        "source_only_bridge_components": False,
    }


def support_tier_policy(tier: str) -> dict[str, int]:
    if tier == "standard":
        return {
            "minimum_total_rows": STANDARD_SUPPORT_TOTAL_ROWS,
            "minimum_total_components": (
                MIN_TASK_TRAIN_COMPONENTS
                + MIN_TASK_VALID_COMPONENTS
                + MIN_TASK_TEST_COMPONENTS
            ),
            "train_rows": MIN_TASK_TRAIN_ROWS,
            "train_components": MIN_TASK_TRAIN_COMPONENTS,
            "validation_rows": MIN_TASK_VALID_ROWS,
            "validation_components": MIN_TASK_VALID_COMPONENTS,
            "test_rows": MIN_TASK_TEST_ROWS,
            "test_components": MIN_TASK_TEST_COMPONENTS,
        }
    if tier == "low_support":
        return {
            "minimum_total_rows": (
                LOW_SUPPORT_MIN_TASK_TRAIN_ROWS
                + LOW_SUPPORT_MIN_TASK_VALID_ROWS
                + LOW_SUPPORT_MIN_TASK_TEST_ROWS
            ),
            "minimum_total_components": (
                LOW_SUPPORT_MIN_TASK_TRAIN_COMPONENTS
                + LOW_SUPPORT_MIN_TASK_VALID_COMPONENTS
                + LOW_SUPPORT_MIN_TASK_TEST_COMPONENTS
            ),
            "train_rows": LOW_SUPPORT_MIN_TASK_TRAIN_ROWS,
            "train_components": LOW_SUPPORT_MIN_TASK_TRAIN_COMPONENTS,
            "validation_rows": LOW_SUPPORT_MIN_TASK_VALID_ROWS,
            "validation_components": LOW_SUPPORT_MIN_TASK_VALID_COMPONENTS,
            "test_rows": LOW_SUPPORT_MIN_TASK_TEST_ROWS,
            "test_components": LOW_SUPPORT_MIN_TASK_TEST_COMPONENTS,
        }
    raise ValueError(f"Unknown task support tier: {tier!r}")


def support_policy() -> dict[str, Any]:
    return {
        "tier_assignment": (
            f"standard when structure-resolved total rows >= {STANDARD_SUPPORT_TOTAL_ROWS}; "
            "otherwise low_support"
        ),
        "locked_low_support_tasks": sorted(LOW_SUPPORT_TASKS),
        "support_tiers": {
            "standard": support_tier_policy("standard"),
            "low_support": support_tier_policy("low_support"),
        },
        "preferred_validation_rows_per_task": PREFERRED_TASK_VALID_ROWS,
        "preferred_validation_components_per_task": PREFERRED_TASK_VALID_COMPONENTS,
        "task_r2_reporting": {
            "minimum_test_rows": R2_SUPPORTED_MIN_TEST_ROWS,
            "minimum_test_components": R2_SUPPORTED_MIN_TEST_COMPONENTS,
            "unsupported_action": "do_not_report_task_level_r2",
        },
    }


def infeasible_summary(
    *,
    source_table: str,
    parent_split: str,
    parent_assignment_sha256: str,
    target_records: list[dict[str, Any]],
    target_structures: Mapping[str, list[dict[str, Any]]],
    components: Mapping[str, Any],
    feasibility: Mapping[str, Any],
    base_seed: int,
    candidate_count: int,
    audit_dir: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "infeasible",
        "source_table": source_table,
        "parent_split": parent_split,
        "parent_assignment_sha256": parent_assignment_sha256,
        "structure_policy": structure_policy(),
        "partition_policy": {
            "candidate_count": int(candidate_count),
            "base_seed": int(base_seed),
            "uses_model_predictions": False,
            "minimum_support": support_policy(),
            "support_tiers": support_policy()["support_tiers"],
            "task_support_tiers": feasibility["support_tiers"],
        },
        "target_pool": {
            "original_rows": len(target_records),
            "retained_rows": len(target_structures["retained"]),
            "excluded_rows": len(target_structures["excluded"]),
            "original_tasks": len(feasibility["tasks"]),
            "canonical_chemicals": int(components["canonical_count"]),
            "structure_components": int(components["component_count"]),
            "acyclic_chemicals": int(components["acyclic_canonical_count"]),
            "similarity_edges_ge_threshold": int(components["similarity_edge_count"]),
            "feasibility": feasibility,
        },
        "audit_files": audit_file_manifest(
            audit_dir, (EXCLUDED_TARGET_CSV, COMPONENT_CSV)
        ),
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    return summary


def write_exclusion_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fields = (
        "stage",
        "record_id",
        "aggregate_id",
        "split_part",
        "task_head",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
        "canonical_smiles",
        "scaffold_smiles",
        "structure_status",
        "exclusion_reason",
        "max_tanimoto_to_heldout",
    )
    write_csv(path, fields, rows)


def write_component_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    group_by_record: Mapping[str, str],
) -> None:
    output = []
    for row in rows:
        output.append(
            {
                "record_id": row["record_id"],
                "aggregate_id": row["aggregate_id"],
                "task_head": row["task_head"],
                "canonical_smiles": row["canonical_smiles"],
                "scaffold_smiles": row["scaffold_smiles"],
                "structure_component": group_by_record[str(row["record_id"])],
            }
        )
    write_csv(
        path,
        (
            "record_id",
            "aggregate_id",
            "task_head",
            "canonical_smiles",
            "scaffold_smiles",
            "structure_component",
        ),
        output,
    )


def write_task_support_csv(
    path: Path,
    support: Mapping[str, Any],
    *,
    reporting_support: Mapping[str, Any],
) -> None:
    rows = []
    for task, values in sorted(support.items()):
        for part, counts in values.items():
            rows.append(
                {
                    "task_head": task,
                    "split_part": part,
                    "rows": counts["rows"],
                    "structure_components": counts["groups"],
                    "support_tier": reporting_support[task]["support_tier"],
                    "low_support": reporting_support[task]["low_support"],
                    "task_r2_supported": reporting_support[task]["task_r2_supported"],
                    "validation_support_warning": (
                        part == "valid"
                        and (
                            counts["rows"] < PREFERRED_TASK_VALID_ROWS
                            or counts["groups"] < PREFERRED_TASK_VALID_COMPONENTS
                        )
                    ),
                }
            )
    write_csv(
        path,
        (
            "task_head",
            "split_part",
            "rows",
            "structure_components",
            "support_tier",
            "low_support",
            "task_r2_supported",
            "validation_support_warning",
        ),
        rows,
    )


def write_overlap_csv(
    path: Path, *, target_audit: Mapping[str, Any], source_audit: Mapping[str, Any]
) -> None:
    rows = []
    for label in target_audit["structure_overlap"]:
        rows.append(
            {
                "boundary": label,
                **target_audit["identity_overlap"][label],
                **target_audit["structure_overlap"][label],
                "max_tanimoto": target_audit["max_cross_part_tanimoto"][label],
            }
        )
    for boundary, overlap in source_audit["structure_overlap"].items():
        rows.append(
            {
                "boundary": boundary,
                **source_audit["exact_identity_overlap"][boundary],
                **overlap,
                "structure_component": 0,
                "max_tanimoto": source_audit["max_tanimoto_to_heldout"][boundary],
            }
        )
    write_csv(
        path,
        (
            "boundary",
            "aggregate_id",
            "result_id",
            "test_id",
            "canonical_parent",
            "murcko_scaffold",
            "structure_component",
            "max_tanimoto",
        ),
        rows,
    )


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_file_manifest(directory: Path, names: Iterable[str]) -> dict[str, Any]:
    output = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        output[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return output


def validate_audit_file_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    for name, expected in manifest.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Locked v1.2.47 audit file changed or is missing: {name}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_locked_json(path: Path, summary: dict[str, Any]) -> None:
    if "contract_sha256" not in summary:
        summary["contract_sha256"] = canonical_sha256(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
