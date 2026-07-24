from __future__ import annotations

import json

import pytest

from scripts.build_v1_2_47_scaffold_family_splits import (
    LOW_SUPPORT_TASKS,
    PART_RATIOS,
    SOURCE_PARTS,
    audit_source_isolation,
    build_route_assignments,
    build_target_structure_components,
    choose_structure_partition,
    count_identity_partition_violations,
    count_support_violations,
    filter_source_structures,
    normalize_record_structures,
    repair_structure_partition,
    support_policy,
    target_feasibility,
)


def record(
    identity: str,
    *,
    task: str = "task",
    smiles: str = "CCO",
    split_part: str = "finetune_mgkg",
    target: float = 1.0,
    test_id: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": identity,
        "aggregate_id": identity,
        "split_part": split_part,
        "task_head": task,
        "target_value": target,
        "smiles": smiles,
        "cas_number": "",
        "dtxsid": "",
        "chemical_name": identity,
        "test_ids": json.dumps([test_id or f"test-{identity}"]),
        "result_ids": json.dumps([f"result-{identity}"]),
    }


def test_normalization_retains_acyclic_and_excludes_invalid_structure() -> None:
    result = normalize_record_structures(
        [record("acyclic", smiles="CCO"), record("invalid", smiles="[Na+]")],
        stage="stage3",
    )
    assert [row["record_id"] for row in result["retained"]] == ["acyclic"]
    assert result["retained"][0]["scaffold_smiles"] == ""
    assert [row["record_id"] for row in result["excluded"]] == ["invalid"]
    assert result["excluded"][0]["exclusion_reason"]


def test_structure_components_use_scaffold_but_not_empty_scaffold_as_key() -> None:
    normalized = normalize_record_structures(
        [
            record("benzene", smiles="c1ccccc1"),
            record("toluene", smiles="Cc1ccccc1"),
            record("ethanol", smiles="CCO"),
            record("butanol", smiles="CCCCO"),
        ],
        stage="stage3",
    )["retained"]
    components = build_target_structure_components(
        normalized,
        tanimoto_threshold=0.999,
        morgan_radius=2,
        morgan_bits=256,
    )
    groups = components["group_by_record"]
    assert groups["benzene"] == groups["toluene"]
    assert groups["ethanol"] != groups["butanol"]
    assert components["acyclic_canonical_count"] == 2


def make_locked_support_fixture() -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    standard_tasks = [f"standard-{index}" for index in range(16)]
    task_sizes = {task: 131 for task in standard_tasks}
    task_sizes.update({"ECx_Population": 85, "ICx_Growth": 51})
    for task, size in task_sizes.items():
        group_count = 11 if size >= 131 else 3
        for index in range(size):
            identity = f"{task}-{index}"
            rows.append(record(identity, task=task, target=float(index)))
            groups[identity] = f"group-{task}-{index % group_count}"
    return rows, groups


def test_target_feasibility_uses_preregistered_two_tier_policy() -> None:
    rows, groups = make_locked_support_fixture()
    result = target_feasibility(rows, retained_records=rows, group_by_record=groups)
    assert result["feasible"] is True
    assert set(result["observed_low_support_tasks"]) == LOW_SUPPORT_TASKS
    assert result["task_support"]["ECx_Population"]["support_tier"] == "low_support"
    assert result["task_support"]["standard-0"]["support_tier"] == "standard"

    reduced = [row for row in rows if row["task_head"] != "ICx_Growth"] + [
        row for row in rows if row["task_head"] == "ICx_Growth"
    ][:30]
    reduced_groups = {str(row["record_id"]): groups[str(row["record_id"])] for row in reduced}
    failed = target_feasibility(
        reduced,
        retained_records=reduced,
        group_by_record=reduced_groups,
    )
    assert failed["feasible"] is False
    assert "ICx_Growth" in failed["failures"]


def test_support_gate_changes_with_task_tier_and_locks_r2_rule() -> None:
    support = {
        "standard": {
            "finetune_mgkg": {"rows": 100, "groups": 5},
            "valid": {"rows": 1, "groups": 1},
            "test": {"rows": 30, "groups": 5},
        },
        "low": {
            "finetune_mgkg": {"rows": 20, "groups": 1},
            "valid": {"rows": 1, "groups": 1},
            "test": {"rows": 10, "groups": 1},
        },
    }
    tiers = {"standard": "standard", "low": "low_support"}
    assert count_support_violations(support, support_tiers=tiers) == 0
    assert count_support_violations(support) == 4
    policy = support_policy()
    assert policy["task_r2_reporting"]["minimum_test_rows"] == 30
    assert policy["task_r2_reporting"]["minimum_test_components"] == 5


def test_candidate_selection_is_prediction_blind_deterministic_and_disjoint() -> None:
    rows: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    for task_index in range(2):
        for index in range(240):
            identity = f"t{task_index}-{index}"
            rows.append(
                record(identity, task=f"task-{task_index}", target=float(index % 25))
            )
            groups[identity] = f"group-{identity}"
    tiers = {"task-0": "standard", "task-1": "standard"}
    first, first_selection = choose_structure_partition(
        rows,
        group_by_record=groups,
        candidate_count=256,
        base_seed=20260721,
        support_tiers=tiers,
    )
    second, second_selection = choose_structure_partition(
        list(reversed(rows)),
        group_by_record=groups,
        candidate_count=256,
        base_seed=20260721,
        support_tiers=tiers,
    )
    assert first == second
    assert first_selection["selected_seed"] == second_selection["selected_seed"]
    assert first_selection["identity_partition_violations"] == 0
    assert set(first.values()) == set(PART_RATIOS)


def test_structure_group_repair_meets_low_support_without_relaxing_gates() -> None:
    rows: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    group_part: dict[str, str] = {}
    background_parts = ["finetune_mgkg"] * 128 + ["valid"] * 32 + ["test"] * 40
    for index, part in enumerate(background_parts):
        identity = f"standard-{index}"
        group = f"standard-group-{index}"
        rows.append(record(identity, task="standard", target=float(index)))
        groups[identity] = group
        group_part[group] = part

    low_groups = [
        ("low-giant", 38, "finetune_mgkg"),
        ("low-train", 3, "finetune_mgkg"),
        ("low-valid", 3, "valid"),
        ("low-test-a", 3, "test"),
        ("low-test-b", 3, "test"),
        ("low-single", 1, "finetune_mgkg"),
    ]
    for group, size, part in low_groups:
        group_part[group] = part
        for index in range(size):
            identity = f"{group}-{index}"
            rows.append(record(identity, task="low", target=float(index)))
            groups[identity] = group

    assignment, metadata = repair_structure_partition(
        rows,
        group_part=group_part,
        group_by_record=groups,
        expected_tasks=["standard", "low"],
        support_tiers={"standard": "standard", "low": "low_support"},
        selected_seed=20260721,
    )
    assert metadata["violations"] == 0
    assert metadata["selected_via_constraint_repair"] is True
    assert metadata["support"]["low"]["test"]["rows"] >= 10
    assert metadata["support"]["low"]["finetune_mgkg"]["rows"] >= 20
    assert set(assignment.values()) == set(PART_RATIOS)


def test_identity_overlap_is_a_candidate_hard_gate() -> None:
    rows = [
        record("a", test_id="shared"),
        record("b", test_id="shared"),
        record("c"),
    ]
    assignment = {"a": "finetune_mgkg", "b": "test", "c": "valid"}
    assert count_identity_partition_violations(rows, assignment=assignment) == 1


def test_source_filter_excludes_exact_scaffold_similarity_and_invalid_rows() -> None:
    target = normalize_record_structures(
        [record("heldout", smiles="c1ccccc1")], stage="stage3"
    )["retained"]
    target_components = build_target_structure_components(target, morgan_bits=256)
    heldout_canonical = {str(target[0]["canonical_smiles"])}
    heldout_scaffolds = {str(target[0]["scaffold_smiles"])}
    heldout_fingerprints = [
        target_components["fingerprints_by_canonical"][str(target[0]["canonical_smiles"])]
    ]
    normalized = normalize_record_structures(
        [
            record("exact", smiles="c1ccccc1", split_part="train"),
            record("scaffold", smiles="Cc1ccccc1", split_part="train"),
            record("far", smiles="CCO", split_part="finetune"),
            record("invalid", smiles="[Na+]", split_part="train"),
        ],
        stage="source",
    )
    kept, excluded = filter_source_structures(
        normalized["retained"],
        invalid_exclusions=normalized["excluded"],
        heldout_canonical=heldout_canonical,
        heldout_scaffolds=heldout_scaffolds,
        heldout_fingerprints=heldout_fingerprints,
        tanimoto_threshold=0.65,
        morgan_radius=2,
        morgan_bits=256,
    )
    assert kept == {"far"}
    reasons = {str(row["record_id"]): str(row["exclusion_reason"]) for row in excluded}
    assert reasons["exact"] == "heldout_canonical_parent"
    assert reasons["scaffold"] == "heldout_murcko_scaffold"
    assert reasons["invalid"]


def test_source_audit_exposes_numeric_boundary_trees() -> None:
    source = normalize_record_structures(
        [
            record("source-a", smiles="CCO", split_part="train"),
            record("source-b", smiles="CCC", split_part="finetune"),
        ],
        stage="source",
    )["retained"]
    target = normalize_record_structures(
        [
            record("valid-a", smiles="c1ccccc1"),
            record("test-a", smiles="C1CCCCC1"),
        ],
        stage="stage3",
    )["retained"]
    assignment = {"valid-a": "valid", "test-a": "test"}
    components = build_target_structure_components(target, morgan_bits=256)
    heldout_canonical = {str(row["canonical_smiles"]) for row in target}
    heldout_scaffolds = {
        str(row["scaffold_smiles"]) for row in target if str(row["scaffold_smiles"])
    }
    heldout_fingerprints = [
        components["fingerprints_by_canonical"][canonical]
        for canonical in sorted(heldout_canonical)
    ]
    audit = audit_source_isolation(
        source,
        kept_source_ids={"source-a", "source-b"},
        excluded_records=[],
        target_records=target,
        assignment=assignment,
        heldout_canonical=heldout_canonical,
        heldout_scaffolds=heldout_scaffolds,
        heldout_fingerprints=heldout_fingerprints,
        tanimoto_threshold=0.65,
        morgan_radius=2,
        morgan_bits=256,
    )
    assert set(audit["remaining_by_part"]) == SOURCE_PARTS
    assert audit["missing_or_invalid_structure_policy"] == "exclude_and_report"
    assert set(audit["structure_overlap"]) == {
        "source_train_vs_stage3_valid",
        "source_train_vs_stage3_test",
        "source_train_vs_stage3_valid_test",
    }
    assert all(
        isinstance(value, float) for value in audit["max_tanimoto_to_heldout"].values()
    )


def parent_row(identity: str, split_part: str) -> dict[str, object]:
    return {
        "record_id": identity,
        "aggregate_id": identity,
        "split_part": split_part,
        "seed": 42,
        "split_type": "parent",
        "source_table": "table",
        "group_key": f"parent-{identity}",
    }


def test_route_assignments_preserve_m00_m10_m11u_semantics() -> None:
    parent = {
        "aquatic": parent_row("aquatic", "train"),
        "soil-ptox": parent_row("soil-ptox", "finetune"),
        "target-train": parent_row("target-train", "finetune_mgkg"),
        "target-valid": parent_row("target-valid", "valid"),
        "target-test": parent_row("target-test", "test"),
    }
    target_assignment = {
        "target-train": "test",
        "target-valid": "finetune_mgkg",
        "target-test": "valid",
    }
    target_groups = {
        identity: f"structure_component_v1:{identity}" for identity in target_assignment
    }
    kept_source = {"aquatic", "soil-ptox"}
    outputs = {
        route: build_route_assignments(
            parent_by_record=parent,
            route=route,
            split_name=f"split-{route}",
            selected_seed=20260721,
            target_assignment=target_assignment,
            target_group_by_record=target_groups,
            kept_source_ids=kept_source,
        )
        for route in ("M00", "M10", "M11U")
    }
    identities = {
        route: {str(row["record_id"]) for row in rows} for route, rows in outputs.items()
    }
    assert identities["M00"] == set(target_assignment)
    assert identities["M10"] == set(target_assignment) | {"aquatic"}
    assert identities["M11U"] == set(target_assignment) | kept_source
    assert all(
        "structure_component_sha256=" in str(row["group_key"])
        for row in outputs["M11U"]
        if str(row["record_id"]).startswith("target-")
    )
