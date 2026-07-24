from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_v1_2_44_second_layer_splits import (
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
)
from scripts.build_v1_2_46_reference_group_splits import (
    PART_RATIOS,
    audit_target_partition,
    build_reference_components,
    choose_reference_partition,
    parse_string_set,
)
from scripts.summarize_v1_2_45_reference_cluster_bootstrap import (
    paired_cluster_bootstrap,
)
from scripts.validate_v1_2_46_reference_group_run import (
    PREDICTION_PARTS,
    validate_run,
)


def target_record(
    record_id: str,
    *,
    task: str,
    reference_numbers: list[str],
    target: float = 1.0,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "aggregate_id": record_id,
        "split_part": "finetune_mgkg",
        "task_head": task,
        "target_value": target,
        "reference_numbers": json.dumps(reference_numbers),
        "test_ids": json.dumps([f"test-{record_id}"]),
        "result_ids": json.dumps([f"result-{record_id}"]),
    }


def test_reference_components_resolve_transitive_multi_reference_rows() -> None:
    rows = [
        target_record("a", task="task", reference_numbers=["A", "B"]),
        target_record("b", task="task", reference_numbers=["B", "C"]),
        target_record("c", task="task", reference_numbers=["D"]),
    ]
    result = build_reference_components(rows)
    assert result["component_count"] == 2
    assert result["group_by_record"]["a"] == result["group_by_record"]["b"]
    assert result["group_by_record"]["a"] != result["group_by_record"]["c"]
    assert result["multi_reference_rows"] == 2


def test_reference_parser_is_fail_closed() -> None:
    assert parse_string_set('["2", "1", "2"]', field="refs") == {"1", "2"}
    with pytest.raises(ValueError, match="valid JSON"):
        parse_string_set("[broken", field="refs")
    with pytest.raises(ValueError, match="JSON array"):
        parse_string_set('"one"', field="refs")


def test_candidate_partition_is_prediction_blind_deterministic_and_disjoint() -> None:
    rows = []
    for task_index in range(2):
        for index in range(240):
            rows.append(
                target_record(
                    f"t{task_index}-{index}",
                    task=f"task-{task_index}",
                    reference_numbers=[f"ref-{task_index}-{index}"],
                    target=float(index % 25),
                )
            )
    components = build_reference_components(rows)
    first, first_selection = choose_reference_partition(
        rows,
        group_by_record=components["group_by_record"],
        candidate_count=256,
        base_seed=20260720,
    )
    second, second_selection = choose_reference_partition(
        list(reversed(rows)),
        group_by_record=components["group_by_record"],
        candidate_count=256,
        base_seed=20260720,
    )
    assert first == second
    assert first_selection["selected_seed"] == second_selection["selected_seed"]
    assert set(first.values()) == set(PART_RATIOS)
    audit = audit_target_partition(
        rows,
        assignment=first,
        group_by_record=components["group_by_record"],
        references_by_record=components["references_by_record"],
    )
    assert all(value == 0 for value in audit["reference_overlap"].values())
    assert all(
        all(value == 0 for value in overlap.values())
        for overlap in audit["identity_overlap"].values()
    )


def normalized_row(identity: str, y: float, pred: float) -> dict[str, object]:
    return {
        "aggregate_id": identity,
        "task_head": "task",
        "y_molkg": y,
        "pred_molkg": pred,
        "y_mgkg": y - 3.0,
        "pred_mgkg": pred - 3.0,
        "result_ids": (f"result-{identity}",),
    }


def test_cluster_bootstrap_preserves_paired_direction() -> None:
    ensembles = {"new": [], "control": []}
    groups = {}
    for group in range(6):
        for within in range(3):
            identity = f"g{group}-r{within}"
            y = float(group + within / 5)
            ensembles["new"].append(normalized_row(identity, y, y + 0.05))
            ensembles["control"].append(normalized_row(identity, y, y + 0.40))
            groups[identity] = f"group-{group}"
    rows, _ = paired_cluster_bootstrap(
        ensembles,
        group_by_aggregate=groups,
        contrasts=(("new_minus_control", "new", "control", "molkg", "test"),),
        replicates=500,
        seed=42,
        equivalence_margin=0.01,
    )
    assert rows[0]["delta_mae"] < 0
    assert rows[0]["delta_mae_ci95_high"] < 0
    assert rows[0]["new_model_mae_superior_95"] is True


def prediction_fixture_rows() -> list[dict[str, str]]:
    rows = []
    for index, part in enumerate(PREDICTION_PARTS):
        rows.append(
            {
                "aggregate_id": f"agg-{index}",
                "result_ids": json.dumps([f"result-{index}"]),
                "split_part": part,
                "task_head": "task",
                "base_task_head": "task",
                "model_head": "task__solid_neglog_mol_kg",
                "target_name": "neg_log10_mol_kg",
                "target_family": "solid_neglog_mol_kg",
                "medium_domain": "soil",
            }
        )
    return rows


def split_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    identity = {}
    for part in PREDICTION_PARTS:
        selected = [row for row in rows if row["split_part"] == part]
        records = [evaluation_identity_record(row, split_part=part) for row in selected]
        identity[part] = {
            "rows": len(records),
            "sha256": canonical_sha256(sorted(records, key=canonical_json)),
        }
    summary: dict[str, object] = {
        "schema": "v1_2_46_reference_group_split_v1",
        "status": "locked",
        "source_table": "aggregated_task_records_ptox_soil_mass_molar_qc",
        "partition_policy": {
            "minimum_support": {
                "train_rows_per_task": 100,
                "validation_rows_per_task": 1,
                "test_rows_per_task": 30,
                "validation_groups_per_task": 1,
                "test_groups_per_task": 5,
            },
            "uses_model_predictions": False,
        },
        "target_pool": {
            "tasks": 18,
            "counts": {"finetune_mgkg": 9724, "valid": 2433, "test": 3042},
            "reference_overlap": {
                "finetune_mgkg_vs_valid": 0,
                "finetune_mgkg_vs_test": 0,
                "valid_vs_test": 0,
            },
            "identity_overlap": {
                "finetune_mgkg_vs_valid": {"aggregate_id": 0},
                "finetune_mgkg_vs_test": {"aggregate_id": 0},
                "valid_vs_test": {"aggregate_id": 0},
            },
            "evaluation_identity": identity,
        },
        "routes": {
            cell: {"split_name": f"split-{cell}"} for cell in ("M00", "M10", "M11U")
        },
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    return summary


def write_run_fixture(tmp_path: Path, *, cell: str) -> tuple[Path, Path]:
    rows = prediction_fixture_rows()
    summary = split_summary(rows)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("history.csv", "best_model.pt", "preprocessing.json"):
        (run_dir / name).write_text("x", encoding="utf-8")
    epochs = {"M00": (0, 0, 30), "M10": (30, 0, 30), "M11U": (30, 20, 30)}[cell]
    weighting = "tanimoto_to_finetune" if cell == "M11U" else "none"
    manifest = {
        "seed": 42,
        "split_name": f"split-{cell}",
        "data_source": {"source_table": "aggregated_task_records_ptox_soil_mass_molar_qc"},
        "head_routing": "task_target",
        "allow_mixed_target_dimensions": True,
        "epochs": epochs[0],
        "finetune_epochs": epochs[1],
        "finetune_mgkg_epochs": epochs[2],
        "ablation": "full",
        "task_filter": {"min_total": 0, "min_train": 0, "min_eval": 0},
        "prediction_output_split_parts": list(PREDICTION_PARTS),
        "ablation_features": {
            "use_descriptors": True,
            "use_fingerprint": True,
            "use_context_numeric": True,
            "use_species_lifestage": True,
            "use_other_categorical_context": True,
            "use_medium_adapter": False,
        },
        "finetune_rows": 10 if cell == "M11U" else 0,
        "finetune": {"freeze": "none", "learning_rate": 0.0001},
        "finetune_mgkg": {
            "freeze": "none",
            "learning_rate": 0.0005,
            "trunk_learning_rate": None,
            "validation_source": "valid",
            "validation_rows": 2433,
            "soil_ptox_replay_fraction_requested": 0.0,
            "stage3_init_checkpoint": {
                "loaded": False,
                "exported": False,
                "stage1_stage2_skipped": False,
            },
        },
        "source_weighting": {"method": weighting, "applied": cell == "M11U"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, summary_path


def test_reference_group_run_validator_accepts_locked_full_finetune(tmp_path: Path) -> None:
    run_dir, summary_path = write_run_fixture(tmp_path, cell="M11U")
    result = validate_run(
        run_dir,
        cell="M11U",
        seed=42,
        split_summary=summary_path,
    )
    assert result["status"] == "ok"


def test_launcher_locks_smoke_formal_matrix_contract() -> None:
    text = Path("scripts/run_v1_2_46_reference_group_matrix_remote.sh").read_text(
        encoding="utf-8"
    )
    assert "for cell in M00 M10 M11U" in text
    assert "SEEDS=(42 2042 3407 8417)" in text
    assert "--finetune-mgkg-monitor-split valid" in text
    assert "--finetune-mgkg-validation-fraction 0" in text
    assert "--finetune-mgkg-freeze none" in text
    assert "--no-medium-adapters" in text
    assert "REBUILD_SPLITS=0" not in text
    assert "[smoke_complete]" in text
    assert "[matrix_complete]" in text
