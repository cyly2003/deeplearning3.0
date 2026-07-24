from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_v1_2_44_second_layer_splits import (
    SOURCE_TABLE,
    audit_route_assignments,
    build_route_assignments,
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
)
from scripts.validate_v1_2_44_matrix_run import (
    PREDICTION_PARTS,
    validate_run,
    validate_split_summary,
)


def parent_row(record_id: str, part: str, group: str) -> dict[str, object]:
    return {
        "split_name": "parent",
        "record_id": record_id,
        "aggregate_id": f"aggregate-{record_id}",
        "split_part": part,
        "seed": 42,
        "split_type": "parent",
        "source_table": SOURCE_TABLE,
        "group_key": group,
    }


def test_route_builder_removes_wrong_stages_and_locks_stage3() -> None:
    stage1 = "stage_contract_v1|stage=aquatic_ptox_stage1|medium_domain=aquatic|target_name=ptox_mol_l|target_family=aquatic_pTox_mol_L"
    stage2 = "stage_contract_v1|stage=soil_ptox_stage2_train|medium_domain=soil|target_name=ptox_mol_l|target_family=aquatic_pTox_mol_L"
    stage3 = "stage_contract_v1|stage=soil_molkg_stage3_train|medium_domain=soil|target_name=neg_log10_mol_kg|target_family=solid_neglog_mol_kg"
    rows = [
        parent_row("water", "train", stage1),
        parent_row("soil", "finetune", stage2),
        parent_row("stage3-train", "finetune_mgkg", stage3),
        parent_row("stage3-valid", "finetune_mgkg", stage3),
        parent_row("stage3-excluded", "finetune_mgkg", stage3),
        parent_row("stage3-test", "test", stage3),
    ]
    eligible = {"stage3-train", "stage3-valid", "stage3-test"}
    validation = {"stage3-valid"}
    routes = {
        route: build_route_assignments(
            rows,
            route=route,
            split_name=route,
            validation_record_ids=validation,
            eligible_stage3_record_ids=eligible,
            eligible_parent_record_ids={"water", "soil", *eligible},
        )
        for route in ("M11", "M00", "M10", "M01")
    }
    assert {row["record_id"] for row in routes["M00"]} == {
        "stage3-train", "stage3-valid", "stage3-test"
    }
    assert {row["record_id"] for row in routes["M10"]} == {
        "water", "stage3-train", "stage3-valid", "stage3-test"
    }
    assert {row["record_id"] for row in routes["M01"]} == {
        "soil", "stage3-train", "stage3-valid", "stage3-test"
    }
    assert next(row for row in routes["M01"] if row["record_id"] == "soil")["split_part"] == "train"
    audits = {route: audit_route_assignments(value, route=route) for route, value in routes.items()}
    assert len({audit["stage3_boundary_sha256"] for audit in audits.values()}) == 1


def make_summary(prediction_rows: list[dict[str, str]]) -> dict[str, object]:
    identity = {}
    for part in PREDICTION_PARTS:
        records = [
            evaluation_identity_record(row, split_part=part)
            for row in prediction_rows
            if row["split_part"] == part
        ]
        identity[part] = {
            "rows": len(records),
            "sha256": canonical_sha256(sorted(records, key=canonical_json)),
        }
    stage_hash = "same-stage3"
    summary: dict[str, object] = {
        "schema_version": 1,
        "matrix_version": "v1.2.44",
        "source_table": SOURCE_TABLE,
        "stage3_boundary": {
            "train_rows": 9724,
            "validation_rows": 2433,
            "test_rows": 3042,
            "stage3_boundary_sha256": stage_hash,
            "evaluation_identity": identity,
            "task_counts": [{"task_head": f"task-{i}"} for i in range(18)],
        },
        "routes": {
            route: {"split_name": f"split-{route}", "stage3_boundary_sha256": stage_hash}
            for route in ("M11", "M00", "M10", "M01")
        },
        "m01_task_filter_exception": {
            "task_filter_min_total": 0,
            "task_filter_min_train": 0,
            "task_filter_min_eval": 0,
            "validation_seed": 17073,
            "train_rows": 80,
            "validation_rows": 20,
        },
        "m10_task_filter_exception": {
            "task_filter_min_total": 0,
            "task_filter_min_train": 0,
            "task_filter_min_eval": 0,
            "validation_seed": 42,
            "train_rows": 100,
            "validation_rows": 10,
        },
    }
    summary["contract_sha256"] = canonical_sha256(summary)
    return summary


def prediction_rows() -> list[dict[str, str]]:
    rows = []
    for index, part in enumerate(PREDICTION_PARTS):
        rows.append(
            {
                "aggregate_id": f"agg-{index}",
                "result_ids": json.dumps([f"result-{index}"]),
                "split_part": part,
                "task_head": "task__solid_neglog_mol_kg",
                "base_task_head": "task",
                "target_name": "neg_log10_mol_kg",
                "target_family": "solid_neglog_mol_kg",
                "medium_domain": "soil",
            }
        )
    return rows


def write_fixture(tmp_path: Path, *, cell: str = "M10") -> tuple[Path, Path]:
    rows = prediction_rows()
    summary = make_summary(rows)
    summary_path = tmp_path / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("history.csv", "best_model.pt", "preprocessing.json"):
        (run_dir / name).write_text("x", encoding="utf-8")
    ablation = "full"
    split = "split-M10"
    epochs = (30, 0, 30)
    min_eval = 30
    min_total = 200
    min_train = 100
    learning_rate = 0.0005
    validation_seed = 42
    if cell == "M00":
        split = "split-M00"; epochs = (0, 0, 30)
    if cell == "M10":
        min_total = 0; min_train = 0; min_eval = 0
    if cell == "M01":
        split = "split-M01"; epochs = (20, 0, 30); min_eval = 0
        min_total = 0; min_train = 0
        learning_rate = 0.0001; validation_seed = 17073
    manifest = {
        "seed": 42,
        "split_name": split,
        "data_source": {"source_table": SOURCE_TABLE},
        "head_routing": "task_target",
        "allow_mixed_target_dimensions": True,
        "epochs": epochs[0],
        "finetune_epochs": epochs[1],
        "finetune_mgkg_epochs": epochs[2],
        "ablation": ablation,
        "task_filter": {"min_total": min_total, "min_train": min_train, "min_eval": min_eval},
        "prediction_output_split_parts": list(PREDICTION_PARTS),
        "ablation_features": {
            "use_descriptors": True,
            "use_fingerprint": True,
            "use_context_numeric": True,
            "use_species_lifestage": True,
            "use_other_categorical_context": True,
            "use_medium_adapter": False,
        },
        "finetune_rows": 0,
        "finetune": {"freeze": "none", "learning_rate": 0.0001},
        "finetune_mgkg": {
            "freeze": "none",
            "validation_source": "valid",
            "validation_rows": 2433,
            "learning_rate": 0.0005,
            "trunk_learning_rate": None,
            "soil_ptox_replay_fraction_requested": 0.0,
            "stage3_init_checkpoint": {
                "loaded": False, "exported": False, "stage1_stage2_skipped": False
            },
        },
        "learning_rate": learning_rate,
        "validation_seed": validation_seed,
        "actual_train_rows": 80 if cell == "M01" else 100,
        "validation_rows": 20 if cell == "M01" else 10,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, summary_path


def test_validator_accepts_m10_and_rejects_prediction_drift(tmp_path: Path) -> None:
    run_dir, summary_path = write_fixture(tmp_path)
    result = validate_run(run_dir, cell="M10", seed=42, split_summary=summary_path)
    assert result["status"] == "ok"
    prediction_path = run_dir / "predictions.csv"
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8-sig", newline="")))
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    assert validate_run(run_dir, cell="M10", seed=42, split_summary=summary_path)["status"] == "ok"
    text = prediction_path.read_text(encoding="utf-8-sig").replace("agg-0", "tampered")
    prediction_path.write_text(text, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="prediction identity mismatch"):
        validate_run(run_dir, cell="M10", seed=42, split_summary=summary_path)


def test_m01_exception_is_explicit_and_seed_locked(tmp_path: Path) -> None:
    run_dir, summary_path = write_fixture(tmp_path, cell="M01")
    assert validate_run(run_dir, cell="M01", seed=42, split_summary=summary_path)["status"] == "ok"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_seed"] = 42
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_seed"):
        validate_run(run_dir, cell="M01", seed=42, split_summary=summary_path)


def test_validator_accepts_strict_m00_on_m11_stage3_boundary(tmp_path: Path) -> None:
    run_dir, summary_path = write_fixture(tmp_path, cell="M00")
    result = validate_run(run_dir, cell="M00", seed=42, split_summary=summary_path)
    assert result["status"] == "ok"


def test_stage3_only_runtime_respects_explicit_zero_stage1_epochs() -> None:
    source = Path("qsar_tl/training/deep_experiment.py").read_text(encoding="utf-8")
    assert 'epochs if epochs is not None else train_cfg.get("epochs", 5)' in source
    assert "resolve_stage1_training_boundary(" in source
    assert "if actual_train_indices\n        else None" in source


def test_split_summary_requires_exact_v40_boundary() -> None:
    summary = make_summary(prediction_rows())
    summary["stage3_boundary"]["test_rows"] = 3041  # type: ignore[index]
    summary["contract_sha256"] = canonical_sha256(
        {key: value for key, value in summary.items() if key != "contract_sha256"}
    )
    with pytest.raises(ValueError, match="boundary changed"):
        validate_split_summary(summary)


def test_launcher_locks_matrix_names_routes_and_runtime_prediction_parts() -> None:
    runner = Path("scripts/run_v1_2_44_second_layer_matrix_remote.sh").read_text(encoding="utf-8")
    assert 'FORMAL_ROOT="outputs/experiments/第二层核心因果实验矩阵_v1_2_44"' in runner
    assert "M00_从头训练_固定评价边界_种子${seed}" in runner
    assert 'M00_SPLIT="M_v1_2_44_M00_仅Stage3从头训练_固定评价边界"' in runner
    assert "M01_土壤pTox后迁移_种子${seed}" in runner
    assert "M11F_完整三阶段冻结主干_种子${seed}" in runner
    assert "M11U复现_完整三阶段全参数微调_种子${seed}" in runner
    assert "PREDICTION_PARTS=(finetune_mgkg finetune_mgkg_validation test)" in runner
    assert "TASK_MIN_EVAL=0" in runner
    assert "TASK_MIN_TOTAL=0; TASK_MIN_TRAIN=0" in runner
    assert "STAGE1_VALID_SEED=17073" in runner
    assert "for cell in M00 M10 M01 M11F M11U B2_CONTEXT B2_MOLECULE" in runner
    assert 'validate_cell M11F "$seed"' in runner
    assert "validate_cell() (" in runner
    assert "SEEDS=(42 2042 3407 8417)" in runner
    assert 'if [[ "$MODE" == "smoke" ]]' in runner
    assert 'scripts/summarize_v1_2_44_second_layer_matrix.py' in runner
    assert '--output-dir "$RUN_ROOT/统一汇总_中文"' in runner
    assert 'REBUILD_SPLITS="${REBUILD_SPLITS:-1}"' in runner
    assert "scripts/validate_v1_2_44_split_state.py" in runner
