from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_v1_2_44_second_layer_splits import (
    canonical_json,
    canonical_sha256,
    evaluation_identity_record,
)
from scripts.summarize_v1_2_47_scaffold_family_matrix import (
    load_test_component_mapping,
)
from scripts.validate_v1_2_47_scaffold_family_run import (
    PREDICTION_PARTS,
    validate_run,
)


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
    tasks = (
        "ECx_Growth",
        "ECx_Mortality",
        "ECx_Population",
        "ECx_Reproduction",
        "ICx_Growth",
        "LOEC_Biochemical",
        "LOEC_GeneticDamage",
        "LOEC_Growth",
        "LOEC_Mortality",
        "LOEC_Physiology",
        "LOEC_Population",
        "LOEC_Reproduction",
        "NOEC_Biochemical",
        "NOEC_Growth",
        "NOEC_Mortality",
        "NOEC_Physiology",
        "NOEC_Population",
        "NOEC_Reproduction",
    )
    low_support = {"ECx_Population", "ICx_Growth"}
    task_counts = {
        task: {
            "finetune_mgkg": {
                "rows": 20 if task in low_support else 100,
                "groups": 1 if task in low_support else 5,
            },
            "valid": {"rows": 1, "groups": 1},
            "test": {
                "rows": 10 if task in low_support else 30,
                "groups": 1 if task in low_support else 5,
            },
        }
        for task in tasks
    }
    support_tiers = {
        "standard": {
            "train_rows": 100,
            "train_components": 5,
            "validation_rows": 1,
            "validation_components": 1,
            "test_rows": 30,
            "test_components": 5,
        },
        "low_support": {
            "train_rows": 20,
            "train_components": 1,
            "validation_rows": 1,
            "validation_components": 1,
            "test_rows": 10,
            "test_components": 1,
        },
    }
    summary: dict[str, object] = {
        "schema": "v1_2_47_scaffold_family_split_v1",
        "status": "locked",
        "source_table": "aggregated_task_records_ptox_soil_mass_molar_qc",
        "partition_policy": {
            "uses_model_predictions": False,
            "tanimoto_threshold": 0.65,
            "support_tiers": support_tiers,
            "task_support_tiers": {
                task: ("low_support" if task in low_support else "standard")
                for task in tasks
            },
            "minimum_support": {
                "locked_low_support_tasks": ["ECx_Population", "ICx_Growth"],
                "support_tiers": support_tiers,
            },
        },
        "target_pool": {
            "original_rows": 15199,
            "tasks": 18,
            "counts": {"finetune_mgkg": 1, "valid": 1, "test": 1},
            "task_counts": task_counts,
            "task_reporting_support": {
                task: {"task_r2_supported": task not in low_support}
                for task in tasks
            },
            "identity_overlap": {
                "train_vs_valid": {"aggregate_id": 0, "result_id": 0},
                "train_vs_test": {"aggregate_id": 0, "result_id": 0},
                "valid_vs_test": {"aggregate_id": 0, "result_id": 0},
            },
            "structure_overlap": {
                "train_vs_valid": {"component": 0, "canonical": 0, "scaffold": 0},
                "train_vs_test": {"component": 0, "canonical": 0, "scaffold": 0},
                "valid_vs_test": {"component": 0, "canonical": 0, "scaffold": 0},
            },
            "max_cross_part_tanimoto": {
                "train_vs_valid": 0.40,
                "train_vs_test": 0.50,
                "valid_vs_test": 0.60,
            },
            "evaluation_identity": identity,
            "component_by_aggregate": {
                "agg-0": "component-train",
                "agg-1": "component-valid",
                "agg-2": "component-test",
            },
        },
        "source_structure_exclusion": {
            "missing_or_invalid_structure_policy": "exclude_and_report",
            "structure_overlap": {
                "source_train_vs_stage3_valid": {
                    "canonical_parent": 0,
                    "murcko_scaffold": 0,
                },
                "source_train_vs_stage3_test": {
                    "canonical_parent": 0,
                    "murcko_scaffold": 0,
                },
                "source_train_vs_stage3_valid_test": {
                    "canonical_parent": 0,
                    "murcko_scaffold": 0,
                },
            },
            "exact_identity_overlap": {
                boundary: {"aggregate_id": 0, "result_id": 0, "test_id": 0}
                for boundary in (
                    "source_train_vs_stage3_valid",
                    "source_train_vs_stage3_test",
                    "source_train_vs_stage3_valid_test",
                )
            },
            "max_tanimoto_to_heldout": {
                "source_train_vs_stage3_valid": 0.40,
                "source_train_vs_stage3_test": 0.50,
                "source_train_vs_stage3_valid_test": 0.50,
            },
        },
        "routes": {
            "M00": {
                "split_name": "split-M00",
                "rows": 3,
                "counts": {"finetune_mgkg": 1, "valid": 1, "test": 1},
            },
            "M10": {
                "split_name": "split-M10",
                "rows": 13,
                "counts": {
                    "train": 10,
                    "finetune_mgkg": 1,
                    "valid": 1,
                    "test": 1,
                },
            },
            "M11U": {
                "split_name": "split-M11U",
                "rows": 23,
                "counts": {
                    "train": 10,
                    "finetune": 10,
                    "finetune_mgkg": 1,
                    "valid": 1,
                    "test": 1,
                },
            },
        },
        "route_semantics": {
            "M00": "target only",
            "M10": "aquatic plus target",
            "M11U": "aquatic plus soil pTox plus target",
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
    with (run_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "validation_samples"])
        writer.writeheader()
        writer.writerow({"phase": "finetune_mgkg", "validation_samples": 1})
    for name in ("best_model.pt", "preprocessing.json"):
        (run_dir / name).write_text("x", encoding="utf-8")
    epochs = {"M00": (0, 0, 30), "M10": (30, 0, 30), "M11U": (30, 20, 30)}[cell]
    weighting = "tanimoto_to_finetune" if cell == "M11U" else "none"
    manifest = {
        "seed": 42,
        "split_name": f"split-{cell}",
        "data_source": {
            "source_table": "aggregated_task_records_ptox_soil_mass_molar_qc"
        },
        "head_routing": "task_target",
        "allow_mixed_target_dimensions": True,
        "epochs": epochs[0],
        "finetune_epochs": epochs[1],
        "finetune_mgkg_epochs": epochs[2],
        "ablation": "full",
        "rows": {"M00": 3, "M10": 13, "M11U": 23}[cell],
        "train_rows": 0 if cell == "M00" else 10,
        "finetune_mgkg_train_rows": 1,
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
            "early_stopping": True,
            "validation_source": "valid",
            "validation_rows": 1,
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


def test_scaffold_family_validator_accepts_locked_full_finetune(tmp_path: Path) -> None:
    run_dir, summary_path = write_run_fixture(tmp_path, cell="M11U")
    result = validate_run(
        run_dir,
        cell="M11U",
        seed=42,
        split_summary=summary_path,
    )
    assert result["status"] == "ok"
    assert result["test_used_for_selection"] is False


def test_test_component_mapping_must_cover_every_prediction() -> None:
    summary = split_summary(prediction_fixture_rows())
    mapping = load_test_component_mapping(summary, test_ids={"agg-2"})
    assert mapping == {"agg-2": "component-test"}


def test_runner_locks_matrix_smoke_validation_and_report_only_contract() -> None:
    text = Path("scripts/run_v1_2_47_scaffold_family_matrix_remote.sh").read_text(
        encoding="utf-8"
    )
    assert "for cell in M00 M10 M11U" in text
    assert "SEEDS=(42 2042 3407 8417)" in text
    assert "--finetune-mgkg-monitor-split valid" in text
    assert "--finetune-mgkg-validation-fraction 0" in text
    assert "--finetune-mgkg-freeze none" in text
    assert "--no-medium-adapters" in text
    assert "--replicates \"$BOOTSTRAP_REPLICATES\"" in text
    assert "[smoke_complete]" in text
    assert "[matrix_complete]" in text
    assert text.index("for cell in M00 M10 M11U") < text.index(
        "summarize_v1_2_47_scaffold_family_matrix.py"
    )


def test_local_launcher_enforces_smoke_before_formal() -> None:
    text = Path("scripts/launch_v1_2_47_remote.ps1").read_text(encoding="utf-8")
    smoke = "run_v1_2_47_scaffold_family_matrix_remote.sh smoke"
    formal = "run_v1_2_47_scaffold_family_matrix_remote.sh formal"
    assert smoke in text and formal in text
    assert text.index(smoke) < text.index(formal)
    assert "REBUILD_SPLITS=0" in text
    assert "setsid bash -lc" in text
