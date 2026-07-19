from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_v1_2_43_g_splits import build_g_splits, load_target_metadata
from scripts.build_v1_2_43_g_splits import stage_contract, stage_sample_record_id
from scripts.summarize_v1_2_43_g_series import (
    assert_final_boundaries,
    assert_screen_validation_identities,
    build_screen_evidence,
    build_test_ensemble_summary,
    canonical_file_sha256,
    canonical_sha256,
    discover_runs,
    load_and_validate_winner_lock,
    select_candidate,
    selection_protocol_fields,
    write_immutable_winner_lock,
)
from scripts.validate_v1_2_43_g_run import validate_run


def make_split_fixture(tmp_path: Path) -> tuple[Path, Path, set[str], set[str], set[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "model.sqlite"
    source_table = "paired"
    parent_split = "parent"
    old_train = {f"old-train-{index:03d}" for index in range(40)}
    old_valid = {f"old-valid-{index:03d}" for index in range(10)}
    outer_test = {f"test-{index:03d}" for index in range(12)}
    with sqlite3.connect(db) as conn:
        conn.execute(
            f'''CREATE TABLE "{source_table}" (
                aggregate_id TEXT PRIMARY KEY,
                task_head TEXT,
                target_name TEXT,
                target_family TEXT,
                medium_domain TEXT,
                target_value_median REAL,
                result_ids TEXT
            )'''
        )
        conn.execute(
            """CREATE TABLE split_assignments (
                split_name TEXT,
                record_id TEXT,
                aggregate_id TEXT,
                split_part TEXT,
                seed INTEGER,
                split_type TEXT,
                source_table TEXT,
                group_key TEXT
            )"""
        )
        assignments = []
        stage1 = {f"stage1-{index:03d}" for index in range(5)}
        stage2 = {f"stage2-{index:03d}" for index in range(5)}
        parts = (
            ("train", stage1),
            ("finetune", stage2),
            ("finetune_mgkg", old_train | old_valid),
            ("test", outer_test),
        )
        for part, identities in parts:
            for position, identity in enumerate(sorted(identities)):
                is_molar = part in {"finetune_mgkg", "test"}
                target_name = "neg_log10_mol_kg" if is_molar else "ptox_mol_l"
                target_family = "solid_neglog_mol_kg" if is_molar else "ptox"
                medium = "soil" if part != "train" else "aquatic"
                task = f"task_{position % 4}"
                conn.execute(
                    f'INSERT INTO "{source_table}" VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        identity,
                        task,
                        target_name,
                        target_family,
                        medium,
                        1.0 + 0.15 * position,
                        json.dumps([f"result-{identity}"]),
                    ),
                )
                record_id = (
                    stage_sample_record_id(identity, medium, target_name, target_family)
                    if part in {"finetune_mgkg", "test"}
                    else f"record-{identity}"
                )
                group_key = (
                    stage_contract(
                        stage=f"fixture_{part}",
                        medium_domain=medium,
                        target_name=target_name,
                        target_family=target_family,
                    )
                    if part in {"finetune_mgkg", "test"}
                    else ("soil" if medium == "soil" else "aquatic")
                )
                assignments.append(
                    (
                        parent_split,
                        record_id,
                        identity,
                        part,
                        42,
                        "parent",
                        source_table,
                        group_key,
                    )
                )
        conn.executemany("INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", assignments)
        conn.commit()

    baseline_root = tmp_path / "baseline"
    run_dir = baseline_root / "v1.2.40_X0_molar_seed42" / "deep" / "full" / parent_split
    run_dir.mkdir(parents=True)
    fields = [
        "aggregate_id",
        "result_ids",
        "split_part",
        "task_head",
        "target_name",
        "target_family",
        "medium_domain",
        "y_true",
        "y_pred",
    ]
    with sqlite3.connect(db) as conn:
        baseline_metadata = {
            row[0]: {
                "result_ids": row[2],
                "task_head": row[3],
                "target_name": row[4],
                "target_family": row[5],
                "medium_domain": row[6],
                "y_true": row[7],
            }
            for row in conn.execute(
                '''SELECT p.aggregate_id, s.record_id, p.result_ids, p.task_head,
                          p.target_name, p.target_family, p.medium_domain,
                          p.target_value_median
                   FROM paired AS p
                   JOIN split_assignments AS s ON s.aggregate_id=p.aggregate_id
                   WHERE s.split_name='parent' AND s.split_part='finetune_mgkg' '''
            )
        }
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for identity in sorted(old_train):
            writer.writerow(
                {
                    "aggregate_id": identity,
                    "split_part": "finetune_mgkg",
                    **baseline_metadata[identity],
                    "y_pred": 1,
                }
            )
        for identity in sorted(old_valid):
            writer.writerow(
                {
                    "aggregate_id": identity,
                    "split_part": "finetune_mgkg_validation",
                    **baseline_metadata[identity],
                    "y_pred": 1,
                }
            )
    return db, baseline_root, old_train, old_valid, outer_test


def call_builder(
    tmp_path: Path,
    *,
    db: Path,
    baseline_root: Path,
    phase: str,
    production_names: bool = False,
) -> dict[str, object]:
    winner_lock = None
    if phase == "final":
        selection = {
            "selected_candidate": "G2",
            "selection_status": "locked_from_fresh_validation",
            **selection_protocol_fields([42, 3407], [42, 2042, 3407, 8417]),
        }
        winner_lock = make_test_winner_lock(tmp_path, selection=selection)
    return build_g_splits(
        db_path=db,
        source_table="paired",
        parent_split="parent",
        old_baseline_root=baseline_root,
        screen_split=("M_v1_2_43_g_screen" if production_names else "g_screen"),
        final_split=("M_v1_2_43_g_final" if production_names else "g_final"),
        phase=phase,
        winner_lock=winner_lock,
        validation_seed=17073,
        quantile_bins=5,
        audit_csv=tmp_path / f"{phase}_audit.csv",
        summary_json=tmp_path / f"{phase}_summary.json",
    )


def make_test_winner_lock(tmp_path: Path, *, selection: dict[str, object]) -> Path:
    selection_path = tmp_path / "selected_candidate.json"
    metrics_path = tmp_path / "screen_seed_metrics.csv"
    candidate_summary_path = tmp_path / "validation_candidate_summary.csv"
    split_summary_path = tmp_path / "screen_split_summary.json"
    manifest_path = tmp_path / "screen_manifest.json"
    predictions_path = tmp_path / "screen_predictions.csv"
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    metrics_path.write_text("candidate,seed\nG0,42\n", encoding="utf-8")
    candidate_summary_path.write_text(
        "candidate,validation_native_mae_mean\nG2,0.5\n", encoding="utf-8"
    )
    split_summary_path.write_text(
        json.dumps(
            {
                "schema": "v1_2_43_g_split_v1",
                "built_split": "M_v1_2_43_g_screen",
                "outer_test_assignments_queried": False,
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    predictions_path.write_text("aggregate_id\na\n", encoding="utf-8")
    runs = [
        {
            "candidate": "G0",
            "phase": "screen",
            "seed": 42,
            "manifest_path": str(manifest_path),
            "predictions_path": str(predictions_path),
            "predictions": [{"aggregate_id": "a"}],
        }
    ]
    screen_evidence = build_screen_evidence(
        runs, split_summary_path=split_summary_path
    )
    canonical_inputs = {
        "fresh_validation_predictions_sha256": screen_evidence[
            "fresh_validation_predictions_sha256"
        ],
        "validation_metric_rows_sha256": canonical_file_sha256(metrics_path),
        "validation_candidate_summary_sha256": canonical_file_sha256(
            candidate_summary_path
        ),
        "selection_sha256": canonical_sha256(selection),
    }
    lock_path = tmp_path / "winner_lock.json"
    write_immutable_winner_lock(
        lock_path,
        selection=selection,
        locked_files={
            "selected_candidate": selection_path,
            "screen_seed_metrics": metrics_path,
            "validation_candidate_summary": candidate_summary_path,
        },
        canonical_inputs=canonical_inputs,
        screen_evidence=screen_evidence,
    )
    return lock_path


def test_locked_target_metadata_resolves_integer_sqlite_aggregate_ids(tmp_path: Path) -> None:
    db = tmp_path / "integer_aggregate_ids.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE paired (
                aggregate_id,
                task_head,
                target_name,
                target_family,
                medium_domain,
                target_value_median,
                result_ids
            )"""
        )
        conn.execute(
            "INSERT INTO paired VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                101,
                "NOEC_GeneticDamage",
                "neg_log10_mol_kg",
                "solid_neglog_mol_kg",
                "soil",
                2.0,
                '["result-101"]',
            ),
        )
        conn.row_factory = sqlite3.Row
        metadata = load_target_metadata(conn, "paired", {"101"})
    assert metadata["101"]["task_head"] == "NOEC_GeneticDamage"


def test_g_split_builder_rotates_validation_and_delays_outer_test(tmp_path: Path) -> None:
    db, baseline_root, old_train, old_valid, outer_test = make_split_fixture(tmp_path)
    screen = call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")
    with sqlite3.connect(db) as conn:
        parts = dict(
            conn.execute(
                "SELECT split_part, COUNT(1) FROM split_assignments WHERE split_name='g_screen' GROUP BY split_part"
            ).fetchall()
        )
        new_valid = {
            row[0]
            for row in conn.execute(
                "SELECT aggregate_id FROM split_assignments WHERE split_name='g_screen' AND split_part='valid'"
            )
        }
        restored_old_valid = conn.execute(
            "SELECT COUNT(1) FROM split_assignments WHERE split_name='g_screen' AND split_part='finetune_mgkg' AND aggregate_id IN (%s)"
            % ",".join("?" for _ in old_valid),
            sorted(old_valid),
        ).fetchone()[0]
    assert parts.get("test", 0) == 0
    assert len(new_valid) == len(old_valid)
    assert new_valid <= old_train
    assert not (new_valid & old_valid)
    assert restored_old_valid == len(old_valid)
    assert screen["new_stage3_training_effective_n"] == len(old_train)

    final = call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="final")
    with sqlite3.connect(db) as conn:
        final_test = conn.execute(
            "SELECT COUNT(1) FROM split_assignments WHERE split_name='g_final' AND split_part='test'"
        ).fetchone()[0]
        final_valid = {
            row[0]
            for row in conn.execute(
                "SELECT aggregate_id FROM split_assignments WHERE split_name='g_final' AND split_part='valid'"
            )
        }
    assert final_test == len(outer_test)
    assert final_valid == new_valid
    assert final["new_validation_aggregate_id_sha256"] == screen["new_validation_aggregate_id_sha256"]


def test_result_id_siblings_never_cross_stage3_train_validation(tmp_path: Path) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    siblings = ("old-train-000", "old-train-001")
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "UPDATE paired SET result_ids = ? WHERE aggregate_id = ?",
            [(json.dumps(["shared-result"]), identity) for identity in siblings],
        )
        conn.commit()
    prediction_path = next(baseline_root.glob("**/predictions.csv"))
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8-sig", newline="")))
    for row in rows:
        if row["aggregate_id"] in siblings:
            row["result_ids"] = json.dumps(["shared-result"])
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")
    with sqlite3.connect(db) as conn:
        valid = {
            row[0]
            for row in conn.execute(
                "SELECT aggregate_id FROM split_assignments WHERE split_name='g_screen' AND split_part='valid'"
            )
        }
        train_results = {
            result_id
            for (payload,) in conn.execute(
                """SELECT p.result_ids FROM split_assignments AS s
                   JOIN paired AS p ON p.aggregate_id=s.aggregate_id
                   WHERE s.split_name='g_screen' AND s.split_part='finetune_mgkg'"""
            )
            for result_id in json.loads(payload)
        }
        valid_results = {
            result_id
            for (payload,) in conn.execute(
                """SELECT p.result_ids FROM split_assignments AS s
                   JOIN paired AS p ON p.aggregate_id=s.aggregate_id
                   WHERE s.split_name='g_screen' AND s.split_part='valid'"""
            )
            for result_id in json.loads(payload)
        }
    assert not (set(siblings) & valid)
    assert not (train_results & valid_results)
    assert summary["new_train_validation_result_ids_overlap_n"] == 0


def test_parent_stage3_extra_identity_fails_closed(tmp_path: Path) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    identity = "parent-extra"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO paired VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identity,
                "task_0",
                "neg_log10_mol_kg",
                "solid_neglog_mol_kg",
                "soil",
                2.5,
                json.dumps(["result-parent-extra"]),
            ),
        )
        conn.execute(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "parent",
                stage_sample_record_id(
                    identity, "soil", "neg_log10_mol_kg", "solid_neglog_mol_kg"
                ),
                identity,
                "finetune_mgkg",
                42,
                "parent",
                "paired",
                stage_contract(
                    stage="fixture_extra",
                    medium_domain="soil",
                    target_name="neg_log10_mol_kg",
                    target_family="solid_neglog_mol_kg",
                ),
            ),
        )
        conn.commit()
    with pytest.raises(ValueError, match="eligible for its locked task filter"):
        call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")


def test_parent_stage3_filter_ineligible_rare_route_is_audited(tmp_path: Path) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    identity = "parent-filtered-rare"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO paired VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identity,
                "task_rare",
                "neg_log10_mol_kg",
                "solid_neglog_mol_kg",
                "soil",
                2.5,
                json.dumps(["result-parent-filtered-rare"]),
            ),
        )
        conn.execute(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "parent",
                stage_sample_record_id(
                    identity, "soil", "neg_log10_mol_kg", "solid_neglog_mol_kg"
                ),
                identity,
                "finetune_mgkg",
                42,
                "parent",
                "paired",
                stage_contract(
                    stage="fixture_filtered_rare",
                    medium_domain="soil",
                    target_name="neg_log10_mol_kg",
                    target_family="solid_neglog_mol_kg",
                ),
            ),
        )
        conn.commit()

    summary = call_builder(
        tmp_path,
        db=db,
        baseline_root=baseline_root,
        phase="screen",
    )

    assert summary["parent_stage3_task_filter_excluded_n"] == 1
    assert summary["parent_stage3_task_filter_excluded_by_task"] == {
        "task_rare|solid_neglog_mol_kg": 1
    }
    exclusions = summary["parent_stage3_task_filter_exclusion_records"]
    assert exclusions == [
        {
            "aggregate_id": identity,
            "record_id": stage_sample_record_id(
                identity, "soil", "neg_log10_mol_kg", "solid_neglog_mol_kg"
            ),
            "task_route": "task_rare|solid_neglog_mol_kg",
            "raw_route_support_n": 1,
            "task_filter_min_total": 200,
            "reason": "entire_task_route_absent_from_v1_2_40_baseline_and_raw_support_lt_min_total",
            "canonical_sha256": exclusions[0]["canonical_sha256"],
        }
    ]
    assert len(exclusions[0]["canonical_sha256"]) == 64
    exclusion_audit = Path(str(summary["parent_stage3_task_filter_exclusion_audit_csv"]))
    assert exclusion_audit.is_file()
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(1) FROM split_assignments WHERE split_name='g_screen' AND aggregate_id=?",
            (identity,),
        ).fetchone()[0] == 0


@pytest.mark.parametrize("result_ids", [None, "", "not-json", "[]", '"scalar"'])
def test_stage3_missing_or_unparseable_result_ids_fail_closed(
    tmp_path: Path, result_ids: object
) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE paired SET result_ids=? WHERE aggregate_id='old-train-000'",
            (result_ids,),
        )
        conn.commit()
    with pytest.raises(ValueError, match="result_ids"):
        call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")


def test_baseline_parent_stage3_scientific_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    prediction_path = next(baseline_root.glob("**/predictions.csv"))
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8-sig", newline="")))
    rows[0]["task_head"] = "tampered-task"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="scientific identities differ"):
        call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")


def test_parent_stage3_wrong_target_contract_fails_closed(tmp_path: Path) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE paired SET target_name='neg_log10_mg_kg' WHERE aggregate_id='old-train-000'"
        )
        conn.commit()
    with pytest.raises(ValueError, match="effective baseline sample set is absent"):
        call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")


def test_screen_ignores_malformed_outer_test_assignment_until_lock(tmp_path: Path) -> None:
    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "parent",
                "malformed-test-record",
                "outer-test-not-in-source",
                "test",
                42,
                "malformed",
                "paired",
                "not-a-contract",
            ),
        )
        conn.commit()
    summary = call_builder(tmp_path, db=db, baseline_root=baseline_root, phase="screen")
    assert summary["outer_test_assignments_queried"] is False
    assert summary["outer_test_counts_reported"] is False
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(1) FROM split_assignments WHERE split_name='g_screen' AND split_part='test'"
        ).fetchone()[0] == 0


def test_validation_selector_requires_threshold_and_both_seeds() -> None:
    rows = [
        {
            "candidate": "G1",
            "complete_screen": True,
            "all_screen_seeds_improve_both": False,
            "paired_seeds_improving_both": 1,
            "delta_r2_vs_g0_mean": 0.010,
            "delta_mae_vs_g0_mean": -0.010,
            "validation_native_r2_mean": 0.71,
            "validation_native_mae_mean": 0.55,
        },
        {
            "candidate": "G2",
            "complete_screen": True,
            "all_screen_seeds_improve_both": True,
            "paired_seeds_improving_both": 2,
            "delta_r2_vs_g0_mean": 0.005,
            "delta_mae_vs_g0_mean": -0.006,
            "validation_native_r2_mean": 0.70,
            "validation_native_mae_mean": 0.54,
        },
        {
            "candidate": "G3",
            "complete_screen": True,
            "all_screen_seeds_improve_both": True,
            "paired_seeds_improving_both": 2,
            "delta_r2_vs_g0_mean": 0.012,
            "delta_mae_vs_g0_mean": -0.005,
            "validation_native_r2_mean": 0.72,
            "validation_native_mae_mean": 0.545,
        },
    ]
    selected = select_candidate(rows, screen_seeds=[42, 3407])
    assert selected["selected_candidate"] == "G2"


def test_screen_identity_audit_fails_closed() -> None:
    rows = [
        {
            "candidate": "G0",
            "phase": "screen",
            "seed": 42,
            "evaluation_part": "validation",
            "n": 10,
            "aggregate_id_sha256": "same",
            "result_ids_sha256": "same-results",
        },
        {
            "candidate": "G0",
            "phase": "screen",
            "seed": 3407,
            "evaluation_part": "validation",
            "n": 10,
            "aggregate_id_sha256": "different",
            "result_ids_sha256": "same-results",
        },
    ]
    with pytest.raises(ValueError, match="identity mismatch"):
        assert_screen_validation_identities(rows, screen_seeds=[42, 3407])


def test_final_boundary_audit_rejects_changed_validation() -> None:
    rows = []
    for phase, candidate, seeds in (
        ("screen", "G0", (42, 3407)),
        ("final", "G0", (42, 2042, 3407, 8417)),
        ("final", "G2", (42, 2042, 3407, 8417)),
    ):
        for seed in seeds:
            rows.append(
                {
                    "candidate": candidate,
                    "phase": phase,
                    "seed": seed,
                    "evaluation_part": "validation",
                    "n": 10,
                    "aggregate_id_sha256": (
                        "changed" if phase == "final" and candidate == "G2" and seed == 8417 else "valid"
                    ),
                    "result_ids_sha256": "valid-results",
                }
            )
            if phase == "final":
                rows.append(
                    {
                        "candidate": candidate,
                        "phase": phase,
                        "seed": seed,
                        "evaluation_part": "test",
                        "n": 12,
                        "aggregate_id_sha256": "test",
                        "result_ids_sha256": "test-results",
                    }
                )
    with pytest.raises(ValueError, match="Final validation boundary differs"):
        assert_final_boundaries(
            rows,
            selected_candidate="G2",
            screen_seeds=[42, 3407],
            final_seeds=[42, 2042, 3407, 8417],
        )


def test_prediction_ensemble_reports_native_and_common_metrics() -> None:
    prediction_sets = {}
    for candidate, offset in (("G0", 0.2), ("G2", 0.1)):
        for seed in (42, 2042, 3407, 8417):
            rows = []
            for index, truth in enumerate((1.0, 2.0, 3.0, 4.0)):
                rows.append(
                    {
                        "aggregate_id": f"agg-{index}",
                        "y_native": truth,
                        "pred_native": truth + offset + seed * 0.0,
                        "y_mgkg": truth - 2.0,
                        "pred_mgkg": truth - 2.0 + offset,
                    }
                )
            prediction_sets[(candidate, "final", seed, "test")] = rows
    output = build_test_ensemble_summary(
        prediction_sets,
        selected_candidate="G2",
        final_seeds=[42, 2042, 3407, 8417],
    )
    assert [row["candidate"] for row in output] == ["G0", "G2"]
    assert output[1]["native_mae"] < output[0]["native_mae"]
    assert output[1]["common_mgkg_mae"] < output[0]["common_mgkg_mae"]


def make_validator_fixture(tmp_path: Path) -> dict[str, object]:
    import torch
    from qsar_tl.training.deep_experiment import seal_stage3_init_contract

    db, baseline_root, _, _, _ = make_split_fixture(tmp_path)
    call_builder(
        tmp_path,
        db=db,
        baseline_root=baseline_root,
        phase="screen",
        production_names=True,
    )
    split_summary = tmp_path / "screen_summary.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("history.csv", "best_model.pt"):
        (run_dir / name).write_bytes(b"x")
    cache = tmp_path / "stage2.pt"
    contract = seal_stage3_init_contract(
        {
            "schema_version": 2,
            "data_identity": {
                "split_name": "M_v1_2_43_g_screen",
                "source_table": "paired",
            },
            "preprocessing": {},
            "base_architecture": {},
            "weighting_and_auxiliary": {},
            "stage12_protocol": {"seed": 42},
        }
    )
    torch.save(
        {"format": "qsar_stage3_init_v1", "contract": contract, "state_dict": {}},
        cache,
    )
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            """SELECT s.aggregate_id, p.result_ids
               FROM split_assignments AS s
               JOIN paired AS p ON p.aggregate_id = s.aggregate_id
               WHERE s.split_name = 'M_v1_2_43_g_screen' AND s.split_part = 'valid'
               ORDER BY s.aggregate_id"""
        ).fetchall()
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "aggregate_id",
                "split_part",
                "target_name",
                "target_family",
                "medium_domain",
                "result_ids",
            ),
        )
        writer.writeheader()
        for identity, result_ids in rows:
            writer.writerow(
                {
                    "aggregate_id": identity,
                    "split_part": "finetune_mgkg_validation",
                    "target_name": "neg_log10_mol_kg",
                    "target_family": "solid_neglog_mol_kg",
                    "medium_domain": "soil",
                    "result_ids": result_ids,
                }
            )
    manifest = {
        "seed": 42,
        "split_name": "M_v1_2_43_g_screen",
        "data_source": {
            "modeling_tables_db": str(db),
            "source_table": "paired",
        },
        "prediction_output_split_parts": ["finetune_mgkg_validation"],
        "epochs": 30,
        "finetune_epochs": 20,
        "finetune_mgkg_epochs": 40,
        "finetune_mgkg_epochs_ran": 40,
        "finetune_mgkg_validation_seed": 17073,
        "ablation_features": {"use_medium_adapter": False},
        "finetune_mgkg": {
            "validation_source": "valid",
            "early_stopping": False,
            "freeze": "heads_only",
            "head_learning_rate": 0.0005,
            "trunk_learning_rate": None,
            "soil_ptox_replay_fraction_requested": 0.0,
            "regression_loss": {"mse_weight": 0.15},
            "target_bin_sampling": {
                "enabled": True,
                "kind": "within_task_equal_width_target_bin_inverse_frequency",
                "requested_bins": 10,
                "min_weight": 0.5,
                "max_weight": 2.0,
                "count_source": "finetune_mgkg_training_only",
            },
            "hierarchical_head": {"enabled": False},
                "stage3_init_checkpoint": {
                    "loaded": True,
                    "stage1_stage2_skipped": True,
                    "path": str(cache),
                    "contract_sha256": contract["sha256"],
                },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "candidate": "G2",
        "phase": "screen",
        "seed": 42,
        "split_name": "M_v1_2_43_g_screen",
        "source_table": "paired",
        "db_path": db,
        "split_summary": split_summary,
        "stage2_cache": cache,
        "smoke": False,
    }


def test_g_run_validator_checks_g2_protocol(tmp_path: Path) -> None:
    fixture = make_validator_fixture(tmp_path)
    run_dir = fixture.pop("run_dir")
    assert isinstance(run_dir, Path)
    result = validate_run(run_dir, **fixture)
    assert result["status"] == "ok"


def test_validator_rejects_wrong_manifest_seed_and_split(tmp_path: Path) -> None:
    seed_fixture = make_validator_fixture(tmp_path / "seed")
    seed_run = seed_fixture.pop("run_dir")
    assert isinstance(seed_run, Path)
    manifest_path = seed_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 3407
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_seed"):
        validate_run(seed_run, **seed_fixture)

    split_fixture = make_validator_fixture(tmp_path / "split")
    split_run = split_fixture.pop("run_dir")
    assert isinstance(split_run, Path)
    manifest_path = split_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_name"] = "stale-split"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_split"):
        validate_run(split_run, **split_fixture)


def test_validator_rejects_stale_current_split_hash(tmp_path: Path) -> None:
    fixture = make_validator_fixture(tmp_path)
    run_dir = fixture.pop("run_dir")
    assert isinstance(run_dir, Path)
    db = fixture["db_path"]
    assert isinstance(db, Path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE split_assignments SET group_key='stale'
               WHERE split_name='M_v1_2_43_g_screen'
                 AND rowid=(SELECT MIN(rowid) FROM split_assignments WHERE split_name='M_v1_2_43_g_screen')"""
        )
        conn.commit()
    with pytest.raises(ValueError, match="assignment hash"):
        validate_run(run_dir, **fixture)


def test_winner_lock_rejects_selection_file_tamper_and_rewrite(tmp_path: Path) -> None:
    selection = {
        "selected_candidate": "G2",
        "selection_status": "locked_from_fresh_validation",
        **selection_protocol_fields([42, 3407], [42, 2042, 3407, 8417]),
    }
    lock_path = make_test_winner_lock(tmp_path, selection=selection)
    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    selection_path = Path(
        lock_payload["locked_files"]["selected_candidate"]["path"]
    )
    selection_path.write_text(json.dumps({**selection, "selected_candidate": "G3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_and_validate_winner_lock(lock_path)

    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable selection content"):
        write_immutable_winner_lock(
            lock_path,
            selection={**selection, "selected_candidate": "G3"},
            locked_files={
                label: Path(item["path"])
                for label, item in lock_payload["locked_files"].items()
            },
            canonical_inputs={
                **lock_payload["canonical_inputs"],
                "selection_sha256": canonical_sha256(
                    {**selection, "selected_candidate": "G3"}
                ),
            },
            screen_evidence=lock_payload["screen_evidence"],
        )


def test_winner_lock_rejects_screen_artifact_tamper(tmp_path: Path) -> None:
    selection = {
        "selected_candidate": "G2",
        "selection_status": "locked_from_fresh_validation",
        **selection_protocol_fields([42, 3407], [42, 2042, 3407, 8417]),
    }
    lock_path = make_test_winner_lock(tmp_path, selection=selection)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    artifact_path = Path(
        payload["screen_evidence"]["screen_run_artifacts"][0]["predictions"]["path"]
    )
    artifact_path.write_text("aggregate_id\ntampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence file hash mismatch"):
        load_and_validate_winner_lock(lock_path)


def test_winner_lock_rejects_screen_split_summary_tamper(tmp_path: Path) -> None:
    selection = {
        "selected_candidate": "G2",
        "selection_status": "locked_from_fresh_validation",
        **selection_protocol_fields([42, 3407], [42, 2042, 3407, 8417]),
    }
    lock_path = make_test_winner_lock(tmp_path, selection=selection)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    summary_path = Path(payload["screen_evidence"]["screen_split_summary"]["path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["new_validation_aggregate_id_sha256"] = "tampered"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence file hash mismatch"):
        load_and_validate_winner_lock(lock_path)


def test_winner_lock_rejects_changed_fresh_rows_with_same_winner(
    tmp_path: Path,
) -> None:
    selection = {
        "selected_candidate": "G2",
        "selection_status": "locked_from_fresh_validation",
        **selection_protocol_fields([42, 3407], [42, 2042, 3407, 8417]),
    }
    lock_path = make_test_winner_lock(tmp_path, selection=selection)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="differ from fresh inputs"):
        write_immutable_winner_lock(
            lock_path,
            selection=selection,
            locked_files={
                label: Path(item["path"])
                for label, item in payload["locked_files"].items()
            },
            canonical_inputs={
                **payload["canonical_inputs"],
                "fresh_validation_predictions_sha256": canonical_sha256(
                    [{"changed": True}]
                ),
            },
            screen_evidence=payload["screen_evidence"],
        )


def test_summarizer_rejects_directory_seed_when_manifest_disagrees(tmp_path: Path) -> None:
    run_dir = tmp_path / "v1.2.43_G0_screen_seed42" / "deep" / "full" / "split"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.csv").write_text("aggregate_id\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 3407,
                "split_name": "M_v1_2_43_g_screen",
                "data_source": {
                    "source_table": "aggregated_task_records_ptox_soil_mass_molar_qc"
                },
                "prediction_output_split_parts": ["finetune_mgkg_validation"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest seed"):
        discover_runs(tmp_path, phases={"screen"})


def test_runner_keeps_g3_as_hierarchy_only() -> None:
    runner = Path("scripts/run_v1_2_43_g_series_remote.sh").read_text(encoding="utf-8")
    g3 = runner.split("    G3)", 1)[1].split("    *)", 1)[0]
    assert "--no-finetune-mgkg-target-bin-sampling" in g3
    assert "--finetune-mgkg-mse-loss-weight 0" in g3
    assert "--finetune-mgkg-hierarchical-head" in g3
    before_lock, after_lock = runner.split("[winner_locked]", 1)
    assert "build_split final" not in before_lock
    assert "build_split final" in after_lock
    assert 'TASK_FILTER_MIN_TOTAL="${TASK_FILTER_MIN_TOTAL:-150}"' in runner
    assert '--task-filter-min-total "$TASK_FILTER_MIN_TOTAL"' in runner
    assert '--task-filter-min-train "$TASK_FILTER_MIN_TRAIN"' in runner
    assert '--task-filter-min-eval "$TASK_FILTER_MIN_EVAL"' in runner
    assert 'SCREEN_CHALLENGERS=(G1 G3)' in runner
    assert 'SCREEN_CHALLENGERS=(G1 G2 G3)' in runner
