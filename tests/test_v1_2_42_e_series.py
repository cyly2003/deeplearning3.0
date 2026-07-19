from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

from qsar_tl.training.meta_ensemble import (
    deterministic_meta_split,
    fit_candidate,
    predict_candidate,
    regression_metrics,
)
from scripts.build_v1_2_42_e_series_oof_splits import build_oof_splits


def synthetic_meta_rows(n: int = 96) -> list[dict[str, object]]:
    rng = np.random.default_rng(734)
    rows = []
    for index in range(n):
        truth = float(rng.normal(3.5, 1.0))
        direct = truth + 0.35 + float(rng.normal(0.0, 0.18))
        transfer = truth - 0.20 + float(rng.normal(0.0, 0.14))
        rows.append(
            {
                "aggregate_id": f"agg-{index:04d}",
                "y_true": truth,
                "direct_pred": direct,
                "transfer_pred": transfer,
                "direct_std": 0.12 + 0.01 * (index % 3),
                "transfer_std": 0.10 + 0.01 * (index % 4),
                "molecular_weight_g_mol_used": 50.0 + index,
                "task_family": "mortality" if index % 2 else "growth",
                "taxon_group_l1": "plant" if index % 3 else "invertebrate",
                "effect_level_x": "" if index % 7 == 0 else 50.0,
            }
        )
    return rows


def test_e_series_meta_candidates_are_finite_and_reproducible() -> None:
    rows = synthetic_meta_rows()
    e0 = regression_metrics(
        [float(row["y_true"]) for row in rows],
        [float(row["transfer_pred"]) for row in rows],
    )
    for candidate in ("E1", "E2", "E3"):
        first = fit_candidate(candidate, rows, seed=42)
        second = fit_candidate(candidate, rows, seed=42)
        first_prediction = predict_candidate(first, rows)
        second_prediction = predict_candidate(second, rows)
        assert np.all(np.isfinite(first_prediction))
        assert np.allclose(first_prediction, second_prediction, atol=1.0e-12)
        assert first["candidate"] == candidate
    e1_prediction = predict_candidate(fit_candidate("E1", rows), rows)
    e1 = regression_metrics([float(row["y_true"]) for row in rows], e1_prediction)
    assert e1.r2 > e0.r2
    assert e1.mae < e0.mae


def test_meta_monitor_split_is_identity_deterministic() -> None:
    rows = synthetic_meta_rows()
    fit_a, monitor_a = deterministic_meta_split(rows, fraction=0.2, seed=7)
    fit_b, monitor_b = deterministic_meta_split(list(reversed(rows)), fraction=0.2, seed=7)
    assert {row["aggregate_id"] for row in fit_a} == {row["aggregate_id"] for row in fit_b}
    assert {row["aggregate_id"] for row in monitor_a} == {
        row["aggregate_id"] for row in monitor_b
    }


def test_oof_split_builder_covers_outer_training_once_and_omits_test(tmp_path: Path) -> None:
    db = tmp_path / "model.sqlite"
    source_table = "paired"
    parent = "parent_transfer"
    with sqlite3.connect(db) as conn:
        conn.execute(
            f'''CREATE TABLE "{source_table}" (
                aggregate_id TEXT PRIMARY KEY,
                task_head TEXT,
                target_name TEXT,
                target_family TEXT,
                medium_domain TEXT
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
        for part, count in (("train", 4), ("finetune", 4), ("finetune_mgkg", 30), ("test", 8)):
            for index in range(count):
                identity = f"{part}-{index:03d}"
                target_name = "neg_log10_mol_kg" if part in {"finetune_mgkg", "test"} else "ptox_mol_l"
                target_family = "solid_neglog_mol_kg" if part in {"finetune_mgkg", "test"} else "ptox"
                medium = "soil" if part != "train" else "aquatic"
                conn.execute(
                    f'INSERT INTO "{source_table}" VALUES (?, ?, ?, ?, ?)',
                    (identity, f"task_{index % 3}", target_name, target_family, medium),
                )
                assignments.append(
                    (
                        parent,
                        f"record-{identity}",
                        identity,
                        part,
                        42,
                        "parent",
                        source_table,
                        "|".join(
                            (
                                "stage_contract_v1",
                                f"stage=parent_{part}",
                                f"medium_domain={medium}",
                                f"target_name={target_name}",
                                f"target_family={target_family}",
                            )
                        ),
                    )
                )
        conn.executemany("INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", assignments)
        conn.commit()

    audit_csv = tmp_path / "audit.csv"
    summary_json = tmp_path / "summary.json"
    summary = build_oof_splits(
        db_path=db,
        source_table=source_table,
        parent_transfer_split=parent,
        direct_prefix="direct_oof",
        transfer_prefix="transfer_oof",
        folds=5,
        seed=123,
        audit_csv=audit_csv,
        summary_json=summary_json,
    )
    assert summary["outer_test_used_in_oof"] is False
    assert summary["oof_validation_coverage_n"] == 30
    with sqlite3.connect(db) as conn:
        validation = conn.execute(
            """SELECT aggregate_id, COUNT(*)
               FROM split_assignments
               WHERE split_name LIKE 'transfer_oof_fold%' AND split_part = 'valid'
               GROUP BY aggregate_id"""
        ).fetchall()
        assert len(validation) == 30
        assert all(count == 1 for _, count in validation)
        outer_test = conn.execute(
            """SELECT COUNT(*) FROM split_assignments
               WHERE split_name LIKE '%_oof_fold%' AND split_part = 'test'"""
        ).fetchone()[0]
        assert outer_test == 0
        non_strict = conn.execute(
            """SELECT COUNT(*) FROM split_assignments
               WHERE split_name LIKE '%_oof_fold%'
                 AND group_key NOT LIKE 'stage_contract_v1|%'"""
        ).fetchone()[0]
        assert non_strict == 0
    with audit_csv.open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 5
    assert json.loads(summary_json.read_text(encoding="utf-8"))["oof_validation_coverage_n"] == 30
