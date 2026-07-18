from __future__ import annotations

import csv
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from scripts.build_paired_soil_molkg_experiment import build_paired_targets
from scripts.build_three_stage_ptox_to_soil_mgkg_split import build_three_stage_split
from qsar_tl.training.baseline import load_split_frame


SOURCE_TABLE = "expanded"
OUTPUT_TABLE = "paired_targets"


def test_paired_molkg_builder_inherits_parent_split_and_roundtrips(tmp_path: Path) -> None:
    db_path = tmp_path / "paired.sqlite"
    audit_path = tmp_path / "conversion.csv"
    summary_path = tmp_path / "summary.json"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f'''
            CREATE TABLE {SOURCE_TABLE} (
                aggregate_id TEXT,
                cas_number TEXT,
                smiles TEXT,
                medium_domain TEXT,
                target_name TEXT,
                target_family TEXT,
                target_basis TEXT,
                target_value_median REAL,
                target_value_mean REAL,
                target_value_std REAL,
                result_ids TEXT,
                unit_family_v2 TEXT,
                standard_unit_v2 TEXT,
                standard_value_mg_kg REAL,
                conversion_path TEXT,
                value_quality TEXT,
                task_head TEXT
            )
            '''
        )
        rows = [
            ("A1", "10", "CC", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.0, 2.0, 0.0, "[10]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("P1", "11", "CCC", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.1, 2.1, 0.0, "[11]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("P2", "12", "CCCC", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.2, 2.2, 0.0, "[12]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("M1", "20", "CCO", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "mg/kg:soil", 2.0, 2.0, 0.3, "[20]", "soil_mg_kg", "mg/kg", 0.01, "raw_to_mgkg", "exact", "ECx_Mortality"),
            # This represents a midpoint aggregate: conversion must not depend
            # on a populated standard_value_mg_kg column.
            ("M2", "21", "CCN", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "mg/kg:soil", 1.5, 1.5, 0.2, "[21]", "soil_mg_kg", "mg/kg", None, "raw_to_mgkg", "midpoint", "ECx_Mortality"),
            ("M3", "22", None, "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "mg/kg:soil", 1.0, 1.0, 0.1, "[22]", "soil_mg_kg", "mg/kg", 0.1, "raw_to_mgkg", "exact", "ECx_Mortality"),
        ]
        conn.executemany(
            f"INSERT INTO {SOURCE_TABLE} VALUES ({', '.join('?' for _ in rows[0])})",
            rows,
        )
        conn.execute(
            "CREATE TABLE task_records_qc (result_id TEXT, molecular_weight_g_mol REAL, molecular_weight_rdkit_g_mol REAL)"
        )
        conn.executemany(
            "INSERT INTO task_records_qc VALUES (?, ?, ?)",
            [("20", 100.0, 100.0), ("21", None, 50.0), ("22", None, None)],
        )
        conn.execute("CREATE TABLE soil_mgkg_parent (aggregate_id TEXT)")
        conn.executemany("INSERT INTO soil_mgkg_parent VALUES (?)", [("M1",), ("M2",), ("M3",)])
        conn.execute(
            """
            CREATE TABLE split_assignments (
                split_name TEXT NOT NULL,
                record_id TEXT,
                aggregate_id TEXT,
                split_part TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split_type TEXT NOT NULL,
                source_table TEXT NOT NULL,
                group_key TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("Parent", None, "M1", "train", 42, "random", "soil_mgkg_parent", "M1"),
                ("Parent", None, "M2", "test", 42, "random", "soil_mgkg_parent", "M2"),
                ("Parent", None, "M3", "test", 42, "random", "soil_mgkg_parent", "M3"),
            ],
        )
        conn.commit()

    summary = build_paired_targets(
        db_path=db_path,
        source_table=SOURCE_TABLE,
        mw_source_table="task_records_qc",
        parent_mgkg_source_table="soil_mgkg_parent",
        parent_mgkg_split="Parent",
        output_table=OUTPUT_TABLE,
        mw_map_table="mw_map",
        matched_mgkg_split="MassMatched",
        molkg_split="MolarMatched",
        audit_csv=audit_path,
        summary_json=summary_path,
    )

    assert summary["parent_soil_mgkg_rows"] == 3
    assert summary["converted_rows"] == 2
    assert summary["conversion_unavailable_rows"] == 1
    assert summary["split_counts"]["MassMatched"] == {"test": 1, "train": 1}
    assert summary["split_counts"]["MolarMatched"] == {"test": 1, "train": 1}
    assert summary["roundtrip_max_abs_error"] < 1e-12
    assert json.loads(summary_path.read_text(encoding="utf-8"))["converted_rows"] == 2

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        molar = conn.execute(
            f"SELECT * FROM {OUTPUT_TABLE} WHERE aggregate_id = 'M1' AND target_name = 'neg_log10_mol_kg'"
        ).fetchone()
        assert molar is not None
        assert math.isclose(float(molar["target_value_median"]), 7.0)
        assert math.isclose(float(molar["target_value_mean"]), 7.0)
        assert math.isclose(float(molar["target_value_std"]), 0.3)
        assert molar["target_family"] == "solid_neglog_mol_kg"
        assert molar["target_basis"] == "mol/kg_from_mg/kg:soil"
        assert molar["parent_target_basis"] == "mg/kg:soil"
        assert molar["unit_family_v2"] == "soil_mol_kg"
        assert molar["standard_unit_v2"] == "mol/kg"
        assert math.isclose(float(molar["standard_value_mol_kg"]), 1.0e-7)
        assert float(molar["molecular_weight_g_mol_used"]) == 100.0
        assert conn.execute(
            f"SELECT COUNT(*) FROM {OUTPUT_TABLE} WHERE aggregate_id = 'M3'"
        ).fetchone()[0] == 0

    with audit_path.open(encoding="utf-8-sig", newline="") as handle:
        audit = {row["aggregate_id"]: row for row in csv.DictReader(handle)}
    assert audit["M2"]["value_quality"] == "midpoint"
    assert audit["M2"]["conversion_status"] == "converted"
    assert audit["M3"]["conversion_status"] == "conversion_unavailable"


def test_three_stage_builder_accepts_the_paired_molkg_target(tmp_path: Path) -> None:
    db_path = tmp_path / "paired_three_stage.sqlite"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            f'''
            CREATE TABLE {SOURCE_TABLE} (
                aggregate_id TEXT, cas_number TEXT, smiles TEXT, medium_domain TEXT,
                target_name TEXT, target_family TEXT, target_basis TEXT,
                target_value_median REAL, result_ids TEXT, unit_family_v2 TEXT,
                standard_unit_v2 TEXT, standard_value_mg_kg REAL,
                conversion_path TEXT, value_quality TEXT,
                task_head TEXT
            )
            '''
        )
        rows = [
            ("A1", "10", "CC", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.0, "[10]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("P1", "11", "CCC", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.1, "[11]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("P2", "12", "CCCC", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "mol/L", 2.2, "[12]", "water_mol_l", "mol/L", None, "identity", "exact", "ECx_Mortality"),
            ("M1", "20", "CCO", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "mg/kg:soil", 2.0, "[20]", "soil_mg_kg", "mg/kg", 0.01, "raw", "exact", "ECx_Mortality"),
            ("M2", "21", "CCN", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "mg/kg:soil", 1.5, "[21]", "soil_mg_kg", "mg/kg", 0.0316227766, "raw", "exact", "ECx_Mortality"),
        ]
        conn.executemany(
            f"INSERT INTO {SOURCE_TABLE} VALUES ({', '.join('?' for _ in rows[0])})",
            rows,
        )
        conn.execute(
            "CREATE TABLE task_records_qc (result_id TEXT, molecular_weight_g_mol REAL, molecular_weight_rdkit_g_mol REAL)"
        )
        conn.executemany(
            "INSERT INTO task_records_qc VALUES (?, ?, ?)",
            [("20", 100.0, 100.0), ("21", 50.0, 50.0)],
        )
        conn.execute("CREATE TABLE soil_mgkg_parent (aggregate_id TEXT)")
        conn.executemany("INSERT INTO soil_mgkg_parent VALUES (?)", [("M1",), ("M2",)])
        conn.execute(
            """
            CREATE TABLE split_assignments (
                split_name TEXT NOT NULL, record_id TEXT, aggregate_id TEXT,
                split_part TEXT NOT NULL, seed INTEGER NOT NULL, split_type TEXT NOT NULL,
                source_table TEXT NOT NULL, group_key TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("Parent", None, "M1", "train", 42, "random", "soil_mgkg_parent", "M1"),
                ("Parent", None, "M2", "test", 42, "random", "soil_mgkg_parent", "M2"),
            ],
        )
        conn.commit()

    build_paired_targets(
        db_path=db_path,
        source_table=SOURCE_TABLE,
        mw_source_table="task_records_qc",
        parent_mgkg_source_table="soil_mgkg_parent",
        parent_mgkg_split="Parent",
        output_table=OUTPUT_TABLE,
        mw_map_table="mw_map",
        matched_mgkg_split="MassMatched",
        molkg_split="MolarMatched",
    )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("SoilPtox", None, "P1", "train", 42, "random", OUTPUT_TABLE, "P1"),
                ("SoilPtox", None, "P2", "test", 42, "random", OUTPUT_TABLE, "P2"),
            ],
        )
        conn.commit()

    summary = build_three_stage_split(
        db_path=db_path,
        source_table=OUTPUT_TABLE,
        soil_ptox_source_table=OUTPUT_TABLE,
        soil_ptox_split_name="SoilPtox",
        soil_mgkg_source_table=OUTPUT_TABLE,
        soil_mgkg_split_name="MolarMatched",
        split_name="ThreeStageMolar",
        stage3_target_name="neg_log10_mol_kg",
        stage3_target_family="solid_neglog_mol_kg",
        stage3_label="soil_molkg",
    )
    assert summary == {"finetune": 1, "finetune_mgkg": 1, "test": 1, "train": 1}
    frame = load_split_frame(
        db_path,
        split_name="ThreeStageMolar",
        source_table=OUTPUT_TABLE,
        allow_mixed_target_dimensions=True,
    )
    stage3 = frame.loc[frame["split_part"].isin(["finetune_mgkg", "test"])]
    assert set(stage3["target_name"]) == {"neg_log10_mol_kg"}
    assert set(stage3["target_family"]) == {"solid_neglog_mol_kg"}
    assert not set(stage3["task_head"]) & {"ECx_Mortality__aquatic_pTox_mol_L"}
