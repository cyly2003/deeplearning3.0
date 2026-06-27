from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_export_task_train_space_writes_task_chemical_and_species_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "toy.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE toy_records (
                aggregate_id INTEGER,
                cas_number TEXT,
                dtxsid TEXT,
                chemical_name TEXT,
                smiles TEXT,
                species_number TEXT,
                latin_name TEXT,
                kingdom TEXT,
                phylum TEXT,
                class_name TEXT,
                tax_order TEXT,
                family TEXT,
                genus TEXT,
                species TEXT,
                task_head TEXT,
                task_family TEXT,
                target_name TEXT,
                target_basis TEXT,
                medium_domain TEXT,
                target_value_median REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO toy_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "64-17-5",
                    "DTXSID-1",
                    "Ethanol",
                    "CCO",
                    "sp1",
                    "Daphnia magna",
                    "Animalia",
                    "Arthropoda",
                    "Branchiopoda",
                    "Diplostraca",
                    "Daphniidae",
                    "Daphnia",
                    "magna",
                    "ECx_Mortality",
                    "ECx",
                    "neg_log10_mg_kg",
                    "mg/kg:NAT",
                    "soil",
                    1.0,
                ),
                (
                    2,
                    "75-04-7",
                    "DTXSID-2",
                    "Ethylamine",
                    "CCN",
                    "sp1",
                    "Daphnia magna",
                    "Animalia",
                    "Arthropoda",
                    "Branchiopoda",
                    "Diplostraca",
                    "Daphniidae",
                    "Daphnia",
                    "magna",
                    "ECx_Mortality",
                    "ECx",
                    "neg_log10_mg_kg",
                    "mg/kg:NAT",
                    "soil",
                    1.2,
                ),
                (
                    3,
                    "7732-18-5",
                    "DTXSID-3",
                    "Water",
                    "O",
                    "sp2",
                    "Pimephales promelas",
                    "Animalia",
                    "Chordata",
                    "Actinopterygii",
                    "Cypriniformes",
                    "Leuciscidae",
                    "Pimephales",
                    "promelas",
                    "NOEC_Reproduction",
                    "NOEC",
                    "neg_log10_mg_kg",
                    "mg/kg:NAT",
                    "soil",
                    2.0,
                ),
            ],
        )
        conn.execute(
            """
            CREATE TABLE split_assignments (
                split_name TEXT,
                record_id TEXT,
                aggregate_id TEXT,
                split_part TEXT,
                seed INTEGER,
                split_type TEXT,
                source_table TEXT,
                group_key TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("toy_split", None, "1", "train", 42, "random_holdout_split", "toy_records", "64-17-5"),
                ("toy_split", None, "2", "finetune", 42, "random_holdout_split", "toy_records", "75-04-7"),
                ("toy_split", None, "3", "test", 42, "random_holdout_split", "toy_records", "7732-18-5"),
            ],
        )
        conn.commit()

    out_dir = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_task_train_space.py",
            "--db",
            str(db_path),
            "--source-table",
            "toy_records",
            "--split-name",
            "toy_split",
            "--out-dir",
            str(out_dir),
            "--target-column",
            "target_value_median",
        ],
        check=True,
    )

    rows = pd.read_csv(out_dir / "task_train_space_rows.csv.gz")
    chemicals = pd.read_csv(out_dir / "task_train_chemical_space.csv")
    species = pd.read_csv(out_dir / "task_train_species_space.csv")

    assert set(rows["split_part"]) == {"train", "finetune"}
    assert len(chemicals) == 2
    assert chemicals["unique_species_count"].tolist() == [1, 1]
    assert len(species) == 2
    assert species.loc[species["latin_name"].eq("Daphnia magna"), "unique_chemical_count"].sum() == 2
