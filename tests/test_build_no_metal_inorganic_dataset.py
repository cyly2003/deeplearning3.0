from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.build_no_metal_inorganic_dataset import build_no_metal_inorganic_dataset


class BuildNoMetalInorganicDatasetTests(unittest.TestCase):
    def test_filters_source_tables_and_target_records_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_db = Path(tmpdir) / "source.sqlite"
            out_db = Path(tmpdir) / "filtered.sqlite"
            with closing(sqlite3.connect(source_db)) as conn:
                conn.execute(
                    """
                    CREATE TABLE target_records (
                        result_id TEXT,
                        aggregate_id TEXT,
                        cas_number TEXT,
                        dtxsid TEXT,
                        smiles TEXT,
                        chemical_class_l1 TEXT,
                        chemical_class_l2 TEXT,
                        value_quality TEXT,
                        target_status TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE aggregated_task_records_aquatic_soil_ptox_qc (
                        aggregate_id TEXT,
                        cas_number TEXT,
                        dtxsid TEXT,
                        smiles TEXT,
                        medium_domain TEXT,
                        target_name TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE aggregated_task_records_soil_ptox_qc (
                        aggregate_id TEXT,
                        cas_number TEXT,
                        dtxsid TEXT,
                        smiles TEXT,
                        medium_domain TEXT,
                        target_name TEXT
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO target_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("r1", "1", "50-00-0", "DTX1", "C=O", "organic", "aldehyde", "exact", "included"),
                        ("r2", "2", "7440-43-9", "DTX2", "[Cd]", "inorganic", "metal_metalloid", "exact", "included"),
                        ("r3", "3", "100-00-5", "DTX3", "Clc1ccc([N+](=O)[O-])cc1", "unknown", "unclassified", "censored", "excluded"),
                    ],
                )
                rows = [
                    ("1", "50-00-0", "DTX1", "C=O", "aquatic", "ptox_mol_l"),
                    ("2", "7440-43-9", "DTX2", "[Cd]", "aquatic", "ptox_mol_l"),
                    ("3", "100-00-5", "DTX3", "Clc1ccc([N+](=O)[O-])cc1", "soil", "ptox_mol_l"),
                ]
                conn.executemany(
                    "INSERT INTO aggregated_task_records_aquatic_soil_ptox_qc VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.executemany(
                    "INSERT INTO aggregated_task_records_soil_ptox_qc VALUES (?, ?, ?, ?, ?, ?)",
                    [rows[2]],
                )
                conn.commit()

            stats = build_no_metal_inorganic_dataset(
                source_db=source_db,
                out_db=out_db,
                tables=(
                    "target_records",
                    "aggregated_task_records_aquatic_soil_ptox_qc",
                    "aggregated_task_records_soil_ptox_qc",
                ),
            )

            self.assertEqual(
                stats["tables"]["aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"]["output_rows"],
                2,
            )
            with closing(sqlite3.connect(source_db)) as conn:
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM aggregated_task_records_aquatic_soil_ptox_qc").fetchone()[0],
                    3,
                )
            with closing(sqlite3.connect(out_db)) as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM target_records").fetchone()[0], 2)
                kept_cas = {
                    row[0]
                    for row in conn.execute(
                        "SELECT cas_number FROM aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"
                    )
                }
                self.assertEqual(kept_cas, {"50-00-0", "100-00-5"})
                self.assertIsNotNone(
                    conn.execute("SELECT value FROM no_metal_inorganic_build_manifest WHERE key='filter'").fetchone()
                )


if __name__ == "__main__":
    unittest.main()
