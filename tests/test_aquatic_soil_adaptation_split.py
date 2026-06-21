from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.build_aquatic_soil_adaptation_split import build_adaptation_splits
from qsar_tl.training.baseline import load_split_frame


class AquaticSoilAdaptationSplitTests(unittest.TestCase):
    def test_builds_transfer_and_soil_only_low_data_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE aggregated_task_records_aquatic_soil_ptox_qc (
                        aggregate_id TEXT,
                        medium_domain TEXT,
                        target_name TEXT,
                        cas_number TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE aggregated_task_records_soil_ptox_qc (
                        aggregate_id TEXT,
                        medium_domain TEXT,
                        target_name TEXT,
                        cas_number TEXT
                    )
                    """
                )
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
                aquatic_rows = [(f"A{idx}", "aquatic", "ptox_mol_l", f"WA-{idx}") for idx in range(4)]
                soil_rows = [(str(idx), "soil", "ptox_mol_l", f"CAS-{idx // 2}") for idx in range(10)]
                conn.executemany(
                    "INSERT INTO aggregated_task_records_aquatic_soil_ptox_qc VALUES (?, ?, ?, ?)",
                    aquatic_rows + soil_rows,
                )
                conn.executemany(
                    "INSERT INTO aggregated_task_records_soil_ptox_qc VALUES (?, ?, ?, ?)",
                    soil_rows,
                )
                base_assignments = []
                for idx in range(10):
                    part = "test" if idx >= 8 else "train"
                    base_assignments.append(
                        (
                            "SoilPtoxQC_C_chemical_holdout_8_2",
                            None,
                            str(idx),
                            part,
                            42,
                            "chemical_holdout_split",
                            "aggregated_task_records_soil_ptox_qc",
                            f"CAS-{idx // 2}",
                        )
                    )
                conn.executemany(
                    "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    base_assignments,
                )
                conn.commit()

            transfer_summary, soil_only_summary = build_adaptation_splits(
                db_path=db_path,
                source_table="aggregated_task_records_aquatic_soil_ptox_qc",
                soil_source_table="aggregated_task_records_soil_ptox_qc",
                soil_split_name="SoilPtoxQC_C_chemical_holdout_8_2",
                split_name="M_qc_aquatic_to_soil_ptox_adapt_C_f50",
                soil_only_split_name="SoilPtoxQC_C_low_f50",
                soil_finetune_fraction=0.5,
                seed=42,
            )

            self.assertEqual(transfer_summary["train"], 4)
            self.assertEqual(transfer_summary["test"], 2)
            self.assertGreaterEqual(transfer_summary["finetune"], 4)
            self.assertEqual(soil_only_summary["test"], 2)
            self.assertEqual(soil_only_summary["train"], transfer_summary["finetune"])

            with closing(sqlite3.connect(db_path)) as conn:
                transfer_test = {
                    row[0]
                    for row in conn.execute(
                        """
                        SELECT aggregate_id FROM split_assignments
                        WHERE split_name = 'M_qc_aquatic_to_soil_ptox_adapt_C_f50'
                          AND split_part = 'test'
                        """
                    )
                }
                soil_only_test = {
                    row[0]
                    for row in conn.execute(
                        """
                        SELECT aggregate_id FROM split_assignments
                        WHERE split_name = 'SoilPtoxQC_C_low_f50'
                          AND split_part = 'test'
                        """
                    )
                }
                soil_only_train = {
                    row[0]
                    for row in conn.execute(
                        """
                        SELECT aggregate_id FROM split_assignments
                        WHERE split_name = 'SoilPtoxQC_C_low_f50'
                          AND split_part = 'train'
                        """
                    )
                }
            self.assertEqual(transfer_test, soil_only_test)
            self.assertFalse(soil_only_train & soil_only_test)

    def test_transfer_load_filters_cross_medium_aggregate_id_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE combined_records (
                        aggregate_id TEXT,
                        medium_domain TEXT,
                        target_name TEXT,
                        task_head TEXT,
                        target_value REAL
                    )
                    """
                )
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
                    "INSERT INTO combined_records VALUES (?, ?, ?, ?, ?)",
                    [
                        ("1", "aquatic", "ptox_mol_l", "ECx_Mortality", 1.0),
                        ("1", "soil", "ptox_mol_l", "ECx_Mortality", 2.0),
                    ],
                )
                conn.execute(
                    "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "M_collision",
                        None,
                        "1",
                        "test",
                        42,
                        "aquatic_soil_adaptation",
                        "combined_records",
                        "soil_test",
                    ),
                )
                conn.commit()

            frame = load_split_frame(db_path, split_name="M_collision", source_table="combined_records")

            self.assertEqual(frame["medium_domain"].tolist(), ["soil"])
            self.assertEqual(frame.attrs["split_join_audit"]["removed_rows_count"], 1)

    def test_builds_soil_mg_kg_low_data_pair_with_target_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE combined_records (
                        aggregate_id TEXT,
                        medium_domain TEXT,
                        target_name TEXT,
                        cas_number TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE soil_records (
                        aggregate_id TEXT,
                        medium_domain TEXT,
                        target_name TEXT,
                        cas_number TEXT
                    )
                    """
                )
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
                aquatic_rows = [(f"A{idx}", "aquatic", "ptox_mol_l", f"WA-{idx}") for idx in range(2)]
                soil_rows = [(f"S{idx}", "soil", "neg_log10_mg_kg", f"CAS-{idx // 2}") for idx in range(5)]
                conn.executemany("INSERT INTO combined_records VALUES (?, ?, ?, ?)", aquatic_rows + soil_rows)
                conn.executemany("INSERT INTO soil_records VALUES (?, ?, ?, ?)", soil_rows)
                assignments = []
                for idx in range(5):
                    assignments.append(
                        (
                            "SoilMg_C_chemical_holdout_8_2",
                            None,
                            f"S{idx}",
                            "test" if idx == 4 else "train",
                            42,
                            "chemical_holdout_split",
                            "soil_records",
                            f"CAS-{idx // 2}",
                        )
                    )
                conn.executemany("INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", assignments)
                conn.commit()

            transfer_summary, soil_only_summary = build_adaptation_splits(
                db_path=db_path,
                source_table="combined_records",
                soil_source_table="soil_records",
                soil_split_name="SoilMg_C_chemical_holdout_8_2",
                split_name="M_aquatic_to_soil_mgkg_adapt_C_f50",
                soil_only_split_name="SoilMg_C_low_f50",
                soil_finetune_fraction=0.5,
                aquatic_target_name="ptox_mol_l",
                soil_target_name="neg_log10_mg_kg",
                seed=42,
            )

            self.assertEqual(transfer_summary["train"], 2)
            self.assertEqual(transfer_summary["test"], 1)
            self.assertGreaterEqual(transfer_summary["finetune"], 2)
            self.assertEqual(soil_only_summary["test"], 1)
            self.assertEqual(soil_only_summary["train"], transfer_summary["finetune"])


if __name__ == "__main__":
    unittest.main()
