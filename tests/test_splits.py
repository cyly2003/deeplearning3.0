from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from qsar_tl.evaluation.splits import (
    MediumTransferRules,
    assign_group_parts,
    assign_medium_transfer_parts,
    assign_random_parts,
    generate_and_write_split,
)


class SplitTests(unittest.TestCase):
    def test_random_split_default_counts(self) -> None:
        parts = assign_random_parts(10, seed=42)
        self.assertEqual(parts.count("train"), 8)
        self.assertEqual(parts.count("valid"), 1)
        self.assertEqual(parts.count("test"), 1)

    def test_chemical_group_split_keeps_test_chemicals_out_of_train(self) -> None:
        records = [
            {"record_id": idx, "cas_number": f"CAS-{idx // 2}", "species_number": f"SP-{idx % 3}"}
            for idx in range(20)
        ]
        parts, _ = assign_group_parts(records, ("cas_number",), seed=7)
        train_chemicals = {row["cas_number"] for row, part in zip(records, parts) if part == "train"}
        test_chemicals = {row["cas_number"] for row, part in zip(records, parts) if part == "test"}
        self.assertFalse(train_chemicals & test_chemicals)

    def test_chemical_species_group_uses_pair_key(self) -> None:
        records = [
            {"record_id": 1, "cas_number": "A", "species_number": "S1"},
            {"record_id": 2, "cas_number": "A", "species_number": "S1"},
            {"record_id": 3, "cas_number": "A", "species_number": "S2"},
            {"record_id": 4, "cas_number": "B", "species_number": "S1"},
        ]
        parts, keys = assign_group_parts(records, ("cas_number", "species_number"), seed=42)
        pair_to_parts: dict[str, set[str]] = {}
        for key, part in zip(keys, parts):
            pair_to_parts.setdefault(key, set()).add(part)
        self.assertTrue(all(len(value) == 1 for value in pair_to_parts.values()))

    def test_medium_transfer_default_domains(self) -> None:
        records = [
            {"record_id": 1, "primary_medium": "Freshwater"},
            {"record_id": 2, "primary_medium": "Soil"},
            {"record_id": 3, "primary_medium": "Unknown"},
        ]
        parts, _ = assign_medium_transfer_parts(records, rules=MediumTransferRules())
        self.assertEqual(parts, ["train", "test", "valid"])

    def test_generate_and_write_split_assignments_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE aggregated_task_records (
                        aggregate_id TEXT,
                        cas_number TEXT,
                        species_number TEXT,
                        target_value REAL
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO aggregated_task_records VALUES (?, ?, ?, ?)",
                    [(str(idx), f"CAS-{idx}", f"SP-{idx}", float(idx)) for idx in range(10)],
                )
                conn.commit()

            summary = generate_and_write_split(
                db_path,
                split_name="unit_random",
                split_type="random_split",
                seed=42,
            )
            self.assertEqual(summary, {"test": 1, "train": 8, "valid": 1})

            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    """
                    SELECT split_name, aggregate_id, split_part, seed
                    FROM split_assignments
                    WHERE split_name = 'unit_random'
                    """
                ).fetchall()
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row[0] == "unit_random" and row[3] == 42 for row in rows))

    def test_baseline_smoke_uses_target_records_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            out_path = Path(tmpdir) / "baseline_metrics.csv"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE target_records (
                        result_id TEXT,
                        cas_number TEXT,
                        species_number TEXT,
                        molecular_weight_g_mol REAL,
                        endpoint TEXT,
                        target_name TEXT,
                        target_status TEXT,
                        target_value REAL
                    )
                    """
                )
                rows = [
                    (
                        str(idx),
                        f"CAS-{idx % 6}",
                        f"SP-{idx % 4}",
                        100.0 + idx,
                        "LC50",
                        "log10_toxicity",
                        "included",
                        float(idx) / 10.0,
                    )
                    for idx in range(30)
                ]
                conn.executemany("INSERT INTO target_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                conn.commit()

            generate_and_write_split(
                db_path,
                split_name="baseline_smoke",
                split_type="random_split",
                seed=42,
            )
            try:
                from qsar_tl.training.baseline import run_baseline
            except ImportError as exc:
                raise unittest.SkipTest(f"scikit-learn baseline dependencies are unavailable: {exc}") from exc

            result = run_baseline(
                db_path,
                split_name="baseline_smoke",
                model_name="random_forest",
                out_path=out_path,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(result.prediction_count, 0)
            self.assertTrue(any(row["split_part"] == "test" for row in result.metrics))


if __name__ == "__main__":
    unittest.main()
