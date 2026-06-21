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
    build_experiment_split_sets,
    generate_and_write_split,
    generate_and_write_experiment_splits,
    validate_min_records,
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

    def test_medium_transfer_prefers_curated_medium_domain(self) -> None:
        records = [
            {
                "record_id": 1,
                "medium_domain": "aquatic",
                "primary_medium": "soil",
                "media_type": "FW",
            },
            {
                "record_id": 2,
                "medium_domain": "soil",
                "primary_medium": "aquatic",
                "media_type": "AQU",
            },
        ]
        parts, _ = assign_medium_transfer_parts(records, rules=MediumTransferRules())
        self.assertEqual(parts, ["train", "test"])

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

            try:
                result = run_baseline(
                    db_path,
                    split_name="baseline_smoke",
                    model_name="random_forest",
                    out_path=out_path,
                )
            except ImportError as exc:
                raise unittest.SkipTest(f"scikit-learn baseline dependencies are unavailable: {exc}") from exc
            self.assertTrue(out_path.exists())
            self.assertGreater(result.prediction_count, 0)
            self.assertTrue(any(row["split_part"] == "test" for row in result.metrics))

    def test_experiment_split_f_keeps_chemicals_in_one_part(self) -> None:
        records = [
            {"aggregate_id": idx, "cas_number": f"CAS-{idx // 3}", "species_number": f"SP-{idx % 4}"}
            for idx in range(60)
        ]
        split_sets = build_experiment_split_sets(
            records,
            code="F",
            source_table="aggregated_task_records",
            id_column="aggregate_id",
            seed=42,
        )
        assignments = split_sets["F_chemical_adapt_7_2_1"]
        chemical_parts: dict[str, set[str]] = {}
        for record, assignment in zip(records, assignments):
            chemical_parts.setdefault(record["cas_number"], set()).add(assignment.split_part)
        self.assertTrue(all(len(parts) == 1 for parts in chemical_parts.values()))
        self.assertTrue({"train", "finetune", "test"}.issubset({item.split_part for item in assignments}))

    def test_experiment_split_d_generates_five_chemical_group_folds(self) -> None:
        records = [
            {"aggregate_id": idx, "cas_number": f"CAS-{idx // 2}", "species_number": f"SP-{idx % 4}"}
            for idx in range(40)
        ]
        split_sets = build_experiment_split_sets(
            records,
            code="D",
            source_table="aggregated_task_records",
            id_column="aggregate_id",
            seed=42,
        )
        self.assertEqual(len(split_sets), 5)
        for assignments in split_sets.values():
            chemical_parts: dict[str, set[str]] = {}
            for record, assignment in zip(records, assignments):
                chemical_parts.setdefault(record["cas_number"], set()).add(assignment.split_part)
            self.assertTrue(all(len(parts) == 1 for parts in chemical_parts.values()))

    def test_generate_and_write_experiment_splits(self) -> None:
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
                    [(str(idx), f"CAS-{idx // 2}", f"SP-{idx % 4}", float(idx)) for idx in range(40)],
                )
                conn.commit()

            summaries = generate_and_write_experiment_splits(
                db_path,
                split_codes=["A", "D"],
                seed=42,
            )

            self.assertIn("A_random_adapt_7_2_1", summaries)
            self.assertIn("D_chemical_group_5fold_fold1", summaries)
            self.assertEqual(len(summaries), 6)

    def test_generate_and_write_experiment_splits_accepts_name_prefix(self) -> None:
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
                    [(str(idx), f"CAS-{idx // 2}", f"SP-{idx % 4}", float(idx)) for idx in range(20)],
                )
                conn.commit()

            summaries = generate_and_write_experiment_splits(
                db_path,
                split_codes=["B"],
                seed=42,
                split_name_prefix="Aquatic_",
            )

            self.assertIn("Aquatic_B_random_8_2", summaries)
            with closing(sqlite3.connect(db_path)) as conn:
                names = {
                    row[0]
                    for row in conn.execute("SELECT DISTINCT split_name FROM split_assignments")
                }
            self.assertEqual(names, {"Aquatic_B_random_8_2"})

    def test_validate_min_records_rejects_low_sample_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "below minimum_modeling_rows=100"):
            validate_min_records([{"aggregate_id": idx} for idx in range(34)], source_table="aggregated_task_records_sediment_ptox", min_records=100)


if __name__ == "__main__":
    unittest.main()
