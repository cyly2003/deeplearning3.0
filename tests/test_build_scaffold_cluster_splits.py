from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.build_scaffold_cluster_splits import build_scaffold_cluster_splits


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(rdkit_available(), "RDKit is required for scaffold/cluster split tests")
class ScaffoldClusterSplitTests(unittest.TestCase):
    def test_scaffold_cluster_split_keeps_structure_groups_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            audit_dir = Path(tmpdir) / "audit"
            with closing(sqlite3.connect(db_path)) as conn:
                create_source_table(conn)
                rows = [
                    ("1", "108-88-3", "DTX1", "Toluene", "Cc1ccccc1", "ECx_Growth"),
                    ("2", "71-43-2", "DTX2", "Benzene", "c1ccccc1", "ECx_Mortality"),
                    ("3", "110-82-7", "DTX3", "Cyclohexane", "C1CCCCC1", "NOEC_Growth"),
                    ("4", "142-82-5", "DTX4", "Heptane", "CCCCCCC", "LOEC_Growth"),
                    ("5", "67-64-1", "DTX5", "Acetone", "CC(=O)C", "LOEC_Reproduction"),
                    ("6", "64-17-5", "DTX6", "Ethanol", "CCO", "NOEC_Reproduction"),
                ]
                conn.executemany("INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?)", rows)
                conn.commit()

            result = build_scaffold_cluster_splits(
                db_path=db_path,
                source_table="source_records",
                split_name_prefix="Toy_",
                audit_dir=audit_dir,
                tanimoto_threshold=0.65,
                seed=7,
            )

            self.assertEqual(result["chemical_count"], 6)
            with closing(sqlite3.connect(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                assignments = conn.execute(
                    """
                    SELECT split_name, aggregate_id, split_part, group_key
                    FROM split_assignments
                    ORDER BY split_name, aggregate_id
                    """
                ).fetchall()
            self.assertTrue(assignments)
            by_split: dict[str, dict[str, set[str]]] = {}
            for row in assignments:
                by_split.setdefault(row["split_name"], {}).setdefault(row["group_key"], set()).add(row["split_part"])
            for split_groups in by_split.values():
                self.assertTrue(all(len(parts) == 1 for parts in split_groups.values()))

            overlap_rows = list(csv.DictReader((audit_dir / "structure_overlap_audit.csv").open(encoding="utf-8")))
            self.assertTrue(overlap_rows)
            self.assertTrue(all(int(row["group_overlap_n"]) == 0 for row in overlap_rows))
            self.assertTrue(all(int(row["canonical_overlap_n"]) == 0 for row in overlap_rows))

    def test_missing_smiles_are_reported_when_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "modeling.sqlite"
            audit_dir = Path(tmpdir) / "audit"
            with closing(sqlite3.connect(db_path)) as conn:
                create_source_table(conn)
                rows = [
                    ("1", "50-00-0", "DTX1", "Formaldehyde", "C=O", "ECx_Growth"),
                    ("2", "missing-cas", "DTX-MISSING", "Missing smiles chemical", "", "NOEC_Growth"),
                    ("3", "64-17-5", "DTX2", "Ethanol", "CCO", "LOEC_Growth"),
                ]
                conn.executemany("INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?)", rows)
                conn.commit()

            build_scaffold_cluster_splits(
                db_path=db_path,
                source_table="source_records",
                split_name_prefix="Toy_",
                audit_dir=audit_dir,
                tanimoto_threshold=0.65,
                seed=11,
            )

            missing_rows = list(csv.DictReader((audit_dir / "missing_structure_report.csv").open(encoding="utf-8")))
            self.assertEqual(len(missing_rows), 1)
            self.assertEqual(missing_rows[0]["dtxsid"], "DTX-MISSING")
            self.assertEqual(missing_rows[0]["structure_status"], "invalid")


def create_source_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE source_records (
            aggregate_id TEXT,
            cas_number TEXT,
            dtxsid TEXT,
            chemical_name TEXT,
            smiles TEXT,
            task_head TEXT
        )
        """
    )


if __name__ == "__main__":
    unittest.main()
