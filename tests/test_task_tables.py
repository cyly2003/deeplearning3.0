from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from qsar_tl.data.task_tables import aggregate_task_records


class TaskTableTests(unittest.TestCase):
    def test_aggregate_task_records_preserves_medium_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tasks.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute(
                    """
                    CREATE TABLE task_records (
                        result_id TEXT,
                        cas_number TEXT,
                        dtxsid TEXT,
                        chemical_name TEXT,
                        smiles TEXT,
                        species_number TEXT,
                        latin_name TEXT,
                        common_name TEXT,
                        task_head TEXT,
                        task_family TEXT,
                        effect_family TEXT,
                        effect_level_x REAL,
                        target_name TEXT,
                        target_basis TEXT,
                        primary_medium TEXT,
                        habitat_labels TEXT,
                        organism_habitat TEXT,
                        media_type TEXT,
                        organism_lifestage TEXT,
                        duration_bin_h REAL,
                        duration_bin_rule TEXT,
                        target_value REAL,
                        task_status TEXT
                    )
                    """
                )
                rows = [
                    (
                        "r1",
                        "50-00-0",
                        "DTXSID",
                        "Formaldehyde",
                        "C=O",
                        "sp1",
                        "Daphnia magna",
                        "water flea",
                        "ECx_Mortality",
                        "ECx",
                        "Mortality",
                        50.0,
                        "ptox_mol_l",
                        "mol/L",
                        "Freshwater",
                        "Water",
                        "Water",
                        "FW",
                        "AD",
                        48.0,
                        "exact",
                        1.0,
                        "included",
                    ),
                    (
                        "r2",
                        "50-00-0",
                        "DTXSID",
                        "Formaldehyde",
                        "C=O",
                        "sp1",
                        "Daphnia magna",
                        "water flea",
                        "ECx_Mortality",
                        "ECx",
                        "Mortality",
                        50.0,
                        "ptox_mol_l",
                        "mol/L",
                        "Freshwater",
                        "Water",
                        "Water",
                        "FW",
                        "AD",
                        48.0,
                        "exact",
                        1.2,
                        "included",
                    ),
                ]
                conn.executemany(
                    """
                    INSERT INTO task_records VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    rows,
                )
                stats = aggregate_task_records(conn)
                self.assertEqual(stats["aggregated_task_records"], 1)
                row = tuple(conn.execute(
                    """
                    SELECT primary_medium, habitat_labels, organism_habitat, media_type, target_value_count
                    FROM aggregated_task_records
                    """
                ).fetchone())
            finally:
                conn.close()
            self.assertEqual(row, ("Freshwater", "Water", "Water", "FW", 2))


if __name__ == "__main__":
    unittest.main()
