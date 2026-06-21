from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from qsar_tl.data.qc_aggregation import build_qc_task_tables
from qsar_tl.data.reference_weighting import parse_publication_year, reference_recency_weight, weighted_mean


def test_reference_year_parsing_and_weighted_mean() -> None:
    assert parse_publication_year("2018") == 2018
    assert parse_publication_year("published 2020 online") == 2020
    assert parse_publication_year("NR") is None
    assert reference_recency_weight("2015", latest_year=2020) == 0.8
    assert weighted_mean([1.0, 3.0], [1.0, 0.8]) == pytest.approx((1.0 + 2.4) / 1.8)


def test_qc_aggregation_uses_reference_weighted_mean_and_conflict_flag() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "qc.sqlite"
        with closing(sqlite3.connect(db_path)) as conn:
            create_task_records_fixture(conn)
            rows = [
                base_task_row(result_id="r1", test_id="t1", reference_number="ref-new", publication_year="2020", target_value=1.0),
                base_task_row(result_id="r2", test_id="t2", reference_number="ref-old", publication_year="2015", target_value=3.0),
            ]
            insert_task_rows(conn, rows)
            stats = build_qc_task_tables(db_path, min_outlier_group_n=50)
            assert stats["qc_included_records"] == 2
            assert stats["aggregated_task_records_qc"] == 1
            row = conn.execute(
                """
                SELECT target_value_median, target_value_weighted_mean,
                       cross_reference_conflict_flag, source_reference_count,
                       reference_numbers
                FROM aggregated_task_records_qc
                """
            ).fetchone()

    expected = (1.0 * 1.0 + 3.0 * 0.8) / 1.8
    assert row[0] == pytest.approx(expected)
    assert row[1] == pytest.approx(expected)
    assert row[2] == 1
    assert row[3] == 2
    assert json.loads(row[4]) == ["ref-new", "ref-old"]


def test_qc_aggregation_keeps_effect_levels_separate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "qc.sqlite"
        with closing(sqlite3.connect(db_path)) as conn:
            create_task_records_fixture(conn)
            rows = [
                base_task_row(result_id="r1", effect_level_x=10.0, target_value=1.0),
                base_task_row(result_id="r2", effect_level_x=50.0, target_value=2.0),
            ]
            insert_task_rows(conn, rows)
            stats = build_qc_task_tables(db_path, min_outlier_group_n=50)

    assert stats["aggregated_task_records_qc"] == 2


def create_task_records_fixture(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE task_records (
            result_id TEXT,
            test_id TEXT,
            reference_number TEXT,
            publication_year TEXT,
            cas_number TEXT,
            dtxsid TEXT,
            chemical_name TEXT,
            smiles TEXT,
            species_number TEXT,
            latin_name TEXT,
            common_name TEXT,
            kingdom TEXT,
            phylum TEXT,
            class_name TEXT,
            tax_order TEXT,
            family TEXT,
            genus TEXT,
            species TEXT,
            task_head TEXT,
            task_family TEXT,
            effect_family TEXT,
            effect_level_x REAL,
            target_name TEXT,
            target_basis TEXT,
            unit_family_v2 TEXT,
            standard_unit_v2 TEXT,
            active_ingredient_basis INTEGER,
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


def base_task_row(
    *,
    result_id: str,
    test_id: str = "t1",
    reference_number: str = "ref1",
    publication_year: str = "2020",
    effect_level_x: float = 50.0,
    target_value: float,
) -> tuple[object, ...]:
    return (
        result_id,
        test_id,
        reference_number,
        publication_year,
        "50-00-0",
        "DTXSID",
        "Formaldehyde",
        "C=O",
        "sp1",
        "Eisenia fetida",
        "earthworm",
        "Animalia",
        "Annelida",
        "Clitellata",
        "Haplotaxida",
        "Lumbricidae",
        "Eisenia",
        "fetida",
        "ECx_Mortality",
        "ECx",
        "Mortality",
        effect_level_x,
        "neg_log10_mg_kg",
        "mg/kg:NAT",
        "soil_mg_kg",
        "mg/kg",
        0,
        "soil",
        "soil",
        "soil",
        "NAT",
        "AD",
        336.0,
        "round_0.5h_ge_24h",
        target_value,
        "included",
    )


def insert_task_rows(conn: sqlite3.Connection, rows: list[tuple[object, ...]]) -> None:
    placeholders = ", ".join("?" for _ in rows[0])
    conn.executemany(f"INSERT INTO task_records VALUES ({placeholders})", rows)
    conn.commit()
