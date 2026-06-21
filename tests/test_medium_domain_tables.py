from __future__ import annotations

import sqlite3

from scripts.build_medium_domain_tables import create_expanded_medium_table, create_subset_tables, classify_medium_domain


def row_with_medium(
    *,
    primary_medium: str | None = None,
    organism_habitat: str | None = None,
    media_type: str | None = None,
    target_name: str | None = None,
    target_basis: str | None = None,
) -> sqlite3.Row:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE rows (
            primary_medium TEXT,
            organism_habitat TEXT,
            media_type TEXT,
            target_name TEXT,
            target_basis TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO rows VALUES (?, ?, ?, ?, ?)",
        (primary_medium, organism_habitat, media_type, target_name, target_basis),
    )
    return conn.execute("SELECT * FROM rows").fetchone()


def test_water_exposure_is_not_overridden_by_species_primary_medium() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(primary_medium="sediment", organism_habitat="Water", media_type="FW")
    )

    assert domain == "aquatic"
    assert "media_type:FW" in reason
    assert conflict is False


def test_sed_media_code_with_trailing_slash_maps_to_sediment() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(primary_medium="soil", organism_habitat="Soil", media_type="SED/")
    )

    assert domain == "sediment"
    assert "media_type:SED" in reason
    assert conflict is True


def test_cul_requires_habitat_context_for_aquatic_assignment() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(organism_habitat="Water", media_type="CUL")
    )

    assert domain == "aquatic"
    assert "CUL_culture_with_water_or_aqueous_exposure" in reason
    assert conflict is False


def test_cul_with_soil_maps_to_soil_culture() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(organism_habitat="Soil", media_type="CUL")
    )

    assert domain == "soil"
    assert "CUL_culture_with_soil_exposure" in reason
    assert conflict is False


def test_conflicting_media_and_habitat_are_flagged() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(organism_habitat="Soil", media_type="FW")
    )

    assert domain == "aquatic"
    assert "media_type:FW" in reason
    assert "organism_habitat:Soil" in reason
    assert conflict is True


def test_neg_log10_mg_kg_without_medium_evidence_is_unknown() -> None:
    domain, reason, conflict = classify_medium_domain(
        row_with_medium(target_name="neg_log10_mg_kg", target_basis="mg/kg:unknown_medium")
    )

    assert domain == "unknown"
    assert reason == "solid_unit_without_medium_evidence"
    assert conflict is False


def test_water_and_sediment_target_family_subset_tables_are_created() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE annotated (
            aggregate_id INTEGER,
            medium_domain TEXT,
            target_name TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO annotated VALUES (?, ?, ?)",
        [
            (1, "aquatic", "ptox_mol_l"),
            (2, "aquatic", "neg_log10_percent"),
            (3, "sediment", "neg_log10_mg_kg"),
            (4, "sediment", "neg_log10_mg_per_organism"),
            (5, "soil", "neg_log10_g_ha"),
        ],
    )

    create_subset_tables(conn, "annotated")

    assert table_count(conn, "aggregated_task_records_aquatic_ptox") == 1
    assert table_count(conn, "aggregated_task_records_aquatic_percent") == 1
    assert table_count(conn, "aggregated_task_records_sediment_mg_kg") == 1
    assert table_count(conn, "aggregated_task_records_sediment_mg_per_organism") == 1
    assert table_count(conn, "aggregated_task_records_soil_g_ha") == 1


def test_subset_tables_accept_suffix_for_qc_outputs() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE annotated (
            aggregate_id INTEGER,
            medium_domain TEXT,
            target_name TEXT
        )
        """
    )
    conn.execute("INSERT INTO annotated VALUES (?, ?, ?)", (1, "soil", "neg_log10_mg_kg"))

    create_subset_tables(conn, "annotated", table_suffix="_qc")

    assert table_count(conn, "aggregated_task_records_soil_mg_kg_qc") == 1


def test_expanded_medium_table_duplicates_conflicting_domains_with_weights() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE annotated (
            aggregate_id INTEGER,
            medium_domain TEXT,
            medium_domains TEXT,
            target_name TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO annotated VALUES (?, ?, ?, ?)",
        (1, "aquatic", '["aquatic", "soil"]', "ptox_mol_l"),
    )

    create_expanded_medium_table(conn, "annotated", "expanded")

    rows = conn.execute(
        "SELECT medium_domain, medium_assignment_weight FROM expanded ORDER BY medium_domain"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [("aquatic", 0.5), ("soil", 0.5)]


def table_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f'SELECT COUNT(1) FROM "{table_name}"').fetchone()[0])
