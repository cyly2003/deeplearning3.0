from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from qsar_tl.data.target_builder import build_target_from_standardized_value


WIDE_TABLE_COLUMNS = [
    "result_id",
    "test_id",
    "reference_number",
    "cas_number",
    "species_number",
    "chemical_name",
    "dtxsid",
    "smiles",
    "molecular_weight_g_mol",
    "molecular_weight_rdkit_g_mol",
    "molecular_weight_source",
    "molecular_weight_status",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "latin_name",
    "common_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "organism_lifestage",
    "media_type",
    "exposure_duration_mean_h",
    "exposure_duration_min_h",
    "exposure_duration_max_h",
    "exposure_duration_standardization_status",
    "num_doses_mean",
    "num_doses_min",
    "num_doses_max",
    "endpoint",
    "effect",
    "measurement",
    "trend",
    "obs_duration_mean_h",
    "obs_duration_standardization_status",
    "conc1_type",
    "conc1_mean_op",
    "conc1_mean_standardized",
    "conc1_min_op",
    "conc1_min_standardized",
    "conc1_max_op",
    "conc1_max_standardized",
    "conc1_standard_unit",
    "conc1_unit_family",
    "conc1_standardization_status",
]


TARGET_COLUMNS = WIDE_TABLE_COLUMNS + [
    "tox_value",
    "tox_value_source",
    "tox_value_imputed",
    "target_value",
    "target_name",
    "target_basis",
    "target_status",
    "excluded_reason",
]


def wide_table_query(limit: int | None = None) -> str:
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    return f"""
    SELECT
        r.result_id,
        t.test_id,
        t.reference_number,
        t.test_cas AS cas_number,
        t.species_number,
        c.chemical_name,
        c.dtxsid,
        c.smiles,
        c.molecular_weight_g_mol,
        c.molecular_weight_rdkit_g_mol,
        c.molecular_weight_source,
        c.molecular_weight_status,
        cc.chemical_class_l1,
        cc.chemical_class_l2,
        cc.chemical_class_l3,
        s.latin_name,
        s.common_name,
        s.kingdom,
        s.phylum_division AS phylum,
        s.class AS class_name,
        s.tax_order,
        s.family,
        s.genus,
        s.species,
        sc.taxon_group_l1,
        sc.taxon_group_l2,
        sc.taxon_group_l3,
        s.primary_medium,
        s.habitat_labels,
        t.organism_habitat,
        t.organism_lifestage,
        t.media_type,
        t.exposure_duration_mean_h,
        t.exposure_duration_min_h,
        t.exposure_duration_max_h,
        t.exposure_duration_standardization_status,
        t.num_doses_mean,
        t.num_doses_min,
        t.num_doses_max,
        r.endpoint,
        r.effect,
        r.measurement,
        r.trend,
        r.obs_duration_mean_h,
        r.obs_duration_standardization_status,
        r.conc1_type,
        r.conc1_mean_op,
        r.conc1_mean_standardized,
        r.conc1_min_op,
        r.conc1_min_standardized,
        r.conc1_max_op,
        r.conc1_max_standardized,
        r.conc1_standard_unit,
        r.conc1_unit_family,
        r.conc1_standardization_status
    FROM results AS r
    JOIN tests AS t ON r.test_id = t.test_id
    LEFT JOIN chemicals AS c ON t.test_cas = c.cas_number
    LEFT JOIN chemical_category_curated AS cc ON CAST(t.test_cas AS TEXT) = cc.cas_number
    LEFT JOIN species AS s ON t.species_number = s.species_number
    LEFT JOIN species_category_curated AS sc ON t.species_number = sc.species_number
    ORDER BY r.result_id
    {limit_clause}
    """


def create_table(conn: sqlite3.Connection, table_name: str, columns: Iterable[str]) -> None:
    column_sql = ", ".join(f'"{column}"' for column in columns)
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({column_sql})')


def insert_rows(conn: sqlite3.Connection, table_name: str, columns: list[str], rows: list[dict[str, object]]) -> None:
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    values = [[row.get(column) for column in columns] for row in rows]
    conn.executemany(
        f'INSERT INTO "{table_name}" ({column_sql}) VALUES ({placeholders})',
        values,
    )


def build_modeling_tables(
    source_db: str | Path,
    output_db: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 10000,
    min_dose_groups_for_midpoint: int = 3,
) -> dict[str, int]:
    source_path = Path(source_db)
    output_path = Path(output_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "wide_records": 0,
        "target_records": 0,
        "included_targets": 0,
        "excluded_targets": 0,
    }

    with sqlite3.connect(source_path) as src, sqlite3.connect(output_path) as out:
        src.row_factory = sqlite3.Row
        out.execute("PRAGMA journal_mode = WAL")
        out.execute("PRAGMA synchronous = NORMAL")
        create_table(out, "wide_records", WIDE_TABLE_COLUMNS)
        create_table(out, "target_records", TARGET_COLUMNS)
        out.execute('DROP TABLE IF EXISTS "build_manifest"')
        out.execute('CREATE TABLE "build_manifest" ("key", "value")')

        cursor = src.execute(wide_table_query(limit=limit))
        while True:
            fetched = cursor.fetchmany(batch_size)
            if not fetched:
                break

            wide_rows: list[dict[str, object]] = []
            target_rows: list[dict[str, object]] = []
            for row in fetched:
                wide_row = {column: row[column] for column in WIDE_TABLE_COLUMNS}
                wide_rows.append(wide_row)

                target = build_target_from_standardized_value(
                    mean_value=row["conc1_mean_standardized"],
                    min_value=row["conc1_min_standardized"],
                    max_value=row["conc1_max_standardized"],
                    dose_group_count=row["num_doses_mean"],
                    unit_family=row["conc1_unit_family"],
                    standard_unit=row["conc1_standard_unit"],
                    molecular_weight_g_mol=row["molecular_weight_g_mol"]
                    or row["molecular_weight_rdkit_g_mol"],
                    medium=row["media_type"] or row["organism_habitat"],
                    min_dose_groups_for_midpoint=min_dose_groups_for_midpoint,
                )
                target_row = wide_row | asdict(target)
                target_rows.append(target_row)

                stats["wide_records"] += 1
                stats["target_records"] += 1
                if target.target_status == "included":
                    stats["included_targets"] += 1
                else:
                    stats["excluded_targets"] += 1

            insert_rows(out, "wide_records", WIDE_TABLE_COLUMNS, wide_rows)
            insert_rows(out, "target_records", TARGET_COLUMNS, target_rows)
            out.commit()

        out.executemany(
            'INSERT INTO "build_manifest" ("key", "value") VALUES (?, ?)',
            [
                ("source_db", str(source_path)),
                ("limit", "" if limit is None else str(limit)),
                ("batch_size", str(batch_size)),
                ("min_dose_groups_for_midpoint", str(min_dose_groups_for_midpoint)),
                ("stats_json", json.dumps(stats, ensure_ascii=False, sort_keys=True)),
            ],
        )
        out.commit()

    return stats


def summarize_target_table(output_db: str | Path) -> list[tuple[str | None, str | None, str | None, int]]:
    with sqlite3.connect(output_db) as conn:
        return conn.execute(
            """
            SELECT target_status, target_name, excluded_reason, COUNT(*) AS n
            FROM target_records
            GROUP BY target_status, target_name, excluded_reason
            ORDER BY n DESC
            """
        ).fetchall()
