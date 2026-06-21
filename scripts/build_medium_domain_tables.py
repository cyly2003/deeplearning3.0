from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.data.medium import classify_exposure_medium


DOMAIN_TABLES = {
    "aquatic": "aggregated_task_records_aquatic",
    "aquatic_plus_sediment": "aggregated_task_records_aquatic_plus_sediment",
    "soil": "aggregated_task_records_soil",
    "sediment": "aggregated_task_records_sediment",
    "solid": "aggregated_task_records_solid",
    "aquatic_ptox": "aggregated_task_records_aquatic_ptox",
    "aquatic_soil_ptox": "aggregated_task_records_aquatic_soil_ptox",
    "aquatic_sediment_ptox": "aggregated_task_records_aquatic_sediment_ptox",
    "aquatic_solid_ptox": "aggregated_task_records_aquatic_solid_ptox",
    "soil_ptox": "aggregated_task_records_soil_ptox",
    "sediment_ptox": "aggregated_task_records_sediment_ptox",
    "solid_ptox": "aggregated_task_records_solid_ptox",
    "soil_mg_kg": "aggregated_task_records_soil_mg_kg",
    "sediment_mg_kg": "aggregated_task_records_sediment_mg_kg",
    "solid_mg_kg": "aggregated_task_records_solid_mg_kg",
    "aquatic_mg_kg": "aggregated_task_records_aquatic_mg_kg",
    "aquatic_percent": "aggregated_task_records_aquatic_percent",
    "sediment_percent": "aggregated_task_records_sediment_percent",
    "aquatic_mg_per_organism": "aggregated_task_records_aquatic_mg_per_organism",
    "sediment_mg_per_organism": "aggregated_task_records_sediment_mg_per_organism",
    "aquatic_mg_per_experimental_unit": "aggregated_task_records_aquatic_mg_per_experimental_unit",
    "sediment_mg_per_experimental_unit": "aggregated_task_records_sediment_mg_per_experimental_unit",
    "soil_g_ha": "aggregated_task_records_soil_g_ha",
    "sediment_g_ha": "aggregated_task_records_sediment_g_ha",
    "soil_l_ha": "aggregated_task_records_soil_l_ha",
    "sediment_l_ha": "aggregated_task_records_sediment_l_ha",
    "soil_seed": "aggregated_task_records_soil_seed",
    "soil_percent": "aggregated_task_records_soil_percent",
}

AQUATIC_MEDIA_CODES = {"FW", "SW", "AQU", "CUL"}
SEDIMENT_MEDIA_CODES = {"SED", "SEDIMENT"}
SOIL_MEDIA_CODES = {"NAT", "ART", "UKS", "LIT", "MIN", "HYP"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build medium-domain annotated and subset tables.")
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--source-table", default="aggregated_task_records")
    parser.add_argument("--annotated-table", default="aggregated_task_records_medium_domain")
    parser.add_argument("--table-suffix", default="", help="Suffix appended to generated subset table names, e.g. _qc")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f'SELECT * FROM "{args.source_table}" ORDER BY aggregate_id').fetchall()
        columns = table_columns(conn, args.source_table)
        annotated_columns = append_missing(
            columns,
            [
                "medium_domain",
                "primary_medium_domain",
                "medium_domains",
                "medium_domain_detail",
                "medium_domain_reason",
                "medium_conflict_flag",
            ],
        )
        create_table(conn, args.annotated_table, annotated_columns)
        annotated_rows = []
        for row in rows:
            payload = {column: row[column] for column in columns}
            payload.update(classify_medium_details(row))
            annotated_rows.append(payload)
        insert_rows(conn, args.annotated_table, annotated_columns, annotated_rows)
        expanded_table = f"{args.annotated_table}_expanded"
        create_expanded_medium_table(conn, args.annotated_table, expanded_table)
        create_subset_tables(conn, expanded_table, table_suffix=args.table_suffix)
        conn.commit()
        for domain, count in domain_counts(conn, args.annotated_table):
            print(f"{domain}: {count}")
        expanded_count = conn.execute(f'SELECT COUNT(1) FROM "{expanded_table}"').fetchone()[0]
        print(f"{expanded_table}: {expanded_count}")
        for name in domain_tables(table_suffix=args.table_suffix).values():
            count = conn.execute(f'SELECT COUNT(1) FROM "{name}"').fetchone()[0]
            print(f"{name}: {count}")


def classify_medium_domain(row: sqlite3.Row) -> tuple[str, str, bool]:
    details = classify_medium_details(row)
    return (
        str(details["medium_domain"]),
        str(details["medium_domain_reason"]),
        bool(details["medium_conflict_flag"]),
    )


def classify_medium_details(row: sqlite3.Row) -> dict[str, object]:
    target_name = clean_text(value_from_row(row, "target_name"))
    classification = classify_exposure_medium(
        organism_habitat=value_from_row(row, "organism_habitat"),
        media_type=value_from_row(row, "media_type"),
        target_basis=value_from_row(row, "target_basis"),
    )
    details = classification.to_row()
    if target_name in {"neg_log10_mg_kg", "neg-log10-mg-kg"}:
        domains = json.loads(str(details["medium_domains"]))
        if domains == ["unknown"]:
            details["medium_domain_reason"] = "solid_unit_without_medium_evidence"
    return details


def create_expanded_medium_table(conn: sqlite3.Connection, annotated_table: str, expanded_table: str) -> None:
    columns = table_columns(conn, annotated_table)
    expanded_columns = append_missing(columns, ["medium_assignment_weight"])
    create_table(conn, expanded_table, expanded_columns)
    rows = conn.execute(f'SELECT * FROM "{annotated_table}" ORDER BY aggregate_id').fetchall()
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = {column: row[column] for column in columns}
        domains = parse_medium_domains(payload.get("medium_domains"))
        weight = 1.0 / len(domains)
        for domain in domains:
            expanded = dict(payload)
            expanded["medium_domain"] = domain
            expanded["medium_assignment_weight"] = weight
            payloads.append(expanded)
    insert_rows(conn, expanded_table, expanded_columns, payloads)


def parse_medium_domains(value: object) -> list[str]:
    if value is None:
        return ["unknown"]
    try:
        domains = json.loads(str(value))
    except json.JSONDecodeError:
        domains = [str(value)]
    cleaned = [str(domain) for domain in domains if str(domain)]
    return cleaned or ["unknown"]


def create_subset_tables(conn: sqlite3.Connection, annotated_table: str, *, table_suffix: str = "") -> None:
    subsets = {
        "aggregated_task_records_aquatic": "medium_domain = 'aquatic'",
        "aggregated_task_records_aquatic_plus_sediment": "medium_domain IN ('aquatic', 'sediment')",
        "aggregated_task_records_soil": "medium_domain = 'soil'",
        "aggregated_task_records_sediment": "medium_domain = 'sediment'",
        "aggregated_task_records_solid": "medium_domain IN ('soil', 'sediment')",
        "aggregated_task_records_aquatic_ptox": "target_name = 'ptox_mol_l' AND medium_domain = 'aquatic'",
        "aggregated_task_records_aquatic_soil_ptox": "target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'soil')",
        "aggregated_task_records_aquatic_sediment_ptox": "target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'sediment')",
        "aggregated_task_records_aquatic_solid_ptox": "target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'soil', 'sediment')",
        "aggregated_task_records_soil_ptox": "target_name = 'ptox_mol_l' AND medium_domain = 'soil'",
        "aggregated_task_records_sediment_ptox": "target_name = 'ptox_mol_l' AND medium_domain = 'sediment'",
        "aggregated_task_records_solid_ptox": "target_name = 'ptox_mol_l' AND medium_domain IN ('soil', 'sediment')",
        "aggregated_task_records_soil_mg_kg": "target_name = 'neg_log10_mg_kg' AND medium_domain = 'soil'",
        "aggregated_task_records_sediment_mg_kg": "target_name = 'neg_log10_mg_kg' AND medium_domain = 'sediment'",
        "aggregated_task_records_solid_mg_kg": "target_name = 'neg_log10_mg_kg' AND medium_domain IN ('soil', 'sediment')",
        "aggregated_task_records_aquatic_mg_kg": "target_name = 'neg_log10_mg_kg' AND medium_domain = 'aquatic'",
        "aggregated_task_records_aquatic_percent": "target_name = 'neg_log10_percent' AND medium_domain = 'aquatic'",
        "aggregated_task_records_sediment_percent": "target_name = 'neg_log10_percent' AND medium_domain = 'sediment'",
        "aggregated_task_records_aquatic_mg_per_organism": "target_name = 'neg_log10_mg_per_organism' AND medium_domain = 'aquatic'",
        "aggregated_task_records_sediment_mg_per_organism": "target_name = 'neg_log10_mg_per_organism' AND medium_domain = 'sediment'",
        "aggregated_task_records_aquatic_mg_per_experimental_unit": "target_name = 'neg_log10_mg_per_experimental_unit' AND medium_domain = 'aquatic'",
        "aggregated_task_records_sediment_mg_per_experimental_unit": "target_name = 'neg_log10_mg_per_experimental_unit' AND medium_domain = 'sediment'",
        "aggregated_task_records_soil_g_ha": "target_name = 'neg_log10_g_ha' AND medium_domain = 'soil'",
        "aggregated_task_records_sediment_g_ha": "target_name = 'neg_log10_g_ha' AND medium_domain = 'sediment'",
        "aggregated_task_records_soil_l_ha": "target_name = 'neg_log10_l_ha' AND medium_domain = 'soil'",
        "aggregated_task_records_sediment_l_ha": "target_name = 'neg_log10_l_ha' AND medium_domain = 'sediment'",
        "aggregated_task_records_soil_seed": "target_name IN ('neg_log10_g_kg_seed', 'neg_log10_ml_kg_seed') AND medium_domain = 'soil'",
        "aggregated_task_records_soil_percent": "target_name = 'neg_log10_percent' AND medium_domain = 'soil'",
    }
    for table_name, where_clause in subsets.items():
        output_table = f"{table_name}{table_suffix}"
        conn.execute(f'DROP TABLE IF EXISTS "{output_table}"')
        conn.execute(
            f"""
            CREATE TABLE "{output_table}" AS
            SELECT *
            FROM "{annotated_table}"
            WHERE {where_clause}
            """
        )


def domain_tables(*, table_suffix: str = "") -> dict[str, str]:
    return {key: f"{value}{table_suffix}" for key, value in DOMAIN_TABLES.items()}


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    if not rows:
        raise ValueError(f"Table not found: {table_name}")
    return [row[1] for row in rows]


def create_table(conn: sqlite3.Connection, table_name: str, columns: list[str]) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    column_sql = ", ".join(f'"{column}"' for column in columns)
    conn.execute(f'CREATE TABLE "{table_name}" ({column_sql})')


def append_missing(columns: list[str], additions: list[str]) -> list[str]:
    result = list(columns)
    present = set(result)
    for column in additions:
        if column not in present:
            result.append(column)
            present.add(column)
    return result


def insert_rows(conn: sqlite3.Connection, table_name: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    values = [[row.get(column) for column in columns] for row in rows]
    conn.executemany(f'INSERT INTO "{table_name}" ({column_sql}) VALUES ({placeholders})', values)


def domain_counts(conn: sqlite3.Connection, annotated_table: str) -> list[tuple[str, int]]:
    return conn.execute(
        f"""
        SELECT medium_domain, COUNT(1) AS n
        FROM "{annotated_table}"
        GROUP BY medium_domain
        ORDER BY n DESC
        """
    ).fetchall()


def value_from_row(row: sqlite3.Row, key: str) -> object:
    if key not in row.keys():
        return None
    return row[key]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", "-")


def normalize_media_code(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    while text.endswith(("/", "*")):
        text = text[:-1]
    return text


if __name__ == "__main__":
    main()
