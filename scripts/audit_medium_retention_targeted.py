from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_MEDIUMS = ("aquatic", "sediment", "soil")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Targeted medium-domain retention audit for key derived tables."
    )
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--medium", action="append", choices=DEFAULT_MEDIUMS)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--min-modeling-rows",
        type=int,
        default=100,
        help="Minimum rows for a dimension-specific table to be considered modeling-ready.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mediums = tuple(args.medium or DEFAULT_MEDIUMS)
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        prepared_tables: dict[str, tuple[str, set[str]]] = {}
        for table_name, where_clause in (
            ("target_records", ""),
            ("target_records", "target_status = 'included'"),
            ("task_records", ""),
            ("task_records", "target_status = 'included'"),
            ("task_records", "task_status = 'included'"),
            ("aggregated_task_records", ""),
        ):
            if not table_exists(conn, table_name):
                print(f"\n[{table_name} {where_clause or 'ALL'}] missing")
                continue
            if table_name not in prepared_tables:
                prepared_tables[table_name] = prepare_audit_table(conn, table_name, mediums)
            audit_table(
                conn,
                source_label=table_name,
                audit_table_name=prepared_tables[table_name][0],
                audit_columns=prepared_tables[table_name][1],
                where_clause=where_clause,
                top=args.top,
            )

        if table_exists(conn, "aggregated_task_records_medium_domain"):
            audit_subset_counts(conn, min_modeling_rows=args.min_modeling_rows)


def audit_table(
    conn: sqlite3.Connection,
    *,
    source_label: str,
    audit_table_name: str,
    audit_columns: set[str],
    where_clause: str,
    top: int,
) -> None:
    label = f"{source_label} {where_clause or 'ALL'}"
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    print(f"\n[{label}]")
    print_grouped(
        conn,
        audit_table_name,
        ("medium_domain",),
        where_sql=where_sql,
        top=100,
    )
    for group_columns in (
        ("medium_domain", "target_name"),
        ("medium_domain", "target_basis"),
        ("medium_domain", "unit_family_v2", "standard_unit_v2"),
        ("medium_domain", "target_status"),
        ("medium_domain", "excluded_reason"),
        ("medium_domain", "task_status"),
        ("medium_domain", "task_excluded_reason"),
        ("medium_domain", "task_head"),
    ):
        existing = tuple(column for column in group_columns if column in audit_columns)
        if len(existing) > 1:
            print_grouped(
                conn,
                audit_table_name,
                existing,
                where_sql=where_sql,
                top=top,
            )


def prepare_audit_table(
    conn: sqlite3.Connection, table_name: str, mediums: tuple[str, ...]
) -> tuple[str, set[str]]:
    source_columns = set(table_columns(conn, table_name))
    kept_columns = [
        column
        for column in (
            "target_name",
            "target_basis",
            "unit_family_v2",
            "standard_unit_v2",
            "target_status",
            "excluded_reason",
            "task_status",
            "task_excluded_reason",
            "task_head",
        )
        if column in source_columns
    ]
    medium_expr = medium_case_expression(source_columns)
    medium_placeholders = ", ".join("?" for _ in mediums)
    temp_name = f"tmp_medium_audit_{table_name}"
    select_columns = ", ".join(quote_identifier(column) for column in kept_columns)
    if select_columns:
        select_columns += ", "
    conn.execute(f"DROP TABLE IF EXISTS {quote_identifier(temp_name)}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {quote_identifier(temp_name)} AS
        SELECT {select_columns}{medium_expr} AS medium_domain
        FROM {quote_identifier(table_name)}
        WHERE {medium_expr} IN ({medium_placeholders})
        """,
        mediums,
    )
    conn.execute(f"CREATE INDEX idx_{temp_name}_medium ON {quote_identifier(temp_name)} (medium_domain)")
    audit_columns = set(kept_columns)
    audit_columns.add("medium_domain")
    return temp_name, audit_columns


def print_grouped(
    conn: sqlite3.Connection,
    table_name: str,
    group_columns: tuple[str, ...],
    *,
    where_sql: str,
    top: int,
) -> None:
    select_parts = []
    group_parts = []
    for column in group_columns:
        quoted = quote_identifier(column)
        select_parts.append(quoted)
        group_parts.append(quoted)
    select_sql = ", ".join(select_parts)
    group_sql = ", ".join(group_parts)
    query = f"""
        SELECT {select_sql}, COUNT(1) AS n
        FROM {quote_identifier(table_name)}
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY n DESC
        LIMIT ?
    """
    rows = conn.execute(query, (top,)).fetchall()
    print(f"by_{'_'.join(group_columns)}")
    print("\t".join((*group_columns, "n")))
    for row in rows:
        values = [format_value(row[column]) for column in group_columns]
        print("\t".join((*values, str(row["n"]))))


def audit_subset_counts(conn: sqlite3.Connection, *, min_modeling_rows: int) -> None:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name LIKE 'aggregated_task_records_%'
        ORDER BY name
        """
    ).fetchall()
    print("\n[subset_tables]")
    print("table\tn\tmodeling_status")
    for row in rows:
        table_name = row["name"]
        if table_name.endswith("_splits") or table_name == "aggregated_task_records_medium_domain":
            continue
        count = conn.execute(f"SELECT COUNT(1) FROM {quote_identifier(table_name)}").fetchone()[0]
        status = "modeling_ready" if count >= min_modeling_rows else f"too_few_rows_lt_{min_modeling_rows}"
        print(f"{table_name}\t{count}\t{status}")


def medium_case_expression(columns: set[str]) -> str:
    if "medium_domain" in columns:
        return quote_identifier("medium_domain")
    primary = normalized_text_sql("primary_medium", columns)
    habitat = normalized_text_sql("organism_habitat", columns)
    media = normalized_media_sql("media_type", columns)
    target_basis = normalized_text_sql("target_basis", columns)
    target_name = normalized_text_sql("target_name", columns)
    return f"""
    CASE
      WHEN {primary} = 'sediment'
        OR {media} IN ('SED', 'SEDIMENT')
        OR {target_basis} LIKE '%sediment%'
        OR {target_basis} LIKE 'mg/kg:sed%'
        THEN 'sediment'
      WHEN {primary} = 'aquatic'
        OR {habitat} = 'water'
        OR {media} IN ('FW', 'SW', 'AQU', 'CUL')
        THEN 'aquatic'
      WHEN {primary} = 'soil'
        OR {habitat} = 'soil'
        OR {media} IN ('NAT', 'ART', 'UKS', 'LIT', 'MIN', 'HYP')
        OR {target_basis} LIKE 'mg/kg:soil%'
        THEN 'soil'
      WHEN {primary} = 'terrestrial'
        OR {habitat} IN ('non-soil', 'nonsoil')
        THEN 'terrestrial_nonsoil'
      WHEN {target_name} IN ('neg_log10_mg_kg', 'neg-log10-mg-kg')
        THEN 'unknown'
      ELSE 'unknown'
    END
    """


def normalized_text_sql(column: str, columns: set[str]) -> str:
    if column not in columns:
        return "''"
    return f"LOWER(REPLACE(TRIM(COALESCE({quote_identifier(column)}, '')), '_', '-'))"


def normalized_media_sql(column: str, columns: set[str]) -> str:
    if column not in columns:
        return "''"
    return f"RTRIM(UPPER(TRIM(COALESCE({quote_identifier(column)}, ''))), '/*')"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})")]


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def format_value(value: object) -> str:
    if value is None:
        return "<null>"
    text = str(value).strip()
    return text if text else "<blank>"


if __name__ == "__main__":
    main()
