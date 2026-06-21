from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast SQL audit for medium-domain retention across derived tables.")
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--top", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        for table_name, where_clause in (
            ("target_records", ""),
            ("target_records", "target_status = 'included'"),
            ("task_records", ""),
            ("task_records", "task_status = 'included'"),
            ("aggregated_task_records", ""),
        ):
            if not table_exists(conn, table_name):
                print(f"\n[{table_name} {where_clause or 'ALL'}] missing")
                continue
            audit_table(conn, table_name, where_clause=where_clause, top=args.top)


def audit_table(conn: sqlite3.Connection, table_name: str, *, where_clause: str, top: int) -> None:
    columns = set(table_columns(conn, table_name))
    medium_expr = medium_case_expression(columns)
    label = f"{table_name} {where_clause or 'ALL'}"
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    print(f"\n[{label}]")
    print_grouped(conn, table_name, medium_expr, ("medium_domain",), where_sql=where_sql, top=100)
    for group_columns in (
        ("medium_domain", "target_name"),
        ("medium_domain", "target_basis"),
        ("medium_domain", "unit_family_v2", "standard_unit_v2"),
        ("medium_domain", "target_status"),
        ("medium_domain", "task_status"),
        ("medium_domain", "excluded_reason"),
        ("medium_domain", "task_excluded_reason"),
        ("medium_domain", "task_head"),
    ):
        existing = tuple(column for column in group_columns if column == "medium_domain" or column in columns)
        if len(existing) > 1:
            print_grouped(conn, table_name, medium_expr, existing, where_sql=where_sql, top=top)


def print_grouped(
    conn: sqlite3.Connection,
    table_name: str,
    medium_expr: str,
    group_columns: tuple[str, ...],
    *,
    where_sql: str,
    top: int,
) -> None:
    select_parts = []
    group_parts = []
    for column in group_columns:
        if column == "medium_domain":
            select_parts.append(f"{medium_expr} AS medium_domain")
            group_parts.append("medium_domain")
        else:
            select_parts.append(quote_identifier(column))
            group_parts.append(quote_identifier(column))
    select_sql = ", ".join(select_parts)
    group_sql = ", ".join(group_parts)
    rows = conn.execute(
        f"""
        SELECT {select_sql}, COUNT(1) AS n
        FROM {quote_identifier(table_name)}
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY n DESC
        LIMIT ?
        """,
        (top,),
    ).fetchall()
    print(f"by_{'_'.join(group_columns)}")
    print("\t".join((*group_columns, "n")))
    for row in rows:
        values = [format_value(row[column]) for column in group_columns]
        print("\t".join((*values, str(row["n"]))))


def medium_case_expression(columns: set[str]) -> str:
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
