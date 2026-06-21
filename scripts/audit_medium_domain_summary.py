from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_medium_domain_tables import DOMAIN_TABLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast SQL summary for medium-domain annotated task tables.")
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--table", default="aggregated_task_records_medium_domain")
    parser.add_argument("--top", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        if not table_exists(conn, args.table):
            raise ValueError(f"Table not found: {args.table}. Run scripts/build_medium_domain_tables.py first.")

        print(f"# medium domain summary: {args.table}")
        print_count_table(conn, args.table)
        for group_columns in (
            ("medium_domain", "target_name"),
            ("medium_domain", "target_basis"),
            ("medium_domain", "unit_family_v2", "standard_unit_v2"),
            ("medium_domain", "unit_conversion_source", "unit_conversion_confidence"),
            ("medium_domain", "medium_domain_reason"),
            ("medium_domain", "task_head"),
            ("medium_domain", "medium_conflict_flag"),
        ):
            print_grouped(conn, args.table, group_columns, top=args.top)

        print("\n## subset_tables")
        for label, table_name in DOMAIN_TABLES.items():
            count = table_count(conn, table_name)
            print(f"{label}\t{table_name}\t{count}")


def print_count_table(conn: sqlite3.Connection, table_name: str) -> None:
    total = table_count(conn, table_name)
    print(f"\n## total\n{total}")
    print_grouped(conn, table_name, ("medium_domain",), top=100)


def print_grouped(
    conn: sqlite3.Connection,
    table_name: str,
    group_columns: tuple[str, ...],
    *,
    top: int,
) -> None:
    existing = set(table_columns(conn, table_name))
    columns = tuple(column for column in group_columns if column in existing)
    if not columns:
        return
    select_sql = ", ".join(quote_identifier(column) for column in columns)
    group_sql = ", ".join(quote_identifier(column) for column in columns)
    rows = conn.execute(
        f"""
        SELECT {select_sql}, COUNT(1) AS n
        FROM {quote_identifier(table_name)}
        GROUP BY {group_sql}
        ORDER BY n DESC
        LIMIT ?
        """,
        (top,),
    ).fetchall()
    print(f"\n## by_{'_'.join(columns)}")
    print("\t".join((*columns, "n")))
    for row in rows:
        values = [format_value(row[column]) for column in columns]
        print("\t".join((*values, str(row["n"]))))


def table_count(conn: sqlite3.Connection, table_name: str) -> int:
    if not table_exists(conn, table_name):
        return 0
    row = conn.execute(f"SELECT COUNT(1) AS n FROM {quote_identifier(table_name)}").fetchone()
    return int(row["n"])


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
