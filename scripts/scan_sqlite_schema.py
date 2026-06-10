from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def scan_schema(sqlite_db: Path) -> str:
    with sqlite3.connect(sqlite_db) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = ? ORDER BY name",
                ("table",),
            )
        ]

        lines: list[str] = []
        lines.append("# SQLite Schema Scan")
        lines.append("")
        lines.append(f"Database: `{sqlite_db}`")
        lines.append(f"Table count: {len(tables)}")
        lines.append("")

        for table in tables:
            quoted = quote_identifier(table)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({quoted})").fetchall()

            lines.append(f"## {table}")
            lines.append("")
            lines.append(f"- Rows: {row_count}")
            lines.append(f"- Columns: {len(columns)}")
            lines.append("")
            lines.append("| Column | Type | Not Null | Primary Key | Default |")
            lines.append("|---|---:|---:|---:|---|")
            for _, name, col_type, notnull, default_value, pk in columns:
                default_text = "" if default_value is None else str(default_value)
                lines.append(
                    f"| `{name}` | `{col_type}` | {notnull} | {pk} | `{default_text}` |"
                )
            lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan SQLite schema into Markdown.")
    parser.add_argument("--db", required=True, type=Path, help="SQLite database path")
    parser.add_argument("--out", required=True, type=Path, help="Markdown output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    markdown = scan_schema(args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(f"Wrote schema report: {args.out.resolve()}")


if __name__ == "__main__":
    main()

