from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def top_counts(conn: sqlite3.Connection, table: str, column: str, limit: int = 20) -> list[tuple[object, int]]:
    table_q = quote_identifier(table)
    column_q = quote_identifier(column)
    return conn.execute(
        f"""
        SELECT {column_q}, COUNT(*) AS n
        FROM {table_q}
        GROUP BY {column_q}
        ORDER BY n DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def write_section(lines: list[str], title: str, rows: list[tuple[object, int]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Value | Count |")
    lines.append("|---|---:|")
    for value, count in rows:
        label = "<NULL>" if value is None else str(value)
        label = label.replace("|", "\\|")
        lines.append(f"| `{label}` | {count} |")
    lines.append("")


def profile(sqlite_db: Path) -> str:
    sections = [
        ("tests", "media_type", "Tests: media_type"),
        ("tests", "organism_habitat", "Tests: organism_habitat"),
        ("tests", "organism_lifestage", "Tests: organism_lifestage"),
        ("tests", "exposure_duration_standardization_status", "Tests: duration standardization"),
        ("results", "endpoint", "Results: endpoint"),
        ("results", "effect", "Results: effect"),
        ("results", "measurement", "Results: measurement"),
        ("results", "conc1_standard_unit", "Results: standardized concentration unit"),
        ("results", "conc1_unit_family", "Results: concentration unit family"),
        ("results", "conc1_standardization_status", "Results: concentration standardization"),
        ("species_category_curated", "taxon_group_l1", "Species: taxon_group_l1"),
        ("species_category_curated", "taxon_group_l2", "Species: taxon_group_l2"),
        ("chemical_category_curated", "chemical_class_l1", "Chemicals: chemical_class_l1"),
    ]

    lines = ["# ECOTOX Clean Database Profile", "", f"Database: `{sqlite_db}`", ""]
    with sqlite3.connect(sqlite_db) as conn:
        for table, column, title in sections:
            write_section(lines, title, top_counts(conn, table, column))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile key categorical fields in ecotox_clean.sqlite.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    text = profile(args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote database profile: {args.out.resolve()}")


if __name__ == "__main__":
    main()

