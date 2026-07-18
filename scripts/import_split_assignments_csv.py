from __future__ import annotations

import argparse
import csv
import sqlite3
from contextlib import closing
from pathlib import Path


FIELDS = (
    "split_name",
    "record_id",
    "aggregate_id",
    "split_part",
    "seed",
    "split_type",
    "source_table",
    "group_key",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import split_assignments rows from a CSV export.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = import_split_assignments(args.db, args.csv)
    for row in summary:
        print(row)


def import_split_assignments(db_path: Path, csv_path: Path) -> list[tuple[str, str, int]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")
    missing = sorted(set(FIELDS) - set(rows[0]))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    split_names = sorted({row["split_name"] for row in rows})
    with closing(sqlite3.connect(db_path)) as conn:
        ensure_split_assignments_table(conn)
        conn.executemany("DELETE FROM split_assignments WHERE split_name = ?", [(name,) for name in split_names])
        conn.executemany(
            """
            INSERT INTO split_assignments (
                split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(row[field] for field in FIELDS) for row in rows],
        )
        conn.commit()
        placeholders = ",".join("?" for _ in split_names)
        return [
            (str(split_name), str(split_part), int(count))
            for split_name, split_part, count in conn.execute(
                f"""
                SELECT split_name, split_part, COUNT(1)
                FROM split_assignments
                WHERE split_name IN ({placeholders})
                GROUP BY split_name, split_part
                ORDER BY split_name, split_part
                """,
                split_names,
            )
        ]


def ensure_split_assignments_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS split_assignments (
            split_name TEXT NOT NULL,
            record_id TEXT,
            aggregate_id TEXT,
            split_part TEXT NOT NULL,
            seed INTEGER NOT NULL,
            split_type TEXT NOT NULL,
            source_table TEXT NOT NULL,
            group_key TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


if __name__ == "__main__":
    main()
