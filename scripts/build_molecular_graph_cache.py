from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from qsar_tl.features.molecular_graph import write_molecular_graph_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reserved molecular graph JSONL cache from SQLite SMILES.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--source-table", default="aggregated_task_records")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit_clause = "" if args.limit is None else f" LIMIT {int(args.limit)}"
    with sqlite3.connect(args.db) as conn:
        rows = conn.execute(
            f"""
            SELECT smiles, COUNT(*) AS n
            FROM "{args.source_table}"
            WHERE smiles IS NOT NULL
              AND TRIM(CAST(smiles AS TEXT)) <> ''
            GROUP BY smiles
            ORDER BY n DESC, smiles
            {limit_clause}
            """
        ).fetchall()
    manifest = write_molecular_graph_cache((row[0] for row in rows), args.out)
    print(manifest)


if __name__ == "__main__":
    main()
