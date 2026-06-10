from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.data.task_tables import build_task_tables, summarize_task_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build task_records and aggregated_task_records tables.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = build_task_tables(args.db, limit=args.limit, batch_size=args.batch_size)
    print(f"Wrote task tables: {args.db.resolve()}")
    print(stats)
    for row in summarize_task_table(args.db):
        print(row)


if __name__ == "__main__":
    main()
