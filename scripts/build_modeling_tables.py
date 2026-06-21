from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.data.modeling_tables import build_modeling_tables, summarize_target_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build wide_records and target_records tables.")
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--min-dose-groups-for-midpoint",
        type=int,
        default=3,
        help="Deprecated compatibility option; min/max midpoint no longer requires a dose-group threshold.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = build_modeling_tables(
        source_db=args.source_db,
        output_db=args.output_db,
        limit=args.limit,
        batch_size=args.batch_size,
        min_dose_groups_for_midpoint=args.min_dose_groups_for_midpoint,
    )
    print(f"Wrote modeling tables: {args.output_db.resolve()}")
    print(stats)
    for row in summarize_target_table(args.output_db):
        print(row)


if __name__ == "__main__":
    main()
