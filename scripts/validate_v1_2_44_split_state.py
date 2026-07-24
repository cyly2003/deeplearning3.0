from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_v1_2_44_second_layer_splits import (
    EXPECTED_PARENT_ASSIGNMENT_SHA256,
    EXPECTED_PARENT_COUNTS,
    PARENT_SPLIT,
    ROUTE_SPLITS,
    SOURCE_TABLE,
    audit_parent_assignments,
    audit_route_assignments,
    read_assignments,
)
from scripts.validate_v1_2_44_matrix_run import validate_split_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quickly validate persisted v1.2.44 split assignments and signed summary."
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--split-summary", required=True, type=Path)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    validate_split_summary(summary)
    if args.source_table != SOURCE_TABLE:
        raise ValueError(f"Unexpected source table: {args.source_table}")
    with closing(sqlite3.connect(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        parent = audit_parent_assignments(
            read_assignments(conn, PARENT_SPLIT, SOURCE_TABLE),
            expected_counts=EXPECTED_PARENT_COUNTS,
            expected_sha256=EXPECTED_PARENT_ASSIGNMENT_SHA256,
        )
        if parent != summary.get("parent"):
            raise ValueError("Persisted v1.2.40 parent assignments differ from signed summary.")
        routes = {}
        for route, split_name in ROUTE_SPLITS.items():
            audit = audit_route_assignments(
                read_assignments(conn, split_name, SOURCE_TABLE), route=route
            )
            observed = {"split_name": split_name, **audit}
            if observed != summary.get("routes", {}).get(route):
                raise ValueError(f"Persisted {route} assignments differ from signed summary.")
            routes[route] = audit["assignment_sha256"]
    print(
        json.dumps(
            {
                "status": "ok",
                "matrix_version": "v1.2.44",
                "contract_sha256": summary["contract_sha256"],
                "parent_assignment_sha256": parent["assignment_sha256"],
                "route_assignment_sha256": routes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
