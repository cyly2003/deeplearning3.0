from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_medium_domain_tables import classify_medium_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit medium-domain sample retention across derived tables.")
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick checks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        for table_name, where_clause in (
            ("target_records", ""),
            ("target_records", "WHERE target_status = 'included'"),
            ("task_records", ""),
            ("task_records", "WHERE task_status = 'included'"),
            ("aggregated_task_records", ""),
        ):
            audit_table(conn, table_name, where_clause=where_clause, limit=args.limit)


def audit_table(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    where_clause: str,
    limit: int | None,
) -> None:
    if not table_exists(conn, table_name):
        print(f"\n[{table_name} {where_clause or 'ALL'}] missing")
        return

    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    domain_counts: Counter[str] = Counter()
    target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exclusion_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[tuple[str, str]] = Counter()
    conflict_counts: Counter[str] = Counter()

    rows = conn.execute(f'SELECT * FROM "{table_name}" {where_clause} {limit_clause}')
    for row in rows:
        domain, reason, conflict = classify_medium_domain(row)
        domain_counts[domain] += 1
        target_counts[domain][str(row["target_name"] if "target_name" in row.keys() else "<missing>")] += 1
        task_counts[domain][str(row["task_head"] if "task_head" in row.keys() else "<missing>")] += 1
        if "target_status" in row.keys():
            status_counts[domain][f"target:{row['target_status']}"] += 1
        if "task_status" in row.keys():
            status_counts[domain][f"task:{row['task_status']}"] += 1
        if "excluded_reason" in row.keys():
            exclusion_counts[domain][f"target:{row['excluded_reason']}"] += 1
        if "task_excluded_reason" in row.keys():
            exclusion_counts[domain][f"task:{row['task_excluded_reason']}"] += 1
        unit = str(
            row["unit_family_v2"]
            if "unit_family_v2" in row.keys() and row["unit_family_v2"] is not None
            else row["conc1_unit_family"]
            if "conc1_unit_family" in row.keys()
            else "<missing>"
        )
        standard_unit = str(
            row["standard_unit_v2"]
            if "standard_unit_v2" in row.keys() and row["standard_unit_v2"] is not None
            else row["conc1_standard_unit"]
            if "conc1_standard_unit" in row.keys()
            else ""
        )
        unit_counts[domain][f"{unit}|{standard_unit}"] += 1
        reason_counts[(domain, reason)] += 1
        if conflict:
            conflict_counts[domain] += 1

    label = f"{table_name} {where_clause or 'ALL'}".strip()
    print(f"\n[{label}]")
    print("domain_counts")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} conflicts={conflict_counts.get(domain, 0)}")
    print("target_by_domain")
    for domain in sorted(target_counts):
        top = ", ".join(f"{name}:{count}" for name, count in target_counts[domain].most_common(8))
        print(f"  {domain}: {top}")
    print("task_by_domain")
    for domain in sorted(task_counts):
        top = ", ".join(f"{name}:{count}" for name, count in task_counts[domain].most_common(8))
        print(f"  {domain}: {top}")
    print("status_by_domain")
    for domain in sorted(status_counts):
        top = ", ".join(f"{name}:{count}" for name, count in status_counts[domain].most_common(8))
        print(f"  {domain}: {top}")
    print("exclusions_by_domain")
    for domain in sorted(exclusion_counts):
        top = ", ".join(f"{name}:{count}" for name, count in exclusion_counts[domain].most_common(8))
        print(f"  {domain}: {top}")
    print("units_by_domain")
    for domain in sorted(unit_counts):
        top = ", ".join(f"{name}:{count}" for name, count in unit_counts[domain].most_common(8))
        print(f"  {domain}: {top}")
    print("top_reasons")
    for (domain, reason), count in reason_counts.most_common(12):
        print(f"  {domain} | {reason}: {count}")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


if __name__ == "__main__":
    main()
