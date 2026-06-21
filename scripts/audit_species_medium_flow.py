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


DEFAULT_TERMS = (
    "earthworm",
    "eisenia",
    "lumbricus",
    "lumbricidae",
    "andrei",
    "fetida",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit medium assignment for selected species terms.")
    parser.add_argument("--db", required=True, type=Path, help="Derived modeling SQLite database")
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS), help="Comma-separated terms matched in common/Latin name")
    parser.add_argument("--limit", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    terms = tuple(term.strip().lower() for term in args.terms.split(",") if term.strip())
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        for table_name, status_column in (
            ("target_records", "target_status"),
            ("task_records", "task_status"),
            ("aggregated_task_records", ""),
        ):
            audit_table(conn, table_name, terms=terms, status_column=status_column, limit=args.limit)


def audit_table(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    terms: tuple[str, ...],
    status_column: str,
    limit: int,
) -> None:
    if not table_exists(conn, table_name):
        print(f"\n[{table_name}] missing")
        return

    domain_counts: Counter[str] = Counter()
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    medium_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    species_counts: Counter[tuple[str, str, str]] = Counter()

    for row in conn.execute(f'SELECT * FROM "{table_name}"'):
        if not matches_terms(row, terms):
            continue
        domain, reason, conflict = classify_medium_domain(row)
        domain_counts[domain] += 1
        status = str(row[status_column]) if status_column and status_column in row.keys() else "<all>"
        status_counts[domain][status] += 1
        task = str(row["task_head"]) if "task_head" in row.keys() else "<missing>"
        task_counts[domain][task] += 1
        medium_counts[
            (
                domain,
                str(row["primary_medium"] if "primary_medium" in row.keys() else ""),
                str(row["organism_habitat"] if "organism_habitat" in row.keys() else ""),
                str(row["media_type"] if "media_type" in row.keys() else ""),
                reason + (";conflict" if conflict else ""),
            )
        ] += 1
        species_counts[
            (
                domain,
                str(row["latin_name"] if "latin_name" in row.keys() else ""),
                str(row["common_name"] if "common_name" in row.keys() else ""),
            )
        ] += 1

    print(f"\n[{table_name}] terms={','.join(terms)}")
    print("domain_counts")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count}")
    print("status_by_domain")
    for domain in sorted(status_counts):
        top = ", ".join(f"{name}:{count}" for name, count in status_counts[domain].most_common(10))
        print(f"  {domain}: {top}")
    print("task_by_domain")
    for domain in sorted(task_counts):
        top = ", ".join(f"{name}:{count}" for name, count in task_counts[domain].most_common(10))
        print(f"  {domain}: {top}")
    print("top_medium_combos")
    for (domain, primary, habitat, media, reason), count in medium_counts.most_common(limit):
        print(f"  {count} | {domain} | primary={primary} habitat={habitat} media={media} | {reason}")
    print("top_species")
    for (domain, latin, common), count in species_counts.most_common(limit):
        print(f"  {count} | {domain} | {latin} | {common}")


def matches_terms(row: sqlite3.Row, terms: tuple[str, ...]) -> bool:
    text = " ".join(
        str(row[column]).lower()
        for column in ("latin_name", "common_name")
        if column in row.keys() and row[column] is not None
    )
    return any(term in text for term in terms)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


if __name__ == "__main__":
    main()

