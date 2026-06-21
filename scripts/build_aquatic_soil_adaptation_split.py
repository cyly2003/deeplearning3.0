from __future__ import annotations

import argparse
import random
import sqlite3
from contextlib import closing
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aquatic train + soil finetune/test pTox adaptation split.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default="aggregated_task_records_aquatic_soil_ptox")
    parser.add_argument("--soil-source-table", default="aggregated_task_records_soil_ptox")
    parser.add_argument("--soil-split-name", default="SoilPtox_C_chemical_holdout_8_2")
    parser.add_argument("--split-name", default="M_aquatic_to_soil_ptox_adapt_C")
    parser.add_argument("--soil-only-split-name", default="")
    parser.add_argument("--soil-finetune-fraction", type=float, default=1.0)
    parser.add_argument("--aquatic-target-name", default="ptox_mol_l")
    parser.add_argument("--soil-target-name", default="ptox_mol_l")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    transfer_summary, soil_only_summary = build_adaptation_splits(
        db_path=args.db,
        source_table=args.source_table,
        soil_source_table=args.soil_source_table,
        soil_split_name=args.soil_split_name,
        split_name=args.split_name,
        soil_only_split_name=args.soil_only_split_name or None,
        soil_finetune_fraction=args.soil_finetune_fraction,
        aquatic_target_name=args.aquatic_target_name,
        soil_target_name=args.soil_target_name,
        seed=args.seed,
    )
    print({"transfer": transfer_summary, "soil_only": soil_only_summary})


def build_adaptation_splits(
    *,
    db_path: Path,
    source_table: str,
    soil_source_table: str,
    soil_split_name: str,
    split_name: str,
    soil_only_split_name: str | None = None,
    soil_finetune_fraction: float = 1.0,
    aquatic_target_name: str = "ptox_mol_l",
    soil_target_name: str = "ptox_mol_l",
    seed: int = 42,
) -> tuple[dict[str, int], dict[str, int] | None]:
    if soil_finetune_fraction <= 0 or soil_finetune_fraction > 1:
        raise ValueError("--soil-finetune-fraction must be in the range (0, 1].")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        ensure_table(conn, source_table)
        ensure_table(conn, soil_source_table)
        ensure_soil_split(conn, soil_split_name, soil_source_table)
        ensure_split_assignments_table(conn)
        conn.execute("DELETE FROM split_assignments WHERE split_name = ?", (split_name,))
        if soil_only_split_name:
            conn.execute("DELETE FROM split_assignments WHERE split_name = ?", (soil_only_split_name,))

        aquatic_rows = conn.execute(
            f"""
            SELECT aggregate_id
            FROM "{source_table}"
            WHERE medium_domain = 'aquatic'
              AND target_name = ?
            ORDER BY aggregate_id
            """,
            (aquatic_target_name,),
        ).fetchall()
        soil_split_rows = conn.execute(
            """
            SELECT aggregate_id, split_part
            FROM split_assignments
            WHERE split_name = ?
              AND source_table = ?
            """,
            (soil_split_name, soil_source_table),
        ).fetchall()
        soil_parts = {
            str(row["aggregate_id"]): row["split_part"]
            for row in soil_split_rows
        }
        soil_train_ids = [aggregate_id for aggregate_id, part in soil_parts.items() if part == "train"]
        selected_soil_train_ids = select_soil_train_ids(
            conn,
            soil_source_table=soil_source_table,
            soil_train_ids=soil_train_ids,
            fraction=soil_finetune_fraction,
            seed=seed,
        )
        selected_soil_train_set = set(selected_soil_train_ids)

        soil_rows = conn.execute(
            f"""
            SELECT aggregate_id
            FROM "{source_table}"
            WHERE medium_domain = 'soil'
              AND target_name = ?
            ORDER BY aggregate_id
            """,
            (soil_target_name,),
        ).fetchall()

        transfer_assignments: list[tuple[str, str | None, str, str, int, str, str, str]] = []
        for row in aquatic_rows:
            aggregate_id = str(row["aggregate_id"])
            transfer_assignments.append(
                (split_name, None, aggregate_id, "train", seed, "aquatic_soil_adaptation", source_table, "aquatic")
            )
        for row in soil_rows:
            aggregate_id = str(row["aggregate_id"])
            soil_part = soil_parts.get(aggregate_id)
            if soil_part == "train" and aggregate_id in selected_soil_train_set:
                split_part = "finetune"
            elif soil_part == "test":
                split_part = "test"
            else:
                continue
            transfer_assignments.append(
                (
                    split_name,
                    None,
                    aggregate_id,
                    split_part,
                    seed,
                    "aquatic_soil_adaptation",
                    source_table,
                    f"soil_{soil_part}",
                )
            )
        write_assignments(conn, transfer_assignments)

        soil_only_summary: dict[str, int] | None = None
        if soil_only_split_name:
            soil_only_assignments: list[tuple[str, str | None, str, str, int, str, str, str]] = []
            for aggregate_id, soil_part in sorted(soil_parts.items(), key=lambda item: natural_sort_key(item[0])):
                if soil_part == "train" and aggregate_id in selected_soil_train_set:
                    split_part = "train"
                elif soil_part == "test":
                    split_part = "test"
                else:
                    continue
                soil_only_assignments.append(
                    (
                        soil_only_split_name,
                        None,
                        aggregate_id,
                        split_part,
                        seed,
                        "soil_low_data_holdout",
                        soil_source_table,
                        f"soil_{soil_part}",
                    )
                )
            write_assignments(conn, soil_only_assignments)
            soil_only_summary = fetch_summary(conn, soil_only_split_name)

        conn.commit()
        transfer_summary = fetch_summary(conn, split_name)
    return transfer_summary, soil_only_summary


def select_soil_train_ids(
    conn: sqlite3.Connection,
    *,
    soil_source_table: str,
    soil_train_ids: list[str],
    fraction: float,
    seed: int,
) -> list[str]:
    if not soil_train_ids:
        raise ValueError("The source soil split has no train rows to sample for finetune.")
    if fraction >= 1:
        return sorted(soil_train_ids, key=natural_sort_key)

    columns = {
        row[1]
        for row in conn.execute(f'PRAGMA table_info("{soil_source_table}")')
    }
    if "cas_number" in columns:
        placeholders = ",".join("?" for _ in soil_train_ids)
        rows = conn.execute(
            f"""
            SELECT aggregate_id, cas_number
            FROM "{soil_source_table}"
            WHERE CAST(aggregate_id AS TEXT) IN ({placeholders})
            ORDER BY aggregate_id
            """,
            soil_train_ids,
        ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            key = str(row["cas_number"] or "<missing>")
            grouped.setdefault(key, []).append(str(row["aggregate_id"]))
        groups = sorted(grouped)
        random.Random(seed).shuffle(groups)
        target_n = max(1, int(round(len(soil_train_ids) * fraction)))
        selected: list[str] = []
        for group in groups:
            selected.extend(grouped[group])
            if len(selected) >= target_n:
                break
        return sorted(selected, key=natural_sort_key)

    shuffled = list(soil_train_ids)
    random.Random(seed).shuffle(shuffled)
    count = max(1, int(round(len(shuffled) * fraction)))
    return sorted(shuffled[:count], key=natural_sort_key)


def natural_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


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


def write_assignments(conn: sqlite3.Connection, assignments: list[tuple[str, str | None, str, str, int, str, str, str]]) -> None:
    if not assignments:
        raise ValueError("No split assignments to write.")
    conn.executemany(
        """
        INSERT INTO split_assignments (
            split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        assignments,
    )


def fetch_summary(conn: sqlite3.Connection, split_name: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT split_part, COUNT(1)
        FROM split_assignments
        WHERE split_name = ?
        GROUP BY split_part
        ORDER BY split_part
        """,
        (split_name,),
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Table not found: {table_name}")


def ensure_soil_split(conn: sqlite3.Connection, split_name: str, source_table: str) -> None:
    count = conn.execute(
        """
        SELECT COUNT(1)
        FROM split_assignments
        WHERE split_name = ?
          AND source_table = ?
        """,
        (split_name, source_table),
    ).fetchone()[0]
    if int(count) == 0:
        raise ValueError(f"Soil split not found: split_name={split_name}, source_table={source_table}")


if __name__ == "__main__":
    main()
