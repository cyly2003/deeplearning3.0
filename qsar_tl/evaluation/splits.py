from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SEED = 42
DEFAULT_RANDOM_FRACTIONS = (0.8, 0.1, 0.1)
DEFAULT_TABLE_CANDIDATES = ("aggregated_task_records", "target_records")


@dataclass(frozen=True)
class SplitAssignment:
    split_name: str
    split_part: str
    seed: int
    split_type: str
    source_table: str
    record_id: str | None = None
    aggregate_id: str | None = None
    group_key: str | None = None


@dataclass(frozen=True)
class MediumTransferRules:
    train_domains: tuple[str, ...] = ("water", "non-soil", "non_soil", "nonsoil")
    test_domains: tuple[str, ...] = ("soil",)
    unknown_part: str = "valid"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")')]


def resolve_source_table(conn: sqlite3.Connection, table_name: str | None = None) -> str:
    if table_name:
        if not table_exists(conn, table_name):
            raise ValueError(f"Source table does not exist: {table_name}")
        return table_name
    for candidate in DEFAULT_TABLE_CANDIDATES:
        if table_exists(conn, candidate):
            return candidate
    raise ValueError(
        "No modeling source table found. Expected aggregated_task_records or target_records."
    )


def resolve_id_column(columns: Iterable[str], preferred: str | None = None) -> str:
    available = set(columns)
    if preferred:
        if preferred not in available:
            raise ValueError(f"ID column does not exist: {preferred}")
        return preferred
    for candidate in ("aggregate_id", "record_id", "result_id", "test_id"):
        if candidate in available:
            return candidate
    raise ValueError("No usable ID column found. Expected aggregate_id, record_id, result_id, or test_id.")


def read_split_records(
    conn: sqlite3.Connection,
    *,
    table_name: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    source_table = resolve_source_table(conn, table_name)
    columns = table_columns(conn, source_table)
    resolved_id = resolve_id_column(columns, id_column)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    rows = conn.execute(f'SELECT * FROM "{source_table}" ORDER BY "{resolved_id}"{limit_clause}').fetchall()
    return source_table, resolved_id, [dict(row) for row in rows]


def create_split_assignments_table(conn: sqlite3.Connection) -> None:
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
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_split_assignments_name_part
        ON split_assignments (split_name, split_part)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_split_assignments_record
        ON split_assignments (split_name, record_id, aggregate_id)
        """
    )


def write_split_assignments(conn: sqlite3.Connection, assignments: list[SplitAssignment]) -> None:
    if not assignments:
        raise ValueError("No split assignments to write.")
    create_split_assignments_table(conn)
    split_name = assignments[0].split_name
    conn.execute("DELETE FROM split_assignments WHERE split_name = ?", (split_name,))
    conn.executemany(
        """
        INSERT INTO split_assignments (
            split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item.split_name,
                item.record_id,
                item.aggregate_id,
                item.split_part,
                item.seed,
                item.split_type,
                item.source_table,
                item.group_key,
            )
            for item in assignments
        ],
    )


def generate_and_write_split(
    db_path: str | Path,
    *,
    split_name: str,
    split_type: str,
    seed: int = DEFAULT_SEED,
    table_name: str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
    random_fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
    medium_rules: MediumTransferRules | None = None,
) -> dict[str, int]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        source_table, resolved_id, records = read_split_records(
            conn,
            table_name=table_name,
            id_column=id_column,
            limit=limit,
        )
        assignments = build_split_assignments(
            records,
            split_name=split_name,
            split_type=split_type,
            source_table=source_table,
            id_column=resolved_id,
            seed=seed,
            random_fractions=random_fractions,
            medium_rules=medium_rules,
        )
        write_split_assignments(conn, assignments)
        conn.commit()
    return summarize_assignments(assignments)


def build_split_assignments(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    split_type: str,
    source_table: str,
    id_column: str,
    seed: int = DEFAULT_SEED,
    random_fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
    medium_rules: MediumTransferRules | None = None,
) -> list[SplitAssignment]:
    if not records:
        raise ValueError("At least one record is required to build a split.")
    normalized_type = split_type.strip().lower()
    if normalized_type == "random_split":
        parts = assign_random_parts(len(records), seed=seed, fractions=random_fractions)
        group_keys = [None] * len(records)
    elif normalized_type == "chemical_group_split":
        parts, group_keys = assign_group_parts(records, ("cas_number",), seed=seed)
    elif normalized_type == "species_group_split":
        parts, group_keys = assign_group_parts(records, ("species_number",), seed=seed)
    elif normalized_type == "chemical_species_group_split":
        parts, group_keys = assign_group_parts(records, ("cas_number", "species_number"), seed=seed)
    elif normalized_type == "medium_transfer_split":
        parts, group_keys = assign_medium_transfer_parts(records, rules=medium_rules or MediumTransferRules())
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")

    assignments: list[SplitAssignment] = []
    for record, part, group_key in zip(records, parts, group_keys):
        record_value = str(record.get(id_column)) if id_column != "aggregate_id" else None
        aggregate_value = str(record.get(id_column)) if id_column == "aggregate_id" else None
        assignments.append(
            SplitAssignment(
                split_name=split_name,
                record_id=record_value,
                aggregate_id=aggregate_value,
                split_part=part,
                seed=seed,
                split_type=normalized_type,
                source_table=source_table,
                group_key=group_key,
            )
        )
    return assignments


def assign_random_parts(
    n_records: int,
    *,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
) -> list[str]:
    train_fraction, valid_fraction, test_fraction = fractions
    if min(fractions) < 0 or abs(train_fraction + valid_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Random split fractions must be non-negative and sum to 1.0.")
    indices = list(range(n_records))
    random.Random(seed).shuffle(indices)
    n_train = int(n_records * train_fraction)
    n_valid = int(n_records * valid_fraction)
    parts = ["test"] * n_records
    for idx in indices[:n_train]:
        parts[idx] = "train"
    for idx in indices[n_train : n_train + n_valid]:
        parts[idx] = "valid"
    return parts


def assign_group_parts(
    records: list[dict[str, Any]],
    group_columns: tuple[str, ...],
    *,
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = DEFAULT_RANDOM_FRACTIONS,
) -> tuple[list[str], list[str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = make_group_key(record, group_columns)
        grouped_indices[key].append(idx)

    groups = list(grouped_indices)
    group_parts = assign_random_parts(len(groups), seed=seed, fractions=fractions)
    parts = ["test"] * len(records)
    group_keys = [""] * len(records)
    for group_key, group_part in zip(groups, group_parts):
        for idx in grouped_indices[group_key]:
            parts[idx] = group_part
            group_keys[idx] = group_key
    return parts, group_keys


def make_group_key(record: dict[str, Any], group_columns: tuple[str, ...]) -> str:
    values = []
    for column in group_columns:
        value = record.get(column)
        values.append("<missing>" if value is None or str(value).strip() == "" else str(value).strip())
    return "||".join(values)


def assign_medium_transfer_parts(
    records: list[dict[str, Any]],
    *,
    rules: MediumTransferRules,
) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    group_keys: list[str] = []
    for record in records:
        medium_text = normalize_medium_text(record)
        domain = classify_medium_domain(medium_text, rules)
        if domain == "test":
            part = "test"
        elif domain == "train":
            part = "train"
        else:
            part = rules.unknown_part
        if part not in {"train", "valid", "test", "excluded"}:
            raise ValueError("unknown_part must be one of train, valid, test, or excluded.")
        parts.append(part)
        group_keys.append(medium_text or "<missing>")
    return parts, group_keys


def normalize_medium_text(record: dict[str, Any]) -> str:
    candidates = (
        record.get("domain"),
        record.get("medium_domain"),
        record.get("primary_medium"),
        record.get("media_type"),
        record.get("organism_habitat"),
        record.get("target_basis"),
    )
    return " ".join(str(value).strip().lower() for value in candidates if value is not None and str(value).strip())


def classify_medium_domain(medium_text: str, rules: MediumTransferRules) -> str:
    if not medium_text:
        return "unknown"
    if any(term in medium_text for term in rules.test_domains):
        return "test"
    if any(term in medium_text for term in rules.train_domains):
        return "train"
    if "water" in medium_text or "aquatic" in medium_text:
        return "train"
    if "soil" in medium_text or "sediment" in medium_text:
        return "test"
    return "unknown"


def summarize_assignments(assignments: list[SplitAssignment]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for assignment in assignments:
        summary[assignment.split_part] = summary.get(assignment.split_part, 0) + 1
    return dict(sorted(summary.items()))
