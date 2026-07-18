from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path


STAGE_SAMPLE_ID_PREFIX = "stage_sample_v1:"
STAGE_CONTRACT_PREFIX = "stage_contract_v1"
PTOX_TARGET_FAMILY = "aquatic_pTox_mol_L"
MGKG_TARGET_FAMILY = "solid_neglog_mg_kg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict aquatic pTox -> soil pTox -> soil mg/kg split assignments. "
            "The pTox and mg/kg targets remain separate output-head families."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", default="aggregated_task_records_medium_domain_qc_expanded")
    parser.add_argument("--soil-ptox-source-table", default="aggregated_task_records_soil_ptox_qc")
    parser.add_argument("--soil-ptox-split-name", default="SoilPtoxQC2_B_random_8_2")
    parser.add_argument("--soil-mgkg-source-table", default="aggregated_task_records_soil_mg_kg_qc")
    parser.add_argument("--soil-mgkg-split-name", default="SoilMgkgQC2_B_random_8_2")
    parser.add_argument("--split-name", default="M_v1_2_39_ptox_to_soil_mgkg_B_random_8_2")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_three_stage_split(
        db_path=args.db,
        source_table=args.source_table,
        soil_ptox_source_table=args.soil_ptox_source_table,
        soil_ptox_split_name=args.soil_ptox_split_name,
        soil_mgkg_source_table=args.soil_mgkg_source_table,
        soil_mgkg_split_name=args.soil_mgkg_split_name,
        split_name=args.split_name,
        seed=args.seed,
    )
    print(summary)


def build_three_stage_split(
    *,
    db_path: Path,
    source_table: str,
    soil_ptox_source_table: str,
    soil_ptox_split_name: str,
    soil_mgkg_source_table: str,
    soil_mgkg_split_name: str,
    split_name: str,
    seed: int = 42,
) -> dict[str, int]:
    """Create one split where each stage has a non-overlapping training role.

    Stage 1 contains aquatic pTox only. Stage 2 contains only the training part
    of the existing soil pTox split. Stage 3 contains only the training part of
    the existing soil mg/kg split; its test rows are the sole external test set.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for table_name in (source_table, soil_ptox_source_table, soil_mgkg_source_table):
            ensure_table(conn, table_name)
        ensure_split_assignments_table(conn)
        ensure_split(conn, soil_ptox_split_name, soil_ptox_source_table)
        ensure_split(conn, soil_mgkg_split_name, soil_mgkg_source_table)

        aquatic_ptox_rows = source_rows(
            conn,
            source_table,
            medium_domain="aquatic",
            target_name="ptox_mol_l",
            target_family=PTOX_TARGET_FAMILY,
        )
        soil_ptox_rows = source_rows(
            conn,
            source_table,
            medium_domain="soil",
            target_name="ptox_mol_l",
            target_family=PTOX_TARGET_FAMILY,
        )
        soil_mgkg_rows = source_rows(
            conn,
            source_table,
            medium_domain="soil",
            target_name="neg_log10_mg_kg",
            target_family=MGKG_TARGET_FAMILY,
        )
        if not aquatic_ptox_rows or not soil_ptox_rows or not soil_mgkg_rows:
            raise ValueError("The three-stage source table is missing one or more required target domains.")

        soil_ptox_parts = split_parts(conn, soil_ptox_split_name, soil_ptox_source_table)
        soil_mgkg_parts = split_parts(conn, soil_mgkg_split_name, soil_mgkg_source_table)
        assert_source_coverage("soil pTox", row_aggregate_ids(soil_ptox_rows), soil_ptox_parts)
        assert_source_coverage("soil mg/kg", row_aggregate_ids(soil_mgkg_rows), soil_mgkg_parts)

        assignments: list[tuple[str, str, str, str, int, str, str, str]] = []
        assignments.extend(
            (
                split_name,
                stage_sample_record_id(row),
                str(row["aggregate_id"]),
                "train",
                seed,
                "strict_three_stage_ptox_to_soil_mgkg",
                source_table,
                stage_group_key(
                    stage="aquatic_ptox_stage1",
                    medium_domain="aquatic",
                    target_name="ptox_mol_l",
                    target_family=PTOX_TARGET_FAMILY,
                ),
            )
            for row in aquatic_ptox_rows
        )
        assignments.extend(
            (
                split_name,
                stage_sample_record_id(row),
                str(row["aggregate_id"]),
                "finetune",
                seed,
                "strict_three_stage_ptox_to_soil_mgkg",
                source_table,
                stage_group_key(
                    stage="soil_ptox_stage2_train",
                    medium_domain="soil",
                    target_name="ptox_mol_l",
                    target_family=PTOX_TARGET_FAMILY,
                ),
            )
            for row in soil_ptox_rows
            if soil_ptox_parts.get(str(row["aggregate_id"])) == "train"
        )
        assignments.extend(
            (
                split_name,
                stage_sample_record_id(row),
                str(row["aggregate_id"]),
                "finetune_mgkg",
                seed,
                "strict_three_stage_ptox_to_soil_mgkg",
                source_table,
                stage_group_key(
                    stage="soil_mgkg_stage3_train",
                    medium_domain="soil",
                    target_name="neg_log10_mg_kg",
                    target_family=MGKG_TARGET_FAMILY,
                ),
            )
            for row in soil_mgkg_rows
            if soil_mgkg_parts.get(str(row["aggregate_id"])) == "train"
        )
        assignments.extend(
            (
                split_name,
                stage_sample_record_id(row),
                str(row["aggregate_id"]),
                "test",
                seed,
                "strict_three_stage_ptox_to_soil_mgkg",
                source_table,
                stage_group_key(
                    stage="soil_mgkg_stage3_test",
                    medium_domain="soil",
                    target_name="neg_log10_mg_kg",
                    target_family=MGKG_TARGET_FAMILY,
                ),
            )
            for row in soil_mgkg_rows
            if soil_mgkg_parts.get(str(row["aggregate_id"])) == "test"
        )
        if not assignments:
            raise ValueError("No assignments were created for the three-stage split.")
        record_ids = [row[1] for row in assignments]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Three-stage source rows were assigned more than once; stage record identities must be unique.")

        conn.execute(
            "DELETE FROM split_assignments WHERE split_name = ? AND source_table = ?",
            (split_name, source_table),
        )
        conn.executemany(
            """
            INSERT INTO split_assignments (
                split_name, record_id, aggregate_id, split_part, seed, split_type, source_table, group_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            assignments,
        )
        inserted_count = conn.execute(
            "SELECT COUNT(*) FROM split_assignments WHERE split_name = ? AND source_table = ?",
            (split_name, source_table),
        ).fetchone()[0]
        if int(inserted_count) != len(assignments):
            raise ValueError(
                "Three-stage assignment count mismatch after insert: "
                f"expected={len(assignments)}, inserted={int(inserted_count)}"
            )
        conn.commit()
        return split_summary(conn, split_name, source_table)


def source_rows(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    medium_domain: str,
    target_name: str,
    target_family: str,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        f'''
        SELECT aggregate_id, medium_domain, target_name, target_family
        FROM "{table_name}"
        WHERE medium_domain = ? AND target_name = ?
        ORDER BY aggregate_id
        ''',
        (medium_domain, target_name),
    ).fetchall()
    unexpected_families = sorted(
        {str(row["target_family"]) for row in rows if str(row["target_family"]) != target_family}
    )
    if unexpected_families:
        raise ValueError(
            "Three-stage target-family contract failed for "
            f"medium_domain={medium_domain!r}, target_name={target_name!r}: "
            f"expected={target_family!r}, observed={unexpected_families}"
        )
    identities = [
        (
            str(row["aggregate_id"]),
            str(row["medium_domain"]),
            str(row["target_name"]),
            str(row["target_family"]),
        )
        for row in rows
    ]
    duplicate_identities = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicate_identities:
        preview = ", ".join("/".join(identity) for identity in duplicate_identities[:3])
        raise ValueError(
            "Three-stage source contains duplicate strict composite identities; "
            f"each stage sample must resolve to exactly one row ({len(duplicate_identities)} duplicates): {preview}"
        )
    return rows


def row_aggregate_ids(rows: list[sqlite3.Row]) -> list[str]:
    return [str(row["aggregate_id"]) for row in rows]


def stage_sample_record_id(row: sqlite3.Row) -> str:
    payload = json.dumps(
        [
            str(row["aggregate_id"]),
            str(row["medium_domain"]),
            str(row["target_name"]),
            str(row["target_family"]),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{STAGE_SAMPLE_ID_PREFIX}{digest}"


def stage_group_key(*, stage: str, medium_domain: str, target_name: str, target_family: str) -> str:
    return "|".join(
        (
            STAGE_CONTRACT_PREFIX,
            f"stage={stage}",
            f"medium_domain={medium_domain}",
            f"target_name={target_name}",
            f"target_family={target_family}",
        )
    )


def split_parts(conn: sqlite3.Connection, split_name: str, source_table: str) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT aggregate_id, split_part
        FROM split_assignments
        WHERE split_name = ? AND source_table = ?
        """,
        (split_name, source_table),
    ).fetchall()
    parts: dict[str, str] = {}
    for row in rows:
        aggregate_id = str(row["aggregate_id"])
        split_part = str(row["split_part"])
        if aggregate_id in parts:
            raise ValueError(
                "Referenced split contains duplicate aggregate assignments: "
                f"split_name={split_name!r}, source_table={source_table!r}, aggregate_id={aggregate_id!r}"
            )
        parts[aggregate_id] = split_part
    return parts


def assert_source_coverage(label: str, source_aggregate_ids: list[str], parts: dict[str, str]) -> None:
    missing = sorted(set(source_aggregate_ids) - set(parts))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{label} source rows are absent from the referenced split ({len(missing)} missing): {preview}")
    allowed = {"train", "test"}
    observed = set(parts.values())
    if not {"train", "test"}.issubset(observed):
        raise ValueError(f"{label} split must contain train and test rows; observed={sorted(observed)}")
    unsupported = observed - allowed
    if unsupported:
        raise ValueError(f"{label} split has unsupported parts for strict staging: {sorted(unsupported)}")


def split_summary(conn: sqlite3.Connection, split_name: str, source_table: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT split_part, COUNT(*) AS n
        FROM split_assignments
        WHERE split_name = ? AND source_table = ?
        GROUP BY split_part
        ORDER BY split_part
        """,
        (split_name, source_table),
    ).fetchall()
    return {str(row["split_part"]): int(row["n"]) for row in rows}


def ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Table not found: {table_name}")


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


def ensure_split(conn: sqlite3.Connection, split_name: str, source_table: str) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM split_assignments WHERE split_name = ? AND source_table = ?",
        (split_name, source_table),
    ).fetchone()[0]
    if int(count) == 0:
        raise ValueError(f"Split not found: split_name={split_name}, source_table={source_table}")


if __name__ == "__main__":
    main()
