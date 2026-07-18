from __future__ import annotations

import argparse
import csv
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
MGKG_TARGET_NAME = "neg_log10_mg_kg"


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
    parser.add_argument("--stage3-target-name", default=MGKG_TARGET_NAME)
    parser.add_argument("--stage3-target-family", default=MGKG_TARGET_FAMILY)
    parser.add_argument("--stage3-label", default="soil_mgkg")
    parser.add_argument("--split-name", default="M_v1_2_39_ptox_to_soil_mgkg_B_random_8_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        help="Optional one-row CSV recording the exact stage-1/source-overlap routing audit.",
    )
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
        stage3_target_name=args.stage3_target_name,
        stage3_target_family=args.stage3_target_family,
        stage3_label=args.stage3_label,
        split_name=args.split_name,
        seed=args.seed,
        audit_csv=args.audit_csv,
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
    stage3_target_name: str = MGKG_TARGET_NAME,
    stage3_target_family: str = MGKG_TARGET_FAMILY,
    stage3_label: str = "soil_mgkg",
    seed: int = 42,
    audit_csv: Path | None = None,
) -> dict[str, int]:
    """Create one split where each stage has a non-overlapping training role.

    Stage 1 contains aquatic-only pTox records. Any aquatic record sharing an
    aggregate identity or raw result source with any soil-pTox candidate is
    removed from stage 1, regardless of whether that soil row later belongs to
    the stage-2 train or test part. Stage 2 contains only the training part of
    the existing soil pTox split. Stage 3 contains only the training part of the
    existing soil target split; its test rows are the sole external test set.
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
        stage3_target_name = str(stage3_target_name).strip()
        stage3_target_family = str(stage3_target_family).strip()
        stage3_label = str(stage3_label).strip().lower().replace("-", "_")
        if not stage3_target_name or not stage3_target_family or not stage3_label:
            raise ValueError("Stage-3 target name, family, and label must be non-empty.")
        soil_mgkg_rows = source_rows(
            conn,
            source_table,
            medium_domain="soil",
            target_name=stage3_target_name,
            target_family=stage3_target_family,
        )
        if not aquatic_ptox_rows or not soil_ptox_rows or not soil_mgkg_rows:
            raise ValueError("The three-stage source table is missing one or more required target domains.")

        aquatic_ptox_rows, routing_audit = route_aquatic_ptox_stage1(
            aquatic_rows=aquatic_ptox_rows,
            soil_ptox_rows=soil_ptox_rows,
        )
        if not aquatic_ptox_rows:
            raise ValueError("Exact source routing removed every aquatic pTox stage-1 candidate.")

        soil_ptox_parts = split_parts(conn, soil_ptox_split_name, soil_ptox_source_table)
        soil_mgkg_parts = split_parts(conn, soil_mgkg_split_name, soil_mgkg_source_table)
        assert_source_coverage("soil pTox", row_aggregate_ids(soil_ptox_rows), soil_ptox_parts)
        assert_source_coverage(
            f"soil stage-3 target {stage3_target_name}",
            row_aggregate_ids(soil_mgkg_rows),
            soil_mgkg_parts,
        )

        split_type = f"strict_three_stage_ptox_to_{stage3_label}"

        assignments: list[tuple[str, str, str, str, int, str, str, str]] = []
        assignments.extend(
            (
                split_name,
                stage_sample_record_id(row),
                str(row["aggregate_id"]),
                "train",
                seed,
                split_type,
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
                split_type,
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
                split_type,
                source_table,
                stage_group_key(
                    stage=f"{stage3_label}_stage3_train",
                    medium_domain="soil",
                    target_name=stage3_target_name,
                    target_family=stage3_target_family,
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
                split_type,
                source_table,
                stage_group_key(
                    stage=f"{stage3_label}_stage3_test",
                    medium_domain="soil",
                    target_name=stage3_target_name,
                    target_family=stage3_target_family,
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
        if audit_csv is not None:
            write_routing_audit(
                audit_csv,
                split_name=split_name,
                source_table=source_table,
                audit=routing_audit,
            )
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
        SELECT aggregate_id, medium_domain, target_name, target_family, result_ids
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


def parse_result_ids(value: object) -> frozenset[str]:
    """Parse the source-result identity list without widening the exclusion key."""
    if value is None or not str(value).strip():
        return frozenset()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"result_ids is not valid JSON: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"result_ids must be a JSON array: {value!r}")
    normalized: set[str] = set()
    for item in parsed:
        if isinstance(item, (dict, list)) or item is None or not str(item).strip():
            raise ValueError(f"result_ids contains an invalid source identity: {value!r}")
        normalized.add(str(item).strip())
    return frozenset(normalized)


def route_aquatic_ptox_stage1(
    *,
    aquatic_rows: list[sqlite3.Row],
    soil_ptox_rows: list[sqlite3.Row],
) -> tuple[list[sqlite3.Row], dict[str, int | str]]:
    """Give all exact aquatic/soil source collisions exclusively to stage 2.

    The exclusion keys are intentionally narrow: exact ``aggregate_id`` and
    exact raw ``result_ids`` only. CAS, test ID, reference, organism, endpoint,
    and similar experimental context are not used as group-level exclusions.
    """
    soil_aggregate_ids = {str(row["aggregate_id"]) for row in soil_ptox_rows}
    soil_result_ids = collect_result_ids(soil_ptox_rows)
    aquatic_aggregate_ids = {str(row["aggregate_id"]) for row in aquatic_rows}
    aquatic_result_ids = collect_result_ids(aquatic_rows)

    routed_rows: list[sqlite3.Row] = []
    excluded_rows: list[sqlite3.Row] = []
    same_aggregate_only = 0
    shared_result_only = 0
    both = 0
    for row in aquatic_rows:
        same_aggregate = str(row["aggregate_id"]) in soil_aggregate_ids
        shared_result = bool(parse_result_ids(row["result_ids"]) & soil_result_ids)
        if same_aggregate and shared_result:
            both += 1
            excluded_rows.append(row)
        elif same_aggregate:
            same_aggregate_only += 1
            excluded_rows.append(row)
        elif shared_result:
            shared_result_only += 1
            excluded_rows.append(row)
        else:
            routed_rows.append(row)

    excluded_total = same_aggregate_only + shared_result_only + both
    soil_rows_overlapping_aquatic = sum(
        1
        for row in soil_ptox_rows
        if str(row["aggregate_id"]) in aquatic_aggregate_ids
        or bool(parse_result_ids(row["result_ids"]) & aquatic_result_ids)
    )
    routed_aggregate_ids = {str(row["aggregate_id"]) for row in routed_rows}
    routed_result_ids = collect_result_ids(routed_rows)
    residual_aggregate_overlap = routed_aggregate_ids & soil_aggregate_ids
    residual_result_overlap = routed_result_ids & soil_result_ids
    if residual_aggregate_overlap or residual_result_overlap:
        raise ValueError(
            "Exact source routing failed to isolate aquatic stage 1 from all soil-pTox candidates: "
            f"aggregate_overlap={len(residual_aggregate_overlap)}, "
            f"result_id_overlap={len(residual_result_overlap)}"
        )

    return routed_rows, {
        "routing_rule": "exact_aggregate_or_raw_result_v1",
        "aquatic_ptox_candidates_before": len(aquatic_rows),
        "soil_ptox_candidates": len(soil_ptox_rows),
        "soil_ptox_candidates_overlapping_aquatic_before": soil_rows_overlapping_aquatic,
        "aquatic_ptox_excluded_total": excluded_total,
        "aquatic_ptox_excluded_same_aggregate_only": same_aggregate_only,
        "aquatic_ptox_excluded_shared_result_only": shared_result_only,
        "aquatic_ptox_excluded_both": both,
        "aquatic_ptox_stage1_after": len(routed_rows),
        "residual_aggregate_id_overlap": len(residual_aggregate_overlap),
        "residual_result_id_overlap": len(residual_result_overlap),
        "aquatic_ptox_excluded_identity_sha256": hash_source_rows(excluded_rows),
        "aquatic_ptox_stage1_identity_sha256": hash_source_rows(routed_rows),
        "soil_ptox_candidates_identity_sha256": hash_source_rows(soil_ptox_rows),
    }


def collect_result_ids(rows: list[sqlite3.Row]) -> set[str]:
    result_ids: set[str] = set()
    for row in rows:
        result_ids.update(parse_result_ids(row["result_ids"]))
    return result_ids


def hash_source_rows(rows: list[sqlite3.Row]) -> str:
    digest = hashlib.sha256()
    identities = sorted(
        json.dumps(
            [str(row["aggregate_id"]), sorted(parse_result_ids(row["result_ids"]))],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    )
    for identity in identities:
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_routing_audit(
    path: Path,
    *,
    split_name: str,
    source_table: str,
    audit: dict[str, int | str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, str | int] = {
        "split_name": split_name,
        "source_table": source_table,
        **audit,
    }
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


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
