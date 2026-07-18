from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


PTOX_TARGET_NAME = "ptox_mol_l"
PTOX_TARGET_FAMILY = "aquatic_pTox_mol_L"
MGKG_TARGET_NAME = "neg_log10_mg_kg"
MGKG_TARGET_FAMILY = "solid_neglog_mg_kg"
MOLKG_TARGET_NAME = "neg_log10_mol_kg"
MOLKG_TARGET_FAMILY = "solid_neglog_mol_kg"
STAGE_SAMPLE_ID_PREFIX = "stage_sample_v1:"
STAGE_CONTRACT_PREFIX = "stage_contract_v1"
TRANSFORM_NAME = "neglog_mgkg_plus_log10_1000_mw_v1"
TRANSFORM_COLUMNS = {
    "target_value_median",
    "target_value_mean",
    "target_value_weighted_mean",
    "target_value_unweighted_median",
    "target_value_min",
    "target_value_max",
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize paired soil mg/kg and mol/kg targets without re-splitting parent records. "
            "The mol/kg target is -log10(mol/kg) = -log10(mg/kg) + log10(1000*MW)."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--source-table",
        default="aggregated_task_records_medium_domain_qc_expanded",
    )
    parser.add_argument("--mw-source-table", default="task_records_qc")
    parser.add_argument("--parent-mgkg-source-table", default="aggregated_task_records_soil_mg_kg_qc")
    parser.add_argument("--parent-mgkg-split", default="SoilMgkgQC2_B_random_8_2")
    parser.add_argument(
        "--output-table",
        default="aggregated_task_records_ptox_soil_mass_molar_qc",
    )
    parser.add_argument("--mw-map-table", default="soil_molkg_molecular_weight_map_v1")
    parser.add_argument("--matched-mgkg-split", default="SoilMgkgMWMatched_B_random_8_2")
    parser.add_argument("--molkg-split", default="SoilMolkgMWMatched_B_random_8_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_paired_targets(
        db_path=args.db,
        source_table=args.source_table,
        mw_source_table=args.mw_source_table,
        parent_mgkg_source_table=args.parent_mgkg_source_table,
        parent_mgkg_split=args.parent_mgkg_split,
        output_table=args.output_table,
        mw_map_table=args.mw_map_table,
        matched_mgkg_split=args.matched_mgkg_split,
        molkg_split=args.molkg_split,
        seed=args.seed,
        audit_csv=args.audit_csv,
        summary_json=args.summary_json,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def build_paired_targets(
    *,
    db_path: Path,
    source_table: str,
    mw_source_table: str,
    parent_mgkg_source_table: str,
    parent_mgkg_split: str,
    output_table: str,
    mw_map_table: str,
    matched_mgkg_split: str,
    molkg_split: str,
    seed: int = 42,
    audit_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    for name in (source_table, mw_source_table, parent_mgkg_source_table, output_table, mw_map_table):
        validate_identifier(name)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    with closing(sqlite3.connect(db_path, timeout=120.0)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.create_function("py_log10", 1, lambda value: math.log10(float(value)))
        for table_name in (source_table, mw_source_table, parent_mgkg_source_table, "split_assignments"):
            ensure_table(conn, table_name)
        source_columns = table_columns(conn, source_table)
        required = {
            "aggregate_id",
            "cas_number",
            "smiles",
            "medium_domain",
            "target_name",
            "target_family",
            "target_basis",
            "target_value_median",
            "standard_value_mg_kg",
            "result_ids",
        }
        missing = sorted(required - set(source_columns))
        if missing:
            raise ValueError(f"Source table is missing required columns: {missing}")

        build_molecular_weight_map(
            conn,
            source_table=source_table,
            mw_source_table=mw_source_table,
            mw_map_table=mw_map_table,
        )
        parent_parts = load_parent_split_parts(
            conn,
            split_name=parent_mgkg_split,
            source_table=parent_mgkg_source_table,
        )
        audit_rows = load_conversion_audit_rows(
            conn,
            source_table=source_table,
            mw_map_table=mw_map_table,
            parent_parts=parent_parts,
        )
        assert_conversion_audit(audit_rows)
        materialize_paired_source_table(
            conn,
            source_table=source_table,
            output_table=output_table,
            mw_map_table=mw_map_table,
            source_columns=source_columns,
        )
        matched_ids = {
            str(row["aggregate_id"])
            for row in audit_rows
            if row["conversion_status"] == "converted"
        }
        write_direct_split(
            conn,
            split_name=matched_mgkg_split,
            source_table=output_table,
            target_name=MGKG_TARGET_NAME,
            target_family=MGKG_TARGET_FAMILY,
            stage="soil_mgkg_mw_matched_direct",
            parent_parts=parent_parts,
            matched_ids=matched_ids,
            seed=seed,
        )
        write_direct_split(
            conn,
            split_name=molkg_split,
            source_table=output_table,
            target_name=MOLKG_TARGET_NAME,
            target_family=MOLKG_TARGET_FAMILY,
            stage="soil_molkg_mw_matched_direct",
            parent_parts=parent_parts,
            matched_ids=matched_ids,
            seed=seed,
        )
        conn.commit()

        summary = summarize_build(
            conn,
            source_table=source_table,
            output_table=output_table,
            matched_mgkg_split=matched_mgkg_split,
            molkg_split=molkg_split,
            audit_rows=audit_rows,
        )

    if audit_csv is not None:
        write_audit_csv(audit_csv, audit_rows)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def build_molecular_weight_map(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    mw_source_table: str,
    mw_map_table: str,
) -> None:
    columns = set(table_columns(conn, mw_source_table))
    required = {"result_id", "molecular_weight_g_mol", "molecular_weight_rdkit_g_mol"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Molecular-weight source is missing columns: {missing}")
    # The 9 GB modeling database historically did not index result_id on the
    # QC task table. The paired conversion resolves exact parent result IDs,
    # so this reusable index prevents an otherwise quadratic JSON-ID join.
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{mw_source_table}_result_id_molkg" '
        f'ON "{mw_source_table}" (result_id)'
    )
    temporary = f"{mw_map_table}__building"
    conn.execute(f'DROP TABLE IF EXISTS "{temporary}"')
    conn.execute(
        f'''
        CREATE TABLE "{temporary}" AS
        WITH candidates AS (
            SELECT
                e.aggregate_id,
                CAST(e.cas_number AS TEXT) AS cas_number,
                CASE
                    WHEN t.molecular_weight_g_mol > 0 THEN t.molecular_weight_g_mol
                    WHEN t.molecular_weight_rdkit_g_mol > 0 THEN t.molecular_weight_rdkit_g_mol
                    ELSE NULL
                END AS molecular_weight_g_mol_used,
                CASE
                    WHEN t.molecular_weight_g_mol > 0 THEN 'molecular_weight_g_mol'
                    WHEN t.molecular_weight_rdkit_g_mol > 0 THEN 'molecular_weight_rdkit_g_mol'
                    ELSE ''
                END AS molecular_weight_source_used
            FROM "{source_table}" e
            JOIN json_each(e.result_ids) source_result
            JOIN "{mw_source_table}" t
              ON t.result_id IN (source_result.value, CAST(source_result.value AS TEXT))
            WHERE e.medium_domain = 'soil'
              AND e.target_name = '{MGKG_TARGET_NAME}'
              AND e.target_family = '{MGKG_TARGET_FAMILY}'
        ), usable AS (
            SELECT * FROM candidates
            WHERE molecular_weight_g_mol_used IS NOT NULL
              AND molecular_weight_g_mol_used > 0
        )
        SELECT
            aggregate_id,
            cas_number,
            MAX(molecular_weight_g_mol_used) AS molecular_weight_g_mol_used,
            MAX(molecular_weight_source_used) AS molecular_weight_source_used,
            COUNT(DISTINCT printf('%.8f', molecular_weight_g_mol_used)) AS distinct_mw_count
        FROM usable
        GROUP BY aggregate_id, cas_number
        '''
    )
    conflict = conn.execute(
        f'SELECT COUNT(*) FROM "{temporary}" WHERE distinct_mw_count != 1'
    ).fetchone()[0]
    if int(conflict) != 0:
        raise ValueError(f"Molecular-weight map contains {int(conflict)} conflicting CAS entries.")
    conn.execute(f'DROP TABLE IF EXISTS "{mw_map_table}"')
    conn.execute(f'ALTER TABLE "{temporary}" RENAME TO "{mw_map_table}"')
    conn.execute(
        f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{mw_map_table}_aggregate" '
        f'ON "{mw_map_table}" (aggregate_id)'
    )


def materialize_paired_source_table(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    output_table: str,
    mw_map_table: str,
    source_columns: list[str],
) -> None:
    temporary = f"{output_table}__building"
    validate_identifier(temporary)
    conn.execute(f'DROP TABLE IF EXISTS "{temporary}"')
    identity_columns = ",\n                ".join(f'e."{column}"' for column in source_columns)
    molkg_columns = ",\n                ".join(
        molkg_expression(column) for column in source_columns
    )
    conn.execute(
        f'''
        CREATE TABLE "{temporary}" AS
        SELECT
            {identity_columns},
            NULL AS standard_value_mol_kg,
            NULL AS molecular_weight_g_mol_used,
            'identity_ptox' AS target_transform,
            target_name AS parent_target_name,
            target_family AS parent_target_family,
            target_basis AS parent_target_basis
        FROM "{source_table}" e
        WHERE e.target_name = ?
          AND e.target_family = ?
          AND e.medium_domain IN ('aquatic', 'soil')

        UNION ALL

        SELECT
            {identity_columns},
            NULL AS standard_value_mol_kg,
            mw.molecular_weight_g_mol_used,
            'identity_mgkg_mw_complete' AS target_transform,
            e.target_name AS parent_target_name,
            e.target_family AS parent_target_family,
            e.target_basis AS parent_target_basis
        FROM "{source_table}" e
        JOIN "{mw_map_table}" mw ON e.aggregate_id = mw.aggregate_id
        WHERE e.target_name = ?
          AND e.target_family = ?
          AND e.medium_domain = 'soil'

        UNION ALL

        SELECT
            {molkg_columns},
            CASE
                WHEN e.standard_value_mg_kg > 0
                THEN e.standard_value_mg_kg / (1000.0 * mw.molecular_weight_g_mol_used)
                ELSE NULL
            END AS standard_value_mol_kg,
            mw.molecular_weight_g_mol_used,
            ? AS target_transform,
            e.target_name AS parent_target_name,
            e.target_family AS parent_target_family,
            e.target_basis AS parent_target_basis
        FROM "{source_table}" e
        JOIN "{mw_map_table}" mw ON e.aggregate_id = mw.aggregate_id
        WHERE e.target_name = ?
          AND e.target_family = ?
          AND e.medium_domain = 'soil'
        ''',
        (
            PTOX_TARGET_NAME,
            PTOX_TARGET_FAMILY,
            MGKG_TARGET_NAME,
            MGKG_TARGET_FAMILY,
            TRANSFORM_NAME,
            MGKG_TARGET_NAME,
            MGKG_TARGET_FAMILY,
        ),
    )
    duplicate = conn.execute(
        f'''
        SELECT COUNT(*) FROM (
            SELECT aggregate_id, medium_domain, target_name, target_family, COUNT(*) AS n
            FROM "{temporary}"
            GROUP BY aggregate_id, medium_domain, target_name, target_family
            HAVING n != 1
        )
        '''
    ).fetchone()[0]
    if int(duplicate) != 0:
        raise ValueError(f"Paired source has {int(duplicate)} duplicate stage identities.")
    conn.execute(f'DROP TABLE IF EXISTS "{output_table}"')
    conn.execute(f'ALTER TABLE "{temporary}" RENAME TO "{output_table}"')
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{output_table}_aggregate" '
        f'ON "{output_table}" (aggregate_id)'
    )
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{output_table}_target" '
        f'ON "{output_table}" (medium_domain, target_name, target_family)'
    )


def molkg_expression(column: str) -> str:
    quoted = f'e."{column}"'
    offset = "(3.0 + py_log10(mw.molecular_weight_g_mol_used))"
    if column in TRANSFORM_COLUMNS:
        return f'CASE WHEN {quoted} IS NULL THEN NULL ELSE {quoted} + {offset} END AS "{column}"'
    if column == "target_name":
        return f"'{MOLKG_TARGET_NAME}' AS target_name"
    if column == "target_family":
        return f"'{MOLKG_TARGET_FAMILY}' AS target_family"
    if column == "target_basis":
        return "'mol/kg_from_mg/kg:soil' AS target_basis"
    if column == "unit_family_v2":
        return "'soil_mol_kg' AS unit_family_v2"
    if column == "standard_unit_v2":
        return "'mol/kg' AS standard_unit_v2"
    if column == "conversion_path":
        return (
            "(COALESCE(e.conversion_path, '') || "
            "'|mg_kg_to_mol_kg_using_molecular_weight') AS conversion_path"
        )
    return quoted


def load_parent_split_parts(
    conn: sqlite3.Connection,
    *,
    split_name: str,
    source_table: str,
) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT aggregate_id, split_part
        FROM split_assignments
        WHERE split_name = ? AND source_table = ?
        ORDER BY rowid
        """,
        (split_name, source_table),
    ).fetchall()
    if not rows:
        raise ValueError(
            f"Parent soil mg/kg split is missing: split={split_name!r}, source={source_table!r}"
        )
    parts: dict[str, str] = {}
    for row in rows:
        aggregate_id = str(row["aggregate_id"])
        split_part = str(row["split_part"])
        if aggregate_id in parts:
            raise ValueError(f"Parent split duplicates aggregate_id={aggregate_id!r}")
        if split_part not in {"train", "test"}:
            raise ValueError(f"Unsupported parent split part {split_part!r}")
        parts[aggregate_id] = split_part
    return parts


def load_conversion_audit_rows(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    mw_map_table: str,
    parent_parts: dict[str, str],
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f'''
        SELECT
            e.aggregate_id,
            e.cas_number,
            e.smiles,
            e.result_ids,
            e.value_quality,
            e.target_value_median AS y_mgkg,
            mw.molecular_weight_g_mol_used,
            mw.molecular_weight_source_used
        FROM "{source_table}" e
        LEFT JOIN "{mw_map_table}" mw ON e.aggregate_id = mw.aggregate_id
        WHERE e.medium_domain = 'soil'
          AND e.target_name = ?
          AND e.target_family = ?
        ORDER BY e.aggregate_id
        ''',
        (MGKG_TARGET_NAME, MGKG_TARGET_FAMILY),
    ).fetchall()
    audit: list[dict[str, Any]] = []
    for row in rows:
        aggregate_id = str(row["aggregate_id"])
        if aggregate_id not in parent_parts:
            raise ValueError(f"Soil mg/kg aggregate is absent from parent split: {aggregate_id}")
        mw = optional_positive_float(row["molecular_weight_g_mol_used"])
        y_mgkg = float(row["y_mgkg"])
        y_molkg = None if mw is None else y_mgkg + math.log10(1000.0 * mw)
        roundtrip_error = None if y_molkg is None else abs(y_mgkg - (y_molkg - math.log10(1000.0 * mw)))
        result_ids = parse_result_ids(row["result_ids"])
        smiles = str(row["smiles"] or "")
        audit.append(
            {
                "aggregate_id": aggregate_id,
                "cas_number": str(row["cas_number"] or ""),
                "parent_split_part": parent_parts[aggregate_id],
                "result_ids_sha256": stable_hash(result_ids),
                "molecular_weight_g_mol_used": "" if mw is None else mw,
                "molecular_weight_source_used": str(row["molecular_weight_source_used"] or ""),
                "conversion_status": "conversion_unavailable" if mw is None else "converted",
                "multifragment_smiles": int("." in smiles),
                "value_quality": str(row["value_quality"] or ""),
                "y_neg_log10_mg_kg": y_mgkg,
                "y_neg_log10_mol_kg": "" if y_molkg is None else y_molkg,
                "roundtrip_abs_error": "" if roundtrip_error is None else roundtrip_error,
            }
        )
    return audit


def assert_conversion_audit(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No soil mg/kg rows were found for paired conversion.")
    converted = [row for row in rows if row["conversion_status"] == "converted"]
    missing = [row for row in rows if row["conversion_status"] != "converted"]
    if not converted:
        raise ValueError("No soil mg/kg rows have usable molecular weight.")
    max_error = max(float(row["roundtrip_abs_error"]) for row in converted)
    if not math.isfinite(max_error) or max_error >= 1e-12:
        raise ValueError(f"mg/kg <-> mol/kg roundtrip error is too large: {max_error}")
    observed_parts = {str(row["parent_split_part"]) for row in rows}
    if observed_parts != {"train", "test"}:
        raise ValueError(f"Parent split must contain train and test rows; observed={observed_parts}")
    if len({str(row["aggregate_id"]) for row in rows}) != len(rows):
        raise ValueError("Soil mg/kg conversion audit contains duplicate aggregate IDs.")
    if len(converted) + len(missing) != len(rows):
        raise AssertionError("Conversion accounting mismatch.")


def write_direct_split(
    conn: sqlite3.Connection,
    *,
    split_name: str,
    source_table: str,
    target_name: str,
    target_family: str,
    stage: str,
    parent_parts: dict[str, str],
    matched_ids: set[str],
    seed: int,
) -> None:
    available = {
        str(row[0])
        for row in conn.execute(
            f'''
            SELECT aggregate_id FROM "{source_table}"
            WHERE medium_domain = 'soil' AND target_name = ? AND target_family = ?
            ''',
            (target_name, target_family),
        )
    }
    if available != matched_ids:
        raise ValueError(
            f"Paired target coverage mismatch for {target_name}: "
            f"available={len(available)}, expected={len(matched_ids)}"
        )
    assignments = [
        (
            split_name,
            stage_sample_record_id(aggregate_id, "soil", target_name, target_family),
            aggregate_id,
            parent_parts[aggregate_id],
            seed,
            "paired_parent_split_mw_complete_v1",
            source_table,
            stage_group_key(
                stage=stage,
                medium_domain="soil",
                target_name=target_name,
                target_family=target_family,
            ),
        )
        for aggregate_id in sorted(matched_ids, key=sortable_id)
    ]
    conn.execute(
        "DELETE FROM split_assignments WHERE split_name = ? AND source_table = ?",
        (split_name, source_table),
    )
    conn.executemany(
        """
        INSERT INTO split_assignments (
            split_name, record_id, aggregate_id, split_part, seed,
            split_type, source_table, group_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        assignments,
    )
    inserted = conn.execute(
        "SELECT COUNT(*) FROM split_assignments WHERE split_name = ? AND source_table = ?",
        (split_name, source_table),
    ).fetchone()[0]
    if int(inserted) != len(assignments):
        raise ValueError(
            f"Paired direct split insert mismatch: expected={len(assignments)}, inserted={inserted}"
        )


def summarize_build(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    output_table: str,
    matched_mgkg_split: str,
    molkg_split: str,
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    converted = [row for row in audit_rows if row["conversion_status"] == "converted"]
    unavailable = [row for row in audit_rows if row["conversion_status"] != "converted"]
    max_error = max(float(row["roundtrip_abs_error"]) for row in converted)
    counts_by_status: dict[str, dict[str, int]] = {}
    for row in audit_rows:
        status = str(row["conversion_status"])
        part = str(row["parent_split_part"])
        counts_by_status.setdefault(status, {}).setdefault(part, 0)
        counts_by_status[status][part] += 1
    split_counts = {}
    for split_name in (matched_mgkg_split, molkg_split):
        split_counts[split_name] = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                """
                SELECT split_part, COUNT(*) FROM split_assignments
                WHERE split_name = ? AND source_table = ?
                GROUP BY split_part ORDER BY split_part
                """,
                (split_name, output_table),
            )
        }
    target_counts = {
        f"{row[0]}|{row[1]}|{row[2]}": int(row[3])
        for row in conn.execute(
            f'''
            SELECT medium_domain, target_name, target_family, COUNT(*)
            FROM "{output_table}"
            GROUP BY medium_domain, target_name, target_family
            ORDER BY medium_domain, target_name, target_family
            '''
        )
    }
    return {
        "schema_version": "paired_soil_mass_molar_v1",
        "source_table": source_table,
        "output_table": output_table,
        "formula": "neg_log10_mol_kg = neg_log10_mg_kg + log10(1000 * molecular_weight_g_mol)",
        "parent_soil_mgkg_rows": len(audit_rows),
        "converted_rows": len(converted),
        "conversion_unavailable_rows": len(unavailable),
        "coverage_fraction": len(converted) / len(audit_rows),
        "multifragment_converted_rows": sum(int(row["multifragment_smiles"]) for row in converted),
        "roundtrip_max_abs_error": max_error,
        "counts_by_conversion_status_and_parent_split": counts_by_status,
        "split_counts": split_counts,
        "target_counts": target_counts,
        "converted_aggregate_id_sha256": stable_hash(
            sorted(str(row["aggregate_id"]) for row in converted)
        ),
        "unavailable_aggregate_id_sha256": stable_hash(
            sorted(str(row["aggregate_id"]) for row in unavailable)
        ),
    }


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_sample_record_id(
    aggregate_id: Any,
    medium_domain: str,
    target_name: str,
    target_family: str,
) -> str:
    payload = json.dumps(
        [str(aggregate_id), medium_domain, target_name, target_family],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{STAGE_SAMPLE_ID_PREFIX}{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


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


def parse_result_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return sorted({str(item) for item in parsed if item not in (None, "")})


def stable_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def optional_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def sortable_id(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def validate_identifier(value: str) -> None:
    if not IDENTIFIER.fullmatch(str(value)):
        raise ValueError(f"Unsafe SQLite identifier: {value!r}")


def ensure_table(conn: sqlite3.Connection, table_name: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Required table does not exist: {table_name}")


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table_name}")')]


if __name__ == "__main__":
    main()
