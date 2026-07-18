from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


DEFAULT_TABLES = (
    "target_records",
    "aggregated_task_records_aquatic_soil_ptox_qc",
    "aggregated_task_records_aquatic_ptox_qc",
    "aggregated_task_records_soil_ptox_qc",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a derived SQLite subset that excludes chemicals classified as "
            "inorganic or metal/metalloid, without modifying the source database."
        )
    )
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--out-db", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing output database.")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(DEFAULT_TABLES),
        help="Source tables to copy after filtering by excluded CAS/DTXSID.",
    )
    parser.add_argument(
        "--suffix",
        default="_no_metal_inorganic",
        help="Suffix appended to copied source-table names except target_records.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = build_no_metal_inorganic_dataset(
        source_db=args.source_db,
        out_db=args.out_db,
        tables=tuple(args.tables),
        suffix=args.suffix,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


def build_no_metal_inorganic_dataset(
    *,
    source_db: Path,
    out_db: Path,
    tables: tuple[str, ...] = DEFAULT_TABLES,
    suffix: str = "_no_metal_inorganic",
    force: bool = False,
) -> dict[str, Any]:
    source_path = source_db.resolve()
    out_path = out_db.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if out_path.exists():
        if not force:
            raise FileExistsError(f"Output database already exists: {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(out_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("ATTACH DATABASE ? AS src", (str(source_path),))
        try:
            ensure_table(conn, "src", "target_records")
            create_excluded_chemicals(conn)
            table_stats: dict[str, dict[str, Any]] = {}
            for table in tables:
                ensure_table(conn, "src", table)
                output_table = "target_records" if table == "target_records" else f"{table}{suffix}"
                table_stats[output_table] = copy_filtered_table(
                    conn,
                    source_table=table,
                    output_table=output_table,
                )
            create_indexes(conn, suffix=suffix)
            manifest = {
                "source_db": str(source_path),
                "out_db": str(out_path),
                "filter": "exclude any CAS/DTXSID classified as chemical_class_l1=inorganic or chemical_class_l2=metal_metalloid in target_records",
                "excluded_chemicals": excluded_summary(conn),
                "tables": table_stats,
            }
            write_manifest(conn, manifest)
            conn.commit()
        finally:
            conn.execute("DETACH DATABASE src")
    return manifest


def create_excluded_chemicals(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS main.excluded_metal_inorganic_chemicals")
    conn.execute(
        """
        CREATE TABLE excluded_metal_inorganic_chemicals AS
        SELECT
            cas_number,
            dtxsid,
            MAX(CASE WHEN LOWER(COALESCE(chemical_class_l1, '')) = 'inorganic' THEN 1 ELSE 0 END) AS has_inorganic,
            MAX(CASE WHEN LOWER(COALESCE(chemical_class_l2, '')) = 'metal_metalloid' THEN 1 ELSE 0 END) AS has_metal_metalloid,
            COUNT(*) AS source_rows
        FROM src.target_records
        GROUP BY cas_number, dtxsid
        HAVING has_inorganic = 1 OR has_metal_metalloid = 1
        """
    )
    conn.execute(
        "CREATE INDEX idx_excluded_metal_inorganic_cas ON excluded_metal_inorganic_chemicals(cas_number)"
    )
    conn.execute(
        "CREATE INDEX idx_excluded_metal_inorganic_dtxsid ON excluded_metal_inorganic_chemicals(dtxsid)"
    )


def copy_filtered_table(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    output_table: str,
) -> dict[str, Any]:
    conn.execute(f'DROP TABLE IF EXISTS main."{output_table}"')
    where = allowed_chemical_where_clause("t")
    conn.execute(
        f"""
        CREATE TABLE main."{output_table}" AS
        SELECT *
        FROM src.{quote_identifier(source_table)} AS t
        WHERE {where}
        """
    )
    input_rows = conn.execute(f"SELECT COUNT(*) FROM src.{quote_identifier(source_table)}").fetchone()[0]
    output_rows = conn.execute(f'SELECT COUNT(*) FROM "{output_table}"').fetchone()[0]
    cas_count = count_distinct_if_column_exists(conn, output_table, "cas_number")
    dtxsid_count = count_distinct_if_column_exists(conn, output_table, "dtxsid")
    smiles_count = count_distinct_if_column_exists(conn, output_table, "smiles")
    return {
        "source_table": source_table,
        "output_table": output_table,
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "excluded_rows": int(input_rows) - int(output_rows),
        "cas_count": cas_count,
        "dtxsid_count": dtxsid_count,
        "smiles_count": smiles_count,
    }


def allowed_chemical_where_clause(table_expr: str) -> str:
    return f"""
        NOT EXISTS (
            SELECT 1
            FROM excluded_metal_inorganic_chemicals ex
            WHERE (
                ex.cas_number IS NOT NULL
                AND TRIM(CAST(ex.cas_number AS TEXT)) <> ''
                AND TRIM(CAST(ex.cas_number AS TEXT)) = TRIM(CAST({table_expr}.cas_number AS TEXT))
            )
            OR (
                ex.dtxsid IS NOT NULL
                AND TRIM(CAST(ex.dtxsid AS TEXT)) <> ''
                AND TRIM(CAST(ex.dtxsid AS TEXT)) = TRIM(CAST({table_expr}.dtxsid AS TEXT))
            )
        )
    """


def create_indexes(conn: sqlite3.Connection, *, suffix: str) -> None:
    for table in ("target_records", *[f"{name}{suffix}" for name in DEFAULT_TABLES if name != "target_records"]):
        if not table_exists(conn, "main", table):
            continue
        columns = set(table_columns(conn, "main", table))
        if "aggregate_id" in columns:
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_aggregate_id" ON "{table}"(aggregate_id)')
        if "cas_number" in columns:
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_cas" ON "{table}"(cas_number)')
        if "dtxsid" in columns:
            conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_dtxsid" ON "{table}"(dtxsid)')
        if {"medium_domain", "target_name"}.issubset(columns):
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{table}_medium_target" ON "{table}"(medium_domain, target_name)'
            )


def excluded_summary(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS keys,
            COUNT(DISTINCT cas_number) AS cas_count,
            COUNT(DISTINCT dtxsid) AS dtxsid_count,
            SUM(source_rows) AS source_rows
        FROM excluded_metal_inorganic_chemicals
        """
    ).fetchone()
    return {
        "keys": int(row[0] or 0),
        "cas_count": int(row[1] or 0),
        "dtxsid_count": int(row[2] or 0),
        "source_rows": int(row[3] or 0),
    }


def write_manifest(conn: sqlite3.Connection, manifest: dict[str, Any]) -> None:
    conn.execute("DROP TABLE IF EXISTS main.no_metal_inorganic_build_manifest")
    conn.execute('CREATE TABLE main.no_metal_inorganic_build_manifest ("key" TEXT, "value" TEXT)')
    rows = [(key, json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value) for key, value in manifest.items()]
    conn.executemany('INSERT INTO no_metal_inorganic_build_manifest ("key", "value") VALUES (?, ?)', rows)


def count_distinct_if_column_exists(conn: sqlite3.Connection, table: str, column: str) -> int | None:
    if column not in table_columns(conn, "main", table):
        return None
    value = conn.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()[0]
    return int(value or 0)


def ensure_table(conn: sqlite3.Connection, schema: str, table: str) -> None:
    if not table_exists(conn, schema, table):
        raise ValueError(f"Table not found: {schema}.{table}")


def table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA {schema}.table_info({quote_identifier(table)})")]


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


if __name__ == "__main__":
    main()
