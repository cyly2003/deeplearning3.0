from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from qsar_tl.features.padel import (
    run_padel_jar,
    write_padel_feature_cache_from_csv,
    write_padel_feature_cache_from_smiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deep-experiment compatible PaDEL descriptor JSONL cache."
    )
    parser.add_argument("--padel-csv", type=Path, default=None, help="PaDEL output CSV with SMILES and descriptors.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database to read unique SMILES from.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL cache path.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--fingerprint-size", type=int, default=512)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--source-table", default="", help="SQLite source table or manifest label.")
    parser.add_argument("--feature-source", default="padel_descriptor_morgan")
    parser.add_argument("--padel-jar", type=Path, default=None, help="Optional PaDEL jar to run before cache conversion.")
    parser.add_argument("--padel-input", type=Path, default=None, help="Input directory/file for PaDEL jar.")
    parser.add_argument("--limit", type=int, default=None, help="Optional SMILES limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=64, help="SMILES batch size for padelpy.from_smiles.")
    parser.add_argument("--padel-threads", type=int, default=1)
    parser.add_argument("--padel-timeout-s", type=int, default=300)
    parser.add_argument("--padel-maxruntime-s", type=int, default=-1)
    parser.add_argument(
        "--no-fallback-single",
        action="store_true",
        help="Disable per-SMILES fallback when a PaDEL batch fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.padel_csv is None and args.db is None:
        raise SystemExit("Provide either --padel-csv or --db.")
    if args.padel_jar is not None:
        if args.padel_csv is None:
            raise SystemExit("--padel-csv is required when --padel-jar is provided.")
        if args.padel_input is None:
            raise SystemExit("--padel-input is required when --padel-jar is provided.")
        run_padel_jar(
            jar_path=args.padel_jar,
            input_path=args.padel_input,
            output_csv=args.padel_csv,
            threads=args.padel_threads,
            timeout_s=args.padel_timeout_s,
        )
    if args.db is not None:
        if not args.source_table:
            raise SystemExit("--source-table is required when --db is provided.")
        smiles_values = read_unique_smiles(args.db, args.source_table, smiles_column=args.smiles_column, limit=args.limit)
        result = write_padel_feature_cache_from_smiles(
            smiles_values,
            args.out,
            fingerprint_size=args.fingerprint_size,
            morgan_radius=args.morgan_radius,
            source_table=args.source_table,
            feature_source=args.feature_source,
            batch_size=args.batch_size,
            timeout_s=args.padel_timeout_s,
            maxruntime_s=args.padel_maxruntime_s,
            threads=args.padel_threads,
            fallback_single=not args.no_fallback_single,
        )
    else:
        result = write_padel_feature_cache_from_csv(
            args.padel_csv,
            args.out,
            smiles_column=args.smiles_column,
            fingerprint_size=args.fingerprint_size,
            morgan_radius=args.morgan_radius,
            source_table=args.source_table,
            feature_source=args.feature_source,
        )
    print(
        json.dumps(
            {
                "out_path": str(result.out_path),
                "manifest_path": str(result.manifest_path),
                "descriptor_count": len(result.descriptor_names),
                "rows_read": result.rows_read,
                "rows_written": result.rows_written,
                "missing_descriptor_values": result.missing_descriptor_values,
                "nonfinite_descriptor_values": result.nonfinite_descriptor_values,
                "morgan_fingerprint_failures": result.morgan_fingerprint_failures,
                "descriptor_generation_failures": result.descriptor_generation_failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_unique_smiles(
    db_path: Path,
    source_table: str,
    *,
    smiles_column: str,
    limit: int | None,
) -> list[str]:
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT "{smiles_column}", COUNT(*) AS n
            FROM "{source_table}"
            WHERE "{smiles_column}" IS NOT NULL
              AND TRIM(CAST("{smiles_column}" AS TEXT)) <> ''
            GROUP BY "{smiles_column}"
            ORDER BY n DESC, "{smiles_column}"
            {limit_clause}
            """
        ).fetchall()
    return [str(row[0]).strip() for row in rows]


if __name__ == "__main__":
    main()
