from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsar_tl.training.censored_loss import censored_audit_summary, censor_operator, censored_direction


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit censored ECOTOX concentration records.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="target_records")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--examples", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows, censored = load_censored_rows(Path(args.db), args.table)
    summary_rows = censored_audit_summary(censored)
    example_rows = build_examples(censored, limit=max(0, int(args.examples)))

    write_csv(out_dir / "censored_audit_summary.csv", summary_rows)
    write_csv(out_dir / "censored_audit_examples.csv", example_rows)
    manifest = {
        "db": str(args.db),
        "table": args.table,
        "rows": total_rows,
        "censored_rows": len(censored),
        "summary_rows": len(summary_rows),
        "example_rows": len(example_rows),
    }
    (out_dir / "censored_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def load_censored_rows(db_path: Path, table: str) -> tuple[int, list[dict[str, Any]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        total = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        rows = conn.execute(
            f"""
            SELECT *
            FROM "{table}"
            WHERE value_quality LIKE 'censored%'
               OR conc1_mean_op IN ('<','>','<=','>=')
               OR conc1_min_op IN ('<','>','<=','>=')
               OR conc1_max_op IN ('<','>','<=','>=')
            """
        ).fetchall()
    return total, [dict(row) for row in rows]


def build_examples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows[:limit]:
        operator = censor_operator(row)
        examples.append(
            {
                "result_id": row.get("result_id", ""),
                "cas_number": row.get("cas_number", ""),
                "chemical_name": row.get("chemical_name", ""),
                "species_number": row.get("species_number", ""),
                "latin_name": row.get("latin_name", ""),
                "endpoint": row.get("endpoint", ""),
                "effect": row.get("effect", ""),
                "measurement": row.get("measurement", ""),
                "operator": operator,
                "censored_direction": censored_direction(operator),
                "tox_value": row.get("tox_value", ""),
                "value_quality": row.get("value_quality", ""),
                "unit_family_v2": row.get("unit_family_v2", ""),
                "standard_unit_v2": row.get("standard_unit_v2", ""),
                "target_status": row.get("target_status", ""),
                "excluded_reason": row.get("excluded_reason", ""),
                "medium_domain": row.get("medium_domain", ""),
            }
        )
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["n"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
