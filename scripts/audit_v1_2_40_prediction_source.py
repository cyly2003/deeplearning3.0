from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.baseline import stage_sample_record_id


DEFAULT_SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
COMPARE_FIELDS = (
    "aggregate_id",
    "task_head",
    "medium_domain",
    "target_name",
    "target_family",
    "result_ids",
    "target_value_median",
    "molecular_weight_g_mol_used",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a completed v1.2.40 predictions.csv with the current source "
            "table, using strict stage-sample identities."
        )
    )
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_prediction_source(
        db_path=args.db,
        predictions_path=args.predictions,
        source_table=args.source_table,
        max_examples=args.max_examples,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def audit_prediction_source(
    *,
    db_path: Path,
    predictions_path: Path,
    source_table: str = DEFAULT_SOURCE_TABLE,
    max_examples: int = 20,
) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    source_by_record: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = f'''SELECT
                aggregate_id, task_head, medium_domain, target_name, target_family,
                target_value_median, molecular_weight_g_mol_used, result_ids
            FROM "{source_table}"'''
        for row in conn.execute(query):
            record_id = stage_sample_record_id(
                row["aggregate_id"],
                row["medium_domain"],
                row["target_name"],
                row["target_family"],
            )
            if record_id in source_by_record:
                raise ValueError(f"Duplicate source stage-sample identity: {record_id}")
            source_by_record[record_id] = {
                "aggregate_id": clean_text(row["aggregate_id"]),
                "task_head": clean_text(row["task_head"]),
                "medium_domain": clean_text(row["medium_domain"]),
                "target_name": clean_text(row["target_name"]),
                "target_family": clean_text(row["target_family"]),
                "result_ids": normalize_result_ids(row["result_ids"]),
                "target_value_median": clean_float(row["target_value_median"]),
                "molecular_weight_g_mol_used": clean_float(
                    row["molecular_weight_g_mol_used"]
                ),
            }

    prediction_rows = 0
    missing_source = 0
    duplicate_prediction = 0
    exact_rows = 0
    mismatch_fields: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    seen_prediction: set[str] = set()
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "aggregate_id",
            "task_head",
            "medium_domain",
            "target_name",
            "target_family",
            "result_ids",
            "molecular_weight_g_mol_used",
            "y_true",
            "split_part",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Predictions missing columns: {sorted(missing_columns)}")
        for row in reader:
            prediction_rows += 1
            split_counts[clean_text(row["split_part"])] += 1
            record_id = stage_sample_record_id(
                row["aggregate_id"],
                row["medium_domain"],
                row["target_name"],
                row["target_family"],
            )
            if record_id in seen_prediction:
                duplicate_prediction += 1
            seen_prediction.add(record_id)
            source = source_by_record.get(record_id)
            if source is None:
                missing_source += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "record_id": record_id,
                            "issue": "missing_source",
                            "prediction_aggregate_id": clean_text(row["aggregate_id"]),
                        }
                    )
                continue
            prediction = {
                "aggregate_id": clean_text(row["aggregate_id"]),
                "task_head": clean_text(row["task_head"]),
                "medium_domain": clean_text(row["medium_domain"]),
                "target_name": clean_text(row["target_name"]),
                "target_family": clean_text(row["target_family"]),
                "result_ids": normalize_result_ids(row["result_ids"]),
                "target_value_median": clean_float(row["y_true"]),
                "molecular_weight_g_mol_used": clean_float(
                    row["molecular_weight_g_mol_used"]
                ),
            }
            row_mismatches: dict[str, dict[str, Any]] = {}
            for field in COMPARE_FIELDS:
                if not equal_value(prediction[field], source[field]):
                    mismatch_fields[field] += 1
                    row_mismatches[field] = {
                        "prediction": prediction[field],
                        "source": source[field],
                    }
            if row_mismatches:
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "record_id": record_id,
                            "issue": "field_mismatch",
                            "fields": row_mismatches,
                        }
                    )
            else:
                exact_rows += 1

    return {
        "status": "match" if not mismatch_fields and not missing_source else "mismatch",
        "db": str(db_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "source_table": source_table,
        "source_rows": len(source_by_record),
        "prediction_rows": prediction_rows,
        "prediction_split_counts": dict(sorted(split_counts.items())),
        "unique_prediction_record_ids": len(seen_prediction),
        "duplicate_prediction_record_ids": duplicate_prediction,
        "exact_rows": exact_rows,
        "missing_source_rows": missing_source,
        "mismatch_fields": dict(sorted(mismatch_fields.items())),
        "examples": examples,
        "interpretation": (
            "A zero-mismatch result proves that the current source values retained in "
            "predictions.csv are the same values used by the completed v1.2.40 run."
        ),
    }


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_result_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"result_ids is not an array: {value!r}")
    return sorted(str(item).strip() for item in value if str(item).strip())


def equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            return left is right
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


if __name__ == "__main__":
    main()
