from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DB = Path("outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite")
DEFAULT_NO_METAL_DB = Path("outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite")
DEFAULT_OUT_DIR = Path("outputs/audits/sample_attrition_20260708")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export sample attrition tables for ECOTOX target, task, QC, pTox, and no-metal/inorganic filtering."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--no-metal-db", type=Path, default=DEFAULT_NO_METAL_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.source_db) as conn, sqlite3.connect(args.no_metal_db) as no_metal:
        conn.row_factory = sqlite3.Row
        no_metal.row_factory = sqlite3.Row
        export_pipeline_stage_summary(conn, no_metal, args.out_dir)
        export_target_exclusions(conn, args.out_dir)
        export_task_exclusions(conn, args.out_dir)
        export_qc_exclusions(conn, args.out_dir)
        export_ptox_branch_flow(conn, no_metal, args.out_dir)
        export_subset_table_counts(conn, args.out_dir)
        export_no_metal_filter_summary(conn, no_metal, args.out_dir)
        export_manifest(args.source_db, args.no_metal_db, args.out_dir)
    print(f"Wrote sample attrition audit tables to {args.out_dir.resolve()}")


def export_pipeline_stage_summary(conn: sqlite3.Connection, no_metal: sqlite3.Connection, out_dir: Path) -> None:
    counts = core_counts(conn, no_metal)
    rows = [
        stage_row(1, "target_build", "Raw ECOTOX result rows after wide join", "wide_records", "result_rows", counts["wide_records"], None, None, "source", "One row per ECOTOX result after joins."),
        stage_row(2, "target_build", "Included ordinary numeric targets", "target_records", "result_rows", counts["target_included"], "Raw ECOTOX result rows after wide join", counts["wide_records"], "exclude", "Rows excluded here are not ordinary regression truths."),
        stage_row(3, "task_mapping", "Task-mappable ordinary targets", "task_records", "result_rows", counts["task_included"], "Included ordinary numeric targets", counts["target_included"], "exclude", "Endpoint/effect/target combinations not mapped to model task heads are removed here."),
        stage_row(4, "toxicity_qc", "QC-included task records", "task_records_qc", "result_rows", counts["qc_included"], "Task-mappable ordinary targets", counts["task_included"], "exclude", "Robust-z toxicity QC exclusions."),
        stage_row(5, "aggregation", "Reference-aware QC aggregate records", "aggregated_task_records_qc", "aggregate_rows", counts["aggregated_qc"], "QC-included task records", counts["qc_included"], "aggregate", "Row decrease is grouping/aggregation, not sample deletion.", represented=represented_result_rows(conn, "aggregated_task_records_qc")),
        stage_row(6, "medium_expansion", "Medium-expanded aggregate records", "aggregated_task_records_medium_domain_qc_expanded", "aggregate_rows", counts["medium_expanded"], "Reference-aware QC aggregate records", counts["aggregated_qc"], "expand", "Ambiguous medium assignments can create weighted expanded rows.", represented=None),
        stage_row(7, "study_scope", "Aquatic plus soil pTox QC aggregates", "aggregated_task_records_aquatic_soil_ptox_qc", "aggregate_rows", counts["aq_soil_ptox"], "Medium-expanded aggregate records", counts["medium_expanded"], "select_scope", "Current water-to-soil pTox modeling scope.", represented=represented_result_rows(conn, "aggregated_task_records_aquatic_soil_ptox_qc")),
        stage_row(8, "study_scope_component", "Aquatic pTox QC aggregates", "aggregated_task_records_aquatic_ptox_qc", "aggregate_rows", counts["aquatic_ptox"], "Aquatic plus soil pTox QC aggregates", counts["aq_soil_ptox"], "component", "Aquatic component of the current pTox modeling scope.", represented=represented_result_rows(conn, "aggregated_task_records_aquatic_ptox_qc")),
        stage_row(9, "study_scope_component", "Soil pTox QC aggregates", "aggregated_task_records_soil_ptox_qc", "aggregate_rows", counts["soil_ptox"], "Aquatic plus soil pTox QC aggregates", counts["aq_soil_ptox"], "component", "Soil component of the current pTox modeling scope.", represented=represented_result_rows(conn, "aggregated_task_records_soil_ptox_qc")),
        stage_row(10, "no_metal_inorganic", "No-metal/inorganic aquatic plus soil pTox aggregates", "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic", "aggregate_rows", counts["no_metal_aq_soil_ptox"], "Aquatic plus soil pTox QC aggregates", counts["aq_soil_ptox"], "exclude_chemical_class", "Exclude any CAS/DTXSID classified as inorganic or metal/metalloid.", represented=represented_result_rows(no_metal, "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic")),
        stage_row(11, "no_metal_inorganic_component", "No-metal/inorganic aquatic pTox aggregates", "aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic", "aggregate_rows", counts["no_metal_aquatic_ptox"], "Aquatic pTox QC aggregates", counts["aquatic_ptox"], "exclude_chemical_class", "Aquatic pTox after inorganic/metal/metalloid filtering.", represented=represented_result_rows(no_metal, "aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic")),
        stage_row(12, "no_metal_inorganic_component", "No-metal/inorganic soil pTox aggregates", "aggregated_task_records_soil_ptox_qc_no_metal_inorganic", "aggregate_rows", counts["no_metal_soil_ptox"], "Soil pTox QC aggregates", counts["soil_ptox"], "exclude_chemical_class", "Soil pTox after inorganic/metal/metalloid filtering.", represented=represented_result_rows(no_metal, "aggregated_task_records_soil_ptox_qc_no_metal_inorganic")),
    ]
    write_csv(out_dir / "pipeline_stage_summary.csv", rows)


def export_target_exclusions(conn: sqlite3.Connection, out_dir: Path) -> None:
    total = count_where(conn, "target_records")
    excluded = count_where(conn, "target_records", "target_status = 'excluded'")
    detail_rows = []
    for row in conn.execute(
        """
        SELECT excluded_reason, value_quality, COUNT(*) AS n
        FROM target_records
        WHERE target_status = 'excluded'
        GROUP BY excluded_reason, value_quality
        ORDER BY n DESC
        """
    ):
        n = int(row["n"])
        detail_rows.append(
            {
                "excluded_reason": row["excluded_reason"],
                "exclusion_category": target_exclusion_category(row["excluded_reason"]),
                "value_quality": row["value_quality"],
                "n": n,
                "percent_of_excluded": pct(n, excluded),
                "percent_of_all_target_records": pct(n, total),
            }
        )
    write_csv(out_dir / "target_exclusion_reason_detail.csv", detail_rows)

    grouped: dict[str, int] = {}
    for row in detail_rows:
        grouped[row["exclusion_category"]] = grouped.get(row["exclusion_category"], 0) + int(row["n"])
    write_csv(
        out_dir / "target_exclusion_reason_grouped.csv",
        [
            {
                "exclusion_category": category,
                "n": n,
                "percent_of_excluded": pct(n, excluded),
                "percent_of_all_target_records": pct(n, total),
            }
            for category, n in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        ],
    )

    status_rows = []
    for row in conn.execute(
        """
        SELECT target_status, target_name, target_family, target_basis, medium_domain, value_quality, COUNT(*) AS n
        FROM target_records
        GROUP BY target_status, target_name, target_family, target_basis, medium_domain, value_quality
        ORDER BY n DESC
        """
    ):
        payload = dict(row)
        payload["percent_of_all_target_records"] = pct(int(row["n"]), total)
        status_rows.append(payload)
    write_csv(out_dir / "target_status_by_family_medium.csv", status_rows)


def export_task_exclusions(conn: sqlite3.Connection, out_dir: Path) -> None:
    total = count_where(conn, "task_records")
    excluded = count_where(conn, "task_records", "task_status = 'excluded'")
    rows = []
    for row in conn.execute(
        """
        SELECT task_excluded_reason, COUNT(*) AS n
        FROM task_records
        WHERE task_status = 'excluded'
        GROUP BY task_excluded_reason
        ORDER BY n DESC
        """
    ):
        n = int(row["n"])
        rows.append(
            {
                "task_excluded_reason": row["task_excluded_reason"],
                "exclusion_category": task_exclusion_category(row["task_excluded_reason"]),
                "n": n,
                "percent_of_excluded_task_records": pct(n, excluded),
                "percent_of_task_records": pct(n, total),
            }
        )
    write_csv(out_dir / "task_exclusion_reason_summary.csv", rows)

    grouped: dict[str, int] = {}
    for row in rows:
        grouped[row["exclusion_category"]] = grouped.get(row["exclusion_category"], 0) + int(row["n"])
    write_csv(
        out_dir / "task_exclusion_reason_grouped.csv",
        [
            {
                "exclusion_category": category,
                "n": n,
                "percent_of_excluded_task_records": pct(n, excluded),
                "percent_of_task_records": pct(n, total),
            }
            for category, n in sorted(grouped.items(), key=lambda item: item[1], reverse=True)
        ],
    )


def export_qc_exclusions(conn: sqlite3.Connection, out_dir: Path) -> None:
    total = count_where(conn, "task_records_qc")
    excluded = count_where(conn, "task_records_qc", "tox_qc_status = 'excluded'")
    rows = []
    for row in conn.execute(
        """
        SELECT tox_qc_reason, target_name, medium_domain, task_head, COUNT(*) AS n
        FROM task_records_qc
        WHERE tox_qc_status = 'excluded'
        GROUP BY tox_qc_reason, target_name, medium_domain, task_head
        ORDER BY n DESC
        """
    ):
        n = int(row["n"])
        payload = dict(row)
        payload["percent_of_qc_excluded"] = pct(n, excluded)
        payload["percent_of_qc_source"] = pct(n, total)
        rows.append(payload)
    write_csv(out_dir / "qc_exclusion_reason_by_target_medium_task.csv", rows)

    reason_rows = []
    for row in conn.execute(
        """
        SELECT tox_qc_reason, COUNT(*) AS n
        FROM task_records_qc
        WHERE tox_qc_status = 'excluded'
        GROUP BY tox_qc_reason
        ORDER BY n DESC
        """
    ):
        n = int(row["n"])
        reason_rows.append(
            {
                "tox_qc_reason": row["tox_qc_reason"],
                "n": n,
                "percent_of_qc_excluded": pct(n, excluded),
                "percent_of_qc_source": pct(n, total),
            }
        )
    write_csv(out_dir / "qc_exclusion_reason_summary.csv", reason_rows)


def export_ptox_branch_flow(conn: sqlite3.Connection, no_metal: sqlite3.Connection, out_dir: Path) -> None:
    rows = []
    definitions = [
        (
            1,
            "target_records included all ordinary numeric targets",
            "target_records",
            "target_status = 'included'",
            None,
            "result_rows",
            "All target-construction included rows before task mapping.",
        ),
        (
            2,
            "target_records included aquatic/soil pTox targets",
            "target_records",
            "target_status = 'included' AND target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'soil')",
            None,
            "result_rows",
            "Target-level pTox rows assigned directly to aquatic or soil medium domains.",
        ),
        (
            3,
            "task_records included aquatic/soil pTox targets",
            "task_records",
            "task_status = 'included' AND target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'soil')",
            None,
            "result_rows",
            "Rows retained after endpoint/effect task mapping.",
        ),
        (
            4,
            "task_records_qc included aquatic/soil pTox targets",
            "task_records_qc",
            "tox_qc_status = 'included' AND target_name = 'ptox_mol_l' AND medium_domain IN ('aquatic', 'soil')",
            None,
            "result_rows",
            "Rows retained after robust-z toxicity QC.",
        ),
        (
            5,
            "aggregated aquatic/soil pTox QC records",
            "aggregated_task_records_aquatic_soil_ptox_qc",
            None,
            None,
            "aggregate_rows",
            "Aggregate rows after reference-aware aggregation and medium-domain expansion; represented_result_rows can count medium-expanded records.",
        ),
        (
            6,
            "no-metal/inorganic aggregated aquatic/soil pTox records",
            "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic",
            None,
            no_metal,
            "aggregate_rows",
            "Final current modeling table after excluding inorganic and metal/metalloid chemicals.",
        ),
    ]
    previous = None
    for order, label, table, where, db_override, row_unit, note in definitions:
        db = db_override or conn
        n = count_where(db, table, where)
        rows.append(
            {
                "stage_order": order,
                "stage": label,
                "table_name": table,
                "row_unit": row_unit,
                "rows": n,
                "represented_result_rows": represented_result_rows(db, table),
                "previous_rows": previous,
                "delta_from_previous": none_if(previous is None, n - previous if previous is not None else None),
                "retained_percent_from_previous": none_if(previous is None, pct(n, previous) if previous else None),
                "note": note,
            }
        )
        previous = n
    write_csv(out_dir / "ptox_branch_flow.csv", rows)

    medium_rows = []
    for table, db in [
        ("aggregated_task_records_aquatic_soil_ptox_qc", conn),
        ("aggregated_task_records_aquatic_ptox_qc", conn),
        ("aggregated_task_records_soil_ptox_qc", conn),
        ("aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic", no_metal),
        ("aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic", no_metal),
        ("aggregated_task_records_soil_ptox_qc_no_metal_inorganic", no_metal),
    ]:
        for row in db.execute(
            f"""
            SELECT medium_domain, target_name, target_family, COUNT(*) AS n,
                   SUM(CAST(target_value_count AS REAL)) AS represented_result_rows
            FROM "{table}"
            GROUP BY medium_domain, target_name, target_family
            ORDER BY n DESC
            """
        ):
            payload = dict(row)
            payload["table_name"] = table
            medium_rows.append(payload)
    write_csv(out_dir / "ptox_aggregate_counts_by_medium.csv", medium_rows)


def export_subset_table_counts(conn: sqlite3.Connection, out_dir: Path) -> None:
    rows = []
    table_names = [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name LIKE 'aggregated_task_records%_qc'
            ORDER BY name
            """
        )
    ]
    for table in table_names:
        rows.append(
            {
                "table_name": table,
                "rows": count_where(conn, table),
                "represented_result_rows": represented_result_rows(conn, table),
            }
        )
    write_csv(out_dir / "aggregated_qc_subset_table_counts.csv", rows)


def export_no_metal_filter_summary(conn: sqlite3.Connection, no_metal: sqlite3.Connection, out_dir: Path) -> None:
    source_to_output = [
        ("target_records", "target_records"),
        ("aggregated_task_records_aquatic_soil_ptox_qc", "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"),
        ("aggregated_task_records_aquatic_ptox_qc", "aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic"),
        ("aggregated_task_records_soil_ptox_qc", "aggregated_task_records_soil_ptox_qc_no_metal_inorganic"),
    ]
    rows = []
    for source_table, output_table in source_to_output:
        input_rows = count_where(conn, source_table)
        output_rows = count_where(no_metal, output_table)
        excluded_rows = input_rows - output_rows
        rows.append(
            {
                "source_table": source_table,
                "output_table": output_table,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "excluded_rows": excluded_rows,
                "retained_percent": pct(output_rows, input_rows),
                "excluded_percent": pct(excluded_rows, input_rows),
                "input_represented_result_rows": represented_result_rows(conn, source_table),
                "output_represented_result_rows": represented_result_rows(no_metal, output_table),
            }
        )
    write_csv(out_dir / "no_metal_inorganic_filter_summary.csv", rows)

    manifest_rows = []
    if table_exists(no_metal, "no_metal_inorganic_build_manifest"):
        for row in no_metal.execute('SELECT "key", "value" FROM no_metal_inorganic_build_manifest ORDER BY "key"'):
            manifest_rows.append({"key": row["key"], "value": row["value"]})
    write_csv(out_dir / "no_metal_inorganic_manifest.csv", manifest_rows)

    key_rows = []
    if table_exists(no_metal, "excluded_metal_inorganic_chemicals"):
        for row in no_metal.execute(
            """
            SELECT
                CASE
                    WHEN has_inorganic = 1 AND has_metal_metalloid = 1 THEN 'inorganic_and_metal_metalloid'
                    WHEN has_inorganic = 1 THEN 'inorganic'
                    WHEN has_metal_metalloid = 1 THEN 'metal_metalloid'
                    ELSE 'unknown'
                END AS exclusion_class,
                COUNT(*) AS chemical_keys,
                COUNT(DISTINCT cas_number) AS cas_count,
                COUNT(DISTINCT dtxsid) AS dtxsid_count,
                SUM(source_rows) AS source_target_records
            FROM excluded_metal_inorganic_chemicals
            GROUP BY exclusion_class
            ORDER BY chemical_keys DESC
            """
        ):
            key_rows.append(dict(row))
    write_csv(out_dir / "no_metal_inorganic_excluded_chemical_keys.csv", key_rows)


def export_manifest(source_db: Path, no_metal_db: Path, out_dir: Path) -> None:
    manifest = {
        "source_db": str(source_db),
        "no_metal_db": str(no_metal_db),
        "outputs": [
            "pipeline_stage_summary.csv",
            "target_exclusion_reason_detail.csv",
            "target_exclusion_reason_grouped.csv",
            "target_status_by_family_medium.csv",
            "task_exclusion_reason_summary.csv",
            "task_exclusion_reason_grouped.csv",
            "qc_exclusion_reason_summary.csv",
            "qc_exclusion_reason_by_target_medium_task.csv",
            "ptox_branch_flow.csv",
            "ptox_aggregate_counts_by_medium.csv",
            "aggregated_qc_subset_table_counts.csv",
            "no_metal_inorganic_filter_summary.csv",
            "no_metal_inorganic_manifest.csv",
            "no_metal_inorganic_excluded_chemical_keys.csv",
        ],
        "notes": [
            "Rows before aggregation are result-level records.",
            "Rows after aggregation are aggregate rows; represented_result_rows gives the summed target_value_count when available.",
            "Aggregation row-count decreases are collapses of replicate/reference/test records, not exclusions.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# Sample Attrition Audit",
        "",
        "This audit exports row-count and reason tables for target construction, task mapping, toxicity QC, pTox scope selection, and no-metal/inorganic filtering.",
        "",
        "Important interpretation: result-level row counts and aggregate row counts are not interchangeable. Aggregation reduces row counts by grouping records and should not be interpreted as deletion.",
        "",
        "Primary table: `pipeline_stage_summary.csv`.",
    ]
    (out_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def core_counts(conn: sqlite3.Connection, no_metal: sqlite3.Connection) -> dict[str, int]:
    return {
        "wide_records": count_where(conn, "wide_records") if table_exists(conn, "wide_records") else count_where(conn, "target_records"),
        "target_included": count_where(conn, "target_records", "target_status = 'included'"),
        "task_included": count_where(conn, "task_records", "task_status = 'included'"),
        "qc_included": count_where(conn, "task_records_qc", "tox_qc_status = 'included'"),
        "aggregated_qc": count_where(conn, "aggregated_task_records_qc"),
        "medium_expanded": count_where(conn, "aggregated_task_records_medium_domain_qc_expanded"),
        "aq_soil_ptox": count_where(conn, "aggregated_task_records_aquatic_soil_ptox_qc"),
        "aquatic_ptox": count_where(conn, "aggregated_task_records_aquatic_ptox_qc"),
        "soil_ptox": count_where(conn, "aggregated_task_records_soil_ptox_qc"),
        "no_metal_aq_soil_ptox": count_where(no_metal, "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"),
        "no_metal_aquatic_ptox": count_where(no_metal, "aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic"),
        "no_metal_soil_ptox": count_where(no_metal, "aggregated_task_records_soil_ptox_qc_no_metal_inorganic"),
    }


def stage_row(
    stage_id: int,
    stage_group: str,
    stage: str,
    table_name: str,
    row_unit: str,
    rows: int,
    basis_stage: str | None,
    basis_rows: int | None,
    operation_type: str,
    note: str,
    represented: int | float | None = None,
) -> dict[str, Any]:
    delta = None if basis_rows is None else rows - basis_rows
    return {
        "stage_id": stage_id,
        "stage_group": stage_group,
        "stage": stage,
        "table_name": table_name,
        "row_unit": row_unit,
        "rows": rows,
        "represented_result_rows": represented,
        "basis_stage": basis_stage,
        "basis_rows": basis_rows,
        "delta_rows": delta,
        "retained_percent_vs_basis": None if basis_rows is None else pct(rows, basis_rows),
        "operation_type": operation_type,
        "note": note,
    }


def count_where(conn: sqlite3.Connection, table: str, where: str | None = None) -> int:
    sql = f'SELECT COUNT(*) FROM "{table}"'
    if where:
        sql += f" WHERE {where}"
    value = conn.execute(sql).fetchone()[0]
    return int(value or 0)


def represented_result_rows(conn: sqlite3.Connection, table: str) -> int | float | None:
    if not table_exists(conn, table):
        return None
    columns = table_columns(conn, table)
    if "target_value_count" not in columns:
        return None
    value = conn.execute(f'SELECT SUM(CAST(target_value_count AS REAL)) FROM "{table}"').fetchone()[0]
    if value is None:
        return None
    if abs(float(value) - int(float(value))) < 1e-9:
        return int(float(value))
    return float(value)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int | float, denominator: int | float | None) -> float | None:
    if denominator is None or float(denominator) == 0.0:
        return None
    return round(float(numerator) / float(denominator) * 100.0, 6)


def none_if(condition: bool, value: Any) -> Any:
    return None if condition else value


def target_exclusion_category(reason: object) -> str:
    text = str(reason or "")
    if text.startswith("unsupported_unit_family"):
        return "missing_or_unsupported_unit"
    if "censored" in text:
        return "censored_value_not_plain_regression_truth"
    if "molecular_weight" in text:
        return "missing_or_invalid_molecular_weight"
    if "missing_mean" in text:
        return "missing_effective_statistic"
    if "non_positive" in text or "non-positive" in text:
        return "non_positive_toxicity_value"
    return text or "unknown"


def task_exclusion_category(reason: object) -> str:
    text = str(reason or "")
    if text.startswith("excluded_endpoint:NR"):
        return "endpoint_NR_excluded"
    if text == "excluded_oral_target":
        return "oral_target_excluded_from_main_task_mapping"
    if text == "bioaccumulation_endpoint_requires_factor_target":
        return "bioaccumulation_requires_factor_target"
    if text.startswith("unsupported_endpoint:") or text.startswith("excluded_endpoint:"):
        return "unsupported_or_excluded_endpoint"
    if text == "unsupported_effect_family":
        return "unsupported_effect_family"
    return text or "unknown"


if __name__ == "__main__":
    main()
