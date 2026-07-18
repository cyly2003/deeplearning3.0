"""Audit the experimental-medium evidence of soil-domain pTox model records.

This script deliberately starts from every raw ECOTOX result ID retained by
the current soil pTox QC modeling table.  It does *not* infer exposure medium
from pTox itself: pTox is only a mol/L target scale.  The output therefore
keeps a traceable chain from model aggregate -> result_id -> test_id ->
reference_number / DOI, while classifying the original ECOTOX ``media_type``
code separately from the project medium-domain routing.

The classification is an evidence audit, not a re-labelling of ECOTOX.  In
particular, an aqueous unit (mg/L or mol/L) does not establish that a record
is pore water, dosing solution, leachate, or bulk-soil exposure.  Such labels
require explicit test-method evidence, which is usually absent from the
minimal ECOTOX flat-file fields retained in this project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_TABLE = "aggregated_task_records_soil_ptox_qc_no_metal_inorganic"

# The official ECOTOX search planner lists these as soil exposure media.  We
# retain soil-like organic substrates separately because they are not
# automatically equivalent to a bulk mineral/natural soil exposure.
DIRECT_SOIL_MATRIX_CODES = {"NAT", "ART", "UKS", "MIN"}
SOIL_RELATED_SUBSTRATE_CODES = {"HUM", "MAN", "LIT"}
AQUEOUS_CODES = {"AQU", "HYP", "FW", "SW"}
CULTURE_CODES = {"CUL", "AGR"}  # CUL = culture; AGR = agar
NONSOIL_SUBSTRATE_CODES = {"FLT", "NONE", "FAB", "POP"}
SEDIMENT_OR_SLUDGE_CODES = {"SED", "SLG"}
MISSING_OR_UNSPECIFIED_CODES = {"", "--", "NC", "NR", "UKN"}

TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "explicit_porewater_or_soil_solution": re.compile(
        r"\b(?:pore\s*water|porewater|soil\s*solution|interstitial\s+water)\b", re.I
    ),
    "extract_leachate_or_elutriate": re.compile(
        r"\b(?:extract|leachate|elutriate|leaching)\b", re.I
    ),
    "hydroponic": re.compile(r"\bhydroponic\b", re.I),
    "aquatic": re.compile(r"\baquatic\b", re.I),
    "soil": re.compile(r"\bsoil\b", re.I),
}


@dataclass(frozen=True)
class EvidenceClass:
    code: str
    label: str
    interpretation: str
    keep_for_soil_matrix_main_analysis: str


EVIDENCE_CLASSES: dict[str, EvidenceClass] = {
    "verified_soil_matrix": EvidenceClass(
        "verified_soil_matrix",
        "ECOTOX direct soil-matrix code",
        "The original media_type is natural, artificial, unspecified, or mineral soil.",
        "yes",
    ),
    "soil_related_substrate": EvidenceClass(
        "soil_related_substrate",
        "ECOTOX direct soil-related substrate code",
        "The original media_type is humus, manure, soil mixture, or litter; it is soil-related but should not be treated as interchangeable with standard bulk soil.",
        "sensitivity_or_separate",
    ),
    "aqueous_or_hydroponic_nonsoil": EvidenceClass(
        "aqueous_or_hydroponic_nonsoil",
        "Aqueous or hydroponic medium",
        "AQU/HYP/FW/SW is aqueous or hydroponic exposure, not a soil matrix.  A soil-associated organism label cannot change this into a soil exposure.",
        "no",
    ),
    "culture_or_agar_nonsoil": EvidenceClass(
        "culture_or_agar_nonsoil",
        "Culture or agar medium",
        "CUL/AGR denotes culture or agar medium, not a direct soil matrix code.",
        "no",
    ),
    "nonsoil_substrate": EvidenceClass(
        "nonsoil_substrate",
        "Non-soil substrate",
        "Filter paper, no substrate, fabric, or plaster of Paris is not a soil matrix.",
        "no",
    ),
    "sediment_or_sludge": EvidenceClass(
        "sediment_or_sludge",
        "Sediment or sludge matrix",
        "Sediment/sludge is a distinct solid matrix and should not be pooled with soil without an explicit study design.",
        "no",
    ),
    "media_mixture_unresolved": EvidenceClass(
        "media_mixture_unresolved",
        "Media mixture with no retained composition",
        "MIX denotes a media mixture with a comment; the retained flat-file fields do not establish that the mixture is soil.",
        "no_until_method_verified",
    ),
    "no_reported_medium": EvidenceClass(
        "no_reported_medium",
        "No usable direct media code",
        "The original media_type is missing, not reported, or unknown.  Organism habitat alone does not establish the test matrix.",
        "no_until_method_verified",
    ),
    "other_unresolved": EvidenceClass(
        "other_unresolved",
        "Other unresolved medium code",
        "The raw medium code is not one of the audited soil, hydroponic, culture, or aqueous categories and needs method-level review.",
        "no_until_method_verified",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-db",
        type=Path,
        default=Path("outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite"),
    )
    parser.add_argument("--raw-db", type=Path, default=Path("ecotox_clean.sqlite"))
    parser.add_argument("--model-table", default=DEFAULT_MODEL_TABLE)
    parser.add_argument(
        "--prediction-csv",
        type=Path,
        default=Path(
            "outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/"
            "split_policy_ensemble_random_8_2_holdout_prediction_rows.csv"
        ),
        help="Optional current-mainline prediction rows used only to quantify test coverage.",
    )
    parser.add_argument(
        "--mainline-summary-csv",
        type=Path,
        default=Path(
            "outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/"
            "split_policy_ensemble_combined_summary.csv"
        ),
        help="Summary used to identify the 29 task heads contributing to the reported random 8:2 metric.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audits/soil_ptox_medium_evidence_20260718"),
    )
    return parser.parse_args()


def media_code(value: Any) -> str:
    return str(value or "").strip().upper().rstrip("/*")


def classify_media(code: str) -> EvidenceClass:
    if code in DIRECT_SOIL_MATRIX_CODES:
        return EVIDENCE_CLASSES["verified_soil_matrix"]
    if code in SOIL_RELATED_SUBSTRATE_CODES:
        return EVIDENCE_CLASSES["soil_related_substrate"]
    if code == "MIX":
        return EVIDENCE_CLASSES["media_mixture_unresolved"]
    if code in AQUEOUS_CODES:
        return EVIDENCE_CLASSES["aqueous_or_hydroponic_nonsoil"]
    if code in CULTURE_CODES:
        return EVIDENCE_CLASSES["culture_or_agar_nonsoil"]
    if code in NONSOIL_SUBSTRATE_CODES:
        return EVIDENCE_CLASSES["nonsoil_substrate"]
    if code in SEDIMENT_OR_SLUDGE_CODES:
        return EVIDENCE_CLASSES["sediment_or_sludge"]
    if code in MISSING_OR_UNSPECIFIED_CODES:
        return EVIDENCE_CLASSES["no_reported_medium"]
    return EVIDENCE_CLASSES["other_unresolved"]


def title_flags(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts)
    return ";".join(name for name, pattern in TEXT_PATTERNS.items() if pattern.search(text))


def exposure_semantic_evidence(flags: str) -> str:
    """Return only evidence that identifies the *concentration basis*.

    Generic occurrences of words such as ``extract`` or ``soil`` in a paper
    title are not enough: they may describe the test chemical or background
    rather than the exposure matrix.  The deliberately narrow pore-water tag
    is therefore used as a lower-bound count of confirmed pore-water records.
    """

    return (
        "explicit_porewater_in_reference_title"
        if "explicit_porewater_or_soil_solution" in flags.split(";")
        else "not_identifiable_from_retained_record_fields"
    )


def chunked(items: Iterable[int], size: int = 900) -> Iterable[list[int]]:
    batch: list[int] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_rows(conn: sqlite3.Connection, query: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(query, parameters)]


def load_model_aggregates(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    return read_rows(conn, f'SELECT * FROM "{table_name}" ORDER BY CAST(aggregate_id AS INTEGER)')


def load_derived_targets(conn: sqlite3.Connection, result_ids: set[int]) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    for batch in chunked(sorted(result_ids)):
        placeholders = ",".join("?" for _ in batch)
        query = f"""
            SELECT * FROM target_records
            WHERE result_id IN ({placeholders})
        """
        for row in read_rows(conn, query, tuple(batch)):
            values[int(row["result_id"])] = row
    return values


def load_raw_records(raw_db: Path, result_ids: set[int]) -> dict[int, dict[str, Any]]:
    values: dict[int, dict[str, Any]] = {}
    with sqlite3.connect(raw_db) as conn:
        conn.row_factory = sqlite3.Row
        for batch in chunked(sorted(result_ids)):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
                SELECT
                    r.result_id, r.test_id, r.conc1_type, r.conc1_mean_op, r.conc1_mean,
                    r.conc1_unit, r.conc1_min_op, r.conc1_min, r.conc1_max_op,
                    r.conc1_max, r.endpoint, r.endpoint_comments, r.effect,
                    r.measurement, r.response_site_comments, r.obs_duration_mean,
                    r.obs_duration_unit,
                    t.reference_number, t.media_type AS raw_media_type,
                    t.organism_habitat AS raw_organism_habitat,
                    t.organism_lifestage AS raw_organism_lifestage,
                    t.exposure_duration_mean, t.exposure_duration_unit,
                    ref.author, ref.title, ref.source, ref.publication_year, ref.doi
                FROM results AS r
                JOIN tests AS t ON r.test_id = t.test_id
                LEFT JOIN "references" AS ref ON t.reference_number = ref.reference_number
                WHERE r.result_id IN ({placeholders})
            """
            for row in conn.execute(query, batch):
                values[int(row["result_id"])] = dict(row)
    return values


def reported_task_heads(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        return set()
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split_policy", "")) == "random_8_2":
                return {item for item in str(row.get("tasks", "")).split(";") if item}
    return set()


def prediction_test_counts(prediction_csv: Path, included_tasks: set[str]) -> tuple[Counter[int], Counter[int]]:
    all_counts: Counter[int] = Counter()
    reported_counts: Counter[int] = Counter()
    if not prediction_csv.exists():
        return all_counts, reported_counts
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split_part", "")).lower() != "test":
                continue
            try:
                aggregate_id = int(str(row["aggregate_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            all_counts[aggregate_id] += 1
            if not included_tasks or str(row.get("task_head", "")) in included_tasks:
                reported_counts[aggregate_id] += 1
    return all_counts, reported_counts


def prediction_metrics_by_evidence(
    prediction_csv: Path,
    aggregate_evidence: dict[int, dict[str, Any]],
    included_tasks: set[str],
) -> list[dict[str, Any]]:
    """Recompute ensemble metrics after stratifying the exact prediction rows."""

    buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    if not prediction_csv.exists():
        return []
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split_part", "")).lower() != "test":
                continue
            try:
                aggregate_id = int(str(row["aggregate_id"]))
                y_true = float(str(row["y_true"]))
                y_pred = float(str(row["y_pred"]))
            except (KeyError, TypeError, ValueError):
                continue
            evidence = aggregate_evidence.get(aggregate_id)
            if evidence is None or not (math.isfinite(y_true) and math.isfinite(y_pred)):
                continue
            scopes = ["all_prediction_rows"]
            if not included_tasks or str(row.get("task_head", "")) in included_tasks:
                scopes.append("reported_main_metric_rows")
            for scope in scopes:
                buckets[(scope, str(evidence["evidence_class"]))].append((y_true, y_pred))
                buckets[(scope, "overall")].append((y_true, y_pred))

    rows: list[dict[str, Any]] = []
    for (scope, evidence_class), values in sorted(buckets.items()):
        y_true = [item[0] for item in values]
        y_pred = [item[1] for item in values]
        n = len(values)
        mean_true = sum(y_true) / n
        ss_res = sum((actual - predicted) ** 2 for actual, predicted in values)
        ss_tot = sum((actual - mean_true) ** 2 for actual in y_true)
        rows.append(
            {
                "scope": scope,
                "evidence_class": evidence_class,
                "n": n,
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else "",
                "rmse": math.sqrt(ss_res / n),
                "mae": sum(abs(actual - predicted) for actual, predicted in values) / n,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.derived_db.exists():
        raise FileNotFoundError(args.derived_db)
    if not args.raw_db.exists():
        raise FileNotFoundError(args.raw_db)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.derived_db) as derived_conn:
        aggregates = load_model_aggregates(derived_conn, args.model_table)
        aggregate_result_ids: dict[int, list[int]] = {}
        all_result_ids: set[int] = set()
        for aggregate in aggregates:
            ids = [int(value) for value in json.loads(str(aggregate["result_ids"]))]
            aggregate_result_ids[int(aggregate["aggregate_id"])] = ids
            all_result_ids.update(ids)
        derived_targets = load_derived_targets(derived_conn, all_result_ids)

    raw_records = load_raw_records(args.raw_db, all_result_ids)
    metric_task_heads = reported_task_heads(args.mainline_summary_csv)
    test_prediction_counts, reported_metric_counts = prediction_test_counts(args.prediction_csv, metric_task_heads)

    raw_output: list[dict[str, Any]] = []
    aggregate_output: list[dict[str, Any]] = []
    aggregate_by_id: dict[int, dict[str, Any]] = {}

    for aggregate in aggregates:
        aggregate_id = int(aggregate["aggregate_id"])
        code = media_code(aggregate.get("media_type"))
        evidence = classify_media(code)
        result_ids = aggregate_result_ids[aggregate_id]
        raw_evidence_classes: set[str] = set()
        raw_codes: set[str] = set()
        raw_test_ids: set[str] = set()
        raw_reference_ids: set[str] = set()
        raw_flags: set[str] = set()
        raw_porewater_result_count = 0
        for result_id in result_ids:
            derived = derived_targets.get(result_id, {})
            raw = raw_records.get(result_id, {})
            raw_code = media_code(raw.get("raw_media_type", derived.get("media_type")))
            raw_evidence = classify_media(raw_code)
            raw_evidence_classes.add(raw_evidence.code)
            raw_codes.add(raw_code)
            raw_test_ids.add(str(raw.get("test_id", derived.get("test_id", ""))))
            raw_reference_ids.add(str(raw.get("reference_number", derived.get("reference_number", ""))))
            flags = title_flags(
                raw.get("title"), raw.get("source"), raw.get("endpoint_comments"), raw.get("response_site_comments")
            )
            raw_flags.update(flag for flag in flags.split(";") if flag)
            semantic_evidence = exposure_semantic_evidence(flags)
            raw_porewater_result_count += int(semantic_evidence == "explicit_porewater_in_reference_title")
            raw_output.append(
                {
                    "aggregate_id": aggregate_id,
                    "model_media_type": aggregate.get("media_type"),
                    "model_primary_medium_domain": aggregate.get("primary_medium_domain"),
                    "model_medium_domains": aggregate.get("medium_domains"),
                    "model_medium_domain_reason": aggregate.get("medium_domain_reason"),
                    "evidence_class": raw_evidence.code,
                    "evidence_label": raw_evidence.label,
                    "main_analysis_recommendation": raw_evidence.keep_for_soil_matrix_main_analysis,
                    "raw_media_code": raw_code,
                    "result_id": result_id,
                    "test_id": raw.get("test_id", derived.get("test_id")),
                    "reference_number": raw.get("reference_number", derived.get("reference_number")),
                    "doi": raw.get("doi"),
                    "reference_title": raw.get("title"),
                    "reference_source": raw.get("source"),
                    "reference_year": raw.get("publication_year"),
                    "raw_organism_habitat": raw.get("raw_organism_habitat", derived.get("organism_habitat")),
                    "raw_conc1_type": raw.get("conc1_type", derived.get("conc1_type")),
                    "raw_conc1_mean_op": raw.get("conc1_mean_op", derived.get("conc1_mean_op")),
                    "raw_conc1_mean": raw.get("conc1_mean", derived.get("conc1_mean")),
                    "raw_conc1_unit": raw.get("conc1_unit", derived.get("conc1_unit")),
                    "raw_conc1_min_op": raw.get("conc1_min_op", derived.get("conc1_min_op")),
                    "raw_conc1_min": raw.get("conc1_min", derived.get("conc1_min")),
                    "raw_conc1_max_op": raw.get("conc1_max_op", derived.get("conc1_max_op")),
                    "raw_conc1_max": raw.get("conc1_max", derived.get("conc1_max")),
                    "raw_endpoint": raw.get("endpoint", derived.get("endpoint")),
                    "raw_effect": raw.get("effect", derived.get("effect")),
                    "raw_measurement": raw.get("measurement", derived.get("measurement")),
                    "raw_endpoint_comments": raw.get("endpoint_comments"),
                    "raw_response_site_comments": raw.get("response_site_comments"),
                    "unit_family_v2": derived.get("unit_family_v2"),
                    "standard_unit_v2": derived.get("standard_unit_v2"),
                    "conversion_path": derived.get("conversion_path"),
                    "target_name": derived.get("target_name"),
                    "target_basis": derived.get("target_basis"),
                    "text_evidence_flags": flags,
                    "exposure_semantic_evidence": semantic_evidence,
                    "raw_record_found": int(bool(raw)),
                }
            )

        evidence_code = evidence.code if len(raw_evidence_classes) == 1 else "mixed_source_evidence"
        evidence_label = evidence.label if len(raw_evidence_classes) == 1 else "Mixed raw-source evidence"
        keep = evidence.keep_for_soil_matrix_main_analysis if len(raw_evidence_classes) == 1 else "no_until_resolved"
        aggregate_row = {
            "aggregate_id": aggregate_id,
            "evidence_class": evidence_code,
            "evidence_label": evidence_label,
            "main_analysis_recommendation": keep,
            "model_media_type": aggregate.get("media_type"),
            "raw_media_codes": ";".join(sorted(raw_codes)),
            "model_primary_medium_domain": aggregate.get("primary_medium_domain"),
            "model_medium_domains": aggregate.get("medium_domains"),
            "model_medium_domain_reason": aggregate.get("medium_domain_reason"),
            "model_medium_conflict_flag": aggregate.get("medium_conflict_flag"),
            "unit_family_v2": aggregate.get("unit_family_v2"),
            "standard_unit_v2": aggregate.get("standard_unit_v2"),
            "conversion_path": aggregate.get("conversion_path"),
            "target_value_count": aggregate.get("target_value_count"),
            "source_result_count": len(result_ids),
            "source_test_count_recomputed": len(raw_test_ids),
            "source_reference_count_recomputed": len(raw_reference_ids),
            "source_text_evidence_flags": ";".join(sorted(raw_flags)),
            "source_raw_records_with_explicit_porewater_title": raw_porewater_result_count,
            "v122_random_holdout_prediction_rows": test_prediction_counts[aggregate_id],
            "v122_random_holdout_reported_metric_rows": reported_metric_counts[aggregate_id],
            "task_head": aggregate.get("task_head"),
            "cas_number": aggregate.get("cas_number"),
            "chemical_name": aggregate.get("chemical_name"),
            "latin_name": aggregate.get("latin_name"),
            "result_ids": aggregate.get("result_ids"),
            "test_ids": aggregate.get("test_ids"),
            "reference_numbers": aggregate.get("reference_numbers"),
        }
        aggregate_output.append(aggregate_row)
        aggregate_by_id[aggregate_id] = aggregate_row

    prediction_metric_rows = prediction_metrics_by_evidence(
        args.prediction_csv, aggregate_by_id, metric_task_heads
    )

    summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in aggregate_output:
        key = (row["evidence_class"], row["unit_family_v2"], row["main_analysis_recommendation"])
        bucket = summary.setdefault(
            key,
            {
                "evidence_class": key[0],
                "unit_family_v2": key[1],
                "main_analysis_recommendation": key[2],
                "aggregate_count": 0,
                "source_result_count": 0,
                "source_raw_records_with_explicit_porewater_title": 0,
                "v122_random_holdout_prediction_rows": 0,
                "v122_random_holdout_reported_metric_rows": 0,
            },
        )
        bucket["aggregate_count"] += 1
        bucket["source_result_count"] += int(row["source_result_count"])
        bucket["source_raw_records_with_explicit_porewater_title"] += int(row["source_raw_records_with_explicit_porewater_title"])
        bucket["v122_random_holdout_prediction_rows"] += int(row["v122_random_holdout_prediction_rows"])
        bucket["v122_random_holdout_reported_metric_rows"] += int(row["v122_random_holdout_reported_metric_rows"])

    references: dict[str, dict[str, Any]] = {}
    for row in raw_output:
        ref_id = str(row["reference_number"])
        ledger = references.setdefault(
            ref_id,
            {
                "reference_number": ref_id,
                "doi": row["doi"],
                "reference_title": row["reference_title"],
                "reference_source": row["reference_source"],
                "reference_year": row["reference_year"],
                "raw_result_count": 0,
                "aggregate_ids": set(),
                "media_codes": set(),
                "evidence_classes": set(),
                "text_evidence_flags": set(),
            },
        )
        ledger["raw_result_count"] += 1
        ledger["aggregate_ids"].add(str(row["aggregate_id"]))
        ledger["media_codes"].add(str(row["raw_media_code"]))
        ledger["evidence_classes"].add(str(row["evidence_class"]))
        ledger["text_evidence_flags"].update(filter(None, str(row["text_evidence_flags"]).split(";")))
    reference_output = [
        {
            **{key: value for key, value in ledger.items() if not isinstance(value, set)},
            "aggregate_count": len(ledger["aggregate_ids"]),
            "aggregate_ids": ";".join(sorted(ledger["aggregate_ids"], key=int)),
            "media_codes": ";".join(sorted(ledger["media_codes"])),
            "evidence_classes": ";".join(sorted(ledger["evidence_classes"])),
            "text_evidence_flags": ";".join(sorted(ledger["text_evidence_flags"])),
        }
        for ledger in references.values()
    ]
    reference_output.sort(key=lambda row: (-int(row["raw_result_count"]), int(row["reference_number"])))
    porewater_reference_output = [
        row for row in reference_output if "explicit_porewater_or_soil_solution" in str(row["text_evidence_flags"])
    ]

    raw_fields = [
        "aggregate_id", "model_media_type", "model_primary_medium_domain", "model_medium_domains",
        "model_medium_domain_reason", "evidence_class", "evidence_label", "main_analysis_recommendation",
        "raw_media_code", "result_id", "test_id", "reference_number", "doi", "reference_title",
        "reference_source", "reference_year", "raw_organism_habitat", "raw_conc1_type",
        "raw_conc1_mean_op", "raw_conc1_mean", "raw_conc1_unit", "raw_conc1_min_op",
        "raw_conc1_min", "raw_conc1_max_op", "raw_conc1_max", "raw_endpoint", "raw_effect",
        "raw_measurement", "raw_endpoint_comments", "raw_response_site_comments", "unit_family_v2",
        "standard_unit_v2", "conversion_path", "target_name", "target_basis", "text_evidence_flags",
        "exposure_semantic_evidence", "raw_record_found",
    ]
    aggregate_fields = [
        "aggregate_id", "evidence_class", "evidence_label", "main_analysis_recommendation",
        "model_media_type", "raw_media_codes", "model_primary_medium_domain", "model_medium_domains",
        "model_medium_domain_reason", "model_medium_conflict_flag", "unit_family_v2", "standard_unit_v2",
        "conversion_path", "target_value_count", "source_result_count", "source_test_count_recomputed",
        "source_reference_count_recomputed", "source_text_evidence_flags", "v122_random_holdout_prediction_rows",
        "source_raw_records_with_explicit_porewater_title", "v122_random_holdout_reported_metric_rows",
        "task_head", "cas_number", "chemical_name", "latin_name", "result_ids", "test_ids", "reference_numbers",
    ]
    reference_fields = [
        "reference_number", "doi", "reference_title", "reference_source", "reference_year",
        "raw_result_count", "aggregate_count", "aggregate_ids", "media_codes", "evidence_classes",
        "text_evidence_flags",
    ]
    summary_rows = sorted(summary.values(), key=lambda row: (-int(row["aggregate_count"]), row["evidence_class"]))
    write_csv(args.output_dir / "raw_source_records.csv", raw_output, raw_fields)
    write_csv(args.output_dir / "model_aggregate_evidence.csv", aggregate_output, aggregate_fields)
    write_csv(
        args.output_dir / "evidence_summary.csv",
        summary_rows,
        ["evidence_class", "unit_family_v2", "main_analysis_recommendation", "aggregate_count", "source_result_count", "source_raw_records_with_explicit_porewater_title", "v122_random_holdout_prediction_rows", "v122_random_holdout_reported_metric_rows"],
    )
    write_csv(args.output_dir / "reference_ledger.csv", reference_output, reference_fields)
    write_csv(args.output_dir / "explicit_porewater_reference_ledger.csv", porewater_reference_output, reference_fields)
    write_csv(
        args.output_dir / "v122_prediction_metrics_by_evidence.csv",
        prediction_metric_rows,
        ["scope", "evidence_class", "n", "r2", "rmse", "mae"],
    )

    manifest = {
        "derived_db": str(args.derived_db),
        "raw_db": str(args.raw_db),
        "model_table": args.model_table,
        "prediction_csv": str(args.prediction_csv),
        "mainline_summary_csv": str(args.mainline_summary_csv),
        "model_aggregate_count": len(aggregate_output),
        "unique_raw_result_count": len(all_result_ids),
        "raw_records_retrieved": len(raw_records),
        "raw_records_missing_from_raw_db": len(all_result_ids) - len(raw_records),
        "raw_records_with_explicit_porewater_title": sum(
            int(row["exposure_semantic_evidence"] == "explicit_porewater_in_reference_title") for row in raw_output
        ),
        "references_with_explicit_porewater_title": len(porewater_reference_output),
        "v122_random_holdout_prediction_rows": sum(test_prediction_counts.values()),
        "v122_random_holdout_reported_metric_rows": sum(reported_metric_counts.values()),
        "v122_reported_metric_task_heads": sorted(metric_task_heads),
        "classification_rule": {
            "verified_soil_matrix": sorted(DIRECT_SOIL_MATRIX_CODES),
            "soil_related_substrate": sorted(SOIL_RELATED_SUBSTRATE_CODES),
            "media_mixture_unresolved": ["MIX"],
            "aqueous_or_hydroponic_nonsoil": sorted(AQUEOUS_CODES),
            "culture_or_agar_nonsoil": sorted(CULTURE_CODES),
            "nonsoil_substrate": sorted(NONSOIL_SUBSTRATE_CODES),
            "sediment_or_sludge": sorted(SEDIMENT_OR_SLUDGE_CODES),
            "no_reported_medium": sorted(MISSING_OR_UNSPECIFIED_CODES),
        },
        "critical_interpretation": (
            "No explicit pore-water, dosing-solution, extract, or leachate field is retained in the raw ECOTOX tables used here. "
            "An mg/L or mol/L unit alone must not be interpreted as pore-water concentration; the title-based pore-water tag is a conservative lower bound, not a complete pore-water census."
        ),
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
