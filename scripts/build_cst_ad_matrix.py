from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_deep_runs import metrics_from_prediction_rows


DEFAULT_CHEMICAL_SCORE_COLUMN = "ad_max_tanimoto_to_train"
DEFAULT_SPECIES_SCORE_COLUMN = "ad_max_taxon_similarity_to_train"
DEFAULT_SPECIES_TASK_COLUMN = "ad_species_task_family_seen_train"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Chemical-Species-Task applicability-domain matrix from "
            "ensemble AD prediction rows."
        )
    )
    parser.add_argument("--prediction-rows", required=True, help="Input *_ad_prediction_rows.csv file.")
    parser.add_argument("--out-dir", required=True, help="Directory for CST-AD outputs.")
    parser.add_argument("--db", default=None, help="Optional modeling SQLite database for train-reference coverage.")
    parser.add_argument("--source-table", default=None, help="Source table used with --db.")
    parser.add_argument("--split-name", default=None, help="Split name used with --db.")
    parser.add_argument("--split-part", default="test")
    parser.add_argument("--chemical-score-column", default=DEFAULT_CHEMICAL_SCORE_COLUMN)
    parser.add_argument("--species-score-column", default=DEFAULT_SPECIES_SCORE_COLUMN)
    parser.add_argument("--species-task-column", default=DEFAULT_SPECIES_TASK_COLUMN)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.10,
        help="Quantile used for automatic chemical/species in-domain thresholds.",
    )
    parser.add_argument(
        "--chemical-threshold",
        type=float,
        default=None,
        help="Override chemical score threshold. Higher scores are treated as more in-domain.",
    )
    parser.add_argument(
        "--species-threshold",
        type=float,
        default=None,
        help="Override taxonomy score threshold. Higher scores are treated as more in-domain.",
    )
    parser.add_argument(
        "--uncertainty-high-quantile",
        type=float,
        default=0.90,
        help="Quantile used for automatic high-uncertainty threshold from ensemble SD.",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=None,
        help="Override ensemble SD high-uncertainty threshold.",
    )
    parser.add_argument("--top-failures", type=int, default=50)
    args = parser.parse_args()

    prediction_rows = Path(args.prediction_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(prediction_rows)
    reference_frame = None
    if args.db or args.source_table or args.split_name:
        if not (args.db and args.source_table and args.split_name):
            raise ValueError("--db, --source-table, and --split-name must be provided together.")
        from qsar_tl.training.baseline import load_split_frame

        reference_frame = load_split_frame(
            args.db,
            source_table=str(args.source_table),
            split_name=str(args.split_name),
        )

    cst = build_cst_ad_frame(
        frame,
        reference_frame=reference_frame,
        split_part=str(args.split_part),
        chemical_score_column=str(args.chemical_score_column),
        species_score_column=str(args.species_score_column),
        species_task_column=str(args.species_task_column),
        threshold_quantile=float(args.threshold_quantile),
        chemical_threshold=args.chemical_threshold,
        species_threshold=args.species_threshold,
        uncertainty_high_quantile=float(args.uncertainty_high_quantile),
        uncertainty_threshold=args.uncertainty_threshold,
    )
    tier_summary = summarize_cst_ad_tiers(cst)
    family_summary = summarize_cst_ad_by_family(cst)
    failure_cases = select_failure_cases(cst, top_n=int(args.top_failures))

    cst_rows_path = out_dir / "cst_ad_prediction_rows.csv"
    tier_summary_path = out_dir / "cst_ad_tier_summary.csv"
    family_summary_path = out_dir / "cst_ad_family_summary.csv"
    failure_cases_path = out_dir / "cst_ad_failure_cases.csv"
    manifest_path = out_dir / "cst_ad_manifest.json"

    cst.to_csv(cst_rows_path, index=False)
    tier_summary.to_csv(tier_summary_path, index=False)
    family_summary.to_csv(family_summary_path, index=False)
    failure_cases.to_csv(failure_cases_path, index=False)
    manifest = {
        "prediction_rows": str(prediction_rows),
        "db": str(args.db) if args.db else None,
        "source_table": str(args.source_table) if args.source_table else None,
        "split_name": str(args.split_name) if args.split_name else None,
        "reference_rows": int(len(reference_frame)) if reference_frame is not None else None,
        "split_part": str(args.split_part),
        "rows": int(len(cst)),
        "chemical_score_column": str(args.chemical_score_column),
        "species_score_column": str(args.species_score_column),
        "species_task_column": str(args.species_task_column),
        "threshold_quantile": float(args.threshold_quantile),
        "chemical_threshold": float(cst["cst_chemical_threshold"].iloc[0]) if len(cst) else None,
        "species_threshold": float(cst["cst_species_threshold"].iloc[0]) if len(cst) else None,
        "uncertainty_high_quantile": float(args.uncertainty_high_quantile),
        "uncertainty_threshold": (
            float(cst["cst_uncertainty_threshold"].iloc[0])
            if len(cst) and cst["cst_uncertainty_threshold"].notna().any()
            else None
        ),
        "outputs": {
            "prediction_rows": str(cst_rows_path),
            "tier_summary": str(tier_summary_path),
            "family_summary": str(family_summary_path),
            "failure_cases": str(failure_cases_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_cst_ad_frame(
    frame: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame | None = None,
    split_part: str = "test",
    chemical_score_column: str = DEFAULT_CHEMICAL_SCORE_COLUMN,
    species_score_column: str = DEFAULT_SPECIES_SCORE_COLUMN,
    species_task_column: str = DEFAULT_SPECIES_TASK_COLUMN,
    threshold_quantile: float = 0.10,
    chemical_threshold: float | None = None,
    species_threshold: float | None = None,
    uncertainty_high_quantile: float = 0.90,
    uncertainty_threshold: float | None = None,
) -> pd.DataFrame:
    if "split_part" not in frame.columns:
        raise ValueError("Input prediction rows must contain split_part.")
    result = frame[frame["split_part"].astype(str).eq(str(split_part))].copy()
    if result.empty:
        raise ValueError(f"No rows found for split_part={split_part!r}.")
    for column in (chemical_score_column, species_score_column, "abs_error", "y_true", "y_pred"):
        if column not in result.columns:
            raise ValueError(f"Missing required column: {column}")

    result["cst_chemical_score"] = pd.to_numeric(result[chemical_score_column], errors="coerce")
    result["cst_species_score"] = pd.to_numeric(result[species_score_column], errors="coerce")
    result["cst_abs_error"] = pd.to_numeric(result["abs_error"], errors="coerce")
    result["cst_ensemble_sd"] = ensemble_prediction_sd(result)
    result["cst_species_task_family_seen"] = boolean_column(result, species_task_column)
    result = add_reference_coverage(result, reference_frame)

    chem_threshold = resolve_threshold(
        result["cst_chemical_score"],
        threshold_quantile,
        override=chemical_threshold,
    )
    species_threshold_value = resolve_threshold(
        result["cst_species_score"],
        threshold_quantile,
        override=species_threshold,
    )
    uncertainty_threshold_value = resolve_uncertainty_threshold(
        result["cst_ensemble_sd"],
        uncertainty_high_quantile,
        override=uncertainty_threshold,
    )
    result["cst_chemical_threshold"] = chem_threshold
    result["cst_species_threshold"] = species_threshold_value
    result["cst_uncertainty_threshold"] = uncertainty_threshold_value
    result["cst_chemical_in_domain"] = result["cst_chemical_score"] >= chem_threshold
    result["cst_species_in_domain"] = result["cst_species_score"] >= species_threshold_value
    if math.isnan(uncertainty_threshold_value):
        result["cst_uncertainty_low"] = True
        result["cst_uncertainty_tier"] = "not_available"
    else:
        result["cst_uncertainty_low"] = result["cst_ensemble_sd"] <= uncertainty_threshold_value
        result["cst_uncertainty_tier"] = np.where(result["cst_uncertainty_low"], "low_or_moderate", "high")

    tiers = [
        assign_cst_tier(
            chemical_in=bool(row.cst_chemical_in_domain),
            species_in=bool(row.cst_species_in_domain),
            species_task_seen=bool(row.cst_species_task_family_seen),
            uncertainty_low=bool(row.cst_uncertainty_low),
        )
        for row in result.itertuples(index=False)
    ]
    result["cst_ad_tier"] = [tier for tier, _reason in tiers]
    result["cst_ad_reason"] = [reason for _tier, reason in tiers]
    return result


def ensemble_prediction_sd(frame: pd.DataFrame) -> pd.Series:
    member_columns = [
        column
        for column in frame.columns
        if column.startswith("y_pred_member_") and not column.endswith("_scaled")
    ]
    if len(member_columns) < 2:
        return pd.Series(np.nan, index=frame.index)
    members = frame.loc[:, sorted(member_columns)].apply(pd.to_numeric, errors="coerce")
    return members.std(axis=1, ddof=0)


def boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def resolve_threshold(series: pd.Series, quantile: float, *, override: float | None) -> float:
    if override is not None:
        return float(override)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Cannot resolve threshold from an empty score column.")
    return float(clean.quantile(quantile))


def resolve_uncertainty_threshold(series: pd.Series, quantile: float, *, override: float | None) -> float:
    if override is not None:
        return float(override)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.quantile(quantile))


def assign_cst_tier(
    *,
    chemical_in: bool,
    species_in: bool,
    species_task_seen: bool,
    uncertainty_low: bool,
) -> tuple[str, str]:
    if not chemical_in and not species_in:
        return "AD-D", "chemical_and_species_extrapolation"
    if not uncertainty_low and (not chemical_in or not species_in):
        return "AD-D", "extrapolation_with_high_ensemble_sd"
    if not uncertainty_low:
        return "AD-C", "high_ensemble_sd"
    if chemical_in and species_in and species_task_seen:
        return "AD-A", "chemical_species_and_species_task_seen"
    if chemical_in and species_in:
        return "AD-B", "chemical_and_species_seen_but_species_task_unseen"
    if not chemical_in:
        return "AD-C", "chemical_extrapolation"
    return "AD-C", "species_extrapolation"


def add_reference_coverage(result: pd.DataFrame, reference_frame: pd.DataFrame | None) -> pd.DataFrame:
    result = result.copy()
    if reference_frame is None:
        result["cst_reference_available"] = False
        return add_empty_reference_columns(result)

    if "split_part" not in reference_frame.columns:
        raise ValueError("reference_frame must contain split_part.")
    train = reference_frame[reference_frame["split_part"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("reference_frame does not contain train rows.")

    train_keys = key_frame(train)
    query_keys = key_frame(result)
    result["cst_reference_available"] = True
    for column in train_keys.columns:
        result[f"cst_{column}"] = query_keys[column]

    count_specs = {
        "train_n_chemical": ("chemical_key",),
        "train_n_species": ("species_key",),
        "train_n_task_head": ("task_head_key",),
        "train_n_task_family": ("task_family_key",),
        "train_n_chemical_species": ("chemical_key", "species_key"),
        "train_n_chemical_task_head": ("chemical_key", "task_head_key"),
        "train_n_species_task_head": ("species_key", "task_head_key"),
        "train_n_species_task_family": ("species_key", "task_family_key"),
        "train_n_cst_exact": ("chemical_key", "species_key", "task_head_key"),
    }
    for output, columns in count_specs.items():
        counts = grouped_counts(train_keys, columns)
        result[f"cst_{output}"] = lookup_counts(query_keys, counts, columns)

    result["cst_task_head_seen_train"] = result["cst_train_n_task_head"] > 0
    result["cst_task_family_seen_train"] = result["cst_train_n_task_family"] > 0
    result["cst_chemical_species_seen_train"] = result["cst_train_n_chemical_species"] > 0
    result["cst_chemical_task_head_seen_train"] = result["cst_train_n_chemical_task_head"] > 0
    result["cst_species_task_head_seen_train"] = result["cst_train_n_species_task_head"] > 0
    result["cst_species_task_family_seen_train_ref"] = result["cst_train_n_species_task_family"] > 0
    result["cst_exact_seen_train"] = result["cst_train_n_cst_exact"] > 0
    result["cst_task_coverage_label"] = [
        task_coverage_label(row)
        for row in result.itertuples(index=False)
    ]
    if "cst_species_task_family_seen" in result.columns:
        result["cst_species_task_family_seen"] = (
            result["cst_species_task_family_seen"] | result["cst_species_task_family_seen_train_ref"]
        )
    return result


def add_empty_reference_columns(result: pd.DataFrame) -> pd.DataFrame:
    key_columns = ("chemical_key", "species_key", "task_head_key", "task_family_key")
    count_columns = (
        "train_n_chemical",
        "train_n_species",
        "train_n_task_head",
        "train_n_task_family",
        "train_n_chemical_species",
        "train_n_chemical_task_head",
        "train_n_species_task_head",
        "train_n_species_task_family",
        "train_n_cst_exact",
    )
    bool_columns = (
        "task_head_seen_train",
        "task_family_seen_train",
        "chemical_species_seen_train",
        "chemical_task_head_seen_train",
        "species_task_head_seen_train",
        "species_task_family_seen_train_ref",
        "exact_seen_train",
    )
    for column in key_columns:
        result[f"cst_{column}"] = ""
    for column in count_columns:
        result[f"cst_{column}"] = pd.NA
    for column in bool_columns:
        result[f"cst_{column}"] = False
    result["cst_task_coverage_label"] = "not_evaluated"
    return result


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    task_head = key_series(frame, ("task_head", "target_name"))
    task_family = key_series(frame, ("task_family", "target_family", "effect_family"))
    missing_family = task_family.eq("") & task_head.ne("")
    task_family = task_family.mask(missing_family, task_head.str.split("_", n=1).str[0])
    return pd.DataFrame(
        {
            "chemical_key": key_series(frame, ("dtxsid", "cas_number", "smiles", "chemical_name")),
            "species_key": key_series(frame, ("species_number", "latin_name")),
            "task_head_key": task_head,
            "task_family_key": task_family,
        },
        index=frame.index,
    )


def key_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        normalized = normalize_text(frame[column])
        values = values.mask(values.eq("") & normalized.ne(""), normalized)
    return values


def normalize_text(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .replace({"<na>": "", "nan": "", "none": ""})
    )


def grouped_counts(keys: pd.DataFrame, columns: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    usable = keys.loc[:, list(columns)].copy()
    usable = usable[(usable != "").all(axis=1)]
    if usable.empty:
        return {}
    grouped = usable.groupby(list(columns), dropna=False).size()
    return {tuple(key if isinstance(key, tuple) else (key,)): int(value) for key, value in grouped.items()}


def lookup_counts(keys: pd.DataFrame, counts: dict[tuple[str, ...], int], columns: tuple[str, ...]) -> list[int]:
    values = []
    for row in keys.loc[:, list(columns)].itertuples(index=False, name=None):
        if not all(row):
            values.append(0)
        else:
            values.append(int(counts.get(tuple(row), 0)))
    return values


def task_coverage_label(row: Any) -> str:
    if bool(getattr(row, "cst_exact_seen_train", False)):
        return "chemical_species_task_seen"
    if bool(getattr(row, "cst_species_task_head_seen_train", False)):
        return "species_task_seen"
    if bool(getattr(row, "cst_species_task_family_seen_train_ref", False)):
        return "species_task_family_seen"
    if bool(getattr(row, "cst_task_head_seen_train", False)):
        return "task_seen_only"
    if bool(getattr(row, "cst_task_family_seen_train", False)):
        return "task_family_seen_only"
    return "task_unseen"


def summarize_cst_ad_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for tier, sub in frame.groupby("cst_ad_tier", dropna=False, sort=True):
        metrics = metrics_from_prediction_rows(dataframe_to_rows(sub))
        rows.append(
            {
                "cst_ad_tier": tier,
                "coverage_n": int(len(sub)),
                "coverage_fraction": float(len(sub) / total) if total else math.nan,
                **metrics,
                "median_abs_error": float(pd.to_numeric(sub["abs_error"], errors="coerce").median()),
                "mean_chemical_score": float(sub["cst_chemical_score"].mean()),
                "mean_species_score": float(sub["cst_species_score"].mean()),
                "mean_ensemble_sd": float(sub["cst_ensemble_sd"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_cst_ad_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["cst_ad_tier"]
    if "task_family" in frame.columns:
        group_columns.append("task_family")
    for keys, sub in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = {column: value for column, value in zip(group_columns, keys)}
        payload.update(metrics_from_prediction_rows(dataframe_to_rows(sub)))
        payload["mean_ensemble_sd"] = float(sub["cst_ensemble_sd"].mean())
        rows.append(payload)
    return pd.DataFrame(rows)


def select_failure_cases(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "sample_id",
            "aggregate_id",
            "task_head",
            "task_family",
            "chemical_name",
            "cas_number",
            "latin_name",
            "family",
            "genus",
            "species",
            "y_true",
            "y_pred",
            "residual",
            "abs_error",
            "cst_chemical_score",
            "cst_species_score",
            "cst_ensemble_sd",
            "cst_ad_tier",
            "cst_ad_reason",
        )
        if column in frame.columns
    ]
    return frame.sort_values("cst_abs_error", ascending=False).loc[:, columns].head(top_n).reset_index(drop=True)


def dataframe_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notnull(frame), "").to_dict(orient="records")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
