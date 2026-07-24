from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ad_common import ANALYSIS_DIR, write_csv, write_json
from ad_support import (
    TAX_DISPLAY_BY_RANK,
    chemical_similarity_matrix,
    context_distance_matrix,
    context_missing_fraction,
    context_neighbor_features,
    fit_context_ranges,
    normalize_category,
    pairwise_tax_rank,
    taxonomy_support,
)


DEFAULT_CONFIG = ANALYSIS_DIR / "config" / "ad_config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build record-level B/T/L support features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stage3 = pd.read_parquet(args.output_dir / "stage3_predictions_ensemble.parquet")
    chemistry = pd.read_parquet(args.output_dir / "stage3_chemical_support_all.parquet")
    context_snapshot = pd.read_parquet(args.output_dir / "stage3_discovery_context_snapshot.parquet")

    structure_cache = pd.read_parquet(args.output_dir / "chemical_structure_cache.parquet")
    all_stage3 = attach_structure_columns(all_stage3, structure_cache)
    context_fields = list(config["context"]["fields"])
    context_snapshot = context_snapshot.rename(columns={"stable_record_id": "record_id"})
    context_columns = ["record_id", *[field for field in context_fields if field not in all_stage3]]
    all_stage3 = all_stage3.merge(
        context_snapshot[context_columns], on="record_id", how="left", validate="one_to_one"
    )
    training = all_stage3.loc[all_stage3["analysis_split"].eq("train")].copy()
    evaluation = chemistry.copy()
    missing_context = [field for field in context_fields if field not in evaluation]
    evaluation = evaluation.merge(
        all_stage3[["record_id", *missing_context]], on="record_id", how="left", validate="one_to_one"
    )

    missing_tokens = {str(value).casefold() for value in config["context"]["semantic_missing_tokens"]}
    numeric_fields = [field for field in context_fields if field in ("effect_level_x", "duration_bin_h")]
    categorical_fields = [field for field in context_fields if field not in numeric_fields]
    ranges = fit_context_ranges(training, numeric_fields)
    write_json(args.output_dir / "context_training_ranges.json", ranges)

    global_tax = taxonomy_support(
        evaluation, training, same_task=False, missing_tokens=missing_tokens
    )
    same_task_tax = taxonomy_support(
        evaluation, training, same_task=True, missing_tokens=missing_tokens
    )
    evaluation = pd.concat(
        [evaluation.reset_index(drop=True), global_tax.reset_index(drop=True), same_task_tax.reset_index(drop=True)],
        axis=1,
    )
    evaluation["context_missing_fraction"] = context_missing_fraction(
        evaluation, context_fields, missing_tokens=missing_tokens
    )

    task_summary = build_task_support(training, evaluation, context_fields, missing_tokens)
    task_lookup = task_summary.set_index("model_head")
    for column in task_summary.columns:
        if column == "model_head":
            continue
        evaluation[column] = evaluation["model_head"].map(task_lookup[column])

    local_columns: list[str] = []
    neighbor_frames: list[pd.DataFrame] = []
    chem_cfg = config["chemical"]
    ctx_cfg = config["context"]
    tax_threshold_rank = {"family": 4, "genus": 5, "species": 6}
    for task, query in evaluation.groupby("model_head", sort=False):
        reference = training.loc[training["model_head"].eq(task)].copy()
        if reference.empty:
            raise ValueError(f"No Stage-3 training reference for task {task}")
        query = query.copy()
        distance = context_distance_matrix(
            query,
            reference,
            numeric_fields=numeric_fields,
            categorical_fields=categorical_fields,
            ranges=ranges,
            missing_tokens=missing_tokens,
        )
        tax_rank = pairwise_tax_rank(query, reference, missing_tokens=missing_tokens)
        similarity = chemical_similarity_matrix(
            query,
            reference,
            radius=int(chem_cfg["radius"]),
            n_bits=int(chem_cfg["n_bits"]),
        )
        features = context_neighbor_features(
            distance,
            k_values=ctx_cfg["k_neighbors_sensitivity"],
            support_thresholds=ctx_cfg["context_support_thresholds"],
        )
        features["B_exp"] = 1.0 - features["context_top5_mean_distance"]
        for chemical_threshold in chem_cfg["candidate_similarity_thresholds"]:
            chemical_ok = np.nan_to_num(similarity, nan=-1.0) >= float(chemical_threshold)
            for tax_label in config["local_support"]["taxonomy_thresholds"]:
                tax_ok = tax_rank >= tax_threshold_rank[tax_label]
                for context_support in ctx_cfg["context_support_thresholds"]:
                    context_ok = distance <= 1.0 - float(context_support)
                    name = local_column_name(chemical_threshold, tax_label, context_support)
                    features[name] = (chemical_ok & tax_ok & context_ok).sum(axis=1)
                    local_columns.append(name)
        feature_frame = pd.DataFrame(features, index=query.index)
        feature_frame.insert(0, "record_id", query["record_id"].to_numpy())
        neighbor_frames.append(feature_frame)
    neighbors = pd.concat(neighbor_frames, ignore_index=True)
    evaluation = evaluation.merge(neighbors, on="record_id", how="left", validate="one_to_one")
    evaluation["B_bottleneck"] = np.minimum(
        evaluation["tax_support_same_task_display"], evaluation["B_exp"]
    )
    evaluation["chemical_entity_id"] = chemical_entity_identity(evaluation)
    evaluation["NAE_M10"] = evaluation["AE_M10"] / evaluation["task_target_iqr"].replace(0, np.nan)
    evaluation["NAE_M00"] = evaluation["AE_M00"] / evaluation["task_target_iqr"].replace(0, np.nan)
    evaluation.to_parquet(args.output_dir / "ad_record_level_unlocked.parquet", index=False)
    write_csv(args.output_dir / "task_support_summary.csv", task_summary)
    context_audit = build_context_audit(
        all_stage3,
        training,
        evaluation,
        context_fields,
        numeric_fields,
        missing_tokens,
        config,
    )
    write_csv(args.output_dir / "context_field_audit.csv", context_audit)
    manifest = {
        "schema_version": 1,
        "record_rows": len(evaluation),
        "validation_rows": int(evaluation["analysis_split"].eq("validation").sum()),
        "test_rows": int(evaluation["analysis_split"].eq("test").sum()),
        "task_count": int(evaluation["model_head"].nunique()),
        "context_fields": context_fields,
        "numeric_context_fields": numeric_fields,
        "categorical_context_fields": categorical_fields,
        "local_support_columns": sorted(set(local_columns)),
        "local_support_column_count": len(set(local_columns)),
        "formal_taxonomy_field": "tax_support_same_task",
        "test_target_used_for_feature_construction": False,
    }
    write_json(args.output_dir / "support_feature_manifest.json", manifest)
    (args.output_dir / "SUPPORT_FEATURE_REPORT.md").write_text(
        render_report(evaluation, task_summary, context_audit, manifest), encoding="utf-8"
    )
    print(args.output_dir / "SUPPORT_FEATURE_REPORT.md")


def attach_structure_columns(all_stage3: pd.DataFrame, structure_cache: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "raw_smiles",
        "canonical_parent",
        "murcko_scaffold",
        "structure_status",
        "structure_reason",
    ]
    mapping = structure_cache[columns].drop_duplicates()
    conflicts = mapping.groupby("raw_smiles")[["canonical_parent", "murcko_scaffold", "structure_status"]].nunique()
    if bool((conflicts > 1).any().any()):
        raise ValueError("A raw SMILES maps to inconsistent normalized structures")
    mapping = mapping.drop_duplicates("raw_smiles")
    output = all_stage3.merge(
        mapping,
        left_on="smiles",
        right_on="raw_smiles",
        how="left",
        validate="many_to_one",
    )
    if output["structure_status"].isna().any():
        raise ValueError("Complete Stage-3 structure mapping is unavailable")
    cas = output["cas_number"].astype("string").fillna("").str.strip()
    dtxsid = output["dtxsid"].astype("string").fillna("").str.strip()
    smiles = output["smiles"].astype("string").fillna("").str.strip()
    aggregate = output["aggregate_id"].astype("string")
    output["raw_chemical_id"] = pd.Series(
        np.where(
            cas.ne(""),
            "cas:" + cas,
            np.where(
                dtxsid.ne(""),
                "dtxsid:" + dtxsid,
                np.where(smiles.ne(""), "smiles:" + smiles, "aggregate:" + aggregate),
            ),
        ),
        index=output.index,
        dtype="string",
    )
    return output


def build_task_support(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    context_fields: list[str],
    missing_tokens: set[str],
) -> pd.DataFrame:
    training = training.copy()
    training["chemical_entity_id"] = chemical_entity_identity(training)
    training["species_identity"] = normalize_category(training["latin_name"], missing_tokens)
    training["genus_identity"] = normalize_category(training["genus"], missing_tokens)
    normalized_context = [normalize_category(training[field], missing_tokens) for field in context_fields]
    training["context_combination"] = pd.concat(normalized_context, axis=1).astype(str).agg("||".join, axis=1)
    rows: list[dict[str, Any]] = []
    for task, group in training.groupby("model_head", sort=True):
        y = pd.to_numeric(group["y_true"], errors="coerce").dropna()
        mean = float(y.mean())
        iqr = float(y.quantile(0.75) - y.quantile(0.25))
        sd = float(y.std(ddof=1))
        validation = evaluation.loc[evaluation["model_head"].eq(task)]
        validation_y = pd.to_numeric(
            validation.loc[validation["analysis_split"].eq("validation"), "y_true"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "model_head": task,
                "n_records_task": len(group),
                "n_unique_chemicals_task": group["chemical_entity_id"].nunique(),
                "n_unique_species_task": group.loc[group["species_identity"].ne("<missing>"), "species_identity"].nunique(),
                "n_unique_genera_task": group.loc[group["genus_identity"].ne("<missing>"), "genus_identity"].nunique(),
                "n_unique_context_combinations_task": group["context_combination"].nunique(),
                "task_target_mean": mean,
                "task_target_iqr": iqr,
                "task_target_sd": sd,
                "task_mean_baseline_train_mae": float((y - mean).abs().mean()),
                "task_mean_baseline_validation_mae": (
                    math.nan if validation_y.empty else float((validation_y - mean).abs().mean())
                ),
            }
        )
    output = pd.DataFrame(rows)
    for source, target in (
        ("n_records_task", "task_records_percentile"),
        ("n_unique_chemicals_task", "task_chemicals_percentile"),
        ("n_unique_species_task", "task_species_percentile"),
    ):
        output[target] = np.log1p(output[source]).rank(method="average", pct=True)
    output["T_index"] = output[
        ["task_records_percentile", "task_chemicals_percentile", "task_species_percentile"]
    ].min(axis=1)
    output["T_tier"] = pd.cut(
        output["T_index"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["T0", "T1", "T2", "T3"],
        include_lowest=True,
    ).astype(str)
    return output


def chemical_entity_identity(frame: pd.DataFrame) -> pd.Series:
    canonical = frame["canonical_parent"].astype("string").fillna("").str.strip()
    fallback = frame.get("raw_chemical_id", frame["aggregate_id"].astype("string")).astype("string")
    return pd.Series(np.where(canonical.ne(""), "canonical:" + canonical, "unavailable:" + fallback), index=frame.index, dtype="string")


def local_column_name(chemical: float, taxonomy: str, context: float) -> str:
    return f"n_local_c{int(round(float(chemical)*100)):02d}_tax_{taxonomy}_b{int(round(float(context)*100)):02d}"


def build_context_audit(
    all_stage3: pd.DataFrame,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    fields: list[str],
    numeric_fields: list[str],
    missing_tokens: set[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for field in fields:
        train_values = normalize_category(training[field], missing_tokens)
        row: dict[str, Any] = {
            "field": field,
            "type": "numeric" if field in numeric_fields else "categorical",
            "included_in_B_exp": True,
            "inclusion_reason": "actual model input with training variation; explicitly locked before calibration",
            "train_semantic_missing_rate": float(train_values.eq("<missing>").mean()),
            "train_unique_nonmissing": int(train_values.loc[train_values.ne("<missing>")].nunique()),
            "tasks_with_nonmissing_variation": 0,
        }
        varying = 0
        for _, group in training.groupby("model_head", sort=False):
            values = normalize_category(group[field], missing_tokens)
            if values.loc[values.ne("<missing>")].nunique() > 1:
                varying += 1
        row["tasks_with_nonmissing_variation"] = varying
        train_vocab = set(train_values.loc[train_values.ne("<missing>")])
        for split in ("validation", "test"):
            values = normalize_category(
                evaluation.loc[evaluation["analysis_split"].eq(split), field], missing_tokens
            )
            row[f"{split}_semantic_missing_rate"] = float(values.eq("<missing>").mean())
            row[f"{split}_unseen_rate"] = (
                math.nan
                if field in numeric_fields
                else float((values.ne("<missing>") & ~values.isin(train_vocab)).mean())
            )
        rows.append(row)
    for field, reason in config["context"]["excluded_fields"].items():
        rows.append(
            {
                "field": field,
                "type": "excluded",
                "included_in_B_exp": False,
                "inclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def render_report(
    evaluation: pd.DataFrame,
    task_summary: pd.DataFrame,
    context_audit: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    validation = evaluation.loc[evaluation["analysis_split"].eq("validation")]
    test = evaluation.loc[evaluation["analysis_split"].eq("test")]
    return "\n".join(
        [
            "# Record-level support-feature report",
            "",
            "> Status: complete and unlocked. No AD tier has yet been assigned; validation calibration is the next step.",
            "",
            "## Biological and experimental support",
            "",
            "- Formal taxonomy support is evaluated within the same task head; global taxonomy support remains a descriptive shared-trunk view.",
            f"- Validation/test family-or-closer same-task tax support: {(validation['tax_support_same_task_rank'] >= 4).mean():.2%}/{(test['tax_support_same_task_rank'] >= 4).mean():.2%}.",
            f"- Validation/test median B_exp: {validation['B_exp'].median():.3f}/{test['B_exp'].median():.3f}.",
            "- B_exp uses equal-weight Gower-type distances within task, with training-only Q05-Q95 numeric ranges and k=5 primary neighbors.",
            "- Semantic missing values are explicit states and are also reported through context_missing_fraction.",
            "",
            "## Task-head support",
            "",
            f"- Tasks: {len(task_summary)}; Stage-3 training records per task range from {task_summary['n_records_task'].min():,} to {task_summary['n_records_task'].max():,}.",
            f"- T0/T1/T2/T3 task counts: {task_summary['T_tier'].value_counts().sort_index().to_dict()}.",
            "- T_index is the minimum empirical percentile of log record, chemical and species counts. It is relative data support, not task difficulty.",
            "",
            "## Joint local support",
            "",
            f"- Precomputed local-neighbor grids: {manifest['local_support_column_count']} combinations.",
            "- Each n_local counts Stage-3 training records with the same task, chemical similarity >= threshold, taxonomy at or above the named level and context support >= threshold.",
            "- Chemical thresholds use >= consistently; structure-unavailable records have zero chemical-qualified local neighbors but retain NaN C values.",
            "",
            "## Leakage boundary",
            "",
            "All vocabularies, robust numeric ranges, task statistics and neighbor references were fit from Stage-3 training rows only. Validation and test targets were not used to construct support features. The record file remains explicitly unlocked until the validation-only rule-selection step.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
