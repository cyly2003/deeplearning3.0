from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsar_tl.evaluation.application_domain import williams_leverage
from qsar_tl.evaluation.splits import build_experiment_split_sets
from qsar_tl.training.deep_experiment import MolecularFeatureBuilder
from scripts.summarize_deep_runs import metrics_from_prediction_rows


TAXON_COLUMNS = ("kingdom", "phylum", "class_name", "tax_order", "family")
TIER_ORDER = ("AD-A", "AD-B", "AD-C", "AD-D")
VARIANT_COLUMNS = {
    "C-only": "cst_c_only_tier",
    "C+S": "cst_cs_tier",
    "C+S+T": "cst_ad_tier",
}


@dataclass(frozen=True)
class ReferenceSpec:
    split_policy: str
    fold: str
    transfer_split_name: str
    soil_split_name: str
    reference_frame: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build random-split mainline Chemical-Species-Task AD outputs without "
            "using ensemble-SD uncertainty tiers."
        )
    )
    parser.add_argument("--db", default="outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite")
    parser.add_argument("--transfer-table", default="aggregated_task_records_aquatic_soil_ptox_qc")
    parser.add_argument("--soil-table", default="aggregated_task_records_soil_ptox_qc")
    parser.add_argument(
        "--prediction-dir",
        default="outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fingerprint-size", type=int, default=512)
    parser.add_argument("--molecular-cache", default="outputs/features/molecular_features_rdkit_morgan512.jsonl")
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--chemical-threshold", type=float, default=0.5)
    parser.add_argument("--chemical-high-threshold", type=float, default=0.7)
    parser.add_argument("--species-threshold", type=float, default=0.8)
    parser.add_argument("--high-error-quantile", type=float, default=0.80)
    parser.add_argument("--embedding-lookup", default="outputs/experiments/final_mainline_train_space/species_embedding_lookup.csv.gz")
    parser.add_argument("--max-prediction-rows", type=int, default=None)
    parser.add_argument("--skip-embedding-diagnostic", action="store_true")
    args = parser.parse_args()
    silence_rdkit_logs()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = Path(args.prediction_dir)

    transfer_frame = read_table(
        Path(args.db),
        str(args.transfer_table),
        columns=needed_reference_columns(),
    )
    soil_frame = read_table(
        Path(args.db),
        str(args.soil_table),
        columns=needed_reference_columns(),
    )

    references = build_random_reference_specs(
        transfer_frame=transfer_frame,
        soil_frame=soil_frame,
        transfer_table=str(args.transfer_table),
        soil_table=str(args.soil_table),
        seed=int(args.split_seed),
    )

    all_rows: list[pd.DataFrame] = []
    embedding_rows: list[pd.DataFrame] = []
    manifest_inputs: list[dict[str, Any]] = []
    for spec in references:
        prediction_path = prediction_file(prediction_dir, spec.split_policy, spec.fold)
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction_rows = pd.read_csv(prediction_path, low_memory=False)
        if args.max_prediction_rows is not None:
            prediction_rows = prediction_rows.head(int(args.max_prediction_rows)).copy()
        cst = build_cst_rows(
            prediction_rows,
            reference_frame=spec.reference_frame,
            split_policy=spec.split_policy,
            fold=spec.fold,
            fingerprint_size=int(args.fingerprint_size),
            molecular_cache=str(args.molecular_cache) if args.molecular_cache else None,
            pca_components=int(args.pca_components),
            chemical_threshold=float(args.chemical_threshold),
            chemical_high_threshold=float(args.chemical_high_threshold),
            species_threshold=float(args.species_threshold),
        )
        all_rows.append(cst)
        if not args.skip_embedding_diagnostic:
            embedding = build_embedding_diagnostic(
                cst,
                reference_frame=spec.reference_frame,
                embedding_lookup_path=Path(args.embedding_lookup),
                split_policy=spec.split_policy,
                fold=spec.fold,
            )
            if not embedding.empty:
                embedding_rows.append(embedding)
        manifest_inputs.append(
            {
                "split_policy": spec.split_policy,
                "fold": spec.fold,
                "prediction_rows": str(prediction_path),
                "transfer_split_name": spec.transfer_split_name,
                "soil_split_name": spec.soil_split_name,
                "reference_rows": int(len(spec.reference_frame)),
                "prediction_rows_used": int(len(prediction_rows)),
            }
        )

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = out_dir / "random_mainline_cst_ad_prediction_rows.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    tier_summary = summarize_tiers(combined)
    family_summary = summarize_family_tiers(combined)
    variant_summary = summarize_variants(combined)
    high_error = summarize_high_error_detection(combined, high_error_quantile=float(args.high_error_quantile))
    failures = select_failure_cases(combined)

    tier_summary.to_csv(out_dir / "cst_ad_tier_summary.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(out_dir / "cst_ad_family_summary.csv", index=False, encoding="utf-8-sig")
    variant_summary.to_csv(out_dir / "cst_ad_variant_comparison.csv", index=False, encoding="utf-8-sig")
    high_error.to_csv(out_dir / "cst_ad_high_error_detection.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out_dir / "cst_ad_failure_cases.csv", index=False, encoding="utf-8-sig")

    embedding_path = None
    if embedding_rows:
        embedding_frame = pd.concat(embedding_rows, ignore_index=True)
        embedding_path = out_dir / "species_embedding_distance_diagnostic.csv"
        embedding_frame.to_csv(embedding_path, index=False, encoding="utf-8-sig")

    axis_tiers = combined.loc[
        :,
        [
            column
            for column in (
                "split_policy",
                "fold",
                "sample_id",
                "aggregate_id",
                "task_head",
                "task_family",
                "chemical_name",
                "cas_number",
                "latin_name",
                "cst_chemical_tier",
                "cst_species_tier",
                "cst_task_tier",
                "cst_c_only_tier",
                "cst_cs_tier",
                "cst_ad_tier",
                "cst_ad_reason",
            )
            if column in combined.columns
        ],
    ]
    axis_tiers.to_csv(out_dir / "cst_ad_axis_tiers.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "db": str(args.db),
        "transfer_table": str(args.transfer_table),
        "soil_table": str(args.soil_table),
        "prediction_dir": str(prediction_dir),
        "out_dir": str(out_dir),
        "split_seed": int(args.split_seed),
        "chemical_threshold": float(args.chemical_threshold),
        "chemical_high_threshold": float(args.chemical_high_threshold),
        "molecular_cache": str(args.molecular_cache) if args.molecular_cache else None,
        "species_threshold": float(args.species_threshold),
        "high_error_quantile": float(args.high_error_quantile),
        "uncertainty_tier": "disabled",
        "reference_definition": "aquatic source train plus soil target finetune pool for the matching random split; test fold excluded",
        "finetune_validation_note": "seed-specific finetune_validation is not separated; coverage uses the target-domain finetune pool.",
        "inputs": manifest_inputs,
        "outputs": {
            "prediction_rows": str(combined_path),
            "axis_tiers": str(out_dir / "cst_ad_axis_tiers.csv"),
            "tier_summary": str(out_dir / "cst_ad_tier_summary.csv"),
            "family_summary": str(out_dir / "cst_ad_family_summary.csv"),
            "variant_comparison": str(out_dir / "cst_ad_variant_comparison.csv"),
            "high_error_detection": str(out_dir / "cst_ad_high_error_detection.csv"),
            "failure_cases": str(out_dir / "cst_ad_failure_cases.csv"),
            "embedding_diagnostic": str(embedding_path) if embedding_path else None,
        },
    }
    (out_dir / "cst_ad_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def silence_rdkit_logs() -> None:
    try:
        from rdkit import RDLogger
    except Exception:
        return
    RDLogger.DisableLog("rdApp.*")


def needed_reference_columns() -> tuple[str, ...]:
    return (
        "aggregate_id",
        "task_head",
        "task_family",
        "target_name",
        "target_family",
        "effect_family",
        "medium_domain",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
        "species_number",
        "latin_name",
        "kingdom",
        "phylum",
        "class_name",
        "tax_order",
        "family",
        "genus",
        "species",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "organism_lifestage",
    )


def read_table(db_path: Path, table: str, *, columns: Iterable[str]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        available = {
            str(row[1])
            for row in conn.execute(f'PRAGMA table_info("{table}")')
        }
        selected = [column for column in columns if column in available]
        if not selected:
            raise ValueError(f"No requested columns are present in table: {table}")
        quoted = ", ".join(f'"{column}"' for column in selected)
        return pd.read_sql_query(f'SELECT {quoted} FROM "{table}" ORDER BY aggregate_id', conn)


def build_random_reference_specs(
    *,
    transfer_frame: pd.DataFrame,
    soil_frame: pd.DataFrame,
    transfer_table: str,
    soil_table: str,
    seed: int,
) -> list[ReferenceSpec]:
    records = soil_frame.astype(object).where(pd.notnull(soil_frame), None).to_dict(orient="records")
    split_sets: dict[str, Any] = {}
    split_sets.update(build_experiment_split_sets(records, code="B", source_table=soil_table, id_column="aggregate_id", seed=seed))
    split_sets.update(build_experiment_split_sets(records, code="E", source_table=soil_table, id_column="aggregate_id", seed=seed))

    specs: list[ReferenceSpec] = []
    specs.append(
        build_reference_spec(
            transfer_frame=transfer_frame,
            assignments=split_sets["B_random_8_2"],
            split_policy="random_8_2",
            fold="holdout",
            transfer_split_name="M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100",
            soil_split_name="SoilPtoxQC2_B_random_8_2",
        )
    )
    for fold in range(1, 6):
        name = f"E_random_5fold_fold{fold}"
        specs.append(
            build_reference_spec(
                transfer_frame=transfer_frame,
                assignments=split_sets[name],
                split_policy="random_5fold",
                fold=f"fold{fold}",
                transfer_split_name=f"M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold{fold}_f100",
                soil_split_name=f"SoilPtoxQC2_E_random_5fold_fold{fold}",
            )
        )
    return specs


def build_reference_spec(
    *,
    transfer_frame: pd.DataFrame,
    assignments: Iterable[Any],
    split_policy: str,
    fold: str,
    transfer_split_name: str,
    soil_split_name: str,
) -> ReferenceSpec:
    train_soil_ids = {
        str(item.aggregate_id)
        for item in assignments
        if str(item.split_part).lower() == "train" and item.aggregate_id is not None
    }
    aggregate_id = normalize_text_series(transfer_frame["aggregate_id"])
    medium = normalize_text_series(transfer_frame.get("medium_domain", pd.Series("", index=transfer_frame.index)))
    target = normalize_text_series(transfer_frame.get("target_name", pd.Series("", index=transfer_frame.index)))
    aquatic = transfer_frame.loc[medium.eq("aquatic") & target.eq("ptox_mol_l")].copy()
    soil_finetune = transfer_frame.loc[
        medium.eq("soil") & target.eq("ptox_mol_l") & aggregate_id.isin(train_soil_ids)
    ].copy()
    aquatic["cst_reference_split_part"] = "train"
    soil_finetune["cst_reference_split_part"] = "finetune"
    reference_frame = pd.concat([aquatic, soil_finetune], ignore_index=True)
    return ReferenceSpec(
        split_policy=split_policy,
        fold=fold,
        transfer_split_name=transfer_split_name,
        soil_split_name=soil_split_name,
        reference_frame=reference_frame,
    )


def prediction_file(prediction_dir: Path, split_policy: str, fold: str) -> Path:
    if split_policy == "random_8_2":
        return prediction_dir / "split_policy_ensemble_random_8_2_holdout_prediction_rows.csv"
    safe_fold = fold.replace(";", "_")
    return prediction_dir / f"split_policy_ensemble_random_5fold_{safe_fold}_prediction_rows.csv"


def build_cst_rows(
    prediction_rows: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame,
    split_policy: str,
    fold: str,
    fingerprint_size: int,
    molecular_cache: str | None,
    pca_components: int,
    chemical_threshold: float,
    chemical_high_threshold: float,
    species_threshold: float,
) -> pd.DataFrame:
    rows = prediction_rows.copy()
    rows = rows[rows["split_part"].astype(str).str.lower().eq("test")].copy()
    drop_member_columns = [
        column for column in rows.columns if column.startswith("y_pred_member_") or column.startswith("y_pred_scaled_member_")
    ]
    rows = rows.drop(columns=drop_member_columns, errors="ignore")
    rows["split_policy"] = split_policy
    rows["fold"] = fold
    rows["cst_reference_rows"] = int(len(reference_frame))
    rows["cst_reference_train_rows"] = int(
        (reference_frame.get("cst_reference_split_part", pd.Series("", index=reference_frame.index)).astype(str) == "train").sum()
    )
    rows["cst_reference_finetune_rows"] = int(
        (reference_frame.get("cst_reference_split_part", pd.Series("", index=reference_frame.index)).astype(str) == "finetune").sum()
    )

    chemical = chemical_ad_scores(
        query_frame=rows,
        reference_frame=reference_frame,
        fingerprint_size=fingerprint_size,
        molecular_cache=molecular_cache,
        pca_components=pca_components,
    )
    for column, values in chemical.items():
        rows[column] = values
    rows["cst_chemical_threshold"] = chemical_threshold
    rows["cst_chemical_high_threshold"] = chemical_high_threshold
    rows["cst_chemical_in_domain"] = rows["cst_chemical_score"] >= chemical_threshold
    rows["cst_chemical_tier"] = [
        chemical_tier(score, leverage_ok, missing, chemical_threshold=chemical_threshold, chemical_high_threshold=chemical_high_threshold)
        for score, leverage_ok, missing in zip(
            rows["cst_chemical_score"],
            rows["cst_chemical_in_domain_williams"],
            rows["cst_chemical_missing"],
        )
    ]

    species = species_ad_scores(rows, reference_frame=reference_frame, species_threshold=species_threshold)
    for column, values in species.items():
        rows[column] = values

    task = task_ad_scores(rows, reference_frame=reference_frame)
    for column, values in task.items():
        rows[column] = values

    rows["cst_c_only_tier"] = [variant_tier((chem,), preferred=False) for chem in rows["cst_chemical_tier"]]
    rows["cst_cs_tier"] = [
        assign_overall_tier(chem, species, "T2", variant="C+S")[0]
        for chem, species in zip(rows["cst_chemical_tier"], rows["cst_species_tier"])
    ]
    assigned = [
        assign_overall_tier(chem, species, task, variant="C+S+T")
        for chem, species, task in zip(rows["cst_chemical_tier"], rows["cst_species_tier"], rows["cst_task_tier"])
    ]
    rows["cst_ad_tier"] = [tier for tier, _reason in assigned]
    rows["cst_ad_reason"] = [reason for _tier, reason in assigned]
    rows["cst_uncertainty_tier"] = "disabled"
    return rows


def chemical_ad_scores(
    *,
    query_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    fingerprint_size: int,
    molecular_cache: str | None,
    pca_components: int,
) -> dict[str, list[Any]]:
    reference_smiles = unique_nonempty_smiles(reference_frame.get("smiles", pd.Series(dtype=object)))
    query_smiles = unique_nonempty_smiles(query_frame.get("smiles", pd.Series(dtype=object)))
    all_smiles = reference_smiles + [smiles for smiles in query_smiles if smiles not in set(reference_smiles)]
    if not reference_smiles or not query_smiles:
        n = len(query_frame)
        return {
            "cst_chemical_score": [0.0] * n,
            "cst_williams_leverage": [math.nan] * n,
            "cst_williams_critical_h": [math.nan] * n,
            "cst_chemical_in_domain_williams": [False] * n,
            "cst_chemical_missing": [True] * n,
        }

    cache_path = molecular_cache if molecular_cache and Path(molecular_cache).exists() else None
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size, cache_path=cache_path)
    descriptors, fingerprints = encode_smiles(all_smiles, encoder)
    ref_count = len(reference_smiles)
    train_mask = np.zeros(len(all_smiles), dtype=bool)
    train_mask[:ref_count] = True
    tanimoto = max_tanimoto(fingerprints[ref_count:], fingerprints[:ref_count])
    leverage, critical_h, _components = williams_leverage(
        descriptors,
        fingerprints,
        train_mask,
        pca_components=pca_components,
    )
    score_by_smiles = {smiles: 1.0 for smiles in reference_smiles}
    score_by_smiles.update({smiles: float(score) for smiles, score in zip(all_smiles[ref_count:], tanimoto)})
    leverage_by_smiles = {smiles: float(value) for smiles, value in zip(all_smiles, leverage)}

    scores = []
    leverages = []
    missing = []
    for value in query_frame.get("smiles", pd.Series("", index=query_frame.index)):
        smiles = normalize_smiles(value)
        is_missing = not bool(smiles)
        scores.append(float(score_by_smiles.get(smiles, 0.0)))
        leverages.append(float(leverage_by_smiles.get(smiles, math.nan)))
        missing.append(is_missing)
    return {
        "cst_chemical_score": scores,
        "cst_williams_leverage": leverages,
        "cst_williams_critical_h": [float(critical_h)] * len(query_frame),
        "cst_chemical_in_domain_williams": [
            bool(math.isfinite(value) and value <= critical_h)
            for value in leverages
        ],
        "cst_chemical_missing": missing,
    }


def unique_nonempty(values: pd.Series) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = normalize_scalar(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def unique_nonempty_smiles(values: pd.Series) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = normalize_smiles(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def encode_smiles(smiles_values: list[str], encoder: MolecularFeatureBuilder) -> tuple[np.ndarray, np.ndarray]:
    descriptors: list[list[float]] = []
    fingerprints: list[list[float]] = []
    for smiles in smiles_values:
        descriptor_row, fingerprint_row = encoder.encode(smiles)
        descriptors.append([safe_float(value) for value in descriptor_row])
        fingerprints.append([1.0 if safe_float(value) > 0.0 else 0.0 for value in fingerprint_row])
    return np.asarray(descriptors, dtype=float), np.asarray(fingerprints, dtype=float)


def max_tanimoto(query: np.ndarray, reference: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    ref = reference.astype(bool)
    qry = query.astype(bool)
    ref_counts = ref.sum(axis=1).astype(float)
    ref_float = ref.astype(float)
    result = np.zeros(qry.shape[0], dtype=float)
    for start in range(0, qry.shape[0], batch_size):
        batch = qry[start : start + batch_size]
        intersections = batch.astype(float) @ ref_float.T
        batch_counts = batch.sum(axis=1).astype(float)[:, None]
        denominators = batch_counts + ref_counts[None, :] - intersections
        sims = np.divide(intersections, denominators, out=np.zeros_like(intersections), where=denominators > 0)
        result[start : start + batch_size] = sims.max(axis=1)
    return result


def chemical_tier(
    score: Any,
    williams_in_domain: Any,
    missing: Any,
    *,
    chemical_threshold: float,
    chemical_high_threshold: float,
) -> str:
    if bool(missing) or not math.isfinite(safe_float(score)):
        return "C2"
    numeric = safe_float(score)
    if numeric >= chemical_high_threshold:
        return "C0"
    if numeric >= chemical_threshold or bool(williams_in_domain):
        return "C1"
    return "C2"


def species_ad_scores(
    query_frame: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame,
    species_threshold: float,
) -> dict[str, list[Any]]:
    ref_profiles = normalized_profiles(reference_frame, TAXON_COLUMNS)
    prefix_sets = {
        level: {
            profile[:level]
            for profile in ref_profiles
            if len(profile) >= level and all(profile[:level])
        }
        for level in range(1, len(TAXON_COLUMNS) + 1)
    }
    ref_species = normalized_set(reference_frame, "latin_name") | normalized_set(reference_frame, "species_number")
    ref_genus = normalized_set(reference_frame, "genus")
    ref_family = normalized_set(reference_frame, "family")
    ref_order = normalized_set(reference_frame, "tax_order")
    ref_lifestage = normalized_set(reference_frame, "organism_lifestage")
    species_task_family = set(zip(species_keys(reference_frame), key_series(reference_frame, ("task_family", "target_family", "effect_family"))))
    species_task_head = set(zip(species_keys(reference_frame), key_series(reference_frame, ("task_head", "target_name"))))

    similarities: list[float] = []
    species_seen: list[bool] = []
    genus_seen: list[bool] = []
    family_seen: list[bool] = []
    order_seen: list[bool] = []
    lifestage_seen: list[bool] = []
    species_task_family_seen: list[bool] = []
    species_task_head_seen: list[bool] = []
    tiers: list[str] = []
    query_profiles = normalized_profiles(query_frame, TAXON_COLUMNS)
    query_species_keys = species_keys(query_frame)
    query_task_family = key_series(query_frame, ("task_family", "target_family", "effect_family"))
    query_task_head = key_series(query_frame, ("task_head", "target_name"))

    for idx, profile in enumerate(query_profiles):
        matched = 0
        for level in range(1, len(TAXON_COLUMNS) + 1):
            prefix = profile[:level]
            if all(prefix) and prefix in prefix_sets[level]:
                matched = level
        similarity = matched / max(len(TAXON_COLUMNS), 1)
        similarities.append(float(similarity))
        species_key = query_species_keys.iloc[idx]
        latin = normalize_row_value(query_frame, idx, "latin_name")
        species_number = normalize_row_value(query_frame, idx, "species_number")
        species_ok = bool((latin and latin in ref_species) or (species_number and species_number in ref_species))
        genus_ok = normalize_row_value(query_frame, idx, "genus") in ref_genus
        family_ok = normalize_row_value(query_frame, idx, "family") in ref_family
        order_ok = normalize_row_value(query_frame, idx, "tax_order") in ref_order
        lifestage_ok = normalize_row_value(query_frame, idx, "organism_lifestage") in ref_lifestage
        stf_ok = bool((species_key, query_task_family.iloc[idx]) in species_task_family)
        sth_ok = bool((species_key, query_task_head.iloc[idx]) in species_task_head)
        species_seen.append(species_ok)
        genus_seen.append(genus_ok)
        family_seen.append(family_ok)
        order_seen.append(order_ok)
        lifestage_seen.append(lifestage_ok)
        species_task_family_seen.append(stf_ok)
        species_task_head_seen.append(sth_ok)
        tiers.append(assign_species_tier(stf_ok, species_ok, genus_ok, family_ok, order_ok, similarity, species_threshold))
    return {
        "cst_species_score": similarities,
        "cst_species_threshold": [float(species_threshold)] * len(query_frame),
        "cst_species_seen_train": species_seen,
        "cst_genus_seen_train": genus_seen,
        "cst_family_seen_train": family_seen,
        "cst_order_seen_train": order_seen,
        "cst_life_stage_seen_train": lifestage_seen,
        "cst_species_task_family_seen_train": species_task_family_seen,
        "cst_species_task_head_seen_train": species_task_head_seen,
        "cst_species_tier": tiers,
    }


def assign_species_tier(
    species_task_family_seen: bool,
    species_seen: bool,
    genus_seen: bool,
    family_seen: bool,
    order_seen: bool,
    taxonomy_similarity: float,
    species_threshold: float,
) -> str:
    if species_task_family_seen:
        return "S0"
    if species_seen:
        return "S1"
    if genus_seen or family_seen:
        return "S2"
    if order_seen or taxonomy_similarity >= species_threshold:
        return "S3"
    return "S4"


def task_ad_scores(query_frame: pd.DataFrame, *, reference_frame: pd.DataFrame) -> dict[str, list[Any]]:
    ref_keys = key_frame(reference_frame)
    query_keys = key_frame(query_frame)
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
    output: dict[str, list[Any]] = {}
    for name, columns in count_specs.items():
        counts = grouped_counts(ref_keys, columns)
        output[f"cst_{name}"] = lookup_counts(query_keys, counts, columns)
    tiers = []
    labels = []
    for idx in range(len(query_frame)):
        row = {key: values[idx] for key, values in output.items()}
        tier, label = assign_task_tier(row)
        tiers.append(tier)
        labels.append(label)
    output["cst_task_tier"] = tiers
    output["cst_task_coverage_label"] = labels
    output["cst_task_head_seen_train"] = [value > 0 for value in output["cst_train_n_task_head"]]
    output["cst_task_family_seen_train"] = [value > 0 for value in output["cst_train_n_task_family"]]
    return output


def assign_task_tier(row: dict[str, int]) -> tuple[str, str]:
    if int(row.get("cst_train_n_cst_exact", 0)) > 0:
        return "T0", "chemical_species_task_seen"
    if int(row.get("cst_train_n_species_task_head", 0)) > 0:
        return "T1", "species_task_head_seen"
    if int(row.get("cst_train_n_species_task_family", 0)) > 0:
        return "T1", "species_task_family_seen"
    if int(row.get("cst_train_n_task_head", 0)) > 0 or int(row.get("cst_train_n_task_family", 0)) > 0:
        return "T2", "task_seen_only"
    return "T3", "task_unseen"


def assign_overall_tier(chemical_tier_value: str, species_tier_value: str, task_tier_value: str, *, variant: str) -> tuple[str, str]:
    high = []
    if chemical_tier_value == "C2":
        high.append("chemical_low_similarity")
    if species_tier_value == "S4":
        high.append("species_low_coverage")
    if variant == "C+S+T" and task_tier_value == "T3":
        high.append("task_unseen")
    if len(high) >= 2:
        return "AD-D", "+".join(high)
    if len(high) == 1:
        return "AD-C", high[0]
    if variant == "C+S" and chemical_tier_value in {"C0", "C1"} and species_tier_value in {"S0", "S1"}:
        return "AD-A", "chemical_and_species_strong"
    if variant == "C+S+T" and chemical_tier_value in {"C0", "C1"} and species_tier_value in {"S0", "S1"} and task_tier_value in {"T0", "T1"}:
        return "AD-A", "chemical_species_task_strong"
    return "AD-B", "moderate_coverage"


def variant_tier(axis_tiers: tuple[str, ...], *, preferred: bool) -> str:
    if axis_tiers == ("C0",):
        return "AD-A"
    if axis_tiers == ("C1",):
        return "AD-B"
    if axis_tiers == ("C2",):
        return "AD-C"
    return "AD-B" if preferred else "AD-C"


def key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    task_head = key_series(frame, ("task_head", "target_name"))
    task_family = key_series(frame, ("task_family", "target_family", "effect_family"))
    return pd.DataFrame(
        {
            "chemical_key": key_series(frame, ("dtxsid", "cas_number", "smiles", "chemical_name")),
            "species_key": species_keys(frame),
            "task_head_key": task_head,
            "task_family_key": task_family.mask(task_family.eq("") & task_head.ne(""), task_head.str.split("_", n=1).str[0]),
        },
        index=frame.index,
    )


def key_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        normalized = normalize_text_series(frame[column])
        values = values.mask(values.eq("") & normalized.ne(""), normalized)
    return values.reset_index(drop=True)


def species_keys(frame: pd.DataFrame) -> pd.Series:
    return key_series(frame, ("latin_name", "species_number"))


def normalized_profiles(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[str, ...]]:
    normalized = [normalize_text_series(frame[column]) if column in frame.columns else pd.Series("", index=frame.index) for column in columns]
    return [tuple(values) for values in zip(*[series.tolist() for series in normalized])]


def normalized_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    return {value for value in normalize_text_series(frame[column]).tolist() if value}


def normalize_text_series(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().str.lower().replace({"<na>": "", "nan": "", "none": ""})


def normalize_row_value(frame: pd.DataFrame, idx: int, column: str) -> str:
    if column not in frame.columns:
        return ""
    return normalize_scalar(frame.iloc[idx][column])


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text.lower()


def normalize_smiles(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def grouped_counts(keys: pd.DataFrame, columns: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    usable = keys.loc[:, list(columns)].copy()
    usable = usable[(usable != "").all(axis=1)]
    if usable.empty:
        return {}
    grouped = usable.groupby(list(columns), dropna=False).size()
    return {tuple(key if isinstance(key, tuple) else (key,)): int(value) for key, value in grouped.items()}


def lookup_counts(keys: pd.DataFrame, counts: dict[tuple[str, ...], int], columns: tuple[str, ...]) -> list[int]:
    values: list[int] = []
    for row in keys.loc[:, list(columns)].itertuples(index=False, name=None):
        if not all(row):
            values.append(0)
        else:
            values.append(int(counts.get(tuple(row), 0)))
    return values


def summarize_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, tier), sub in frame.groupby(["split_policy", "cst_ad_tier"], dropna=False, sort=True):
        metrics = metrics_from_prediction_rows(dataframe_to_rows(sub))
        policy_total = int((frame["split_policy"].astype(str) == str(policy)).sum())
        rows.append(
            {
                "split_policy": policy,
                "cst_ad_tier": tier,
                "coverage_n": int(len(sub)),
                "coverage_fraction": float(len(sub) / policy_total) if policy_total else math.nan,
                **metrics,
                "median_abs_error": float(pd.to_numeric(sub["abs_error"], errors="coerce").median()),
                "mean_chemical_score": float(pd.to_numeric(sub["cst_chemical_score"], errors="coerce").mean()),
                "mean_species_score": float(pd.to_numeric(sub["cst_species_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_family_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, family, tier), sub in frame.groupby(["split_policy", "task_family", "cst_ad_tier"], dropna=False, sort=True):
        metrics = metrics_from_prediction_rows(dataframe_to_rows(sub))
        family_total = int((frame["split_policy"].astype(str).eq(str(policy)) & frame["task_family"].astype(str).eq(str(family))).sum())
        rows.append(
            {
                "split_policy": policy,
                "task_family": family,
                "cst_ad_tier": tier,
                "coverage_n": int(len(sub)),
                "coverage_fraction": float(len(sub) / family_total) if family_total else math.nan,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_variants(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, column in VARIANT_COLUMNS.items():
        for (policy, tier), sub in frame.groupby(["split_policy", column], dropna=False, sort=True):
            metrics = metrics_from_prediction_rows(dataframe_to_rows(sub))
            policy_total = int((frame["split_policy"].astype(str) == str(policy)).sum())
            rows.append(
                {
                    "variant": variant,
                    "split_policy": policy,
                    "cst_ad_tier": tier,
                    "coverage_n": int(len(sub)),
                    "coverage_fraction": float(len(sub) / policy_total) if policy_total else math.nan,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_high_error_detection(frame: pd.DataFrame, *, high_error_quantile: float) -> pd.DataFrame:
    data = frame.copy()
    data["cst_abs_error_numeric"] = pd.to_numeric(data["abs_error"], errors="coerce")
    thresholds = data.groupby(["split_policy", "task_family"])["cst_abs_error_numeric"].transform(
        lambda values: values.quantile(high_error_quantile)
    )
    data["cst_high_error"] = data["cst_abs_error_numeric"] >= thresholds
    rows: list[dict[str, Any]] = []
    for variant, column in VARIANT_COLUMNS.items():
        for policy, sub in data.groupby("split_policy", sort=True):
            predicted_high = sub[column].astype(str).isin({"AD-C", "AD-D"}).to_numpy(dtype=bool)
            truth = sub["cst_high_error"].to_numpy(dtype=bool)
            score = sub[column].astype(str).map({"AD-A": 0.0, "AD-B": 1.0, "AD-C": 2.0, "AD-D": 3.0}).fillna(1.0).to_numpy(dtype=float)
            tp = int(np.logical_and(predicted_high, truth).sum())
            fp = int(np.logical_and(predicted_high, ~truth).sum())
            fn = int(np.logical_and(~predicted_high, truth).sum())
            precision = tp / (tp + fp) if tp + fp else math.nan
            recall = tp / (tp + fn) if tp + fn else math.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall and math.isfinite(precision) and math.isfinite(recall) else math.nan
            rows.append(
                {
                    "variant": variant,
                    "split_policy": policy,
                    "high_error_quantile": float(high_error_quantile),
                    "n": int(len(sub)),
                    "high_error_n": int(truth.sum()),
                    "predicted_high_risk_n": int(predicted_high.sum()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auroc": auroc(truth, score),
                    "average_precision": average_precision(truth, score),
                }
            )
    return pd.DataFrame(rows)


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true]
    negatives = scores[~y_true]
    if len(positives) == 0 or len(negatives) == 0:
        return math.nan
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for value in positives:
        wins += float((value > negatives).sum())
        wins += 0.5 * float((value == negatives).sum())
    return wins / total


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    truth = y_true[order]
    positives = int(truth.sum())
    if positives == 0:
        return math.nan
    tp = 0
    precisions: list[float] = []
    for idx, is_positive in enumerate(truth, start=1):
        if is_positive:
            tp += 1
            precisions.append(tp / idx)
    return float(np.mean(precisions)) if precisions else math.nan


def select_failure_cases(frame: pd.DataFrame, *, limit: int = 200) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "split_policy",
            "fold",
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
            "cst_chemical_tier",
            "cst_species_tier",
            "cst_task_tier",
            "cst_ad_tier",
            "cst_ad_reason",
        )
        if column in frame.columns
    ]
    ranked = frame.assign(_abs_error=pd.to_numeric(frame["abs_error"], errors="coerce")).sort_values("_abs_error", ascending=False)
    return ranked.loc[:, columns].head(limit).reset_index(drop=True)


def build_embedding_diagnostic(
    cst_frame: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame,
    embedding_lookup_path: Path,
    split_policy: str,
    fold: str,
) -> pd.DataFrame:
    if not embedding_lookup_path.exists():
        return pd.DataFrame()
    lookup = pd.read_csv(embedding_lookup_path, low_memory=False)
    emb_columns = [column for column in lookup.columns if column.startswith("emb_")]
    if not emb_columns:
        return pd.DataFrame()
    lookup = lookup.copy()
    lookup["species_key"] = species_keys(lookup)
    ref_keys = set(species_keys(reference_frame))
    query = cst_frame.copy()
    query["species_key"] = species_keys(query)
    ref_lookup = lookup[lookup["species_key"].isin(ref_keys)].drop_duplicates("species_key")
    query_lookup = query.merge(lookup.drop_duplicates("species_key"), on="species_key", how="left", suffixes=("", "_embedding"))
    if ref_lookup.empty or query_lookup.empty:
        return pd.DataFrame()
    ref_matrix = ref_lookup.loc[:, emb_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    query_matrix = query_lookup.loc[:, emb_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    distances = nearest_cosine_distance(query_matrix, ref_matrix)
    output = query.loc[
        :,
        [
            column
            for column in (
                "split_policy",
                "fold",
                "sample_id",
                "aggregate_id",
                "task_head",
                "task_family",
                "latin_name",
                "cst_species_tier",
                "cst_species_score",
                "abs_error",
            )
            if column in query.columns
        ],
    ].copy()
    output["split_policy"] = split_policy
    output["fold"] = fold
    output["species_key"] = query["species_key"].to_numpy()
    output["species_embedding_nearest_cosine_distance"] = distances
    output["embedding_reference_species_n"] = int(len(ref_lookup))
    output["embedding_lookup_source"] = str(embedding_lookup_path)
    return output


def nearest_cosine_distance(query: np.ndarray, reference: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    query = np.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
    reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
    ref_norm = np.linalg.norm(reference, axis=1)
    ref_norm = np.where(ref_norm > 1e-12, ref_norm, 1.0)
    ref_unit = reference / ref_norm[:, None]
    output = np.full(query.shape[0], np.nan, dtype=float)
    for start in range(0, query.shape[0], batch_size):
        batch = query[start : start + batch_size]
        norm = np.linalg.norm(batch, axis=1)
        usable = norm > 1e-12
        if not bool(usable.any()):
            continue
        batch_unit = batch[usable] / norm[usable, None]
        sims = batch_unit @ ref_unit.T
        output[start : start + batch_size][usable] = 1.0 - sims.max(axis=1)
    return output


def dataframe_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notnull(frame), "").to_dict(orient="records")


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


if __name__ == "__main__":
    main()
