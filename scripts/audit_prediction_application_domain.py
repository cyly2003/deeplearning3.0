from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsar_tl.evaluation.application_domain import (
    DEFAULT_TAXON_COLUMNS,
    ad_warning,
    williams_leverage,
)
from qsar_tl.training.baseline import load_split_frame
from qsar_tl.training.deep_experiment import MolecularFeatureBuilder
from scripts.summarize_deep_runs import focus_prediction_rows, metrics_from_prediction_rows


EVAL_SPLITS = {"test", "validation", "finetune_validation"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prediction-level chemical and species AD audit outputs.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--molecular-cache", default=None)
    parser.add_argument("--fingerprint-size", type=int, default=512)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--tanimoto-threshold", type=float, default=0.5)
    parser.add_argument("--proxy-distance-threshold", type=float, default=1.5)
    parser.add_argument("--taxon-columns", default="kingdom,phylum,class_name,tax_order,family")
    parser.add_argument("--taxon-similarity-threshold", type=float, default=0.8)
    args = parser.parse_args()

    predictions = read_csv(Path(args.predictions))
    focus_rows = [row for split in EVAL_SPLITS for row in focus_prediction_rows(predictions, split_part=split)]
    if not focus_rows:
        raise ValueError(f"No soil ECx/NOEC/LOEC evaluation predictions found: {args.predictions}")
    requested_ids = {str(row.get("aggregate_id", "") or row.get("sample_id", "")) for row in focus_rows}
    requested_ids.discard("")

    frame = load_split_frame(args.db, split_name=args.split_name, source_table=args.source_table)
    if "aggregate_id" not in frame.columns:
        raise ValueError("AD audit requires aggregate_id in the source table.")
    train_frame = frame[frame["split_part"].astype("string").str.lower() == "train"].copy()
    query_frame = frame[frame["aggregate_id"].astype("string").isin(requested_ids)].copy()
    if train_frame.empty or query_frame.empty:
        raise ValueError("AD audit requires train reference rows and query prediction rows.")

    taxon_columns = tuple(item.strip() for item in args.taxon_columns.split(",") if item.strip()) or DEFAULT_TAXON_COLUMNS
    ad_rows = build_ad_rows(
        train_frame,
        query_frame,
        fingerprint_size=int(args.fingerprint_size),
        cache_path=args.molecular_cache,
        pca_components=int(args.pca_components),
        tanimoto_threshold=float(args.tanimoto_threshold),
        proxy_distance_threshold=float(args.proxy_distance_threshold),
        taxon_columns=taxon_columns,
        taxon_similarity_threshold=float(args.taxon_similarity_threshold),
    )
    ad_by_id = {str(row.get("aggregate_id", "")): row for row in ad_rows}
    merged = []
    for row in focus_rows:
        key = str(row.get("aggregate_id", "") or row.get("sample_id", ""))
        ad = ad_by_id.get(key, {})
        if not ad:
            continue
        merged.append({**row, **{f"ad_{key}": value for key, value in ad.items() if key not in row}})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "chemical_ad_metrics.csv", chemical_ad_metrics(merged))
    write_csv(out_dir / "species_ad_metrics.csv", species_ad_metrics(merged))
    write_csv(out_dir / "ad_stratified_metrics.csv", stratified_ad_metrics(merged))
    write_csv(out_dir / "ad_failure_cases.csv", failure_cases(merged))
    write_csv(out_dir / "ad_prediction_rows.csv", merged)
    manifest = {
        "db": str(args.db),
        "source_table": args.source_table,
        "split_name": args.split_name,
        "predictions": str(args.predictions),
        "focus_prediction_rows": len(focus_rows),
        "ad_rows": len(ad_rows),
        "merged_rows": len(merged),
        "tanimoto_threshold": float(args.tanimoto_threshold),
        "proxy_distance_threshold": float(args.proxy_distance_threshold),
        "taxon_columns": list(taxon_columns),
        "taxon_similarity_threshold": float(args.taxon_similarity_threshold),
    }
    (out_dir / "ad_audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_ad_rows(
    train_frame: Any,
    query_frame: Any,
    *,
    fingerprint_size: int,
    cache_path: str | None,
    pca_components: int,
    tanimoto_threshold: float,
    proxy_distance_threshold: float,
    taxon_columns: tuple[str, ...],
    taxon_similarity_threshold: float,
) -> list[dict[str, Any]]:
    import pandas as pd

    train_chemical_frame = unique_chemical_reference_frame(train_frame)
    combined = pd.concat([train_chemical_frame, query_frame], ignore_index=True)
    train_mask = np.zeros(len(combined), dtype=bool)
    train_mask[: len(train_chemical_frame)] = True
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size, cache_path=cache_path)
    print(
        "[ad-audit] "
        f"train_rows={len(train_frame)} unique_chemical_refs={len(train_chemical_frame)} "
        f"query_rows={len(query_frame)}",
        flush=True,
    )
    descriptor_matrix, fingerprint_matrix, _ = build_molecular_matrices_for_encoder(combined, encoder)
    print("[ad-audit] molecular_matrices_ready", flush=True)
    leverage, critical_h, pca_used = williams_leverage(
        descriptor_matrix,
        fingerprint_matrix,
        train_mask,
        pca_components=pca_components,
    )
    print("[ad-audit] williams_ready", flush=True)
    query_start = len(train_frame)
    query_start = len(train_chemical_frame)
    query_fingerprint_matrix = fingerprint_matrix[query_start:]
    query_descriptor_matrix = descriptor_matrix[query_start:]
    tanimoto = max_tanimoto_to_unique_reference(
        query_fingerprint_matrix,
        fingerprint_matrix[train_mask],
        batch_size=64,
    )
    print("[ad-audit] tanimoto_ready", flush=True)
    proxy_distance = proxy_distance_to_reference(
        query_descriptor_matrix,
        descriptor_matrix[train_mask],
        batch_size=128,
    )
    print("[ad-audit] proxy_distance_ready", flush=True)
    taxon_similarity = query_max_taxon_similarity_to_train(train_frame, query_frame, taxon_columns)
    context = query_species_context_flags(train_frame, query_frame)
    print("[ad-audit] species_ad_ready", flush=True)
    rows = []
    for query_offset, combined_idx in enumerate(range(query_start, len(combined))):
        row = query_frame.iloc[query_offset]
        chemical_ok = bool(leverage[combined_idx] <= critical_h or tanimoto[query_offset] >= tanimoto_threshold)
        species_ok = bool(taxon_similarity[query_offset] >= taxon_similarity_threshold)
        payload = {
            "aggregate_id": row.get("aggregate_id", ""),
            "split_part": row.get("split_part", ""),
            "task_head": row.get("task_head", ""),
            "task_family": row.get("task_family", ""),
            "medium_domain": row.get("medium_domain", ""),
            "cas_number": row.get("cas_number", ""),
            "chemical_name": row.get("chemical_name", ""),
            "species_number": row.get("species_number", ""),
            "latin_name": row.get("latin_name", ""),
            "williams_leverage": float(leverage[combined_idx]),
            "williams_critical_h": float(critical_h),
            "pca_components_used": int(pca_used),
            "chemical_in_domain_williams": bool(leverage[combined_idx] <= critical_h),
            "max_tanimoto_to_train": float(tanimoto[query_offset]),
            "chemical_in_domain_tanimoto": bool(tanimoto[query_offset] >= tanimoto_threshold),
            "proxy_distance_to_train": float(proxy_distance[query_offset]),
            "chemical_in_domain_proxy": bool(proxy_distance[query_offset] <= proxy_distance_threshold),
            "chemical_in_domain_either": chemical_ok,
            "max_taxon_similarity_to_train": float(taxon_similarity[query_offset]),
            "taxon_distance_to_train": float(1.0 - taxon_similarity[query_offset]),
            "species_in_domain_taxon": species_ok,
            "overall_in_domain": chemical_ok and species_ok,
            "ad_warning": ad_warning(chemical_ok, species_ok),
        }
        for key, values in context.items():
            payload[key] = bool(values[query_offset])
        rows.append(payload)
    return rows


def unique_chemical_reference_frame(frame: Any) -> Any:
    for column in ("smiles", "cas_number", "dtxsid"):
        if column in frame.columns:
            return frame.drop_duplicates(subset=[column], keep="first").copy()
    return frame.copy()


def query_max_taxon_similarity_to_train(train_frame: Any, query_frame: Any, taxon_columns: tuple[str, ...]) -> np.ndarray:
    train_prefixes: dict[int, set[tuple[str, ...]]] = {level: set() for level in range(1, len(taxon_columns) + 1)}
    for profile in normalized_taxon_profiles(train_frame, taxon_columns):
        for level in range(1, len(profile) + 1):
            prefix = profile[:level]
            if all(prefix):
                train_prefixes[level].add(prefix)
    values = []
    max_level = max(len(taxon_columns), 1)
    for profile in normalized_taxon_profiles(query_frame, taxon_columns):
        matched = 0
        for level in range(1, len(profile) + 1):
            prefix = profile[:level]
            if all(prefix) and prefix in train_prefixes[level]:
                matched = level
        values.append(matched / max_level)
    return np.asarray(values, dtype=float)


def query_species_context_flags(train_frame: Any, query_frame: Any) -> dict[str, list[bool]]:
    latin_seen = normalized_set(train_frame, "latin_name")
    genus_seen = normalized_set(train_frame, "genus")
    family_seen = normalized_set(train_frame, "family")
    order_seen = normalized_set(train_frame, "tax_order")
    lifestage_seen = normalized_set(train_frame, "organism_lifestage")
    train_latin = normalized_column(train_frame, "latin_name")
    train_task_family = normalized_column(train_frame, "task_family")
    species_task_seen = {
        (latin, family)
        for latin, family in zip(train_latin, train_task_family)
        if latin and family
    }
    query_latin = normalized_column(query_frame, "latin_name")
    query_task_family = normalized_column(query_frame, "task_family")
    return {
        "species_seen_train": [value in latin_seen for value in query_latin],
        "genus_seen_train": [value in genus_seen for value in normalized_column(query_frame, "genus")],
        "family_seen_train": [value in family_seen for value in normalized_column(query_frame, "family")],
        "order_seen_train": [value in order_seen for value in normalized_column(query_frame, "tax_order")],
        "life_stage_seen_train": [
            value in lifestage_seen for value in normalized_column(query_frame, "organism_lifestage")
        ],
        "species_task_family_seen_train": [
            (latin, family) in species_task_seen for latin, family in zip(query_latin, query_task_family)
        ],
    }


def normalized_taxon_profiles(frame: Any, taxon_columns: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    if not taxon_columns:
        return []
    columns = [normalized_column(frame, column) for column in taxon_columns]
    return zip(*columns)


def normalized_column(frame: Any, column: str) -> list[str]:
    if column not in frame.columns:
        return [""] * len(frame)
    return frame[column].astype("string").fillna("").str.strip().str.lower().tolist()


def normalized_set(frame: Any, column: str) -> set[str]:
    return {value for value in normalized_column(frame, column) if value}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip().lower()


def max_tanimoto_to_unique_reference(
    query_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    train = np.unique(reference_matrix.astype(np.uint8), axis=0).astype(bool)
    query = query_matrix.astype(bool)
    if train.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=float)
    train_counts = train.sum(axis=1).astype(np.float32)
    train_float = train.astype(np.float32)
    maxima = np.zeros(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], batch_size):
        batch = query[start : start + batch_size].astype(np.float32)
        intersections = batch @ train_float.T
        batch_counts = batch.sum(axis=1).astype(np.float32)[:, None]
        denominators = batch_counts + train_counts[None, :] - intersections
        similarities = np.divide(intersections, denominators, out=np.zeros_like(intersections), where=denominators > 0)
        maxima[start : start + batch_size] = similarities.max(axis=1)
    return maxima.astype(float)


def proxy_distance_to_reference(
    query_descriptor_matrix: np.ndarray,
    reference_descriptor_matrix: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    if query_descriptor_matrix.shape[0] == 0:
        return np.asarray([], dtype=float)
    query_proxy = query_descriptor_matrix[:, :3].astype(float, copy=False)
    train_proxy = reference_descriptor_matrix[:, :3].astype(float, copy=False)
    proxy = np.vstack([train_proxy, query_proxy])
    proxy = np.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)
    train = np.unique(proxy[: len(train_proxy)], axis=0)
    if train.shape[0] == 0:
        return np.full(query_proxy.shape[0], np.inf, dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    train_scaled = (train - mean) / std
    query_scaled = (np.nan_to_num(query_proxy, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    output = np.zeros(query_scaled.shape[0], dtype=float)
    for start in range(0, query_scaled.shape[0], batch_size):
        batch = query_scaled[start : start + batch_size]
        diff = batch[:, None, :] - train_scaled[None, :, :]
        distances = np.sqrt(np.mean(diff * diff, axis=2))
        output[start : start + batch_size] = distances.min(axis=1)
    return output


def build_molecular_matrices_for_encoder(frame: Any, encoder: MolecularFeatureBuilder) -> tuple[np.ndarray, np.ndarray, str]:
    class Config:
        fingerprint_size: int
        molecular_cache_path: str | None

        def __init__(self, fingerprint_size: int, molecular_cache_path: str | None) -> None:
            self.fingerprint_size = fingerprint_size
            self.molecular_cache_path = molecular_cache_path

    # Reuse the tested matrix builder by passing an equivalent lightweight config.
    cfg = Config(encoder.fingerprint_size, None)
    descriptor_rows = []
    fingerprint_rows = []
    for smiles in frame.get("smiles", []):
        descriptors, fingerprint = encoder.encode(smiles)
        descriptor_rows.append([safe_float(value) for value in descriptors])
        fingerprint_rows.append([1.0 if safe_float(value) > 0.0 else 0.0 for value in fingerprint])
    return np.asarray(descriptor_rows, dtype=float), np.asarray(fingerprint_rows, dtype=float), encoder.source


def chemical_ad_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return grouped_metric_rows(
        rows,
        (
            "split_part",
            "ad_chemical_in_domain_either",
            "ad_chemical_in_domain_williams",
            "ad_chemical_in_domain_tanimoto",
            "ad_chemical_in_domain_proxy",
            "ad_ad_warning",
        ),
    )


def species_ad_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return grouped_metric_rows(
        rows,
        (
            "split_part",
            "ad_species_in_domain_taxon",
            "ad_species_seen_train",
            "ad_genus_seen_train",
            "ad_family_seen_train",
            "ad_order_seen_train",
            "ad_life_stage_seen_train",
            "ad_species_task_family_seen_train",
        ),
    )


def stratified_ad_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return grouped_metric_rows(rows, ("split_part", "ad_overall_in_domain", "ad_ad_warning", "task_family"))


def grouped_metric_rows(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(column, "")) for column in columns)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        output.append({**{column: value for column, value in zip(columns, key)}, **metrics_from_prediction_rows(items)})
    return output


def failure_cases(rows: list[dict[str, Any]], *, limit: int = 200) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get("abs_error", 0.0) or 0.0), reverse=True)
    columns = (
        "aggregate_id",
        "split_part",
        "task_head",
        "task_family",
        "medium_domain",
        "cas_number",
        "chemical_name",
        "species_number",
        "latin_name",
        "y_true",
        "y_pred",
        "abs_error",
        "ad_ad_warning",
        "ad_overall_in_domain",
        "ad_max_tanimoto_to_train",
        "ad_proxy_distance_to_train",
        "ad_chemical_in_domain_proxy",
        "ad_williams_leverage",
        "ad_williams_critical_h",
        "ad_max_taxon_similarity_to_train",
        "ad_species_seen_train",
        "ad_species_task_family_seen_train",
    )
    return [{column: row.get(column, "") for column in columns} for row in ranked[:limit]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


if __name__ == "__main__":
    main()
