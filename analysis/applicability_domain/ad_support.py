from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ad_chemistry import build_fingerprints


TAX_LEVELS = (
    ("higher", 1, ("phylum", "kingdom")),
    ("class", 2, ("class_name",)),
    ("order", 3, ("tax_order",)),
    ("family", 4, ("family",)),
    ("genus", 5, ("genus",)),
    ("species", 6, ("latin_name",)),
)
TAX_LABEL_BY_RANK = {0: "unseen", 1: "higher", 2: "class", 3: "order", 4: "family", 5: "genus", 6: "species"}
TAX_DISPLAY_BY_RANK = {0: 0.0, 1: 0.25, 2: 0.40, 3: 0.55, 4: 0.70, 5: 0.85, 6: 1.0}
SEMANTIC_MISSING_DEFAULT = {"", "na", "n/a", "nr", "not reported", "unknown", "<missing>"}


def normalize_category(series: pd.Series, missing_tokens: set[str] | None = None) -> pd.Series:
    missing_tokens = missing_tokens or SEMANTIC_MISSING_DEFAULT
    output = series.astype("string").fillna("").str.strip().str.casefold()
    return output.mask(output.isin(missing_tokens), "<missing>")


def taxonomy_support(
    evaluation: pd.DataFrame,
    training: pd.DataFrame,
    *,
    same_task: bool,
    missing_tokens: set[str],
) -> pd.DataFrame:
    output_rank = np.zeros(len(evaluation), dtype=np.int8)
    if same_task:
        groups = {
            str(task): group for task, group in training.groupby("model_head", sort=False)
        }
    else:
        groups = {"__global__": training}
    group_sets = {
        key: {
            field: set(normalize_category(group[field], missing_tokens)) - {"<missing>"}
            for _, _, fields in TAX_LEVELS
            for field in fields
        }
        for key, group in groups.items()
    }
    for index, (_, row) in enumerate(evaluation.iterrows()):
        key = str(row["model_head"]) if same_task else "__global__"
        reference_sets = group_sets.get(key)
        if reference_sets is None:
            continue
        best = 0
        for _, rank, fields in TAX_LEVELS:
            matched = False
            for field in fields:
                query_value = normalize_scalar(row.get(field), missing_tokens)
                if query_value == "<missing>":
                    continue
                if query_value in reference_sets[field]:
                    matched = True
                    break
            if matched:
                best = rank
        output_rank[index] = best
    label = "same_task" if same_task else "global"
    return pd.DataFrame(
        {
            f"tax_support_{label}_rank": output_rank,
            f"tax_support_{label}": [TAX_LABEL_BY_RANK[int(value)] for value in output_rank],
            f"tax_support_{label}_display": [TAX_DISPLAY_BY_RANK[int(value)] for value in output_rank],
        },
        index=evaluation.index,
    )


def pairwise_tax_rank(
    query: pd.DataFrame, reference: pd.DataFrame, *, missing_tokens: set[str]
) -> np.ndarray:
    ranks = np.zeros((len(query), len(reference)), dtype=np.int8)
    for _, rank, fields in TAX_LEVELS:
        matched_level = np.zeros_like(ranks, dtype=bool)
        for field in fields:
            left = normalize_category(query[field], missing_tokens).to_numpy(object)[:, None]
            right = normalize_category(reference[field], missing_tokens).to_numpy(object)[None, :]
            matched_level |= (left == right) & (left != "<missing>") & (right != "<missing>")
        ranks[matched_level] = rank
    return ranks


def fit_context_ranges(
    training: pd.DataFrame, numeric_fields: Iterable[str], *, duration_field: str = "duration_bin_h"
) -> dict[str, dict[str, float | str]]:
    output: dict[str, dict[str, float | str]] = {}
    for field in numeric_fields:
        values = pd.to_numeric(training[field], errors="coerce")
        transform = "log1p" if field == duration_field else "identity"
        transformed = np.log1p(values.clip(lower=0)) if transform == "log1p" else values
        valid = transformed.dropna()
        q05 = float(valid.quantile(0.05)) if not valid.empty else math.nan
        q95 = float(valid.quantile(0.95)) if not valid.empty else math.nan
        output[field] = {
            "transform": transform,
            "q05": q05,
            "q95": q95,
            "range": q95 - q05 if math.isfinite(q05) and math.isfinite(q95) else math.nan,
            "included": bool(math.isfinite(q05) and math.isfinite(q95) and q95 > q05),
        }
    return output


def context_distance_matrix(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    numeric_fields: list[str],
    categorical_fields: list[str],
    ranges: dict[str, dict[str, float | str]],
    missing_tokens: set[str],
) -> np.ndarray:
    distances: list[np.ndarray] = []
    for field in numeric_fields:
        info = ranges[field]
        if not info["included"]:
            continue
        left = pd.to_numeric(query[field], errors="coerce").to_numpy(float)
        right = pd.to_numeric(reference[field], errors="coerce").to_numpy(float)
        if info["transform"] == "log1p":
            left = np.log1p(np.clip(left, 0, None))
            right = np.log1p(np.clip(right, 0, None))
        left_missing = np.isnan(left)[:, None]
        right_missing = np.isnan(right)[None, :]
        distance = np.abs(left[:, None] - right[None, :]) / float(info["range"])
        distance = np.clip(distance, 0.0, 1.0)
        distance[left_missing & right_missing] = 0.0
        distance[left_missing ^ right_missing] = 1.0
        distances.append(distance)
    for field in categorical_fields:
        left = normalize_category(query[field], missing_tokens).to_numpy(object)[:, None]
        right = normalize_category(reference[field], missing_tokens).to_numpy(object)[None, :]
        distances.append((left != right).astype(float))
    if not distances:
        raise ValueError("No context fields remain after training-only range fitting")
    return np.mean(np.stack(distances, axis=0), axis=0)


def context_missing_fraction(
    frame: pd.DataFrame, fields: list[str], *, missing_tokens: set[str]
) -> np.ndarray:
    indicators = []
    for field in fields:
        normalized = normalize_category(frame[field], missing_tokens)
        indicators.append(normalized.eq("<missing>").to_numpy(float))
    return np.mean(np.stack(indicators, axis=1), axis=1)


def context_neighbor_features(
    distance: np.ndarray,
    *,
    k_values: Iterable[int],
    support_thresholds: Iterable[float],
) -> dict[str, np.ndarray]:
    ordered = np.sort(distance, axis=1)
    output: dict[str, np.ndarray] = {"context_nearest_distance": ordered[:, 0]}
    for k in sorted(set(int(value) for value in k_values)):
        actual = min(k, ordered.shape[1])
        output[f"context_top{k}_mean_distance"] = ordered[:, :actual].mean(axis=1)
        output[f"context_top{k}_actual_k"] = np.full(len(ordered), actual, dtype=int)
    for support in support_thresholds:
        output[f"n_context_support_ge_{support:.2f}"] = (distance <= 1.0 - support).sum(axis=1)
    return output


def chemical_similarity_matrix(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    radius: int,
    n_bits: int,
) -> np.ndarray:
    from rdkit import DataStructs

    canonicals = set(query.loc[query["structure_status"].eq("ok"), "canonical_parent"])
    canonicals.update(reference.loc[reference["structure_status"].eq("ok"), "canonical_parent"])
    fingerprints = build_fingerprints(canonicals, radius=radius, n_bits=n_bits)
    output = np.full((len(query), len(reference)), np.nan, dtype=float)
    reference_values = reference["canonical_parent"].astype(str).tolist()
    valid_reference = [index for index, value in enumerate(reference_values) if value in fingerprints]
    valid_fps = [fingerprints[reference_values[index]] for index in valid_reference]
    cache: dict[str, np.ndarray] = {}
    for index, (_, row) in enumerate(query.iterrows()):
        canonical = str(row["canonical_parent"] or "")
        if canonical not in fingerprints:
            continue
        if canonical not in cache:
            values = np.full(len(reference), np.nan, dtype=float)
            values[valid_reference] = DataStructs.BulkTanimotoSimilarity(
                fingerprints[canonical], valid_fps
            )
            cache[canonical] = values
        output[index] = cache[canonical]
    return output


def normalize_scalar(value: Any, missing_tokens: set[str]) -> str:
    if value is None:
        return "<missing>"
    try:
        if bool(pd.isna(value)):
            return "<missing>"
    except (TypeError, ValueError):
        pass
    text = str(value).strip().casefold()
    return "<missing>" if text in missing_tokens else text
