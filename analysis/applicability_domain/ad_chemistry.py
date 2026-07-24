from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ad_common import ROOT


def normalize_structure_table(frame: pd.DataFrame, *, smiles_column: str = "smiles") -> pd.DataFrame:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from rdkit import RDLogger
    from scripts.build_scaffold_cluster_splits import normalize_structure

    RDLogger.DisableLog("rdApp.*")
    cache: dict[str, dict[str, str]] = {}
    rows: list[dict[str, str]] = []
    for value in frame[smiles_column].fillna("").astype(str):
        if value not in cache:
            cache[value] = normalize_structure(value)
        item = cache[value]
        rows.append(
            {
                "raw_smiles": value,
                "canonical_parent": item["canonical_smiles"],
                "murcko_scaffold": item["scaffold_smiles"],
                "structure_status": "ok" if item["structure_status"] == "ok" else "unavailable",
                "structure_reason": "" if item["structure_status"] == "ok" else item["parse_error"],
            }
        )
    return pd.DataFrame(rows, index=frame.index)


def build_fingerprints(canonical_parents: Iterable[str], *, radius: int = 2, n_bits: int = 2048) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    output: dict[str, Any] = {}
    for canonical in sorted({str(value) for value in canonical_parents if str(value)}):
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise ValueError(f"Canonical parent cannot be reparsed: {canonical}")
        output[canonical] = generator.GetFingerprint(molecule)
    return output


def save_fingerprint_cache(path: Path, fingerprints: dict[str, Any], *, n_bits: int = 2048) -> None:
    from rdkit import DataStructs

    canonicals = sorted(fingerprints)
    packed = np.zeros((len(canonicals), n_bits // 8), dtype=np.uint8)
    for index, canonical in enumerate(canonicals):
        bit_string = DataStructs.BitVectToBinaryText(fingerprints[canonical])
        packed[index] = np.frombuffer(bit_string, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, canonical_parent=np.asarray(canonicals, dtype=object), packed_bits=packed)


def chemical_support(
    query: pd.DataFrame,
    reference_structures: pd.DataFrame,
    *,
    prefix: str,
    radius: int,
    n_bits: int,
    top_k: int,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    from rdkit import DataStructs

    reference = (
        reference_structures.loc[
            reference_structures["structure_status"].eq("ok"),
            ["canonical_parent", "murcko_scaffold"],
        ]
        .drop_duplicates("canonical_parent")
        .sort_values("canonical_parent")
        .reset_index(drop=True)
    )
    reference_fps = build_fingerprints(reference["canonical_parent"], radius=radius, n_bits=n_bits)
    reference_canonicals = reference["canonical_parent"].tolist()
    reference_fp_list = [reference_fps[value] for value in reference_canonicals]
    reference_scaffolds = set(reference["murcko_scaffold"].loc[reference["murcko_scaffold"].ne("")])
    query_fps = build_fingerprints(query["canonical_parent"], radius=radius, n_bits=n_bits)
    thresholds = list(thresholds)
    cache: dict[str, dict[str, Any]] = {}
    for canonical in sorted(set(query.loc[query["structure_status"].eq("ok"), "canonical_parent"])):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query_fps[canonical], reference_fp_list), dtype=float
        )
        order = np.argsort(-similarities)
        k = min(top_k, len(similarities))
        best = int(order[0])
        cache[canonical] = {
            "max": float(similarities[best]),
            "nearest": reference_canonicals[best],
            "topk": float(similarities[order[:k]].mean()),
            **{f"n_ge_{threshold:.2f}": int((similarities >= threshold).sum()) for threshold in thresholds},
        }
    rows: list[dict[str, Any]] = []
    reference_set = set(reference_canonicals)
    for _, item in query.iterrows():
        canonical = str(item["canonical_parent"] or "")
        if item["structure_status"] != "ok" or not canonical:
            row = {
                f"C_{prefix}": math.nan,
                f"nearest_{prefix}_canonical_parent": "",
                f"top{top_k}_mean_similarity_{prefix}": math.nan,
                f"exact_parent_seen_{prefix}": False,
                f"scaffold_seen_{prefix}": False,
            }
            row.update({f"n_{prefix}_chemical_ge_{threshold:.2f}": 0 for threshold in thresholds})
        else:
            values = cache[canonical]
            row = {
                f"C_{prefix}": values["max"],
                f"nearest_{prefix}_canonical_parent": values["nearest"],
                f"top{top_k}_mean_similarity_{prefix}": values["topk"],
                f"exact_parent_seen_{prefix}": canonical in reference_set,
                f"scaffold_seen_{prefix}": bool(item["murcko_scaffold"]) and item["murcko_scaffold"] in reference_scaffolds,
            }
            row.update(
                {
                    f"n_{prefix}_chemical_ge_{threshold:.2f}": values[f"n_ge_{threshold:.2f}"]
                    for threshold in thresholds
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, index=query.index)


def structure_summary(frame: pd.DataFrame, *, stage: str) -> dict[str, Any]:
    valid = frame["structure_status"].eq("ok")
    scaffolds = frame.loc[valid & frame["murcko_scaffold"].ne(""), "murcko_scaffold"]
    scaffold_sizes = scaffolds.value_counts()
    return {
        "stage": stage,
        "n_records": len(frame),
        "n_raw_chemical_ids": frame["raw_chemical_id"].nunique(dropna=True) if "raw_chemical_id" in frame else math.nan,
        "n_raw_smiles": frame["raw_smiles"].nunique(dropna=True),
        "structure_valid_rows": int(valid.sum()),
        "structure_unavailable_rows": int((~valid).sum()),
        "structure_valid_fraction": float(valid.mean()),
        "n_canonical_parents": frame.loc[valid, "canonical_parent"].nunique(),
        "n_nonempty_scaffolds": scaffolds.nunique(),
        "singleton_scaffold_fraction": (
            math.nan if scaffold_sizes.empty else float((scaffold_sizes == 1).mean())
        ),
        "median_chemicals_per_scaffold": (
            math.nan if scaffold_sizes.empty else float(scaffold_sizes.median())
        ),
    }
