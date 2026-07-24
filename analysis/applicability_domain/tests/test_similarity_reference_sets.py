from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_target_exact_parent_flag_uses_stage3_training_structures_only() -> None:
    manifest = pd.read_parquet(ROOT / "stage3_manifest.parquet")
    cache = pd.read_parquet(ROOT / "chemical_structure_cache.parquet")
    mapping = (
        cache.loc[cache["stage"].eq("stage3_train"), ["raw_smiles", "canonical_parent"]]
        .drop_duplicates("raw_smiles")
        .set_index("raw_smiles")["canonical_parent"]
    )
    train_parent = set(
        manifest.loc[manifest["analysis_split"].eq("train"), "smiles"].astype("string").fillna("").map(mapping).dropna()
    )
    train_parent.discard("")
    records = pd.read_parquet(ROOT / "ad_record_level.parquet")
    available = records.loc[records["structure_status"].eq("ok")]
    expected = available["canonical_parent"].isin(train_parent).to_numpy(bool)
    observed = available["exact_parent_seen_target"].to_numpy(bool)
    assert np.array_equal(expected, observed)


def test_structure_unavailable_is_not_silently_encoded_as_zero_similarity() -> None:
    records = pd.read_parquet(ROOT / "ad_record_level.parquet")
    unavailable = records.loc[records["structure_status"].ne("ok")]
    assert len(unavailable) > 0
    assert unavailable["C_target"].isna().all()
    assert unavailable["C_source"].isna().all()
    assert unavailable["ad_tier_reason"].str.contains("structure_unavailable", regex=False).all()
