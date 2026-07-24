from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "train": "cf14390672af38a494fc5fdd6540a5d56ff9f8924c3c4479240333f975ae23b6",
    "validation": "edacfcec1367f932d191334f0b6b2fa3a2c87a1899adf980eac3bdad509a92b3",
    "test": "e9559640dc209d8e44a38101e09989905ae5b10e6f33e1c64fc9d14d124bc9e0",
}


def record_hash(values: pd.Series) -> str:
    payload = json.dumps(sorted(values.astype(str)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_stage3_strict_record_identity_matches_locked_boundary() -> None:
    manifest = pd.read_parquet(ROOT / "stage3_manifest.parquet")
    assert len(manifest) == 15199
    assert manifest["record_id"].is_unique
    assert manifest.groupby("analysis_split").size().to_dict() == {
        "test": 3042,
        "train": 9724,
        "validation": 2433,
    }
    for split, expected in EXPECTED_HASHES.items():
        assert record_hash(manifest.loc[manifest["analysis_split"].eq(split), "record_id"]) == expected


def test_prediction_and_support_rows_align_one_to_one() -> None:
    manifest = pd.read_parquet(ROOT / "stage3_manifest.parquet")
    records = pd.read_parquet(ROOT / "ad_record_level.parquet")
    expected = manifest.loc[manifest["analysis_split"].isin(["validation", "test"]), "record_id"]
    assert records["record_id"].is_unique
    assert set(records["record_id"]) == set(expected)
    assert records[["M00_prediction", "M10_prediction", "y_true"]].notna().all().all()


def test_discovery_uses_locked_boundary_hash() -> None:
    audit = json.loads((ROOT / "ad_discovery_audit.json").read_text(encoding="utf-8"))
    assert audit["locked_boundary_sha256"] == "a2febaa7ac4679c6f273a50210315a69118490c4e38b887e4f72958bfbc2ff24"
