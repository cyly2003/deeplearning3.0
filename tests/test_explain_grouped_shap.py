from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.explain_grouped_shap import (
    build_feature_catalog,
    descriptor_percentile_pdp,
    fingerprint_bit_contrast,
    reproducible_sample_indices,
    summarize_shap_values,
)


def _arrays() -> dict[str, object]:
    return {
        "numeric": np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        "fingerprint": np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "categorical": {"species": np.asarray([1, 2, 1], dtype=np.int64)},
        "adapter_id": np.asarray([1, 1, 1], dtype=np.int64),
        "target": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "target_scaler": None,
        "target_scale_key": "ECx_Growth|solid_neglog_mg_kg",
        "numeric_names": ["MolWt", "TPSA", "duration_log1p_h"],
        "categorical_names": ["species"],
    }


def _predictor(_model: object, arrays: dict[str, object], _task_head: str) -> np.ndarray:
    numeric = np.asarray(arrays["numeric"], dtype=float)
    fingerprint = np.asarray(arrays["fingerprint"], dtype=float)
    return numeric[:, 0] + 2.0 * numeric[:, 1] + 3.0 * fingerprint[:, 0] - fingerprint[:, 1]


def test_feature_catalog_assigns_descriptor_context_and_fingerprint_groups() -> None:
    catalog = build_feature_catalog(
        _arrays(),
        preprocessing={"molecular_descriptor_names": ["MolWt", "TPSA"]},
    )

    assert catalog["feature_group"].tolist() == [
        "molecular_descriptor",
        "molecular_descriptor",
        "species_context",
        "morgan_fingerprint",
        "morgan_fingerprint",
        "species_context",
    ]
    assert catalog.loc[3, "feature"] == "morgan_bit_0"


def test_sampling_is_reproducible() -> None:
    first = reproducible_sample_indices(10, background_rows=4, explain_rows=3, seed=91)
    second = reproducible_sample_indices(10, background_rows=4, explain_rows=3, seed=91)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_summary_and_pdp_outputs_have_expected_fields(monkeypatch) -> None:
    arrays = _arrays()
    catalog = build_feature_catalog(arrays, preprocessing={"molecular_descriptor_names": ["MolWt", "TPSA"]})
    shap_values = np.asarray(
        [[0.5, 0.2, 0.1, 0.7, 0.05, 0.1], [0.4, 0.3, 0.2, 0.5, 0.1, 0.2]], dtype=float
    )
    ranking = summarize_shap_values(shap_values, catalog)
    descriptor_ranking = ranking[ranking["feature_group"] == "molecular_descriptor"]
    fingerprint_ranking = ranking[ranking["feature_group"] == "morgan_fingerprint"]
    monkeypatch.setattr("scripts.explain_grouped_shap.predict", _predictor)

    pdp = descriptor_percentile_pdp(
        object(),
        arrays,
        task_head="ECx_Growth",
        descriptor_ranking=descriptor_ranking,
        preprocessing={"numeric_stats": {"MolWt": {"mean": 10.0, "std": 2.0}, "TPSA": {"mean": 3.0, "std": 1.0}}},
        top_k=1,
        grid_size=3,
    )
    contrast = fingerprint_bit_contrast(
        object(),
        arrays,
        task_head="ECx_Growth",
        fingerprint_ranking=fingerprint_ranking,
        top_k=1,
    )

    assert ranking.iloc[0]["feature"] == "morgan_bit_0"
    assert len(pdp) == 3
    assert {"percentile", "raw_value_approx", "mean_prediction"}.issubset(pdp.columns)
    assert len(contrast) == 1
    assert contrast.iloc[0]["mean_0_to_1_contrast"] == 3.0
