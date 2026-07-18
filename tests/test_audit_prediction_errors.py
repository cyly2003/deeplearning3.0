from __future__ import annotations

import math

import pandas as pd

from scripts.audit_prediction_errors import (
    build_quality_labels,
    group_summary,
    high_error_threshold,
    prepare_audit_rows,
)


def test_prepare_audit_rows_marks_high_error_and_labels_quality() -> None:
    frame = pd.DataFrame(
        {
            "split_part": ["test", "test", "test"],
            "task_head": ["NOEC_Growth", "NOEC_Growth", "ECx_Mortality"],
            "task_family": ["NOEC", "NOEC", "ECx"],
            "chemical_name": ["Chem A", "Chem A", "Chem B"],
            "cas_number": ["1-1-1", "1-1-1", "2-2-2"],
            "latin_name": ["Species a", "Species b", "Species b"],
            "y_true": [1.0, 5.0, 3.0],
            "y_pred": [1.1, 2.0, 2.5],
            "cst_ad_tier": ["AD-A", "AD-B", "AD-A"],
            "value_quality": ["exact", "censored", "exact"],
            "unit_conversion_confidence": ["high", "medium", "high"],
            "toxicity_bin_status": ["active", "active", "active"],
            "toxicity_bin_boundary_flag": [0, 1, 0],
            "chemical_class_l2": ["organic", "metal_metalloid", "organic"],
        }
    )

    audited = prepare_audit_rows(frame, high_error_quantile=0.8, high_error_absolute=1.0)

    assert audited["is_high_error"].tolist() == [False, True, False]
    assert audited.loc[1, "quality_label"] == "value=censored|unit=medium|bin=active|bin_boundary|chem=metal_metalloid"
    assert audited.loc[0, "chemical_label"] == "Chem A [1-1-1]"
    assert audited.loc[1, "error_direction"] == "underprediction"


def test_group_summary_computes_error_metrics() -> None:
    frame = pd.DataFrame(
        {
            "task_head": ["A", "A", "B"],
            "chemical_label": ["c1", "c2", "c2"],
            "species_label": ["s1", "s1", "s2"],
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 3.0, 2.0],
            "is_high_error": [False, True, True],
        }
    )

    summary = group_summary(frame, ["task_head"])
    rows = {row["task_head"]: row for row in summary.to_dict(orient="records")}

    assert rows["A"]["n"] == 2
    assert math.isclose(rows["A"]["mae"], 0.5)
    assert rows["A"]["high_error_n"] == 1
    assert rows["B"]["r2"] != rows["B"]["r2"]


def test_high_error_threshold_uses_stricter_absolute_or_quantile() -> None:
    values = pd.Series([0.1, 0.2, 1.2, 3.0])

    assert high_error_threshold(values, quantile=0.5, absolute=1.0) == 1.0
    assert high_error_threshold(values, quantile=0.9, absolute=1.0) > 1.0


def test_build_quality_labels_falls_back_to_prediction_fields() -> None:
    labels = build_quality_labels(
        pd.DataFrame(
            {
                "toxicity_bin_status": ["active"],
                "toxicity_bin_boundary_flag": ["0"],
                "conversion_path": ["raw_to_mol_L_to_pTox"],
            }
        )
    )

    assert labels.tolist() == ["bin=active|conversion=raw_to_mol_L_to_pTox"]
