from __future__ import annotations

import pandas as pd

from scripts.audit_label_conflicts import prepare_rows, summarize_conflict_groups, unique_token_count


def test_summarize_conflict_groups_flags_range_and_repeated_high_error() -> None:
    frame = pd.DataFrame(
        {
            "chemical_label": ["Chem A", "Chem A", "Chem A", "Chem B"],
            "species_label": ["Species 1", "Species 1", "Species 1", "Species 2"],
            "task_head": ["NOEC_Growth", "NOEC_Growth", "NOEC_Growth", "ECx_Mortality"],
            "target_basis": ["mol/L", "mol/L", "mol/L", "mol/L"],
            "aggregate_id": ["1", "2", "3", "4"],
            "split_policy": ["random_5fold", "random_5fold", "random_8_2", "random_5fold"],
            "fold": ["fold1", "fold2", "holdout", "fold1"],
            "ad_tier": ["AD-A", "AD-A", "AD-A", "AD-B"],
            "y_true": [1.0, 4.0, 5.0, 2.0],
            "y_pred": [1.1, 1.0, 2.0, 2.1],
            "abs_error": [0.1, 3.0, 3.0, 0.1],
            "residual": [-0.1, 3.0, 3.0, -0.1],
            "is_high_error": [False, True, True, False],
            "target_value_min": [1.0, 3.0, 5.0, 2.0],
            "target_value_max": [1.0, 4.5, 5.0, 2.0],
            "target_value_count": [1, 2, 1, 1],
            "cross_reference_range": [0.0, 1.5, 0.0, 0.0],
            "cross_reference_conflict_flag": [0, 1, 0, 0],
            "reference_numbers": ['["R1"]', '["R2","R3"]', '["R4"]', '["R5"]'],
        }
    )
    rows = prepare_rows(frame)

    summary = summarize_conflict_groups(
        rows,
        group_columns=("chemical_label", "species_label", "task_head", "target_basis"),
        group_range_threshold=2.0,
        source_range_threshold=1.0,
        min_high_error_n=2,
        min_high_error_fraction=0.5,
    )

    first = summary.iloc[0]
    assert first["chemical_label"] == "Chem A"
    assert first["group_label_conflict"]
    assert first["source_conflict"]
    assert first["repeated_high_error"]
    assert first["in_domain_high_error"]
    assert first["multi_reference"]
    assert first["conflict_risk_score"] == 5


def test_unique_token_count_handles_json_and_delimiters() -> None:
    values = pd.Series(['["R1", "R2"]', "R2;R3", "", None])

    assert unique_token_count(values) == 3
