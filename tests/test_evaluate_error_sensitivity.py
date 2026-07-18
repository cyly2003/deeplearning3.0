from __future__ import annotations

import pandas as pd

from scripts.evaluate_error_sensitivity import (
    attach_group_flags,
    build_scenarios,
    prepare_groups,
    prepare_rows,
    summarize_by_dimension,
    summarize_scenarios,
)


def test_sensitivity_excluding_label_or_source_conflict_reduces_mae() -> None:
    rows = pd.DataFrame(
        {
            "chemical_label": ["Chem A", "Chem A", "Chem B", "Chem B"],
            "species_label": ["Species 1", "Species 1", "Species 2", "Species 2"],
            "task_head": ["NOEC_Growth", "NOEC_Growth", "NOEC_Growth", "NOEC_Growth"],
            "task_family": ["NOEC", "NOEC", "NOEC", "NOEC"],
            "target_basis": ["mol/L", "mol/L", "mol/L", "mol/L"],
            "split_policy": ["random_5fold"] * 4,
            "ad_tier": ["AD-A"] * 4,
            "chemical_class_l1": ["organic"] * 4,
            "chemical_class_l2": ["unclassified"] * 4,
            "taxon_group_l1": ["plant"] * 4,
            "y_true": [1.0, 5.0, 2.0, 2.2],
            "y_pred": [4.0, 2.0, 2.1, 2.0],
            "is_high_error": [True, True, False, False],
        }
    )
    groups = pd.DataFrame(
        {
            "chemical_label": ["Chem A", "Chem B"],
            "species_label": ["Species 1", "Species 2"],
            "task_head": ["NOEC_Growth", "NOEC_Growth"],
            "target_basis": ["mol/L", "mol/L"],
            "group_label_conflict": [True, False],
            "source_conflict": [True, False],
            "repeated_high_error": [True, False],
            "in_domain_high_error": [True, False],
            "multi_reference": [True, False],
            "conflict_risk_score": [5, 0],
            "prediction_n": [2, 2],
            "mae": [3.0, 0.15],
        }
    )

    data = attach_group_flags(
        prepare_rows(rows),
        prepare_groups(groups, group_columns=("chemical_label", "species_label", "task_head", "target_basis")),
        group_columns=("chemical_label", "species_label", "task_head", "target_basis"),
    )
    overall = summarize_scenarios(data, build_scenarios(data))

    baseline = overall[overall["scenario"].eq("baseline_all")].iloc[0]
    filtered = overall[overall["scenario"].eq("exclude_label_or_source_conflict")].iloc[0]
    assert baseline["n"] == 4
    assert filtered["n"] == 2
    assert filtered["retained_fraction"] == 0.5
    assert filtered["mae"] < baseline["mae"]


def test_dimension_summary_uses_dimension_baseline_counts() -> None:
    rows = pd.DataFrame(
        {
            "chemical_label": ["Chem A", "Chem A", "Chem B", "Chem C"],
            "species_label": ["Species 1", "Species 1", "Species 2", "Species 3"],
            "task_head": ["NOEC_Growth", "NOEC_Growth", "ECx_Mortality", "ECx_Mortality"],
            "task_family": ["NOEC", "NOEC", "ECx", "ECx"],
            "target_basis": ["mol/L"] * 4,
            "split_policy": ["random_5fold"] * 4,
            "ad_tier": ["AD-A"] * 4,
            "chemical_class_l1": ["organic"] * 4,
            "chemical_class_l2": ["unclassified"] * 4,
            "taxon_group_l1": ["plant"] * 4,
            "y_true": [1.0, 5.0, 2.0, 2.2],
            "y_pred": [4.0, 2.0, 2.1, 2.0],
            "is_high_error": [True, True, False, False],
        }
    )
    groups = pd.DataFrame(
        {
            "chemical_label": ["Chem A", "Chem B", "Chem C"],
            "species_label": ["Species 1", "Species 2", "Species 3"],
            "task_head": ["NOEC_Growth", "ECx_Mortality", "ECx_Mortality"],
            "target_basis": ["mol/L"] * 3,
            "group_label_conflict": [True, False, False],
            "source_conflict": [False, False, False],
            "repeated_high_error": [True, False, False],
            "in_domain_high_error": [True, False, False],
            "multi_reference": [False, False, False],
            "conflict_risk_score": [3, 0, 0],
            "prediction_n": [2, 1, 1],
        }
    )
    data = attach_group_flags(
        prepare_rows(rows),
        prepare_groups(groups, group_columns=("chemical_label", "species_label", "task_head", "target_basis")),
        group_columns=("chemical_label", "species_label", "task_head", "target_basis"),
    )

    summary = summarize_by_dimension(data, build_scenarios(data), dimensions=["task_head"], min_n=1)
    ecx_filtered = summary[
        summary["scenario"].eq("exclude_label_or_source_conflict")
        & summary["task_head"].eq("ECx_Mortality")
    ].iloc[0]
    assert ecx_filtered["n"] == 2
    assert ecx_filtered["retained_fraction_of_dimension"] == 1.0
