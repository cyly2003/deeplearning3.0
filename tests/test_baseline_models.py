from __future__ import annotations

import pandas as pd

from qsar_tl.training.baseline import (
    add_duration_nonlinear_features,
    normalize_model_name,
    select_feature_columns,
    task_skip_reason,
    validate_single_target_dimension,
)


def test_baseline_model_aliases_cover_requested_models() -> None:
    assert normalize_model_name("RF") == "random_forest"
    assert normalize_model_name("XGB") == "xgboost"
    assert normalize_model_name("LGBM") == "lightgbm"
    assert normalize_model_name("PLS") == "pls"
    assert normalize_model_name("Extratress") == "extra_trees"
    assert normalize_model_name("ElasticNet") == "elastic_net"
    assert normalize_model_name("MLPRegressor") == "mlp"


def test_duration_nonlinear_features_are_added() -> None:
    frame = pd.DataFrame({"duration_bin_h": [24.0, 96.0, None]})

    transformed = add_duration_nonlinear_features(frame)

    assert "duration_log1p_h" in transformed.columns
    assert "duration_sqrt_h" in transformed.columns
    assert "duration_inv_log1p_h" in transformed.columns
    assert "duration_rbf_96h" in transformed.columns


def test_task_skip_reason_enforces_minimum_counts() -> None:
    frame = pd.DataFrame({"split_part": ["train", "train", "test"]})

    assert task_skip_reason(frame, min_total=4, min_train=1, min_eval=1) == "total_samples_lt_4"
    assert task_skip_reason(frame, min_total=0, min_train=3, min_eval=1) == "train_samples_lt_3"
    assert task_skip_reason(frame, min_total=0, min_train=1, min_eval=2) == "eval_samples_lt_2"
    assert task_skip_reason(frame, min_total=3, min_train=2, min_eval=1) is None


def test_unit_v2_metadata_are_excluded_from_baseline_features() -> None:
    frame = pd.DataFrame(
        {
            "target_value_median": [1.0],
            "target_value_weighted_mean": [1.0],
            "target_value_unweighted_median": [1.0],
            "target_value_weighted_std": [0.1],
            "task_head": ["ECx_Mortality"],
            "conc1_mean": [10.0],
            "conc1_unit": ["mg/L"],
            "unit_family_v2": ["water_mg_l"],
            "standard_unit_v2": ["mg/L"],
            "standard_value_mg_l": [10.0],
            "standard_value_mol_l": [0.0001],
            "unit_conversion_source": ["raw_unit_rule"],
            "unit_conversion_confidence": ["high"],
            "unit_conversion_note": ["mg/L_to_mg/L"],
            "conversion_path": ["raw_to_mg_L_to_mol_L_to_pTox"],
            "active_ingredient_basis": [1],
            "acid_equivalent_basis": [0],
            "target_family": ["aquatic_pTox_mol_L"],
            "value_quality": ["exact"],
            "chemical_class_l1": ["organic"],
        }
    )

    features = select_feature_columns(frame, target_column="target_value_median", task_column="task_head")

    assert "chemical_class_l1" in features
    assert "conc1_mean" not in features
    assert "conc1_unit" not in features
    assert "unit_family_v2" not in features
    assert "standard_unit_v2" not in features
    assert "unit_conversion_source" not in features
    assert "target_family" not in features
    assert "standard_value_mg_l" not in features
    assert "target_value_weighted_mean" not in features
    assert "target_value_unweighted_median" not in features
    assert "target_value_weighted_std" not in features


def test_validate_single_target_dimension_allows_one_target_scale() -> None:
    frame = pd.DataFrame({"target_name": ["ptox_mol_l", "ptox_mol_l", None]})

    validate_single_target_dimension(frame, split_name="B_random_8_2", source_table="aggregated_task_records_aquatic_ptox")


def test_validate_single_target_dimension_rejects_mixed_target_scales() -> None:
    frame = pd.DataFrame({"target_name": ["ptox_mol_l", "neg_log10_mg_kg", "ptox_mol_l"]})

    try:
        validate_single_target_dimension(frame, split_name="B_random_8_2", source_table="aggregated_task_records_aquatic")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected mixed target dimensions to be rejected")

    assert "Mixed target dimensions" in message
    assert "aggregated_task_records_aquatic" in message
    assert "*_ptox" in message


def test_validate_single_target_dimension_prefers_target_family_when_available() -> None:
    frame = pd.DataFrame(
        {
            "target_family": ["aquatic_pTox_mol_L", "solid_neglog_mg_kg"],
            "target_name": ["ptox_mol_l", "ptox_mol_l"],
        }
    )

    try:
        validate_single_target_dimension(frame, split_name="mixed", source_table="records")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected mixed target families to be rejected")

    assert "target_family" in message
