from __future__ import annotations

from qsar_tl.data.target_builder import build_target_from_standardized_value, choose_toxicity_value


def test_min_max_midpoint_no_longer_requires_dose_group_count() -> None:
    result = choose_toxicity_value(
        mean_value=None,
        min_value=2.0,
        max_value=6.0,
        dose_group_count=None,
        min_dose_groups_for_midpoint=999,
    )

    assert result.value == 4.0
    assert result.source == "min_max_midpoint"
    assert result.imputed is True
    assert result.value_quality == "midpoint"


def test_target_builder_uses_min_max_midpoint_without_dose_group_count() -> None:
    result = build_target_from_standardized_value(
        mean_value=None,
        min_value=10.0,
        max_value=10.0,
        dose_group_count=None,
        unit_family="water_mg_l",
        standard_unit="mg/L",
        molecular_weight_g_mol=100.0,
        medium="FW",
        min_dose_groups_for_midpoint=999,
    )

    assert result.target_status == "included"
    assert result.target_name == "ptox_mol_l"
    assert result.target_family == "aquatic_pTox_mol_L"
    assert result.tox_value_source == "min_max_midpoint"
    assert result.value_quality == "midpoint"


def test_censored_mean_is_kept_for_audit_but_excluded_from_default_target() -> None:
    result = build_target_from_standardized_value(
        mean_value=10.0,
        min_value=None,
        max_value=None,
        dose_group_count=None,
        unit_family="water_mg_l",
        standard_unit="mg/L",
        molecular_weight_g_mol=100.0,
        medium="aquatic",
        mean_op="<",
    )

    assert result.target_status == "excluded"
    assert result.tox_value == 10.0
    assert result.value_quality == "censored"
    assert result.excluded_reason == "censored_toxicity_value"
