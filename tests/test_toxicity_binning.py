from __future__ import annotations

from qsar_tl.training.toxicity_binning import (
    ToxicityBinningConfig,
    assign_toxicity_bin,
    load_toxicity_bin_scheme,
)


def _scheme() -> dict:
    return load_toxicity_bin_scheme("authority_v1")


def test_soil_mg_kg_hard_bins_and_boundary_flags() -> None:
    cfg = ToxicityBinningConfig(enabled=True)
    values = [
        (95.0, "soil_higher_toxicity_screening", True),
        (100.0, "soil_higher_toxicity_screening", True),
        (105.0, "soil_low_toxicity_screening", True),
        (1000.0, "soil_low_toxicity_screening", True),
        (1050.0, "soil_very_low_or_practically_nontoxic_screening", True),
        (1200.0, "soil_very_low_or_practically_nontoxic_screening", False),
    ]

    for value, expected_label, expected_boundary in values:
        assignment = assign_toxicity_bin(
            {"unit_family_v2": "soil_mg_kg", "standard_value_mg_kg": value},
            _scheme(),
            config=cfg,
        )
        assert assignment.status == "active"
        assert assignment.label == expected_label
        assert assignment.boundary_flag is expected_boundary


def test_water_mg_l_epa_categories_at_boundaries() -> None:
    cfg = ToxicityBinningConfig(enabled=True)
    values = [
        (0.1, "water_very_high_toxicity"),
        (1.0, "water_high_toxicity"),
        (10.0, "water_moderate_toxicity"),
        (100.0, "water_low_toxicity"),
        (100.1, "water_very_low_or_practically_nontoxic"),
    ]

    for value, expected_label in values:
        assignment = assign_toxicity_bin(
            {"unit_family_v2": "water_mg_l", "standard_value_mg_l": value},
            _scheme(),
            config=cfg,
        )
        assert assignment.status == "active"
        assert assignment.label == expected_label


def test_water_mol_l_converts_to_mg_l_using_molecular_weight() -> None:
    assignment = assign_toxicity_bin(
        {"unit_family_v2": "water_mol_l", "standard_value_mol_l": 0.0001},
        _scheme(),
        config=ToxicityBinningConfig(enabled=True),
        descriptor_mol_weight=100.0,
    )

    assert assignment.status == "active"
    assert assignment.value == 10.0
    assert assignment.value_unit == "mg/L"
    assert assignment.conversion == "mol_l_to_mg_l_using_molecular_weight"
    assert assignment.label == "water_moderate_toxicity"


def test_water_mol_l_without_molecular_weight_is_ineligible() -> None:
    assignment = assign_toxicity_bin(
        {"unit_family_v2": "water_mol_l", "standard_value_mol_l": 0.0001, "smiles": ""},
        _scheme(),
        config=ToxicityBinningConfig(enabled=True),
    )

    assert assignment.index == -1
    assert assignment.status == "missing_molecular_weight"


def test_unsupported_unit_family_gets_status_without_active_label() -> None:
    assignment = assign_toxicity_bin(
        {"unit_family_v2": "soil_g_ha", "standard_value_g_ha": 100.0},
        _scheme(),
        config=ToxicityBinningConfig(enabled=True),
    )

    assert assignment.index == -1
    assert assignment.status == "no_authoritative_threshold"
    assert assignment.label == ""
