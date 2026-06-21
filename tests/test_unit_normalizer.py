from __future__ import annotations

from qsar_tl.data.target_builder import build_target_from_standardized_value
from qsar_tl.data.unit_normalizer import normalize_concentration_values


def normalize_raw(value: float, unit: str, *, organism_habitat: str | None = None, media_type: str | None = None):
    return normalize_concentration_values(
        standardized_mean=None,
        standardized_min=None,
        standardized_max=None,
        unit_family="other",
        standard_unit=None,
        raw_mean=value,
        raw_min=None,
        raw_max=None,
        raw_unit=unit,
        molecular_weight_g_mol=100.0,
        organism_habitat=organism_habitat,
        media_type=media_type,
    )


def test_soil_mg_kg_raw_units_are_standardized() -> None:
    result = normalize_raw(2.5, "mg/kg dry soil")

    assert result.mean_value_v2 == 2.5
    assert result.unit_family_v2 == "soil_mg_kg"
    assert result.standard_unit_v2 == "mg/kg"
    assert result.unit_conversion_confidence == "high"


def test_soil_ug_g_maps_to_mg_kg() -> None:
    result = normalize_raw(3.0, "ug/g soil")

    assert result.mean_value_v2 == 3.0
    assert result.unit_family_v2 == "soil_mg_kg"


def test_area_application_units_are_kept_as_separate_target_scale() -> None:
    result = normalize_raw(1.0, "AI kg/ha")

    assert result.mean_value_v2 == 1000.0
    assert result.unit_family_v2 == "soil_g_ha"
    assert result.standard_unit_v2 == "g/ha"
    assert result.active_ingredient_basis is True


def test_water_raw_units_are_standardized_for_ptox_path() -> None:
    result = normalize_raw(5.0, "ug/ml")

    assert result.mean_value_v2 == 5.0
    assert result.unit_family_v2 == "water_mg_l"
    assert result.standard_unit_v2 == "mg/L"
    assert result.standard_value_mg_l == 5.0
    assert result.standard_value_mol_l == 5.0 / 1000.0 / 100.0
    assert result.conversion_path == "raw_to_mg_L_to_mol_L_to_pTox"


def test_ppm_is_aqueous_mg_l_only_with_aquatic_context() -> None:
    result = normalize_raw(2.0, "ppm", organism_habitat="Water", media_type="FW")

    assert result.unit_family_v2 == "water_mg_l"
    assert result.standard_unit_v2 == "mg/L"
    assert result.mean_value_v2 == 2.0
    assert result.unit_conversion_note == "ppm_assumed_mg_per_l_for_aqueous_matrix"


def test_ppm_is_solid_mg_kg_with_soil_context() -> None:
    result = normalize_raw(2.0, "ppm", organism_habitat="Soil", media_type="NAT")

    assert result.unit_family_v2 == "soil_mg_kg"
    assert result.standard_unit_v2 == "mg/kg"
    assert result.standard_value_mg_kg == 2.0


def test_seed_treatment_units_are_kept_separate() -> None:
    result = normalize_raw(250.0, "AI g/100 kg sd")

    assert result.mean_value_v2 == 2.5
    assert result.unit_family_v2 == "seed_g_kg"
    assert result.standard_unit_v2 == "g/kg seed"


def test_target_builder_includes_g_ha_as_distinct_target() -> None:
    result = build_target_from_standardized_value(
        mean_value=1000.0,
        min_value=None,
        max_value=None,
        dose_group_count=None,
        unit_family="soil_g_ha",
        standard_unit="g/ha",
        molecular_weight_g_mol=100.0,
        medium="NAT",
    )

    assert result.target_status == "included"
    assert result.target_name == "neg_log10_g_ha"
    assert result.target_basis == "g/ha"
