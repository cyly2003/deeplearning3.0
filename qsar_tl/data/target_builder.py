from __future__ import annotations

from dataclasses import dataclass
from math import log10


@dataclass(frozen=True)
class ToxicityValueResult:
    value: float | None
    source: str
    imputed: bool
    excluded_reason: str | None = None


def choose_toxicity_value(
    mean_value: float | None,
    min_value: float | None,
    max_value: float | None,
    dose_group_count: int | None,
    min_dose_groups_for_midpoint: int = 3,
) -> ToxicityValueResult:
    if mean_value is not None:
        return ToxicityValueResult(value=float(mean_value), source="mean", imputed=False)

    if (
        min_value is not None
        and max_value is not None
        and dose_group_count is not None
        and dose_group_count >= min_dose_groups_for_midpoint
    ):
        return ToxicityValueResult(
            value=(float(min_value) + float(max_value)) / 2.0,
            source="min_max_midpoint",
            imputed=True,
        )

    return ToxicityValueResult(
        value=None,
        source="excluded",
        imputed=False,
        excluded_reason="missing_mean_and_invalid_min_max_midpoint",
    )


def neg_log10(value: float) -> float:
    if value <= 0:
        raise ValueError("Cannot apply -log10 to non-positive values.")
    return -log10(value)


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NR", "NC", "NA", "N/A", "NONE", "NULL"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


@dataclass(frozen=True)
class TargetBuildResult:
    target_value: float | None
    target_name: str | None
    target_basis: str | None
    target_status: str
    tox_value: float | None
    tox_value_source: str
    tox_value_imputed: bool
    excluded_reason: str | None


def build_target_from_standardized_value(
    *,
    mean_value: object,
    min_value: object,
    max_value: object,
    dose_group_count: object,
    unit_family: str | None,
    standard_unit: str | None,
    molecular_weight_g_mol: object,
    medium: str | None,
    min_dose_groups_for_midpoint: int = 3,
) -> TargetBuildResult:
    """Build the first-version modeling target from standardized ECOTOX fields.

    Rules are intentionally conservative:
    - mean is preferred.
    - if mean is missing, min/max midpoint is allowed only when dose groups >= 3.
    - water mg/L is converted to mol/L using molecular weight.
    - water mol/L is used directly.
    - soil/sediment mg/kg is modeled as -log10(mg/kg).
    - oral mg/kg/d is modeled as -log10(mg/kg/d) for the auxiliary branch.
    """

    selected = choose_toxicity_value(
        mean_value=parse_float(mean_value),
        min_value=parse_float(min_value),
        max_value=parse_float(max_value),
        dose_group_count=parse_int(dose_group_count),
        min_dose_groups_for_midpoint=min_dose_groups_for_midpoint,
    )
    if selected.value is None:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_basis=None,
            target_status="excluded",
            tox_value=None,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            excluded_reason=selected.excluded_reason,
        )

    if selected.value <= 0:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_basis=None,
            target_status="excluded",
            tox_value=selected.value,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            excluded_reason="non_positive_toxicity_value",
        )

    family = (unit_family or "").strip()
    unit = (standard_unit or "").strip()
    medium_norm = (medium or "").strip()

    try:
        if family == "water_mol_l" or unit == "mol/L":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="ptox_mol_l",
                target_basis="mol/L",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                excluded_reason=None,
            )

        if family == "water_mg_l" or unit == "mg/L":
            mw = parse_float(molecular_weight_g_mol)
            if mw is None or mw <= 0:
                return TargetBuildResult(
                    target_value=None,
                    target_name=None,
                    target_basis=None,
                    target_status="excluded",
                    tox_value=selected.value,
                    tox_value_source=selected.source,
                    tox_value_imputed=selected.imputed,
                    excluded_reason="missing_or_invalid_molecular_weight_for_mg_l_to_mol_l",
                )
            mol_l = selected.value / 1000.0 / mw
            return TargetBuildResult(
                target_value=neg_log10(mol_l),
                target_name="ptox_mol_l",
                target_basis="mol/L_from_mg/L",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                excluded_reason=None,
            )

        if family == "soil_mg_kg" or unit == "mg/kg":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_mg_kg",
                target_basis=f"mg/kg:{medium_norm or 'unknown_medium'}",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                excluded_reason=None,
            )

        if family == "oral_mg_kg_d" or unit == "mg/kg/d":
            return TargetBuildResult(
                target_value=neg_log10(selected.value),
                target_name="neg_log10_mg_kg_bw_day",
                target_basis="mg/kg/day",
                target_status="included",
                tox_value=selected.value,
                tox_value_source=selected.source,
                tox_value_imputed=selected.imputed,
                excluded_reason=None,
            )
    except ValueError as exc:
        return TargetBuildResult(
            target_value=None,
            target_name=None,
            target_basis=None,
            target_status="excluded",
            tox_value=selected.value,
            tox_value_source=selected.source,
            tox_value_imputed=selected.imputed,
            excluded_reason=str(exc),
        )

    return TargetBuildResult(
        target_value=None,
        target_name=None,
        target_basis=None,
        target_status="excluded",
        tox_value=selected.value,
        tox_value_source=selected.source,
        tox_value_imputed=selected.imputed,
        excluded_reason=f"unsupported_unit_family:{family or unit or 'missing'}",
    )
