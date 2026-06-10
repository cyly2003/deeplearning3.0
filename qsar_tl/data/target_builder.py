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

