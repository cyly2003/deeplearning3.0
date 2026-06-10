from __future__ import annotations

import re
from dataclasses import dataclass


EC_LC_ENDPOINT_RE = re.compile(r"^(EC|LC)(\d+(?:\.\d+)?)$")

EXCLUDED_ENDPOINTS = {
    "BAF",
    "BCF",
    "BCFD",
    "LOEL",
    "LT50",
    "MATC",
    "NOEL",
}

MORTALITY_CODES = {"MOR", "MORT", "SURV"}
GROWTH_CODES = {"GRO", "WGHT", "LGTH", "GGRO", "BMAS"}
REPRODUCTION_CODES = {"REP", "GERM", "PROG", "FCND", "GREP", "FERZ"}
POPULATION_CODES = {"POP", "ABND", "PGRT", "GPOP"}
IMMOBILIZATION_CODES = {"ITX", "IMBL"}


@dataclass(frozen=True)
class TaskMappingResult:
    task_head: str | None
    task_family: str | None
    effect_family: str | None
    effect_level_x: float | None
    task_status: str
    task_excluded_reason: str | None


def normalize_code(value: object) -> str:
    """Normalize compact ECOTOX endpoint/effect/measurement codes."""

    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    text = text.lstrip("~")
    while text.endswith(("/", "*")):
        text = text[:-1]
    return text


def parse_effect_level_x(endpoint: object) -> float | None:
    endpoint_norm = normalize_code(endpoint)
    match = EC_LC_ENDPOINT_RE.match(endpoint_norm)
    if not match:
        return None
    return float(match.group(2))


def classify_endpoint(endpoint: object) -> tuple[str | None, float | None, str | None]:
    endpoint_norm = normalize_code(endpoint)
    if not endpoint_norm:
        return None, None, "missing_endpoint"

    if endpoint_norm.startswith("NR"):
        return None, None, "excluded_endpoint:NR"

    if endpoint_norm in EXCLUDED_ENDPOINTS:
        return None, None, f"excluded_endpoint:{endpoint_norm}"

    if endpoint_norm in {"NOEC", "LOEC"}:
        return endpoint_norm, None, None

    effect_level = parse_effect_level_x(endpoint_norm)
    if effect_level is not None:
        return "ECx", effect_level, None

    return None, None, f"unsupported_endpoint:{endpoint_norm}"


def classify_effect(effect: object, measurement: object) -> tuple[str | None, str | None]:
    codes = {normalize_code(effect), normalize_code(measurement)}
    codes.discard("")

    if codes & MORTALITY_CODES:
        return "Mortality", None
    if codes & GROWTH_CODES:
        return "Growth", None
    if codes & REPRODUCTION_CODES:
        return "Reproduction", None
    if codes & POPULATION_CODES:
        return "Population", None
    if codes & IMMOBILIZATION_CODES:
        return "Immobilization", None

    if not codes:
        return None, "missing_effect_and_measurement"
    return None, "unsupported_effect_family"


def map_task_head(
    *,
    endpoint: object,
    effect: object,
    measurement: object,
    target_name: object = None,
    target_basis: object = None,
) -> TaskMappingResult:
    """Map an included target record to a first-batch ECOTOX-QSAR task head."""

    target_name_norm = normalize_code(target_name)
    target_basis_norm = normalize_code(target_basis)
    if target_name_norm == "NEG_LOG10_MG_KG_BW_DAY" or target_basis_norm == "MG/KG/DAY":
        return TaskMappingResult(
            task_head=None,
            task_family=None,
            effect_family=None,
            effect_level_x=None,
            task_status="excluded",
            task_excluded_reason="excluded_oral_target",
        )

    task_family, effect_level, endpoint_reason = classify_endpoint(endpoint)
    if endpoint_reason is not None:
        return TaskMappingResult(
            task_head=None,
            task_family=None,
            effect_family=None,
            effect_level_x=None,
            task_status="excluded",
            task_excluded_reason=endpoint_reason,
        )

    effect_family, effect_reason = classify_effect(effect, measurement)
    if effect_reason is not None:
        return TaskMappingResult(
            task_head=None,
            task_family=task_family,
            effect_family=None,
            effect_level_x=effect_level,
            task_status="excluded",
            task_excluded_reason=effect_reason,
        )

    return TaskMappingResult(
        task_head=f"{task_family}_{effect_family}",
        task_family=task_family,
        effect_family=effect_family,
        effect_level_x=effect_level,
        task_status="included",
        task_excluded_reason=None,
    )
