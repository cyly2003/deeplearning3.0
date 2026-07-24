from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ad_common import ANALYSIS_DIR, canonical_sha256, regression_metrics, within_task_r2, write_csv
from ad_support import TAX_LABEL_BY_RANK


DEFAULT_CONFIG = ANALYSIS_DIR / "config" / "ad_config.yaml"
TAX_RANK = {value: key for key, value in TAX_LABEL_BY_RANK.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate and lock AD rules using validation only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = pd.read_parquet(args.output_dir / "ad_record_level_unlocked.parquet")
    validation = records.loc[records["analysis_split"].eq("validation")].copy()
    if set(validation["analysis_split"]) != {"validation"}:
        raise ValueError("Calibration input must contain validation rows only")
    candidates = evaluate_candidates(validation, config)
    candidates["pareto"] = pareto_mask(candidates)
    selected = select_rule(candidates)
    rule = build_locked_rule(selected, validation, config)
    tier = apply_rule(validation, rule)
    validation["ad_tier"] = tier
    validation["ad_tier_reason"] = tier_reason(validation, rule, tier)
    summary = summarize_tiers(validation, prediction="M10_prediction")
    curves = coverage_error_curves(validation, rule, split="validation")

    write_csv(args.output_dir / "ad_candidate_rules.csv", candidates)
    write_csv(args.output_dir / "ad_validation_pareto.csv", candidates.loc[candidates["pareto"]])
    write_csv(args.output_dir / "ad_summary_by_tier_validation.csv", summary)
    write_csv(args.output_dir / "ad_coverage_error_curves_validation.csv", curves)
    write_locked_json(args.output_dir / "ad_rule_locked.json", rule)
    (args.output_dir / "ad_rule_selection_note.md").write_text(
        render_selection_note(selected, rule, summary, candidates), encoding="utf-8"
    )
    print(args.output_dir / "ad_rule_locked.json")


def evaluate_candidates(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    calibration = config["calibration"]
    rows: list[dict[str, Any]] = []
    candidate_values = product(
        (0.50, 0.65),
        (0.70, 0.80),
        ("family", "genus"),
        (0.50, 0.60),
        (0.70, 0.80),
        (1, 3),
        (3, 5, 10),
        (0.40, 0.60),
    )
    overall_mae = float(validation["AE_M10"].mean())
    for index, values in enumerate(candidate_values, start=1):
        c_mod, c_high, tax_high, b_mod, b_high, n_mod, n_high, missing_high = values
        if c_high < c_mod or b_high < b_mod or n_high < n_mod:
            continue
        params = {
            "c_mod": c_mod,
            "c_high": c_high,
            "tax_mod": "family",
            "tax_high": tax_high,
            "b_mod": b_mod,
            "b_high": b_high,
            "n_mod": n_mod,
            "n_high": n_high,
            "missing_mod": 0.60,
            "missing_high": missing_high,
        }
        rule = {"thresholds": params}
        tier = apply_rule(validation, rule)
        stats = tier_candidate_stats(validation, tier)
        high_coverage = stats["high_n"] / len(validation)
        hm_coverage = (stats["high_n"] + stats["moderate_n"]) / len(validation)
        high_target = calibration["high_coverage_target"]
        hm_target = calibration["high_plus_moderate_coverage_target"]
        coverage_ok = high_target[0] <= high_coverage <= high_target[1] and hm_target[0] <= hm_coverage <= hm_target[1]
        minimum_rows = int(calibration["minimum_tier_rows"])
        tier_size_ok = min(stats["high_n"], stats["moderate_n"], stats["low_n"]) >= minimum_rows
        monotonic_mae = stats["high_mae"] <= stats["moderate_mae"] <= stats["low_mae"]
        monotonic_median = stats["high_median_ae"] <= stats["moderate_median_ae"] <= stats["low_median_ae"]
        monotonic_p90 = stats["high_p90_ae"] <= stats["moderate_p90_ae"] <= stats["low_p90_ae"]
        validation_direction = (
            stats["high_mae"] < stats["low_mae"]
            and stats["high_median_ae"] < stats["low_median_ae"]
            and stats["high_p90_ae"] < stats["low_p90_ae"]
        )
        max_task_share_ok = stats["high_max_task_share"] <= 0.35
        feasible = coverage_ok and tier_size_ok and max_task_share_ok
        direction_gain = (
            (stats["low_mae"] - stats["high_mae"])
            + (stats["low_median_ae"] - stats["high_median_ae"])
            + (stats["low_p90_ae"] - stats["high_p90_ae"])
        ) / max(overall_mae, 1e-12)
        coverage_score = -abs(high_coverage - np.mean(high_target)) - abs(hm_coverage - np.mean(hm_target))
        stability_score = stats["task_high_better_fraction"] * min(stats["eligible_task_count"] / 10.0, 1.0)
        simplicity_penalty = 0.01 * (n_high / 5.0) + 0.01 * (tax_high == "genus")
        score = 2.0 * direction_gain + stability_score + coverage_score - simplicity_penalty
        rows.append(
            {
                "candidate_id": f"AD-CAND-{index:04d}",
                **params,
                **stats,
                "high_coverage": high_coverage,
                "high_plus_moderate_coverage": hm_coverage,
                "coverage_ok": coverage_ok,
                "tier_size_ok": tier_size_ok,
                "max_task_share_ok": max_task_share_ok,
                "monotonic_mae": monotonic_mae,
                "monotonic_median_ae": monotonic_median,
                "monotonic_p90_ae": monotonic_p90,
                "validation_direction": validation_direction,
                "feasible": feasible,
                "selection_score": score,
            }
        )
    return pd.DataFrame(rows)


def apply_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    t = rule["thresholds"]
    high_local = local_column(t["c_high"], t["tax_high"], t["b_high"])
    mod_local = local_column(t["c_mod"], t["tax_mod"], t["b_mod"])
    common = frame["structure_status"].eq("ok")
    moderate = (
        common
        & frame["C_target"].ge(t["c_mod"])
        & frame["tax_support_same_task_rank"].ge(TAX_RANK[t["tax_mod"]])
        & frame["B_exp"].ge(t["b_mod"])
        & frame[mod_local].ge(t["n_mod"])
        & frame["context_missing_fraction"].le(t["missing_mod"])
    )
    high = (
        moderate
        & frame["C_target"].ge(t["c_high"])
        & frame["tax_support_same_task_rank"].ge(TAX_RANK[t["tax_high"]])
        & frame["B_exp"].ge(t["b_high"])
        & frame[high_local].ge(t["n_high"])
        & frame["context_missing_fraction"].le(t["missing_high"])
        & frame["T_tier"].ne("T0")
    )
    return pd.Series(np.where(high, "High", np.where(moderate, "Moderate", "Low/outside")), index=frame.index, dtype="string")


def tier_candidate_stats(frame: pd.DataFrame, tier: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, prefix in (("High", "high"), ("Moderate", "moderate"), ("Low/outside", "low")):
        selected = frame.loc[tier.eq(label)]
        output[f"{prefix}_n"] = len(selected)
        output[f"{prefix}_mae"] = float(selected["AE_M10"].mean()) if len(selected) else math.nan
        output[f"{prefix}_median_ae"] = float(selected["AE_M10"].median()) if len(selected) else math.nan
        output[f"{prefix}_p90_ae"] = float(selected["AE_M10"].quantile(0.90)) if len(selected) else math.nan
    high = frame.loc[tier.eq("High")]
    output["high_tasks"] = int(high["model_head"].nunique())
    output["high_max_task_share"] = (
        math.nan if high.empty else float(high["model_head"].value_counts(normalize=True).max())
    )
    minimum = 15
    comparisons = []
    for _, group in frame.assign(_tier=tier).groupby("model_head", sort=False):
        high_group = group.loc[group["_tier"].eq("High"), "AE_M10"]
        low_group = group.loc[group["_tier"].eq("Low/outside"), "AE_M10"]
        if len(high_group) >= minimum and len(low_group) >= minimum:
            comparisons.append(float(high_group.mean()) < float(low_group.mean()))
    output["eligible_task_count"] = len(comparisons)
    output["task_high_better_count"] = int(sum(comparisons))
    output["task_high_better_fraction"] = (
        math.nan if not comparisons else float(np.mean(comparisons))
    )
    return output


def pareto_mask(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates.loc[candidates["feasible"]].copy()
    mask = pd.Series(False, index=candidates.index)
    if eligible.empty:
        eligible = candidates.copy()
    objectives = [
        ("high_mae", "min"),
        ("high_p90_ae", "min"),
        ("high_coverage", "max"),
        ("task_high_better_fraction", "max"),
    ]
    for index, row in eligible.iterrows():
        dominated = False
        for other_index, other in eligible.iterrows():
            if other_index == index:
                continue
            no_worse = True
            strictly_better = False
            for column, direction in objectives:
                left = float(other[column]) if pd.notna(other[column]) else (-math.inf if direction == "max" else math.inf)
                right = float(row[column]) if pd.notna(row[column]) else (-math.inf if direction == "max" else math.inf)
                if direction == "min":
                    no_worse &= left <= right
                    strictly_better |= left < right
                else:
                    no_worse &= left >= right
                    strictly_better |= left > right
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            mask.loc[index] = True
    return mask


def select_rule(candidates: pd.DataFrame) -> pd.Series:
    pool = candidates.loc[candidates["pareto"] & candidates["feasible"] & candidates["validation_direction"]]
    if pool.empty:
        pool = candidates.loc[candidates["pareto"] & candidates["feasible"]]
    if pool.empty:
        pool = candidates.loc[candidates["pareto"]]
    if pool.empty:
        pool = candidates
    return pool.sort_values(
        ["selection_score", "high_coverage", "candidate_id"], ascending=[False, False, True]
    ).iloc[0]


def build_locked_rule(
    selected: pd.Series, validation: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    thresholds = {
        key: to_builtin(selected[key])
        for key in (
            "c_mod",
            "c_high",
            "tax_mod",
            "tax_high",
            "b_mod",
            "b_high",
            "n_mod",
            "n_high",
            "missing_mod",
            "missing_high",
        )
    }
    thresholds["n_mod"] = int(thresholds["n_mod"])
    thresholds["n_high"] = int(thresholds["n_high"])
    local_high = local_column(thresholds["c_high"], thresholds["tax_high"], thresholds["b_high"])
    n95 = float(validation[local_high].quantile(0.95))
    l_scale_denominator = math.log10(n95 + 1.0) if n95 > 0 else 1.0
    validation_pass = bool(
        selected["feasible"]
        and selected["validation_direction"]
        and selected["high_mae"] <= selected["moderate_mae"] <= selected["low_mae"]
    )
    return {
        "schema_version": 1,
        "status": "locked_validation_supported" if validation_pass else "locked_descriptive_only",
        "candidate_id": selected["candidate_id"],
        "selection_split": "validation",
        "selection_uses_test": False,
        "thresholds": thresholds,
        "formal_taxonomy_field": "tax_support_same_task_rank",
        "task_support_high_rule": "T_tier != T0",
        "structure_unavailable_policy": "Low/outside with C_target NaN",
        "local_high_column": local_high,
        "local_moderate_column": local_column(
            thresholds["c_mod"], thresholds["tax_mod"], thresholds["b_mod"]
        ),
        "support_floor_l_scale_denominator": l_scale_denominator,
        "support_floor_l_scale_source": "validation 95th percentile of locked high local count",
        "validation_summary": {
            key: to_builtin(selected[key])
            for key in selected.index
            if key
            in {
                "high_n",
                "moderate_n",
                "low_n",
                "high_coverage",
                "high_plus_moderate_coverage",
                "high_mae",
                "moderate_mae",
                "low_mae",
                "high_p90_ae",
                "moderate_p90_ae",
                "low_p90_ae",
                "eligible_task_count",
                "task_high_better_count",
                "task_high_better_fraction",
                "feasible",
                "validation_direction",
            }
        },
        "configuration_sha256": canonical_sha256(config),
        "validation_record_id_sha256": canonical_sha256(sorted(validation["record_id"].astype(str))),
        "immutable_after_creation": True,
    }


def summarize_tiers(frame: pd.DataFrame, *, prediction: str) -> pd.DataFrame:
    rows = []
    for tier in ("High", "Moderate", "Low/outside"):
        selected = frame.loc[frame["ad_tier"].eq(tier)]
        metric = regression_metrics(selected, prediction=prediction)
        rows.append(
            {
                "split": selected["analysis_split"].iloc[0] if len(selected) else "validation",
                "tier": tier,
                "n": len(selected),
                "coverage": len(selected) / len(frame),
                "unique_chemicals": selected["chemical_entity_id"].nunique(),
                "unique_species": normalize_species(selected).nunique(),
                "tasks": selected["model_head"].nunique(),
                **metric,
                "within_task_r2": within_task_r2(selected, prediction=prediction) if len(selected) else math.nan,
                "median_ae": selected["AE_M10"].median(),
                "p90_ae": selected["AE_M10"].quantile(0.90),
                "median_nae": selected["NAE_M10"].median(),
                "median_prediction_sd": selected["M10_prediction_sd"].median(),
            }
        )
    return pd.DataFrame(rows)


def coverage_error_curves(frame: pd.DataFrame, rule: dict[str, Any], *, split: str) -> pd.DataFrame:
    t = rule["thresholds"]
    local = local_column(t["c_high"], t["tax_high"], t["b_high"])
    l_scaled = np.clip(
        np.log10(frame[local].to_numpy(float) + 1.0)
        / float(rule["support_floor_l_scale_denominator"]),
        0,
        1,
    )
    scores = {
        "chemical_only": frame["C_target"].fillna(-1.0).to_numpy(float),
        "bio_context_only": np.minimum(
            frame["tax_support_same_task_display"].to_numpy(float),
            frame["B_exp"].to_numpy(float),
        ),
        "joint_CBTL": np.min(
            np.stack(
                [
                    frame["C_target"].fillna(-1.0).to_numpy(float),
                    frame["tax_support_same_task_display"].to_numpy(float),
                    frame["B_exp"].to_numpy(float),
                    frame["T_index"].to_numpy(float),
                    l_scaled,
                ],
                axis=1,
            ),
            axis=1,
        ),
        "seed_disagreement_only": -frame["M10_prediction_sd"].to_numpy(float),
    }
    rows = []
    for method, score in scores.items():
        order = np.argsort(-score, kind="mergesort")
        for coverage in np.linspace(0.10, 1.0, 19):
            n = max(1, int(round(len(frame) * coverage)))
            selected = frame.iloc[order[:n]]
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "retained_coverage": n / len(frame),
                    "n": n,
                    "mae": float(selected["AE_M10"].mean()),
                    "median_ae": float(selected["AE_M10"].median()),
                    "p90_ae": float(selected["AE_M10"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def tier_reason(frame: pd.DataFrame, rule: dict[str, Any], tier: pd.Series) -> pd.Series:
    t = rule["thresholds"]
    reasons = []
    for index, row in frame.iterrows():
        if tier.loc[index] == "High":
            reasons.append("all_high_conditions_met")
            continue
        if tier.loc[index] == "Moderate":
            reasons.append("moderate_conditions_met_but_one_or_more_high_conditions_failed")
            continue
        failed = []
        if row["structure_status"] != "ok":
            failed.append("structure_unavailable")
        elif not row["C_target"] >= t["c_mod"]:
            failed.append("low_C_target")
        if row["tax_support_same_task_rank"] < TAX_RANK[t["tax_mod"]]:
            failed.append("taxonomy_below_moderate")
        if row["B_exp"] < t["b_mod"]:
            failed.append("B_exp_below_moderate")
        if row[rule["local_moderate_column"]] < t["n_mod"]:
            failed.append("local_density_below_moderate")
        if row["context_missing_fraction"] > t["missing_mod"]:
            failed.append("context_missingness_above_moderate")
        reasons.append("|".join(failed) if failed else "moderate_rule_failed")
    return pd.Series(reasons, index=frame.index, dtype="string")


def local_column(chemical: float, taxonomy: str, context: float) -> str:
    return f"n_local_c{int(round(float(chemical)*100)):02d}_tax_{taxonomy}_b{int(round(float(context)*100)):02d}"


def normalize_species(frame: pd.DataFrame) -> pd.Series:
    return frame["latin_name"].astype("string").fillna("<missing>").str.strip().str.casefold()


def write_locked_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Locked AD rule already exists and differs: {path}. Remove only after an explicit protocol reset."
            )
        return
    path.write_text(text, encoding="utf-8")


def render_selection_note(
    selected: pd.Series,
    rule: dict[str, Any],
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Validation-only AD rule selection",
            "",
            f"> Locked candidate: `{selected['candidate_id']}`; status: `{rule['status']}`.",
            "",
            "## Selection boundary",
            "",
            "- Candidate thresholds were evaluated only on the 2,433 Stage-3 validation records.",
            "- The outer-test targets were not passed to the calibration function.",
            "- The search was a coarse, interpretable grid with nested High within Moderate rules.",
            "- Pareto objectives balanced High-tier MAE/P90 AE, coverage and task-wise direction; final selection used a predeclared direction/coverage score rather than minimum validation MAE alone.",
            "",
            "## Locked thresholds",
            "",
            "```json",
            json.dumps(rule["thresholds"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Validation result",
            "",
            markdown_table(summary),
            "",
            f"Candidates/Pareto candidates: {len(candidates)}/{int(candidates['pareto'].sum())}.",
            f"Eligible task comparisons with >=15 High and >=15 Low rows: {int(selected['eligible_task_count'])}; High had lower MAE in {int(selected['task_high_better_count'])}.",
            "",
            "The rule is now immutable for outer-test evaluation. Final use of the term calibrated applicability domain still depends on whether the validation direction reproduces on test and is not driven by task composition.",
            "",
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
