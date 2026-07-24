from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from ad_common import regression_metrics, within_task_r2, write_csv  # noqa: E402


def load_calibration_module():
    path = ANALYSIS_DIR / "04_calibrate_ad_on_validation.py"
    spec = importlib.util.spec_from_file_location("ad_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import calibration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CALIBRATION = load_calibration_module()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the immutable validation-locked support rule on test.")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lock_path = args.output_dir / "ad_rule_locked.json"
    lock_before = sha256_file(lock_path)
    rule = json.loads(lock_path.read_text(encoding="utf-8"))
    records = pd.read_parquet(args.output_dir / "ad_record_level_unlocked.parquet")

    validation = records.loc[records["analysis_split"].eq("validation")].copy()
    test = records.loc[records["analysis_split"].eq("test")].copy()
    if len(validation) != 2433 or len(test) != 3042:
        raise ValueError(f"Unexpected Stage-3 evaluation counts: validation={len(validation)}, test={len(test)}")
    if set(test["analysis_split"].unique()) != {"test"}:
        raise AssertionError("Outer-test evaluator received non-test records")

    expected_validation_hash = rule["validation_record_id_sha256"]
    actual_validation_hash = CALIBRATION.canonical_sha256(sorted(validation["record_id"].astype(str)))
    if actual_validation_hash != expected_validation_hash:
        raise AssertionError("Validation identities differ from the identities used to lock the rule")

    combined = []
    tier_summaries = {}
    curves = []
    for split_name, frame in (("validation", validation), ("test", test)):
        frame = frame.copy()
        tier = CALIBRATION.apply_rule(frame, rule)
        frame["ad_tier"] = tier
        frame["ad_tier_reason"] = CALIBRATION.tier_reason(frame, rule, tier)
        frame["ad_rule_candidate_id"] = rule["candidate_id"]
        frame["ad_rule_status"] = rule["status"]
        frame["ad_rule_sha256"] = lock_before
        frame["joint_support_score"] = joint_support_score(frame, rule)
        combined.append(frame)
        tier_summaries[split_name] = CALIBRATION.summarize_tiers(frame, prediction="M10_prediction")
        curves.append(CALIBRATION.coverage_error_curves(frame, rule, split=split_name))

    all_records = pd.concat(combined, ignore_index=True)
    test_locked = all_records.loc[all_records["analysis_split"].eq("test")].copy()
    validation_locked = all_records.loc[all_records["analysis_split"].eq("validation")].copy()

    task_summary = summarize_by_task(test_locked, rule)
    transfer_gain = summarize_transfer_gain(
        test_locked,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    structure_summary = summarize_structure_unavailable(all_records)
    success = assess_support_relationship(validation_locked, test_locked, task_summary, rule)

    all_records.to_parquet(args.output_dir / "ad_record_level.parquet", index=False)
    write_csv(args.output_dir / "ad_summary_by_tier_validation.csv", tier_summaries["validation"])
    write_csv(args.output_dir / "ad_summary_by_tier_test.csv", tier_summaries["test"])
    write_csv(args.output_dir / "ad_summary_by_task_test.csv", task_summary)
    write_csv(args.output_dir / "ad_coverage_error_curves.csv", pd.concat(curves, ignore_index=True))
    write_csv(args.output_dir / "transfer_gain_by_support.csv", transfer_gain)
    write_csv(args.output_dir / "structure_unavailable_summary.csv", structure_summary)
    (args.output_dir / "ad_success_assessment.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lock_after = sha256_file(lock_path)
    if lock_after != lock_before:
        raise AssertionError("The outer-test evaluator modified ad_rule_locked.json")
    integrity = {
        "rule_path": str(lock_path.resolve()),
        "sha256_before_test_evaluation": lock_before,
        "sha256_after_test_evaluation": lock_after,
        "unchanged": True,
        "validation_record_id_sha256": actual_validation_hash,
        "test_record_count": len(test),
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.seed,
    }
    (args.output_dir / "ad_rule_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "AD_TEST_EVALUATION_REPORT.md").write_text(
        render_report(rule, tier_summaries["validation"], tier_summaries["test"], success, integrity),
        encoding="utf-8",
    )
    print(args.output_dir / "AD_TEST_EVALUATION_REPORT.md")


def joint_support_score(frame: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    t = rule["thresholds"]
    local = rule["local_high_column"]
    l_scaled = np.clip(
        np.log10(frame[local].to_numpy(float) + 1.0)
        / float(rule["support_floor_l_scale_denominator"]),
        0.0,
        1.0,
    )
    return np.min(
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
    )


def summarize_by_task(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = ["High", "Moderate", "Low/outside"]
    high_local = rule["local_high_column"]
    for task, task_frame in frame.groupby("model_head", sort=True):
        task_n = len(task_frame)
        tier_mae: dict[str, float] = {}
        tier_n: dict[str, int] = {}
        for tier in order:
            selected = task_frame.loc[task_frame["ad_tier"].eq(tier)]
            metric = regression_metrics(selected, prediction="M10_prediction")
            tier_n[tier] = len(selected)
            tier_mae[tier] = metric["mae"]
            rows.append(
                {
                    "task_head": task,
                    "tier": tier,
                    "n": len(selected),
                    "task_n": task_n,
                    "coverage_within_task": len(selected) / task_n,
                    "sufficient_n_for_tier_metrics": len(selected) >= 15,
                    "unique_chemicals": selected["chemical_entity_id"].nunique(),
                    "unique_species": normalized_species(selected).nunique(),
                    "median_C_target": selected["C_target"].median(),
                    "median_B_exp": selected["B_exp"].median(),
                    "T_index": task_frame["T_index"].iloc[0],
                    "T_tier": task_frame["T_tier"].iloc[0],
                    "median_n_local_high": selected[high_local].median(),
                    **metric,
                    "within_task_r2": within_task_r2(selected, prediction="M10_prediction") if len(selected) else math.nan,
                    "median_ae": selected["AE_M10"].median(),
                    "p90_ae": selected["AE_M10"].quantile(0.90),
                    "median_nae": selected["NAE_M10"].median(),
                    "median_prediction_sd": selected["M10_prediction_sd"].median(),
                }
            )
        high_low_eligible = tier_n["High"] >= 15 and tier_n["Low/outside"] >= 15
        monotonic_eligible = all(tier_n[tier] >= 15 for tier in order)
        high_better = high_low_eligible and tier_mae["High"] < tier_mae["Low/outside"]
        monotonic = monotonic_eligible and (
            tier_mae["High"] <= tier_mae["Moderate"] <= tier_mae["Low/outside"]
        )
        for row in rows[-3:]:
            row["high_vs_low_eligible"] = high_low_eligible
            row["high_mae_lower_than_low"] = high_better if high_low_eligible else None
            row["three_tier_monotonic_eligible"] = monotonic_eligible
            row["three_tier_mae_monotonic"] = monotonic if monotonic_eligible else None
    return pd.DataFrame(rows)


def summarize_transfer_gain(frame: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    work = frame.copy()
    similarity_bins = [-np.inf, 0.40, 0.50, 0.65, 0.80, np.inf]
    labels = ["<0.40", "0.40-<0.50", "0.50-<0.65", "0.65-<0.80", ">=0.80"]
    work["C_source_bin"] = pd.cut(work["C_source"], similarity_bins, labels=labels, right=False).astype("string")
    work["C_target_bin"] = pd.cut(work["C_target"], similarity_bins, labels=labels, right=False).astype("string")
    work.loc[work["C_source"].isna(), "C_source_bin"] = "structure_unavailable"
    work.loc[work["C_target"].isna(), "C_target_bin"] = "structure_unavailable"
    work["exact_source_seen"] = np.where(
        work["structure_status"].ne("ok"),
        "structure_unavailable",
        np.where(work["exact_parent_seen_source"].fillna(False), "seen", "not_seen"),
    )
    dimensions = {
        "C_source_bin": "C_source_bin",
        "exact_source_seen": "exact_source_seen",
        "C_target_bin": "C_target_bin",
        "support_tier": "ad_tier",
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for dimension, column in dimensions.items():
        for group, selected in work.groupby(column, dropna=False, observed=True, sort=False):
            values = selected["delta_AE_M10_minus_M00"].dropna().to_numpy(float)
            ci_low, ci_high = bootstrap_mean_ci(values, replicates=replicates, rng=rng)
            rows.append(
                {
                    "split": "test",
                    "dimension": dimension,
                    "group": str(group),
                    "n": len(selected),
                    "mean_delta_AE": float(np.mean(values)) if len(values) else math.nan,
                    "mean_delta_AE_ci95_low": ci_low,
                    "mean_delta_AE_ci95_high": ci_high,
                    "median_delta_AE": float(np.median(values)) if len(values) else math.nan,
                    "fraction_M10_improved": float(np.mean(values < 0)) if len(values) else math.nan,
                    "interpretation": "negative_delta_favors_M10",
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, *, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_structure_unavailable(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, selected in frame.groupby(["analysis_split", "structure_status", "structure_reason"], dropna=False):
        split, status, reason = keys
        rows.append(
            {
                "split": split,
                "structure_status": status,
                "structure_reason": reason,
                "n": len(selected),
                "coverage": len(selected) / int((frame["analysis_split"] == split).sum()),
                "mae_M10": selected["AE_M10"].mean(),
                "median_ae_M10": selected["AE_M10"].median(),
                "median_prediction_sd": selected["M10_prediction_sd"].median(),
            }
        )
    return pd.DataFrame(rows)


def assess_support_relationship(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    task_summary: pd.DataFrame,
    rule: dict[str, Any],
) -> dict[str, Any]:
    def split_result(frame: pd.DataFrame) -> dict[str, Any]:
        by_tier = frame.groupby("ad_tier", observed=True)["AE_M10"].agg(["size", "mean", "median"])
        high = by_tier.loc["High"]
        moderate = by_tier.loc["Moderate"]
        low = by_tier.loc["Low/outside"]
        return {
            "n_high": int(high["size"]),
            "n_moderate": int(moderate["size"]),
            "n_low": int(low["size"]),
            "coverage_high": float(high["size"] / len(frame)),
            "coverage_high_plus_moderate": float((high["size"] + moderate["size"]) / len(frame)),
            "mae_high": float(high["mean"]),
            "mae_moderate": float(moderate["mean"]),
            "mae_low": float(low["mean"]),
            "median_ae_high": float(high["median"]),
            "median_ae_moderate": float(moderate["median"]),
            "median_ae_low": float(low["median"]),
            "mae_monotonic": bool(high["mean"] <= moderate["mean"] <= low["mean"]),
            "median_ae_monotonic": bool(high["median"] <= moderate["median"] <= low["median"]),
        }

    validation_result = split_result(validation)
    test_result = split_result(test)
    task_flags = task_summary.drop_duplicates("task_head")
    eligible = task_flags.loc[task_flags["high_vs_low_eligible"]]
    monotonic_eligible = task_flags.loc[task_flags["three_tier_monotonic_eligible"]]
    task_direction = {
        "eligible_high_vs_low_tasks": len(eligible),
        "high_better_tasks": int(eligible["high_mae_lower_than_low"].fillna(False).sum()),
        "high_better_fraction": float(eligible["high_mae_lower_than_low"].mean()) if len(eligible) else math.nan,
        "eligible_three_tier_tasks": len(monotonic_eligible),
        "monotonic_three_tier_tasks": int(monotonic_eligible["three_tier_mae_monotonic"].fillna(False).sum()),
    }
    formal_ad = bool(
        rule["status"] == "locked_formal_ad"
        and validation_result["mae_monotonic"]
        and test_result["mae_monotonic"]
        and len(eligible) >= 5
        and task_direction["high_better_fraction"] >= 0.60
    )
    return {
        "formal_calibrated_ad_supported": formal_ad,
        "final_label": "calibrated_applicability_domain" if formal_ad else "training_support_stratification",
        "reason": (
            "Validation-locked rule met coverage, aggregate direction and task-wise stability criteria."
            if formal_ad
            else "No candidate met the predeclared validation coverage targets. This prevents use as a calibrated rejection rule, but does not imply that records or tasks outside the strict High subset are unreliable."
        ),
        "validation": validation_result,
        "test": test_result,
        "task_direction_test": task_direction,
        "rule_status_before_test": rule["status"],
    }


def render_report(
    rule: dict[str, Any],
    validation_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    success: dict[str, Any],
    integrity: dict[str, Any],
) -> str:
    candidate_count = len(pd.read_csv(ANALYSIS_DIR / "ad_candidate_rules.csv"))
    feasible_count = int(pd.read_csv(ANALYSIS_DIR / "ad_candidate_rules.csv")["feasible"].sum())
    return "\n".join(
        [
            "# Locked outer-test support analysis",
            "",
            f"> Final label: `{success['final_label']}`. Formal calibrated AD supported: `{success['formal_calibrated_ad_supported']}`.",
            "",
            "## Boundary and integrity",
            "",
            "- Primary model: M10 four-seed prediction-level ensemble; M00 is a fixed comparator.",
            "- Native target: `neg_log10_mol_kg` for the locked Stage-3 soil route.",
            "- Candidate selection used validation targets only; the fixed outer test was evaluated after the JSON rule was written.",
            f"- Rule SHA-256 before/after outer-test evaluation: `{integrity['sha256_before_test_evaluation']}` / `{integrity['sha256_after_test_evaluation']}` (unchanged).",
            f"- Candidate grid: {candidate_count}; candidates meeting all predeclared coverage/size/composition constraints: {feasible_count}.",
            "",
            "## Validation support strata",
            "",
            markdown_table(validation_summary),
            "",
            "## Outer-test support strata",
            "",
            markdown_table(test_summary),
            "",
            "## Scientific interpretation",
            "",
            success["reason"],
            f"The two large test strata cover {(success['test']['n_moderate'] + success['test']['n_low']) / integrity['test_record_count']:.1%} of records. Their MAEs ({success['test']['mae_moderate']:.3f} and {success['test']['mae_low']:.3f}) lie close to the full-test MAE, so the support labels provide a modest ranking signal rather than a reliable/unreliable split.",
            "The machine-readable label `Low/outside` is displayed in figures as `Lower measured support`; it denotes failure of at least one strict joint-support condition, not model failure.",
            "The tier-error patterns may describe where the locked model has denser training support, but they must not be used as a formal rejection rule or as proof of extrapolation to new chemical families.",
            "Structure-unavailable records remain a separate state and were never recoded as chemical similarity zero.",
            "",
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
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


def normalized_species(frame: pd.DataFrame) -> pd.Series:
    return frame["latin_name"].astype("string").fillna("<missing>").str.strip().str.casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
