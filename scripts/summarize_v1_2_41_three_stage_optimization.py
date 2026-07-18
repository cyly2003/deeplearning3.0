from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_v1_2_40_paired_mass_molar import (
    metrics,
    normalize_predictions,
    stable_hash,
    unique_by_aggregate,
    write_csv,
)


CANDIDATE_PATTERN = re.compile(r"_(S[1-3])_seed(\d+)")
BASELINE_PATTERN = re.compile(r"_X0_molar_seed(\d+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the v1.2.41 stage-3 challenger using validation only, then "
            "summarize the locked winner without using test metrics for selection."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--screen-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--final-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Read and write validation rows only until the winner is locked.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.baseline_root, baseline=True)
    runs.extend(discover_runs(args.root, baseline=False))
    rows, prediction_sets = summarize_runs(runs, include_test=not args.selection_only)
    assert_validation_identities(rows, screen_seeds=args.screen_seeds)
    screen_rows = build_validation_screen(rows, screen_seeds=args.screen_seeds)
    selection = select_candidate(screen_rows, screen_seeds=args.screen_seeds)
    ensemble_rows = (
        []
        if args.selection_only
        else build_ensemble_summary(
            prediction_sets,
            selected_candidate=selection.get("selected_candidate"),
            final_seeds=args.final_seeds,
        )
    )
    write_csv(args.output_dir / "all_seed_metrics.csv", rows)
    write_csv(args.output_dir / "validation_candidate_summary.csv", screen_rows)
    write_csv(args.output_dir / "ensemble_metrics.csv", ensemble_rows)
    selection.update(
        {
            "selection_uses_test": False,
            "screen_seeds": list(args.screen_seeds),
            "final_seeds": list(args.final_seeds),
            "primary_target": "neg_log10_mol_kg",
            "paired_reporting_target": "neg_log10_mg_kg",
            "selection_metrics": ["validation_native_r2", "validation_native_mae"],
            "selection_only": bool(args.selection_only),
        }
    )
    (args.output_dir / "selected_candidate.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True))


def discover_runs(root: Path, *, baseline: bool) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    pattern = BASELINE_PATTERN if baseline else CANDIDATE_PATTERN
    for manifest_path in root.glob("**/manifest.json"):
        match = pattern.search("/".join(manifest_path.parts))
        predictions_path = manifest_path.parent / "predictions.csv"
        if match is None or not predictions_path.is_file():
            continue
        if baseline:
            candidate, seed = "S0", int(match.group(1))
        else:
            candidate, seed = str(match.group(1)), int(match.group(2))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        runs.append(
            {
                "candidate": candidate,
                "seed": seed,
                "manifest": manifest,
                "predictions": predictions,
                "run_dir": str(manifest_path.parent),
            }
        )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        key = (str(run["candidate"]), int(run["seed"]))
        if key in unique:
            raise ValueError(f"Duplicate optimization run for candidate/seed={key}")
        unique[key] = run
    return [unique[key] for key in sorted(unique)]


def summarize_runs(
    runs: list[dict[str, Any]], *, include_test: bool
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        candidate = str(run["candidate"])
        seed = int(run["seed"])
        normalized = normalize_predictions(run["predictions"], scale="molar")
        parts = ("validation", "test") if include_test else ("validation",)
        for part in parts:
            part_rows = [row for row in normalized if row["evaluation_part"] == part]
            if not part_rows:
                continue
            native = metrics(part_rows, truth="y_native", prediction="pred_native")
            common = metrics(part_rows, truth="y_mgkg", prediction="pred_mgkg")
            manifest = run["manifest"]
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "evaluation_part": part,
                    "n": native["n"],
                    "native_r2": native["r2"],
                    "native_rmse": native["rmse"],
                    "native_mae": native["mae"],
                    "common_mgkg_r2": common["r2"],
                    "common_mgkg_rmse": common["rmse"],
                    "common_mgkg_mae": common["mae"],
                    "aggregate_id_sha256": stable_hash(
                        row["aggregate_id"] for row in part_rows
                    ),
                    "result_ids_sha256": stable_hash(
                        result_id
                        for row in part_rows
                        for result_id in row["result_ids"]
                    ),
                    "stage3_epochs_ran": manifest.get("finetune_mgkg_epochs_ran"),
                    "stage3_mse_weight": (
                        manifest.get("finetune_mgkg", {})
                        .get("regression_loss", {})
                        .get("mse_weight")
                    ),
                    "swa_phase": manifest.get("swa", {}).get("phase"),
                    "swa_updates": manifest.get("swa", {}).get("updates"),
                    "swa_applied": manifest.get("swa", {}).get("applied"),
                    "run_dir": run["run_dir"],
                }
            )
            prediction_sets[(candidate, seed, part)] = part_rows
    return sorted(rows, key=lambda row: (row["candidate"], row["seed"], row["evaluation_part"])), prediction_sets


def assert_validation_identities(rows: list[dict[str, Any]], *, screen_seeds: list[int]) -> None:
    validation = {
        (str(row["candidate"]), int(row["seed"])): row
        for row in rows
        if row["evaluation_part"] == "validation"
    }
    missing_baseline = [seed for seed in screen_seeds if ("S0", seed) not in validation]
    if missing_baseline:
        raise ValueError(f"Missing S0 validation baseline for screen seeds: {missing_baseline}")
    for (candidate, seed), row in validation.items():
        if candidate == "S0" or seed not in screen_seeds:
            continue
        baseline = validation[("S0", seed)]
        identity_fields = ("n", "aggregate_id_sha256", "result_ids_sha256")
        mismatches = {
            field: {"candidate": row[field], "baseline": baseline[field]}
            for field in identity_fields
            if row[field] != baseline[field]
        }
        if mismatches:
            raise ValueError(
                f"Validation identity mismatch for candidate={candidate} seed={seed}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )


def build_validation_screen(
    rows: list[dict[str, Any]], *, screen_seeds: list[int]
) -> list[dict[str, Any]]:
    wanted = set(screen_seeds)
    validation = [
        row
        for row in rows
        if row["evaluation_part"] == "validation" and int(row["seed"]) in wanted
    ]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_by_seed: dict[int, dict[str, Any]] = {}
    for row in validation:
        by_candidate[str(row["candidate"])].append(row)
        if row["candidate"] == "S0":
            baseline_by_seed[int(row["seed"])] = row
    output: list[dict[str, Any]] = []
    for candidate, candidate_rows in sorted(by_candidate.items()):
        candidate_by_seed = {int(row["seed"]): row for row in candidate_rows}
        complete = all(
            seed in candidate_by_seed and seed in baseline_by_seed
            for seed in screen_seeds
        )
        paired = [
            (candidate_by_seed[seed], baseline_by_seed[seed])
            for seed in screen_seeds
            if seed in candidate_by_seed and seed in baseline_by_seed
        ]
        complete = complete and len(paired) == len(screen_seeds)
        r2_values = [float(row["native_r2"]) for row in candidate_rows]
        mae_values = [float(row["native_mae"]) for row in candidate_rows]
        output.append(
            {
                "candidate": candidate,
                "complete_screen": complete,
                "seed_count": len(candidate_rows),
                "seeds": ",".join(str(seed) for seed in sorted(candidate_by_seed)),
                "validation_native_r2_mean": statistics.fmean(r2_values),
                "validation_native_r2_sd": statistics.stdev(r2_values) if len(r2_values) > 1 else 0.0,
                "validation_native_mae_mean": statistics.fmean(mae_values),
                "validation_native_mae_sd": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0,
                "delta_r2_vs_s0_mean": (
                    statistics.fmean(
                        float(candidate_row["native_r2"]) - float(baseline_row["native_r2"])
                        for candidate_row, baseline_row in paired
                    )
                    if paired
                    else None
                ),
                "delta_mae_vs_s0_mean": (
                    statistics.fmean(
                        float(candidate_row["native_mae"]) - float(baseline_row["native_mae"])
                        for candidate_row, baseline_row in paired
                    )
                    if paired
                    else None
                ),
                "paired_seeds_improving_both": sum(
                    float(candidate_row["native_r2"]) > float(baseline_row["native_r2"])
                    and float(candidate_row["native_mae"]) < float(baseline_row["native_mae"])
                    for candidate_row, baseline_row in paired
                ),
            }
        )
    return output


def select_candidate(rows: list[dict[str, Any]], *, screen_seeds: list[int]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["candidate"] != "S0"
        and bool(row["complete_screen"])
        and float(row["delta_r2_vs_s0_mean"]) > 0
        and float(row["delta_mae_vs_s0_mean"]) < 0
        and int(row["paired_seeds_improving_both"]) >= 1
    ]
    if not eligible:
        return {
            "selected_candidate": None,
            "selection_status": "no_challenger_improves_both_validation_metrics",
        }
    eligible.sort(
        key=lambda row: (
            float(row["validation_native_mae_mean"]),
            -float(row["validation_native_r2_mean"]),
            str(row["candidate"]),
        )
    )
    selected = eligible[0]
    return {
        "selected_candidate": selected["candidate"],
        "selection_status": "locked_from_validation",
        "selected_validation_native_r2_mean": selected["validation_native_r2_mean"],
        "selected_validation_native_mae_mean": selected["validation_native_mae_mean"],
        "selected_delta_r2_vs_s0_mean": selected["delta_r2_vs_s0_mean"],
        "selected_delta_mae_vs_s0_mean": selected["delta_mae_vs_s0_mean"],
    }


def build_ensemble_summary(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    selected_candidate: str | None,
    final_seeds: list[int],
) -> list[dict[str, Any]]:
    candidates = ["S0"] + ([selected_candidate] if selected_candidate else [])
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        for part in ("validation", "test"):
            sets = [prediction_sets.get((candidate, seed, part)) for seed in final_seeds]
            if any(rows is None for rows in sets):
                continue
            indexed = [unique_by_aggregate(rows or []) for rows in sets]
            identities = set(indexed[0])
            if any(set(item) != identities for item in indexed[1:]):
                raise ValueError(f"Ensemble identities differ for {candidate} {part}")
            ensemble: list[dict[str, Any]] = []
            for aggregate_id in sorted(identities):
                source_rows = [item[aggregate_id] for item in indexed]
                reference = dict(source_rows[0])
                reference["pred_native_ensemble"] = statistics.fmean(
                    float(row["pred_native"]) for row in source_rows
                )
                reference["pred_mgkg_ensemble"] = statistics.fmean(
                    float(row["pred_mgkg"]) for row in source_rows
                )
                ensemble.append(reference)
            native = metrics(ensemble, truth="y_native", prediction="pred_native_ensemble")
            common = metrics(ensemble, truth="y_mgkg", prediction="pred_mgkg_ensemble")
            output.append(
                {
                    "candidate": candidate,
                    "evaluation_part": part,
                    "seed_count": len(final_seeds),
                    "seeds": ",".join(str(seed) for seed in final_seeds),
                    "n": native["n"],
                    "native_r2": native["r2"],
                    "native_rmse": native["rmse"],
                    "native_mae": native["mae"],
                    "common_mgkg_r2": common["r2"],
                    "common_mgkg_rmse": common["rmse"],
                    "common_mgkg_mae": common["mae"],
                    "aggregate_id_sha256": stable_hash(identities),
                }
            )
    return output


if __name__ == "__main__":
    main()
