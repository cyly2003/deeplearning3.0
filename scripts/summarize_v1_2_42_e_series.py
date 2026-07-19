from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qsar_tl.training.meta_ensemble import (  # noqa: E402
    fit_candidate,
    predict_candidate,
    regression_metrics,
)


TARGET_NAME = "neg_log10_mol_kg"
OOF_PATTERN = re.compile(r"_OOF_([DX])_seed(\d+)_fold(\d+)")
BASELINE_PATTERN = re.compile(r"_(D_molar|X0_molar)_seed(\d+)")
VALIDATION_PARTS = {"valid"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select E1-E3 from two-seed OOF validation only, then refit the "
            "locked winner on four-seed OOF predictions and evaluate the outer test once."
        )
    )
    parser.add_argument("--oof-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--screen-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--final-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--meta-seed", type=int, default=4242)
    parser.add_argument("--selection-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen_rows = build_paired_oof_rows(
        args.oof_root,
        seeds=args.screen_seeds,
        folds=args.folds,
    )
    selection_ids = load_selection_identities(args.baseline_root, seed=42)
    selection, validation_rows, screen_models = select_candidate(
        screen_rows,
        selection_ids=selection_ids,
        meta_seed=args.meta_seed,
    )
    selection.update(
        {
            "schema": "v1_2_42_e_series_selection_v1",
            "screen_seeds": list(args.screen_seeds),
            "final_seeds": list(args.final_seeds),
            "folds": int(args.folds),
            "selection_uses_test": False,
            "selection_identity_source": "v1.2.40_X0_molar_seed42_finetune_mgkg_validation",
            "selection_identity_n": len(selection_ids),
            "selection_identity_sha256": stable_hash(selection_ids),
            "outer_test_read": not bool(args.selection_only),
        }
    )
    write_csv(args.output_dir / "validation_candidate_summary.csv", validation_rows)
    write_json(args.output_dir / "selected_candidate.json", selection)
    write_json(args.output_dir / "screen_models.json", screen_models)
    if args.selection_only or not selection.get("selected_candidate"):
        write_csv(args.output_dir / "ensemble_metrics.csv", [])
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
        return

    locked = str(selection["selected_candidate"])
    final_oof_rows = build_paired_oof_rows(
        args.oof_root,
        seeds=args.final_seeds,
        folds=args.folds,
    )
    final_model = fit_candidate(locked, final_oof_rows, seed=args.meta_seed)
    test_rows = load_baseline_test_ensemble(
        args.baseline_root,
        seeds=args.final_seeds,
    )
    winner_prediction = predict_candidate(final_model, test_rows)
    for row, prediction in zip(test_rows, winner_prediction):
        row["winner_pred"] = float(prediction)
    metrics_rows = build_final_metrics(
        validation_rows,
        test_rows=test_rows,
        winner=locked,
        seeds=args.final_seeds,
    )
    write_csv(args.output_dir / "ensemble_metrics.csv", metrics_rows)
    write_json(args.output_dir / "final_model.json", final_model)
    write_csv(
        args.output_dir / "winner_test_predictions.csv",
        export_test_predictions(test_rows, winner=locked),
    )
    audit = {
        "schema": "v1_2_42_e_series_protocol_audit_v1",
        "selection_uses_test": False,
        "outer_test_read_after_winner_lock": True,
        "oof_outer_test_in_training": False,
        "screen_oof_n": len(screen_rows),
        "final_oof_n": len(final_oof_rows),
        "outer_test_n": len(test_rows),
        "locked_winner": locked,
        "screen_oof_identity_sha256": stable_hash(row["aggregate_id"] for row in screen_rows),
        "final_oof_identity_sha256": stable_hash(row["aggregate_id"] for row in final_oof_rows),
        "outer_test_identity_sha256": stable_hash(row["aggregate_id"] for row in test_rows),
    }
    write_json(args.output_dir / "oof_protocol_audit.json", audit)
    print(json.dumps({**selection, **audit}, ensure_ascii=False, sort_keys=True))


def build_paired_oof_rows(root: Path, *, seeds: Sequence[int], folds: int) -> list[dict[str, Any]]:
    runs = discover_oof_runs(root)
    expected = {
        (architecture, int(seed), fold)
        for architecture in ("D", "X")
        for seed in seeds
        for fold in range(1, folds + 1)
    }
    missing = sorted(expected - set(runs))
    if missing:
        raise ValueError(f"Missing required OOF runs: {missing[:10]}")
    by_architecture: dict[str, dict[int, dict[str, dict[str, Any]]]] = {"D": {}, "X": {}}
    for architecture in ("D", "X"):
        for seed in seeds:
            combined: dict[str, dict[str, Any]] = {}
            for fold in range(1, folds + 1):
                for row in read_prediction_rows(runs[(architecture, int(seed), fold)]):
                    identity = row["aggregate_id"]
                    if identity in combined:
                        raise ValueError(
                            f"OOF identity predicted more than once for {architecture} seed={seed}: {identity}"
                        )
                    combined[identity] = row
            by_architecture[architecture][int(seed)] = combined

    reference_ids = set(by_architecture["D"][int(seeds[0])])
    if not reference_ids:
        raise ValueError("OOF predictions contain no target rows.")
    for architecture in ("D", "X"):
        for seed in seeds:
            observed = set(by_architecture[architecture][int(seed)])
            if observed != reference_ids:
                raise ValueError(
                    f"OOF identity coverage mismatch for {architecture} seed={seed}: "
                    f"expected={len(reference_ids)} observed={len(observed)}"
                )

    output: list[dict[str, Any]] = []
    for identity in sorted(reference_ids):
        direct_rows = [by_architecture["D"][int(seed)][identity] for seed in seeds]
        transfer_rows = [by_architecture["X"][int(seed)][identity] for seed in seeds]
        reference = direct_rows[0]
        assert_truth_alignment(identity, direct_rows + transfer_rows)
        output.append(
            {
                **reference,
                "direct_pred": statistics.fmean(row["y_pred"] for row in direct_rows),
                "direct_std": population_std(row["y_pred"] for row in direct_rows),
                "transfer_pred": statistics.fmean(row["y_pred"] for row in transfer_rows),
                "transfer_std": population_std(row["y_pred"] for row in transfer_rows),
                "seed_count": len(seeds),
            }
        )
    return output


def discover_oof_runs(root: Path) -> dict[tuple[str, int, int], Path]:
    runs: dict[tuple[str, int, int], Path] = {}
    for path in root.glob("**/predictions.csv"):
        match = OOF_PATTERN.search("/".join(path.parts))
        if match is None:
            continue
        key = (match.group(1), int(match.group(2)), int(match.group(3)))
        if key in runs:
            raise ValueError(f"Duplicate OOF run for {key}: {path}")
        manifest = path.parent / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"OOF predictions are missing manifest: {path}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        requested_parts = set(payload.get("prediction_output_split_parts", []))
        required_part = "valid"
        if requested_parts != {required_part}:
            raise ValueError(
                f"OOF prediction export contract differs for {key}: {sorted(requested_parts)}"
            )
        if key[0] == "X" and payload.get("finetune_mgkg_validation_source") != "internal_finetune_fraction":
            raise ValueError(f"Transfer OOF stage-3 monitor is not internal to the training fold: {key}")
        runs[key] = path
    return runs


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("target_name", "")) != TARGET_NAME:
                continue
            split_part = str(raw.get("split_part", "")).strip().lower()
            if split_part not in VALIDATION_PARTS:
                raise ValueError(f"OOF file contains non-held-out target row: {path} part={split_part}")
            identity = str(raw.get("aggregate_id", "")).strip()
            mw = required_positive(raw.get("molecular_weight_g_mol_used"), "molecular weight")
            rows.append(
                {
                    "aggregate_id": identity,
                    "result_ids": parse_result_ids(raw.get("result_ids")),
                    "y_true": required_float(raw.get("y_true"), "y_true"),
                    "y_pred": required_float(raw.get("y_pred"), "y_pred"),
                    "molecular_weight_g_mol_used": mw,
                    "task_family": str(raw.get("task_family") or raw.get("base_task_head") or raw.get("task_head") or ""),
                    "taxon_group_l1": str(raw.get("taxon_group_l1") or ""),
                    "effect_level_x": optional_float(raw.get("effect_level_x")),
                }
            )
    if not rows:
        raise ValueError(f"No {TARGET_NAME} held-out rows found in {path}")
    return rows


def load_selection_identities(root: Path, *, seed: int) -> set[str]:
    path = discover_baseline_prediction(root, cell="X0_molar", seed=seed)
    identities: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("target_name", "")) != TARGET_NAME:
                continue
            if str(row.get("split_part", "")).strip().lower() != "finetune_mgkg_validation":
                continue
            identity = str(row.get("aggregate_id", "")).strip()
            if identity in identities:
                raise ValueError(f"Duplicate selection identity in baseline: {identity}")
            identities.add(identity)
    if not identities:
        raise ValueError("The locked v1.2.40 validation identity set is empty.")
    return identities


def select_candidate(
    rows: list[dict[str, Any]],
    *,
    selection_ids: set[str],
    meta_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    observed = {row["aggregate_id"] for row in rows}
    missing = selection_ids - observed
    if missing:
        raise ValueError(f"Locked selection identities are absent from OOF predictions: {sorted(missing)[:5]}")
    fit_rows = [row for row in rows if row["aggregate_id"] not in selection_ids]
    validation = [row for row in rows if row["aggregate_id"] in selection_ids]
    if not fit_rows or not validation:
        raise ValueError("Meta train/selection partition is empty.")
    candidates: dict[str, Sequence[float]] = {
        "E0": [row["transfer_pred"] for row in validation]
    }
    models: dict[str, Any] = {}
    for candidate in ("E1", "E2", "E3"):
        model = fit_candidate(candidate, fit_rows, seed=meta_seed)
        models[candidate] = model
        candidates[candidate] = predict_candidate(model, validation)
    truth = [row["y_true"] for row in validation]
    baseline_metrics = regression_metrics(truth, candidates["E0"])
    metrics_rows: list[dict[str, Any]] = []
    for candidate, prediction in candidates.items():
        current = regression_metrics(truth, prediction)
        improves_both = current.r2 > baseline_metrics.r2 and current.mae < baseline_metrics.mae
        metrics_rows.append(
            {
                "candidate": candidate,
                "evaluation_part": "selection_validation_oof",
                "n": current.n,
                "native_r2": current.r2,
                "native_rmse": current.rmse,
                "native_mae": current.mae,
                "delta_r2_vs_e0": current.r2 - baseline_metrics.r2,
                "delta_mae_vs_e0": current.mae - baseline_metrics.mae,
                "improves_both_vs_e0": improves_both,
                "selection_identity_sha256": stable_hash(selection_ids),
            }
        )
    eligible = [row for row in metrics_rows if row["candidate"] != "E0" and row["improves_both_vs_e0"]]
    eligible.sort(key=lambda row: (float(row["native_mae"]), -float(row["native_r2"]), str(row["candidate"])))
    if not eligible:
        selection = {
            "selected_candidate": None,
            "selection_status": "no_candidate_improves_both_oof_validation_metrics",
            "meta_train_n": len(fit_rows),
        }
    else:
        winner = eligible[0]
        selection = {
            "selected_candidate": winner["candidate"],
            "selection_status": "locked_from_oof_validation",
            "selected_validation_native_r2": winner["native_r2"],
            "selected_validation_native_mae": winner["native_mae"],
            "selected_delta_r2_vs_e0": winner["delta_r2_vs_e0"],
            "selected_delta_mae_vs_e0": winner["delta_mae_vs_e0"],
            "meta_train_n": len(fit_rows),
        }
    return selection, metrics_rows, models


def load_baseline_test_ensemble(root: Path, *, seeds: Sequence[int]) -> list[dict[str, Any]]:
    by_cell: dict[str, dict[int, dict[str, dict[str, Any]]]] = {"D_molar": {}, "X0_molar": {}}
    for cell in by_cell:
        for seed in seeds:
            path = discover_baseline_prediction(root, cell=cell, seed=int(seed))
            indexed: dict[str, dict[str, Any]] = {}
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for raw in csv.DictReader(handle):
                    if str(raw.get("target_name", "")) != TARGET_NAME:
                        continue
                    if str(raw.get("split_part", "")).strip().lower() != "test":
                        continue
                    identity = str(raw.get("aggregate_id", "")).strip()
                    if identity in indexed:
                        raise ValueError(f"Duplicate outer-test identity for {cell} seed={seed}: {identity}")
                    indexed[identity] = {
                        "aggregate_id": identity,
                        "result_ids": parse_result_ids(raw.get("result_ids")),
                        "y_true": required_float(raw.get("y_true"), "y_true"),
                        "y_pred": required_float(raw.get("y_pred"), "y_pred"),
                        "molecular_weight_g_mol_used": required_positive(
                            raw.get("molecular_weight_g_mol_used"), "molecular weight"
                        ),
                        "task_family": str(raw.get("task_family") or raw.get("base_task_head") or raw.get("task_head") or ""),
                        "taxon_group_l1": str(raw.get("taxon_group_l1") or ""),
                        "effect_level_x": optional_float(raw.get("effect_level_x")),
                    }
            by_cell[cell][int(seed)] = indexed
    reference_ids = set(by_cell["D_molar"][int(seeds[0])])
    for cell in by_cell:
        for seed in seeds:
            if set(by_cell[cell][int(seed)]) != reference_ids:
                raise ValueError(f"Outer-test identity mismatch for {cell} seed={seed}")
    output: list[dict[str, Any]] = []
    for identity in sorted(reference_ids):
        direct_rows = [by_cell["D_molar"][int(seed)][identity] for seed in seeds]
        transfer_rows = [by_cell["X0_molar"][int(seed)][identity] for seed in seeds]
        assert_truth_alignment(identity, direct_rows + transfer_rows)
        output.append(
            {
                **direct_rows[0],
                "direct_pred": statistics.fmean(row["y_pred"] for row in direct_rows),
                "direct_std": population_std(row["y_pred"] for row in direct_rows),
                "transfer_pred": statistics.fmean(row["y_pred"] for row in transfer_rows),
                "transfer_std": population_std(row["y_pred"] for row in transfer_rows),
            }
        )
    if not output:
        raise ValueError("No outer-test target rows found in the locked baseline runs.")
    return output


def discover_baseline_prediction(root: Path, *, cell: str, seed: int) -> Path:
    matches = []
    for path in root.glob("**/predictions.csv"):
        match = BASELINE_PATTERN.search("/".join(path.parts))
        if match and match.group(1) == cell and int(match.group(2)) == seed:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one baseline run for {cell} seed={seed}, found {len(matches)}")
    return matches[0]


def build_final_metrics(
    validation_rows: list[dict[str, Any]],
    *,
    test_rows: list[dict[str, Any]],
    winner: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    output = [dict(row) for row in validation_rows]
    truth_native = [row["y_true"] for row in test_rows]
    offset = [math.log10(1000.0 * row["molecular_weight_g_mol_used"]) for row in test_rows]
    truth_mgkg = [truth - current for truth, current in zip(truth_native, offset)]
    for candidate, key in (("E0", "transfer_pred"), (winner, "winner_pred")):
        prediction_native = [row[key] for row in test_rows]
        prediction_mgkg = [prediction - current for prediction, current in zip(prediction_native, offset)]
        native = regression_metrics(truth_native, prediction_native)
        common = regression_metrics(truth_mgkg, prediction_mgkg)
        output.append(
            {
                "candidate": candidate,
                "evaluation_part": "final_test",
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "n": native.n,
                "native_r2": native.r2,
                "native_rmse": native.rmse,
                "native_mae": native.mae,
                "common_mgkg_r2": common.r2,
                "common_mgkg_rmse": common.rmse,
                "common_mgkg_mae": common.mae,
                "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in test_rows),
            }
        )
    return output


def export_test_predictions(rows: Sequence[Mapping[str, Any]], *, winner: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        offset = math.log10(1000.0 * float(row["molecular_weight_g_mol_used"]))
        output.append(
            {
                "aggregate_id": row["aggregate_id"],
                "task_family": row["task_family"],
                "taxon_group_l1": row["taxon_group_l1"],
                "molecular_weight_g_mol_used": row["molecular_weight_g_mol_used"],
                "y_true_mol_kg": row["y_true"],
                "e0_pred_mol_kg": row["transfer_pred"],
                f"{winner.lower()}_pred_mol_kg": row["winner_pred"],
                "y_true_neg_log10_mg_kg": float(row["y_true"]) - offset,
                "e0_pred_neg_log10_mg_kg": float(row["transfer_pred"]) - offset,
                f"{winner.lower()}_pred_neg_log10_mg_kg": float(row["winner_pred"]) - offset,
            }
        )
    return output


def assert_truth_alignment(identity: str, rows: Sequence[Mapping[str, Any]]) -> None:
    truth = [float(row["y_true"]) for row in rows]
    if max(truth) - min(truth) > 1.0e-10:
        raise ValueError(f"Truth mismatch across paired predictions for {identity}")
    mw = [float(row["molecular_weight_g_mol_used"]) for row in rows]
    if max(mw) - min(mw) > 1.0e-10:
        raise ValueError(f"Molecular-weight mismatch across paired predictions for {identity}")


def population_std(values: Iterable[float]) -> float:
    normalized = [float(value) for value in values]
    return statistics.pstdev(normalized) if len(normalized) > 1 else 0.0


def parse_result_ids(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return sorted({str(item) for item in parsed if item not in (None, "")})


def required_float(value: Any, label: str) -> float:
    parsed = optional_float(value)
    if parsed is None:
        raise ValueError(f"Missing or non-finite {label}: {value!r}")
    return parsed


def required_positive(value: Any, label: str) -> float:
    parsed = required_float(value, label)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive: {value!r}")
    return parsed


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def stable_hash(values: Iterable[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else ["status"]
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
