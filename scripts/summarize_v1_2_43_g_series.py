from __future__ import annotations

import argparse
import csv
import hashlib
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


RUN_PATTERN = re.compile(r"_(G[0-3])_(screen|final)_seed(\d+)")
MIN_MEAN_DELTA_R2 = 0.005
MAX_MEAN_DELTA_MAE = -0.005
SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
PHASE_SPLITS = {
    "screen": "M_v1_2_43_g_screen",
    "final": "M_v1_2_43_g_final",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lock the v1.2.43 G-series winner using the fresh validation set only, "
            "then summarize G0 and the locked winner on the untouched outer test."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--screen-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--final-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--screen-split-summary", required=True, type=Path)
    parser.add_argument("--selection-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.selection_only:
        runs = discover_runs(args.root, phases={"screen"})
        rows, _ = summarize_runs(runs, include_test=False)
        assert_screen_validation_identities(rows, screen_seeds=args.screen_seeds)
        validation_summary = build_validation_summary(rows, screen_seeds=args.screen_seeds)
        selection = select_candidate(validation_summary, screen_seeds=args.screen_seeds)
        selection.update(selection_protocol_fields(args.screen_seeds, args.final_seeds))
        canonical_inputs = selection_canonical_inputs(
            runs=runs,
            validation_rows=rows,
            validation_summary=validation_summary,
            selection=selection,
        )
        screen_evidence = build_screen_evidence(
            runs,
            split_summary_path=args.screen_split_summary,
        )
        lock_path = args.output_dir / "winner_lock.json"
        if lock_path.exists():
            existing_lock = load_and_validate_winner_lock(lock_path)
            if existing_lock.get("canonical_inputs") != canonical_inputs:
                raise ValueError(
                    "Existing immutable winner lock differs from fresh validation rows, "
                    "summary, or selection content."
                )
            if existing_lock.get("screen_evidence") != screen_evidence:
                raise ValueError(
                    "Existing immutable winner lock differs from the current screen split "
                    "or run artifact identity."
                )
            print(json.dumps(existing_lock, ensure_ascii=False, sort_keys=True))
            return
        screen_metrics_path = args.output_dir / "screen_seed_metrics.csv"
        validation_summary_path = args.output_dir / "validation_candidate_summary.csv"
        selection_path = args.output_dir / "selected_candidate.json"
        write_csv(screen_metrics_path, rows)
        write_csv(validation_summary_path, validation_summary)
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_immutable_winner_lock(
            lock_path,
            selection=selection,
            locked_files={
                "screen_seed_metrics": screen_metrics_path,
                "validation_candidate_summary": validation_summary_path,
                "selected_candidate": selection_path,
            },
            canonical_inputs=canonical_inputs,
            screen_evidence=screen_evidence,
        )
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
        return

    lock = load_and_validate_winner_lock(args.output_dir / "winner_lock.json")
    winner = lock.get("selected_candidate")
    if winner not in {"G1", "G2", "G3"}:
        raise ValueError("Final summary requires a locked non-baseline winner.")
    runs = discover_runs(args.root, phases={"screen", "final"})
    current_screen_evidence = build_screen_evidence(
        [run for run in runs if run["phase"] == "screen"],
        split_summary_path=args.screen_split_summary,
    )
    if current_screen_evidence != lock.get("screen_evidence"):
        raise ValueError(
            "Final summary screen split/run artifacts differ from the immutable winner lock."
        )
    unexpected_final = sorted(
        {
            str(run["candidate"])
            for run in runs
            if run["phase"] == "final"
            and run["candidate"] not in {"G0", str(winner)}
        }
    )
    if unexpected_final:
        raise ValueError(
            "Final output root contains stale non-locked candidates: "
            f"{unexpected_final}"
        )
    rows, prediction_sets = summarize_runs(runs, include_test=True)
    assert_screen_validation_identities(rows, screen_seeds=args.screen_seeds)
    assert_final_boundaries(
        rows,
        selected_candidate=str(winner),
        screen_seeds=args.screen_seeds,
        final_seeds=args.final_seeds,
    )
    final_rows = [row for row in rows if row["phase"] == "final"]
    final_summary = build_final_single_model_summary(
        rows,
        selected_candidate=str(winner),
        final_seeds=args.final_seeds,
    )
    ensemble = build_test_ensemble_summary(
        prediction_sets,
        selected_candidate=str(winner),
        final_seeds=args.final_seeds,
    )
    write_csv(args.output_dir / "final_seed_metrics.csv", final_rows)
    write_csv(args.output_dir / "final_single_model_summary.csv", final_summary)
    write_csv(args.output_dir / "ensemble_metrics.csv", ensemble)
    print(json.dumps(lock, ensure_ascii=False, sort_keys=True))


def selection_protocol_fields(
    screen_seeds: list[int], final_seeds: list[int]
) -> dict[str, Any]:
    return {
        "selection_uses_test": False,
        "outer_test_assignments_read": False,
        "outer_test_metrics_read": False,
        "screen_seeds": list(screen_seeds),
        "final_seeds": list(final_seeds),
        "validation_seed": 17073,
        "min_mean_delta_r2": MIN_MEAN_DELTA_R2,
        "max_mean_delta_mae": MAX_MEAN_DELTA_MAE,
        "requires_every_screen_seed_improve_both": True,
        "primary_target": "neg_log10_mol_kg",
        "paired_reporting_target": "neg_log10_mg_kg",
    }


def write_immutable_winner_lock(
    path: Path,
    *,
    selection: dict[str, Any],
    locked_files: dict[str, Path],
    canonical_inputs: dict[str, str],
    screen_evidence: dict[str, Any],
) -> dict[str, Any]:
    required_inputs = {
        "fresh_validation_predictions_sha256",
        "validation_metric_rows_sha256",
        "validation_candidate_summary_sha256",
        "selection_sha256",
    }
    if set(canonical_inputs) != required_inputs:
        raise ValueError("Cannot lock incomplete canonical validation inputs.")
    if canonical_inputs["selection_sha256"] != canonical_sha256(selection):
        raise ValueError("Cannot lock a selection with a mismatched canonical hash.")
    validate_screen_evidence(screen_evidence)
    required_file_labels = {
        "screen_seed_metrics",
        "validation_candidate_summary",
        "selected_candidate",
    }
    if set(locked_files) != required_file_labels:
        raise ValueError("Cannot lock an incomplete set of validation selection files.")
    if (
        canonical_inputs["validation_metric_rows_sha256"]
        != canonical_file_sha256(locked_files["screen_seed_metrics"])
        or canonical_inputs["validation_candidate_summary_sha256"]
        != canonical_file_sha256(locked_files["validation_candidate_summary"])
        or canonical_inputs["fresh_validation_predictions_sha256"]
        != screen_evidence.get("fresh_validation_predictions_sha256")
    ):
        raise ValueError("Cannot lock derived files or screen artifacts that differ from fresh inputs.")
    payload = {
        "schema": "v1_2_43_g_winner_lock_v2",
        "selected_candidate": selection.get("selected_candidate"),
        "selection_status": selection.get("selection_status"),
        "selection_uses_test": False,
        "outer_test_assignments_read": False,
        "outer_test_metrics_read": False,
        "screen_seeds": selection.get("screen_seeds", []),
        "selection_thresholds": {
            "min_mean_delta_r2": selection.get("min_mean_delta_r2"),
            "max_mean_delta_mae": selection.get("max_mean_delta_mae"),
            "requires_every_screen_seed_improve_both": selection.get(
                "requires_every_screen_seed_improve_both"
            ),
        },
        "canonical_inputs": canonical_inputs,
        "screen_evidence": screen_evidence,
        "locked_files": {
            label: {
                "path": str(file_path.resolve()),
                "sha256": file_sha256(file_path),
                "canonical_sha256": canonical_file_sha256(file_path),
            }
            for label, file_path in sorted(locked_files.items())
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != encoded:
            raise ValueError(
                "winner_lock.json already exists with different immutable selection content."
            )
        return payload
    path.write_text(encoded, encoding="utf-8")
    return payload


def load_and_validate_winner_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("Final summary requires the immutable winner_lock.json.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "v1_2_43_g_winner_lock_v2":
        raise ValueError("Unsupported or missing winner-lock schema.")
    if payload.get("selection_uses_test") is not False:
        raise ValueError("Winner lock does not attest validation-only selection.")
    locked_files = payload.get("locked_files")
    required_locked_files = {
        "screen_seed_metrics",
        "validation_candidate_summary",
        "selected_candidate",
    }
    if not isinstance(locked_files, dict) or set(locked_files) != required_locked_files:
        raise ValueError("Winner lock is missing locked selection files.")
    for label, item in locked_files.items():
        if not isinstance(item, dict):
            raise ValueError(f"Malformed winner-lock entry: {label}")
        locked_path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        expected_canonical = str(item.get("canonical_sha256", ""))
        if not locked_path.is_file() or file_sha256(locked_path) != expected:
            raise ValueError(f"Winner-lock selection file hash mismatch: {label}")
        if not expected_canonical or canonical_file_sha256(locked_path) != expected_canonical:
            raise ValueError(f"Winner-lock selection canonical hash mismatch: {label}")
    selection_entry = locked_files.get("selected_candidate")
    if not isinstance(selection_entry, dict):
        raise ValueError("Winner lock does not include selected_candidate.json.")
    selection = json.loads(Path(str(selection_entry["path"])).read_text(encoding="utf-8"))
    if (
        selection.get("selected_candidate") != payload.get("selected_candidate")
        or selection.get("selection_status") != payload.get("selection_status")
    ):
        raise ValueError("Winner-lock candidate/status differs from selected_candidate.json.")
    canonical_inputs = payload.get("canonical_inputs")
    required_canonical_inputs = {
        "fresh_validation_predictions_sha256",
        "validation_metric_rows_sha256",
        "validation_candidate_summary_sha256",
        "selection_sha256",
    }
    if (
        not isinstance(canonical_inputs, dict)
        or set(canonical_inputs) != required_canonical_inputs
        or any(not str(value) for value in canonical_inputs.values())
    ):
        raise ValueError("Winner lock lacks complete canonical validation-input hashes.")
    if canonical_inputs["selection_sha256"] != canonical_sha256(selection):
        raise ValueError("Winner-lock selection canonical content differs from selected_candidate.json.")
    validate_screen_evidence(payload.get("screen_evidence"))
    if (
        canonical_inputs["validation_metric_rows_sha256"]
        != str(locked_files["screen_seed_metrics"].get("canonical_sha256", ""))
        or canonical_inputs["validation_candidate_summary_sha256"]
        != str(
            locked_files["validation_candidate_summary"].get("canonical_sha256", "")
        )
    ):
        raise ValueError("Winner-lock derived CSV canonical hashes do not match fresh inputs.")
    if (
        canonical_inputs["fresh_validation_predictions_sha256"]
        != payload["screen_evidence"].get("fresh_validation_predictions_sha256")
    ):
        raise ValueError("Winner-lock fresh prediction hash differs from screen artifacts.")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_file_sha256(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return canonical_sha256(json.loads(path.read_text(encoding="utf-8-sig")))
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return canonical_tabular_sha256(list(csv.DictReader(handle)))
    raise ValueError(f"Unsupported canonical lock file type: {path}")


def selection_canonical_inputs(
    *,
    runs: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    validation_summary: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, str]:
    fresh_predictions = canonical_fresh_predictions(runs)
    return {
        "fresh_validation_predictions_sha256": canonical_sha256(fresh_predictions),
        "validation_metric_rows_sha256": canonical_tabular_sha256(validation_rows),
        "validation_candidate_summary_sha256": canonical_tabular_sha256(
            validation_summary
        ),
        "selection_sha256": canonical_sha256(selection),
    }


def canonical_tabular_sha256(rows: list[dict[str, Any]]) -> str:
    normalized = [
        {
            str(key): "" if value is None else str(value)
            for key, value in row.items()
        }
        for row in rows
    ]
    return canonical_sha256(normalized)


def canonical_fresh_predictions(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": str(run["candidate"]),
            "phase": str(run["phase"]),
            "seed": int(run["seed"]),
            "row": dict(row),
        }
        for run in sorted(
            runs,
            key=lambda item: (
                str(item["candidate"]),
                str(item["phase"]),
                int(item["seed"]),
            ),
        )
        for row in run["predictions"]
    ]


def build_screen_evidence(
    runs: list[dict[str, Any]],
    *,
    split_summary_path: Path,
) -> dict[str, Any]:
    if not split_summary_path.is_file():
        raise ValueError(f"Screen split summary is missing: {split_summary_path}")
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    if (
        split_summary.get("schema") != "v1_2_43_g_split_v1"
        or split_summary.get("built_split") != PHASE_SPLITS["screen"]
        or split_summary.get("outer_test_assignments_queried") is not False
    ):
        raise ValueError("Screen split summary does not attest the locked no-test screen split.")
    artifacts = []
    for run in sorted(
        runs,
        key=lambda item: (str(item["candidate"]), int(item["seed"])),
    ):
        manifest_path = Path(str(run["manifest_path"]))
        predictions_path = Path(str(run["predictions_path"]))
        artifacts.append(
            {
                "candidate": str(run["candidate"]),
                "phase": str(run["phase"]),
                "seed": int(run["seed"]),
                "manifest": {
                    "path": str(manifest_path.resolve()),
                    "sha256": file_sha256(manifest_path),
                    "canonical_sha256": canonical_file_sha256(manifest_path),
                },
                "predictions": {
                    "path": str(predictions_path.resolve()),
                    "sha256": file_sha256(predictions_path),
                    "canonical_sha256": canonical_file_sha256(predictions_path),
                },
            }
        )
    return {
        "screen_split_summary": {
            "path": str(split_summary_path.resolve()),
            "sha256": file_sha256(split_summary_path),
            "canonical_sha256": canonical_file_sha256(split_summary_path),
        },
        "screen_run_artifacts": artifacts,
        "screen_run_artifacts_canonical_sha256": canonical_sha256(artifacts),
        "fresh_validation_predictions_sha256": canonical_sha256(
            canonical_fresh_predictions(runs)
        ),
    }


def validate_screen_evidence(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Winner lock lacks screen split/run artifact evidence.")
    split_entry = value.get("screen_split_summary")
    artifacts = value.get("screen_run_artifacts")
    expected_artifacts_hash = str(value.get("screen_run_artifacts_canonical_sha256", ""))
    fresh_predictions_hash = str(value.get("fresh_validation_predictions_sha256", ""))
    if (
        not isinstance(split_entry, dict)
        or not isinstance(artifacts, list)
        or not artifacts
        or not fresh_predictions_hash
    ):
        raise ValueError("Winner lock has incomplete screen split/run artifact evidence.")
    _validate_evidence_file(split_entry, label="screen_split_summary")
    if canonical_sha256(artifacts) != expected_artifacts_hash:
        raise ValueError("Winner-lock screen run-artifact canonical hash mismatch.")
    seen: set[tuple[str, int]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("phase") != "screen":
            raise ValueError("Winner lock contains malformed screen run-artifact evidence.")
        key = (str(artifact.get("candidate", "")), int(artifact.get("seed", -1)))
        if key in seen:
            raise ValueError(f"Winner lock contains duplicate screen artifact: {key}")
        seen.add(key)
        _validate_evidence_file(artifact.get("manifest"), label=f"manifest:{key}")
        _validate_evidence_file(artifact.get("predictions"), label=f"predictions:{key}")


def _validate_evidence_file(value: Any, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Winner lock has malformed evidence file entry: {label}")
    path = Path(str(value.get("path", "")))
    expected = str(value.get("sha256", ""))
    expected_canonical = str(value.get("canonical_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise ValueError(f"Winner-lock evidence file hash mismatch: {label}")
    if not expected_canonical or canonical_file_sha256(path) != expected_canonical:
        raise ValueError(f"Winner-lock evidence canonical hash mismatch: {label}")


def discover_runs(root: Path, *, phases: set[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest_path in root.glob("**/manifest.json"):
        match = RUN_PATTERN.search("/".join(manifest_path.parts))
        predictions_path = manifest_path.parent / "predictions.csv"
        if match is None or not predictions_path.is_file():
            continue
        candidate, phase, seed_text = match.groups()
        if phase not in phases:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        path_seed = int(seed_text)
        manifest_seed = manifest.get("seed")
        if manifest_seed is None or int(manifest_seed) != path_seed:
            raise ValueError(
                "G-series manifest seed does not match the directory identity: "
                f"path_seed={path_seed} manifest_seed={manifest_seed} path={manifest_path}"
            )
        expected_split = PHASE_SPLITS[phase]
        if manifest.get("split_name") != expected_split:
            raise ValueError(
                f"G-series manifest split mismatch for phase={phase}: "
                f"expected={expected_split} actual={manifest.get('split_name')}"
            )
        manifest_source = manifest.get("data_source", {}).get("source_table")
        if manifest_source != SOURCE_TABLE:
            raise ValueError(
                f"G-series manifest source-table mismatch: {manifest_source!r}"
            )
        expected_prediction_parts = (
            ["finetune_mgkg_validation"]
            if phase == "screen"
            else ["finetune_mgkg_validation", "test"]
        )
        if sorted(manifest.get("prediction_output_split_parts", [])) != sorted(
            expected_prediction_parts
        ):
            raise ValueError(
                f"G-series manifest prediction boundary mismatch for phase={phase}."
            )
        with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        runs.append(
            {
                "candidate": candidate,
                "phase": phase,
                "seed": int(manifest_seed),
                "manifest": manifest,
                "predictions": predictions,
                "run_dir": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "predictions_path": str(predictions_path),
            }
        )
    unique: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in runs:
        key = (str(run["candidate"]), str(run["phase"]), int(run["seed"]))
        if key in unique:
            raise ValueError(f"Duplicate G-series run for candidate/phase/seed={key}")
        unique[key] = run
    return [unique[key] for key in sorted(unique)]


def summarize_runs(
    runs: list[dict[str, Any]],
    *,
    include_test: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, str], list[dict[str, Any]]],
]:
    rows: list[dict[str, Any]] = []
    prediction_sets: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        normalized = normalize_predictions(run["predictions"], scale="molar")
        wanted_parts = ("validation", "test") if include_test else ("validation",)
        for part in wanted_parts:
            part_rows = [row for row in normalized if row["evaluation_part"] == part]
            if not part_rows:
                continue
            native = metrics(part_rows, truth="y_native", prediction="pred_native")
            common = metrics(part_rows, truth="y_mgkg", prediction="pred_mgkg")
            manifest = run["manifest"]
            stage3 = manifest.get("finetune_mgkg", {})
            rows.append(
                {
                    "candidate": run["candidate"],
                    "phase": run["phase"],
                    "seed": run["seed"],
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
                    "stage3_freeze": stage3.get("freeze"),
                    "stage3_mse_weight": stage3.get("regression_loss", {}).get("mse_weight"),
                    "run_dir": run["run_dir"],
                }
            )
            prediction_sets[
                (str(run["candidate"]), str(run["phase"]), int(run["seed"]), part)
            ] = part_rows
    rows.sort(key=lambda row: (row["phase"], row["candidate"], row["seed"], row["evaluation_part"]))
    return rows, prediction_sets


def assert_screen_validation_identities(
    rows: list[dict[str, Any]], *, screen_seeds: list[int]
) -> None:
    screen = [
        row
        for row in rows
        if row["phase"] == "screen"
        and row["evaluation_part"] == "validation"
        and int(row["seed"]) in set(screen_seeds)
    ]
    by_key = {(str(row["candidate"]), int(row["seed"])): row for row in screen}
    missing_g0 = [seed for seed in screen_seeds if ("G0", seed) not in by_key]
    if missing_g0:
        raise ValueError(f"Missing G0 screen validation runs for seeds: {missing_g0}")
    reference = by_key[("G0", screen_seeds[0])]
    identity_fields = ("n", "aggregate_id_sha256", "result_ids_sha256")
    for key, row in by_key.items():
        mismatches = {
            field: {"actual": row[field], "reference": reference[field]}
            for field in identity_fields
            if row[field] != reference[field]
        }
        if mismatches:
            raise ValueError(
                f"Fresh validation identity mismatch for {key}: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )


def build_validation_summary(
    rows: list[dict[str, Any]], *, screen_seeds: list[int]
) -> list[dict[str, Any]]:
    wanted = set(screen_seeds)
    validation = [
        row
        for row in rows
        if row["phase"] == "screen"
        and row["evaluation_part"] == "validation"
        and int(row["seed"]) in wanted
    ]
    by_candidate: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in validation:
        by_candidate[str(row["candidate"])][int(row["seed"])] = row
    baseline = by_candidate.get("G0", {})
    output: list[dict[str, Any]] = []
    for candidate, seed_rows in sorted(by_candidate.items()):
        paired = [
            (seed_rows[seed], baseline[seed])
            for seed in screen_seeds
            if seed in seed_rows and seed in baseline
        ]
        r2_values = [float(seed_rows[seed]["native_r2"]) for seed in sorted(seed_rows)]
        mae_values = [float(seed_rows[seed]["native_mae"]) for seed in sorted(seed_rows)]
        deltas_r2 = [float(row["native_r2"]) - float(g0["native_r2"]) for row, g0 in paired]
        deltas_mae = [float(row["native_mae"]) - float(g0["native_mae"]) for row, g0 in paired]
        improving = [delta_r2 > 0 and delta_mae < 0 for delta_r2, delta_mae in zip(deltas_r2, deltas_mae)]
        output.append(
            {
                "candidate": candidate,
                "complete_screen": len(paired) == len(screen_seeds),
                "seed_count": len(seed_rows),
                "seeds": ",".join(str(seed) for seed in sorted(seed_rows)),
                "validation_native_r2_mean": statistics.fmean(r2_values),
                "validation_native_r2_sd": statistics.stdev(r2_values) if len(r2_values) > 1 else 0.0,
                "validation_native_mae_mean": statistics.fmean(mae_values),
                "validation_native_mae_sd": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0,
                "delta_r2_vs_g0_mean": statistics.fmean(deltas_r2) if deltas_r2 else None,
                "delta_mae_vs_g0_mean": statistics.fmean(deltas_mae) if deltas_mae else None,
                "paired_seeds_improving_both": sum(improving),
                "all_screen_seeds_improve_both": len(improving) == len(screen_seeds) and all(improving),
            }
        )
    return output


def assert_final_boundaries(
    rows: list[dict[str, Any]],
    *,
    selected_candidate: str,
    screen_seeds: list[int],
    final_seeds: list[int],
) -> None:
    by_key = {
        (
            str(row["candidate"]),
            str(row["phase"]),
            int(row["seed"]),
            str(row["evaluation_part"]),
        ): row
        for row in rows
    }
    screen_reference = by_key.get(("G0", "screen", screen_seeds[0], "validation"))
    if screen_reference is None:
        raise ValueError("Missing screen validation reference for final-boundary audit.")
    identity_fields = ("n", "aggregate_id_sha256", "result_ids_sha256")
    test_reference: dict[str, Any] | None = None
    for candidate in ("G0", selected_candidate):
        for seed in final_seeds:
            validation = by_key.get((candidate, "final", seed, "validation"))
            test = by_key.get((candidate, "final", seed, "test"))
            if validation is None or test is None:
                raise ValueError(
                    f"Missing final validation/test metrics for candidate={candidate} seed={seed}"
                )
            if any(validation[field] != screen_reference[field] for field in identity_fields):
                raise ValueError(
                    f"Final validation boundary differs from locked screen validation: "
                    f"candidate={candidate} seed={seed}"
                )
            if test_reference is None:
                test_reference = test
            elif any(test[field] != test_reference[field] for field in identity_fields):
                raise ValueError(
                    f"Final outer-test boundary differs across candidate/seed: "
                    f"candidate={candidate} seed={seed}"
                )


def select_candidate(
    rows: list[dict[str, Any]], *, screen_seeds: list[int]
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["candidate"] != "G0"
        and bool(row["complete_screen"])
        and bool(row["all_screen_seeds_improve_both"])
        and int(row["paired_seeds_improving_both"]) == len(screen_seeds)
        and float(row["delta_r2_vs_g0_mean"]) >= MIN_MEAN_DELTA_R2
        and float(row["delta_mae_vs_g0_mean"]) <= MAX_MEAN_DELTA_MAE
    ]
    if not eligible:
        return {
            "selected_candidate": None,
            "selection_status": "no_challenger_passed_preregistered_validation_thresholds",
        }
    eligible.sort(
        key=lambda row: (
            float(row["validation_native_mae_mean"]),
            -float(row["validation_native_r2_mean"]),
            str(row["candidate"]),
        )
    )
    winner = eligible[0]
    return {
        "selected_candidate": winner["candidate"],
        "selection_status": "locked_from_fresh_validation",
        "selected_validation_native_r2_mean": winner["validation_native_r2_mean"],
        "selected_validation_native_mae_mean": winner["validation_native_mae_mean"],
        "selected_delta_r2_vs_g0_mean": winner["delta_r2_vs_g0_mean"],
        "selected_delta_mae_vs_g0_mean": winner["delta_mae_vs_g0_mean"],
    }


def build_final_single_model_summary(
    rows: list[dict[str, Any]],
    *,
    selected_candidate: str,
    final_seeds: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in ("G0", selected_candidate):
        candidate_rows = [
            row
            for row in rows
            if row["phase"] == "final"
            and row["candidate"] == candidate
            and row["evaluation_part"] == "test"
            and int(row["seed"]) in set(final_seeds)
        ]
        by_seed = {int(row["seed"]): row for row in candidate_rows}
        missing = [seed for seed in final_seeds if seed not in by_seed]
        if missing:
            raise ValueError(f"Missing final {candidate} test runs for seeds: {missing}")
        item: dict[str, Any] = {
            "candidate": candidate,
            "evaluation_part": "test",
            "aggregation": "single_model_seed_mean",
            "seed_count": len(final_seeds),
            "seeds": ",".join(str(seed) for seed in final_seeds),
        }
        for field in (
            "native_r2",
            "native_rmse",
            "native_mae",
            "common_mgkg_r2",
            "common_mgkg_rmse",
            "common_mgkg_mae",
        ):
            values = [float(by_seed[seed][field]) for seed in final_seeds]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def build_test_ensemble_summary(
    prediction_sets: dict[tuple[str, str, int, str], list[dict[str, Any]]],
    *,
    selected_candidate: str,
    final_seeds: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reference_identities: set[str] | None = None
    reference_truth: dict[str, tuple[float, float]] | None = None
    for candidate in ("G0", selected_candidate):
        sets = [prediction_sets.get((candidate, "final", seed, "test")) for seed in final_seeds]
        if any(rows is None for rows in sets):
            raise ValueError(f"Incomplete final prediction ensemble for {candidate}")
        indexed = [unique_by_aggregate(rows or []) for rows in sets]
        identities = set(indexed[0])
        if any(set(item) != identities for item in indexed[1:]):
            raise ValueError(f"Final test identities differ across seeds for {candidate}")
        candidate_truth = {
            identity: (
                float(indexed[0][identity]["y_native"]),
                float(indexed[0][identity]["y_mgkg"]),
            )
            for identity in identities
        }
        for seed_index in indexed[1:]:
            if any(
                abs(float(seed_index[identity]["y_native"]) - candidate_truth[identity][0]) > 1e-12
                or abs(float(seed_index[identity]["y_mgkg"]) - candidate_truth[identity][1]) > 1e-12
                for identity in identities
            ):
                raise ValueError(f"Final test truth differs across seeds for {candidate}")
        if reference_identities is None:
            reference_identities = identities
            reference_truth = candidate_truth
        elif identities != reference_identities or candidate_truth != reference_truth:
            raise ValueError("G0 and the locked winner do not share an identical final test boundary.")
        ensemble: list[dict[str, Any]] = []
        for identity in sorted(identities):
            source_rows = [item[identity] for item in indexed]
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
                "evaluation_part": "test",
                "aggregation": "prediction_ensemble",
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
