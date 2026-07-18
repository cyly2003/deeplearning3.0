from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CELLS = ("D_mass", "D_molar", "X0_mass", "X0_molar")
VALIDATION_PARTS = {"validation", "finetune_mgkg_validation"}
TEST_PART = "test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired soil mass/molar runs on both native and common "
            "-log10(mg/kg) scales."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root)
    if not runs:
        raise ValueError(f"No paired v1.2.40 runs found under {args.root}")
    seed_rows, identity_rows, prediction_sets = summarize_runs(runs)
    contrast_rows = summarize_contrasts(prediction_sets)
    multi_seed_rows = summarize_multi_seed(seed_rows)
    write_csv(args.output_dir / "paired_seed_metrics.csv", seed_rows)
    write_csv(args.output_dir / "paired_target_contrasts.csv", contrast_rows)
    write_csv(args.output_dir / "paired_identity_audit.csv", identity_rows)
    write_csv(args.output_dir / "paired_multi_seed_summary.csv", multi_seed_rows)
    summary = {
        "runs": len(runs),
        "cells": sorted({row["cell"] for row in seed_rows}),
        "seeds": sorted({int(row["seed"]) for row in seed_rows}),
        "selection_uses_test": False,
        "primary_scale": "neg_log10_mg_kg",
        "native_molar_scale": "neg_log10_mol_kg",
        "source_table": args.source_table,
        "db": str(args.db),
        "outputs": {
            "seed_metrics": str(args.output_dir / "paired_seed_metrics.csv"),
            "target_contrasts": str(args.output_dir / "paired_target_contrasts.csv"),
            "identity_audit": str(args.output_dir / "paired_identity_audit.csv"),
            "multi_seed_summary": str(args.output_dir / "paired_multi_seed_summary.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest_path in root.glob("**/manifest.json"):
        predictions_path = manifest_path.parent / "predictions.csv"
        if not predictions_path.is_file():
            continue
        cell = identify_cell(manifest_path)
        if cell is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed = int(manifest.get("seed", seed_from_path(manifest_path)))
        with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        runs.append(
            {
                "cell": cell,
                "seed": seed,
                "manifest": manifest,
                "predictions": predictions,
                "run_dir": str(manifest_path.parent),
            }
        )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        key = (str(run["cell"]), int(run["seed"]))
        if key in unique:
            raise ValueError(f"Duplicate paired run for cell/seed={key}")
        unique[key] = run
    return [unique[key] for key in sorted(unique)]


def summarize_runs(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int, str], list[dict[str, Any]]]]:
    seed_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        cell = str(run["cell"])
        seed = int(run["seed"])
        scale = "molar" if cell.endswith("_molar") else "mass"
        architecture = "direct" if cell.startswith("D_") else "transfer"
        normalized = normalize_predictions(run["predictions"], scale=scale)
        train_rows = [row for row in normalized if row["evaluation_part"] == "target_train"]
        mw_only_native = fit_mw_only(train_rows, target="y_native")
        mw_only_common = fit_mw_only(train_rows, target="y_mgkg")
        for evaluation_part in ("validation", "test"):
            rows = [row for row in normalized if row["evaluation_part"] == evaluation_part]
            if not rows:
                continue
            native = metrics(rows, truth="y_native", prediction="pred_native")
            common = metrics(rows, truth="y_mgkg", prediction="pred_mgkg")
            single = metrics(
                [row for row in rows if not row["multifragment"]],
                truth="y_mgkg",
                prediction="pred_mgkg",
            )
            multi = metrics(
                [row for row in rows if row["multifragment"]],
                truth="y_mgkg",
                prediction="pred_mgkg",
            )
            mw_native_metrics = evaluate_mw_only(rows, mw_only_native, truth="y_native")
            mw_common_metrics = evaluate_mw_only(rows, mw_only_common, truth="y_mgkg")
            seed_rows.append(
                {
                    "cell": cell,
                    "architecture": architecture,
                    "target_parameterization": scale,
                    "seed": seed,
                    "evaluation_part": evaluation_part,
                    "n": common["n"],
                    "native_r2": native["r2"],
                    "native_rmse": native["rmse"],
                    "native_mae": native["mae"],
                    "common_mgkg_r2": common["r2"],
                    "common_mgkg_rmse": common["rmse"],
                    "common_mgkg_mae": common["mae"],
                    "single_fragment_n": single["n"],
                    "single_fragment_mgkg_r2": single["r2"],
                    "single_fragment_mgkg_rmse": single["rmse"],
                    "single_fragment_mgkg_mae": single["mae"],
                    "multifragment_n": multi["n"],
                    "multifragment_mgkg_r2": multi["r2"],
                    "multifragment_mgkg_rmse": multi["rmse"],
                    "multifragment_mgkg_mae": multi["mae"],
                    "mw_only_native_r2": mw_native_metrics["r2"],
                    "mw_only_native_rmse": mw_native_metrics["rmse"],
                    "mw_only_native_mae": mw_native_metrics["mae"],
                    "mw_only_common_mgkg_r2": mw_common_metrics["r2"],
                    "mw_only_common_mgkg_rmse": mw_common_metrics["rmse"],
                    "mw_only_common_mgkg_mae": mw_common_metrics["mae"],
                    "error_log10_mw_correlation": pearson(
                        [row["pred_mgkg"] - row["y_mgkg"] for row in rows],
                        [row["log10_mw"] for row in rows],
                    ),
                    "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in rows),
                    "run_dir": run["run_dir"],
                }
            )
            prediction_sets[(cell, seed, evaluation_part)] = rows
            identity_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "evaluation_part": evaluation_part,
                    "n": len(rows),
                    "aggregate_id_sha256": stable_hash(row["aggregate_id"] for row in rows),
                    "result_ids_sha256": stable_hash(
                        result_id
                        for row in rows
                        for result_id in row["result_ids"]
                    ),
                    "max_roundtrip_abs_error": max(row["roundtrip_abs_error"] for row in rows),
                    "validation_seed": run["manifest"].get("validation_seed"),
                    "finetune_validation_seed": run["manifest"].get("finetune_validation_seed"),
                    "finetune_mgkg_validation_seed": run["manifest"].get("finetune_mgkg_validation_seed"),
                }
            )
    assert_paired_identities(identity_rows)
    return seed_rows, identity_rows, prediction_sets


def normalize_predictions(rows: list[dict[str, str]], *, scale: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        split_part = str(row.get("split_part", "")).strip().lower()
        original_part = str(row.get("original_split_part", "")).strip().lower()
        if split_part in VALIDATION_PARTS or split_part == "validation":
            evaluation_part = "validation"
        elif split_part == TEST_PART:
            evaluation_part = "test"
        elif split_part in {"train", "finetune_mgkg"}:
            evaluation_part = "target_train"
        else:
            continue
        # Transfer predictions contain stage-1 and stage-2 rows. Only the
        # final soil target has a molecular weight recorded by the paired table.
        mw = optional_positive(row.get("molecular_weight_g_mol_used"))
        if mw is None:
            continue
        y_native = optional_float(row.get("y_true"))
        pred_native = optional_float(row.get("y_pred"))
        if y_native is None or pred_native is None:
            continue
        offset = math.log10(1000.0 * mw)
        if scale == "molar":
            y_mgkg = y_native - offset
            pred_mgkg = pred_native - offset
            roundtrip = abs(y_native - (y_mgkg + offset))
        else:
            y_mgkg = y_native
            pred_mgkg = pred_native
            roundtrip = 0.0
        normalized.append(
            {
                "aggregate_id": str(row.get("aggregate_id", "")),
                "task_head": str(row.get("base_task_head") or row.get("task_head") or ""),
                "evaluation_part": evaluation_part,
                "original_split_part": original_part,
                "y_native": y_native,
                "pred_native": pred_native,
                "y_mgkg": y_mgkg,
                "pred_mgkg": pred_mgkg,
                "molecular_weight": mw,
                "log10_mw": math.log10(mw),
                "multifragment": "." in str(row.get("smiles", "") or ""),
                "result_ids": parse_result_ids(row.get("result_ids")),
                "roundtrip_abs_error": roundtrip,
            }
        )
    return normalized


def summarize_contrasts(
    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    architectures = (("direct", "D_mass", "D_molar"), ("transfer", "X0_mass", "X0_molar"))
    seeds = sorted({key[1] for key in prediction_sets})
    for architecture, mass_cell, molar_cell in architectures:
        for seed in seeds:
            for part in ("validation", "test"):
                mass = prediction_sets.get((mass_cell, seed, part))
                molar = prediction_sets.get((molar_cell, seed, part))
                if mass is None or molar is None:
                    continue
                mass_by_id = unique_by_aggregate(mass)
                molar_by_id = unique_by_aggregate(molar)
                if set(mass_by_id) != set(molar_by_id):
                    raise ValueError(
                        f"Mass/molar paired IDs differ for {architecture} seed={seed} part={part}"
                    )
                truth_error = max(
                    abs(mass_by_id[key]["y_mgkg"] - molar_by_id[key]["y_mgkg"])
                    for key in mass_by_id
                )
                if truth_error >= 1e-12:
                    raise ValueError(
                        f"Back-converted molar truth differs from mass truth: max_error={truth_error}"
                    )
                mass_metrics = metrics(list(mass_by_id.values()), truth="y_mgkg", prediction="pred_mgkg")
                molar_metrics = metrics(list(molar_by_id.values()), truth="y_mgkg", prediction="pred_mgkg")
                rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "evaluation_part": part,
                        "paired_n": len(mass_by_id),
                        "mass_common_mgkg_r2": mass_metrics["r2"],
                        "molar_common_mgkg_r2": molar_metrics["r2"],
                        "molar_minus_mass_r2": difference(molar_metrics["r2"], mass_metrics["r2"]),
                        "mass_common_mgkg_rmse": mass_metrics["rmse"],
                        "molar_common_mgkg_rmse": molar_metrics["rmse"],
                        "molar_minus_mass_rmse": difference(molar_metrics["rmse"], mass_metrics["rmse"]),
                        "mass_common_mgkg_mae": mass_metrics["mae"],
                        "molar_common_mgkg_mae": molar_metrics["mae"],
                        "molar_minus_mass_mae": difference(molar_metrics["mae"], mass_metrics["mae"]),
                        "truth_roundtrip_max_abs_error": truth_error,
                        "paired_aggregate_id_sha256": stable_hash(mass_by_id),
                    }
                )
    return rows


def summarize_multi_seed(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["cell"]), str(row["evaluation_part"]))].append(row)
    output: list[dict[str, Any]] = []
    fields = (
        "native_r2",
        "native_rmse",
        "native_mae",
        "common_mgkg_r2",
        "common_mgkg_rmse",
        "common_mgkg_mae",
        "single_fragment_mgkg_r2",
    )
    for (cell, part), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "cell": cell,
            "evaluation_part": part,
            "seed_count": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
            item[f"{field}_mean"] = statistics.fmean(values) if values else None
            item[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        output.append(item)
    return output


def fit_mw_only(
    rows: list[dict[str, Any]], *, target: str
) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None
    x = [row["log10_mw"] for row in rows]
    y = [row[target] for row in rows]
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        return (y_mean, 0.0)
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denominator
    return (y_mean - slope * x_mean, slope)


def evaluate_mw_only(
    rows: list[dict[str, Any]],
    model: tuple[float, float] | None,
    *,
    truth: str,
) -> dict[str, float | int | None]:
    if model is None:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}
    intercept, slope = model
    copies = [
        {**row, "mw_only_prediction": intercept + slope * row["log10_mw"]}
        for row in rows
    ]
    return metrics(copies, truth=truth, prediction="mw_only_prediction")


def metrics(
    rows: list[dict[str, Any]], *, truth: str, prediction: str
) -> dict[str, float | int | None]:
    pairs = [
        (float(row[truth]), float(row[prediction]))
        for row in rows
        if row.get(truth) not in (None, "") and row.get(prediction) not in (None, "")
    ]
    if not pairs:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}
    observed = [pair[0] for pair in pairs]
    residuals = [pair[1] - pair[0] for pair in pairs]
    mean_observed = statistics.fmean(observed)
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((value - mean_observed) ** 2 for value in observed)
    return {
        "n": len(pairs),
        "r2": None if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / len(pairs)),
        "mae": statistics.fmean(abs(value) for value in residuals),
    }


def assert_paired_identities(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["evaluation_part"]))].append(row)
    for key, group in grouped.items():
        if len(group) != len(CELLS):
            continue
        aggregate_hashes = {str(row["aggregate_id_sha256"]) for row in group}
        if len(aggregate_hashes) != 1:
            raise ValueError(f"Paired evaluation identities differ across cells for seed/part={key}")
        if max(float(row["max_roundtrip_abs_error"]) for row in group) >= 1e-12:
            raise ValueError(f"Roundtrip error exceeds tolerance for seed/part={key}")


def unique_by_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["aggregate_id"])
        if key in result:
            raise ValueError(f"Duplicate aggregate prediction: {key}")
        result[key] = row
    return result


def identify_cell(path: Path) -> str | None:
    text = "/".join(path.parts)
    for cell in CELLS:
        if f"_{cell}_seed" in text:
            return cell
    return None


def seed_from_path(path: Path) -> int:
    text = "/".join(path.parts)
    marker = "_seed"
    position = text.rfind(marker)
    if position < 0:
        raise ValueError(f"Cannot infer seed from path: {path}")
    digits = []
    for char in text[position + len(marker) :]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        raise ValueError(f"Cannot infer seed from path: {path}")
    return int("".join(digits))


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


def stable_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def optional_positive(value: Any) -> float | None:
    parsed = optional_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = math.sqrt(
        sum((xi - x_mean) ** 2 for xi in x)
        * sum((yi - y_mean) ** 2 for yi in y)
    )
    return None if denominator <= 0 else numerator / denominator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
