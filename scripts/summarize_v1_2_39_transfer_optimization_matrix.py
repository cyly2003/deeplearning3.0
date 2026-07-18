from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


VALIDATION_SPLIT = "finetune_mgkg_validation"
TEST_SPLIT = "test"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the pTox-to-mg/kg transfer matrix without using test data for selection."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def pooled_metrics(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    pairs = [
        (float(row["y_true"]), float(row["y_pred"]))
        for row in rows
        if row.get("y_true") not in (None, "") and row.get("y_pred") not in (None, "")
    ]
    if not pairs:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}
    observed = [pair[0] for pair in pairs]
    predicted = [pair[1] for pair in pairs]
    residuals = [estimate - truth for truth, estimate in pairs]
    mean_observed = sum(observed) / len(observed)
    ss_res = sum(value * value for value in residuals)
    ss_tot = sum((value - mean_observed) ** 2 for value in observed)
    return {
        "n": len(pairs),
        "r2": None if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / len(pairs)),
        "mae": sum(abs(value) for value in residuals) / len(pairs),
    }


def matrix_cell(run_dir: Path, manifest: dict[str, Any]) -> str:
    name = run_dir.parts[-4] if len(run_dir.parts) >= 4 else run_dir.name
    for label in ("T0", "T1", "T2", "T3", "T4", "T5"):
        if f"_{label}_" in name or name.startswith(f"v1.2.39_{label}_"):
            return label
    run_name = str(manifest.get("run_name", ""))
    for label in ("T0", "T1", "T2", "T3", "T4", "T5"):
        if f"_{label}_" in run_name:
            return label
    return ""


def read_run(run_dir: Path, seed: int) -> dict[str, Any] | None:
    predictions_path = run_dir / "predictions.csv"
    manifest_path = run_dir / "manifest.json"
    if not predictions_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", seed)) != seed and f"seed{seed}" not in str(run_dir):
        return None
    cell = matrix_cell(run_dir, manifest)
    if not cell:
        return None
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    validation = pooled_metrics(
        [row for row in predictions if row.get("split_part") == VALIDATION_SPLIT]
    )
    test = pooled_metrics([row for row in predictions if row.get("split_part") == TEST_SPLIT])
    protocol = manifest.get("finetune_mgkg", {}) or {}
    adapter = manifest.get("mgkg_residual_adapter", {}) or {}
    return {
        "matrix_cell": cell,
        "seed": seed,
        "selection_rank": "",
        "selection_metric": "internal_validation_r2",
        "selection_uses_test": False,
        "validation_n": validation["n"],
        "validation_r2": validation["r2"],
        "validation_rmse": validation["rmse"],
        "validation_mae": validation["mae"],
        "test_n_report_only": test["n"],
        "test_r2_report_only": test["r2"],
        "test_rmse_report_only": test["rmse"],
        "test_mae_report_only": test["mae"],
        "mgkg_epochs": protocol.get("epochs"),
        "head_only_epochs": protocol.get("head_only_epochs"),
        "freeze": protocol.get("freeze"),
        "batch_size": protocol.get("batch_size"),
        "head_learning_rate": protocol.get("head_learning_rate"),
        "trunk_learning_rate": protocol.get("trunk_learning_rate"),
        "replay_fraction": protocol.get("soil_ptox_replay_fraction_requested"),
        "residual_adapter": adapter.get("enabled", False),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    for manifest_path in args.root.glob("**/manifest.json"):
        row = read_run(manifest_path.parent, args.seed)
        if row is not None:
            rows.append(row)
    latest_by_cell: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest_by_cell[row["matrix_cell"]] = row
    rows = list(latest_by_cell.values())
    rows.sort(
        key=lambda row: (
            row["validation_r2"] is None,
            -(row["validation_r2"] if row["validation_r2"] is not None else -math.inf),
            row["validation_rmse"] if row["validation_rmse"] is not None else math.inf,
            row["matrix_cell"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["selection_rank"] = rank
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["matrix_cell", "seed", "selection_rank"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
