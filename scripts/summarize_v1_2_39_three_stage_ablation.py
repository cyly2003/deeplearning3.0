from __future__ import annotations

"""Summarize the prespecified v1.2.39 soil mg/kg random-8:2 ablations.

The summary is intentionally restricted to the final soil mg/kg test domain.
MS3 has the identical input contract as M1 (molecular inputs without context),
so it is surfaced as an alias instead of duplicating a stochastic training run.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VERSION = "v1.2.39"
RUN_NAME_PREFIX = "three_stage_protocolfix_"
RANDOM8_SPLIT = "M_v1_2_39_ptox_to_soil_mgkg_B_random_8_2"
VARIANTS = (
    {
        "label": "Full",
        "run_label": "full_no_adapter_random8_2",
        "ablation": "full",
        "input_contract": "Descriptors + Morgan fingerprint + full context; medium adapters disabled.",
        "reuses": "",
    },
    {
        "label": "M1_no_context",
        "run_label": "M1_no_context",
        "ablation": "no_context",
        "input_contract": "Descriptors + Morgan fingerprint only; all numeric, duration, taxonomic, and categorical context removed.",
        "reuses": "",
    },
    {
        "label": "M2_no_species_lifestage",
        "run_label": "M2_no_species_lifestage",
        "ablation": "no_species_lifestage",
        "input_contract": "Full model without taxonomic/species/life-stage categorical context.",
        "reuses": "",
    },
    {
        "label": "M3_no_toxicity_binning",
        "run_label": "M3_no_toxicity_binning",
        "ablation": "full",
        "input_contract": "Full input model without the auxiliary toxicity-bin classification loss.",
        "reuses": "",
    },
    {
        "label": "M4_no_source_weighting",
        "run_label": "M4_no_source_weighting",
        "ablation": "full",
        "input_contract": "Full input model with uniform aquatic pretraining weights.",
        "reuses": "",
    },
    {
        "label": "M6_no_molecular_residual",
        "run_label": "M6_no_molecular_residual",
        "ablation": "no_molecular_residual",
        "input_contract": "Full input model without the molecular residual branch.",
        "reuses": "",
    },
    {
        "label": "MS1_descriptors_with_context",
        "run_label": "MS1_descriptors_with_context",
        "ablation": "descriptors_with_context",
        "input_contract": "Molecular descriptors + full context; Morgan fingerprint removed.",
        "reuses": "",
    },
    {
        "label": "MS2_fingerprint_with_context",
        "run_label": "MS2_fingerprint_with_context",
        "ablation": "fingerprint_with_context",
        "input_contract": "Morgan fingerprint + full context; molecular descriptors removed.",
        "reuses": "",
    },
    {
        "label": "MS3_molecular_only_no_context",
        "run_label": "M1_no_context",
        "ablation": "no_context",
        "input_contract": "Descriptors + Morgan fingerprint only; identical model-input state to M1.",
        "reuses": "M1_no_context",
    },
    {
        "label": "MS4_no_molecular_input",
        "run_label": "MS4_no_molecular_input",
        "ablation": "no_molecular_input",
        "input_contract": "All descriptor and fingerprint values removed; full experimental context retained.",
        "reuses": "",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the v1.2.39 soil mg/kg random-8:2 ablation matrix.")
    parser.add_argument("--root", required=True, help="v1.2.39 experiment root directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV, JSON, and figure outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_variant(root, variant) for variant in VARIANTS]
    baseline = next((row for row in rows if row["label"] == "Full"), None)
    for row in rows:
        append_deltas(row, baseline)

    fields = [
        "label",
        "run_label",
        "ablation",
        "reuses",
        "status",
        "input_contract",
        "run_dir",
        "n",
        "r2",
        "rmse",
        "mae",
        "delta_r2_vs_full",
        "delta_rmse_vs_full",
        "delta_mae_vs_full",
    ]
    write_csv(out_dir / "soil_mgkg_random8_2_ablation_summary.csv", rows, fields)
    write_csv(out_dir / "soil_mgkg_random8_2_ablation_deltas.csv", rows, fields)
    (out_dir / "soil_mgkg_random8_2_ablation_contract.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "split_name": RANDOM8_SPLIT,
                "evaluation_contract": "soil test rows with target_name=neg_log10_mg_kg only",
                "adapter_contract": "medium adapters are disabled for every variant",
                "ms3_reuse": "MS3 is an alias of M1 because both retain descriptors and fingerprint while removing all context.",
                "variants": list(VARIANTS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_metric_deltas(rows, out_dir / "soil_mgkg_random8_2_ablation_delta_mae.png")

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


def summarize_variant(root: Path, variant: dict[str, str]) -> dict[str, Any]:
    run_name = f"{VERSION}_{RUN_NAME_PREFIX}{variant['run_label']}_seed42"
    run_dir = root / run_name / "deep" / variant["ablation"] / RANDOM8_SPLIT
    prediction_path = run_dir / "predictions.csv"
    row: dict[str, Any] = {
        "label": variant["label"],
        "run_label": variant["run_label"],
        "ablation": variant["ablation"],
        "reuses": variant["reuses"],
        "input_contract": variant["input_contract"],
        "run_dir": str(run_dir),
        "status": "reused_pending_source" if variant["reuses"] else "missing",
        "n": "",
        "r2": "",
        "rmse": "",
        "mae": "",
    }
    if not prediction_path.exists():
        return row

    values: list[tuple[float, float]] = []
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            if item.get("split_part") != "test" or item.get("target_name") != "neg_log10_mg_kg":
                continue
            try:
                y_true = float(item["y_true"])
                y_pred = float(item["y_pred"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(y_true) and math.isfinite(y_pred):
                values.append((y_true, y_pred))
    if not values:
        row["status"] = "no_eligible_soil_mgkg_test_rows"
        return row

    y_true = np.asarray([item[0] for item in values], dtype=float)
    y_pred = np.asarray([item[1] for item in values], dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    row.update(
        {
            "status": "reused" if variant["reuses"] else "complete",
            "n": int(y_true.size),
            "r2": "" if ss_tot <= 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
        }
    )
    return row


def append_deltas(row: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    for metric in ("r2", "rmse", "mae"):
        row[f"delta_{metric}_vs_full"] = ""
    if baseline is None:
        return
    for metric in ("r2", "rmse", "mae"):
        if not isinstance(row.get(metric), float) or not isinstance(baseline.get(metric), float):
            continue
        row[f"delta_{metric}_vs_full"] = float(row[metric] - baseline[metric])


def plot_metric_deltas(rows: list[dict[str, Any]], output_path: Path) -> None:
    eligible = [row for row in rows if row["label"] != "Full" and isinstance(row.get("delta_mae_vs_full"), float)]
    figure, axis = plt.subplots(figsize=(10, max(3.5, 0.48 * max(1, len(eligible)) + 1.2)), layout="constrained")
    if not eligible:
        axis.text(0.5, 0.5, "No completed ablation results yet", ha="center", va="center")
        axis.set_axis_off()
    else:
        labels = [str(row["label"]) for row in eligible]
        values = [float(row["delta_mae_vs_full"]) for row in eligible]
        colors = ["#C44E52" if value > 0 else "#4C9F70" for value in values]
        positions = np.arange(len(eligible))
        axis.barh(positions, values, color=colors)
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Delta MAE vs full model (soil mg/kg test)")
        axis.set_title("v1.2.39 random 8:2 ablation effects")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
