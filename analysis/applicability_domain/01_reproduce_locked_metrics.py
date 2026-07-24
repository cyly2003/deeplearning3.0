from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ad_common import (
    ANALYSIS_DIR,
    SEEDS,
    build_route_ensembles,
    regression_metrics,
    within_task_r2,
    write_csv,
)


EXPECTED_TEST = {
    "M10": {"n": 3042, "r2": 0.7154353368, "rmse": 0.7551770292, "mae": 0.5457618347},
    "M00": {"n": 3042, "r2": 0.6801113449, "rmse": 0.8006776498, "mae": 0.5792099757},
}
TOLERANCE = 1e-6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce locked M10/M00 row-wise ensembles.")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = build_route_ensembles()
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        selected = ensemble.loc[ensemble["analysis_split"].eq(split)]
        for route in ("M10", "M00"):
            metric = regression_metrics(selected, prediction=f"{route}_prediction")
            metric.update(
                {
                    "route": route,
                    "split": split,
                    "aggregation": "four_seed_rowwise_prediction_mean",
                    "seeds": ",".join(map(str, SEEDS)),
                    "prediction_sd_ddof": 1,
                    "within_task_r2": within_task_r2(
                        selected, prediction=f"{route}_prediction"
                    ),
                }
            )
            for name in ("r2", "rmse", "mae"):
                expected = EXPECTED_TEST.get(route, {}).get(name) if split == "test" else None
                metric[f"expected_{name}"] = expected
                metric[f"abs_diff_{name}"] = (
                    math.nan if expected is None else abs(float(metric[name]) - expected)
                )
            rows.append(metric)
    result = pd.DataFrame(rows)
    assert_locked_test_metrics(result)
    write_csv(args.output_dir / "locked_metric_reproduction.csv", result)
    ensemble.to_parquet(args.output_dir / "stage3_predictions_ensemble.parquet", index=False)
    report = render_report(result)
    (args.output_dir / "LOCKED_METRICS_REPRODUCTION.md").write_text(report, encoding="utf-8")
    print(args.output_dir / "LOCKED_METRICS_REPRODUCTION.md")


def assert_locked_test_metrics(result: pd.DataFrame) -> None:
    test = result.loc[result["split"].eq("test")].set_index("route")
    for route, expected in EXPECTED_TEST.items():
        if int(test.loc[route, "n"]) != expected["n"]:
            raise ValueError(f"{route} n mismatch: {test.loc[route, 'n']} != {expected['n']}")
        for metric in ("r2", "rmse", "mae"):
            diff = abs(float(test.loc[route, metric]) - expected[metric])
            if diff > TOLERANCE:
                raise ValueError(
                    f"{route} {metric} mismatch: observed={test.loc[route, metric]}, "
                    f"expected={expected[metric]}, abs_diff={diff}"
                )


def render_report(result: pd.DataFrame) -> str:
    lines = [
        "# Locked metric reproduction",
        "",
        "> Gate status: PASS. Metrics were recomputed after averaging the four predictions for each stable record; model-level metrics were not averaged.",
        "",
        "## Boundary and statistic",
        "",
        "- Target: native `neg_log10_mol_kg` (soil mol/kg log scale).",
        "- Seeds: 42, 2042, 3407 and 8417.",
        "- Outer test: fixed 3,042-record random boundary.",
        "- Prediction disagreement: sample standard deviation across four seeds (`ddof=1`), reported as a diagnostic rather than complete uncertainty.",
        "- Within-task R²: `1 - sum((y-yhat)^2) / sum((y-mean(y within evaluation task))^2)`.",
        "- Pass tolerance for locked outer-test R²/RMSE/MAE: absolute difference <= 1e-6.",
        "",
        "## Recomputed results",
        "",
        "| Split | Route | n | R² | RMSE | MAE | Within-task R² | Max locked abs diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.to_dict("records"):
        diffs = [row[f"abs_diff_{name}"] for name in ("r2", "rmse", "mae")]
        finite = [float(value) for value in diffs if pd.notna(value)]
        max_diff = max(finite) if finite else math.nan
        lines.append(
            f"| {row['split']} | {row['route']} | {int(row['n']):,} | "
            f"{row['r2']:.9f} | {row['rmse']:.9f} | {row['mae']:.9f} | "
            f"{row['within_task_r2']:.9f} | "
            f"{('NA' if not math.isfinite(max_diff) else f'{max_diff:.2e}')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The Phase-0 reproduction gate is satisfied for both M10 and M00. The analysis may proceed to chemical-space reconstruction and record-level C/B/T/L support features without retraining.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
