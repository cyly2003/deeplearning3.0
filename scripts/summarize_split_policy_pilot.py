from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from scripts.summarize_deep_runs import (
    focus_prediction_rows,
    metrics_from_prediction_rows,
    task_family,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize same-strategy transfer runs across random 8:2 and random k-fold split policies."
    )
    parser.add_argument("--root", required=True, help="Experiment root containing */deep/*/*/predictions.csv.")
    parser.add_argument("--out-dir", required=True, help="Output directory for split-policy summary CSVs.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    combined_inputs: dict[str, list[dict[str, str]]] = defaultdict(list)
    family_inputs: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for pred_path in sorted(root.glob("*/deep/*/*/predictions.csv")):
        run_dir = pred_path.parent
        run_name = strip_version_prefix(run_dir.parents[2].name)
        split_name = run_dir.name
        policy = split_policy_label(split_name)
        fold = fold_label(split_name)
        if policy == "other":
            continue

        rows = read_prediction_rows(pred_path)
        focus = focus_prediction_rows(rows, split_part="test")
        if not focus:
            continue

        run_rows.append(
            {
                "run": run_name,
                "split_name": split_name,
                "split_policy": policy,
                "fold": fold,
                **metrics_from_prediction_rows(focus),
            }
        )
        combined_inputs[policy].extend(focus)
        for row in focus:
            family_inputs[(policy, task_family(row))].append(row)

    policy_rows = [
        {"split_policy": policy, "folds_combined": folds_combined(policy, run_rows), **metrics_from_prediction_rows(rows)}
        for policy, rows in sorted(combined_inputs.items())
        if rows
    ]
    family_rows = [
        {
            "split_policy": policy,
            "task_family": family,
            "folds_combined": folds_combined(policy, run_rows),
            **metrics_from_prediction_rows(rows),
        }
        for (policy, family), rows in sorted(family_inputs.items())
        if rows
    ]

    write_csv(
        out_dir / "split_policy_run_summary.csv",
        sorted(run_rows, key=lambda row: (str(row["split_policy"]), str(row["fold"]), str(row["run"]))),
        [
            "run",
            "split_name",
            "split_policy",
            "fold",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
        ],
    )
    write_csv(
        out_dir / "split_policy_combined_summary.csv",
        policy_rows,
        ["split_policy", "folds_combined", "n", "r2", "rmse", "mae", "huber_loss", "task_count", "tasks"],
    )
    write_csv(
        out_dir / "split_policy_family_summary.csv",
        family_rows,
        [
            "split_policy",
            "task_family",
            "folds_combined",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
        ],
    )


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_policy_label(split_name: str) -> str:
    if "adapt_B_random_8_2" in split_name or "B_random_8_2" in split_name:
        return "random_8_2"
    if "adapt_E_random_5fold" in split_name or "E_random_5fold" in split_name:
        return "random_5fold"
    if "adapt_G_scaffold_cluster_8_2" in split_name or "G_scaffold_cluster_8_2" in split_name:
        return "scaffold_cluster_8_2"
    if "adapt_H_scaffold_cluster_5fold" in split_name or "H_scaffold_cluster_5fold" in split_name:
        return "scaffold_cluster_5fold"
    return "other"


def fold_label(split_name: str) -> str:
    match = re.search(r"fold(\d+)", split_name)
    if match:
        return f"fold{match.group(1)}"
    return "holdout"


def folds_combined(policy: str, run_rows: list[dict[str, object]]) -> str:
    folds = sorted(
        {str(row["fold"]) for row in run_rows if str(row.get("split_policy", "")) == policy},
        key=lambda value: (value != "holdout", value),
    )
    return ";".join(folds)


def strip_version_prefix(value: str) -> str:
    parts = value.split("_", 1)
    if parts and parts[0].startswith("v") and len(parts) == 2:
        return parts[1]
    return value


if __name__ == "__main__":
    main()
