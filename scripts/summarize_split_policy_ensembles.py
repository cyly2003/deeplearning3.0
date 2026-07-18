from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.summarize_deep_runs import focus_prediction_rows, metrics_from_prediction_rows, task_family, write_csv


DEFAULT_KEY_COLUMNS = (
    "sample_id",
    "aggregate_id",
    "split_name",
    "split_part",
    "task_head",
    "target_scale_key",
    "target_name",
    "target_family",
    "medium_domain",
    "species_number",
    "latin_name",
    "y_true",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build seed-ensemble summaries for random 8:2 and random 5-fold split-policy runs."
    )
    parser.add_argument("--root", required=True, help="Experiment root containing */deep/*/*/predictions.csv.")
    parser.add_argument("--out-dir", required=True, help="Output directory for ensemble summary CSVs.")
    parser.add_argument("--seeds", nargs="+", required=True, help="Seed values to ensemble, e.g. 42 1042 2042.")
    parser.add_argument("--split-part", default="test")
    parser.add_argument("--key-columns", default=",".join(DEFAULT_KEY_COLUMNS))
    parser.add_argument("--write-prediction-rows", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(str(seed).strip() for seed in args.seeds if str(seed).strip())
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for an ensemble.")
    key_columns = tuple(column.strip() for column in str(args.key_columns).split(",") if column.strip())

    index = index_prediction_paths(root)
    fold_rows: list[dict[str, Any]] = []
    combined_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_inputs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for policy, fold in ensemble_groups(index):
        run_paths = []
        missing = []
        for seed in seeds:
            path = index.get((policy, fold, seed))
            if path is None:
                missing.append(seed)
            else:
                run_paths.append(path)
        if missing:
            raise FileNotFoundError(f"Missing {policy}/{fold} predictions for seed(s): {', '.join(missing)}")
        ensemble = build_ensemble_prediction_rows(
            run_paths=run_paths,
            split_part=str(args.split_part),
            key_columns=key_columns,
        )
        row_dicts = focus_prediction_rows(dataframe_to_rows(ensemble), split_part=str(args.split_part))
        if not row_dicts:
            continue
        fold_rows.append(
            {
                "split_policy": policy,
                "fold": fold,
                "seeds": ";".join(seeds),
                **metrics_from_prediction_rows(row_dicts),
            }
        )
        combined_inputs[policy].extend(row_dicts)
        for row in row_dicts:
            family_inputs[(policy, task_family(row))].append(row)
        if args.write_prediction_rows:
            safe_fold = fold.replace(";", "_")
            ensemble.to_csv(out_dir / f"split_policy_ensemble_{policy}_{safe_fold}_prediction_rows.csv", index=False)

    combined_rows = [
        {
            "split_policy": policy,
            "folds_combined": folds_combined(policy, fold_rows),
            "seeds": ";".join(seeds),
            **metrics_from_prediction_rows(rows),
        }
        for policy, rows in sorted(combined_inputs.items())
    ]
    family_rows = [
        {
            "split_policy": policy,
            "task_family": family,
            "folds_combined": folds_combined(policy, fold_rows),
            "seeds": ";".join(seeds),
            **metrics_from_prediction_rows(rows),
        }
        for (policy, family), rows in sorted(family_inputs.items())
    ]

    write_csv(
        out_dir / "split_policy_ensemble_fold_summary.csv",
        sorted(fold_rows, key=lambda row: (str(row["split_policy"]), str(row["fold"]))),
        [
            "split_policy",
            "fold",
            "seeds",
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
        out_dir / "split_policy_ensemble_combined_summary.csv",
        combined_rows,
        ["split_policy", "folds_combined", "seeds", "n", "r2", "rmse", "mae", "huber_loss", "task_count", "tasks"],
    )
    write_csv(
        out_dir / "split_policy_ensemble_family_summary.csv",
        family_rows,
        [
            "split_policy",
            "task_family",
            "folds_combined",
            "seeds",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
        ],
    )


def index_prediction_paths(root: Path) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    for path in sorted(root.glob("*/deep/*/*/predictions.csv")):
        run_name = strip_version_prefix(path.parent.parents[2].name)
        split_name = path.parent.name
        policy = split_policy_label(split_name)
        fold = fold_label(split_name)
        seed = seed_label(run_name)
        if policy == "other" or not seed:
            continue
        key = (policy, fold, seed)
        if key in index:
            raise ValueError(f"Duplicate prediction path for {key}: {index[key]} and {path}")
        index[key] = path
    return index


def ensemble_groups(index: dict[tuple[str, str, str], Path]) -> list[tuple[str, str]]:
    groups = sorted({(policy, fold) for policy, fold, _seed in index})
    return groups


def build_ensemble_prediction_rows(
    *,
    run_paths: list[Path],
    split_part: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    frames = [load_prediction_rows(path, split_part=split_part) for path in run_paths]
    if len(frames) < 2:
        raise ValueError("At least two prediction files are required.")
    reference = frames[0].copy()
    missing_columns = [column for column in key_columns if column not in reference.columns]
    if missing_columns:
        raise ValueError(f"Missing key columns in first run: {missing_columns}")
    for index, frame in enumerate(frames[1:], start=2):
        frame_missing = [column for column in key_columns if column not in frame.columns]
        if frame_missing:
            raise ValueError(f"Missing key columns in run {index}: {frame_missing}")
        if not reference.loc[:, key_columns].equals(frame.loc[:, key_columns]):
            raise ValueError(f"Prediction rows are not aligned for run {index}: {run_paths[index - 1]}")

    pred_columns = []
    for index, frame in enumerate(frames, start=1):
        column = f"y_pred_member_{index}"
        reference[column] = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy()
        pred_columns.append(column)
        if "y_pred_scaled" in frame.columns:
            scaled_column = f"y_pred_scaled_member_{index}"
            reference[scaled_column] = pd.to_numeric(frame["y_pred_scaled"], errors="coerce").to_numpy()

    reference["y_pred"] = reference[pred_columns].mean(axis=1)
    scaled_columns = [f"y_pred_scaled_member_{index}" for index in range(1, len(frames) + 1)]
    if all(column in reference.columns for column in scaled_columns):
        reference["y_pred_scaled"] = reference[scaled_columns].mean(axis=1)
    reference["residual"] = pd.to_numeric(reference["y_true"], errors="raise") - reference["y_pred"]
    reference["abs_error"] = reference["residual"].abs()
    reference["ensemble_members"] = ";".join(str(path.parent.parents[2].name) for path in run_paths)
    return reference


def load_prediction_rows(path: Path, *, split_part: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    if "split_part" not in frame.columns:
        raise ValueError(f"Missing split_part column: {path}")
    return frame[frame["split_part"].astype(str).eq(str(split_part))].reset_index(drop=True)


def dataframe_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notnull(frame), "").to_dict(orient="records")


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


def seed_label(run_name: str) -> str:
    match = re.search(r"seed(\d+)", run_name)
    return match.group(1) if match else ""


def folds_combined(policy: str, rows: list[dict[str, Any]]) -> str:
    folds = sorted(
        {str(row["fold"]) for row in rows if str(row.get("split_policy", "")) == policy},
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
