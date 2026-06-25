from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_ad_gate import gate_rows
from scripts.summarize_deep_runs import metrics_from_prediction_rows, task_family


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
        description="Average prediction rows across seed runs and summarize ensemble metrics."
    )
    parser.add_argument("--audit-root", required=True, help="Directory containing */ad_prediction_rows.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for ensemble summary CSV files")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Ensemble group in the form output_name=run_dir_a,run_dir_b,run_dir_c",
    )
    parser.add_argument("--split-part", default="test")
    parser.add_argument("--key-columns", default=",".join(DEFAULT_KEY_COLUMNS))
    parser.add_argument("--write-prediction-rows", action="store_true")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    out_dir = Path(args.out_dir)
    key_columns = tuple(column.strip() for column in str(args.key_columns).split(",") if column.strip())
    groups = parse_groups(args.group)
    result = summarize_seed_ensembles(
        audit_root=audit_root,
        groups=groups,
        split_part=str(args.split_part),
        out_dir=out_dir,
        key_columns=key_columns,
        write_prediction_rows=bool(args.write_prediction_rows),
    )
    print(
        {
            "groups": sorted(groups),
            "focus_summary": str(result.focus_summary),
            "family_summary": str(result.family_summary),
            "ad_gate_summary": str(result.ad_gate_summary),
        }
    )


class EnsembleSummaryPaths:
    def __init__(self, *, focus_summary: Path, family_summary: Path, ad_gate_summary: Path) -> None:
        self.focus_summary = focus_summary
        self.family_summary = family_summary
        self.ad_gate_summary = ad_gate_summary


def parse_groups(values: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for value in values:
        name, sep, members = value.partition("=")
        name = name.strip()
        runs = [member.strip() for member in members.split(",") if member.strip()]
        if not sep or not name or len(runs) < 2:
            raise ValueError(f"Invalid --group value: {value!r}")
        if name in groups:
            raise ValueError(f"Duplicate ensemble group: {name}")
        groups[name] = runs
    return groups


def summarize_seed_ensembles(
    *,
    audit_root: Path,
    groups: dict[str, list[str]],
    split_part: str,
    out_dir: Path,
    key_columns: tuple[str, ...] = DEFAULT_KEY_COLUMNS,
    write_prediction_rows: bool = False,
) -> EnsembleSummaryPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    ad_rows: list[dict[str, Any]] = []

    for group_name, run_names in groups.items():
        ensemble = build_ensemble_prediction_rows(
            audit_root=audit_root,
            run_names=run_names,
            split_part=split_part,
            key_columns=key_columns,
        )
        row_dicts = dataframe_to_rows(ensemble)
        focus_rows.append({"run": group_name, "split_part": split_part, "tier": "all_test", **metrics_from_prediction_rows(row_dicts)})
        for family, sub in ensemble.groupby("task_family", dropna=False):
            family_rows.append(
                {
                    "run": group_name,
                    "split_part": split_part,
                    "task_family": family if str(family) else task_family({}),
                    **metrics_from_prediction_rows(dataframe_to_rows(sub)),
                }
            )
        ad_rows.extend(gate_rows(group_name, row_dicts, split_part=split_part))
        if write_prediction_rows:
            ensemble.to_csv(out_dir / f"{group_name}_ad_prediction_rows.csv", index=False)

    focus_summary = out_dir / "seed_mean_ensemble_focus_summary.csv"
    family_summary = out_dir / "seed_mean_ensemble_family_summary.csv"
    ad_gate_summary = out_dir / "seed_mean_ensemble_ad_gate_summary.csv"
    write_csv(focus_summary, focus_rows)
    write_csv(family_summary, family_rows)
    write_csv(ad_gate_summary, ad_rows)
    return EnsembleSummaryPaths(
        focus_summary=focus_summary,
        family_summary=family_summary,
        ad_gate_summary=ad_gate_summary,
    )


def build_ensemble_prediction_rows(
    *,
    audit_root: Path,
    run_names: list[str],
    split_part: str,
    key_columns: tuple[str, ...] = DEFAULT_KEY_COLUMNS,
) -> pd.DataFrame:
    frames = [load_prediction_rows(audit_root / run / "ad_prediction_rows.csv", split_part=split_part) for run in run_names]
    if not frames:
        raise ValueError("At least one run is required")
    reference = frames[0].copy()
    missing_columns = [column for column in key_columns if column not in reference.columns]
    if missing_columns:
        raise ValueError(f"Missing key columns in first run: {missing_columns}")
    for index, frame in enumerate(frames[1:], start=2):
        frame_missing = [column for column in key_columns if column not in frame.columns]
        if frame_missing:
            raise ValueError(f"Missing key columns in run {index}: {frame_missing}")
        if not reference.loc[:, key_columns].equals(frame.loc[:, key_columns]):
            raise ValueError(f"Prediction rows are not aligned for run {index}: {run_names[index - 1]}")

    pred_columns = []
    for index, frame in enumerate(frames, start=1):
        column = f"y_pred_member_{index}"
        reference[column] = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy()
        pred_columns.append(column)
        if "y_pred_scaled" in frame.columns:
            scaled_column = f"y_pred_scaled_member_{index}"
            reference[scaled_column] = pd.to_numeric(frame["y_pred_scaled"], errors="coerce").to_numpy()

    reference["y_pred"] = reference[pred_columns].mean(axis=1)
    if all(f"y_pred_scaled_member_{index}" in reference.columns for index in range(1, len(frames) + 1)):
        reference["y_pred_scaled"] = reference[[f"y_pred_scaled_member_{index}" for index in range(1, len(frames) + 1)]].mean(axis=1)
    reference["residual"] = pd.to_numeric(reference["y_true"], errors="raise") - reference["y_pred"]
    reference["abs_error"] = reference["residual"].abs()
    reference["ensemble_members"] = ";".join(run_names)
    return reference


def load_prediction_rows(path: Path, *, split_part: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "split_part" not in frame.columns:
        raise ValueError(f"Missing split_part column: {path}")
    return frame[frame["split_part"].astype(str).eq(str(split_part))].reset_index(drop=True)


def dataframe_to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.astype(object).where(pd.notnull(frame), "").to_dict(orient="records")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["n"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
