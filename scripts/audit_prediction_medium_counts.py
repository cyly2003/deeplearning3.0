from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qsar_tl.data.medium import classify_exposure_medium


def main() -> None:
    args = build_parser().parse_args()
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir) if args.out_dir else predictions_path.parent / "medium_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(predictions_path)
    if frame.empty:
        raise ValueError(f"Empty predictions file: {predictions_path}")

    add_inferred_medium(frame)
    write_count(
        frame,
        ["split_part", "medium_domain"],
        out_dir / "counts_by_split_medium.csv",
    )
    write_count(
        frame,
        ["split_part", "task_head", "target_name", "medium_domain"],
        out_dir / "counts_by_split_task_target_medium.csv",
    )
    write_count(
        frame,
        ["split_part", "medium_domain", "primary_medium", "media_type", "target_basis"],
        out_dir / "counts_by_split_medium_raw_fields.csv",
    )
    write_count(
        frame,
        ["split_part", "medium_domain", "inferred_medium_domain", "medium_conflict_flag"],
        out_dir / "counts_by_recorded_vs_inferred_medium.csv",
    )
    mismatched = frame[
        frame["inferred_medium_domain"].notna()
        & frame["inferred_medium_domain"].ne("")
        & frame["medium_domain"].astype(str).ne(frame["inferred_medium_domain"].astype(str))
    ].copy()
    keep = [
        column
        for column in [
            "sample_id",
            "aggregate_id",
            "record_id",
            "split_part",
            "task_head",
            "target_name",
            "medium_domain",
            "inferred_medium_domain",
            "primary_medium",
            "media_type",
            "target_basis",
            "medium_conflict_flag",
            "medium_inference_reason",
            "cas_number",
            "chemical_name",
            "latin_name",
        ]
        if column in mismatched.columns
    ]
    mismatched[keep].to_csv(out_dir / "medium_mismatch_rows.csv", index=False, encoding="utf-8-sig")
    print(f"audit_dir={out_dir}")
    print(f"mismatched_rows={len(mismatched)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit medium-domain counts in a deep predictions.csv file.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv from a deep run.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run>/medium_audit.")
    return parser


def add_inferred_medium(frame: pd.DataFrame) -> None:
    inferred_domains: list[str] = []
    conflict_flags: list[int] = []
    reasons: list[str] = []
    for _, row in frame.iterrows():
        classification = classify_exposure_medium(
            organism_habitat=row.get("organism_habitat", None),
            media_type=row.get("media_type", None),
            target_basis=row.get("target_basis", None),
        )
        inferred_domains.append(classification.medium_domain)
        conflict_flags.append(int(classification.medium_conflict_flag))
        reasons.append(classification.medium_domain_reason)
    frame["inferred_medium_domain"] = inferred_domains
    frame["medium_conflict_flag"] = conflict_flags
    frame["medium_inference_reason"] = reasons


def write_count(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return
    counts = (
        frame.groupby(existing, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", *existing], ascending=[False, *([True] * len(existing))])
    )
    counts.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
