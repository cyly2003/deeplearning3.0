from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts.summarize_split_policy_pilot import main


def test_summarizes_random_holdout_and_random_fivefold(tmp_path, monkeypatch) -> None:
    root = tmp_path / "experiments"
    out_dir = tmp_path / "summary"
    write_predictions(
        root
        / "v1.2.15_transfer_f100_anchor_random8_2_seed42"
        / "deep"
        / "full"
        / "M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100"
        / "predictions.csv",
        [
            {"split_part": "test", "medium_domain": "soil", "task_head": "ECx_Growth", "y_true": "1.0", "y_pred": "1.5"},
            {"split_part": "test", "medium_domain": "soil", "task_head": "NOEC_Growth", "y_true": "2.0", "y_pred": "2.5"},
            {"split_part": "train", "medium_domain": "aquatic", "task_head": "ECx_Growth", "y_true": "3.0", "y_pred": "3.0"},
        ],
    )
    write_predictions(
        root
        / "v1.2.15_transfer_f100_anchor_random5fold_fold1_seed42"
        / "deep"
        / "full"
        / "M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold1_f100"
        / "predictions.csv",
        [
            {"split_part": "test", "medium_domain": "soil", "task_head": "ECx_Growth", "y_true": "1.0", "y_pred": "1.0"},
            {"split_part": "test", "medium_domain": "soil", "task_head": "LOEC_Reproduction", "y_true": "4.0", "y_pred": "3.0"},
        ],
    )

    monkeypatch.setattr(sys, "argv", ["summarize_split_policy_pilot", "--root", str(root), "--out-dir", str(out_dir)])
    main()

    combined = read_csv(out_dir / "split_policy_combined_summary.csv")
    by_policy = {row["split_policy"]: row for row in combined}
    assert set(by_policy) == {"random_8_2", "random_5fold"}
    assert by_policy["random_8_2"]["n"] == "2"
    assert by_policy["random_5fold"]["folds_combined"] == "fold1"


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split_part", "medium_domain", "task_head", "y_true", "y_pred"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
