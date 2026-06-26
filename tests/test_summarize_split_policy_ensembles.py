from __future__ import annotations

import csv
import sys
from pathlib import Path

from scripts.summarize_split_policy_ensembles import main


def test_builds_seed_ensemble_for_holdout_and_fold(tmp_path, monkeypatch) -> None:
    root = tmp_path / "experiments"
    out_dir = tmp_path / "summary"
    for seed, offset in [("42", 0.0), ("1042", 1.0)]:
        write_predictions(
            root
            / f"v1.2.15_transfer_f100_anchor_random8_2_seed{seed}_cebin"
            / "deep"
            / "full"
            / "M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100"
            / "predictions.csv",
            prediction_rows(offset=offset),
        )
        write_predictions(
            root
            / f"v1.2.15_transfer_f100_anchor_random5fold_fold1_seed{seed}_cebin"
            / "deep"
            / "full"
            / "M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold1_f100"
            / "predictions.csv",
            prediction_rows(offset=offset),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_split_policy_ensembles",
            "--root",
            str(root),
            "--out-dir",
            str(out_dir),
            "--seeds",
            "42",
            "1042",
        ],
    )
    main()

    combined = read_csv(out_dir / "split_policy_ensemble_combined_summary.csv")
    by_policy = {row["split_policy"]: row for row in combined}
    assert set(by_policy) == {"random_8_2", "random_5fold"}
    assert by_policy["random_8_2"]["n"] == "2"
    assert by_policy["random_8_2"]["seeds"] == "42;1042"
    assert by_policy["random_5fold"]["folds_combined"] == "fold1"


def prediction_rows(*, offset: float) -> list[dict[str, str]]:
    return [
        {
            "sample_id": "1",
            "aggregate_id": "1",
            "split_name": "split",
            "split_part": "test",
            "task_head": "ECx_Growth",
            "target_scale_key": "ECx_Growth|soil",
            "target_name": "ptox_mol_l",
            "target_family": "soil_pTox",
            "medium_domain": "soil",
            "species_number": "sp1",
            "latin_name": "Species one",
            "y_true": "2.0",
            "y_pred": str(1.0 + offset),
        },
        {
            "sample_id": "2",
            "aggregate_id": "2",
            "split_name": "split",
            "split_part": "test",
            "task_head": "NOEC_Growth",
            "target_scale_key": "NOEC_Growth|soil",
            "target_name": "ptox_mol_l",
            "target_family": "soil_pTox",
            "medium_domain": "soil",
            "species_number": "sp2",
            "latin_name": "Species two",
            "y_true": "3.0",
            "y_pred": str(3.0 + offset),
        },
    ]


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
