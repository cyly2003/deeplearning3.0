from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scripts.summarize_v1_2_40_paired_mass_molar import (
    load_runs,
    summarize_contrasts,
    summarize_runs,
)


def test_paired_summary_backconverts_molar_predictions_to_mass_scale(tmp_path: Path) -> None:
    cells = ("D_mass", "D_molar", "X0_mass", "X0_molar")
    for cell in cells:
        run_dir = tmp_path / f"v1.2.40_{cell}_seed42" / "deep" / "full" / "Split"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": 42,
                    "validation_seed": 17073 if cell.startswith("D_") else 42,
                    "finetune_validation_seed": 42,
                    "finetune_mgkg_validation_seed": 42,
                }
            ),
            encoding="utf-8",
        )
        rows = []
        for aggregate_id, split_part, y_mgkg, pred_mgkg, mw in (
            ("T1", "train" if cell.startswith("D_") else "finetune_mgkg", 1.0, 1.1, 100.0),
            ("V1", "validation" if cell.startswith("D_") else "finetune_mgkg_validation", 2.0, 2.2, 50.0),
            ("V2", "validation" if cell.startswith("D_") else "finetune_mgkg_validation", 2.5, 2.4, 80.0),
            ("E1", "test", 3.0, 2.8, 200.0),
            ("E2", "test", 4.0, 4.1, 400.0),
        ):
            offset = math.log10(1000.0 * mw)
            is_molar = cell.endswith("_molar")
            rows.append(
                {
                    "aggregate_id": aggregate_id,
                    "split_part": split_part,
                    "original_split_part": split_part,
                    "base_task_head": "ECx_Mortality",
                    "task_head": "ECx_Mortality",
                    "y_true": y_mgkg + offset if is_molar else y_mgkg,
                    "y_pred": pred_mgkg + offset if is_molar else pred_mgkg,
                    "molecular_weight_g_mol_used": mw,
                    "smiles": "CC",
                    "result_ids": json.dumps([aggregate_id]),
                }
            )
        with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    runs = load_runs(tmp_path)
    assert len(runs) == 4
    seed_rows, identity_rows, prediction_sets = summarize_runs(runs)
    assert len(seed_rows) == 8
    assert len(identity_rows) == 8
    contrasts = summarize_contrasts(prediction_sets)
    assert len(contrasts) == 4
    for row in contrasts:
        assert abs(float(row["molar_minus_mass_r2"])) < 1e-12
        assert abs(float(row["molar_minus_mass_rmse"])) < 1e-12
        assert abs(float(row["truth_roundtrip_max_abs_error"])) < 1e-12
