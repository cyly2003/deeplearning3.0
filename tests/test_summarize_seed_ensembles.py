from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_seed_ensembles import build_ensemble_prediction_rows, summarize_seed_ensembles


def test_seed_ensemble_averages_aligned_prediction_rows(tmp_path: Path) -> None:
    audit_root = tmp_path / "audits"
    write_run(
        audit_root,
        "seed1",
        [
            prediction_row(sample_id=1, task_head="ECx_Growth", y_true=1.0, y_pred=0.0),
            prediction_row(sample_id=2, task_head="NOEC_Growth", y_true=3.0, y_pred=4.0),
        ],
    )
    write_run(
        audit_root,
        "seed2",
        [
            prediction_row(sample_id=1, task_head="ECx_Growth", y_true=1.0, y_pred=2.0),
            prediction_row(sample_id=2, task_head="NOEC_Growth", y_true=3.0, y_pred=2.0),
        ],
    )

    out_dir = tmp_path / "summary"
    summarize_seed_ensembles(
        audit_root=audit_root,
        groups={"ensemble": ["seed1", "seed2"]},
        split_part="test",
        out_dir=out_dir,
        write_prediction_rows=True,
    )

    focus = pd.read_csv(out_dir / "seed_mean_ensemble_focus_summary.csv")
    ad_gate = pd.read_csv(out_dir / "seed_mean_ensemble_ad_gate_summary.csv")
    prediction_rows = pd.read_csv(out_dir / "ensemble_ad_prediction_rows.csv")

    assert focus.loc[0, "n"] == 2
    assert focus.loc[0, "mae"] == 0.0
    assert focus.loc[0, "r2"] == 1.0
    assert set(ad_gate["tier"]) >= {"all_test", "overall_in_domain", "species_task_family_seen_train"}
    assert prediction_rows["y_pred"].tolist() == [1.0, 3.0]
    assert prediction_rows["abs_error"].tolist() == [0.0, 0.0]


def test_seed_ensemble_rejects_unaligned_rows(tmp_path: Path) -> None:
    audit_root = tmp_path / "audits"
    write_run(audit_root, "seed1", [prediction_row(sample_id=1, task_head="ECx_Growth", y_true=1.0, y_pred=1.0)])
    write_run(audit_root, "seed2", [prediction_row(sample_id=2, task_head="ECx_Growth", y_true=1.0, y_pred=1.0)])

    with pytest.raises(ValueError, match="not aligned"):
        build_ensemble_prediction_rows(
            audit_root=audit_root,
            run_names=["seed1", "seed2"],
            split_part="test",
        )


def write_run(audit_root: Path, run_name: str, rows: list[dict[str, object]]) -> None:
    run_dir = audit_root / run_name
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "ad_prediction_rows.csv", index=False)


def prediction_row(*, sample_id: int, task_head: str, y_true: float, y_pred: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "aggregate_id": sample_id,
        "split_name": "M_v2_aquatic_to_soil_ptox_adapt_C_f100",
        "split_part": "test",
        "task_head": task_head,
        "task_family": task_head.split("_", 1)[0],
        "target_scale_key": f"{task_head}|aquatic_pTox_mol_L",
        "target_name": "ptox_mol_l",
        "target_family": "aquatic_pTox_mol_L",
        "medium_domain": "soil",
        "species_number": 100 + sample_id,
        "latin_name": f"Species {sample_id}",
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_scaled": y_true,
        "y_pred_scaled": y_pred,
        "residual": y_true - y_pred,
        "abs_error": abs(y_true - y_pred),
        "ad_overall_in_domain": True,
        "ad_ad_warning": "in_domain",
        "ad_species_task_family_seen_train": True,
        "ad_species_seen_train": True,
        "ad_family_seen_train": True,
    }
