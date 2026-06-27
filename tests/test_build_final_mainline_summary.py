from __future__ import annotations

import pandas as pd

from scripts.build_final_mainline_summary import build_summary_tables


def test_build_summary_tables_outputs_main_and_full_task_scopes(tmp_path, monkeypatch):
    chemical_dir = tmp_path / "chemical"
    random_dir = tmp_path / "random"
    out_dir = tmp_path / "out"
    chemical_dir.mkdir()
    random_dir.mkdir()

    pd.DataFrame(
        [
            {
                "run": "anchor_tanimoto_a1_5seed_ensemble",
                "split_part": "test",
                "tier": "all_test",
                "n": 3,
                "r2": 0.3,
                "rmse": 1.2,
                "mae": 0.9,
                "huber_loss": 0.5,
                "task_count": 2,
                "tasks": "ECx_Growth;ICx_Growth",
            }
        ]
    ).to_csv(chemical_dir / "seed_mean_ensemble_focus_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run": "anchor_tanimoto_a1_5seed_ensemble",
                "split_part": "test",
                "task_family": "ECx",
                "n": 2,
                "r2": 0.4,
                "rmse": 1.0,
                "mae": 0.8,
                "huber_loss": 0.4,
                "task_count": 1,
                "tasks": "ECx_Growth",
            }
        ]
    ).to_csv(chemical_dir / "seed_mean_ensemble_family_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run": "anchor_tanimoto_a1_5seed_ensemble",
                "split_part": "test",
                "task_family": "ECx",
                "task_head": "ECx_Growth",
                "n": 2,
                "r2": 0.4,
                "rmse": 1.0,
                "mae": 0.8,
                "huber_loss": 0.4,
            },
            {
                "run": "anchor_tanimoto_a1_5seed_ensemble",
                "split_part": "test",
                "task_family": "ICx",
                "task_head": "ICx_Growth",
                "n": 1,
                "r2": 0.1,
                "rmse": 1.5,
                "mae": 1.1,
                "huber_loss": 0.7,
            },
        ]
    ).to_csv(chemical_dir / "seed_mean_ensemble_task_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "split_policy": "random_8_2",
                "folds_combined": "holdout",
                "seeds": "42;1042",
                "n": 4,
                "r2": 0.7,
                "rmse": 0.8,
                "mae": 0.6,
                "huber_loss": 0.3,
                "task_count": 1,
                "tasks": "ECx_Growth",
            }
        ]
    ).to_csv(random_dir / "split_policy_ensemble_combined_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "split_policy": "random_8_2",
                "task_family": "ECx",
                "folds_combined": "holdout",
                "seeds": "42;1042",
                "n": 4,
                "r2": 0.7,
                "rmse": 0.8,
                "mae": 0.6,
                "huber_loss": 0.3,
                "task_count": 1,
                "tasks": "ECx_Growth",
            }
        ]
    ).to_csv(random_dir / "split_policy_ensemble_family_summary.csv", index=False)
    task_row = {
        "split_policy": "random_8_2",
        "folds_combined": "holdout",
        "seeds": "42;1042",
        "task_family": "ECx",
        "task_head": "ECx_Growth",
        "n": 4,
        "r2": 0.7,
        "rmse": 0.8,
        "mae": 0.6,
        "huber_loss": 0.3,
    }
    pd.DataFrame([task_row]).to_csv(random_dir / "split_policy_ensemble_main_task_summary.csv", index=False)
    pd.DataFrame(
        [
            task_row,
            {
                **task_row,
                "task_family": "LDx",
                "task_head": "LDx_Mortality",
                "n": 5,
            },
        ]
    ).to_csv(random_dir / "split_policy_ensemble_task_summary.csv", index=False)

    monkeypatch.chdir(tmp_path)
    outputs = build_summary_tables(chemical_dir=chemical_dir, random_dir=random_dir, out_dir=out_dir)

    task_main = pd.read_csv(outputs["task_30task"])
    task_all = pd.read_csv(outputs["task_35task"])
    overall = pd.read_csv(outputs["overall"])

    assert set(task_main["task_family"]) == {"ECx"}
    assert set(task_all["task_family"]) == {"ECx", "ICx", "LDx"}
    assert set(overall["evaluation_policy"]) == {"chemical_holdout_f100_5seed", "random_8_2_2seed"}
    assert int(overall.loc[overall["evaluation_policy"].eq("random_8_2_2seed"), "ensemble_seed_count"].iloc[0]) == 2
