from __future__ import annotations

from pathlib import Path

import pytest

from qsar_tl.training.deep_experiment import resolve_stage1_monitor_split


ROOT = Path(__file__).resolve().parents[1]


def test_staged_training_auto_monitor_isolated_from_finetune_rows() -> None:
    assert (
        resolve_stage1_monitor_split(
            "auto",
            staged_training=True,
        )
        == "internal_train_fraction"
    )


def test_staged_training_rejects_downstream_stage1_monitor() -> None:
    with pytest.raises(ValueError, match="cannot monitor downstream"):
        resolve_stage1_monitor_split(
            "finetune",
            staged_training=True,
        )


def test_nonstaged_training_keeps_existing_auto_monitor_contract() -> None:
    assert (
        resolve_stage1_monitor_split(
            "auto",
            staged_training=False,
        )
        == "auto"
    )


def test_v138_runner_disables_medium_adapters_and_uses_new_run_identity() -> None:
    runner = (ROOT / "scripts" / "run_v1_2_38_soil_mgkg_random_local.ps1").read_text(
        encoding="utf-8"
    )
    assert '"--no-medium-adapters"' in runner
    assert '"--no-censored-loss"' in runner
    assert '"--censored-loss",' not in runner
    assert "soil_mgkg_no_adapter_" in runner
    assert "no_censored" in runner


def test_v139_runner_uses_stage_local_validation_and_no_censored_ablation() -> None:
    runner = (
        ROOT / "scripts" / "run_v1_2_39_ptox_to_soil_mgkg_3stage_local.ps1"
    ).read_text(encoding="utf-8")
    summary = (
        ROOT / "scripts" / "summarize_v1_2_39_three_stage_ablation.py"
    ).read_text(encoding="utf-8")

    assert '"--monitor-split", "internal_train_fraction"' in runner
    assert '"--validation-fraction", "0.1"' in runner
    assert '"--finetune-validation-fraction", "0.2"' in runner
    assert '"--finetune-mgkg-validation-fraction", "0.2"' in runner
    assert '"--no-censored-loss"' in runner
    assert '"--censored-loss",' not in runner
    assert "M5_no_censored_loss" not in runner
    assert "M5_no_censored_loss" not in summary
    assert "three_stage_protocolfix_" in summary
    assert '"protocolfix"' in runner
