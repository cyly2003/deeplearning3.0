from __future__ import annotations

from pathlib import Path

import pytest

from qsar_tl.training.deep_experiment import (
    SwaConfig,
    build_regression_loss,
    resolve_stage1_monitor_split,
    should_update_swa,
)
from qsar_tl.training.deep_train import DeepTrainingConfig
from qsar_tl.training.train import build_parser


ROOT = Path(__file__).resolve().parents[1]


def test_g_series_stage3_cli_exposes_sampling_hierarchy_and_checkpoint_controls() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--finetune-mgkg-target-bin-sampling",
            "--finetune-mgkg-target-bins",
            "10",
            "--finetune-mgkg-sampling-min-weight",
            "0.5",
            "--finetune-mgkg-sampling-max-weight",
            "2.0",
            "--finetune-mgkg-hierarchical-head",
            "--finetune-mgkg-hierarchical-family-tau",
            "128",
            "--finetune-mgkg-hierarchical-task-tau",
            "64",
            "--finetune-mgkg-init-checkpoint",
            "stage2.pt",
            "--export-finetune-mgkg-init-checkpoint",
            "export.pt",
        ]
    )

    assert args.finetune_mgkg_target_bin_sampling is True
    assert args.finetune_mgkg_target_bins == 10
    assert args.finetune_mgkg_sampling_min_weight == 0.5
    assert args.finetune_mgkg_sampling_max_weight == 2.0
    assert args.finetune_mgkg_hierarchical_head is True
    assert args.finetune_mgkg_hierarchical_family_tau == 128
    assert args.finetune_mgkg_hierarchical_task_tau == 64
    assert args.finetune_mgkg_init_checkpoint == "stage2.pt"
    assert args.export_finetune_mgkg_init_checkpoint == "export.pt"


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
    assert '"--monitor-split", "internal_train_fraction"' in runner
    assert '"--validation-fraction", "0.1"' in runner
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
    assert "three_stage_protocolfix_routingfix_" in summary
    assert '"protocolfix_routingfix"' in runner
    assert "--audit-csv" in runner
    assert "routing_audit.csv" in runner
    assert '"--source-weight-cache-dir", "outputs\\cache\\source_weights"' in runner
    assert "Assert-RoutingAudit -TransferSplit $SplitName" in runner
    assert '"main" {' in runner and "Invoke-Splits" in runner
    assert '"ablations" {' in runner and "Invoke-Random8Split" in runner


def test_remote_v138_v139_launchers_preserve_paired_protocol() -> None:
    v138 = (ROOT / "scripts" / "run_v1_2_38_soil_mgkg_random_remote.sh").read_text(encoding="utf-8")
    v139 = (ROOT / "scripts" / "run_v1_2_39_ptox_to_soil_mgkg_3stage_remote.sh").read_text(encoding="utf-8")

    for runner in (v138, v139):
        assert "--batch-size 512" in runner
        assert "--no-medium-adapters" in runner
        assert "--no-censored-loss" in runner
        assert "--monitor-split internal_train_fraction" in runner
        assert "--validation-fraction 0.1" in runner
    assert "--source-weighting-method none" in v138
    assert "--source-weighting-method tanimoto_to_finetune" in v139
    assert "--source-weight-cache-dir" in v139
    assert "--finetune-mgkg-epochs" in v139
    assert "--audit-csv" in v139
    for runner in (v138, v139):
        assert 'MODEL_SEED="${MODEL_SEED_OVERRIDE:-42}"' in runner
        assert '--seed "$MODEL_SEED"' in runner
        assert "seed${MODEL_SEED}" in runner
    assert "build_three_stage_ptox_to_soil_mgkg_split.py" in v139
    assert "--seed 42" in v139
    assert 'FINETUNE_MGKG_FREEZE="${FINETUNE_MGKG_FREEZE_OVERRIDE:-none}"' in v139
    assert '--finetune-mgkg-freeze "$FINETUNE_MGKG_FREEZE"' in v139
    assert 'FREEZE_SUFFIX="_mgkg_${FINETUNE_MGKG_FREEZE}"' in v139


def test_stage_local_huber_mse_hybrid_matches_declared_weights() -> None:
    torch = pytest.importorskip("torch")
    predictions = torch.tensor([0.0, 2.0])
    targets = torch.tensor([0.0, 0.0])
    config = DeepTrainingConfig(huber_delta=1.0, mse_loss_weight=0.3)
    loss = build_regression_loss(config)(predictions, targets)
    huber = torch.nn.functional.huber_loss(predictions, targets, delta=1.0)
    mse = torch.nn.functional.mse_loss(predictions, targets)
    assert float(loss) == pytest.approx(0.7 * float(huber) + 0.3 * float(mse))


def test_stage3_swa_updates_only_in_requested_phase() -> None:
    config = SwaConfig(enabled=True, phase="finetune_mgkg", start_epoch=31)
    assert not should_update_swa(config, phase="pretrain", epoch=40)
    assert not should_update_swa(config, phase="finetune", epoch=40)
    assert not should_update_swa(config, phase="finetune_mgkg", epoch=30)
    assert sum(
        should_update_swa(config, phase="finetune_mgkg", epoch=epoch)
        for epoch in range(1, 41)
    ) == 10


def test_v141_runner_locks_stage3_protocol_and_validation_only_selection() -> None:
    runner = (
        ROOT / "scripts" / "run_v1_2_41_three_stage_optimization_matrix_remote.sh"
    ).read_text(encoding="utf-8")
    summary = (
        ROOT / "scripts" / "summarize_v1_2_41_three_stage_optimization.py"
    ).read_text(encoding="utf-8")

    assert "STAGE1_EPOCHS=30" in runner
    assert "STAGE2_EPOCHS=20" in runner
    assert "STAGE3_EPOCHS=40" in runner
    assert "--finetune-mgkg-trunk-learning-rate 0.0001" in runner
    assert "--no-finetune-mgkg-early-stopping" in runner
    assert "--swa-phase finetune_mgkg" in runner
    assert "--swa-start-epoch 31" in runner
    assert "--finetune-mgkg-mse-loss-weight 0.3" in runner
    assert "SCREEN_SEEDS=(3407 42)" in runner
    assert "FINAL_SEEDS=(42 2042 3407 8417)" in runner
    assert 'for candidate in "${CANDIDATES[@]}"' in runner
    assert '"selection_uses_test": False' in summary
    assert 'row["evaluation_part"] == "validation"' in summary
    assert "--selection-only" in runner
    assert "assert_validation_identities" in summary
    assert '"result_ids_sha256"' in summary
