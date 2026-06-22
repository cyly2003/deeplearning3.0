from scripts.summarize_deep_runs import common_task_summary


def _prediction(task: str, y_true: float, y_pred: float) -> dict[str, str]:
    return {
        "task_head": task,
        "y_true": str(y_true),
        "y_pred": str(y_pred),
    }


def test_common_task_summary_uses_task_intersection_and_skips_smoke() -> None:
    rows = [
        {
            "run": "soil_f20_authority_bin_aux_lw005_augN001",
            "split_name": "SoilPtoxQC2_C_low_f20",
            "split_key": "f20",
            "prediction_split_part": "test",
            "rows": [_prediction("ECx_Growth", 1, 1), _prediction("NOEC_Growth", 2, 3)],
        },
        {
            "run": "transfer_f20_source_alpha1_authority_bin_aux_lw005",
            "split_name": "M_v2_aquatic_to_soil_ptox_adapt_C_f20",
            "split_key": "f20",
            "prediction_split_part": "test",
            "rows": [
                _prediction("ECx_Growth", 1, 2),
                _prediction("NOEC_Growth", 2, 2),
                _prediction("LOEC_Growth", 3, 9),
            ],
        },
        {
            "run": "smoke_soil_f100_authority_bin_aux_lw005",
            "split_name": "SoilPtoxQC2_C_low_f100",
            "split_key": "f100",
            "prediction_split_part": "test",
            "rows": [_prediction("ECx_Growth", 1, 1)],
        },
    ]

    summary = common_task_summary(rows)

    assert len(summary) == 2
    assert {row["comparison_group"] for row in summary} == {"f20"}
    assert {row["run"] for row in summary} == {
        "soil_f20_authority_bin_aux_lw005_augN001",
        "transfer_f20_source_alpha1_authority_bin_aux_lw005",
    }
    assert all(row["task_count"] == 2 for row in summary)
    assert all(row["tasks"] == "ECx_Growth;NOEC_Growth" for row in summary)
    assert all(not str(row["run"]).startswith("smoke_") for row in summary)


def test_common_task_summary_adds_fullc_vs_transfer_f100_group() -> None:
    rows = [
        {
            "run": "soil_fullC_authority_bin_aux_lw005_core",
            "split_name": "SoilPtoxQC2_C_chemical_holdout_8_2",
            "split_key": "fullC",
            "prediction_split_part": "test",
            "rows": [_prediction("ECx_Growth", 1, 1), _prediction("NOEC_Growth", 2, 2)],
        },
        {
            "run": "transfer_f100_source_alpha1_authority_bin_aux_lw005",
            "split_name": "M_v2_aquatic_to_soil_ptox_adapt_C_f100",
            "split_key": "f100",
            "prediction_split_part": "test",
            "rows": [
                _prediction("ECx_Growth", 1, 1),
                _prediction("NOEC_Growth", 2, 2),
                _prediction("LOEC_Growth", 3, 3),
            ],
        },
    ]

    summary = common_task_summary(rows)

    assert len(summary) == 2
    assert {row["comparison_group"] for row in summary} == {"fullC_vs_transfer_f100"}
    assert all(row["task_count"] == 2 for row in summary)
    assert all(row["tasks"] == "ECx_Growth;NOEC_Growth" for row in summary)
