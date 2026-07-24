from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsar_tl.training.traditional_comparison import (
    EXPECTED_COUNTS,
    EXPECTED_TASKS,
    EXPECTED_TARGET,
    EXPECTED_TARGET_FAMILY,
    inverse_target_scale,
    regression_metrics,
    validate_snapshot,
    within_task_r2,
)
from scripts.summarize_v1_2_53_traditional_ml_fixed_boundary import (
    add_holm_significance,
    paired_species_endpoint_stratified_bootstrap,
    within_group_r2,
)


def test_regression_metrics_exact_prediction() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_within_task_r2_removes_task_level_offsets() -> None:
    frame = pd.DataFrame(
        {
            "model_head": ["A", "A", "B", "B"],
            "y_true": [0.0, 1.0, 10.0, 11.0],
            "y_pred": [0.0, 1.0, 10.0, 11.0],
        }
    )
    assert within_task_r2(frame) == pytest.approx(1.0)
    assert within_group_r2(frame, "model_head") == pytest.approx(1.0)


def test_inverse_target_scale_uses_task_specific_stats() -> None:
    values = np.asarray([0.0, 1.0], dtype=float)
    stats = {
        "A": {"mean": 2.0, "std": 3.0},
        "B": {"mean": 10.0, "std": 2.0},
    }
    result = inverse_target_scale(values, ["A", "B"], stats)
    assert result.tolist() == pytest.approx([2.0, 12.0])


def test_validate_snapshot_fails_closed_on_count_change() -> None:
    rows = sum(EXPECTED_COUNTS.values())
    split = (
        ["train"] * EXPECTED_COUNTS["train"]
        + ["validation"] * EXPECTED_COUNTS["validation"]
        + ["test"] * EXPECTED_COUNTS["test"]
    )
    tasks = [f"task_{idx % EXPECTED_TASKS}" for idx in range(rows)]
    frame = pd.DataFrame(
        {
            "stable_record_id": [f"id_{idx}" for idx in range(rows)],
            "aggregate_id": np.arange(rows),
            "analysis_split": split,
            "model_head": tasks,
            "target_name": EXPECTED_TARGET,
            "target_family": EXPECTED_TARGET_FAMILY,
            "target_scale_key": tasks,
            "smiles": "C",
            "y_true": np.linspace(0.0, 1.0, rows),
        }
    )
    audit = validate_snapshot(frame, expected_record_id_sha256=None)
    assert audit["split_counts"] == EXPECTED_COUNTS
    assert audit["tasks"] == EXPECTED_TASKS
    with pytest.raises(ValueError, match="row counts changed"):
        validate_snapshot(frame.iloc[:-1].copy(), expected_record_id_sha256=None)


def test_paired_species_endpoint_bootstrap_preserves_pairing() -> None:
    identities = [f"id_{idx}" for idx in range(8)]
    truth = np.linspace(0.0, 3.5, len(identities))
    common = {
        "stable_record_id": identities,
        "aggregate_id": np.arange(len(identities)),
        "model_head": ["A"] * 4 + ["B"] * 4,
        "latin_name": ["Species one"] * 4 + ["Species two"] * 4,
        "species_endpoint": ["Species one||A"] * 4 + ["Species two||B"] * 4,
        "y_true": truth,
    }
    deep = pd.DataFrame({**common, "y_pred": truth + 0.05})
    traditional = pd.DataFrame({**common, "y_pred": truth + 0.50})
    result, _, intervals = paired_species_endpoint_stratified_bootstrap(
        {"deep": deep, "traditional": traditional},
        contrasts=(
            (
                "deep_minus_traditional",
                "deep",
                "traditional",
                "matched_full",
                "architecture_and_training_system_advantage",
            ),
        ),
        replicates=1_000,
        seed=20260720,
    )
    row = result.iloc[0]
    assert row["delta_mae"] < 0.0
    assert row["delta_rmse"] < 0.0
    assert row["delta_r2"] > 0.0
    assert len(intervals) == 2
    adjusted = add_holm_significance(result)
    assert adjusted.iloc[0]["deep_better_significance_mae"] in {"*", "**", "***"}
