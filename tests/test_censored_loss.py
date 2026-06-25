from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qsar_tl.training.censored_loss import censored_audit_summary, censored_direction, censored_hinge_loss


def test_censored_hinge_respects_ptox_direction() -> None:
    predictions = torch.tensor([4.0, 2.0, 2.5], dtype=torch.float32)
    bounds = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)

    loss = censored_hinge_loss(predictions, bounds, ["right", "left", "right"])

    assert torch.isclose(loss, torch.tensor((1.0**2 + 1.0**2 + 0.0) / 3.0))


def test_censored_hinge_accepts_direction_ids() -> None:
    predictions = torch.tensor([4.0, 2.0], dtype=torch.float32)
    bounds = torch.tensor([3.0, 3.0], dtype=torch.float32)

    loss = censored_hinge_loss(predictions, bounds, torch.tensor([1, -1]))

    assert torch.isclose(loss, torch.tensor(1.0))


def test_censored_audit_summary_derives_task_family_from_endpoint() -> None:
    rows = [
        {
            "endpoint": "EC50",
            "effect": "MOR",
            "measurement": "",
            "conc1_mean_op": ">",
            "unit_family_v2": "water_mg_l",
            "medium_domain": "aquatic",
            "value_quality": "censored",
            "target_status": "excluded",
            "excluded_reason": "censored_toxicity_value",
        }
    ]

    summary = censored_audit_summary(rows)

    assert censored_direction(">") == "right"
    assert summary[0]["task_family"] == "ECx"
    assert summary[0]["task_head"] == "ECx_Mortality"
    assert summary[0]["operator"] == ">"
    assert summary[0]["n"] == 1
