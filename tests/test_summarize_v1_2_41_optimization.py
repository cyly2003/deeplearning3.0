from __future__ import annotations

import pytest

from scripts.summarize_v1_2_41_three_stage_optimization import (
    assert_validation_identities,
    build_validation_screen,
    select_candidate,
)


def row(candidate: str, seed: int, *, r2: float, mae: float, identity: str = "same") -> dict:
    return {
        "candidate": candidate,
        "seed": seed,
        "evaluation_part": "validation",
        "n": 10,
        "native_r2": r2,
        "native_mae": mae,
        "aggregate_id_sha256": identity,
        "result_ids_sha256": identity,
    }


def test_selection_requires_complete_paired_baselines() -> None:
    rows = [
        row("S0", 42, r2=0.60, mae=0.60),
        row("S1", 42, r2=0.61, mae=0.59),
        row("S1", 3407, r2=0.62, mae=0.58),
    ]
    with pytest.raises(ValueError, match="Missing S0 validation baseline"):
        assert_validation_identities(rows, screen_seeds=[42, 3407])


def test_selection_fails_on_validation_identity_mismatch() -> None:
    rows = [
        row("S0", 42, r2=0.60, mae=0.60),
        row("S1", 42, r2=0.61, mae=0.59, identity="different"),
    ]
    with pytest.raises(ValueError, match="Validation identity mismatch"):
        assert_validation_identities(rows, screen_seeds=[42])


def test_validation_only_pareto_rule_selects_improving_candidate() -> None:
    rows = [
        row("S0", 42, r2=0.60, mae=0.60),
        row("S0", 3407, r2=0.62, mae=0.58),
        row("S1", 42, r2=0.63, mae=0.57),
        row("S1", 3407, r2=0.64, mae=0.56),
        row("S2", 42, r2=0.65, mae=0.61),
        row("S2", 3407, r2=0.66, mae=0.60),
    ]
    assert_validation_identities(rows, screen_seeds=[42, 3407])
    summary = build_validation_screen(rows, screen_seeds=[42, 3407])
    selection = select_candidate(summary, screen_seeds=[42, 3407])
    assert selection["selected_candidate"] == "S1"
