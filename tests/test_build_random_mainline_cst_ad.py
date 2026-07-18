from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.build_random_mainline_cst_ad import (
    assign_overall_tier,
    assign_species_tier,
    average_precision,
    auroc,
    chemical_tier,
    summarize_tiers,
)


def test_chemical_tier_uses_tanimoto_and_williams_fallback() -> None:
    assert chemical_tier(0.85, False, False, chemical_threshold=0.5, chemical_high_threshold=0.7) == "C0"
    assert chemical_tier(0.55, False, False, chemical_threshold=0.5, chemical_high_threshold=0.7) == "C1"
    assert chemical_tier(0.10, True, False, chemical_threshold=0.5, chemical_high_threshold=0.7) == "C1"
    assert chemical_tier(0.10, False, False, chemical_threshold=0.5, chemical_high_threshold=0.7) == "C2"
    assert chemical_tier(np.nan, True, True, chemical_threshold=0.5, chemical_high_threshold=0.7) == "C2"


def test_species_and_overall_tiers_separate_strong_moderate_and_low_coverage() -> None:
    assert assign_species_tier(True, False, False, False, False, 0.0, 0.8) == "S0"
    assert assign_species_tier(False, True, False, False, False, 0.0, 0.8) == "S1"
    assert assign_species_tier(False, False, True, False, False, 0.0, 0.8) == "S2"
    assert assign_species_tier(False, False, False, False, True, 0.4, 0.8) == "S3"
    assert assign_species_tier(False, False, False, False, False, 0.9, 0.8) == "S3"
    assert assign_species_tier(False, False, False, False, False, 0.2, 0.8) == "S4"

    assert assign_overall_tier("C0", "S0", "T0", variant="C+S+T") == (
        "AD-A",
        "chemical_species_task_strong",
    )
    assert assign_overall_tier("C1", "S2", "T2", variant="C+S+T") == ("AD-B", "moderate_coverage")
    assert assign_overall_tier("C2", "S2", "T2", variant="C+S+T") == ("AD-C", "chemical_low_similarity")
    assert assign_overall_tier("C2", "S4", "T3", variant="C+S+T") == (
        "AD-D",
        "chemical_low_similarity+species_low_coverage+task_unseen",
    )


def test_summarize_tiers_uses_split_policy_denominator() -> None:
    frame = pd.DataFrame(
        {
            "split_policy": ["random_8_2", "random_8_2", "random_5fold", "random_5fold", "random_5fold"],
            "cst_ad_tier": ["AD-A", "AD-B", "AD-A", "AD-A", "AD-B"],
            "y_true": [1.0, 2.0, 1.0, 2.0, 3.0],
            "y_pred": [1.1, 3.0, 1.1, 2.2, 2.5],
            "abs_error": [0.1, 1.0, 0.1, 0.2, 0.5],
            "cst_chemical_score": [1.0, 0.2, 1.0, 0.9, 0.3],
            "cst_species_score": [1.0, 0.2, 1.0, 1.0, 0.4],
            "task_head": ["ECx", "NOEC", "ECx", "ECx", "NOEC"],
        }
    )

    summary = summarize_tiers(frame)

    by_policy_tier = {
        (row["split_policy"], row["cst_ad_tier"]): row
        for row in summary.to_dict(orient="records")
    }
    assert by_policy_tier[("random_8_2", "AD-A")]["coverage_fraction"] == 0.5
    assert by_policy_tier[("random_5fold", "AD-A")]["coverage_fraction"] == 2 / 3
    assert by_policy_tier[("random_5fold", "AD-B")]["coverage_n"] == 1
    assert math.isclose(by_policy_tier[("random_8_2", "AD-B")]["mae"], 1.0)


def test_high_error_ranking_metrics_are_directionally_correct() -> None:
    truth = np.asarray([True, False, True, False])
    scores = np.asarray([3.0, 2.0, 1.0, 0.0])

    assert auroc(truth, scores) == 0.75
    assert math.isclose(average_precision(truth, scores), (1.0 + 2 / 3) / 2)
