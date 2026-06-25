from __future__ import annotations

import math

import pandas as pd

from scripts.build_cst_ad_matrix import build_cst_ad_frame, summarize_cst_ad_tiers


def test_build_cst_ad_frame_assigns_expected_tiers() -> None:
    frame = pd.DataFrame(
        {
            "split_part": ["test", "test", "test", "test"],
            "ad_max_tanimoto_to_train": [0.9, 0.9, 0.2, 0.2],
            "ad_max_taxon_similarity_to_train": [1.0, 1.0, 1.0, 0.2],
            "ad_species_task_family_seen_train": [True, False, False, False],
            "y_true": [1.0, 1.0, 1.0, 1.0],
            "y_pred": [1.1, 1.3, 1.5, 2.0],
            "abs_error": [0.1, 0.3, 0.5, 1.0],
            "y_pred_member_1": [1.0, 1.2, 1.4, 1.0],
            "y_pred_member_2": [1.2, 1.4, 1.6, 3.0],
        }
    )

    result = build_cst_ad_frame(
        frame,
        chemical_threshold=0.5,
        species_threshold=0.5,
        uncertainty_threshold=0.5,
    )

    assert list(result["cst_ad_tier"]) == ["AD-A", "AD-B", "AD-C", "AD-D"]
    assert result["cst_chemical_in_domain"].tolist() == [True, True, False, False]
    assert result["cst_species_in_domain"].tolist() == [True, True, True, False]
    assert result["cst_species_task_family_seen"].tolist() == [True, False, False, False]


def test_build_cst_ad_frame_supports_missing_ensemble_members() -> None:
    frame = pd.DataFrame(
        {
            "split_part": ["test"],
            "ad_max_tanimoto_to_train": [1.0],
            "ad_max_taxon_similarity_to_train": [1.0],
            "ad_species_task_family_seen_train": ["true"],
            "y_true": [2.0],
            "y_pred": [2.1],
            "abs_error": [0.1],
        }
    )

    result = build_cst_ad_frame(frame, chemical_threshold=0.5, species_threshold=0.5)

    assert result.loc[0, "cst_ad_tier"] == "AD-A"
    assert result.loc[0, "cst_uncertainty_tier"] == "not_available"
    assert math.isnan(result.loc[0, "cst_ensemble_sd"])


def test_summarize_cst_ad_tiers_reports_coverage_and_mae() -> None:
    frame = pd.DataFrame(
        {
            "split_part": ["test", "test"],
            "ad_max_tanimoto_to_train": [0.9, 0.2],
            "ad_max_taxon_similarity_to_train": [1.0, 0.2],
            "ad_species_task_family_seen_train": [True, False],
            "y_true": [1.0, 2.0],
            "y_pred": [1.2, 3.0],
            "abs_error": [0.2, 1.0],
            "task_head": ["ECx_Growth", "NOEC_Growth"],
        }
    )
    result = build_cst_ad_frame(frame, chemical_threshold=0.5, species_threshold=0.5)

    summary = summarize_cst_ad_tiers(result)

    by_tier = {row["cst_ad_tier"]: row for row in summary.to_dict(orient="records")}
    assert by_tier["AD-A"]["coverage_fraction"] == 0.5
    assert by_tier["AD-A"]["mae"] == 0.19999999999999996
    assert by_tier["AD-D"]["coverage_n"] == 1
