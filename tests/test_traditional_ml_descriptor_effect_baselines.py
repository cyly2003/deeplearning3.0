from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scripts.run_traditional_ml_descriptor_effect_baselines import (
    DatasetBundle,
    SubtaskSpec,
    assign_outer_split,
    build_subtasks,
    effect_level_feature_frame,
    is_molecular_size_related_descriptor,
    normalize_model_name,
    select_qsar_descriptor_columns,
    select_subtask,
)


def test_effect_level_features_encode_missing_levels_without_context() -> None:
    features = effect_level_feature_frame(pd.Series([50.0, None, 10.0]))

    assert list(features.columns) == [
        "effect_level_x",
        "effect_level_x_fraction",
        "effect_level_x_log1p",
        "effect_level_x_present",
    ]
    assert features.loc[0, "effect_level_x_fraction"] == 0.5
    assert features.loc[1, "effect_level_x"] == 0.0
    assert features.loc[1, "effect_level_x_present"] == 0.0
    assert features.loc[2, "effect_level_x_present"] == 1.0


def test_select_subtask_filters_rows_without_adding_species_context_features() -> None:
    frame = pd.DataFrame(
        {
            "latin_name": ["A", "A", "B"],
            "task_head": ["ECx_Growth", "NOEC_Growth", "ECx_Growth"],
            "target_value_median": [1.0, 2.0, 3.0],
        }
    )
    features = pd.DataFrame(
        {
            "rdkit_MolWt": [100.0, 110.0, 120.0],
            "effect_level_x": [50.0, 0.0, 50.0],
            "effect_level_x_fraction": [0.5, 0.0, 0.5],
            "effect_level_x_log1p": [np.log1p(50.0), 0.0, np.log1p(50.0)],
            "effect_level_x_present": [1.0, 0.0, 1.0],
        }
    )
    bundle = DatasetBundle(
        frame=frame,
        features=features,
        feature_columns=list(features.columns),
        descriptor_cache_path=Path("descriptor_cache.csv.gz"),
        invalid_smiles_count=0,
    )

    selected_frame, selected_features = select_subtask(
        bundle,
        SubtaskSpec(domain="soil", scope="species_endpoint", name="A__ECx_Growth", latin_name="A", task_head="ECx_Growth"),
    )

    assert len(selected_frame) == 1
    assert selected_frame.iloc[0]["latin_name"] == "A"
    assert "latin_name" not in selected_features.columns
    assert "task_head" not in selected_features.columns


def test_build_subtasks_and_split_are_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "latin_name": ["Species A"] * 8 + ["Species B"] * 4,
            "task_head": ["ECx_Growth"] * 6 + ["NOEC_Growth"] * 6,
        }
    )
    skipped: list[dict[str, object]] = []

    subtasks = build_subtasks(
        frame,
        domain="soil",
        scopes=["endpoint", "species_endpoint"],
        min_total=3,
        max_endpoint_tasks=2,
        max_species=2,
        max_species_endpoint_tasks=3,
        skipped_rows=skipped,
    )
    split_a = assign_outer_split(12, seed=42, validation_fraction=0.25)
    split_b = assign_outer_split(12, seed=42, validation_fraction=0.25)

    assert {task.scope for task in subtasks} == {"endpoint", "species_endpoint"}
    assert np.array_equal(split_a, split_b)
    assert (split_a == "validation").sum() == 3


def test_qsar_descriptor_union_filters_to_literature_guided_columns() -> None:
    selected = select_qsar_descriptor_columns(
        [
            "rdkit_MolWt",
            "rdkit_MolLogP",
            "rdkit_TPSA",
            "rdkit_Chi0",
            "rdkit_SlogP_VSA1",
            "rdkit_Ipc",
            "unrelated_column",
        ]
    )

    assert selected == [
        "rdkit_MolWt",
        "rdkit_MolLogP",
        "rdkit_TPSA",
        "rdkit_Chi0",
        "rdkit_SlogP_VSA1",
    ]


def test_qsar_descriptor_sensitivity_drops_molecular_size_related_columns() -> None:
    selected = select_qsar_descriptor_columns(
        [
            "rdkit_MolWt",
            "rdkit_ExactMolWt",
            "rdkit_HeavyAtomCount",
            "rdkit_TPSA",
            "rdkit_LabuteASA",
            "rdkit_RingCount",
            "rdkit_BCUT2D_MWHI",
            "rdkit_Chi0",
            "rdkit_SlogP_VSA1",
            "rdkit_MolLogP",
            "rdkit_MaxPartialCharge",
        ],
        exclude_molecular_size_related=True,
    )

    assert selected == ["rdkit_MolLogP", "rdkit_MaxPartialCharge"]
    assert is_molecular_size_related_descriptor("MolWt")
    assert is_molecular_size_related_descriptor("SlogP_VSA1")
    assert not is_molecular_size_related_descriptor("MolLogP")


def test_requested_traditional_model_aliases_are_supported() -> None:
    assert normalize_model_name("XGBoost") == "xgboost"
    assert normalize_model_name("LIGHTGBM") == "lightgbm"
    assert normalize_model_name("RF") == "random_forest"
    assert normalize_model_name("KNN") == "knn"
    assert normalize_model_name("Pls") == "pls"


def test_include_species_keeps_low_sample_endpoint_for_audit() -> None:
    frame = pd.DataFrame(
        {
            "latin_name": ["Eisenia fetida"] * 10 + ["Daphnia magna"] * 60,
            "task_head": ["ECx_Growth"] * 10 + ["ECx_Mortality"] * 60,
        }
    )
    skipped: list[dict[str, object]] = []

    subtasks = build_subtasks(
        frame,
        domain="soil",
        scopes=["species_endpoint"],
        min_total=50,
        max_endpoint_tasks=2,
        max_species=2,
        max_species_endpoint_tasks=1,
        skipped_rows=skipped,
        include_species={"Eisenia fetida"},
    )

    assert any(task.latin_name == "Eisenia fetida" and task.task_head == "ECx_Growth" for task in subtasks)
