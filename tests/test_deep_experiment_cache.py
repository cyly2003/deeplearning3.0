from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from qsar_tl.modeling.dataset import AggregatedTaskDataset
from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork, TOXICITY_BIN_LOGITS_KEY
from qsar_tl.features.descriptor_groups import resolve_descriptor_group_indices
from qsar_tl.features.molecular_graph import write_molecular_graph_cache
from qsar_tl.features.padel import write_padel_feature_cache_from_csv
from qsar_tl.training.deep_experiment import (
    ABLATION_SPECS,
    EffectLevelWeightingConfig,
    MOLECULAR_SIZE_RELATED_DESCRIPTOR_NAMES,
    RAW_NUMERIC_CLIP_ABS,
    MolecularFeatureBuilder,
    SourceWeightingConfig,
    active_categorical_columns,
    apply_effect_level_frequency_weights,
    apply_finetune_freeze,
    apply_source_similarity_weights,
    build_censored_training_samples,
    batch_weighted_loss,
    build_deep_samples,
    build_molecular_feature_cache,
    build_noisy_index_dataset,
    build_numeric_feature_names,
    build_preprocessing_manifest,
    coral_loss,
    descriptor_mask_indices,
    descriptor_proxy_indices,
    encode_category_id,
    fit_categorical_maps,
    fit_adapter_map,
    fit_numeric_stats,
    fit_target_scaler,
    fit_zscore_correction,
    load_molecular_feature_cache,
    load_molecular_feature_cache_descriptor_names,
    max_tanimoto_similarity,
    masked_descriptors,
    metrics_by_group,
    min_proxy_distance,
    normalize_clipped_weights,
    predict_all,
    raw_numeric_matrix,
    split_finetune_validation_indices,
    split_training_validation_indices,
    ZScoreCorrectionConfig,
    CensoredLossConfig,
)
from qsar_tl.training.deep_train import DeepTrainingConfig, dataloader_runtime_options
from qsar_tl.training.deep_train import collate_aggregated_task_batch
from qsar_tl.training.train import build_run_dir


def test_molecular_feature_cache_loads_jsonl(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.jsonl"
    payload = {
        "smiles": "CCO",
        "descriptors": [1.0, 2.0],
        "fingerprint": [1.0, 0.0, 1.0, 0.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    cache = load_molecular_feature_cache(cache_path, fingerprint_size=4)

    assert cache["CCO"] == ([1.0, 2.0], [1.0, 0.0, 1.0, 0.0])


def test_molecular_feature_builder_prefers_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.jsonl"
    payload = {
        "smiles": "CCO",
        "descriptors": [3.0, 4.0],
        "fingerprint": [0.0, 1.0, 0.0, 1.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)

    assert builder.source == "rdkit_cache"
    assert builder.encode("CCO") == ([3.0, 4.0], [0.0, 1.0, 0.0, 1.0])


def test_molecular_feature_builder_reads_descriptor_names_from_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "padel_features.jsonl"
    payload = {
        "smiles": "CCO",
        "feature_source": "padel_descriptor_morgan",
        "descriptor_names": ["MW", "TopoPSA", "ALogP"],
        "descriptors": [46.0, 20.2, -0.1],
        "fingerprint": [1.0, 0.0, 1.0, 0.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)

    assert builder.source == "padel_descriptor_morgan"
    assert load_molecular_feature_cache_descriptor_names(cache_path, fingerprint_size=4) == ("MW", "TopoPSA", "ALogP")
    assert builder.descriptor_names(3) == ("MW", "TopoPSA", "ALogP")


def test_molecular_feature_builder_cache_miss_keeps_cache_descriptor_width(tmp_path: Path) -> None:
    cache_path = tmp_path / "padel_features.jsonl"
    payload = {
        "smiles": "CCO",
        "feature_source": "padel_descriptor_morgan",
        "descriptor_names": ["MW", "TopoPSA", "ALogP"],
        "descriptors": [46.0, 20.2, -0.1],
        "fingerprint": [1.0, 0.0, 1.0, 0.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)
    descriptors, fingerprint = builder.encode("")

    assert descriptors == [0.0, 0.0, 0.0]
    assert len(fingerprint) == 4


def test_molecular_feature_builder_pads_cached_descriptors_to_cache_schema(tmp_path: Path) -> None:
    cache_path = tmp_path / "padel_features.jsonl"
    payloads = [
        {
            "smiles": "CCO",
            "feature_source": "padel_descriptor_morgan",
            "descriptor_names": ["MW", "TopoPSA", "ALogP"],
            "descriptors": [46.0, 20.2, -0.1],
            "fingerprint": [1.0, 0.0, 1.0, 0.0],
        },
        {
            "smiles": "CCC",
            "feature_source": "padel_descriptor_morgan",
            "descriptor_names": ["MW", "TopoPSA", "ALogP"],
            "descriptors": [44.0],
            "fingerprint": [0.0, 1.0, 0.0, 1.0],
        },
    ]
    cache_path.write_text("\n".join(json.dumps(payload) for payload in payloads) + "\n", encoding="utf-8")

    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)

    assert builder.encode("CCC") == ([44.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0])


def test_numeric_matrix_clips_extreme_cache_descriptors(tmp_path: Path) -> None:
    cache_path = tmp_path / "padel_features.jsonl"
    payload = {
        "smiles": "CCO",
        "feature_source": "padel_descriptor_morgan",
        "descriptor_names": ["HugePositive", "HugeNegative"],
        "descriptors": [1.0e308, -1.0e308],
        "fingerprint": [1.0, 0.0, 1.0, 0.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    frame = _tiny_frame()
    frame["smiles"] = "CCO"
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)

    matrix = raw_numeric_matrix(frame, builder, descriptor_names=builder.descriptor_names())
    stats = fit_numeric_stats(frame, builder, descriptor_names=builder.descriptor_names())

    assert np.isfinite(matrix).all()
    assert np.max(np.abs(matrix[:, :2])) <= RAW_NUMERIC_CLIP_ABS
    assert all(np.isfinite([mean, std]).all() for mean, std in stats.values())


def test_molecular_feature_builder_treats_nan_smiles_as_missing() -> None:
    builder = MolecularFeatureBuilder(fingerprint_size=4)
    builder._rdkit_available = True

    def fail_rdkit(smiles: str) -> tuple[list[float], list[float]]:
        raise AssertionError(f"missing SMILES should not reach RDKit: {smiles!r}")

    builder._encode_rdkit = fail_rdkit  # type: ignore[method-assign]

    assert builder.encode(float("nan")) == builder._encode_fallback("")
    assert builder.encode("nan") == builder._encode_fallback("")


def test_build_molecular_feature_cache_accepts_source_table(tmp_path: Path) -> None:
    db_path = tmp_path / "modeling.sqlite"
    out_path = tmp_path / "features.jsonl"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE aggregated_task_records (smiles TEXT)")
        conn.execute("CREATE TABLE aggregated_task_records_aquatic_soil_ptox_qc (smiles TEXT)")
        conn.execute("INSERT INTO aggregated_task_records VALUES ('CCO')")
        conn.execute("INSERT INTO aggregated_task_records_aquatic_soil_ptox_qc VALUES ('CCN')")
        conn.commit()

    manifest = build_molecular_feature_cache(
        db_path,
        out_path=out_path,
        fingerprint_size=4,
        source_table="aggregated_task_records_aquatic_soil_ptox_qc",
    )

    payloads = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert manifest["source_table"] == "aggregated_task_records_aquatic_soil_ptox_qc"
    assert [payload["smiles"] for payload in payloads] == ["CCN"]


def test_no_fingerprint_ablation_masks_fingerprint(tmp_path: Path) -> None:
    frame = _tiny_frame()
    cache_path = _cache_path(tmp_path)
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=cache_path)
    spec = ABLATION_SPECS["no_fingerprint"]
    maps = fit_categorical_maps(frame, ablation=spec)
    stats = fit_numeric_stats(frame, builder, ablation=spec)

    samples = build_deep_samples(
        frame,
        encoder=builder,
        categorical_maps=maps,
        numeric_stats=stats,
        target_column="target_value",
        ablation=spec,
    )

    assert samples[0]["fingerprint"] == [0.0, 0.0, 0.0, 0.0]


def test_no_molecular_size_descriptors_masks_weight_related_group() -> None:
    spec = ABLATION_SPECS["no_molecular_size_descriptors"]

    masked = masked_descriptors([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], spec)

    assert masked == [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert spec.masked_descriptor_names == MOLECULAR_SIZE_RELATED_DESCRIPTOR_NAMES


def test_descriptor_mask_uses_supplied_descriptor_names() -> None:
    spec = ABLATION_SPECS["no_molecular_size_descriptors"]

    masked = masked_descriptors(
        [1.0, 2.0, 3.0],
        spec,
        descriptor_names=("Unrelated", "MolWt", "TPSA"),
    )

    assert masked == [1.0, 0.0, 0.0]
    assert descriptor_mask_indices(3, spec, descriptor_names=("Unrelated", "MolWt", "TPSA")) == (1, 2)


def test_preprocessing_manifest_records_molecular_size_descriptor_mask() -> None:
    spec = ABLATION_SPECS["no_molecular_size_descriptors"]
    manifest = build_preprocessing_manifest(
        categorical_maps={},
        adapter_map={},
        numeric_stats={str(idx): (0.0, 1.0) for idx in range(8)},
        ablation=spec,
        fingerprint_size=4,
        encoder_source="rdkit_cache",
        cache_path=None,
        descriptor_count=8,
    )

    assert manifest["masked_descriptor_names"] == list(MOLECULAR_SIZE_RELATED_DESCRIPTOR_NAMES)
    assert manifest["masked_descriptor_indices"] == [0, 1, 3, 4, 5, 6, 7]


def test_preprocessing_manifest_uses_cache_descriptor_names() -> None:
    spec = ABLATION_SPECS["full"]
    manifest = build_preprocessing_manifest(
        categorical_maps={},
        adapter_map={},
        numeric_stats={str(idx): (0.0, 1.0) for idx in range(3)},
        ablation=spec,
        fingerprint_size=4,
        encoder_source="padel_descriptor_morgan",
        cache_path="features.jsonl",
        descriptor_count=3,
        descriptor_names=("MW", "TopoPSA", "ALogP"),
        descriptor_encoder={"mode": "dense_head", "head_dim": 8},
    )

    assert manifest["molecular_descriptor_names"] == ("MW", "TopoPSA", "ALogP")
    assert manifest["numeric_feature_names"][:3] == ("MW", "TopoPSA", "ALogP")
    assert manifest["descriptor_encoder"]["mode"] == "dense_head"


def test_no_context_ablation_removes_categorical_context(tmp_path: Path) -> None:
    frame = _tiny_frame()
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=_cache_path(tmp_path))
    spec = ABLATION_SPECS["no_context"]

    samples = build_deep_samples(
        frame,
        encoder=builder,
        categorical_maps=fit_categorical_maps(frame, ablation=spec),
        numeric_stats=fit_numeric_stats(frame, builder, ablation=spec),
        target_column="target_value",
        ablation=spec,
    )

    assert active_categorical_columns(spec) == ()
    assert samples[0]["categorical_ids"] == {}
    assert all(value == 0.0 for value in samples[0]["molecular_numeric"][2:])


def test_build_censored_training_samples_uses_reference_split(tmp_path: Path) -> None:
    db_path = tmp_path / "targets.sqlite"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE target_records (
                record_id TEXT,
                cas_number TEXT,
                dtxsid TEXT,
                species_number TEXT,
                endpoint TEXT,
                effect TEXT,
                measurement TEXT,
                target_name TEXT,
                target_basis TEXT,
                target_status TEXT,
                value_quality TEXT,
                excluded_reason TEXT,
                conc1_mean_op TEXT,
                conc1_min_op TEXT,
                conc1_max_op TEXT,
                standard_value_mol_l REAL,
                medium_domain TEXT,
                smiles TEXT,
                latin_name TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO target_records VALUES (
                'r1','50-00-0','DTXSID1','sp1','EC50','MOR','','ptox_mol_l','mol/L',
                'excluded','censored','censored_toxicity_value','>','','',
                0.001,'aquatic','CCO','Daphnia magna'
            )
            """
        )
        conn.commit()
    frame = pd.DataFrame(
        [
            {
                "aggregate_id": "a1",
                "cas_number": "50-00-0",
                "dtxsid": "DTXSID1",
                "species_number": "sp1",
                "task_head": "ECx_Mortality",
                "target_name": "ptox_mol_l",
                "target_basis": "mol/L",
                "target_family": "aquatic_pTox_mol_L",
                "medium_domain": "aquatic",
                "split_part": "train",
                "smiles": "CCO",
                "target_value": 3.0,
            }
        ]
    )
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=_cache_path(tmp_path))
    samples, summary = build_censored_training_samples(
        db_path,
        frame,
        encoder=builder,
        categorical_maps=fit_categorical_maps(frame, ablation=ABLATION_SPECS["full"]),
        adapter_map=fit_adapter_map(frame, ablation=ABLATION_SPECS["full"]),
        numeric_stats=fit_numeric_stats(frame, builder, ablation=ABLATION_SPECS["full"]),
        target_column="target_value",
        target_scaler=fit_target_scaler(frame, target_column="target_value", mode="none"),
        zscore_correction=None,
        ablation=ABLATION_SPECS["full"],
        config=CensoredLossConfig(enabled=True, weight=0.03),
        kept_task_heads=("ECx_Mortality",),
        split_parts=("train",),
    )

    assert summary["usable_rows"] == 1
    assert summary["train_rows"] == 1
    assert samples[0]["censored_direction"] == "right"
    assert samples[0]["censored_direction_id"] == 1
    assert samples[0]["target_value_raw"] == pytest.approx(3.0)


def test_no_molecular_residual_ablation_disables_residual_layer() -> None:
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=2,
            fingerprint_dim=4,
            categorical_cardinalities={},
            task_heads=("ECx_Mortality",),
            hidden_dims=(8,),
            use_molecular_residual=False,
        )
    )

    assert model.molecular_residual is None


def test_descriptor_dense_head_forward_shape() -> None:
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=5,
            fingerprint_dim=4,
            descriptor_count=3,
            descriptor_encoder_mode="dense_head",
            descriptor_head_dim=6,
            categorical_cardinalities={},
            task_heads=("ECx_Mortality",),
            hidden_dims=(8,),
        )
    )

    outputs = model(
        torch.zeros((2, 5), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        {},
    )

    assert outputs["ECx_Mortality"].shape == (2,)


def test_descriptor_prior_clustered_head_forward_shape() -> None:
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=5,
            fingerprint_dim=4,
            descriptor_count=3,
            descriptor_encoder_mode="prior_clustered_heads",
            descriptor_group_head_dim=4,
            descriptor_group_indices={"size": (0, 1), "partition": (2,)},
            categorical_cardinalities={},
            task_heads=("ECx_Mortality",),
            hidden_dims=(8,),
        )
    )

    outputs = model(
        torch.zeros((2, 5), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        {},
    )

    assert outputs["ECx_Mortality"].shape == (2,)


def test_apply_finetune_freeze_heads_embeddings() -> None:
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=2,
            fingerprint_dim=4,
            categorical_cardinalities={"family": 3},
            adapter_count=2,
            task_heads=("ECx_Mortality",),
            hidden_dims=(8,),
        )
    )

    trainable = apply_finetune_freeze(model, "heads_embeddings")

    assert trainable
    assert any(parameter.requires_grad for parameter in model.heads.parameters())
    assert any(parameter.requires_grad for parameter in model.embeddings.parameters())
    assert any(parameter.requires_grad for parameter in model.adapters.parameters())
    assert not any(parameter.requires_grad for parameter in model.trunk.parameters())
    if model.molecular_residual is not None:
        assert not any(parameter.requires_grad for parameter in model.molecular_residual.parameters())


def test_preprocessing_manifest_records_feature_schema(tmp_path: Path) -> None:
    frame = _tiny_frame()
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=_cache_path(tmp_path))
    spec = ABLATION_SPECS["full"]
    maps = fit_categorical_maps(frame, ablation=spec)
    adapter_map = fit_adapter_map(frame, ablation=spec)
    stats = fit_numeric_stats(frame, builder, ablation=spec)

    manifest = build_preprocessing_manifest(
        categorical_maps=maps,
        adapter_map=adapter_map,
        numeric_stats=stats,
        ablation=spec,
        fingerprint_size=4,
        encoder_source=builder.source,
        cache_path=str(_cache_path(tmp_path)),
        descriptor_count=2,
    )

    assert manifest["schema_version"] == 2
    assert manifest["fingerprint_size"] == 4
    assert "duration_log1p_h" in manifest["numeric_feature_names"]
    assert "species_number" not in manifest["categorical_columns"]
    assert "family" in manifest["categorical_columns"]
    assert "genus" in manifest["categorical_columns"]
    assert manifest["adapter_map"] == {"aquatic|ptox_mol_l": 1}
    assert manifest["adapter_cardinality"] == 2
    assert manifest["numeric_stats"]["descriptor_0"]["index"] == 0
    assert "mean" in manifest["numeric_stats"]["descriptor_0"]
    assert "std" in manifest["numeric_stats"]["descriptor_0"]


def test_validation_split_prefers_finetune_part() -> None:
    samples = [
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "finetune", "task_head": "ECx_Mortality"},
    ]

    train, validation, source = split_training_validation_indices(
        samples,
        train_indices=[0, 1],
        seed=42,
        validation_fraction=0.5,
    )

    assert train == [0, 1]
    assert validation == [2]
    assert source == "finetune"


def test_validation_split_falls_back_to_train_fraction() -> None:
    samples = [
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "train", "task_head": "ECx_Growth"},
        {"split_part": "train", "task_head": "ECx_Growth"},
    ]

    train, validation, source = split_training_validation_indices(
        samples,
        train_indices=[0, 1, 2, 3],
        seed=42,
        validation_fraction=0.5,
    )

    assert source == "internal_train_fraction"
    assert len(train) == 2
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)


def test_validation_split_can_force_internal_train_fraction() -> None:
    samples = [
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "train", "task_head": "ECx_Mortality"},
        {"split_part": "finetune", "task_head": "ECx_Mortality"},
    ]

    train, validation, source = split_training_validation_indices(
        samples,
        train_indices=[0, 1],
        seed=42,
        validation_fraction=0.5,
        monitor_split="internal_train_fraction",
    )

    assert source == "internal_train_fraction"
    assert len(train) == 1
    assert len(validation) == 1
    assert validation != [2]


def test_finetune_validation_split_holds_out_finetune_rows() -> None:
    samples = [
        {"split_part": "finetune", "task_head": "ECx_Mortality"},
        {"split_part": "finetune", "task_head": "ECx_Mortality"},
        {"split_part": "finetune", "task_head": "NOEC_Growth"},
        {"split_part": "finetune", "task_head": "NOEC_Growth"},
    ]

    train, validation, source = split_finetune_validation_indices(
        samples,
        finetune_indices=[0, 1, 2, 3],
        seed=42,
        validation_fraction=0.5,
    )

    assert source == "internal_finetune_fraction"
    assert len(train) == 2
    assert len(validation) == 2
    assert set(train).isdisjoint(validation)


def test_noisy_index_dataset_is_reproducible() -> None:
    dataset = AggregatedTaskDataset(
        samples=[
            {
                "sample_id": "a",
                "molecular_numeric": [0.0, 1.0],
                "fingerprint": [1.0, 0.0],
                "categorical_ids": {},
                "task_head": "ECx_Mortality",
                "target_value": 1.0,
            }
        ],
        fingerprint_size=2,
    )

    first = build_noisy_index_dataset(
        dataset,
        [0],
        replicates=2,
        numeric_noise_std=0.1,
        target_noise_std=0.0,
        seed=123,
    )
    second = build_noisy_index_dataset(
        dataset,
        [0],
        replicates=2,
        numeric_noise_std=0.1,
        target_noise_std=0.0,
        seed=123,
    )

    assert len(first) == 2
    assert first[0]["molecular_numeric"] == [0.0, 1.0]
    assert first[1]["molecular_numeric"] == second[1]["molecular_numeric"]
    assert first[1]["molecular_numeric"] != [0.0, 1.0]


def test_target_scaler_uses_fit_indices_only() -> None:
    frame = pd.DataFrame(
        [
            {"split_part": "train", "task_head": "A", "target_name": "water", "target_value": 10.0},
            {"split_part": "train", "task_head": "A", "target_name": "water", "target_value": 12.0},
            {"split_part": "test", "task_head": "A", "target_name": "water", "target_value": 1000.0},
        ]
    )

    scaler = fit_target_scaler(
        frame,
        target_column="target_value",
        mode="per_task_target",
        fit_indices=[0, 1],
    )

    stat = scaler.stats["A|water"]
    assert stat["mean"] == 11.0
    assert stat["std"] == 1.0
    assert scaler.inverse_transform("A|water", scaler.transform("A|water", 12.0)) == 12.0


def test_categorical_map_separates_rare_and_unknown() -> None:
    frame = pd.DataFrame(
        [
            {"family": "Daphniidae"},
            {"family": "Daphniidae"},
            {"family": "RareFamily"},
        ]
    )

    maps = fit_categorical_maps(frame, min_count=2)
    family_map = maps["family"]

    assert family_map["RareFamily"] == family_map["<rare>"]
    assert encode_category_id("NotSeenInTraining", family_map) == family_map["<unknown>"]
    assert encode_category_id("Daphniidae", family_map) != family_map["<rare>"]


def test_batch_weighted_loss_is_normalized_by_task_weights() -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, molecular_numeric, fingerprint, categorical_ids, adapter_ids=None):
            del molecular_numeric, fingerprint, categorical_ids, adapter_ids
            return {
                "A": torch.zeros(3, dtype=torch.float32),
                "B": torch.zeros(3, dtype=torch.float32),
            }

    loss = batch_weighted_loss(
        ConstantModel(),
        torch.zeros((3, 0), dtype=torch.float32),
        torch.zeros((3, 0), dtype=torch.float32),
        {},
        None,
        torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32),
        ["A", "A", "B"],
        torch.nn.HuberLoss(delta=1.0, reduction="mean"),
        DeepTrainingConfig(task_weights={"A": 1.0, "B": 1.0}),
        torch.device("cpu"),
    )

    assert torch.isclose(loss, torch.tensor(1.5))


def test_predictions_include_metadata_and_raw_scale(tmp_path: Path) -> None:
    frame = _tiny_frame()
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=_cache_path(tmp_path))
    spec = ABLATION_SPECS["full"]
    scaler = fit_target_scaler(
        frame,
        target_column="target_value",
        mode="per_task_target",
        fit_indices=[0],
    )
    samples = build_deep_samples(
        frame,
        encoder=builder,
        categorical_maps=fit_categorical_maps(frame.iloc[[0]], ablation=spec),
        adapter_map=fit_adapter_map(frame.iloc[[0]], ablation=spec),
        numeric_stats=fit_numeric_stats(frame.iloc[[0]], builder, ablation=spec),
        target_column="target_value",
        target_scaler=scaler,
        ablation=spec,
    )
    dataset = AggregatedTaskDataset(samples=samples, fingerprint_size=4)

    class ZeroModel(torch.nn.Module):
        def forward(self, molecular_numeric, fingerprint, categorical_ids, adapter_ids=None):
            del fingerprint, categorical_ids, adapter_ids
            return {"ECx_Mortality": torch.zeros(molecular_numeric.shape[0], dtype=torch.float32)}

    predictions = predict_all(
        ZeroModel(),
        dataset,
        samples,
        batch_size=2,
        device=torch.device("cpu"),
        target_scaler=scaler,
    )

    assert predictions[0]["sample_id"] == "a1"
    assert predictions[0]["target_name"] == "ptox_mol_l"
    assert predictions[0]["medium_domain"] == "aquatic"
    assert predictions[0]["effect_level_x"] == 50.0
    assert predictions[0]["y_true"] == 1.0
    assert predictions[0]["y_pred"] == 1.0
    assert "y_pred_scaled" in predictions[0]


def test_predictions_route_interleaved_heads_when_auxiliary_output_is_first() -> None:
    task_heads = ["A", "B", "A", "B"]
    samples = [
        {
            "sample_id": f"row-{idx}",
            "split_part": "test",
            "task_head": task_head,
            "molecular_numeric": [float(idx)],
            "fingerprint": [1.0, 0.0],
            "categorical_ids": {},
            "target_value": float(idx),
        }
        for idx, task_head in enumerate(task_heads)
    ]
    dataset = AggregatedTaskDataset(samples=samples, fingerprint_size=2)

    class InterleavedModel(torch.nn.Module):
        def forward(self, molecular_numeric, fingerprint, categorical_ids, adapter_ids=None):
            del fingerprint, categorical_ids, adapter_ids
            row = torch.arange(molecular_numeric.shape[0], dtype=torch.float32)
            return {
                TOXICITY_BIN_LOGITS_KEY: torch.zeros((molecular_numeric.shape[0], 3)),
                "A": row + 10.0,
                "B": row + 20.0,
            }

    predictions = predict_all(
        InterleavedModel(),
        dataset,
        samples,
        batch_size=4,
        device=torch.device("cpu"),
    )

    assert [row["y_pred_scaled"] for row in predictions] == [10.0, 21.0, 12.0, 23.0]


def test_effect_level_metrics_keep_x_levels_separate() -> None:
    predictions = [
        {
            "split_part": "test",
            "task_head": "ECx_Mortality",
            "target_name": "ptox_mol_l",
            "medium_domain": "soil",
            "effect_level_x": 10.0,
            "y_true": 1.0,
            "y_pred": 1.1,
        },
        {
            "split_part": "test",
            "task_head": "ECx_Mortality",
            "target_name": "ptox_mol_l",
            "medium_domain": "soil",
            "effect_level_x": 50.0,
            "y_true": 2.0,
            "y_pred": 1.9,
        },
    ]

    rows = metrics_by_group(
        predictions,
        huber_delta=1.0,
        group_columns=("split_part", "task_head", "target_name", "medium_domain", "effect_level_x"),
    )

    assert {row["effect_level_x"] for row in rows} == {"10", "50"}


def test_zscore_correction_clips_test_sample_with_train_parameters(tmp_path: Path) -> None:
    frame = _tiny_frame()
    builder = MolecularFeatureBuilder(fingerprint_size=4, cache_path=_cache_path(tmp_path))
    spec = ABLATION_SPECS["full"]
    train_frame = frame.iloc[[0]]
    descriptor_count = len(builder.encode("CCO")[0])
    numeric_stats = fit_numeric_stats(train_frame, builder, ablation=spec)
    correction = fit_zscore_correction(
        train_frame,
        builder,
        numeric_stats=numeric_stats,
        feature_names=build_numeric_feature_names(descriptor_count),
        config=ZScoreCorrectionConfig(enabled=True, threshold=1.0),
        ablation=spec,
    )
    samples = build_deep_samples(
        frame,
        encoder=builder,
        categorical_maps=fit_categorical_maps(train_frame, ablation=spec),
        adapter_map=fit_adapter_map(train_frame, ablation=spec),
        numeric_stats=numeric_stats,
        target_column="target_value",
        zscore_correction=correction,
        ablation=spec,
    )

    assert max(samples[1]["molecular_numeric"]) <= 1.0
    assert min(samples[1]["molecular_numeric"]) >= -1.0


def test_source_similarity_weighting_defaults_to_unit_weights() -> None:
    samples = [
        {"split_part": "train", "medium_domain": "aquatic", "fingerprint": [1.0, 0.0]},
        {"split_part": "finetune", "medium_domain": "soil", "fingerprint": [1.0, 0.0]},
    ]

    summary = apply_source_similarity_weights(samples, SourceWeightingConfig())

    assert summary["applied"] is False
    assert [sample["sample_weight"] for sample in samples] == [1.0, 1.0]


def test_source_similarity_weighting_upweights_target_like_source_samples() -> None:
    samples = [
        {"split_part": "train", "medium_domain": "aquatic", "fingerprint": [1.0, 0.0, 0.0, 0.0]},
        {"split_part": "train", "medium_domain": "aquatic", "fingerprint": [0.0, 1.0, 0.0, 0.0]},
        {"split_part": "finetune", "medium_domain": "soil", "fingerprint": [1.0, 0.0, 0.0, 0.0]},
    ]
    config = SourceWeightingConfig(enabled=True, method="tanimoto_to_finetune", alpha=1.0)

    summary = apply_source_similarity_weights(samples, config)

    assert summary["applied"] is True
    assert samples[0]["sample_weight"] > samples[1]["sample_weight"]
    assert samples[2]["sample_weight"] == 1.0


def test_source_similarity_weight_cache_is_exact_and_reusable(tmp_path: Path) -> None:
    samples = [
        {"sample_id": "source-a", "split_part": "train", "medium_domain": "aquatic", "fingerprint": [1.0, 0.0]},
        {"sample_id": "source-b", "split_part": "train", "medium_domain": "aquatic", "fingerprint": [0.0, 1.0]},
        {"sample_id": "target", "split_part": "finetune", "medium_domain": "soil", "fingerprint": [1.0, 0.0]},
    ]
    config = SourceWeightingConfig(enabled=True, method="tanimoto_to_finetune", alpha=1.0)

    cold = apply_source_similarity_weights(samples, config, cache_dir=tmp_path)
    cold_weights = [sample["sample_weight"] for sample in samples]
    warm = apply_source_similarity_weights(samples, config, cache_dir=tmp_path)

    assert cold["cache_hit"] is False
    assert warm["cache_hit"] is True
    assert warm["cache_key"] == cold["cache_key"]
    assert [sample["sample_weight"] for sample in samples] == cold_weights
    assert Path(str(warm["cache_path"])).exists()

    samples[0]["fingerprint"] = [0.0, 1.0]
    changed = apply_source_similarity_weights(samples, config, cache_dir=tmp_path)
    assert changed["cache_hit"] is False
    assert changed["cache_key"] != cold["cache_key"]


def test_dataloader_runtime_options_enable_cuda_prefetch_only_with_workers() -> None:
    assert dataloader_runtime_options("cpu", 0) == {"num_workers": 0, "pin_memory": False}
    assert dataloader_runtime_options("cuda:0", 0) == {"num_workers": 0, "pin_memory": True}
    assert dataloader_runtime_options("cuda:0", 4) == {
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
    }
    assert dataloader_runtime_options("cuda:0", 4, persistent_workers=True) == {
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": True,
    }


def test_proxy_distance_weighting_upweights_proxy_near_source_samples() -> None:
    samples = [
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "fingerprint": [1.0, 0.0],
            "molecular_numeric": [0.0, 0.0, 0.0, 9.0],
        },
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "fingerprint": [0.0, 1.0],
            "molecular_numeric": [8.0, 8.0, 8.0, 9.0],
        },
        {
            "split_part": "finetune",
            "medium_domain": "soil",
            "fingerprint": [0.0, 1.0],
            "molecular_numeric": [0.1, 0.1, 0.1, 9.0],
        },
    ]
    config = SourceWeightingConfig(enabled=True, method="proxy_distance_to_finetune", alpha=1.0)

    summary = apply_source_similarity_weights(samples, config)

    assert summary["applied"] is True
    assert "proxy_distance_mean" in summary
    assert samples[0]["sample_weight"] > samples[1]["sample_weight"]
    assert samples[2]["sample_weight"] == 1.0


def test_tanimoto_proxy_weighting_reports_similarity_and_proxy_distance() -> None:
    samples = [
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "fingerprint": [1.0, 0.0, 0.0],
            "molecular_numeric": [0.0, 0.0, 0.0],
        },
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "fingerprint": [0.0, 1.0, 0.0],
            "molecular_numeric": [4.0, 4.0, 4.0],
        },
        {
            "split_part": "finetune",
            "medium_domain": "soil",
            "fingerprint": [1.0, 0.0, 0.0],
            "molecular_numeric": [0.1, 0.1, 0.1],
        },
    ]
    config = SourceWeightingConfig(enabled=True, method="tanimoto_proxy_to_finetune", alpha=1.0)

    summary = apply_source_similarity_weights(samples, config)

    assert summary["applied"] is True
    assert "similarity_mean" in summary
    assert "proxy_distance_mean" in summary
    assert samples[0]["sample_weight"] > samples[1]["sample_weight"]


def test_max_tanimoto_similarity_deduplicates_without_changing_scores() -> None:
    source = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    expected = []
    for row in source.astype(bool):
        row_scores = []
        for ref in target.astype(bool):
            intersection = np.logical_and(row, ref).sum()
            union = np.logical_or(row, ref).sum()
            row_scores.append(intersection / union if union else 0.0)
        expected.append(max(row_scores))

    actual = max_tanimoto_similarity(source, target, chunk_size=1)

    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_min_proxy_distance_matches_direct_euclidean_minimum() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    pooled = np.vstack([source, target])
    std = np.where(pooled.std(axis=0) > 1e-12, pooled.std(axis=0), 1.0)
    scaled_source = (source - pooled.mean(axis=0)) / std
    scaled_target = (target - pooled.mean(axis=0)) / std
    expected = np.asarray(
        [np.sqrt(np.min(np.sum((scaled_target - row[None, :]) ** 2, axis=1))) for row in scaled_source],
        dtype=np.float32,
    )

    actual = min_proxy_distance(source, target, batch_size=2)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_effect_level_weighting_defaults_to_existing_weights() -> None:
    samples = [
        {"split_part": "train", "task_head": "ECx_Mortality", "effect_level_x": 10.0, "sample_weight": 1.25},
        {"split_part": "finetune", "task_head": "ECx_Mortality", "effect_level_x": 50.0, "sample_weight": 0.75},
    ]

    summary = apply_effect_level_frequency_weights(
        samples,
        EffectLevelWeightingConfig(),
        train_indices=[0, 1],
    )

    assert summary["applied"] is False
    assert [sample["sample_weight"] for sample in samples] == [1.25, 0.75]


def test_effect_level_weighting_uses_train_and_finetune_train_only() -> None:
    samples = [
        {"split_part": "train", "task_head": "ECx_Mortality", "effect_level_x": 10.0, "sample_weight": 1.0},
        {"split_part": "train", "task_head": "ECx_Growth", "effect_level_x": 50.0, "sample_weight": 1.0},
        {"split_part": "train", "task_head": "ECx_Reproduction", "effect_level_x": 50.0, "sample_weight": 1.0},
        {"split_part": "finetune", "task_head": "ECx_Mortality", "effect_level_x": 50.0, "sample_weight": 1.0},
        {
            "split_part": "finetune_validation",
            "task_head": "ECx_Mortality",
            "effect_level_x": 10.0,
            "sample_weight": 1.0,
        },
        {"split_part": "test", "task_head": "ECx_Mortality", "effect_level_x": 10.0, "sample_weight": 1.0},
        {"split_part": "train", "task_head": "NOEC_Mortality", "effect_level_x": None, "sample_weight": 1.0},
    ]

    summary = apply_effect_level_frequency_weights(
        samples,
        EffectLevelWeightingConfig(enabled=True, beta=0.5),
        train_indices=[0, 1, 2, 3],
    )

    assert summary["applied"] is True
    assert summary["level_counts"] == {"10": 1, "50": 3}
    assert summary["weighted_samples"] == 4
    assert samples[0]["sample_weight"] > 1.0
    assert 0.5 <= samples[1]["sample_weight"] <= 1.0
    assert samples[4]["sample_weight"] == 1.0
    assert samples[5]["sample_weight"] == 1.0
    assert samples[6]["sample_weight"] == 1.0


def test_effect_level_weighting_multiplies_source_weights() -> None:
    samples = [
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "task_head": "ECx_Mortality",
            "effect_level_x": 10.0,
            "fingerprint": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "task_head": "ECx_Mortality",
            "effect_level_x": 50.0,
            "fingerprint": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "split_part": "train",
            "medium_domain": "aquatic",
            "task_head": "ECx_Growth",
            "effect_level_x": 50.0,
            "fingerprint": [0.0, 0.0, 1.0, 0.0],
        },
        {
            "split_part": "finetune",
            "medium_domain": "soil",
            "task_head": "ECx_Mortality",
            "effect_level_x": 50.0,
            "fingerprint": [1.0, 0.0, 0.0, 0.0],
        },
    ]

    apply_source_similarity_weights(
        samples,
        SourceWeightingConfig(enabled=True, method="tanimoto_to_finetune", alpha=1.0),
    )
    source_weights = [sample["sample_weight"] for sample in samples]
    summary = apply_effect_level_frequency_weights(
        samples,
        EffectLevelWeightingConfig(enabled=True, beta=0.5),
        train_indices=[0, 1, 2, 3],
    )

    assert summary["applied"] is True
    assert samples[0]["sample_weight"] > source_weights[0]
    assert samples[1]["sample_weight"] < source_weights[1]
    assert samples[2]["sample_weight"] < source_weights[2]
    assert samples[3]["sample_weight"] < source_weights[3]


def test_clipped_effect_level_weights_keep_mean_one() -> None:
    weights = np.asarray([0.05] * 100 + [1.0] * 10 + [25.0] * 2, dtype=np.float32)

    clipped = normalize_clipped_weights(weights, 0.5, 3.0)

    assert clipped.min() >= 0.5
    assert clipped.max() <= 3.0
    assert abs(float(clipped.mean()) - 1.0) < 1e-6


def test_collate_preserves_training_metadata() -> None:
    batch = [
        {
            "molecular_numeric": [0.1],
            "fingerprint": [1.0, 0.0],
            "categorical_ids": {},
            "adapter_id": 0,
            "task_head": "ECx_Mortality",
            "target_value": 1.0,
            "sample_weight": 1.5,
            "toxicity_bin_index": 2,
            "split_part": "train",
            "medium_domain": "aquatic",
        }
    ]

    collated = collate_aggregated_task_batch(batch)

    assert float(collated["sample_weight"][0]) == 1.5
    assert int(collated["toxicity_bin_index"][0]) == 2
    assert collated["split_part"] == ["train"]
    assert collated["medium_domain"] == ["aquatic"]


def test_coral_loss_is_zero_for_identical_representations() -> None:
    shared = torch.tensor([[1.0, 2.0], [3.0, 5.0], [4.0, 8.0]])

    assert float(coral_loss(shared, shared).detach()) < 1e-8


def test_padel_csv_cache_writer_outputs_compatible_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "padel.csv"
    csv_path.write_text(
        "smiles,MW,TopoPSA,ALogP\nCCO,46.07,20.2,-0.1\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "padel_features.jsonl"

    result = write_padel_feature_cache_from_csv(
        csv_path,
        out_path,
        smiles_column="smiles",
        fingerprint_size=8,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert result.rows_written == 1
    assert payload["feature_source"] == "padel_descriptor_morgan"
    assert payload["descriptor_names"] == ["MW", "TopoPSA", "ALogP"]
    assert len(payload["fingerprint"]) == 8
    assert result.manifest_path.exists()


def test_descriptor_group_rules_resolve_to_indices() -> None:
    groups = {
        "size": {"names": ["MW"], "prefixes": ["nAtom"]},
        "partition": {"contains": ["logp"]},
    }

    resolved = resolve_descriptor_group_indices(groups, ("MW", "nAtomP", "ALogP", "TopoPSA"))

    assert resolved["size"] == (0, 1)
    assert resolved["partition"] == (2,)


def test_molecular_graph_cache_writes_deep_graph_encoder_interface(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    out_path = tmp_path / "graphs.jsonl"

    manifest = write_molecular_graph_cache(["CCO", "CCO"], out_path)

    assert manifest["training_integration"] == "deep_graph_encoder"
    assert manifest["written"] == 1
    payload = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["smiles"] == "CCO"
    assert payload["atom_features"]
    assert "edge_index" in payload


def test_build_run_dir_uses_version_and_chinese_name() -> None:
    run_dir = build_run_dir(
        Path("outputs/experiments"),
        config={"experiment": {"version": "v1.0.0", "name_zh": "训练优化重构_水相pTox随机划分对照"}},
        ablation="full",
        split_name="AquaticPtox_B_random_8_2",
    )

    assert "v1.0.0_训练优化重构_水相pTox随机划分对照" in str(run_dir)
    assert run_dir.name == "AquaticPtox_B_random_8_2"


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "aggregate_id": "a1",
                "smiles": "CCO",
                "split_part": "train",
                "task_head": "ECx_Mortality",
                "effect_level_x": 50.0,
                "target_value": 1.0,
                "target_name": "ptox_mol_l",
                "duration_bin_h": 24.0,
                "duration_log1p_h": 3.2,
                "target_value_count": 2,
                "species_number": "sp1",
                "latin_name": "Daphnia magna",
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class_name": "Branchiopoda",
                "tax_order": "Diplostraca",
                "family": "Daphniidae",
                "genus": "Daphnia",
                "species": "magna",
                "medium_domain": "aquatic",
                "media_type": "water",
                "organism_lifestage": "adult",
                "target_basis": "active ingredient",
                "effect_family": "Mortality",
            },
            {
                "aggregate_id": "a2",
                "smiles": "CCO",
                "split_part": "test",
                "task_head": "ECx_Mortality",
                "effect_level_x": 10.0,
                "target_value": 2.0,
                "target_name": "ptox_mol_l",
                "duration_bin_h": 48.0,
                "duration_log1p_h": 3.9,
                "target_value_count": 1,
                "species_number": "sp2",
                "latin_name": "Pimephales promelas",
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class_name": "Actinopterygii",
                "tax_order": "Cypriniformes",
                "family": "Leuciscidae",
                "genus": "Pimephales",
                "species": "promelas",
                "medium_domain": "aquatic",
                "media_type": "water",
                "organism_lifestage": "juvenile",
                "target_basis": "active ingredient",
                "effect_family": "Mortality",
            },
        ]
    )


def _cache_path(tmp_path: Path) -> Path:
    cache_path = tmp_path / "features.jsonl"
    payload = {
        "smiles": "CCO",
        "descriptors": [3.0, 4.0],
        "fingerprint": [0.0, 1.0, 0.0, 1.0],
    }
    cache_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return cache_path
