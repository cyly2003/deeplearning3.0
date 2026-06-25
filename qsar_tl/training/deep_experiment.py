from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qsar_tl.evaluation.metrics import regression_metrics
from qsar_tl.modeling.dataset import AggregatedTaskDataset
from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork, TOXICITY_BIN_LOGITS_KEY
from qsar_tl.training.censored_loss import censored_direction, censored_hinge_loss
from qsar_tl.training.baseline import (
    EVAL_SPLIT_PARTS,
    add_duration_nonlinear_features,
    load_split_frame,
    task_skip_reason,
)
from qsar_tl.training.deep_train import (
    DeepTrainingConfig,
    build_optimizer,
    collate_aggregated_task_batch,
    set_torch_seed,
)
from qsar_tl.training.ordinal_binning import ordinal_softmax_loss
from qsar_tl.training.toxicity_binning import (
    ToxicityBinningConfig,
    assign_toxicity_bin,
    load_toxicity_bin_scheme,
    summarize_toxicity_bins,
    toxicity_bin_class_count,
)


MOLECULAR_DESCRIPTOR_NAMES = (
    "MolWt",
    "TPSA",
    "MolLogP",
    "HeavyAtomCount",
    "NumHAcceptors",
    "NumHDonors",
    "RingCount",
    "RotatableBonds",
)
EFFECT_LEVEL_NUMERIC_COLUMNS = (
    "effect_level_x",
    "effect_level_x_fraction",
    "effect_level_x_log1p",
    "effect_level_x_present",
)
DURATION_NUMERIC_COLUMNS = (
    "duration_bin_h",
    "duration_log1p_h",
    "duration_sqrt_h",
    "duration_inv_log1p_h",
    "duration_rbf_24h",
    "duration_rbf_48h",
    "duration_rbf_96h",
    "duration_rbf_168h",
    "duration_rbf_336h",
    "duration_rbf_720h",
)
CONTEXT_NUMERIC_COLUMNS = EFFECT_LEVEL_NUMERIC_COLUMNS + DURATION_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = (
    "latin_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
    "target_basis",
    "effect_family",
)
MISSING_CATEGORY_TOKEN = "<missing>"
UNKNOWN_CATEGORY_TOKEN = "<unknown>"
RARE_CATEGORY_TOKEN = "<rare>"
GLOBAL_TARGET_SCALE_KEY = "__global__"
PREDICTION_METADATA_COLUMNS = (
    "sample_id",
    "aggregate_id",
    "record_id",
    "split_name",
    "split_part",
    "task_head",
    "task_group",
    "task_family",
    "effect_level_x",
    "target_name",
    "target_family",
    "target_basis",
    "target_column",
    "target_scale_key",
    "unit_family_v2",
    "standard_unit_v2",
    "standard_value_mg_l",
    "standard_value_mol_l",
    "standard_value_mg_kg",
    "standard_value_g_ha",
    "standard_value_mg_kg_diet",
    "standard_value_mg_kg_bw_day",
    "conversion_path",
    "original_split_part",
    "medium_domain",
    "primary_medium",
    "media_type",
    "cas_number",
    "dtxsid",
    "chemical_name",
    "smiles",
    "species_number",
    "latin_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
    "organism_lifestage",
    "effect_family",
    "adapter_name",
    "adapter_id",
    "toxicity_bin_index",
    "toxicity_bin_label",
    "toxicity_bin_scheme",
    "toxicity_bin_source",
    "toxicity_bin_boundary_flag",
    "toxicity_bin_status",
    "toxicity_bin_value",
    "toxicity_bin_value_unit",
    "toxicity_bin_conversion",
    "perturbation_replicate",
    "perturbation_numeric_noise_std",
)


@dataclass(frozen=True)
class DeepExperimentResult:
    out_dir: Path
    metrics_path: Path
    history_path: Path
    manifest_path: Path
    preprocessing_path: Path
    predictions_path: Path
    best_model_path: Path | None
    encoder_source: str
    trained_tasks: tuple[str, ...]
    rows: int
    ablation: str
    best_epoch: int
    early_stopping_enabled: bool


@dataclass(frozen=True)
class AblationSpec:
    name: str
    use_descriptors: bool = True
    use_fingerprint: bool = True
    use_context_numeric: bool = True
    use_duration_features: bool = True
    use_species_lifestage: bool = True
    use_other_categorical_context: bool = True
    use_molecular_residual: bool = True
    use_medium_adapter: bool = True


@dataclass(frozen=True)
class TargetScaler:
    mode: str
    target_column: str
    fit_split_parts: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    def transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return (float(value) - float(stat["mean"])) / float(stat["std"])

    def inverse_transform(self, key: str, value: float) -> float:
        if self.mode in {"none", "identity"}:
            return float(value)
        stat = self.stats.get(key) or self.stats[GLOBAL_TARGET_SCALE_KEY]
        return float(value) * float(stat["std"]) + float(stat["mean"])

    def to_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_column": self.target_column,
            "fit_split_parts": list(self.fit_split_parts),
            "stats": self.stats,
        }


@dataclass(frozen=True)
class FeatureNoiseConfig:
    train_replicates: int = 1
    finetune_replicates: int = 1
    numeric_noise_std: float = 0.0
    target_noise_std: float = 0.0
    seed: int = 42

    def active_for_train(self) -> bool:
        return self.train_replicates > 1 and (self.numeric_noise_std > 0 or self.target_noise_std > 0)

    def active_for_finetune(self) -> bool:
        return self.finetune_replicates > 1 and (self.numeric_noise_std > 0 or self.target_noise_std > 0)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "train_replicates": self.train_replicates,
            "finetune_replicates": self.finetune_replicates,
            "numeric_noise_std": self.numeric_noise_std,
            "target_noise_std": self.target_noise_std,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PerturbationConfig:
    enabled: bool = False
    split_parts: tuple[str, ...] = ("test",)
    replicates: int = 1
    numeric_noise_std: float = 0.0
    seed: int = 42

    def active(self) -> bool:
        return self.enabled and self.replicates > 1 and self.numeric_noise_std > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "split_parts": list(self.split_parts),
            "replicates": self.replicates,
            "numeric_noise_std": self.numeric_noise_std,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SourceWeightingConfig:
    enabled: bool = False
    method: str = "none"
    alpha: float = 0.0
    source_split_parts: tuple[str, ...] = ("train",)
    target_split_parts: tuple[str, ...] = ("finetune",)
    source_domains: tuple[str, ...] = ("aquatic",)
    target_domains: tuple[str, ...] = ("soil",)
    min_weight: float = 0.25
    max_weight: float = 2.0

    def active(self) -> bool:
        return self.enabled and self.method not in {"", "none", "off", "false"} and self.alpha > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "alpha": self.alpha,
            "source_split_parts": list(self.source_split_parts),
            "target_split_parts": list(self.target_split_parts),
            "source_domains": list(self.source_domains),
            "target_domains": list(self.target_domains),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class EffectLevelWeightingConfig:
    enabled: bool = False
    beta: float = 0.0
    split_parts: tuple[str, ...] = ("train", "finetune")
    task_prefixes: tuple[str, ...] = ("ECx", "LCx", "ICx", "LDx")
    min_weight: float = 0.5
    max_weight: float = 3.0

    def active(self) -> bool:
        return self.enabled and self.beta > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "beta": self.beta,
            "split_parts": list(self.split_parts),
            "task_prefixes": list(self.task_prefixes),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class DomainAlignmentConfig:
    enabled: bool = False
    method: str = "none"
    weight: float = 0.0
    source_split_parts: tuple[str, ...] = ("train",)
    target_split_parts: tuple[str, ...] = ("finetune",)
    source_domains: tuple[str, ...] = ("aquatic",)
    target_domains: tuple[str, ...] = ("soil",)
    phases: tuple[str, ...] = ("pretrain",)

    def active(self) -> bool:
        return self.enabled and self.method == "coral" and self.weight > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "weight": self.weight,
            "source_split_parts": list(self.source_split_parts),
            "target_split_parts": list(self.target_split_parts),
            "source_domains": list(self.source_domains),
            "target_domains": list(self.target_domains),
            "phases": list(self.phases),
        }


@dataclass(frozen=True)
class CensoredLossConfig:
    enabled: bool = False
    method: str = "hinge"
    weight: float = 0.0
    margin: float = 0.0
    split_parts: tuple[str, ...] = ("train", "finetune")
    include_ops: tuple[str, ...] = ("<", "<=", ">", ">=")

    def active(self) -> bool:
        return self.enabled and self.method == "hinge" and self.weight > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "weight": self.weight,
            "margin": self.margin,
            "split_parts": list(self.split_parts),
            "include_ops": list(self.include_ops),
        }


@dataclass(frozen=True)
class SwaConfig:
    enabled: bool = False
    phase: str = "finetune"
    start_epoch: int = 15

    def active(self) -> bool:
        return self.enabled and self.start_epoch > 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "start_epoch": self.start_epoch,
        }


@dataclass(frozen=True)
class ZScoreCorrectionConfig:
    enabled: bool = True
    threshold: float = 6.0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": "clip_standardized_zscore",
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class ZScoreCorrection:
    enabled: bool
    threshold: float
    feature_names: tuple[str, ...]
    fit_split_parts: tuple[str, ...]
    stats: dict[str, dict[str, float]]

    def transform(self, index: int, value: float) -> float:
        if not self.enabled:
            return float(value)
        if not math.isfinite(float(value)):
            return 0.0
        return max(-self.threshold, min(self.threshold, float(value)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": "clip_standardized_zscore",
            "threshold": self.threshold,
            "fit_split_parts": list(self.fit_split_parts),
            "feature_names": list(self.feature_names),
            "stats": self.stats,
        }


ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": AblationSpec("full"),
    "no_fingerprint": AblationSpec("no_fingerprint", use_fingerprint=False),
    "no_descriptors": AblationSpec("no_descriptors", use_descriptors=False),
    "descriptors_only": AblationSpec(
        "descriptors_only",
        use_fingerprint=False,
        use_context_numeric=False,
        use_duration_features=False,
        use_species_lifestage=False,
        use_other_categorical_context=False,
        use_medium_adapter=False,
    ),
    "no_species_lifestage": AblationSpec("no_species_lifestage", use_species_lifestage=False),
    "no_duration": AblationSpec("no_duration", use_duration_features=False),
    "no_context": AblationSpec(
        "no_context",
        use_context_numeric=False,
        use_duration_features=False,
        use_species_lifestage=False,
        use_other_categorical_context=False,
        use_medium_adapter=False,
    ),
    "no_molecular_residual": AblationSpec("no_molecular_residual", use_molecular_residual=False),
    "no_medium_adapter": AblationSpec("no_medium_adapter", use_medium_adapter=False),
}

SPECIES_LIFESTAGE_COLUMNS = frozenset(
    (
        "latin_name",
        "kingdom",
        "phylum",
        "class_name",
        "tax_order",
        "family",
        "genus",
        "species",
        "taxon_group_l1",
        "taxon_group_l2",
        "taxon_group_l3",
        "organism_lifestage",
    )
)
DURATION_CONTEXT_COLUMNS = frozenset(DURATION_NUMERIC_COLUMNS)


def run_deep_experiment(
    db_path: str | Path,
    *,
    split_name: str,
    out_dir: str | Path,
    config: Mapping[str, Any],
    limit: int | None = None,
    seed: int = 42,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    scheduler: str | None = None,
    device: str | None = None,
    ablation: str = "full",
    source_table: str | None = None,
    early_stopping: bool | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    validation_fraction: float | None = None,
    monitor_split: str | None = None,
    finetune_epochs: int | None = None,
    finetune_learning_rate: float | None = None,
    finetune_batch_size: int | None = None,
    finetune_scheduler: str | None = None,
    finetune_freeze: str | None = None,
    finetune_validation_fraction: float | None = None,
    weight_decay: float | None = None,
    dropout: float | None = None,
    target_standardization: str | None = None,
    augment_train_replicates: int | None = None,
    augment_finetune_replicates: int | None = None,
    augment_numeric_noise_std: float | None = None,
    augment_target_noise_std: float | None = None,
    test_noise_replicates: int | None = None,
    test_noise_numeric_std: float | None = None,
    feature_zscore_correction: bool | None = None,
    feature_zscore_threshold: float | None = None,
    metric_min_n: int | None = None,
    source_weighting_method: str | None = None,
    source_weighting_alpha: float | None = None,
    effect_level_weighting_enabled: bool | None = None,
    effect_level_weighting_beta: float | None = None,
    toxicity_binning_enabled: bool | None = None,
    toxicity_binning_mode: str | None = None,
    toxicity_binning_loss_weight: float | None = None,
    toxicity_binning_scheme: str | None = None,
    censored_loss_enabled: bool | None = None,
    censored_loss_weight: float | None = None,
    censored_loss_margin: float | None = None,
    domain_alignment_method: str | None = None,
    domain_alignment_weight: float | None = None,
    swa_enabled: bool | None = None,
    swa_start_epoch: int | None = None,
) -> DeepExperimentResult:
    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_spec = get_ablation_spec(ablation)
    if not bool(config.get("model", {}).get("use_medium_adapters", True)):
        ablation_spec = replace(ablation_spec, use_medium_adapter=False)

    frame = load_split_frame(db_path, split_name=split_name, source_table=source_table, limit=limit)
    split_join_audit = dict(frame.attrs.get("split_join_audit", {}))
    frame = add_duration_nonlinear_features(frame)
    target_column = "target_value_median" if "target_value_median" in frame.columns else "target_value"
    frame = frame[frame[target_column].notna()].copy()
    if frame.empty:
        raise ValueError("No target rows available for deep training.")

    filter_cfg = config.get("experiment", {}).get("task_filter", {})
    min_total = int(filter_cfg.get("min_total", 0))
    min_train = int(filter_cfg.get("min_train", 1))
    min_eval = int(filter_cfg.get("min_eval", 1))
    skipped_tasks: dict[str, str] = {}
    kept_frames = []
    for task_head, task_frame in frame.groupby("task_head", dropna=False):
        task_label = "default" if task_head is None else str(task_head)
        reason = task_skip_reason(
            task_frame,
            min_total=min_total,
            min_train=min_train,
            min_eval=min_eval,
        )
        if reason is not None:
            skipped_tasks[task_label] = reason
            continue
        kept_frames.append(task_frame)
    if not kept_frames:
        raise ValueError(f"No task heads passed sample thresholds. Skipped: {skipped_tasks}")
    frame = _concat_frames(kept_frames)

    train_cfg = config.get("training", {})
    reporting_cfg = config.get("reporting", {}) if isinstance(config.get("reporting", {}), dict) else {}
    metric_min_group_n = max(1, int(metric_min_n if metric_min_n is not None else reporting_cfg.get("min_metric_group_n", 5)))
    finetune_cfg = train_cfg.get("finetune", {}) if isinstance(train_cfg.get("finetune", {}), dict) else {}
    augmentation_cfg = _feature_noise_config(
        train_cfg,
        seed=seed,
        train_replicates_override=augment_train_replicates,
        finetune_replicates_override=augment_finetune_replicates,
        numeric_noise_std_override=augment_numeric_noise_std,
        target_noise_std_override=augment_target_noise_std,
    )
    perturbation_cfg = _perturbation_config(
        train_cfg,
        seed=seed,
        replicates_override=test_noise_replicates,
        numeric_noise_std_override=test_noise_numeric_std,
    )
    source_weighting_cfg = _source_weighting_config(
        train_cfg,
        method_override=source_weighting_method,
        alpha_override=source_weighting_alpha,
    )
    effect_level_weighting_cfg = _effect_level_weighting_config(
        train_cfg,
        enabled_override=effect_level_weighting_enabled,
        beta_override=effect_level_weighting_beta,
    )
    toxicity_binning_cfg = _toxicity_binning_config(
        train_cfg,
        enabled_override=toxicity_binning_enabled,
        mode_override=toxicity_binning_mode,
        loss_weight_override=toxicity_binning_loss_weight,
        scheme_override=toxicity_binning_scheme,
    )
    toxicity_bin_scheme = load_toxicity_bin_scheme(toxicity_binning_cfg.scheme)
    toxicity_bin_count = toxicity_bin_class_count(toxicity_bin_scheme) if toxicity_binning_cfg.enabled else 0
    censored_loss_cfg = _censored_loss_config(
        train_cfg,
        enabled_override=censored_loss_enabled,
        weight_override=censored_loss_weight,
        margin_override=censored_loss_margin,
    )
    domain_alignment_cfg = _domain_alignment_config(
        train_cfg,
        method_override=domain_alignment_method,
        weight_override=domain_alignment_weight,
    )
    swa_cfg = _swa_config(
        train_cfg,
        enabled_override=swa_enabled,
        start_epoch_override=swa_start_epoch,
    )
    requested_finetune_epochs = int(
        finetune_epochs if finetune_epochs is not None else finetune_cfg.get("epochs", 0)
    )
    finetune_requested = requested_finetune_epochs > 0
    finetune_freeze_mode = str(
        finetune_freeze if finetune_freeze is not None else finetune_cfg.get("freeze", "none")
    ).strip().lower()

    fingerprint_size = int(config.get("features", {}).get("molecule", {}).get("morgan_n_bits", 512))
    cache_path = _molecular_cache_path(config)
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size, cache_path=cache_path)
    descriptor_count = _descriptor_count(frame, encoder, ablation_spec)
    numeric_feature_names = build_numeric_feature_names(descriptor_count)

    early_cfg = _early_stopping_config(
        config,
        enabled_override=early_stopping,
        patience_override=early_stopping_patience,
        min_delta_override=early_stopping_min_delta,
        validation_fraction_override=validation_fraction,
        monitor_split_override=monitor_split,
    )
    split_probe_samples = [
        {"split_part": str(row.get("split_part")), "task_head": str(row.get("task_head"))}
        for _, row in frame.iterrows()
    ]
    train_indices = [idx for idx, sample in enumerate(split_probe_samples) if sample["split_part"] == "train"]
    if not train_indices:
        raise ValueError("No train samples found for deep experiment.")
    actual_train_indices, validation_indices, validation_source = split_training_validation_indices(
        split_probe_samples,
        train_indices=train_indices,
        seed=seed,
        validation_fraction=early_cfg["validation_fraction"],
        monitor_split=early_cfg["monitor_split"],
    )
    early_enabled = bool(early_cfg["enabled"] and validation_indices)
    if not early_enabled:
        actual_train_indices = train_indices
        validation_indices = []
        validation_source = ""

    finetune_indices = [idx for idx, sample in enumerate(split_probe_samples) if sample["split_part"] == "finetune"]
    finetune_train_indices, finetune_validation_indices, finetune_validation_source = split_finetune_validation_indices(
        split_probe_samples,
        finetune_indices=finetune_indices,
        seed=seed,
        validation_fraction=float(
            finetune_validation_fraction
            if finetune_validation_fraction is not None
            else finetune_cfg.get("validation_fraction", 0.0)
        ),
        monitor_split=str(finetune_cfg.get("monitor_split", "auto")),
    )
    preprocessing_indices = list(actual_train_indices)
    if finetune_requested:
        preprocessing_indices.extend(finetune_train_indices)
    preprocessing_frame = frame.iloc[sorted(set(preprocessing_indices))].copy()
    category_min_count = int(
        train_cfg.get(
            "categorical_min_count",
            config.get("features", {}).get("categorical_min_count", 1),
        )
    )
    categorical_maps = fit_categorical_maps(
        preprocessing_frame,
        ablation=ablation_spec,
        min_count=category_min_count,
    )
    adapter_map = fit_adapter_map(preprocessing_frame, ablation=ablation_spec)
    numeric_stats = fit_numeric_stats(preprocessing_frame, encoder, ablation=ablation_spec)
    zscore_cfg = _zscore_correction_config(
        train_cfg,
        enabled_override=feature_zscore_correction,
        threshold_override=feature_zscore_threshold,
    )
    zscore_correction = fit_zscore_correction(
        preprocessing_frame,
        encoder,
        numeric_stats=numeric_stats,
        feature_names=numeric_feature_names,
        config=zscore_cfg,
        ablation=ablation_spec,
    )
    target_scaler = fit_target_scaler(
        frame,
        target_column=target_column,
        mode=str(target_standardization or train_cfg.get("target_standardization", "per_task_target")),
        fit_indices=preprocessing_indices,
    )
    samples = build_deep_samples(
        frame,
        encoder=encoder,
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        target_column=target_column,
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        ablation=ablation_spec,
        toxicity_binning_config=toxicity_binning_cfg,
        toxicity_bin_scheme=toxicity_bin_scheme,
    )
    censored_summary = {
        **censored_loss_cfg.to_manifest(),
        "candidate_rows": 0,
        "usable_rows": 0,
        "train_rows": 0,
        "finetune_rows": 0,
        "skipped_rows": {},
    }
    mark_internal_validation_samples(
        samples,
        indices=finetune_validation_indices,
        split_part="finetune_validation",
    )
    if censored_loss_cfg.active():
        censored_samples, censored_summary = build_censored_training_samples(
            db_path,
            frame,
            encoder=encoder,
            categorical_maps=categorical_maps,
            adapter_map=adapter_map,
            numeric_stats=numeric_stats,
            target_column=target_column,
            target_scaler=target_scaler,
            zscore_correction=zscore_correction,
            ablation=ablation_spec,
            config=censored_loss_cfg,
            kept_task_heads=tuple(sorted(set(str(sample.get("task_head")) for sample in samples))),
            split_parts=tuple(sorted({str(sample.get("split_part")) for sample in samples})),
        )
        censored_start = len(samples)
        samples.extend(censored_samples)
        for offset, sample in enumerate(censored_samples):
            idx = censored_start + offset
            split_part = str(sample.get("split_part", "")).lower()
            if split_part == "train":
                actual_train_indices.append(idx)
                train_indices.append(idx)
            elif split_part == "finetune":
                finetune_indices.append(idx)
                finetune_train_indices.append(idx)
    source_weighting_summary = apply_source_similarity_weights(samples, source_weighting_cfg)
    effect_level_weighting_summary = apply_effect_level_frequency_weights(
        samples,
        effect_level_weighting_cfg,
        train_indices=actual_train_indices + finetune_train_indices,
    )
    weighting_history_fields = sample_weighting_history_fields(
        source_weighting_summary,
        effect_level_weighting_summary,
    )
    toxicity_binning_summary = summarize_toxicity_bins(samples, toxicity_binning_cfg, toxicity_bin_count)
    dataset = AggregatedTaskDataset(samples=samples, fingerprint_size=fingerprint_size)

    train_dataset = build_noisy_index_dataset(
        dataset,
        actual_train_indices,
        replicates=augmentation_cfg.train_replicates,
        numeric_noise_std=augmentation_cfg.numeric_noise_std,
        target_noise_std=augmentation_cfg.target_noise_std,
        seed=augmentation_cfg.seed,
    )
    validation_dataset = _IndexDataset(dataset, validation_indices) if validation_indices else None
    task_heads = dataset.task_heads()
    categorical_cardinalities = {
        column: max(mapping.values(), default=0) + 1
        for column, mapping in categorical_maps.items()
    }
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=dataset.numeric_dim(),
            fingerprint_dim=dataset.fingerprint_dim(),
            categorical_cardinalities=categorical_cardinalities,
            adapter_count=max(adapter_map.values(), default=0) + 1 if adapter_map else 0,
            effect_level_numeric_indices=effect_level_feature_indices(numeric_feature_names),
            task_heads=task_heads,
            hidden_dims=_hidden_dims(config),
            dropout=float(dropout if dropout is not None else config.get("model", {}).get("dropout", 0.15)),
            use_molecular_residual=ablation_spec.use_molecular_residual,
            use_adapters=ablation_spec.use_medium_adapter,
            toxicity_bin_count=toxicity_bin_count,
            toxicity_binning_mode=toxicity_binning_cfg.mode if toxicity_binning_cfg.enabled else "none",
        )
    )

    loss_cfg = train_cfg.get("loss", {}) if isinstance(train_cfg.get("loss", {}), dict) else {}
    task_weights = resolve_task_weights(
        samples,
        train_indices=actual_train_indices,
        train_cfg=train_cfg,
        task_heads=task_heads,
    )
    train_config = DeepTrainingConfig(
        epochs=int(epochs or train_cfg.get("epochs", 5)),
        batch_size=int(batch_size or train_cfg.get("batch_size", 256)),
        learning_rate=float(learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 3e-4)),
        huber_delta=float(loss_cfg.get("delta", 1.0)),
        task_weights=task_weights,
        optimizer=str(train_cfg.get("optimizer", "adamw")),
        weight_decay=float(weight_decay if weight_decay is not None else train_cfg.get("weight_decay", 1e-4)),
        gradient_clip_norm=_optional_float(train_cfg.get("gradient_clip_norm", 5.0)),
        scheduler=str(scheduler if scheduler is not None else train_cfg.get("scheduler", "none")),
        device=_resolve_device(device or train_cfg.get("device", "cpu")),
        seed=seed,
        num_workers=int(train_cfg.get("num_workers", 0)),
        toxicity_bin_loss_weight=toxicity_binning_cfg.loss_weight if toxicity_binning_cfg.active() else 0.0,
        toxicity_binning_mode=toxicity_binning_cfg.mode,
        censored_loss_weight=censored_loss_cfg.weight if censored_loss_cfg.active() else 0.0,
        censored_loss_margin=censored_loss_cfg.margin,
    )
    finetune_config = DeepTrainingConfig(
        epochs=requested_finetune_epochs,
        batch_size=int(finetune_batch_size or finetune_cfg.get("batch_size", train_config.batch_size)),
        learning_rate=float(
            finetune_learning_rate
            if finetune_learning_rate is not None
            else finetune_cfg.get("learning_rate", train_config.learning_rate * 0.2)
        ),
        huber_delta=train_config.huber_delta,
        task_weights=train_config.task_weights,
        optimizer=str(finetune_cfg.get("optimizer", train_config.optimizer)),
        weight_decay=float(finetune_cfg.get("weight_decay", train_config.weight_decay)),
        gradient_clip_norm=_optional_float(finetune_cfg.get("gradient_clip_norm", train_config.gradient_clip_norm)),
        scheduler=str(finetune_scheduler if finetune_scheduler is not None else finetune_cfg.get("scheduler", train_config.scheduler)),
        device=train_config.device,
        seed=seed,
        num_workers=train_config.num_workers,
        toxicity_bin_loss_weight=train_config.toxicity_bin_loss_weight,
        toxicity_binning_mode=train_config.toxicity_binning_mode,
        censored_loss_weight=train_config.censored_loss_weight,
        censored_loss_margin=train_config.censored_loss_margin,
    )

    set_torch_seed(seed)
    torch_device = torch.device(train_config.device)
    model.to(torch_device)
    optimizer = build_optimizer(model.parameters(), train_config)
    scheduler = build_scheduler(optimizer, train_config, train_config.epochs)
    loss_fn = torch.nn.HuberLoss(delta=train_config.huber_delta, reduction="mean")
    swa_model: Any | None = None
    swa_updates = 0
    swa_last_global_epoch = 0
    swa_applied = False
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_aggregated_task_batch,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            collate_fn=collate_aggregated_task_batch,
        )
        if validation_dataset is not None
        else None
    )
    domain_alignment_summary = {
        **domain_alignment_cfg.to_manifest(),
        "pretrain_reference_samples": 0,
        "finetune_reference_samples": 0,
        "pretrain_alignment_steps": 0,
        "finetune_alignment_steps": 0,
    }
    pretrain_alignment_indices = domain_alignment_reference_indices(
        samples,
        domain_alignment_cfg,
        phase="pretrain",
    )
    domain_alignment_summary["pretrain_reference_samples"] = len(pretrain_alignment_indices)
    pretrain_alignment_batches = (
        cycle_dataloader(
            DataLoader(
                _IndexDataset(dataset, pretrain_alignment_indices),
                batch_size=train_config.batch_size,
                shuffle=True,
                num_workers=train_config.num_workers,
                collate_fn=collate_aggregated_task_batch,
            )
        )
        if pretrain_alignment_indices
        else None
    )

    history = []
    best_epoch = 0
    best_monitor_loss = float("inf")
    best_state: dict[str, Any] | None = None
    no_improve_epochs = 0
    for epoch in range(1, train_config.epochs + 1):
        epoch_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            train_config,
            torch_device,
            alignment_batches=pretrain_alignment_batches,
            alignment_weight=domain_alignment_cfg.weight if pretrain_alignment_batches is not None else 0.0,
        )
        domain_alignment_summary["pretrain_alignment_steps"] += int(epoch_loss.get("alignment_steps", 0))
        validation_loss = (
            evaluate_loss(model, validation_loader, loss_fn, train_config, torch_device)
            if validation_loader is not None
            else None
        )
        monitor_loss = validation_loss["mean_loss"] if validation_loss is not None else epoch_loss["mean_loss"]
        if early_enabled:
            improved = monitor_loss < best_monitor_loss - float(early_cfg["min_delta"])
            if improved:
                best_monitor_loss = monitor_loss
                best_epoch = epoch
                no_improve_epochs = 0
                best_state = clone_state_dict(model)
            else:
                no_improve_epochs += 1
        else:
            best_monitor_loss = monitor_loss
            best_epoch = epoch
            no_improve_epochs = 0
            best_state = clone_state_dict(model)
        row = {
            "phase": "pretrain",
            "epoch": epoch,
            "global_epoch": epoch,
            **weighting_history_fields,
            **epoch_loss,
            "validation_loss": "" if validation_loss is None else validation_loss["mean_loss"],
            "validation_task_loss": "" if validation_loss is None else validation_loss.get("mean_task_loss", ""),
            "validation_toxicity_bin_loss": "" if validation_loss is None else validation_loss.get("mean_toxicity_bin_loss", ""),
            "validation_toxicity_bin_samples": 0 if validation_loss is None else validation_loss.get("toxicity_bin_samples", 0),
            "validation_censored_loss": "" if validation_loss is None else validation_loss.get("mean_censored_loss", ""),
            "validation_censored_samples": 0 if validation_loss is None else validation_loss.get("censored_samples", 0),
            "validation_samples": 0 if validation_loss is None else validation_loss["samples"],
            "monitor_loss": monitor_loss,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "learning_rate": current_learning_rate(optimizer),
            "swa_updates": swa_updates,
        }
        if should_update_swa(swa_cfg, phase="pretrain", epoch=epoch):
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)
            swa_updates += 1
            swa_last_global_epoch = epoch
            row["swa_updates"] = swa_updates
        history.append(row)
        step_scheduler(scheduler, monitor_loss)
        validation_msg = "" if validation_loss is None else f" validation_loss={validation_loss['mean_loss']:.6f}"
        print(
            f"[epoch {epoch}] mean_loss={epoch_loss['mean_loss']:.6f}{validation_msg} "
            f"samples={epoch_loss['samples']} best_epoch={best_epoch}",
            flush=True,
        )
        if early_enabled and no_improve_epochs >= int(early_cfg["patience"]):
            print(
                f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                f"monitor_loss={best_monitor_loss:.6f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    finetune_ran = 0
    finetune_best_epoch = 0
    finetune_best_monitor_loss = float("inf")
    finetune_no_improve_epochs = 0
    finetune_early_enabled = bool(
        finetune_validation_indices and bool(finetune_cfg.get("early_stopping", True))
    )
    finetune_patience = int(finetune_cfg.get("early_stopping_patience", early_cfg["patience"]))
    finetune_min_delta = float(finetune_cfg.get("early_stopping_min_delta", early_cfg["min_delta"]))
    if finetune_requested and finetune_train_indices:
        trainable_parameters = apply_finetune_freeze(model, finetune_freeze_mode)
        finetune_dataset = build_noisy_index_dataset(
            dataset,
            finetune_train_indices,
            replicates=augmentation_cfg.finetune_replicates,
            numeric_noise_std=augmentation_cfg.numeric_noise_std,
            target_noise_std=augmentation_cfg.target_noise_std,
            seed=augmentation_cfg.seed + 100_000,
        )
        finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=finetune_config.batch_size,
            shuffle=True,
            num_workers=finetune_config.num_workers,
            collate_fn=collate_aggregated_task_batch,
        )
        finetune_validation_loader = (
            DataLoader(
                _IndexDataset(dataset, finetune_validation_indices),
                batch_size=finetune_config.batch_size,
                shuffle=False,
                num_workers=finetune_config.num_workers,
                collate_fn=collate_aggregated_task_batch,
            )
            if finetune_validation_indices
            else None
        )
        finetune_alignment_indices = domain_alignment_reference_indices(
            samples,
            domain_alignment_cfg,
            phase="finetune",
        )
        domain_alignment_summary["finetune_reference_samples"] = len(finetune_alignment_indices)
        finetune_alignment_batches = (
            cycle_dataloader(
                DataLoader(
                    _IndexDataset(dataset, finetune_alignment_indices),
                    batch_size=finetune_config.batch_size,
                    shuffle=True,
                    num_workers=finetune_config.num_workers,
                    collate_fn=collate_aggregated_task_batch,
                )
            )
            if finetune_alignment_indices
            else None
        )
        optimizer = build_optimizer(trainable_parameters, finetune_config)
        scheduler = build_scheduler(optimizer, finetune_config, finetune_config.epochs)
        finetune_best_state: dict[str, Any] | None = None
        for epoch in range(1, finetune_config.epochs + 1):
            epoch_loss = train_one_epoch(
                model,
                finetune_loader,
                optimizer,
                loss_fn,
                finetune_config,
                torch_device,
                alignment_batches=finetune_alignment_batches,
                alignment_weight=domain_alignment_cfg.weight if finetune_alignment_batches is not None else 0.0,
            )
            domain_alignment_summary["finetune_alignment_steps"] += int(epoch_loss.get("alignment_steps", 0))
            finetune_ran = epoch
            global_epoch = len(history) + 1
            finetune_validation_loss = (
                evaluate_loss(model, finetune_validation_loader, loss_fn, finetune_config, torch_device)
                if finetune_validation_loader is not None
                else None
            )
            monitor_loss = (
                finetune_validation_loss["mean_loss"]
                if finetune_validation_loss is not None
                else epoch_loss["mean_loss"]
            )
            if finetune_early_enabled:
                improved = monitor_loss < finetune_best_monitor_loss - finetune_min_delta
                if improved:
                    finetune_best_monitor_loss = monitor_loss
                    finetune_best_epoch = global_epoch
                    finetune_no_improve_epochs = 0
                    finetune_best_state = clone_state_dict(model)
                else:
                    finetune_no_improve_epochs += 1
            else:
                finetune_best_monitor_loss = monitor_loss
                finetune_best_epoch = global_epoch
                finetune_no_improve_epochs = 0
                finetune_best_state = clone_state_dict(model)
            row = {
                "phase": "finetune",
                "epoch": epoch,
                "global_epoch": global_epoch,
                **weighting_history_fields,
                **epoch_loss,
                "validation_loss": "" if finetune_validation_loss is None else finetune_validation_loss["mean_loss"],
                "validation_task_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_task_loss", ""),
                "validation_toxicity_bin_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_toxicity_bin_loss", ""),
                "validation_toxicity_bin_samples": 0 if finetune_validation_loss is None else finetune_validation_loss.get("toxicity_bin_samples", 0),
                "validation_censored_loss": "" if finetune_validation_loss is None else finetune_validation_loss.get("mean_censored_loss", ""),
                "validation_censored_samples": 0 if finetune_validation_loss is None else finetune_validation_loss.get("censored_samples", 0),
                "validation_samples": 0 if finetune_validation_loss is None else finetune_validation_loss["samples"],
                "monitor_loss": monitor_loss,
                "best_epoch": finetune_best_epoch,
                "no_improve_epochs": finetune_no_improve_epochs,
                "learning_rate": current_learning_rate(optimizer),
                "swa_updates": swa_updates,
            }
            if should_update_swa(swa_cfg, phase="finetune", epoch=epoch):
                if swa_model is None:
                    swa_model = torch.optim.swa_utils.AveragedModel(model)
                swa_model.update_parameters(model)
                swa_updates += 1
                swa_last_global_epoch = global_epoch
                row["swa_updates"] = swa_updates
            history.append(row)
            step_scheduler(scheduler, monitor_loss)
            best_epoch = finetune_best_epoch
            best_monitor_loss = finetune_best_monitor_loss
            best_state = finetune_best_state
            validation_msg = (
                ""
                if finetune_validation_loss is None
                else f" validation_loss={finetune_validation_loss['mean_loss']:.6f}"
            )
            print(
                f"[finetune epoch {epoch}] mean_loss={epoch_loss['mean_loss']:.6f}{validation_msg} "
                f"samples={epoch_loss['samples']} best_epoch={best_epoch}",
                flush=True,
            )
            if finetune_early_enabled and finetune_no_improve_epochs >= finetune_patience:
                print(
                    f"[finetune early-stop] epoch={epoch} best_epoch={finetune_best_epoch} "
                    f"monitor_loss={finetune_best_monitor_loss:.6f}",
                    flush=True,
                )
                break
        if finetune_best_state is not None:
            model.load_state_dict(finetune_best_state)

    if swa_model is not None and swa_updates > 0:
        model.load_state_dict(swa_model.module.state_dict())
        best_state = clone_state_dict(model)
        best_epoch = swa_last_global_epoch or best_epoch
        swa_applied = True

    predictions = predict_all(
        model,
        dataset,
        samples,
        batch_size=train_config.batch_size,
        device=torch_device,
        target_scaler=target_scaler,
    )
    metrics_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        min_n_for_summary=metric_min_group_n,
    )
    filtered_metric_rows = filter_summary_metric_rows(metrics_rows)
    effect_level_metric_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        group_columns=("split_part", "task_head", "target_name", "medium_domain", "effect_level_x"),
        min_n_for_summary=metric_min_group_n,
    )
    filtered_effect_level_metric_rows = filter_summary_metric_rows(effect_level_metric_rows)
    toxicity_bin_metric_rows = metrics_by_group(
        predictions,
        huber_delta=train_config.huber_delta,
        group_columns=(
            "split_part",
            "task_head",
            "target_name",
            "medium_domain",
            "toxicity_bin_status",
            "toxicity_bin_label",
            "toxicity_bin_boundary_flag",
        ),
        min_n_for_summary=metric_min_group_n,
    )
    toxicity_bin_boundary_audit_rows = build_toxicity_bin_boundary_audit_rows(predictions)
    split_medium_audit_rows = build_split_medium_audit_rows(frame, split_join_audit=split_join_audit)
    perturbation_predictions: list[dict[str, Any]] = []
    perturbation_summary_rows: list[dict[str, Any]] = []
    perturbation_metric_rows: list[dict[str, Any]] = []
    if perturbation_cfg.active():
        perturbation_predictions = predict_with_feature_perturbation(
            model,
            dataset,
            samples,
            split_parts=perturbation_cfg.split_parts,
            replicates=perturbation_cfg.replicates,
            numeric_noise_std=perturbation_cfg.numeric_noise_std,
            seed=perturbation_cfg.seed,
            batch_size=train_config.batch_size,
            device=torch_device,
            target_scaler=target_scaler,
        )
        perturbation_summary_rows = summarize_perturbation_predictions(perturbation_predictions)
        perturbation_metric_rows = metrics_by_group(
            perturbation_mean_predictions(perturbation_summary_rows),
            huber_delta=train_config.huber_delta,
            min_n_for_summary=metric_min_group_n,
        )

    history_path = output_dir / "history.csv"
    metrics_path = output_dir / "metrics.csv"
    metrics_filtered_path = output_dir / "metrics_filtered.csv"
    effect_level_metrics_path = output_dir / "effect_level_metrics.csv"
    effect_level_metrics_filtered_path = output_dir / "effect_level_metrics_filtered.csv"
    toxicity_bin_metrics_path = output_dir / "toxicity_bin_metrics.csv"
    toxicity_bin_boundary_audit_path = output_dir / "toxicity_bin_boundary_audit.csv"
    split_medium_audit_path = output_dir / "split_medium_audit.csv"
    predictions_path = output_dir / "predictions.csv"
    perturbation_predictions_path = output_dir / "test_noise_predictions.csv"
    perturbation_summary_path = output_dir / "test_noise_summary.csv"
    perturbation_metrics_path = output_dir / "test_noise_metrics.csv"
    manifest_path = output_dir / "manifest.json"
    preprocessing_path = output_dir / "preprocessing.json"
    best_model_path = output_dir / "best_model.pt"
    write_rows(history_path, history)
    write_rows(metrics_path, metrics_rows)
    write_rows(metrics_filtered_path, filtered_metric_rows)
    write_rows(effect_level_metrics_path, effect_level_metric_rows)
    write_rows(effect_level_metrics_filtered_path, filtered_effect_level_metric_rows)
    write_rows(toxicity_bin_metrics_path, toxicity_bin_metric_rows)
    write_rows(toxicity_bin_boundary_audit_path, toxicity_bin_boundary_audit_rows)
    write_rows(split_medium_audit_path, split_medium_audit_rows)
    write_rows(predictions_path, predictions)
    if perturbation_predictions:
        write_rows(perturbation_predictions_path, perturbation_predictions)
        write_rows(perturbation_summary_path, perturbation_summary_rows)
        write_rows(perturbation_metrics_path, perturbation_metric_rows)
    if best_state is not None:
        torch.save(best_state, best_model_path)
    else:
        best_model_path = None
    manifest = {
        "split_name": split_name,
        "rows": len(samples),
        "train_rows": len(train_indices),
        "actual_train_rows": len(actual_train_indices),
        "finetune_rows": len(finetune_indices),
        "finetune_train_rows": len(finetune_train_indices),
        "finetune_validation_rows": len(finetune_validation_indices),
        "finetune_validation_source": finetune_validation_source,
        "validation_rows": len(validation_indices),
        "validation_source": validation_source,
        "task_heads": list(task_heads),
        "skipped_tasks": skipped_tasks,
        "fingerprint_size": fingerprint_size,
        "encoder_source": encoder.source,
        "device": train_config.device,
        "epochs": train_config.epochs,
        "finetune_epochs": finetune_config.epochs,
        "finetune_epochs_ran": finetune_ran,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_monitor_loss": best_monitor_loss if math.isfinite(best_monitor_loss) else None,
        "early_stopping": {
            "enabled": early_enabled,
            "requested": bool(early_cfg["enabled"]),
            "patience": int(early_cfg["patience"]),
            "min_delta": float(early_cfg["min_delta"]),
            "validation_fraction": float(early_cfg["validation_fraction"]),
            "monitor_split": early_cfg["monitor_split"],
        },
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "optimizer": train_config.optimizer,
        "weight_decay": train_config.weight_decay,
        "gradient_clip_norm": train_config.gradient_clip_norm,
        "scheduler": train_config.scheduler,
        "target_standardization": target_scaler.to_manifest(),
        "feature_zscore_correction": zscore_correction.to_manifest(),
        "metric_reporting": {
            "min_metric_group_n": metric_min_group_n,
            "filtered_metrics_path": str(metrics_filtered_path),
            "effect_level_metrics_path": str(effect_level_metrics_path),
            "effect_level_metrics_filtered_path": str(effect_level_metrics_filtered_path),
            "toxicity_bin_metrics_path": str(toxicity_bin_metrics_path),
            "toxicity_bin_boundary_audit_path": str(toxicity_bin_boundary_audit_path),
        },
        "split_join_audit": split_join_audit,
        "split_medium_audit_path": str(split_medium_audit_path),
        "augmentation": augmentation_cfg.to_manifest(),
        "test_perturbation": perturbation_cfg.to_manifest(),
        "source_weighting": source_weighting_summary,
        "effect_level_weighting": effect_level_weighting_summary,
        "toxicity_binning": toxicity_binning_summary,
        "censored_loss": censored_summary,
        "domain_alignment": domain_alignment_summary,
        "swa": {
            **swa_cfg.to_manifest(),
            "updates": swa_updates,
            "last_global_epoch": swa_last_global_epoch,
            "applied": swa_applied,
        },
        "task_weighting": str(train_cfg.get("task_weighting", "balanced")),
        "task_weights": train_config.task_weights,
        "categorical_min_count": category_min_count,
        "finetune_batch_size": finetune_config.batch_size,
        "finetune_learning_rate": finetune_config.learning_rate,
        "finetune": {
            "requested": finetune_requested,
            "enabled": bool(finetune_requested and finetune_indices),
            "epochs": finetune_config.epochs,
            "epochs_ran": finetune_ran,
            "rows": len(finetune_indices),
            "train_rows": len(finetune_train_indices),
            "validation_rows": len(finetune_validation_indices),
            "validation_source": finetune_validation_source,
            "early_stopping": finetune_early_enabled,
            "early_stopping_patience": finetune_patience,
            "early_stopping_min_delta": finetune_min_delta,
            "learning_rate": finetune_config.learning_rate,
            "batch_size": finetune_config.batch_size,
            "freeze": finetune_freeze_mode,
        },
        "numeric_dim": dataset.numeric_dim(),
        "fingerprint_dim": dataset.fingerprint_dim(),
        "categorical_cardinalities": categorical_cardinalities,
        "adapter_cardinality": max(adapter_map.values(), default=0) + 1 if adapter_map else 0,
        "adapter_map": adapter_map,
        "ablation": ablation_spec.name,
        "ablation_features": {
            "use_descriptors": ablation_spec.use_descriptors,
            "use_fingerprint": ablation_spec.use_fingerprint,
            "use_context_numeric": ablation_spec.use_context_numeric,
            "use_duration_features": ablation_spec.use_duration_features,
            "use_species_lifestage": ablation_spec.use_species_lifestage,
            "use_other_categorical_context": ablation_spec.use_other_categorical_context,
            "use_molecular_residual": ablation_spec.use_molecular_residual,
            "use_medium_adapter": ablation_spec.use_medium_adapter,
        },
    }
    preprocessing = build_preprocessing_manifest(
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        ablation=ablation_spec,
        fingerprint_size=fingerprint_size,
        encoder_source=encoder.source,
        cache_path=cache_path,
        descriptor_count=descriptor_count,
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        categorical_min_count=category_min_count,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing_path.write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")

    return DeepExperimentResult(
        out_dir=output_dir,
        metrics_path=metrics_path,
        history_path=history_path,
        manifest_path=manifest_path,
        preprocessing_path=preprocessing_path,
        predictions_path=predictions_path,
        best_model_path=best_model_path,
        encoder_source=encoder.source,
        trained_tasks=task_heads,
        rows=len(samples),
        ablation=ablation_spec.name,
        best_epoch=best_epoch,
        early_stopping_enabled=early_enabled,
    )


def get_ablation_spec(name: str | None) -> AblationSpec:
    key = (name or "full").strip().lower()
    if key not in ABLATION_SPECS:
        allowed = ", ".join(sorted(ABLATION_SPECS))
        raise ValueError(f"Unknown ablation '{name}'. Allowed values: {allowed}")
    return ABLATION_SPECS[key]


def build_preprocessing_manifest(
    *,
    categorical_maps: dict[str, dict[str, int]],
    adapter_map: dict[str, int] | None = None,
    numeric_stats: dict[str, tuple[float, float]],
    ablation: AblationSpec,
    fingerprint_size: int,
    encoder_source: str,
    cache_path: str | None,
    descriptor_count: int | None = None,
    target_scaler: TargetScaler | None = None,
    zscore_correction: ZScoreCorrection | None = None,
    categorical_min_count: int = 1,
) -> dict[str, Any]:
    descriptor_names = _descriptor_feature_names(descriptor_count or len(MOLECULAR_DESCRIPTOR_NAMES))
    numeric_feature_names = build_numeric_feature_names(descriptor_count or len(MOLECULAR_DESCRIPTOR_NAMES))
    numeric_feature_names = numeric_feature_names[: len(numeric_stats)]
    return {
        "schema_version": 2,
        "ablation": ablation.name,
        "fingerprint_size": int(fingerprint_size),
        "encoder_source": encoder_source,
        "molecular_feature_cache": cache_path or "",
        "target_standardization": {} if target_scaler is None else target_scaler.to_manifest(),
        "feature_zscore_correction": {} if zscore_correction is None else zscore_correction.to_manifest(),
        "numeric_feature_names": numeric_feature_names,
        "molecular_descriptor_names": descriptor_names,
        "context_numeric_columns": list(CONTEXT_NUMERIC_COLUMNS),
        "effect_level_numeric_columns": list(EFFECT_LEVEL_NUMERIC_COLUMNS),
        "effect_level_numeric_indices": list(effect_level_feature_indices(numeric_feature_names)),
        "duration_context_columns": sorted(DURATION_CONTEXT_COLUMNS),
        "categorical_columns": list(active_categorical_columns(ablation)),
        "categorical_min_count": int(categorical_min_count),
        "categorical_special_tokens": {
            "missing": MISSING_CATEGORY_TOKEN,
            "unknown": UNKNOWN_CATEGORY_TOKEN,
            "rare": RARE_CATEGORY_TOKEN,
        },
        "categorical_maps": categorical_maps,
        "adapter_columns": ["medium_domain", "target_family_or_target_name"],
        "adapter_map": adapter_map or {},
        "adapter_cardinality": max((adapter_map or {}).values(), default=0) + 1 if adapter_map else 0,
        "numeric_stats": {
            name: {
                "index": idx,
                "mean": float(numeric_stats[str(idx)][0]),
                "std": float(numeric_stats[str(idx)][1]),
            }
            for idx, name in enumerate(numeric_feature_names)
        },
        "ablation_features": {
            "use_descriptors": ablation.use_descriptors,
            "use_fingerprint": ablation.use_fingerprint,
            "use_context_numeric": ablation.use_context_numeric,
            "use_duration_features": ablation.use_duration_features,
            "use_species_lifestage": ablation.use_species_lifestage,
            "use_other_categorical_context": ablation.use_other_categorical_context,
            "use_molecular_residual": ablation.use_molecular_residual,
            "use_medium_adapter": ablation.use_medium_adapter,
        },
    }


def _descriptor_count(frame: Any, encoder: MolecularFeatureBuilder, ablation: AblationSpec) -> int:
    for value in frame.get("smiles", []):
        descriptors = masked_descriptors(encoder.encode(value)[0], ablation)
        return len(descriptors)
    return len(MOLECULAR_DESCRIPTOR_NAMES)


def _descriptor_feature_names(count: int) -> list[str]:
    if count == len(MOLECULAR_DESCRIPTOR_NAMES):
        return list(MOLECULAR_DESCRIPTOR_NAMES)
    return [f"descriptor_{idx}" for idx in range(count)]


def build_numeric_feature_names(descriptor_count: int) -> tuple[str, ...]:
    return tuple(_descriptor_feature_names(descriptor_count) + list(CONTEXT_NUMERIC_COLUMNS))


def effect_level_feature_indices(feature_names: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    effect_columns = set(EFFECT_LEVEL_NUMERIC_COLUMNS)
    return tuple(idx for idx, name in enumerate(feature_names) if name in effect_columns)


def apply_finetune_freeze(model: Any, mode: str) -> list[Any]:
    normalized = (mode or "none").strip().lower()
    if normalized in {"", "none", "all"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    for parameter in model.parameters():
        parameter.requires_grad = False

    if normalized == "heads_only":
        modules = [model.heads]
    elif normalized == "heads_embeddings":
        modules = [model.heads, model.embeddings, model.adapters]
    else:
        allowed = "none, heads_only, heads_embeddings"
        raise ValueError(f"Unsupported finetune freeze mode '{mode}'. Allowed values: {allowed}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters for finetune freeze mode '{mode}'.")
    return trainable


class MolecularFeatureBuilder:
    def __init__(self, *, fingerprint_size: int = 512, cache_path: str | Path | None = None) -> None:
        self.fingerprint_size = fingerprint_size
        self.cache = load_molecular_feature_cache(cache_path, fingerprint_size=fingerprint_size) if cache_path else {}
        self._rdkit_available = self._check_rdkit()
        if self.cache:
            self.source = "rdkit_cache"
        elif self._rdkit_available:
            self.source = "rdkit"
        else:
            self.source = "stable_smiles_fallback"

    @staticmethod
    def _check_rdkit() -> bool:
        try:
            import rdkit  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _normalize_smiles(smiles: object) -> str:
        if smiles is None:
            return ""
        try:
            if isinstance(smiles, (float, np.floating)) and not math.isfinite(float(smiles)):
                return ""
        except (TypeError, ValueError):
            pass
        text = str(smiles).strip()
        if text.lower() in {"", "nan", "none", "null", "na", "n/a", "<na>"}:
            return ""
        return text

    def encode(self, smiles: object) -> tuple[list[float], list[float]]:
        text = self._normalize_smiles(smiles)
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        if self._rdkit_available and text.strip():
            try:
                return self._encode_rdkit(text)
            except Exception:
                return self._encode_fallback(text)
        return self._encode_fallback(text)

    def _encode_rdkit(self, smiles: str) -> tuple[list[float], list[float]]:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._encode_fallback(smiles)
        descriptors = [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.HeavyAtomCount(mol)),
            float(Descriptors.NumHAcceptors(mol)),
            float(Descriptors.NumHDonors(mol)),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(Descriptors.NumRotatableBonds(mol)),
        ]
        generator = GetMorganGenerator(radius=2, fpSize=self.fingerprint_size)
        fingerprint = generator.GetFingerprint(mol)
        return descriptors, [float(bit) for bit in fingerprint.ToBitString()]

    def _encode_fallback(self, smiles: str) -> tuple[list[float], list[float]]:
        text = smiles or ""
        counts = Counter(text)
        descriptors = [
            float(len(text)),
            float(sum(ch.isupper() for ch in text)),
            float(sum(ch.islower() for ch in text)),
            float(counts.get("C", 0)),
            float(counts.get("N", 0)),
            float(counts.get("O", 0)),
            float(counts.get("S", 0)),
            float(counts.get("P", 0)),
        ]
        bits = [0.0] * self.fingerprint_size
        for ngram in _smiles_ngrams(text):
            digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).hexdigest()
            bits[int(digest, 16) % self.fingerprint_size] = 1.0
        return descriptors, bits


def _molecular_cache_path(config: Mapping[str, Any]) -> str | None:
    value = config.get("experiment", {}).get("molecular_feature_cache")
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def load_molecular_feature_cache(
    cache_path: str | Path | None,
    *,
    fingerprint_size: int,
) -> dict[str, tuple[list[float], list[float]]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    cache: dict[str, tuple[list[float], list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            smiles = str(row.get("smiles", ""))
            descriptors = [float(value) for value in row.get("descriptors", [])]
            fingerprint = [float(value) for value in row.get("fingerprint", [])]
            if len(fingerprint) != fingerprint_size:
                continue
            cache[smiles] = (descriptors, fingerprint)
    return cache


def build_molecular_feature_cache(
    db_path: str | Path,
    *,
    out_path: str | Path,
    fingerprint_size: int = 512,
    limit: int | None = None,
    source_table: str = "aggregated_task_records",
) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoder = MolecularFeatureBuilder(fingerprint_size=fingerprint_size)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT smiles, COUNT(*) AS n
            FROM "{source_table}"
            WHERE smiles IS NOT NULL
              AND TRIM(CAST(smiles AS TEXT)) <> ''
            GROUP BY smiles
            ORDER BY n DESC, smiles
            {limit_clause}
            """
        ).fetchall()
    written = 0
    fallback_count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for smiles, count in rows:
            descriptors, fingerprint = encoder.encode(smiles)
            if encoder.source != "rdkit" or len(descriptors) != len(MOLECULAR_DESCRIPTOR_NAMES):
                fallback_count += 1
            payload = {
                "smiles": smiles,
                "source_count": int(count),
                "descriptor_names": list(MOLECULAR_DESCRIPTOR_NAMES),
                "descriptors": descriptors,
                "fingerprint_size": fingerprint_size,
                "fingerprint": fingerprint,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    manifest = {
        "out_path": str(out),
        "source_table": source_table,
        "fingerprint_size": fingerprint_size,
        "encoder_source": encoder.source,
        "unique_smiles": len(rows),
        "written": written,
        "fallback_count": fallback_count,
    }
    (out.with_suffix(out.suffix + ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _smiles_ngrams(text: str) -> list[str]:
    if not text:
        return ["<missing>"]
    grams = [text[idx : idx + width] for width in (1, 2, 3) for idx in range(max(len(text) - width + 1, 0))]
    return grams or [text]


def fit_categorical_maps(
    frame: Any,
    *,
    ablation: AblationSpec | None = None,
    min_count: int = 1,
) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for column in active_categorical_columns(ablation or ABLATION_SPECS["full"]):
        if column not in frame.columns:
            continue
        counts = Counter(_category_value(value) for value in frame[column])
        mapping = {
            MISSING_CATEGORY_TOKEN: 1,
            UNKNOWN_CATEGORY_TOKEN: 2,
            RARE_CATEGORY_TOKEN: 3,
        }
        values = sorted(
            value
            for value, count in counts.items()
            if value != MISSING_CATEGORY_TOKEN and count >= max(int(min_count), 1)
        )
        mapping.update({value: idx + 4 for idx, value in enumerate(values)})
        rare_values = sorted(
            value
            for value, count in counts.items()
            if value != MISSING_CATEGORY_TOKEN and count < max(int(min_count), 1)
        )
        mapping.update({value: mapping[RARE_CATEGORY_TOKEN] for value in rare_values})
        maps[column] = mapping
    return maps


def fit_adapter_map(frame: Any, *, ablation: AblationSpec | None = None) -> dict[str, int]:
    spec = ablation or ABLATION_SPECS["full"]
    if not spec.use_medium_adapter:
        return {}
    values = sorted({adapter_name(row) for _, row in frame.iterrows()})
    return {value: idx + 1 for idx, value in enumerate(values)}


def raw_numeric_matrix(frame: Any, encoder: MolecularFeatureBuilder, *, ablation: AblationSpec | None = None) -> np.ndarray:
    spec = ablation or ABLATION_SPECS["full"]
    molecular_rows = [masked_descriptors(encoder.encode(value)[0], spec) for value in frame.get("smiles", [])]
    context_rows = [_context_numeric(row, spec) for _, row in frame.iterrows()]
    matrix = np.array([mol + ctx for mol, ctx in zip(molecular_rows, context_rows)], dtype=float)
    if matrix.size == 0:
        matrix = np.zeros((1, len(MOLECULAR_DESCRIPTOR_NAMES) + len(CONTEXT_NUMERIC_COLUMNS)))
    return matrix


def fit_numeric_stats(frame: Any, encoder: MolecularFeatureBuilder, *, ablation: AblationSpec | None = None) -> dict[str, tuple[float, float]]:
    matrix = raw_numeric_matrix(frame, encoder, ablation=ablation)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    return {str(idx): (float(mean), float(std if std > 1e-12 else 1.0)) for idx, (mean, std) in enumerate(zip(means, stds))}


def fit_zscore_correction(
    frame: Any,
    encoder: MolecularFeatureBuilder,
    *,
    numeric_stats: dict[str, tuple[float, float]],
    feature_names: tuple[str, ...],
    config: ZScoreCorrectionConfig,
    ablation: AblationSpec | None = None,
) -> ZScoreCorrection:
    split_parts = tuple(sorted({_category_value(value) for value in frame.get("split_part", [])}))
    threshold = max(float(config.threshold), 1.0)
    if not config.enabled:
        return ZScoreCorrection(
            enabled=False,
            threshold=threshold,
            feature_names=feature_names,
            fit_split_parts=split_parts,
            stats={},
        )
    matrix = raw_numeric_matrix(frame, encoder, ablation=ablation)
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(feature_names[: matrix.shape[1]]):
        mean, std = numeric_stats[str(idx)]
        std = std if std > 1e-12 else 1.0
        values = np.nan_to_num(matrix[:, idx].astype(float), nan=float(mean))
        z_values = (values - mean) / std
        clipped = np.abs(z_values) > threshold
        stats[name] = {
            "index": float(idx),
            "mean": float(mean),
            "std": float(std),
            "lower_raw": float(mean - threshold * std),
            "upper_raw": float(mean + threshold * std),
            "clipped_fit_count": float(np.count_nonzero(clipped)),
            "clipped_fit_fraction": float(np.count_nonzero(clipped) / max(len(values), 1)),
        }
    return ZScoreCorrection(
        enabled=True,
        threshold=threshold,
        feature_names=feature_names,
        fit_split_parts=split_parts,
        stats=stats,
    )


def build_deep_samples(
    frame: Any,
    *,
    encoder: MolecularFeatureBuilder,
    categorical_maps: dict[str, dict[str, int]],
    numeric_stats: dict[str, tuple[float, float]],
    target_column: str,
    target_scaler: TargetScaler | None = None,
    zscore_correction: ZScoreCorrection | None = None,
    adapter_map: dict[str, int] | None = None,
    ablation: AblationSpec | None = None,
    toxicity_binning_config: ToxicityBinningConfig | None = None,
    toxicity_bin_scheme: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = ablation or ABLATION_SPECS["full"]
    samples: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        descriptors, fingerprint = encoder.encode(row.get("smiles"))
        numeric = masked_descriptors(descriptors, spec) + _context_numeric(row, spec)
        fingerprint = masked_fingerprint(fingerprint, spec)
        normalized = []
        for idx, value in enumerate(numeric):
            mean, std = numeric_stats[str(idx)]
            value = 0.0 if value is None or not math.isfinite(float(value)) else float(value)
            z_value = (value - mean) / std
            normalized.append(zscore_correction.transform(idx, z_value) if zscore_correction is not None else z_value)
        categorical_ids = {
            column: encode_category_id(row.get(column), mapping)
            for column, mapping in categorical_maps.items()
        }
        row_adapter_name = adapter_name(row)
        adapter_id = (adapter_map or {}).get(row_adapter_name, 0) if spec.use_medium_adapter else 0
        scale_key = target_scale_key(row, "none" if target_scaler is None else target_scaler.mode)
        raw_target = float(row[target_column])
        scaled_target = raw_target if target_scaler is None else target_scaler.transform(scale_key, raw_target)
        metadata = sample_metadata(row, target_column=target_column, scale_key=scale_key)
        toxicity_fields: dict[str, Any] = {}
        if toxicity_binning_config is not None and toxicity_bin_scheme is not None:
            descriptor_mol_weight = (
                descriptors[0]
                if descriptors and getattr(encoder, "source", "") != "stable_smiles_fallback"
                else None
            )
            toxicity_fields = assign_toxicity_bin(
                row,
                toxicity_bin_scheme,
                config=toxicity_binning_config,
                descriptor_mol_weight=descriptor_mol_weight,
            ).as_sample_fields()
        samples.append(
            {
                "sample_id": row.get("aggregate_id"),
                "molecular_numeric": normalized,
                "fingerprint": fingerprint,
                "categorical_ids": categorical_ids,
                "adapter_name": row_adapter_name,
                "adapter_id": adapter_id,
                "task_head": str(row.get("task_head")),
                "target_value": float(scaled_target),
                "target_value_raw": raw_target,
                "target_value_scaled": float(scaled_target),
                "split_part": str(row.get("split_part")),
                **metadata,
                **toxicity_fields,
            }
        )
    return samples


def build_censored_training_samples(
    db_path: str | Path,
    reference_frame: Any,
    *,
    encoder: MolecularFeatureBuilder,
    categorical_maps: dict[str, dict[str, int]],
    adapter_map: dict[str, int] | None,
    numeric_stats: dict[str, tuple[float, float]],
    target_column: str,
    target_scaler: TargetScaler | None,
    zscore_correction: ZScoreCorrection | None,
    ablation: AblationSpec,
    config: CensoredLossConfig,
    kept_task_heads: tuple[str, ...],
    split_parts: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    split_part_set = {part.lower() for part in config.split_parts}
    kept_tasks = {str(task) for task in kept_task_heads}
    reference_lookup = censored_reference_split_lookup(reference_frame)
    fallback_aquatic_train = any(
        str(row.get("medium_domain", "")).lower() == "aquatic"
        and str(row.get("split_part", "")).lower() == "train"
        for _, row in reference_frame.iterrows()
    )
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    candidates = 0
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        for raw in conn.execute(
            """
            SELECT *
            FROM target_records
            WHERE value_quality LIKE 'censored%'
              AND target_status = 'excluded'
            """
        ):
            candidates += 1
            row = dict(raw)
            operator = first_censor_operator(row)
            if operator not in config.include_ops:
                skipped["unsupported_operator"] += 1
                continue
            direction = censored_direction(operator)
            direction_id = 1 if direction == "right" else -1 if direction == "left" else 0
            if direction_id == 0:
                skipped["missing_direction"] += 1
                continue
            task = map_censored_task(row)
            if not task:
                skipped["unmapped_task"] += 1
                continue
            if task["task_head"] not in kept_tasks:
                skipped["task_not_trained"] += 1
                continue
            bound = censored_bound_value(row)
            if bound is None:
                skipped["missing_or_unsupported_bound"] += 1
                continue
            split_part = resolve_censored_split_part(
                row,
                task_head=task["task_head"],
                reference_lookup=reference_lookup,
                fallback_aquatic_train=fallback_aquatic_train,
            )
            if not split_part:
                skipped["unmatched_split"] += 1
                continue
            if split_part not in split_part_set:
                skipped["split_not_trainable"] += 1
                continue
            payload = dict(row)
            payload.update(task)
            payload["aggregate_id"] = f"censored:{payload.get('record_id', len(rows))}"
            payload["split_part"] = split_part
            payload["target_value"] = bound
            payload["target_value_median"] = bound
            payload["censored_direction"] = direction
            payload["censored_direction_id"] = direction_id
            payload["censored_operator"] = operator
            payload["censored_bound_raw"] = bound
            payload["censored_value_quality"] = payload.get("value_quality", "")
            rows.append(payload)
    if not rows:
        return [], {
            **config.to_manifest(),
            "candidate_rows": candidates,
            "usable_rows": 0,
            "train_rows": 0,
            "finetune_rows": 0,
            "skipped_rows": dict(sorted(skipped.items())),
        }
    censored_frame = add_duration_nonlinear_features(pd.DataFrame(rows))
    samples = build_deep_samples(
        censored_frame,
        encoder=encoder,
        categorical_maps=categorical_maps,
        adapter_map=adapter_map,
        numeric_stats=numeric_stats,
        target_column=target_column,
        target_scaler=target_scaler,
        zscore_correction=zscore_correction,
        ablation=ablation,
    )
    for sample, row in zip(samples, rows):
        sample["censored_direction"] = row["censored_direction"]
        sample["censored_direction_id"] = row["censored_direction_id"]
        sample["censored_operator"] = row["censored_operator"]
        sample["censored_bound_raw"] = row["censored_bound_raw"]
        sample["censored_value_quality"] = row["censored_value_quality"]
    split_counts = Counter(str(sample.get("split_part", "")).lower() for sample in samples)
    return samples, {
        **config.to_manifest(),
        "candidate_rows": candidates,
        "usable_rows": len(samples),
        "train_rows": int(split_counts.get("train", 0)),
        "finetune_rows": int(split_counts.get("finetune", 0)),
        "skipped_rows": dict(sorted(skipped.items())),
    }


def censored_reference_split_lookup(frame: Any) -> dict[tuple[str, str, str, str, str, str], set[str]]:
    lookup: dict[tuple[str, str, str, str, str, str], set[str]] = {}
    for _, row in frame.iterrows():
        for identifier in (row.get("cas_number"), row.get("dtxsid")):
            key = censored_match_key(
                identifier,
                row.get("species_number"),
                row.get("task_head"),
                row.get("target_name"),
                row.get("target_basis"),
                row.get("medium_domain"),
            )
            if key[0]:
                lookup.setdefault(key, set()).add(str(row.get("split_part", "")).lower())
    return lookup


def resolve_censored_split_part(
    row: Mapping[str, Any],
    *,
    task_head: str,
    reference_lookup: dict[tuple[str, str, str, str, str, str], set[str]],
    fallback_aquatic_train: bool,
) -> str:
    parts: set[str] = set()
    for identifier in (row.get("cas_number"), row.get("dtxsid")):
        key = censored_match_key(
            identifier,
            row.get("species_number"),
            task_head,
            row.get("target_name"),
            row.get("target_basis"),
            row.get("medium_domain"),
        )
        parts.update(reference_lookup.get(key, set()))
    if len(parts) == 1:
        return next(iter(parts))
    if not parts and fallback_aquatic_train and str(row.get("medium_domain", "")).lower() == "aquatic":
        return "train"
    return ""


def censored_match_key(
    identifier: Any,
    species_number: Any,
    task_head: Any,
    target_name: Any,
    target_basis: Any,
    medium_domain: Any,
) -> tuple[str, str, str, str, str, str]:
    return (
        clean_match_value(identifier),
        clean_match_value(species_number),
        clean_match_value(task_head),
        clean_match_value(target_name),
        clean_match_value(target_basis),
        clean_match_value(medium_domain),
    )


def clean_match_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def first_censor_operator(row: Mapping[str, Any]) -> str:
    for column in ("conc1_mean_op", "conc1_min_op", "conc1_max_op"):
        value = str(row.get(column, "") or "").strip()
        if value in {"<", "<=", ">", ">="}:
            return value
    return ""


def map_censored_task(row: Mapping[str, Any]) -> dict[str, str]:
    from qsar_tl.data.task_mapping import map_task_head

    mapping = map_task_head(
        endpoint=row.get("endpoint"),
        effect=row.get("effect"),
        measurement=row.get("measurement"),
        target_name=row.get("target_name"),
        target_basis=row.get("target_basis"),
    )
    if not mapping.task_head:
        return {}
    return {
        "task_head": str(mapping.task_head),
        "task_family": str(mapping.task_family or ""),
        "effect_family": str(mapping.effect_family or ""),
    }


def censored_bound_value(row: Mapping[str, Any]) -> float | None:
    target_name = str(row.get("target_name", "") or "").strip()
    candidates = []
    if target_name == "ptox_mol_l":
        candidates = ("standard_value_mol_l",)
    elif target_name == "neg_log10_mg_kg":
        candidates = ("standard_value_mg_kg",)
    elif target_name == "neg_log10_g_ha":
        candidates = ("standard_value_g_ha",)
    elif target_name == "neg_log10_mg_kg_diet":
        candidates = ("standard_value_mg_kg_diet",)
    elif target_name == "neg_log10_mg_kg_bw_day":
        candidates = ("standard_value_mg_kg_bw_day",)
    else:
        unit_family = str(row.get("unit_family_v2", "") or "").strip()
        if unit_family == "water_mol_l":
            candidates = ("standard_value_mol_l",)
        elif unit_family == "soil_mg_kg":
            candidates = ("standard_value_mg_kg",)
    for column in candidates:
        value = optional_float(row.get(column))
        if value is not None and value > 0:
            return -math.log10(value)
    return None


def active_categorical_columns(ablation: AblationSpec) -> tuple[str, ...]:
    columns: list[str] = []
    for column in CATEGORICAL_COLUMNS:
        if column in SPECIES_LIFESTAGE_COLUMNS and not ablation.use_species_lifestage:
            continue
        if column not in SPECIES_LIFESTAGE_COLUMNS and not ablation.use_other_categorical_context:
            continue
        columns.append(column)
    return tuple(columns)


def fit_target_scaler(
    frame: Any,
    *,
    target_column: str,
    mode: str = "per_task_target",
    fit_indices: list[int] | tuple[int, ...] | None = None,
) -> TargetScaler:
    normalized_mode = (mode or "per_task_target").strip().lower()
    allowed_modes = {
        "none",
        "identity",
        "global",
        "per_task",
        "per_target",
        "per_task_target",
        "per_adapter",
        "per_task_adapter",
    }
    if normalized_mode not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValueError(f"Unsupported target_standardization '{mode}'. Allowed values: {allowed}")
    if fit_indices is None:
        fit_frame = frame[frame["split_part"].astype("string").str.lower() == "train"].copy()
    else:
        fit_frame = frame.iloc[list(fit_indices)].copy()
    if fit_frame.empty:
        raise ValueError("Cannot fit target scaler without training rows.")
    split_parts = tuple(sorted({_category_value(value) for value in fit_frame.get("split_part", [])}))
    if normalized_mode in {"none", "identity"}:
        return TargetScaler(
            mode=normalized_mode,
            target_column=target_column,
            fit_split_parts=split_parts,
            stats={GLOBAL_TARGET_SCALE_KEY: _target_stats([0.0])},
        )
    stats: dict[str, dict[str, float]] = {
        GLOBAL_TARGET_SCALE_KEY: _target_stats([safe_number(value) for value in fit_frame[target_column]])
    }
    if normalized_mode == "global":
        return TargetScaler(
            mode=normalized_mode,
            target_column=target_column,
            fit_split_parts=split_parts,
            stats=stats,
        )
    grouped_values: dict[str, list[float]] = {}
    for _, row in fit_frame.iterrows():
        key = target_scale_key(row, normalized_mode)
        grouped_values.setdefault(key, []).append(safe_number(row[target_column]))
    for key, values in grouped_values.items():
        stats[key] = _target_stats(values)
    return TargetScaler(
        mode=normalized_mode,
        target_column=target_column,
        fit_split_parts=split_parts,
        stats=stats,
    )


def target_scale_key(row: Mapping[str, Any], mode: str) -> str:
    normalized_mode = (mode or "per_task_target").strip().lower()
    task = _category_value(row.get("task_head"))
    target = _target_dimension_value(row)
    if normalized_mode in {"none", "identity", "global"}:
        return GLOBAL_TARGET_SCALE_KEY
    if normalized_mode == "per_task":
        return task
    if normalized_mode == "per_target":
        return target
    if normalized_mode == "per_adapter":
        return adapter_name(row)
    if normalized_mode == "per_task_adapter":
        return f"{task}|{adapter_name(row)}"
    return f"{task}|{target}"


def _target_stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        finite = [0.0]
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    return {
        "count": float(len(finite)),
        "mean": mean,
        "std": std if std > 1e-12 else 1.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def resolve_task_weights(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    train_cfg: Mapping[str, Any],
    task_heads: tuple[str, ...],
) -> dict[str, float]:
    raw_manual = train_cfg.get("task_weights", {}) or {}
    manual = {str(key): float(value) for key, value in raw_manual.items()}
    mode = str(train_cfg.get("task_weighting", "balanced")).strip().lower()
    group_by_task: dict[str, str] = {}
    for idx in train_indices:
        task_head = str(samples[idx].get("task_head", ""))
        task_group = str(samples[idx].get("task_group", "") or "")
        if task_head and task_head not in group_by_task:
            group_by_task[task_head] = task_group
    main_weight = float(train_cfg.get("main_task_weight", 1.0))
    toxicity_aux_weight = float(train_cfg.get("toxicity_aux_task_weight", 0.35))
    bioaccum_aux_weight = float(train_cfg.get("bioaccumulation_aux_task_weight", 0.2))
    weights = {}
    for task_head in task_heads:
        task_group = group_by_task.get(task_head, "")
        if task_group == "toxicity_aux":
            base_weight = toxicity_aux_weight
        elif task_group == "bioaccumulation_aux":
            base_weight = bioaccum_aux_weight
        else:
            base_weight = main_weight
        weights[task_head] = base_weight * float(manual.get(task_head, 1.0))
    if mode in {"none", "manual"}:
        return weights
    if mode != "balanced":
        raise ValueError("training.task_weighting must be 'balanced', 'manual', or 'none'.")
    counts = Counter(str(samples[idx]["task_head"]) for idx in train_indices)
    if not counts:
        return weights
    reference = float(np.median([count for count in counts.values()]))
    exponent = float(train_cfg.get("task_weight_exponent", 0.5))
    min_weight = float(train_cfg.get("task_weight_min", 0.25))
    max_weight = float(train_cfg.get("task_weight_max", 4.0))
    for task_head in task_heads:
        count = max(int(counts.get(task_head, 0)), 1)
        balanced = (reference / count) ** exponent
        weights[task_head] = max(min_weight, min(max_weight, balanced * weights[task_head]))
    return weights


def encode_category_id(value: Any, mapping: Mapping[str, int]) -> int:
    token = _category_value(value)
    if token in mapping:
        return int(mapping[token])
    return int(mapping.get(UNKNOWN_CATEGORY_TOKEN, 0))


def sample_metadata(row: Mapping[str, Any], *, target_column: str, scale_key: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for column in PREDICTION_METADATA_COLUMNS:
        if column == "sample_id":
            metadata[column] = clean_metadata_value(row.get("aggregate_id", row.get("sample_id", "")))
        elif column == "target_column":
            metadata[column] = target_column
        elif column == "target_scale_key":
            metadata[column] = scale_key
        elif column in row:
            metadata[column] = clean_metadata_value(row.get(column))
    return metadata


def clean_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return value


def safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def adapter_name(row: Mapping[str, Any]) -> str:
    medium = _category_value(
        row.get("medium_domain")
        if row.get("medium_domain") is not None
        else row.get("primary_medium")
    )
    target = _target_dimension_value(row)
    return f"{medium}|{target}"


def _target_dimension_value(row: Mapping[str, Any]) -> str:
    value = row.get("target_family")
    if value is None or str(value).strip() == "":
        value = row.get("target_name")
    return _category_value(value)


def masked_descriptors(descriptors: list[float], ablation: AblationSpec) -> list[float]:
    if ablation.use_descriptors:
        return descriptors
    return [0.0] * len(descriptors)


def masked_fingerprint(fingerprint: list[float], ablation: AblationSpec) -> list[float]:
    if ablation.use_fingerprint:
        return fingerprint
    return [0.0] * len(fingerprint)


def _context_numeric(row: Mapping[str, Any], ablation: AblationSpec | None = None) -> list[float]:
    spec = ablation or ABLATION_SPECS["full"]
    effect_level = optional_float(row.get("effect_level_x"))
    has_effect_level = effect_level is not None
    effect_level_value = float(effect_level or 0.0)
    values = []
    for column in CONTEXT_NUMERIC_COLUMNS:
        if not spec.use_context_numeric:
            values.append(0.0)
            continue
        if column == "effect_level_x":
            values.append(effect_level_value)
            continue
        if column == "effect_level_x_fraction":
            values.append(effect_level_value / 100.0)
            continue
        if column == "effect_level_x_log1p":
            values.append(math.log1p(max(effect_level_value, 0.0)))
            continue
        if column == "effect_level_x_present":
            values.append(1.0 if has_effect_level else 0.0)
            continue
        if column in DURATION_CONTEXT_COLUMNS and not spec.use_duration_features:
            values.append(0.0)
            continue
        raw = row.get(column, 0.0)
        try:
            values.append(float(raw) if raw is not None and raw == raw else 0.0)
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _category_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def _concat_frames(frames: list[Any]) -> Any:
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


class _IndexDataset:
    def __init__(self, dataset: AggregatedTaskDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


class _NoisyIndexDataset:
    def __init__(
        self,
        dataset: AggregatedTaskDataset,
        indices: list[int],
        *,
        replicates: int,
        numeric_noise_std: float,
        target_noise_std: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.replicates = max(int(replicates), 1)
        self.numeric_noise_std = max(float(numeric_noise_std), 0.0)
        self.target_noise_std = max(float(target_noise_std), 0.0)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.indices) * self.replicates

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index, replicate_id = self.source(index)
        sample = dict(self.dataset[source_index])
        sample["categorical_ids"] = dict(sample.get("categorical_ids", {}))
        sample["molecular_numeric"] = list(sample.get("molecular_numeric", []))
        sample["fingerprint"] = list(sample.get("fingerprint", []))
        if replicate_id > 0:
            rng = np.random.default_rng(self.seed + source_index * 1009 + replicate_id * 9176)
            if self.numeric_noise_std > 0 and sample["molecular_numeric"]:
                noise = rng.normal(0.0, self.numeric_noise_std, size=len(sample["molecular_numeric"]))
                sample["molecular_numeric"] = [
                    float(value) + float(delta)
                    for value, delta in zip(sample["molecular_numeric"], noise)
                ]
            if self.target_noise_std > 0:
                sample["target_value"] = float(sample["target_value"]) + float(
                    rng.normal(0.0, self.target_noise_std)
                )
        sample["perturbation_replicate"] = replicate_id
        sample["perturbation_numeric_noise_std"] = self.numeric_noise_std
        return sample

    def source(self, index: int) -> tuple[int, int]:
        if not self.indices:
            raise IndexError("Cannot index an empty noisy dataset.")
        base_position = index // self.replicates
        replicate_id = index % self.replicates
        return self.indices[base_position], replicate_id


def build_noisy_index_dataset(
    dataset: AggregatedTaskDataset,
    indices: list[int],
    *,
    replicates: int,
    numeric_noise_std: float,
    target_noise_std: float,
    seed: int,
) -> Any:
    if max(int(replicates), 1) <= 1 or (float(numeric_noise_std) <= 0 and float(target_noise_std) <= 0):
        return _IndexDataset(dataset, indices)
    return _NoisyIndexDataset(
        dataset,
        indices,
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        target_noise_std=target_noise_std,
        seed=seed,
    )


def mark_internal_validation_samples(
    samples: list[dict[str, Any]],
    *,
    indices: list[int],
    split_part: str,
) -> None:
    for idx in indices:
        if idx < 0 or idx >= len(samples):
            continue
        sample = dict(samples[idx])
        sample.setdefault("original_split_part", sample.get("split_part", ""))
        sample["split_part"] = split_part
        samples[idx] = sample


def apply_source_similarity_weights(
    samples: list[dict[str, Any]],
    config: SourceWeightingConfig,
) -> dict[str, Any]:
    for sample in samples:
        sample["sample_weight"] = 1.0
    summary: dict[str, Any] = {
        **config.to_manifest(),
        "applied": False,
        "weighted_samples": 0,
        "target_reference_samples": 0,
    }
    if not config.active():
        return summary
    allowed_methods = {
        "tanimoto",
        "tanimoto_to_target",
        "tanimoto_to_finetune",
        "proxy_distance_to_finetune",
        "tanimoto_proxy_to_finetune",
    }
    if config.method not in allowed_methods:
        raise ValueError(
            "training.source_weighting.method must be 'none', 'tanimoto', "
            "'tanimoto_to_target', 'tanimoto_to_finetune', "
            "'proxy_distance_to_finetune', or 'tanimoto_proxy_to_finetune'."
        )
    source_indices = select_samples_by_domain_and_split(
        samples,
        split_parts=config.source_split_parts,
        domains=config.source_domains,
    )
    target_indices = select_samples_by_domain_and_split(
        samples,
        split_parts=config.target_split_parts,
        domains=config.target_domains,
    )
    summary["weighted_samples"] = len(source_indices)
    summary["target_reference_samples"] = len(target_indices)
    if not source_indices or not target_indices:
        return summary
    similarities: np.ndarray | None = None
    distances: np.ndarray | None = None
    raw_weights: np.ndarray | None = None
    if config.method in {"tanimoto", "tanimoto_to_target", "tanimoto_to_finetune", "tanimoto_proxy_to_finetune"}:
        source_fp = np.asarray([samples[idx].get("fingerprint", []) for idx in source_indices], dtype=np.float32)
        target_fp = np.asarray([samples[idx].get("fingerprint", []) for idx in target_indices], dtype=np.float32)
        if source_fp.ndim != 2 or target_fp.ndim != 2 or source_fp.shape[1] != target_fp.shape[1]:
            return summary
        similarities = max_tanimoto_similarity(source_fp, target_fp)
        raw_weights = 1.0 + float(config.alpha) * similarities
    if config.method in {"proxy_distance_to_finetune", "tanimoto_proxy_to_finetune"}:
        source_proxy = proxy_descriptor_matrix(samples, source_indices)
        target_proxy = proxy_descriptor_matrix(samples, target_indices)
        distances = min_proxy_distance(source_proxy, target_proxy)
        proxy_weights = np.exp(-float(config.alpha) * distances)
        raw_weights = proxy_weights if raw_weights is None else raw_weights * proxy_weights
    if raw_weights is None:
        return summary
    mean_weight = float(np.mean(raw_weights)) if raw_weights.size else 1.0
    if mean_weight > 0 and math.isfinite(mean_weight):
        raw_weights = raw_weights / mean_weight
    lower = min(float(config.min_weight), float(config.max_weight))
    upper = max(float(config.min_weight), float(config.max_weight))
    clipped = np.clip(raw_weights, lower, upper)
    for idx, weight in zip(source_indices, clipped):
        samples[idx]["sample_weight"] = float(weight)
    summary.update(
        {
            "applied": True,
            "weight_min": float(np.min(clipped)),
            "weight_mean": float(np.mean(clipped)),
            "weight_max": float(np.max(clipped)),
        }
    )
    if similarities is not None and similarities.size:
        summary.update(
            {
                "similarity_min": float(np.min(similarities)),
                "similarity_mean": float(np.mean(similarities)),
                "similarity_max": float(np.max(similarities)),
            }
        )
    if distances is not None and distances.size:
        summary.update(
            {
                "proxy_distance_min": float(np.min(distances)),
                "proxy_distance_mean": float(np.mean(distances)),
                "proxy_distance_max": float(np.max(distances)),
            }
        )
    return summary


def proxy_descriptor_matrix(samples: list[dict[str, Any]], indices: list[int]) -> np.ndarray:
    proxy_indices = [0, 1, 2]
    rows: list[list[float]] = []
    for idx in indices:
        values = list(samples[idx].get("molecular_numeric", []))
        rows.append([safe_proxy_float(values[pos]) if pos < len(values) else 0.0 for pos in proxy_indices])
    return np.asarray(rows, dtype=np.float32)


def safe_proxy_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def min_proxy_distance(source_proxy: np.ndarray, target_proxy: np.ndarray, *, batch_size: int = 2048) -> np.ndarray:
    if source_proxy.ndim != 2 or target_proxy.ndim != 2 or source_proxy.shape[1] != target_proxy.shape[1]:
        return np.full(source_proxy.shape[0] if source_proxy.ndim >= 1 else 0, np.inf, dtype=np.float32)
    if source_proxy.shape[0] == 0 or target_proxy.shape[0] == 0:
        return np.full(source_proxy.shape[0], np.inf, dtype=np.float32)
    pooled = np.vstack([source_proxy, target_proxy])
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    source_scaled = ((np.nan_to_num(source_proxy, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32)
    target_scaled = ((np.nan_to_num(target_proxy, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std).astype(np.float32)
    target_norm = np.sum(target_scaled * target_scaled, axis=1, dtype=np.float32)[None, :]
    out = np.empty(source_scaled.shape[0], dtype=np.float32)
    chunk = max(1, int(batch_size))
    for start in range(0, source_scaled.shape[0], chunk):
        batch = source_scaled[start : start + chunk]
        batch_norm = np.sum(batch * batch, axis=1, dtype=np.float32)[:, None]
        distances_sq = batch_norm + target_norm - 2.0 * (batch @ target_scaled.T)
        out[start : start + chunk] = np.sqrt(np.maximum(np.min(distances_sq, axis=1), 0.0))
    return out


def apply_effect_level_frequency_weights(
    samples: list[dict[str, Any]],
    config: EffectLevelWeightingConfig,
    *,
    train_indices: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        **config.to_manifest(),
        "applied": False,
        "training_reference_samples": len(set(train_indices)),
        "weighted_samples": 0,
        "eligible_levels": 0,
        "level_counts": {},
    }
    if not config.active():
        return summary
    lower = min(float(config.min_weight), float(config.max_weight))
    upper = max(float(config.min_weight), float(config.max_weight))
    split_set = {str(value).strip().lower() for value in config.split_parts if str(value).strip()}
    prefix_set = tuple(str(value).strip().lower() for value in config.task_prefixes if str(value).strip())
    eligible: list[tuple[int, str]] = []
    seen_indices = sorted({int(idx) for idx in train_indices if 0 <= int(idx) < len(samples)})
    for idx in seen_indices:
        sample = samples[idx]
        split_part = str(sample.get("split_part", "")).strip().lower()
        if split_set and split_part not in split_set:
            continue
        task_head = str(sample.get("task_head", "")).strip().lower()
        if prefix_set and not task_head.startswith(prefix_set):
            continue
        effect_level = optional_float(sample.get("effect_level_x"))
        if effect_level is None:
            continue
        eligible.append((idx, effect_level_label(effect_level)))
    counts = Counter(level for _, level in eligible)
    summary["weighted_samples"] = len(eligible)
    summary["eligible_levels"] = len(counts)
    summary["level_counts"] = dict(sorted(counts.items()))
    if not eligible or not counts:
        return summary
    raw_weights = np.asarray([float(counts[level]) ** (-float(config.beta)) for _, level in eligible], dtype=np.float32)
    raw_mean = float(np.mean(raw_weights)) if raw_weights.size else 1.0
    normalized = raw_weights / raw_mean if raw_mean > 0 and math.isfinite(raw_mean) else raw_weights
    clipped = normalize_clipped_weights(normalized, lower, upper)
    final_weights: list[float] = []
    for (idx, _), effect_weight in zip(eligible, clipped):
        current = float(samples[idx].get("sample_weight", 1.0) or 1.0)
        final_weight = current * float(effect_weight)
        samples[idx]["sample_weight"] = final_weight
        final_weights.append(final_weight)
    final_array = np.asarray(final_weights, dtype=np.float32)
    summary.update(
        {
            "applied": True,
            "raw_weight_min": float(np.min(raw_weights)),
            "raw_weight_mean": raw_mean,
            "raw_weight_max": float(np.max(raw_weights)),
            "effect_weight_min": float(np.min(clipped)),
            "effect_weight_mean": float(np.mean(clipped)),
            "effect_weight_max": float(np.max(clipped)),
            "final_weight_min": float(np.min(final_array)),
            "final_weight_mean": float(np.mean(final_array)),
            "final_weight_max": float(np.max(final_array)),
        }
    )
    return summary


def normalize_clipped_weights(weights: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if values.size == 0:
        return values
    raw_lower = float(lower)
    raw_upper = float(upper)
    lower = min(raw_lower, raw_upper)
    upper = max(raw_lower, raw_upper)
    if lower > 1.0 or upper < 1.0:
        return np.clip(values, lower, upper)

    def clipped_mean(scale: float) -> float:
        return float(np.mean(np.clip(values * scale, lower, upper)))

    low = 0.0
    high = 1.0
    while clipped_mean(high) < 1.0 and high < 1_000_000.0:
        high *= 2.0
    for _ in range(48):
        mid = (low + high) / 2.0
        if clipped_mean(mid) < 1.0:
            low = mid
        else:
            high = mid
    return np.clip(values * high, lower, upper)


def sample_weighting_history_fields(
    source_weighting_summary: Mapping[str, Any],
    effect_level_weighting_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_weighting_applied": bool(source_weighting_summary.get("applied", False)),
        "source_weighting_alpha": source_weighting_summary.get("alpha", ""),
        "source_weighted_samples": source_weighting_summary.get("weighted_samples", 0),
        "source_weight_min": source_weighting_summary.get("weight_min", ""),
        "source_weight_mean": source_weighting_summary.get("weight_mean", ""),
        "source_weight_max": source_weighting_summary.get("weight_max", ""),
        "effect_level_weighting_applied": bool(effect_level_weighting_summary.get("applied", False)),
        "effect_level_weighting_beta": effect_level_weighting_summary.get("beta", ""),
        "effect_level_weighted_samples": effect_level_weighting_summary.get("weighted_samples", 0),
        "effect_level_weight_min": effect_level_weighting_summary.get("effect_weight_min", ""),
        "effect_level_weight_mean": effect_level_weighting_summary.get("effect_weight_mean", ""),
        "effect_level_weight_max": effect_level_weighting_summary.get("effect_weight_max", ""),
        "final_sample_weight_min": effect_level_weighting_summary.get(
            "final_weight_min",
            source_weighting_summary.get("weight_min", ""),
        ),
        "final_sample_weight_mean": effect_level_weighting_summary.get(
            "final_weight_mean",
            source_weighting_summary.get("weight_mean", ""),
        ),
        "final_sample_weight_max": effect_level_weighting_summary.get(
            "final_weight_max",
            source_weighting_summary.get("weight_max", ""),
        ),
    }


def select_samples_by_domain_and_split(
    samples: list[dict[str, Any]],
    *,
    split_parts: tuple[str, ...],
    domains: tuple[str, ...],
) -> list[int]:
    split_set = {str(value).strip().lower() for value in split_parts if str(value).strip()}
    domain_set = {str(value).strip().lower() for value in domains if str(value).strip()}
    indices: list[int] = []
    for idx, sample in enumerate(samples):
        split_part = str(sample.get("split_part", "")).strip().lower()
        medium_domain = str(sample.get("medium_domain", "")).strip().lower()
        if split_set and split_part not in split_set:
            continue
        if domain_set and medium_domain not in domain_set:
            continue
        indices.append(idx)
    return indices


def max_tanimoto_similarity(source_fp: np.ndarray, target_fp: np.ndarray, *, chunk_size: int = 2048) -> np.ndarray:
    source_binary = (source_fp > 0).astype(np.uint8, copy=False)
    target_binary = (target_fp > 0).astype(np.uint8, copy=False)
    if source_binary.shape[0] == 0 or target_binary.shape[0] == 0:
        return np.zeros(source_binary.shape[0], dtype=np.float32)
    source_unique, source_inverse = np.unique(source_binary, axis=0, return_inverse=True)
    target_unique = np.unique(target_binary, axis=0)
    source = source_unique.astype(np.float32, copy=False)
    target = target_unique.astype(np.float32, copy=False)
    target_sums = target.sum(axis=1, keepdims=True).T
    unique_scores = np.zeros(source.shape[0], dtype=np.float32)
    for start in range(0, source.shape[0], chunk_size):
        chunk = source[start : start + chunk_size]
        intersections = chunk @ target.T
        denominators = chunk.sum(axis=1, keepdims=True) + target_sums - intersections
        sims = np.divide(
            intersections,
            np.maximum(denominators, 1.0),
            out=np.zeros_like(intersections, dtype=np.float32),
            where=denominators > 0,
        )
        unique_scores[start : start + chunk.shape[0]] = sims.max(axis=1)
    return unique_scores[source_inverse]


def domain_alignment_reference_indices(
    samples: list[dict[str, Any]],
    config: DomainAlignmentConfig,
    *,
    phase: str,
) -> list[int]:
    if not config.active() or phase not in set(config.phases):
        return []
    if phase == "pretrain":
        return select_samples_by_domain_and_split(
            samples,
            split_parts=config.target_split_parts,
            domains=config.target_domains,
        )
    if phase == "finetune":
        return select_samples_by_domain_and_split(
            samples,
            split_parts=config.source_split_parts,
            domains=config.source_domains,
        )
    return []


def cycle_dataloader(dataloader: Any) -> Any:
    while True:
        for batch in dataloader:
            yield batch


def should_update_swa(config: SwaConfig, *, phase: str, epoch: int) -> bool:
    return config.active() and config.phase == phase and int(epoch) >= int(config.start_epoch)


def _hidden_dims(config: Mapping[str, Any]) -> tuple[int, ...]:
    hidden_dim = int(config.get("model", {}).get("hidden_dim", 256))
    return (hidden_dim, max(32, hidden_dim // 2))


def _resolve_device(device: str) -> str:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _early_stopping_config(
    config: Mapping[str, Any],
    *,
    enabled_override: bool | None,
    patience_override: int | None,
    min_delta_override: float | None,
    validation_fraction_override: float | None,
    monitor_split_override: str | None,
) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    raw = train_cfg.get("early_stopping", {}) if isinstance(train_cfg.get("early_stopping", {}), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False) if enabled_override is None else enabled_override),
        "patience": int(patience_override if patience_override is not None else raw.get("patience", 15)),
        "min_delta": float(min_delta_override if min_delta_override is not None else raw.get("min_delta", 0.0)),
        "validation_fraction": float(
            validation_fraction_override if validation_fraction_override is not None else raw.get("validation_fraction", 0.1)
        ),
        "monitor_split": str(monitor_split_override if monitor_split_override is not None else raw.get("monitor_split", "auto")),
    }


def _feature_noise_config(
    train_cfg: Mapping[str, Any],
    *,
    seed: int,
    train_replicates_override: int | None = None,
    finetune_replicates_override: int | None = None,
    numeric_noise_std_override: float | None = None,
    target_noise_std_override: float | None = None,
) -> FeatureNoiseConfig:
    raw = train_cfg.get("augmentation", {}) if isinstance(train_cfg.get("augmentation", {}), dict) else {}
    return FeatureNoiseConfig(
        train_replicates=max(
            1,
            int(train_replicates_override if train_replicates_override is not None else raw.get("train_replicates", 1)),
        ),
        finetune_replicates=max(
            1,
            int(
                finetune_replicates_override
                if finetune_replicates_override is not None
                else raw.get("finetune_replicates", raw.get("train_replicates", 1))
            ),
        ),
        numeric_noise_std=max(
            0.0,
            float(
                numeric_noise_std_override
                if numeric_noise_std_override is not None
                else raw.get("numeric_noise_std", 0.0)
            ),
        ),
        target_noise_std=max(
            0.0,
            float(
                target_noise_std_override
                if target_noise_std_override is not None
                else raw.get("target_noise_std", 0.0)
            ),
        ),
        seed=int(raw.get("seed", seed)),
    )


def _zscore_correction_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    threshold_override: float | None = None,
) -> ZScoreCorrectionConfig:
    raw = train_cfg.get("feature_zscore_correction", {}) if isinstance(train_cfg.get("feature_zscore_correction", {}), dict) else {}
    enabled = bool(raw.get("enabled", True) if enabled_override is None else enabled_override)
    threshold = float(threshold_override if threshold_override is not None else raw.get("threshold", 6.0))
    return ZScoreCorrectionConfig(enabled=enabled, threshold=max(threshold, 1.0))


def _perturbation_config(
    train_cfg: Mapping[str, Any],
    *,
    seed: int,
    replicates_override: int | None = None,
    numeric_noise_std_override: float | None = None,
) -> PerturbationConfig:
    raw = train_cfg.get("test_perturbation", {}) if isinstance(train_cfg.get("test_perturbation", {}), dict) else {}
    split_parts_raw = raw.get("split_parts", ["test"])
    if isinstance(split_parts_raw, str):
        split_parts = tuple(part.strip() for part in split_parts_raw.split(",") if part.strip())
    else:
        split_parts = tuple(str(part).strip() for part in split_parts_raw if str(part).strip())
    replicates = max(1, int(replicates_override if replicates_override is not None else raw.get("replicates", 1)))
    numeric_noise_std = max(
        0.0,
        float(
            numeric_noise_std_override
            if numeric_noise_std_override is not None
            else raw.get("numeric_noise_std", 0.0)
        ),
    )
    return PerturbationConfig(
        enabled=bool(raw.get("enabled", False) or replicates_override is not None or numeric_noise_std_override is not None),
        split_parts=split_parts or ("test",),
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        seed=int(raw.get("seed", seed + 200_000)),
    )


def _source_weighting_config(
    train_cfg: Mapping[str, Any],
    *,
    method_override: str | None = None,
    alpha_override: float | None = None,
) -> SourceWeightingConfig:
    raw = train_cfg.get("source_weighting", {}) if isinstance(train_cfg.get("source_weighting", {}), dict) else {}
    method = str(method_override if method_override is not None else raw.get("method", "none")).strip().lower()
    alpha = max(0.0, float(alpha_override if alpha_override is not None else raw.get("alpha", 0.0)))
    return SourceWeightingConfig(
        enabled=bool(raw.get("enabled", False) or method_override is not None or alpha_override is not None),
        method=method,
        alpha=alpha,
        source_split_parts=_string_tuple(raw.get("source_split_parts", ("train",)), lower=True) or ("train",),
        target_split_parts=_string_tuple(raw.get("target_split_parts", ("finetune",)), lower=True) or ("finetune",),
        source_domains=_string_tuple(raw.get("source_domains", ("aquatic",)), lower=True) or ("aquatic",),
        target_domains=_string_tuple(raw.get("target_domains", ("soil",)), lower=True) or ("soil",),
        min_weight=max(0.0, float(raw.get("min_weight", 0.25))),
        max_weight=max(0.0, float(raw.get("max_weight", 2.0))),
    )


def _effect_level_weighting_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    beta_override: float | None = None,
) -> EffectLevelWeightingConfig:
    raw = (
        train_cfg.get("effect_level_weighting", {})
        if isinstance(train_cfg.get("effect_level_weighting", {}), dict)
        else {}
    )
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if beta_override is not None:
        enabled = True
    beta = max(0.0, float(beta_override if beta_override is not None else raw.get("beta", 0.0)))
    return EffectLevelWeightingConfig(
        enabled=enabled,
        beta=beta,
        split_parts=_string_tuple(raw.get("split_parts", ("train", "finetune")), lower=True)
        or ("train", "finetune"),
        task_prefixes=_string_tuple(raw.get("task_prefixes", ("ECx", "LCx", "ICx", "LDx")), lower=False)
        or ("ECx", "LCx", "ICx", "LDx"),
        min_weight=max(0.0, float(raw.get("min_weight", 0.5))),
        max_weight=max(0.0, float(raw.get("max_weight", 3.0))),
    )


def _toxicity_binning_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    mode_override: str | None = None,
    loss_weight_override: float | None = None,
    scheme_override: str | None = None,
) -> ToxicityBinningConfig:
    raw = (
        train_cfg.get("toxicity_binning", {})
        if isinstance(train_cfg.get("toxicity_binning", {}), dict)
        else {}
    )
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if mode_override is not None or loss_weight_override is not None or scheme_override is not None:
        enabled = True
    mode = str(mode_override if mode_override is not None else raw.get("mode", "aux_classification")).strip().lower()
    loss_weight = max(0.0, float(loss_weight_override if loss_weight_override is not None else raw.get("loss_weight", 0.05)))
    return ToxicityBinningConfig(
        enabled=enabled,
        scheme=str(scheme_override if scheme_override is not None else raw.get("scheme", "authority_v1")).strip()
        or "authority_v1",
        mode=mode,
        loss_weight=loss_weight,
        boundary_policy=str(raw.get("boundary_policy", "hard")).strip().lower() or "hard",
        boundary_tolerance=max(0.0, float(raw.get("boundary_tolerance", 0.05))),
        threshold_multiplier=max(1e-12, float(raw.get("threshold_multiplier", 1.0))),
        require_active_bin_for_regression=bool(raw.get("require_active_bin_for_regression", False)),
    )


def _censored_loss_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    weight_override: float | None = None,
    margin_override: float | None = None,
) -> CensoredLossConfig:
    raw = train_cfg.get("censored_loss", {}) if isinstance(train_cfg.get("censored_loss", {}), dict) else {}
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    if weight_override is not None or margin_override is not None:
        enabled = True
    method = str(raw.get("method", "hinge")).strip().lower() or "hinge"
    weight = max(0.0, float(weight_override if weight_override is not None else raw.get("weight", 0.0)))
    margin = max(0.0, float(margin_override if margin_override is not None else raw.get("margin", 0.0)))
    return CensoredLossConfig(
        enabled=enabled,
        method=method,
        weight=weight,
        margin=margin,
        split_parts=_string_tuple(raw.get("split_parts", ("train", "finetune")), lower=True)
        or ("train", "finetune"),
        include_ops=_string_tuple(raw.get("include_ops", ("<", "<=", ">", ">=")), lower=False)
        or ("<", "<=", ">", ">="),
    )


def _domain_alignment_config(
    train_cfg: Mapping[str, Any],
    *,
    method_override: str | None = None,
    weight_override: float | None = None,
) -> DomainAlignmentConfig:
    raw = train_cfg.get("domain_alignment", {}) if isinstance(train_cfg.get("domain_alignment", {}), dict) else {}
    method = str(method_override if method_override is not None else raw.get("method", "none")).strip().lower()
    weight = max(0.0, float(weight_override if weight_override is not None else raw.get("weight", 0.0)))
    return DomainAlignmentConfig(
        enabled=bool(raw.get("enabled", False) or method_override is not None or weight_override is not None),
        method=method,
        weight=weight,
        source_split_parts=_string_tuple(raw.get("source_split_parts", ("train",)), lower=True) or ("train",),
        target_split_parts=_string_tuple(raw.get("target_split_parts", ("finetune",)), lower=True) or ("finetune",),
        source_domains=_string_tuple(raw.get("source_domains", ("aquatic",)), lower=True) or ("aquatic",),
        target_domains=_string_tuple(raw.get("target_domains", ("soil",)), lower=True) or ("soil",),
        phases=_string_tuple(raw.get("phases", ("pretrain",)), lower=True) or ("pretrain",),
    )


def _swa_config(
    train_cfg: Mapping[str, Any],
    *,
    enabled_override: bool | None = None,
    start_epoch_override: int | None = None,
) -> SwaConfig:
    raw = train_cfg.get("swa", {}) if isinstance(train_cfg.get("swa", {}), dict) else {}
    enabled = bool(raw.get("enabled", False) if enabled_override is None else enabled_override)
    start_epoch = max(1, int(start_epoch_override if start_epoch_override is not None else raw.get("start_epoch", 15)))
    return SwaConfig(
        enabled=enabled,
        phase=str(raw.get("phase", "finetune")).strip().lower() or "finetune",
        start_epoch=start_epoch,
    )


def _string_tuple(value: Any, *, lower: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    if lower:
        return tuple(item.lower() for item in items if item)
    return tuple(item for item in items if item)


def split_training_validation_indices(
    samples: list[dict[str, Any]],
    *,
    train_indices: list[int],
    seed: int,
    validation_fraction: float,
    monitor_split: str = "auto",
) -> tuple[list[int], list[int], str]:
    requested = (monitor_split or "auto").strip().lower()
    force_internal = requested in {"internal_train_fraction", "train_fraction"}
    explicit_parts = [] if requested == "auto" or force_internal else [requested]
    preferred_parts = [] if force_internal else (explicit_parts or ["finetune", "validation", "val"])
    for part in preferred_parts:
        indices = [idx for idx, sample in enumerate(samples) if str(sample.get("split_part", "")).lower() == part]
        if indices:
            return list(train_indices), indices, part

    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(train_indices) < 2:
        return list(train_indices), [], ""

    rng = np.random.default_rng(seed)
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for idx in train_indices:
        task = str(samples[idx].get("task_head", ""))
        by_task.setdefault(task, []).append(idx)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    actual_train = [idx for idx in train_indices if idx not in validation_set]
    return actual_train, sorted(validation), "internal_train_fraction" if validation else ""


def split_finetune_validation_indices(
    samples: list[dict[str, Any]],
    *,
    finetune_indices: list[int],
    seed: int,
    validation_fraction: float,
    monitor_split: str = "auto",
) -> tuple[list[int], list[int], str]:
    requested = (monitor_split or "auto").strip().lower()
    if requested not in {"", "auto", "internal", "internal_finetune_fraction", "finetune_fraction"}:
        validation = [
            idx
            for idx, sample in enumerate(samples)
            if str(sample.get("split_part", "")).lower() == requested
        ]
        if validation:
            return list(finetune_indices), validation, requested

    fraction = max(0.0, min(float(validation_fraction), 0.5))
    if fraction <= 0 or len(finetune_indices) < 2:
        return list(finetune_indices), [], ""

    rng = np.random.default_rng(seed + 17_031)
    validation: list[int] = []
    by_task: dict[str, list[int]] = {}
    for idx in finetune_indices:
        task = str(samples[idx].get("task_head", ""))
        by_task.setdefault(task, []).append(idx)
    for indices in by_task.values():
        if len(indices) < 2:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_validation = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
        validation.extend(shuffled[:n_validation])
    validation_set = set(validation)
    finetune_train = [idx for idx in finetune_indices if idx not in validation_set]
    if not finetune_train:
        return list(finetune_indices), [], ""
    return finetune_train, sorted(validation), "internal_finetune_fraction" if validation else ""


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_scheduler(optimizer: Any, config: DeepTrainingConfig, epochs: int) -> Any | None:
    scheduler_name = (config.scheduler or "none").strip().lower()
    if scheduler_name in {"", "none", "off", "false"}:
        return None
    import torch

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
    raise ValueError("training.scheduler must be 'none', 'cosine', or 'reduce_on_plateau'.")


def step_scheduler(scheduler: Any | None, metric: float) -> None:
    if scheduler is None:
        return
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(float(metric))
    else:
        scheduler.step()


def current_learning_rate(optimizer: Any) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "false"}:
        return None
    return float(value)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    config: DeepTrainingConfig,
    device: Any,
    *,
    alignment_batches: Any | None = None,
    alignment_weight: float = 0.0,
) -> dict[str, Any]:
    import torch

    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_toxicity_bin_loss = 0.0
    total_toxicity_bin_samples = 0
    total_censored_loss = 0.0
    total_censored_samples = 0
    total_alignment_loss = 0.0
    alignment_steps = 0
    total_samples = 0
    max_grad_norm = 0.0
    for batch in dataloader:
        molecular_numeric = batch["molecular_numeric"].to(device)
        fingerprint = batch["fingerprint"].to(device)
        categorical_ids = {key: value.to(device) for key, value in batch["categorical_ids"].items()}
        adapter_ids = batch.get("adapter_id")
        adapter_ids = adapter_ids.to(device) if adapter_ids is not None else None
        targets = batch["target_value"].to(device)
        sample_weights = batch.get("sample_weight")
        sample_weights = sample_weights.to(device) if sample_weights is not None else None
        toxicity_bin_index = batch.get("toxicity_bin_index")
        toxicity_bin_index = toxicity_bin_index.to(device) if toxicity_bin_index is not None else None
        censored_direction_id = batch.get("censored_direction_id")
        censored_direction_id = censored_direction_id.to(device) if censored_direction_id is not None else None
        task_heads = list(batch["task_head"])
        optimizer.zero_grad()
        loss_components = batch_weighted_loss(
            model,
            molecular_numeric,
            fingerprint,
            categorical_ids,
            adapter_ids,
            targets,
            task_heads,
            loss_fn,
            config,
            device,
            sample_weights=sample_weights,
            toxicity_bin_index=toxicity_bin_index,
            toxicity_bin_loss_weight=config.toxicity_bin_loss_weight,
            censored_direction_id=censored_direction_id,
            censored_loss_weight=config.censored_loss_weight,
            censored_loss_margin=config.censored_loss_margin,
            return_components=True,
        )
        task_loss = loss_components["regression_loss"]
        loss = loss_components["loss"]
        alignment_loss = None
        if alignment_batches is not None and alignment_weight > 0:
            reference_batch = next(alignment_batches)
            alignment_loss = coral_batch_loss(
                model,
                batch,
                reference_batch,
                device=device,
            )
            loss = loss + float(alignment_weight) * alignment_loss
        loss.backward()
        if config.gradient_clip_norm is not None and config.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
            max_grad_norm = max(max_grad_norm, float(grad_norm.detach().cpu()))
        optimizer.step()
        batch_size = len(task_heads)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_task_loss += float(task_loss.detach().cpu()) * batch_size
        toxicity_bin_samples = int(loss_components.get("toxicity_bin_samples", 0))
        if toxicity_bin_samples > 0:
            total_toxicity_bin_loss += float(loss_components["toxicity_bin_loss"].detach().cpu()) * toxicity_bin_samples
            total_toxicity_bin_samples += toxicity_bin_samples
        censored_samples = int(loss_components.get("censored_samples", 0))
        if censored_samples > 0:
            total_censored_loss += float(loss_components["censored_loss"].detach().cpu()) * censored_samples
            total_censored_samples += censored_samples
        if alignment_loss is not None:
            total_alignment_loss += float(alignment_loss.detach().cpu())
            alignment_steps += 1
        total_samples += batch_size
    return {
        "mean_loss": total_loss / max(total_samples, 1),
        "mean_task_loss": total_task_loss / max(total_samples, 1),
        "mean_toxicity_bin_loss": total_toxicity_bin_loss / max(total_toxicity_bin_samples, 1),
        "toxicity_bin_samples": total_toxicity_bin_samples,
        "toxicity_bin_loss_weight": config.toxicity_bin_loss_weight,
        "mean_censored_loss": total_censored_loss / max(total_censored_samples, 1),
        "censored_samples": total_censored_samples,
        "censored_loss_weight": config.censored_loss_weight,
        "mean_alignment_loss": total_alignment_loss / max(alignment_steps, 1) if alignment_steps else 0.0,
        "alignment_steps": alignment_steps,
        "samples": total_samples,
        "max_grad_norm": max_grad_norm,
    }


def evaluate_loss(model: Any, dataloader: Any, loss_fn: Any, config: DeepTrainingConfig, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_toxicity_bin_loss = 0.0
    total_toxicity_bin_samples = 0
    total_censored_loss = 0.0
    total_censored_samples = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            molecular_numeric = batch["molecular_numeric"].to(device)
            fingerprint = batch["fingerprint"].to(device)
            categorical_ids = {key: value.to(device) for key, value in batch["categorical_ids"].items()}
            adapter_ids = batch.get("adapter_id")
            adapter_ids = adapter_ids.to(device) if adapter_ids is not None else None
            targets = batch["target_value"].to(device)
            task_heads = list(batch["task_head"])
            toxicity_bin_index = batch.get("toxicity_bin_index")
            toxicity_bin_index = toxicity_bin_index.to(device) if toxicity_bin_index is not None else None
            censored_direction_id = batch.get("censored_direction_id")
            censored_direction_id = censored_direction_id.to(device) if censored_direction_id is not None else None
            loss_components = batch_weighted_loss(
                model,
                molecular_numeric,
                fingerprint,
                categorical_ids,
                adapter_ids,
                targets,
                task_heads,
                loss_fn,
                config,
                device,
                sample_weights=None,
                toxicity_bin_index=toxicity_bin_index,
                toxicity_bin_loss_weight=config.toxicity_bin_loss_weight,
                censored_direction_id=censored_direction_id,
                censored_loss_weight=config.censored_loss_weight,
                censored_loss_margin=config.censored_loss_margin,
                return_components=True,
            )
            loss = loss_components["loss"]
            batch_size = len(task_heads)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_task_loss += float(loss_components["regression_loss"].detach().cpu()) * batch_size
            toxicity_bin_samples = int(loss_components.get("toxicity_bin_samples", 0))
            if toxicity_bin_samples > 0:
                total_toxicity_bin_loss += float(loss_components["toxicity_bin_loss"].detach().cpu()) * toxicity_bin_samples
                total_toxicity_bin_samples += toxicity_bin_samples
            censored_samples = int(loss_components.get("censored_samples", 0))
            if censored_samples > 0:
                total_censored_loss += float(loss_components["censored_loss"].detach().cpu()) * censored_samples
                total_censored_samples += censored_samples
            total_samples += batch_size
    return {
        "mean_loss": total_loss / max(total_samples, 1),
        "mean_task_loss": total_task_loss / max(total_samples, 1),
        "mean_toxicity_bin_loss": total_toxicity_bin_loss / max(total_toxicity_bin_samples, 1),
        "toxicity_bin_samples": total_toxicity_bin_samples,
        "mean_censored_loss": total_censored_loss / max(total_censored_samples, 1),
        "censored_samples": total_censored_samples,
        "samples": total_samples,
    }


def batch_weighted_loss(
    model: Any,
    molecular_numeric: Any,
    fingerprint: Any,
    categorical_ids: dict[str, Any],
    adapter_ids: Any,
    targets: Any,
    task_heads: list[str],
    loss_fn: Any,
    config: DeepTrainingConfig,
    device: Any,
    sample_weights: Any | None = None,
    toxicity_bin_index: Any | None = None,
    toxicity_bin_loss_weight: float = 0.0,
    censored_direction_id: Any | None = None,
    censored_loss_weight: float = 0.0,
    censored_loss_margin: float = 0.0,
    return_components: bool = False,
) -> Any:
    import torch

    outputs = model(molecular_numeric, fingerprint, categorical_ids, adapter_ids=adapter_ids)
    censored_ids = (
        torch.zeros(targets.shape[0], dtype=torch.long, device=device)
        if censored_direction_id is None
        else censored_direction_id.to(device=device, dtype=torch.long)
    )
    censored_mask = censored_ids != 0
    losses = []
    weight_sum = 0.0
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        task_mask = torch.tensor([head == task_head for head in task_heads], dtype=torch.bool, device=device)
        mask = task_mask & ~censored_mask
        if not bool(mask.any()):
            continue
        weight = float(config.task_weights.get(task_head, 1.0))
        weight_sum += weight
        losses.append(
            weight
            * masked_huber_loss(
                outputs[task_head][mask],
                targets[mask],
                loss_fn=loss_fn,
                config=config,
                sample_weights=None if sample_weights is None else sample_weights[mask],
            )
        )
    if not losses:
        regression_loss = torch.tensor(0.0, device=device)
    else:
        regression_loss = torch.stack(losses).sum() / max(weight_sum, 1e-12)
    toxicity_bin_loss, toxicity_bin_samples = toxicity_bin_auxiliary_loss(
        outputs,
        toxicity_bin_index,
        mode=config.toxicity_binning_mode,
    )
    censored_loss, censored_samples = censored_multitask_loss(
        outputs,
        targets=targets,
        task_heads=task_heads,
        censored_direction_id=censored_ids,
        margin=censored_loss_margin,
    )
    if toxicity_bin_loss is not None and float(toxicity_bin_loss_weight) > 0:
        total_loss = regression_loss + float(toxicity_bin_loss_weight) * toxicity_bin_loss
    else:
        total_loss = regression_loss
        if toxicity_bin_loss is None:
            toxicity_bin_loss = torch.tensor(0.0, device=device)
    if censored_loss is not None and float(censored_loss_weight) > 0:
        total_loss = total_loss + float(censored_loss_weight) * censored_loss
    elif censored_loss is None:
        censored_loss = torch.tensor(0.0, device=device)
    if return_components:
        return {
            "loss": total_loss,
            "regression_loss": regression_loss,
            "toxicity_bin_loss": toxicity_bin_loss,
            "toxicity_bin_samples": toxicity_bin_samples,
            "censored_loss": censored_loss,
            "censored_samples": censored_samples,
        }
    return total_loss


def censored_multitask_loss(
    outputs: Mapping[str, Any],
    *,
    targets: Any,
    task_heads: list[str],
    censored_direction_id: Any,
    margin: float = 0.0,
) -> tuple[Any | None, int]:
    import torch

    losses = []
    count = 0
    device = targets.device
    direction_ids = censored_direction_id.to(device=device, dtype=censored_direction_id.dtype)
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        mask = torch.tensor([head == task_head for head in task_heads], dtype=torch.bool, device=device)
        mask = mask & (direction_ids != 0)
        task_count = int(mask.sum().detach().cpu())
        if task_count <= 0:
            continue
        losses.append(censored_hinge_loss(outputs[task_head][mask], targets[mask], direction_ids[mask], margin=margin))
        count += task_count
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), count


def toxicity_bin_auxiliary_loss(
    outputs: Mapping[str, Any],
    toxicity_bin_index: Any | None,
    *,
    mode: str = "aux_classification",
) -> tuple[Any | None, int]:
    import torch

    logits = outputs.get(TOXICITY_BIN_LOGITS_KEY)
    if logits is None or toxicity_bin_index is None:
        return None, 0
    targets = toxicity_bin_index.to(device=logits.device, dtype=torch.long)
    mask = targets >= 0
    count = int(mask.sum().detach().cpu())
    if count <= 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device), 0
    if str(mode or "").strip().lower() == "ordinal":
        return ordinal_softmax_loss(logits, targets), count
    return torch.nn.functional.cross_entropy(logits[mask], targets[mask]), count


def masked_huber_loss(
    predictions: Any,
    targets: Any,
    *,
    loss_fn: Any,
    config: DeepTrainingConfig,
    sample_weights: Any | None = None,
) -> Any:
    import torch

    if sample_weights is None:
        return loss_fn(predictions, targets)
    losses = torch.nn.functional.huber_loss(
        predictions,
        targets,
        reduction="none",
        delta=float(config.huber_delta),
    )
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype).clamp(min=0.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0:
        return losses.mean()
    return (losses * weights).sum() / weight_sum


def coral_batch_loss(
    model: Any,
    batch: Mapping[str, Any],
    reference_batch: Mapping[str, Any],
    *,
    device: Any,
) -> Any:
    main_shared = shared_representation(model, batch, device=device)
    reference_shared = shared_representation(model, reference_batch, device=device)
    return coral_loss(main_shared, reference_shared)


def shared_representation(model: Any, batch: Mapping[str, Any], *, device: Any) -> Any:
    molecular_numeric = batch["molecular_numeric"].to(device)
    fingerprint = batch["fingerprint"].to(device)
    categorical_ids = {key: value.to(device) for key, value in batch["categorical_ids"].items()}
    adapter_ids = batch.get("adapter_id")
    adapter_ids = adapter_ids.to(device) if adapter_ids is not None else None
    return model.encode_shared(
        molecular_numeric=molecular_numeric,
        fingerprint=fingerprint,
        categorical_ids=categorical_ids,
        adapter_ids=adapter_ids,
    )


def coral_loss(source: Any, target: Any) -> Any:
    import torch

    if source.shape[0] < 2 or target.shape[0] < 2:
        return torch.zeros((), dtype=source.dtype, device=source.device)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T.matmul(source_centered) / max(source.shape[0] - 1, 1)
    target_cov = target_centered.T.matmul(target_centered) / max(target.shape[0] - 1, 1)
    dim = max(int(source.shape[1]), 1)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * dim * dim)


def predict_all(
    model: Any,
    dataset: AggregatedTaskDataset,
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    device: Any,
    target_scaler: TargetScaler | None = None,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_aggregated_task_batch)
    model.eval()
    predictions: list[dict[str, Any]] = []
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            molecular_numeric = batch["molecular_numeric"].to(device)
            fingerprint = batch["fingerprint"].to(device)
            categorical_ids = {key: value.to(device) for key, value in batch["categorical_ids"].items()}
            adapter_ids = batch.get("adapter_id")
            adapter_ids = adapter_ids.to(device) if adapter_ids is not None else None
            outputs = model(molecular_numeric, fingerprint, categorical_ids, adapter_ids=adapter_ids)
            task_heads = list(batch["task_head"])
            targets = batch["target_value"].detach().cpu().tolist()
            for row_idx, task_head in enumerate(task_heads):
                sample_meta = samples[cursor + row_idx]
                y_pred_scaled = float(outputs[task_head][row_idx].detach().cpu())
                scale_key = str(sample_meta.get("target_scale_key", GLOBAL_TARGET_SCALE_KEY))
                y_pred = (
                    y_pred_scaled
                    if target_scaler is None
                    else target_scaler.inverse_transform(scale_key, y_pred_scaled)
                )
                y_true = float(sample_meta.get("target_value_raw", targets[row_idx]))
                prediction_row = {
                    column: sample_meta.get(column, "")
                    for column in PREDICTION_METADATA_COLUMNS
                    if column in sample_meta
                }
                prediction_row.update(
                    {
                        "split_part": sample_meta["split_part"],
                        "task_head": task_head,
                        "target_name": sample_meta.get("target_name", ""),
                        "medium_domain": sample_meta.get("medium_domain", ""),
                        "y_true": y_true,
                        "y_pred": float(y_pred),
                        "y_true_scaled": float(targets[row_idx]),
                        "y_pred_scaled": y_pred_scaled,
                        "target_standardization": "" if target_scaler is None else target_scaler.mode,
                        "residual": y_true - float(y_pred),
                        "abs_error": abs(y_true - float(y_pred)),
                    }
                )
                predictions.append(
                    prediction_row
                )
            cursor += len(task_heads)
    return predictions


def predict_with_feature_perturbation(
    model: Any,
    dataset: AggregatedTaskDataset,
    samples: list[dict[str, Any]],
    *,
    split_parts: tuple[str, ...],
    replicates: int,
    numeric_noise_std: float,
    seed: int,
    batch_size: int,
    device: Any,
    target_scaler: TargetScaler | None = None,
) -> list[dict[str, Any]]:
    selected_parts = {part.lower() for part in split_parts}
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if str(sample.get("split_part", "")).lower() in selected_parts
    ]
    if not indices:
        return []
    noisy_dataset = _NoisyIndexDataset(
        dataset,
        indices,
        replicates=replicates,
        numeric_noise_std=numeric_noise_std,
        target_noise_std=0.0,
        seed=seed,
    )
    noisy_samples: list[dict[str, Any]] = []
    for noisy_idx in range(len(noisy_dataset)):
        source_index, replicate_id = noisy_dataset.source(noisy_idx)
        sample = dict(samples[source_index])
        sample["perturbation_replicate"] = replicate_id
        sample["perturbation_numeric_noise_std"] = float(numeric_noise_std)
        noisy_samples.append(sample)
    return predict_all(
        model,
        noisy_dataset,
        noisy_samples,
        batch_size=batch_size,
        device=device,
        target_scaler=target_scaler,
    )


def summarize_perturbation_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (
            str(row.get("sample_id", "")),
            str(row.get("split_part", "")),
            str(row.get("task_head", "")),
            str(row.get("target_name", "")),
            str(row.get("medium_domain", "")),
            effect_level_label(row.get("effect_level_x", "")),
        )
        grouped.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for (sample_id, split_part, task_head, target_name, medium_domain, effect_level_x), items in sorted(grouped.items()):
        y_true = float(items[0]["y_true"])
        values = np.array([float(item["y_pred"]) for item in items], dtype=float)
        rows.append(
            {
                "sample_id": sample_id,
                "split_part": split_part,
                "task_head": task_head,
                "target_name": target_name,
                "medium_domain": medium_domain,
                "effect_level_x": effect_level_x,
                "n_replicates": len(items),
                "y_true": y_true,
                "y_pred_mean": float(np.mean(values)),
                "y_pred_std": float(np.std(values)),
                "abs_error_mean_prediction": abs(y_true - float(np.mean(values))),
                "mean_abs_error_across_replicates": float(np.mean(np.abs(y_true - values))),
            }
        )
    return rows


def perturbation_mean_predictions(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "split_part": row["split_part"],
            "task_head": row["task_head"],
            "target_name": row["target_name"],
            "medium_domain": row["medium_domain"],
            "effect_level_x": row.get("effect_level_x", ""),
            "y_true": row["y_true"],
            "y_pred": row["y_pred_mean"],
        }
        for row in summary_rows
    ]


def metrics_by_group(
    predictions: list[dict[str, Any]],
    *,
    huber_delta: float,
    group_columns: tuple[str, ...] = ("split_part", "task_head", "target_name", "medium_domain"),
    min_n_for_summary: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = {}
    for row in predictions:
        key = tuple(metric_group_value(row, column) for column in group_columns)
        bucket = grouped.setdefault(key, {"y_true": [], "y_pred": []})
        bucket["y_true"].append(float(row["y_true"]))
        bucket["y_pred"].append(float(row["y_pred"]))
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        metrics = regression_metrics(values["y_true"], values["y_pred"], huber_delta=huber_delta)
        n = len(values["y_true"])
        row = {column: value for column, value in zip(group_columns, key)}
        row.update({"n": n, **metrics})
        row.update(metric_summary_status(row, min_n_for_summary=min_n_for_summary))
        rows.append(row)
    return rows


def metric_group_value(row: Mapping[str, Any], column: str) -> str:
    if column == "effect_level_x":
        return effect_level_label(row.get("effect_level_x", ""))
    return str(row.get(column, ""))


def effect_level_label(value: Any) -> str:
    number = optional_float(value)
    if number is None:
        return "none"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:g}"


def metric_summary_status(row: Mapping[str, Any], *, min_n_for_summary: int) -> dict[str, Any]:
    n = int(float(row.get("n", 0) or 0))
    r2 = optional_float(row.get("r2"))
    if n < int(min_n_for_summary):
        return {
            "metric_valid_for_summary": 0,
            "metric_exclusion_reason": f"n_below_{int(min_n_for_summary)}",
        }
    if r2 is None:
        return {
            "metric_valid_for_summary": 0,
            "metric_exclusion_reason": "undefined_r2",
        }
    return {"metric_valid_for_summary": 1, "metric_exclusion_reason": ""}


def filter_summary_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("metric_valid_for_summary", 0) or 0) == 1]


def build_toxicity_bin_boundary_audit_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], int] = {}
    for row in predictions:
        key = (
            str(row.get("split_part", "")),
            str(row.get("medium_domain", "")),
            str(row.get("unit_family_v2", "")),
            str(row.get("toxicity_bin_status", "")),
            str(row.get("toxicity_bin_boundary_flag", "")),
        )
        grouped[key] = grouped.get(key, 0) + 1
    rows = [
        {
            "split_part": split_part,
            "medium_domain": medium_domain,
            "unit_family_v2": unit_family,
            "toxicity_bin_status": status,
            "toxicity_bin_boundary_flag": boundary_flag,
            "n": count,
        }
        for (split_part, medium_domain, unit_family, status, boundary_flag), count in sorted(grouped.items())
    ]
    return rows or [{"n": 0}]


def build_split_medium_audit_rows(frame: Any, *, split_join_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "split_part" in frame.columns and "medium_domain" in frame.columns:
        counts = frame.groupby(["split_part", "medium_domain"], dropna=False).size().reset_index(name="n")
        for _, item in counts.iterrows():
            rows.append(
                {
                    "audit_section": "loaded_rows_by_split_medium",
                    "split_part": clean_metadata_value(item["split_part"]),
                    "medium_domain": clean_metadata_value(item["medium_domain"]),
                    "n": int(item["n"]),
                }
            )
    removed_rows = split_join_audit.get("removed_rows", []) if isinstance(split_join_audit, Mapping) else []
    for item in removed_rows:
        rows.append({"audit_section": "removed_by_split_medium_contract", **dict(item)})
    if not rows:
        rows.append({"audit_section": "loaded_rows_by_split_medium", "n": int(len(frame))})
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
