from __future__ import annotations

"""Reproducible grouped SHAP and PDP outputs for trained deep QSAR models.

The script reconstructs model inputs solely from the run's manifest and
preprocessing artifacts. It is intended for one target scale and one task head
at a time, so the exported attributions retain an unambiguous response scale.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork
from qsar_tl.training.baseline import add_duration_nonlinear_features, load_split_frame
from qsar_tl.training.deep_experiment import (
    MolecularFeatureBuilder,
    TargetScaler,
    ZScoreCorrection,
    build_deep_samples,
    get_ablation_spec,
    apply_head_routing,
    normalize_head_routing,
)


GLOBAL_COLUMNS = ["feature", "feature_group", "feature_index", "mean_abs_shap"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create grouped SHAP rankings and descriptor/fingerprint PDP outputs for a deep QSAR run."
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing best_model.pt and run artifacts.")
    parser.add_argument("--manifest", default=None, help="Optional manifest.json path; defaults to MODEL_DIR/manifest.json.")
    parser.add_argument(
        "--preprocessing",
        default=None,
        help="Optional preprocessing.json path; defaults to MODEL_DIR/preprocessing.json.",
    )
    parser.add_argument("--db", required=True, help="SQLite modeling-table database used by the trained run.")
    parser.add_argument("--source-table", default=None, help="Source table; defaults to manifest data_source.source_table.")
    parser.add_argument("--split-name", default=None, help="Split name; defaults to the model manifest.")
    parser.add_argument("--split-part", default="test", help="Split part to explain, normally test.")
    parser.add_argument(
        "--target-family",
        required=True,
        help="Exact target_family to explain. This prevents mixing incompatible target scales.",
    )
    parser.add_argument("--task-head", default=None, help="Task head to explain; defaults to the largest filtered group.")
    parser.add_argument("--out-dir", required=True, help="Directory for attribution tables, plots, and provenance.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for background/explanation sampling.")
    parser.add_argument("--background-rows", type=int, default=48, help="Maximum SHAP background rows.")
    parser.add_argument("--explain-rows", type=int, default=32, help="Maximum rows receiving SHAP attributions.")
    parser.add_argument(
        "--shap-max-evals",
        type=int,
        default=1200,
        help="Permutation SHAP evaluation budget per explained sample; auto-raised to the SHAP minimum.",
    )
    parser.add_argument("--global-top-k", type=int, default=20)
    parser.add_argument("--group-top-k", type=int, default=12)
    parser.add_argument("--descriptor-pdp-top-k", type=int, default=5)
    parser.add_argument("--descriptor-pdp-grid-size", type=int, default=11)
    parser.add_argument("--fingerprint-contrast-top-k", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.background_rows < 1 or args.explain_rows < 1:
        raise ValueError("--background-rows and --explain-rows must both be positive.")
    if args.descriptor_pdp_grid_size < 2:
        raise ValueError("--descriptor-pdp-grid-size must be at least 2.")

    model_dir = Path(args.model_dir)
    manifest_path = Path(args.manifest) if args.manifest else model_dir / "manifest.json"
    preprocessing_path = Path(args.preprocessing) if args.preprocessing else model_dir / "preprocessing.json"
    manifest = _read_json(manifest_path)
    preprocessing = _read_json(preprocessing_path)
    split_name = str(args.split_name or manifest.get("split_name") or "")
    source_table = str(args.source_table or _manifest_source_table(manifest) or "")
    if not split_name:
        raise ValueError("Missing --split-name and no split_name was found in the manifest.")
    if not source_table:
        raise ValueError("Missing --source-table and no data_source.source_table was found in the manifest.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame, target_column, task_head = load_explanation_frame(
        db_path=args.db,
        source_table=source_table,
        split_name=split_name,
        split_part=args.split_part,
        target_family=args.target_family,
        task_head=args.task_head,
        head_routing=normalize_head_routing(manifest.get("head_routing", "task")),
    )
    arrays = build_arrays_from_artifacts(frame, preprocessing=preprocessing, target_column=target_column)
    _require_single_scale_key(arrays)
    model = load_model_from_artifacts(
        model_dir,
        manifest=manifest,
        preprocessing=preprocessing,
        device=_resolve_device(args.device),
    )

    background_indices, explain_indices = reproducible_sample_indices(
        len(frame),
        background_rows=args.background_rows,
        explain_rows=args.explain_rows,
        seed=args.seed,
    )
    sampled_rows = sample_provenance(frame, background_indices, explain_indices)
    sampled_rows.to_csv(out_dir / "sampled_rows.csv", index=False)

    shap_values = compute_permutation_shap(
        model,
        arrays,
        task_head=task_head,
        background_indices=background_indices,
        explain_indices=explain_indices,
        max_evals=args.shap_max_evals,
        seed=args.seed,
    )
    catalog = build_feature_catalog(arrays, preprocessing=preprocessing)
    global_ranking = summarize_shap_values(shap_values, catalog)
    global_ranking.to_csv(out_dir / "shap_global_feature_ranking.csv", index=False)
    _plot_ranking(global_ranking.head(args.global_top_k), out_dir / "shap_global_feature_ranking.png", "Global mean |SHAP value|")

    group_outputs = {
        "descriptor": "molecular_descriptor",
        "species_context": "species_context",
        "morgan_fingerprint": "morgan_fingerprint",
    }
    group_rankings: dict[str, pd.DataFrame] = {}
    for output_name, feature_group in group_outputs.items():
        ranking = global_ranking[global_ranking["feature_group"] == feature_group].copy()
        ranking.to_csv(out_dir / f"shap_{output_name}_ranking.csv", index=False)
        _plot_ranking(
            ranking.head(args.group_top_k),
            out_dir / f"shap_{output_name}_ranking.png",
            f"{output_name.replace('_', ' ').title()} mean |SHAP value|",
        )
        group_rankings[output_name] = ranking

    explained_arrays = slice_arrays(arrays, explain_indices)
    descriptor_pdp = descriptor_percentile_pdp(
        model,
        explained_arrays,
        task_head=task_head,
        descriptor_ranking=group_rankings["descriptor"],
        preprocessing=preprocessing,
        top_k=args.descriptor_pdp_top_k,
        grid_size=args.descriptor_pdp_grid_size,
    )
    descriptor_pdp.to_csv(out_dir / "descriptor_percentile_pdp.csv", index=False)
    _plot_descriptor_pdp(descriptor_pdp, out_dir / "descriptor_percentile_pdp.png")

    fingerprint_contrast = fingerprint_bit_contrast(
        model,
        explained_arrays,
        task_head=task_head,
        fingerprint_ranking=group_rankings["morgan_fingerprint"],
        top_k=args.fingerprint_contrast_top_k,
    )
    fingerprint_contrast.to_csv(out_dir / "fingerprint_bit_contrast.csv", index=False)
    _plot_fingerprint_contrast(fingerprint_contrast, out_dir / "fingerprint_bit_contrast.png")

    provenance = {
        "schema_version": 1,
        "method": "shap.PermutationExplainer",
        "model_dir": str(model_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "preprocessing": str(preprocessing_path.resolve()),
        "database": str(Path(args.db).resolve()),
        "source_table": source_table,
        "split_name": split_name,
        "split_part": args.split_part,
        "target_family": args.target_family,
        "target_column": target_column,
        "task_head": task_head,
        "target_scale_key": arrays["target_scale_key"],
        "adapter_id": int(arrays["adapter_id"][0]),
        "sampling": {
            "seed": int(args.seed),
            "eligible_rows": int(len(frame)),
            "background_rows": int(len(background_indices)),
            "explain_rows": int(len(explain_indices)),
            "sampled_rows_file": "sampled_rows.csv",
        },
        "shap": {
            "requested_max_evals": int(args.shap_max_evals),
            "effective_max_evals": int(max(args.shap_max_evals, 2 * len(catalog) + 1)),
            "feature_count": int(len(catalog)),
        },
        "interpretation_contract": {
            "numeric_features": "model-input z scores after the saved training-only preprocessing transform",
            "categorical_features": "observed category identifiers only; the permutation masker swaps observed values",
            "fingerprint_contrast": "counterfactual 0-to-1 bit contrast while holding every other reconstructed model input fixed",
            "adapter": "held fixed after target_family filtering; it is not treated as an explanatory feature",
        },
    }
    (out_dir / "attribution_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for path in sorted(out_dir.iterdir()):
        print(path)


def load_explanation_frame(
    *,
    db_path: str | Path,
    source_table: str,
    split_name: str,
    split_part: str,
    target_family: str,
    task_head: str | None,
    head_routing: str,
) -> tuple[pd.DataFrame, str, str]:
    frame = load_split_frame(
        db_path,
        split_name=split_name,
        source_table=source_table,
        limit=None,
        allow_mixed_target_dimensions=True,
    )
    frame = add_duration_nonlinear_features(frame)
    target_column = "target_value_median" if "target_value_median" in frame.columns else "target_value"
    required = {"split_part", "task_head", "target_family", target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source table {source_table!r} lacks required columns: {missing}")
    target_match = frame["target_family"].fillna("").astype(str).str.strip().eq(str(target_family).strip())
    frame = frame[(frame["split_part"] == split_part) & target_match & frame[target_column].notna()].copy()
    if frame.empty:
        raise ValueError(
            "No eligible rows after filtering "
            f"split_part={split_part!r}, target_family={target_family!r}, source_table={source_table!r}."
        )
    requested_head = str(task_head) if task_head else ""
    if requested_head:
        frame = frame[frame["task_head"].astype(str) == requested_head].copy()
    else:
        requested_head = str(frame["task_head"].value_counts().idxmax())
        frame = frame[frame["task_head"].astype(str) == requested_head].copy()
    if frame.empty:
        raise ValueError(f"No eligible rows for task_head={requested_head!r}.")
    frame = apply_head_routing(frame, mode=head_routing).reset_index(drop=True)
    model_heads = sorted(frame["model_head"].astype(str).unique().tolist())
    if len(model_heads) != 1:
        raise ValueError(f"Explanation rows resolved to multiple model heads: {model_heads}")
    return frame, target_column, model_heads[0]


def build_arrays_from_artifacts(
    frame: pd.DataFrame,
    *,
    preprocessing: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    cache_path = str(preprocessing.get("molecular_feature_cache") or "") or None
    encoder = MolecularFeatureBuilder(fingerprint_size=int(preprocessing["fingerprint_size"]), cache_path=cache_path)
    numeric_stats = {
        str(item["index"]): (float(item["mean"]), float(item["std"]))
        for item in preprocessing["numeric_stats"].values()
    }
    samples = build_deep_samples(
        frame,
        encoder=encoder,
        categorical_maps=preprocessing.get("categorical_maps", {}),
        adapter_map=preprocessing.get("adapter_map", {}),
        numeric_stats=numeric_stats,
        target_column=target_column,
        target_scaler=_target_scaler_from_preprocessing(preprocessing),
        zscore_correction=_zscore_correction_from_preprocessing(preprocessing),
        ablation=get_ablation_spec(str(preprocessing.get("ablation", "full"))),
    )
    if not samples:
        raise ValueError("No deep samples could be reconstructed for explanation.")
    scale_keys = [str(sample.get("target_scale_key", "__global__")) for sample in samples]
    return {
        "numeric": np.asarray([sample["molecular_numeric"] for sample in samples], dtype=np.float32),
        "fingerprint": np.asarray([sample["fingerprint"] for sample in samples], dtype=np.float32),
        "categorical": {
            column: np.asarray([sample["categorical_ids"].get(column, 0) for sample in samples], dtype=np.int64)
            for column in preprocessing.get("categorical_columns", [])
        },
        "adapter_id": np.asarray([sample.get("adapter_id", 0) for sample in samples], dtype=np.int64),
        "target": np.asarray([sample["target_value_raw"] for sample in samples], dtype=np.float32),
        "target_scaler": _target_scaler_from_preprocessing(preprocessing),
        "target_scale_key": scale_keys[0] if scale_keys else "__global__",
        "numeric_names": list(preprocessing["numeric_feature_names"]),
        "categorical_names": list(preprocessing.get("categorical_columns", [])),
    }


def load_model_from_artifacts(
    model_dir: Path,
    *,
    manifest: dict[str, Any],
    preprocessing: dict[str, Any],
    device: str,
) -> EcotoxMultiTaskNetwork:
    import torch

    if int(manifest.get("graph_embedding_dim", 0)) > 0 or bool(
        (manifest.get("ablation_features") or {}).get("use_molecular_graph")
    ):
        raise ValueError("This grouped SHAP script currently supports non-graph deep runs only.")
    state = _load_state_dict(model_dir / "best_model.pt")
    model_dropout = 0.1 if _state_uses_dropout(state) else 0.0
    descriptor_encoder = preprocessing.get("descriptor_encoder", {}) or manifest.get("descriptor_encoder", {}) or {}
    descriptor_groups = {
        str(name): tuple(int(index) for index in indices)
        for name, indices in (descriptor_encoder.get("groups", {}) or {}).items()
        if isinstance(indices, list)
    }
    categorical_embeddings = {
        match.group(1): tuple(value.shape)
        for key, value in state.items()
        if (match := re.match(r"^embeddings\.([^.]*)\.weight$", key)) is not None
    }
    categorical_cardinalities = {name: int(shape[0]) for name, shape in categorical_embeddings.items()}
    categorical_embedding_dims = {name: int(shape[1]) for name, shape in categorical_embeddings.items()}
    adapter_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"^adapters\.(\d+)\.", key)) is not None
    }
    hidden_dims = _hidden_dims_from_state(state)
    bin_weight = state.get("toxicity_bin_classifier.weight")
    ablation_features = manifest.get("ablation_features", {}) or {}
    mgkg_adapter = manifest.get("mgkg_residual_adapter", {}) or {}
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=int(manifest.get("numeric_dim", len(preprocessing["numeric_feature_names"]))),
            fingerprint_dim=int(manifest.get("fingerprint_dim", preprocessing["fingerprint_size"])),
            categorical_cardinalities=categorical_cardinalities,
            categorical_embedding_dims=categorical_embedding_dims,
            effect_level_numeric_indices=tuple(int(index) for index in preprocessing.get("effect_level_numeric_indices", [])),
            descriptor_count=len(preprocessing.get("molecular_descriptor_names", [])),
            descriptor_encoder_mode=str(descriptor_encoder.get("mode", "raw")),
            descriptor_head_dim=int(descriptor_encoder.get("head_dim", 64)),
            descriptor_group_head_dim=int(descriptor_encoder.get("group_head_dim", 16)),
            descriptor_group_indices=descriptor_groups,
            adapter_count=max(adapter_ids, default=-1) + 1,
            hidden_dims=hidden_dims,
            dropout=model_dropout,
            use_molecular_residual="molecular_residual.weight" in state,
            use_adapters=bool(adapter_ids),
            toxicity_bin_count=0 if bin_weight is None else int(bin_weight.shape[0]),
            toxicity_binning_mode="none" if bin_weight is None else "aux_classification",
            task_heads=tuple(str(head) for head in manifest["task_heads"]),
            use_mgkg_residual_adapter=bool(mgkg_adapter.get("enabled", False)),
            mgkg_residual_adapter_bottleneck=int(mgkg_adapter.get("bottleneck_dim", 32)),
            mgkg_residual_adapter_heads=tuple(str(value) for value in mgkg_adapter.get("task_heads", [])),
        )
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict(model: EcotoxMultiTaskNetwork, arrays: dict[str, Any], task_head: str) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        numeric = torch.as_tensor(arrays["numeric"], dtype=torch.float32, device=device)
        fingerprint = torch.as_tensor(arrays["fingerprint"], dtype=torch.float32, device=device)
        categorical = {
            key: torch.as_tensor(value, dtype=torch.long, device=device)
            for key, value in arrays["categorical"].items()
        }
        adapter_ids = torch.as_tensor(arrays["adapter_id"], dtype=torch.long, device=device)
        y_scaled = model(numeric, fingerprint, categorical, adapter_ids=adapter_ids)[task_head].detach().cpu().numpy()
    target_scaler = arrays.get("target_scaler")
    if target_scaler is None:
        return np.asarray(y_scaled, dtype=np.float32)
    scale_key = str(arrays["target_scale_key"])
    return np.asarray([target_scaler.inverse_transform(scale_key, float(value)) for value in y_scaled], dtype=np.float32)


def reproducible_sample_indices(
    n_rows: int,
    *,
    background_rows: int,
    explain_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 1:
        raise ValueError("Cannot sample an empty explanation frame.")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows, dtype=np.int64)
    background = rng.choice(indices, size=min(background_rows, n_rows), replace=False)
    explain = rng.choice(indices, size=min(explain_rows, n_rows), replace=False)
    return np.sort(background), np.sort(explain)


def compute_permutation_shap(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    *,
    task_head: str,
    background_indices: np.ndarray,
    explain_indices: np.ndarray,
    max_evals: int,
    seed: int,
) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency failure is environment-specific
        raise RuntimeError(
            "The optional 'shap' package is required for this analysis. Install it in the training environment "
            "before running the explanation script."
        ) from exc

    np.random.seed(seed)
    background = pack_model_features(arrays, background_indices)
    values = pack_model_features(arrays, explain_indices)

    def model_fn(matrix: np.ndarray) -> np.ndarray:
        return predict(model, unpack_model_features(matrix, arrays), task_head)

    try:
        explainer = shap.Explainer(model_fn, background, algorithm="permutation", seed=seed)
    except TypeError:  # pragma: no cover - exercised with older SHAP releases
        explainer = shap.Explainer(model_fn, background, algorithm="permutation")
    effective_max_evals = max(int(max_evals), 2 * background.shape[1] + 1)
    result = explainer(values, max_evals=effective_max_evals)
    shap_values = np.asarray(result.values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]
    if shap_values.shape != values.shape:
        raise ValueError(f"Unexpected SHAP value shape {shap_values.shape}; expected {values.shape}.")
    return shap_values


def build_feature_catalog(arrays: dict[str, Any], *, preprocessing: dict[str, Any]) -> pd.DataFrame:
    descriptor_names = set(str(name) for name in preprocessing.get("molecular_descriptor_names", []))
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(arrays["numeric_names"]):
        rows.append(
            {
                "feature": str(name),
                "feature_group": "molecular_descriptor" if str(name) in descriptor_names else "species_context",
                "feature_index": int(index),
            }
        )
    offset = len(rows)
    for bit_index in range(arrays["fingerprint"].shape[1]):
        rows.append(
            {
                "feature": f"morgan_bit_{bit_index}",
                "feature_group": "morgan_fingerprint",
                "feature_index": int(bit_index),
            }
        )
    for category_index, name in enumerate(arrays["categorical_names"]):
        rows.append(
            {
                "feature": str(name),
                "feature_group": "species_context",
                "feature_index": int(offset + arrays["fingerprint"].shape[1] + category_index),
            }
        )
    return pd.DataFrame(rows, columns=["feature", "feature_group", "feature_index"])


def summarize_shap_values(shap_values: np.ndarray, catalog: pd.DataFrame) -> pd.DataFrame:
    if shap_values.ndim != 2 or shap_values.shape[1] != len(catalog):
        raise ValueError("SHAP matrix and feature catalog dimensions do not match.")
    result = catalog.copy()
    result["mean_abs_shap"] = np.mean(np.abs(shap_values), axis=0)
    return result.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


def descriptor_percentile_pdp(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    *,
    task_head: str,
    descriptor_ranking: pd.DataFrame,
    preprocessing: dict[str, Any],
    top_k: int,
    grid_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stats = preprocessing.get("numeric_stats", {})
    for _, ranking_row in descriptor_ranking.head(top_k).iterrows():
        feature = str(ranking_row["feature"])
        feature_index = int(ranking_row["feature_index"])
        values = arrays["numeric"][:, feature_index]
        percentiles = np.linspace(0.05, 0.95, grid_size)
        grid = np.quantile(values, percentiles)
        stat = stats.get(feature, {})
        mean = float(stat.get("mean", 0.0))
        std = float(stat.get("std", 1.0))
        for percentile, input_value in zip(percentiles, grid):
            trial = copy_arrays(arrays)
            trial["numeric"][:, feature_index] = float(input_value)
            prediction = predict(model, trial, task_head)
            rows.append(
                {
                    "feature": feature,
                    "feature_index": feature_index,
                    "percentile": float(percentile),
                    "model_input_z": float(input_value),
                    "raw_value_approx": float(mean + std * float(input_value)),
                    "mean_prediction": float(np.mean(prediction)),
                    "std_prediction": float(np.std(prediction)),
                    "n_reference_rows": int(len(prediction)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "feature_index",
            "percentile",
            "model_input_z",
            "raw_value_approx",
            "mean_prediction",
            "std_prediction",
            "n_reference_rows",
        ],
    )


def fingerprint_bit_contrast(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    *,
    task_head: str,
    fingerprint_ranking: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, ranking_row in fingerprint_ranking.head(top_k).iterrows():
        bit_index = int(ranking_row["feature_index"])
        zero_trial = copy_arrays(arrays)
        one_trial = copy_arrays(arrays)
        zero_trial["fingerprint"][:, bit_index] = 0.0
        one_trial["fingerprint"][:, bit_index] = 1.0
        zero_prediction = predict(model, zero_trial, task_head)
        one_prediction = predict(model, one_trial, task_head)
        contrast = one_prediction - zero_prediction
        rows.append(
            {
                "feature": str(ranking_row["feature"]),
                "fingerprint_bit": bit_index,
                "mean_abs_shap": float(ranking_row["mean_abs_shap"]),
                "observed_bit_prevalence": float(np.mean(arrays["fingerprint"][:, bit_index] > 0.5)),
                "mean_prediction_bit_0": float(np.mean(zero_prediction)),
                "mean_prediction_bit_1": float(np.mean(one_prediction)),
                "mean_0_to_1_contrast": float(np.mean(contrast)),
                "std_0_to_1_contrast": float(np.std(contrast)),
                "n_reference_rows": int(len(contrast)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "fingerprint_bit",
            "mean_abs_shap",
            "observed_bit_prevalence",
            "mean_prediction_bit_0",
            "mean_prediction_bit_1",
            "mean_0_to_1_contrast",
            "std_0_to_1_contrast",
            "n_reference_rows",
        ],
    )


def pack_model_features(arrays: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    parts = [arrays["numeric"][indices], arrays["fingerprint"][indices]]
    if arrays["categorical_names"]:
        parts.append(np.column_stack([arrays["categorical"][name][indices] for name in arrays["categorical_names"]]))
    return np.column_stack(parts).astype(np.float32)


def unpack_model_features(values: np.ndarray, template: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    n_numeric = template["numeric"].shape[1]
    n_fingerprint = template["fingerprint"].shape[1]
    numeric = values[:, :n_numeric]
    fingerprint = values[:, n_numeric : n_numeric + n_fingerprint]
    offset = n_numeric + n_fingerprint
    categorical = {
        name: np.rint(values[:, offset + index]).clip(min=0).astype(np.int64)
        for index, name in enumerate(template["categorical_names"])
    }
    adapter_ids = np.asarray(template["adapter_id"], dtype=np.int64)
    if len(np.unique(adapter_ids)) != 1:
        raise ValueError("The target_family/task filter must yield one adapter_id for SHAP explanation.")
    return {
        "numeric": numeric,
        "fingerprint": fingerprint,
        "categorical": categorical,
        "adapter_id": np.full(len(values), int(adapter_ids[0]), dtype=np.int64),
        "target": np.zeros(len(values), dtype=np.float32),
        "target_scaler": template.get("target_scaler"),
        "target_scale_key": template.get("target_scale_key", "__global__"),
        "numeric_names": template["numeric_names"],
        "categorical_names": template["categorical_names"],
    }


def slice_arrays(arrays: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    return {
        "numeric": arrays["numeric"][indices].copy(),
        "fingerprint": arrays["fingerprint"][indices].copy(),
        "categorical": {name: values[indices].copy() for name, values in arrays["categorical"].items()},
        "adapter_id": arrays["adapter_id"][indices].copy(),
        "target": arrays["target"][indices].copy(),
        "target_scaler": arrays.get("target_scaler"),
        "target_scale_key": arrays.get("target_scale_key", "__global__"),
        "numeric_names": list(arrays["numeric_names"]),
        "categorical_names": list(arrays["categorical_names"]),
    }


def copy_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    return slice_arrays(arrays, np.arange(arrays["numeric"].shape[0]))


def sample_provenance(frame: pd.DataFrame, background_indices: np.ndarray, explain_indices: np.ndarray) -> pd.DataFrame:
    columns = [
        column
        for column in ("aggregate_id", "record_id", "casrn", "cas_number", "smiles", "task_head", "target_family", "split_part")
        if column in frame.columns
    ]
    rows: list[pd.DataFrame] = []
    for role, indices in (("background", background_indices), ("explain", explain_indices)):
        selected = frame.iloc[indices][columns].copy()
        selected.insert(0, "analysis_row_index", indices.astype(int))
        selected.insert(1, "sampling_role", role)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _target_scaler_from_preprocessing(preprocessing: dict[str, Any]) -> TargetScaler | None:
    payload = preprocessing.get("target_standardization") or {}
    if not payload:
        return None
    return TargetScaler(
        mode=str(payload.get("mode", "none")),
        target_column=str(payload.get("target_column", "target_value")),
        fit_split_parts=tuple(str(value) for value in payload.get("fit_split_parts", [])),
        stats={
            str(key): {str(stat_key): float(stat_value) for stat_key, stat_value in value.items()}
            for key, value in dict(payload.get("stats", {})).items()
        },
    )


def _zscore_correction_from_preprocessing(preprocessing: dict[str, Any]) -> ZScoreCorrection | None:
    payload = preprocessing.get("feature_zscore_correction") or {}
    if not payload:
        return None
    return ZScoreCorrection(
        enabled=bool(payload.get("enabled", False)),
        threshold=float(payload.get("threshold", 6.0)),
        feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
        fit_split_parts=tuple(str(value) for value in payload.get("fit_split_parts", [])),
        stats={
            str(key): {str(stat_key): float(stat_value) for stat_key, stat_value in value.items()}
            for key, value in dict(payload.get("stats", {})).items()
        },
    )


def _manifest_source_table(manifest: dict[str, Any]) -> str:
    source = manifest.get("data_source", {}) or {}
    return str(source.get("source_table", ""))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required run artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch versions
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload in {path}")
    return state


def _hidden_dims_from_state(state: dict[str, Any]) -> tuple[int, ...]:
    rows = []
    for key, value in state.items():
        match = re.match(r"^trunk\.(\d+)\.weight$", key)
        if match is not None:
            rows.append((int(match.group(1)), int(value.shape[0])))
    hidden_dims = tuple(output_dim for _, output_dim in sorted(rows))
    if not hidden_dims:
        raise ValueError("Checkpoint does not contain trunk linear layers.")
    return hidden_dims


def _state_uses_dropout(state: dict[str, Any]) -> bool:
    indices = [
        int(match.group(1))
        for key in state
        if (match := re.match(r"^trunk\.(\d+)\.weight$", key)) is not None
    ]
    return len(indices) > 1 and sorted(indices)[1] - sorted(indices)[0] == 3


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is unavailable.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_single_scale_key(arrays: dict[str, Any]) -> None:
    adapter_ids = np.asarray(arrays["adapter_id"], dtype=np.int64)
    if len(np.unique(adapter_ids)) != 1:
        raise ValueError(
            "Filtered explanation rows map to multiple adapters. Refine --target-family or choose a compatible split."
        )


def _plot_ranking(frame: pd.DataFrame, path: Path, title: str) -> None:
    if frame.empty:
        return
    display = frame.iloc[::-1]
    fig, axis = plt.subplots(figsize=(7.5, max(2.8, 0.34 * len(display) + 1.2)))
    colors = {
        "molecular_descriptor": "#3B7EA1",
        "species_context": "#4E9F6D",
        "morgan_fingerprint": "#C77B30",
    }
    axis.barh(display["feature"], display["mean_abs_shap"], color=[colors.get(group, "#777777") for group in display["feature_group"]])
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_descriptor_pdp(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    features = list(frame["feature"].drop_duplicates())
    fig, axes = plt.subplots(len(features), 1, figsize=(7.2, max(3.0, 2.35 * len(features))), squeeze=False)
    for axis, feature in zip(axes[:, 0], features):
        selected = frame[frame["feature"] == feature]
        axis.plot(selected["percentile"], selected["mean_prediction"], marker="o", color="#3B7EA1")
        axis.fill_between(
            selected["percentile"],
            selected["mean_prediction"] - selected["std_prediction"],
            selected["mean_prediction"] + selected["std_prediction"],
            color="#3B7EA1",
            alpha=0.16,
        )
        axis.set_ylabel("Mean prediction")
        axis.set_title(feature)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Descriptor percentile in explained rows")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_fingerprint_contrast(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    display = frame.iloc[::-1]
    fig, axis = plt.subplots(figsize=(7.5, max(2.8, 0.34 * len(display) + 1.2)))
    axis.barh(display["feature"], display["mean_0_to_1_contrast"], color="#C77B30")
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Mean prediction contrast: bit 1 minus bit 0")
    axis.set_title("Top fingerprint-bit counterfactual contrasts")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
