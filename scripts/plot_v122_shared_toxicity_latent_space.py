from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork
from qsar_tl.training.baseline import add_duration_nonlinear_features, load_split_frame
from qsar_tl.training.deep_experiment import (
    MolecularFeatureBuilder,
    TargetScaler,
    ZScoreCorrection,
    build_deep_samples,
    get_ablation_spec,
)


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "v1_2_22_no_metal_random_split_remote"
    / "v1.2.22_no_metal_random8_2_seed2042_cebin_lw0025_censored_w0p01"
    / "deep"
    / "full"
    / "M_v2_aquatic_to_soil_ptox_no_metal_adapt_B_random_8_2_f100"
)
DEFAULT_DB = PROJECT_ROOT / "outputs" / "derived" / "modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures" / "supp_shared_toxicity_latent_v122_20260717"
DEFAULT_STYLE = PROJECT_ROOT / "style_journal_clean_v1.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the v1.2.22 no-metal random-mainline 128-D shared toxicity representation "
            "and draw PCA/t-SNE/optional UMAP latent-space panels."
        )
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Directory with best_model.pt, manifest.json, and preprocessing.json.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite modeling database used by the v1.2.22 no-metal run.")
    parser.add_argument("--source-table", default=None, help="Defaults to manifest split_join_audit.source_table.")
    parser.add_argument("--split-name", default=None, help="Defaults to manifest split_name.")
    parser.add_argument("--split-parts", nargs="+", default=["train", "finetune"], help="Split parts to visualize.")
    parser.add_argument("--endpoint-families", nargs="+", default=["ECx", "NOEC", "LOEC"], help="Endpoint families to include.")
    parser.add_argument(
        "--sampling-mode",
        choices=["balanced", "original", "soil_focused"],
        default="balanced",
        help=(
            "balanced: cap each medium x endpoint x split stratum; "
            "original: random sample preserving the filtered candidate distribution; "
            "soil_focused: keep soil-focused soil:aquatic sampling."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=9000, help="Maximum rows after sampling. Use <=0 for no global cap.")
    parser.add_argument("--max-per-stratum", type=int, default=800, help="Maximum rows per medium x endpoint x split stratum.")
    parser.add_argument(
        "--soil-aquatic-ratio",
        type=float,
        default=3.0,
        help="For soil_focused mode, sample up to this many aquatic rows per soil row.",
    )
    parser.add_argument("--max-tsne-rows", type=int, default=6000, help="Maximum rows passed to t-SNE/UMAP.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--style", default=str(DEFAULT_STYLE))
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style = load_style(Path(args.style))
    apply_matplotlib_style(style)

    manifest = read_json(run_dir / "manifest.json")
    preprocessing = read_json(run_dir / "preprocessing.json")
    split_name = args.split_name or str(manifest["split_name"])
    source_table = args.source_table or str((manifest.get("split_join_audit") or {}).get("source_table", ""))
    if not source_table:
        raise ValueError("Missing source table. Pass --source-table or use a manifest with split_join_audit.source_table.")

    frame = load_visualization_frame(
        Path(args.db),
        split_name=split_name,
        source_table=source_table,
        task_heads=tuple(str(value) for value in manifest["task_heads"]),
        split_parts=tuple(args.split_parts),
        endpoint_families=tuple(args.endpoint_families),
        sampling_mode=str(args.sampling_mode),
        seed=int(args.seed),
        max_rows=int(args.max_rows),
        max_per_stratum=int(args.max_per_stratum),
        soil_aquatic_ratio=float(args.soil_aquatic_ratio),
    )
    frame_path = out_dir / "latent_input_rows.csv.gz"
    frame.to_csv(frame_path, index=False, compression="gzip")

    samples = build_samples(frame, preprocessing=preprocessing)
    arrays = samples_to_arrays(samples, preprocessing=preprocessing)
    raw_input = raw_input_matrix(arrays)
    model = load_model(run_dir, manifest=manifest, preprocessing=preprocessing)
    latent = encode_shared(model, arrays, batch_size=int(args.batch_size), apply_adapters=False)
    post_adapter_latent = encode_shared(model, arrays, batch_size=int(args.batch_size), apply_adapters=True)
    if latent.shape[1] != 128:
        raise ValueError(f"Expected a 128-D shared representation, got {latent.shape[1]} dimensions.")

    metadata = build_metadata_frame(samples)
    np.save(out_dir / "raw_input_descriptor_fingerprint_numeric_context.npy", raw_input)
    np.save(out_dir / "shared_toxicity_latent_128d.npy", latent)
    np.save(out_dir / "shared_toxicity_latent_post_adapter_128d.npy", post_adapter_latent)
    metadata.to_csv(out_dir / "shared_toxicity_latent_metadata.csv", index=False)

    raw_reductions = compute_reductions(
        raw_input,
        metadata,
        out_dir=out_dir,
        skipped_prefix="raw_input",
        seed=int(args.seed),
        max_tsne_rows=int(args.max_tsne_rows),
        skip_tsne=bool(args.skip_tsne),
        skip_umap=bool(args.skip_umap),
    )
    for method, coords in raw_reductions.items():
        coords.to_csv(out_dir / f"raw_input_{method.lower()}_coordinates.csv", index=False)

    latent_reductions = compute_reductions(
        latent,
        metadata,
        out_dir=out_dir,
        skipped_prefix="latent",
        seed=int(args.seed),
        max_tsne_rows=int(args.max_tsne_rows),
        skip_tsne=bool(args.skip_tsne),
        skip_umap=bool(args.skip_umap),
    )
    for method, coords in latent_reductions.items():
        coords.to_csv(out_dir / f"latent_{method.lower()}_coordinates.csv", index=False)

    diagnostics = pd.concat(
        [
            compute_diagnostics(raw_input, metadata, reductions=raw_reductions, space_name="raw_input"),
            compute_diagnostics(latent, metadata, reductions=latent_reductions, space_name="latent_128d"),
            compute_diagnostics(post_adapter_latent, metadata, reductions={}, space_name="post_adapter_128d"),
        ],
        ignore_index=True,
    )
    diagnostics.to_csv(out_dir / "latent_space_diagnostics.csv", index=False)

    plot_latent_by_medium(latent_reductions, style, out_dir / "supp_shared_toxicity_latent_by_medium")
    plot_latent_by_endpoint(latent_reductions, style, out_dir / "supp_shared_toxicity_latent_by_endpoint")
    plot_input_vs_latent_umap_by_medium(
        raw_reductions,
        latent_reductions,
        style,
        out_dir / "supp_input_vs_latent_umap_by_medium",
    )
    plot_input_vs_latent_umap_by_endpoint(
        raw_reductions,
        latent_reductions,
        style,
        out_dir / "supp_input_vs_latent_umap_by_endpoint",
    )
    write_manifest(
        out_dir / "figure_manifest.json",
        args=args,
        manifest=manifest,
        preprocessing=preprocessing,
        frame=frame,
        raw_input=raw_input,
        latent=latent,
        post_adapter_latent=post_adapter_latent,
        raw_reductions=raw_reductions,
        latent_reductions=latent_reductions,
        diagnostics=diagnostics,
    )
    write_readme(out_dir / "README.md", diagnostics=diagnostics, reductions=latent_reductions)

    print(f"out_dir={out_dir}")
    print(f"rows={len(metadata)}")
    print(f"raw_input_shape={raw_input.shape[0]}x{raw_input.shape[1]}")
    print(f"latent_shape={latent.shape[0]}x{latent.shape[1]}")
    print(f"post_adapter_latent_shape={post_adapter_latent.shape[0]}x{post_adapter_latent.shape[1]}")
    print(f"diagnostics={out_dir / 'latent_space_diagnostics.csv'}")


def load_visualization_frame(
    db_path: Path,
    *,
    split_name: str,
    source_table: str,
    task_heads: tuple[str, ...],
    split_parts: tuple[str, ...],
    endpoint_families: tuple[str, ...],
    sampling_mode: str,
    seed: int,
    max_rows: int,
    max_per_stratum: int,
    soil_aquatic_ratio: float,
) -> pd.DataFrame:
    frame = load_split_frame(db_path, split_name=split_name, source_table=source_table, limit=None)
    frame = add_duration_nonlinear_features(frame)
    target_column = "target_value_median" if "target_value_median" in frame.columns else "target_value"
    frame = frame[frame[target_column].notna()].copy()
    frame = frame[frame["task_head"].astype(str).isin(task_heads)].copy()
    frame["split_part"] = frame["split_part"].astype(str)
    frame["medium_domain"] = frame["medium_domain"].astype(str)
    frame["endpoint_family"] = frame["task_head"].astype(str).map(endpoint_family)
    frame = frame[frame["split_part"].str.lower().isin({value.lower() for value in split_parts})].copy()
    frame = frame[frame["endpoint_family"].isin(set(endpoint_families))].copy()
    frame = frame[frame["medium_domain"].str.lower().isin({"aquatic", "soil"})].copy()
    if frame.empty:
        raise ValueError("No rows left after endpoint/split/medium filtering.")

    sort_columns = [column for column in ("medium_domain", "endpoint_family", "split_part", "task_head", "aggregate_id") if column in frame.columns]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame.attrs["candidate_counts"] = candidate_count_summary(frame)
    sampled = sample_frame(
        frame,
        mode=sampling_mode,
        max_rows=max_rows,
        max_per_stratum=max_per_stratum,
        soil_aquatic_ratio=soil_aquatic_ratio,
        seed=seed,
    )
    sampled.attrs["candidate_counts"] = frame.attrs["candidate_counts"]
    sampled.attrs["sampling_mode"] = sampling_mode
    sampled.attrs["sampling_parameters"] = {
        "max_rows": int(max_rows),
        "max_per_stratum": int(max_per_stratum),
        "soil_aquatic_ratio": float(soil_aquatic_ratio),
    }
    sampled.attrs["sampled_counts"] = candidate_count_summary(sampled)
    result = sampled.reset_index(drop=True)
    result.attrs = dict(sampled.attrs)
    return result


def sample_frame(
    frame: pd.DataFrame,
    *,
    mode: str,
    max_rows: int,
    max_per_stratum: int,
    soil_aquatic_ratio: float,
    seed: int,
) -> pd.DataFrame:
    normalized = str(mode or "balanced").strip().lower()
    if normalized == "balanced":
        return balanced_sample(
            frame,
            group_columns=("medium_domain", "endpoint_family", "split_part"),
            max_per_group=max_per_stratum,
            max_rows=max_rows,
            seed=seed,
        )
    if normalized == "original":
        if max_rows > 0 and len(frame) > max_rows:
            return frame.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)
        return frame.copy().reset_index(drop=True)
    if normalized == "soil_focused":
        return soil_focused_sample(
            frame,
            max_rows=max_rows,
            aquatic_per_soil=soil_aquatic_ratio,
            seed=seed,
        )
    raise ValueError(f"Unsupported sampling mode: {mode!r}")


def balanced_sample(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    max_per_group: int,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    sampled_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(list(group_columns), dropna=False):
        if len(group) > max_per_group:
            sampled_parts.append(group.sample(n=max_per_group, random_state=seed))
        else:
            sampled_parts.append(group)
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=seed)
    return sampled.sort_values(list(group_columns)).reset_index(drop=True)


def soil_focused_sample(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    aquatic_per_soil: float,
    seed: int,
) -> pd.DataFrame:
    if aquatic_per_soil <= 0:
        raise ValueError("--soil-aquatic-ratio must be positive.")
    soil = frame[frame["medium_domain"].astype(str).str.lower() == "soil"].copy()
    aquatic = frame[frame["medium_domain"].astype(str).str.lower() == "aquatic"].copy()
    if soil.empty or aquatic.empty:
        raise ValueError("soil_focused sampling requires both soil and aquatic rows.")
    target_soil = len(soil)
    target_aquatic = min(len(aquatic), int(round(target_soil * aquatic_per_soil)))
    if max_rows > 0 and target_soil + target_aquatic > max_rows:
        target_soil = max(1, int(math.floor(max_rows / (1.0 + aquatic_per_soil))))
        target_aquatic = max(1, min(len(aquatic), max_rows - target_soil))
        target_soil = min(target_soil, len(soil))
    sampled_soil = soil if target_soil >= len(soil) else soil.sample(n=target_soil, random_state=seed)
    sampled_aquatic = aquatic.sample(n=target_aquatic, random_state=seed)
    sampled = pd.concat([sampled_soil, sampled_aquatic], ignore_index=True)
    return sampled.sort_values(["medium_domain", "endpoint_family", "split_part", "task_head"]).reset_index(drop=True)


def candidate_count_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "by_medium": {str(key): int(value) for key, value in frame["medium_domain"].astype(str).value_counts().items()},
        "by_endpoint_family": {str(key): int(value) for key, value in frame["endpoint_family"].astype(str).value_counts().items()},
        "by_split_medium": {
            f"{str(split_part)}|{str(medium)}": int(value)
            for (split_part, medium), value in frame.groupby(["split_part", "medium_domain"], dropna=False).size().items()
        },
    }


def build_samples(frame: pd.DataFrame, *, preprocessing: Mapping[str, Any]) -> list[dict[str, Any]]:
    cache_path = preprocessing.get("molecular_feature_cache")
    encoder = MolecularFeatureBuilder(
        fingerprint_size=int(preprocessing["fingerprint_size"]),
        cache_path=None if cache_path is None else str(cache_path),
    )
    numeric_stats = {
        str(int(value["index"])): (float(value["mean"]), float(value["std"]))
        for value in dict(preprocessing["numeric_stats"]).values()
    }
    scaler = target_scaler_from_preprocessing(preprocessing)
    zscore = zscore_from_preprocessing(preprocessing)
    return build_deep_samples(
        frame,
        encoder=encoder,
        categorical_maps=dict(preprocessing.get("categorical_maps", {})),
        numeric_stats=numeric_stats,
        target_column=str((preprocessing.get("target_standardization") or {}).get("target_column", "target_value_median")),
        target_scaler=scaler,
        zscore_correction=zscore,
        adapter_map=dict(preprocessing.get("adapter_map", {})),
        descriptor_names=tuple(preprocessing.get("molecular_descriptor_names", [])),
        ablation=get_ablation_spec(str(preprocessing.get("ablation", "full"))),
    )


def samples_to_arrays(samples: list[dict[str, Any]], *, preprocessing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "numeric": np.asarray([sample["molecular_numeric"] for sample in samples], dtype=np.float32),
        "fingerprint": np.asarray([sample["fingerprint"] for sample in samples], dtype=np.float32),
        "categorical": {
            column: np.asarray([sample["categorical_ids"].get(column, 0) for sample in samples], dtype=np.int64)
            for column in preprocessing.get("categorical_columns", [])
        },
        "adapter_id": np.asarray([sample.get("adapter_id", 0) for sample in samples], dtype=np.int64),
    }


def raw_input_matrix(arrays: Mapping[str, Any]) -> np.ndarray:
    """Raw model-input feature block without direct medium-label categorical IDs.

    This combines standardized molecular descriptors, Morgan fingerprints, and
    numeric effect/duration context. Direct categorical medium labels are
    excluded so the raw-space comparison is not a trivial label echo.
    """

    return np.hstack([arrays["numeric"], arrays["fingerprint"]]).astype(np.float32, copy=False)


def load_model(run_dir: Path, *, manifest: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> EcotoxMultiTaskNetwork:
    import torch

    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    hidden_dims = infer_hidden_dims(state)
    trunk_input_dim = int(state["trunk.0.weight"].shape[1])
    numeric_dim = int(preprocessing.get("numeric_feature_names") and len(preprocessing["numeric_feature_names"]) or manifest["numeric_dim"])
    fingerprint_dim = int(preprocessing.get("fingerprint_size", manifest.get("fingerprint_dim", 0)))
    descriptor_names = tuple(str(value) for value in preprocessing.get("molecular_descriptor_names", []))
    categorical_cardinalities = {str(key): int(value) for key, value in dict(manifest.get("categorical_cardinalities", {})).items()}
    descriptor_encoder = dict(preprocessing.get("descriptor_encoder") or manifest.get("descriptor_encoder") or {})
    descriptor_groups = {
        str(key): tuple(int(index) for index in value)
        for key, value in dict(descriptor_encoder.get("groups", {}) or {}).items()
        if isinstance(value, list)
    }
    toxicity_cfg = dict(manifest.get("toxicity_binning") or {})
    ablation_features = dict(manifest.get("ablation_features") or preprocessing.get("ablation_features") or {})
    graph_cfg = dict(manifest.get("molecular_graph") or preprocessing.get("molecular_graph") or {})
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=numeric_dim,
            fingerprint_dim=fingerprint_dim,
            task_heads=tuple(str(value) for value in manifest["task_heads"]),
            categorical_cardinalities=categorical_cardinalities,
            effect_level_numeric_indices=tuple(int(value) for value in preprocessing.get("effect_level_numeric_indices", [])),
            descriptor_count=len(descriptor_names),
            descriptor_encoder_mode=str(descriptor_encoder.get("mode", "raw")),
            descriptor_head_dim=int(descriptor_encoder.get("head_dim", 64)),
            descriptor_group_head_dim=int(descriptor_encoder.get("group_head_dim", 16)),
            descriptor_group_indices=descriptor_groups,
            graph_atom_feature_dim=int(graph_cfg.get("atom_feature_dim", 0) or 0),
            graph_edge_feature_dim=int(graph_cfg.get("edge_feature_dim", 0) or 0),
            graph_embedding_dim=int(graph_cfg.get("embedding_dim", 0) or 0),
            graph_message_steps=int(graph_cfg.get("message_steps", 2) or 2),
            adapter_count=int(manifest.get("adapter_cardinality", preprocessing.get("adapter_cardinality", 0))),
            hidden_dims=hidden_dims,
            dropout=0.10,
            use_molecular_residual=bool(ablation_features.get("use_molecular_residual", True)),
            use_adapters=bool(ablation_features.get("use_medium_adapter", True)),
            toxicity_bin_count=int(toxicity_cfg.get("class_count", 0)) if bool(toxicity_cfg.get("enabled", False)) else 0,
            toxicity_binning_mode=str(toxicity_cfg.get("mode", "none")),
        )
    )
    actual_input_dim = int(model.trunk[0].weight.shape[1])
    if actual_input_dim != trunk_input_dim:
        raise ValueError(f"Model reconstruction mismatch: trunk input {actual_input_dim}, checkpoint {trunk_input_dim}.")
    model.load_state_dict(state)
    model.eval()
    return model


def infer_hidden_dims(state: Mapping[str, Any]) -> tuple[int, ...]:
    rows: list[tuple[int, int]] = []
    for key, value in state.items():
        if key.startswith("trunk.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                rows.append((int(parts[1]), int(value.shape[0])))
    if not rows:
        raise ValueError("Could not infer trunk hidden dimensions from checkpoint.")
    return tuple(dim for _, dim in sorted(rows))


def encode_shared(
    model: EcotoxMultiTaskNetwork,
    arrays: Mapping[str, Any],
    *,
    batch_size: int,
    apply_adapters: bool,
) -> np.ndarray:
    import torch

    n_rows = arrays["numeric"].shape[0]
    chunks: list[np.ndarray] = []
    original_adapters = model.adapters
    if not apply_adapters:
        model.adapters = torch.nn.ModuleList()
    with torch.no_grad():
        try:
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                numeric = torch.as_tensor(arrays["numeric"][start:end], dtype=torch.float32)
                fingerprint = torch.as_tensor(arrays["fingerprint"][start:end], dtype=torch.float32)
                categorical = {
                    key: torch.as_tensor(value[start:end], dtype=torch.long)
                    for key, value in arrays["categorical"].items()
                }
                adapter_id = torch.as_tensor(arrays["adapter_id"][start:end], dtype=torch.long)
                shared = model.encode_shared(
                    molecular_numeric=numeric,
                    fingerprint=fingerprint,
                    categorical_ids=categorical,
                    adapter_ids=adapter_id if apply_adapters else None,
                )
                chunks.append(shared.detach().cpu().numpy().astype(np.float32))
        finally:
            if not apply_adapters:
                model.adapters = original_adapters
    return np.vstack(chunks)


def compute_reductions(
    values: np.ndarray,
    metadata: pd.DataFrame,
    *,
    out_dir: Path,
    skipped_prefix: str,
    seed: int,
    max_tsne_rows: int,
    skip_tsne: bool,
    skip_umap: bool,
) -> dict[str, pd.DataFrame]:
    scaled = StandardScaler().fit_transform(values)
    base_metadata = metadata.reset_index(drop=True)
    reductions: dict[str, pd.DataFrame] = {}

    pca = PCA(n_components=2, random_state=seed)
    pca_xy = pca.fit_transform(scaled)
    reductions["PCA"] = coordinate_frame(base_metadata, pca_xy, "PCA")

    nonlinear_indices = nonlinear_sample_indices(base_metadata, max_rows=max_tsne_rows, seed=seed)
    nonlinear_scaled = scaled[nonlinear_indices]
    nonlinear_metadata = base_metadata.iloc[nonlinear_indices].reset_index(drop=True)
    if not skip_tsne and len(nonlinear_metadata) >= 10:
        perplexity = max(5, min(30, (len(nonlinear_metadata) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
            metric="euclidean",
        )
        reductions["t-SNE"] = coordinate_frame(nonlinear_metadata, tsne.fit_transform(nonlinear_scaled), "t-SNE")

    if not skip_umap:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, max(5, len(nonlinear_metadata) - 1)),
                min_dist=0.15,
                metric="euclidean",
                random_state=seed,
            )
            reductions["UMAP"] = coordinate_frame(nonlinear_metadata, reducer.fit_transform(nonlinear_scaled), "UMAP")
        except Exception as exc:
            skipped = pd.DataFrame(
                [{"method": "UMAP", "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}]
            )
            skipped.to_csv(out_dir / f"{skipped_prefix}_umap_skipped.csv", index=False)
    return reductions


def nonlinear_sample_indices(metadata: pd.DataFrame, *, max_rows: int, seed: int) -> np.ndarray:
    if len(metadata) <= max_rows:
        return np.arange(len(metadata), dtype=int)
    sampled = balanced_sample(
        metadata.reset_index(names="_row_index"),
        group_columns=("medium_domain", "endpoint_family", "split_part"),
        max_per_group=max(1, math.ceil(max_rows / max(1, metadata.groupby(["medium_domain", "endpoint_family", "split_part"]).ngroups))),
        max_rows=max_rows,
        seed=seed,
    )
    return sampled["_row_index"].to_numpy(dtype=int)


def coordinate_frame(metadata: pd.DataFrame, coordinates: np.ndarray, method: str) -> pd.DataFrame:
    result = metadata.copy()
    result.insert(0, "method", method)
    result["x"] = coordinates[:, 0]
    result["y"] = coordinates[:, 1]
    return result


def compute_diagnostics(
    values: np.ndarray,
    metadata: pd.DataFrame,
    *,
    reductions: Mapping[str, pd.DataFrame],
    space_name: str,
) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(values)
    rows = []
    for label_column in ("medium_domain", "endpoint_family", "split_part"):
        rows.append(
            {
                "space": space_name,
                "metric": f"silhouette_by_{label_column}",
                "value": safe_silhouette(scaled, metadata[label_column].astype(str).to_numpy()),
                "n": len(metadata),
                "interpretation": "values near 0 indicate weak global separation; larger positive values indicate stronger separation",
            }
        )
        rows.append(
            {
                "space": space_name,
                "metric": f"knn15_purity_by_{label_column}",
                "value": knn_purity(scaled, metadata[label_column].astype(str).to_numpy(), k=15),
                "n": len(metadata),
                "interpretation": "fraction of nearest neighbors sharing the same label",
            }
        )
    for method, coords in reductions.items():
        xy = coords[["x", "y"]].to_numpy(dtype=float)
        for label_column in ("medium_domain", "endpoint_family", "split_part"):
            rows.append(
                {
                    "space": f"{space_name}_{method}",
                    "metric": f"silhouette_by_{label_column}",
                    "value": safe_silhouette(xy, coords[label_column].astype(str).to_numpy()),
                    "n": len(coords),
                    "interpretation": "2-D projection diagnostic; use as visual support, not as model-selection metric",
                }
            )
    return pd.DataFrame(rows)


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(values, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def knn_purity(values: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if len(values) <= 1:
        return float("nan")
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(values)), metric="euclidean")
    neighbors.fit(values)
    indices = neighbors.kneighbors(values, return_distance=False)[:, 1:]
    if indices.size == 0:
        return float("nan")
    same = labels[indices] == labels[:, None]
    return float(np.mean(same))


def plot_latent_by_medium(reductions: Mapping[str, pd.DataFrame], style: Mapping[str, Any], out_base: Path) -> None:
    palette = {
        "aquatic": style_color(style, "okabe_ito", "blue", "#0072B2"),
        "soil": style_color(style, "okabe_ito", "vermillion", "#D55E00"),
    }
    endpoint_markers = {"ECx": "o", "NOEC": "^", "LOEC": "s"}
    plot_reduction_grid(
        reductions,
        style,
        out_base,
        color_column="medium_domain",
        palette=palette,
        marker_column="endpoint_family",
        markers=endpoint_markers,
        legend_title_color="Medium",
        legend_title_marker="Endpoint",
    )


def plot_latent_by_endpoint(reductions: Mapping[str, pd.DataFrame], style: Mapping[str, Any], out_base: Path) -> None:
    palette = {
        "ECx": style_color(style, "okabe_ito", "blue", "#0072B2"),
        "NOEC": style_color(style, "okabe_ito", "bluish_green", "#009E73"),
        "LOEC": style_color(style, "okabe_ito", "orange", "#E69F00"),
    }
    medium_markers = {"aquatic": "o", "soil": "^"}
    plot_reduction_grid(
        reductions,
        style,
        out_base,
        color_column="endpoint_family",
        palette=palette,
        marker_column="medium_domain",
        markers=medium_markers,
        legend_title_color="Endpoint",
        legend_title_marker="Medium",
    )


def plot_input_vs_latent_umap_by_medium(
    raw_reductions: Mapping[str, pd.DataFrame],
    latent_reductions: Mapping[str, pd.DataFrame],
    style: Mapping[str, Any],
    out_base: Path,
) -> None:
    palette = {
        "aquatic": style_color(style, "okabe_ito", "blue", "#0072B2"),
        "soil": style_color(style, "okabe_ito", "vermillion", "#D55E00"),
    }
    endpoint_markers = {"ECx": "o", "NOEC": "^", "LOEC": "s"}
    plot_space_comparison_umap(
        raw_reductions,
        latent_reductions,
        style,
        out_base,
        color_column="medium_domain",
        palette=palette,
        marker_column="endpoint_family",
        markers=endpoint_markers,
        legend_title_color="Medium",
        legend_title_marker="Endpoint",
    )


def plot_input_vs_latent_umap_by_endpoint(
    raw_reductions: Mapping[str, pd.DataFrame],
    latent_reductions: Mapping[str, pd.DataFrame],
    style: Mapping[str, Any],
    out_base: Path,
) -> None:
    palette = {
        "ECx": style_color(style, "okabe_ito", "blue", "#0072B2"),
        "NOEC": style_color(style, "okabe_ito", "bluish_green", "#009E73"),
        "LOEC": style_color(style, "okabe_ito", "orange", "#E69F00"),
    }
    medium_markers = {"aquatic": "o", "soil": "s"}
    plot_space_comparison_umap(
        raw_reductions,
        latent_reductions,
        style,
        out_base,
        color_column="endpoint_family",
        palette=palette,
        marker_column="medium_domain",
        markers=medium_markers,
        legend_title_color="Endpoint",
        legend_title_marker="Medium",
    )


def plot_space_comparison_umap(
    raw_reductions: Mapping[str, pd.DataFrame],
    latent_reductions: Mapping[str, pd.DataFrame],
    style: Mapping[str, Any],
    out_base: Path,
    *,
    color_column: str,
    palette: Mapping[str, str],
    marker_column: str,
    markers: Mapping[str, str],
    legend_title_color: str,
    legend_title_marker: str,
) -> None:
    method = "UMAP" if "UMAP" in raw_reductions and "UMAP" in latent_reductions else "PCA"
    panels = [
        ("A", "Raw input space", raw_reductions[method]),
        ("B", "128-D shared representation", latent_reductions[method]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 86 / 25.4), squeeze=False)
    for ax, (panel_label, title, coords) in zip(axes.ravel(), panels):
        for marker_value, marker in markers.items():
            marker_subset = coords[coords[marker_column].astype(str) == marker_value]
            for color_value, color in palette.items():
                subset = marker_subset[marker_subset[color_column].astype(str) == color_value]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["x"],
                    subset["y"],
                    s=9,
                    c=color,
                    marker=marker,
                    alpha=0.42,
                    linewidths=0.15,
                    edgecolors="#FFFFFF",
                    rasterized=True,
                )
        ax.set_xlabel(f"{title}\n{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.set_title(f"{panel_label}  {title}", loc="left", fontweight="bold", pad=4)
        style_axes(ax, style)
    add_bottom_legends(
        fig,
        style,
        palette=palette,
        markers=markers,
        legend_title_color=legend_title_color,
        legend_title_marker=legend_title_marker,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=1.4)
    save_figure(fig, out_base)


def plot_reduction_grid(
    reductions: Mapping[str, pd.DataFrame],
    style: Mapping[str, Any],
    out_base: Path,
    *,
    color_column: str,
    palette: Mapping[str, str],
    marker_column: str,
    markers: Mapping[str, str],
    legend_title_color: str,
    legend_title_marker: str,
) -> None:
    methods = list(reductions)
    fig_width = max(85, 58 * len(methods)) / 25.4
    fig_height = 78 / 25.4
    fig, axes = plt.subplots(1, len(methods), figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.ravel()
    for panel_idx, (ax, method) in enumerate(zip(axes_flat, methods), start=1):
        coords = reductions[method]
        for marker_value, marker in markers.items():
            marker_subset = coords[coords[marker_column].astype(str) == marker_value]
            for color_value, color in palette.items():
                subset = marker_subset[marker_subset[color_column].astype(str) == color_value]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["x"],
                    subset["y"],
                    s=9,
                    c=color,
                    marker=marker,
                    alpha=0.42,
                    linewidths=0.15,
                    edgecolors="#FFFFFF",
                    rasterized=True,
                )
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        ax.set_title(f"{chr(64 + panel_idx)}  {method}", loc="left", fontweight="bold", pad=4)
        style_axes(ax, style)
    add_bottom_legends(
        fig,
        style,
        palette=palette,
        markers=markers,
        legend_title_color=legend_title_color,
        legend_title_marker=legend_title_marker,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.2)
    save_figure(fig, out_base)


def add_bottom_legends(
    fig: Any,
    style: Mapping[str, Any],
    *,
    palette: Mapping[str, str],
    markers: Mapping[str, str],
    legend_title_color: str,
    legend_title_marker: str,
) -> None:
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="none", label=key, markerfacecolor=value, markeredgecolor="none", linestyle="none", markersize=5)
        for key, value in palette.items()
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=value, color="#333333", label=key, markerfacecolor="#BBBBBB", markeredgecolor="none", linestyle="none", markersize=5)
        for key, value in markers.items()
    ]
    legend_fontsize = float((style.get("font_sizes_pt", {}) or {}).get("legend", 7.0))
    legend_title_fontsize = float((style.get("font_sizes_pt", {}) or {}).get("legend_title", 7.5))
    fig.legend(
        handles=color_handles,
        title=legend_title_color,
        loc="lower center",
        bbox_to_anchor=(0.36, 0.015),
        ncol=max(1, len(color_handles)),
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.legend(
        handles=marker_handles,
        title=legend_title_marker,
        loc="lower center",
        bbox_to_anchor=(0.70, 0.015),
        ncol=max(1, len(marker_handles)),
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def build_metadata_frame(samples: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, sample in enumerate(samples):
        task_head = str(sample.get("task_head", ""))
        rows.append(
            {
                "row_index": idx,
                "sample_id": sample.get("sample_id", ""),
                "aggregate_id": sample.get("aggregate_id", sample.get("sample_id", "")),
                "record_id": sample.get("record_id", ""),
                "split_part": str(sample.get("split_part", "")),
                "medium_domain": str(sample.get("medium_domain", "")),
                "endpoint_family": endpoint_family(task_head),
                "task_head": task_head,
                "effect_family": sample.get("effect_family", ""),
                "target_name": sample.get("target_name", ""),
                "target_value_raw": sample.get("target_value_raw", ""),
                "cas_number": sample.get("cas_number", ""),
                "dtxsid": sample.get("dtxsid", ""),
                "chemical_name": sample.get("chemical_name", ""),
                "species_number": sample.get("species_number", ""),
                "latin_name": sample.get("latin_name", ""),
                "adapter_name": sample.get("adapter_name", ""),
                "adapter_id": sample.get("adapter_id", ""),
            }
        )
    return pd.DataFrame(rows)


def target_scaler_from_preprocessing(preprocessing: Mapping[str, Any]) -> TargetScaler | None:
    payload = dict(preprocessing.get("target_standardization") or {})
    if not payload:
        return None
    return TargetScaler(
        mode=str(payload.get("mode", "none")),
        target_column=str(payload.get("target_column", "target_value")),
        fit_split_parts=tuple(str(value) for value in payload.get("fit_split_parts", [])),
        stats={
            str(key): {str(stat_key): float(stat_value) for stat_key, stat_value in dict(value).items()}
            for key, value in dict(payload.get("stats", {})).items()
        },
    )


def zscore_from_preprocessing(preprocessing: Mapping[str, Any]) -> ZScoreCorrection | None:
    payload = dict(preprocessing.get("feature_zscore_correction") or {})
    if not payload:
        return None
    return ZScoreCorrection(
        enabled=bool(payload.get("enabled", True)),
        threshold=float(payload.get("threshold", 6.0)),
        feature_names=tuple(str(value) for value in payload.get("feature_names", [])),
        fit_split_parts=tuple(str(value) for value in payload.get("fit_split_parts", [])),
        stats={
            str(key): {str(stat_key): float(stat_value) for stat_key, stat_value in dict(value).items()}
            for key, value in dict(payload.get("stats", {})).items()
        },
    )


def endpoint_family(task_head: str) -> str:
    return str(task_head).split("_", 1)[0]


def load_style(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_matplotlib_style(style: Mapping[str, Any]) -> None:
    for key, value in dict(style.get("matplotlib_rc", {}) or {}).items():
        plt.rcParams[key] = value
    regular = ((style.get("fonts", {}) or {}).get("english", {}) or {}).get("regular")
    family = ((style.get("fonts", {}) or {}).get("english", {}) or {}).get("family_name", "Arial")
    if regular and Path(str(regular)).exists():
        font_manager.fontManager.addfont(str(regular))
        plt.rcParams["font.family"] = family
    else:
        ensure_available_font()
    sizes = dict(style.get("font_sizes_pt", {}) or {})
    plt.rcParams["axes.labelsize"] = float(sizes.get("axis_label", 8.0))
    plt.rcParams["xtick.labelsize"] = float(sizes.get("tick_label", 7.0))
    plt.rcParams["ytick.labelsize"] = float(sizes.get("tick_label", 7.0))
    plt.rcParams["legend.fontsize"] = float(sizes.get("legend", 7.0))
    plt.rcParams["axes.titlesize"] = float(sizes.get("panel_label", 9.0))


def ensure_available_font() -> None:
    family = plt.rcParams.get("font.family", ["DejaVu Sans"])
    candidates = family if isinstance(family, list) else [family]
    for candidate in candidates:
        try:
            font_manager.findfont(str(candidate), fallback_to_default=False)
            return
        except Exception:
            continue
    plt.rcParams["font.family"] = "DejaVu Sans"


def style_axes(ax: Any, style: Mapping[str, Any]) -> None:
    axes_cfg = dict(style.get("axes", {}) or {})
    grid_cfg = dict(style.get("grid", {}) or {})
    for spine in ax.spines.values():
        spine.set_color(axes_cfg.get("spine_color", "#333333"))
        spine.set_linewidth(float(axes_cfg.get("spine_linewidth", 0.75)))
    ax.grid(
        bool(grid_cfg.get("show", True)),
        color=grid_cfg.get("color", "#DDE3EA"),
        linewidth=float(grid_cfg.get("linewidth", 0.45)),
        linestyle=grid_cfg.get("linestyle", "--"),
        alpha=float(grid_cfg.get("alpha", 0.72)),
        zorder=0,
    )
    ax.tick_params(direction="out", width=0.65, length=3.0, color="#333333")


def style_color(style: Mapping[str, Any], group: str, key: str, default: str) -> str:
    return str(((style.get("colors", {}) or {}).get(group, {}) or {}).get(key, default))


def save_figure(fig: Any, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf", ".tiff"):
        fig.savefig(out_base.with_suffix(suffix), dpi=600)
    plt.close(fig)


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    frame: pd.DataFrame,
    raw_input: np.ndarray,
    latent: np.ndarray,
    post_adapter_latent: np.ndarray,
    raw_reductions: Mapping[str, pd.DataFrame],
    latent_reductions: Mapping[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
) -> None:
    payload = {
        "purpose": "Supplementary comparison of raw input space and the v1.2.22 mainline pre-adapter shared toxicity representation.",
        "boundary": "v1.2.22 no-metal/inorganic random interpolation mainline; CAS/fixed chemical holdout is deprecated and not used here.",
        "run_dir": str(Path(args.run_dir)),
        "db": str(Path(args.db)),
        "split_name": manifest.get("split_name"),
        "source_table": (manifest.get("split_join_audit") or {}).get("source_table"),
        "selected_split_parts": list(args.split_parts),
        "selected_endpoint_families": list(args.endpoint_families),
        "sampling_mode": str(args.sampling_mode),
        "sampling_parameters": dict(frame.attrs.get("sampling_parameters", {})),
        "candidate_counts_before_sampling": dict(frame.attrs.get("candidate_counts", {})),
        "sampled_counts": dict(frame.attrs.get("sampled_counts", {})),
        "sampled_rows": int(len(frame)),
        "latent_shape": [int(latent.shape[0]), int(latent.shape[1])],
        "medium_counts": frame["medium_domain"].astype(str).value_counts().to_dict(),
        "endpoint_counts": frame["endpoint_family"].astype(str).value_counts().to_dict(),
        "task_counts": frame["task_head"].astype(str).value_counts().to_dict(),
        "raw_input_definition": (
            "Concatenated standardized molecular descriptors, Morgan fingerprint bits, "
            "and numeric effect/duration context. Direct categorical medium labels are excluded "
            "to avoid a tautological medium-separation comparison."
        ),
        "raw_input_shape": [int(raw_input.shape[0]), int(raw_input.shape[1])],
        "latent_definition": "128-D shared trunk representation before medium-specific adapters are applied.",
        "post_adapter_latent_shape": [int(post_adapter_latent.shape[0]), int(post_adapter_latent.shape[1])],
        "post_adapter_note": "Saved for audit only; main latent-space figures use the pre-adapter shared representation.",
        "diagnostics": diagnostics.to_dict(orient="records"),
        "raw_reduction_methods": {key: int(len(value)) for key, value in raw_reductions.items()},
        "latent_reduction_methods": {key: int(len(value)) for key, value in latent_reductions.items()},
        "preprocessing": {
            "encoder_source": preprocessing.get("encoder_source"),
            "fingerprint_size": preprocessing.get("fingerprint_size"),
            "numeric_feature_names": preprocessing.get("numeric_feature_names"),
            "molecular_descriptor_names": preprocessing.get("molecular_descriptor_names"),
            "effect_level_numeric_indices": preprocessing.get("effect_level_numeric_indices"),
            "target_standardization": preprocessing.get("target_standardization", {}).get("mode"),
            "feature_zscore_correction": preprocessing.get("feature_zscore_correction", {}).get("method"),
        },
        "note": (
            "The local v1.2.22 artifact contains the post-finetuning best_model.pt. "
            "A standalone aquatic-pretrained checkpoint was not present locally, so this output visualizes "
            "the fine-tuned shared representation."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(path: Path, *, diagnostics: pd.DataFrame, reductions: Mapping[str, pd.DataFrame]) -> None:
    medium_raw = metric_value(diagnostics, "raw_input", "silhouette_by_medium_domain")
    medium_latent = metric_value(diagnostics, "latent_128d", "silhouette_by_medium_domain")
    medium_post_adapter = metric_value(diagnostics, "post_adapter_128d", "silhouette_by_medium_domain")
    endpoint_raw = metric_value(diagnostics, "raw_input", "silhouette_by_endpoint_family")
    endpoint_latent = metric_value(diagnostics, "latent_128d", "silhouette_by_endpoint_family")
    manifest_path = path.parent / "figure_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    sampling_mode = str(manifest.get("sampling_mode", "unknown"))
    sampled_counts = manifest.get("sampled_counts", {}) if isinstance(manifest.get("sampled_counts", {}), dict) else {}
    by_medium = sampled_counts.get("by_medium", {}) if isinstance(sampled_counts.get("by_medium", {}), dict) else {}
    methods = ", ".join(reductions.keys())
    text = f"""# v1.2.22 Shared Toxicity Latent Space

This folder contains a Supplementary Figure candidate for the v1.2.22 no-metal/inorganic random-interpolation mainline.
CAS/fixed chemical holdout is deprecated and is not used for this visualization.

Sampling mode: `{sampling_mode}`.
Sampled medium counts: {json.dumps(by_medium, ensure_ascii=False)}.

Generated outputs:

- `supp_shared_toxicity_latent_by_medium.*`: color = medium, marker = endpoint family.
- `supp_shared_toxicity_latent_by_endpoint.*`: color = endpoint family, marker = medium.
- `supp_input_vs_latent_umap_by_medium.*`: raw input UMAP vs 128-D latent UMAP, color = medium.
- `supp_input_vs_latent_umap_by_endpoint.*`: raw input UMAP vs 128-D latent UMAP, color = endpoint family.
- `raw_input_descriptor_fingerprint_numeric_context.npy`: raw feature block used for the input-space comparison.
- `shared_toxicity_latent_128d.npy`: extracted pre-adapter 128-D shared representation.
- `shared_toxicity_latent_post_adapter_128d.npy`: post-adapter representation, kept for audit.
- `shared_toxicity_latent_metadata.csv`: row metadata paired with the latent matrix.
- `latent_*_coordinates.csv`: 2-D coordinates for {methods}.
- `raw_input_*_coordinates.csv`: 2-D coordinates for the raw input space.
- `latent_space_diagnostics.csv`: silhouette and kNN-purity diagnostics.

Current quantitative check:

- Raw input silhouette by medium_domain: {medium_raw:.4g}
- Pre-adapter latent 128-D silhouette by medium_domain: {medium_latent:.4g}
- Post-adapter latent 128-D silhouette by medium_domain: {medium_post_adapter:.4g}
- Raw input silhouette by endpoint_family: {endpoint_raw:.4g}
- Latent 128-D silhouette by endpoint_family: {endpoint_latent:.4g}

Raw input is defined as descriptors + fingerprint + numeric effect/duration context, excluding direct categorical medium labels.
The main latent figures use the pre-adapter shared trunk representation because this is the appropriate layer for a shared-encoder alignment claim.
Interpretation should use the raw-vs-latent diagnostic values directly; do not claim medium compression unless medium separability is actually lower in the latent representation.
"""
    path.write_text(text, encoding="utf-8")


def metric_value(frame: pd.DataFrame, space: str, metric: str) -> float:
    subset = frame[(frame["space"] == space) & (frame["metric"] == metric)]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0]["value"])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
