from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager

from qsar_tl.config import load_config
from qsar_tl.evaluation.metrics import regression_metrics
from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork
from qsar_tl.training.baseline import add_duration_nonlinear_features, load_split_frame
from qsar_tl.training.deep_experiment import (
    ABLATION_SPECS,
    MolecularFeatureBuilder,
    TargetScaler,
    build_deep_samples,
    get_ablation_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain a trained deep QSAR model with permutation, PDP, and SHAP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="Directory containing best_model.pt, manifest.json, preprocessing.json")
    parser.add_argument("--db", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--source-table", default=None)
    parser.add_argument("--task-head", default=None, help="Task head to explain; defaults to largest test task")
    parser.add_argument("--split-part", default="test")
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--pdp-grid-size", type=int, default=25)
    parser.add_argument("--shap-rows", type=int, default=32)
    parser.add_argument("--shap-max-evals", type=int, default=1200)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--style", default="style_journal_clean_v1.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "explainability")
    out_dir.mkdir(parents=True, exist_ok=True)
    style = _load_style(Path(args.style))
    _apply_matplotlib_style(style)

    manifest = _read_json(run_dir / "manifest.json")
    preprocessing = _read_json(run_dir / "preprocessing.json")
    split_name = args.split_name or manifest["split_name"]
    db_path = args.db or config.get("data", {}).get("modeling_tables_db")
    if not db_path:
        raise ValueError("Missing --db or data.modeling_tables_db.")

    frame = load_split_frame(db_path, split_name=split_name, source_table=args.source_table, limit=None)
    frame = add_duration_nonlinear_features(frame)
    target_column = "target_value_median" if "target_value_median" in frame.columns else "target_value"
    frame = frame[frame[target_column].notna()].copy()
    task_head = args.task_head or _largest_task(frame, split_part=args.split_part)
    frame = frame[(frame["split_part"] == args.split_part) & (frame["task_head"] == task_head)].copy()
    if frame.empty:
        raise ValueError(f"No rows for split_part={args.split_part!r}, task_head={task_head!r}.")
    if len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=int(config.get("project", {}).get("seed", 42))).copy()

    arrays = build_arrays(frame, config=config, preprocessing=preprocessing, target_column=target_column)
    model = load_model(run_dir, manifest=manifest, preprocessing=preprocessing, config=config)
    y_pred = predict(model, arrays, task_head)
    metrics = regression_metrics(
        arrays["target"].tolist(),
        y_pred.tolist(),
        huber_delta=float(config.get("training", {}).get("loss", {}).get("delta", 1.0)),
    )
    baseline_path = out_dir / "baseline_metrics.json"
    baseline_path.write_text(json.dumps({"split_name": split_name, "split_part": args.split_part, "task_head": task_head, "n": len(frame), **metrics}, ensure_ascii=False, indent=2), encoding="utf-8")

    permutation = permutation_importance(
        model,
        arrays,
        task_head=task_head,
        baseline_rmse=float(metrics["rmse"]),
        repeats=args.permutation_repeats,
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    permutation_path = out_dir / "permutation_importance.csv"
    permutation.to_csv(permutation_path, index=False)
    plot_importance(permutation.head(20), style, out_dir / "permutation_importance_top20")

    pdp = duration_pdp(
        model,
        arrays,
        frame,
        preprocessing=preprocessing,
        task_head=task_head,
        grid_size=args.pdp_grid_size,
    )
    pdp_path = out_dir / "pdp_duration.csv"
    pdp.to_csv(pdp_path, index=False)
    plot_pdp(pdp, style, out_dir / "pdp_exposure_duration")

    shap_path = ""
    if not args.skip_shap:
        shap_values = shap_importance(
            model,
            arrays,
            task_head=task_head,
            max_rows=args.shap_rows,
            max_evals=args.shap_max_evals,
            seed=int(config.get("project", {}).get("seed", 42)),
        )
        shap_path = str(out_dir / "shap_feature_importance.csv")
        shap_values.to_csv(shap_path, index=False)
        plot_importance(shap_values.head(20), style, out_dir / "shap_importance_top20", value_column="mean_abs_shap")

    print(f"baseline={baseline_path}")
    print(f"permutation={permutation_path}")
    print(f"pdp={pdp_path}")
    if shap_path:
        print(f"shap={shap_path}")
    print(f"figures={out_dir}")


def build_arrays(frame: pd.DataFrame, *, config: dict[str, Any], preprocessing: dict[str, Any], target_column: str) -> dict[str, Any]:
    cache_path = preprocessing.get("molecular_feature_cache") or config.get("experiment", {}).get("molecular_feature_cache")
    encoder = MolecularFeatureBuilder(fingerprint_size=int(preprocessing["fingerprint_size"]), cache_path=cache_path)
    stats = {
        str(item["index"]): (float(item["mean"]), float(item["std"]))
        for item in preprocessing["numeric_stats"].values()
    }
    ablation = get_ablation_spec(str(preprocessing.get("ablation", "full")))
    target_scaler = _target_scaler_from_preprocessing(preprocessing)
    samples = build_deep_samples(
        frame,
        encoder=encoder,
        categorical_maps=preprocessing.get("categorical_maps", {}),
        adapter_map=preprocessing.get("adapter_map", {}),
        numeric_stats=stats,
        target_column=target_column,
        target_scaler=target_scaler,
        ablation=ablation,
    )
    numeric = np.asarray([sample["molecular_numeric"] for sample in samples], dtype=np.float32)
    fingerprint = np.asarray([sample["fingerprint"] for sample in samples], dtype=np.float32)
    categorical = {
        column: np.asarray([sample["categorical_ids"].get(column, 0) for sample in samples], dtype=np.int64)
        for column in preprocessing.get("categorical_columns", [])
    }
    scale_keys = [str(sample.get("target_scale_key", "__global__")) for sample in samples]
    scale_key = max(set(scale_keys), key=scale_keys.count) if scale_keys else "__global__"
    return {
        "numeric": numeric,
        "fingerprint": fingerprint,
        "categorical": categorical,
        "adapter_id": np.asarray([sample.get("adapter_id", 0) for sample in samples], dtype=np.int64),
        "target": np.asarray([sample["target_value_raw"] for sample in samples], dtype=np.float32),
        "target_scaled": np.asarray([sample["target_value_scaled"] for sample in samples], dtype=np.float32),
        "target_scaler": target_scaler,
        "target_scale_key": scale_key,
        "numeric_names": list(preprocessing["numeric_feature_names"]),
        "categorical_names": list(preprocessing.get("categorical_columns", [])),
        "adapter_cardinality": int(preprocessing.get("adapter_cardinality", 0)),
    }


def load_model(run_dir: Path, *, manifest: dict[str, Any], preprocessing: dict[str, Any], config: dict[str, Any]) -> EcotoxMultiTaskNetwork:
    import torch

    hidden_dim = int(config.get("model", {}).get("hidden_dim", 256))
    ablation_features = manifest.get("ablation_features", {}) or {}
    descriptor_names = tuple(preprocessing.get("molecular_descriptor_names", []))
    descriptor_encoder = preprocessing.get("descriptor_encoder", {}) or manifest.get("descriptor_encoder", {}) or {}
    descriptor_groups = {
        str(key): tuple(int(index) for index in value)
        for key, value in (descriptor_encoder.get("groups", {}) or {}).items()
        if isinstance(value, list)
    }
    mgkg_adapter = manifest.get("mgkg_residual_adapter", {}) or {}
    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=len(preprocessing["numeric_feature_names"]),
            fingerprint_dim=int(preprocessing["fingerprint_size"]),
            categorical_cardinalities={
                key: int(value) for key, value in manifest.get("categorical_cardinalities", {}).items()
            },
            adapter_count=int(manifest.get("adapter_cardinality", preprocessing.get("adapter_cardinality", 0))),
            descriptor_count=int(len(descriptor_names)),
            descriptor_encoder_mode=str(descriptor_encoder.get("mode", "raw")),
            descriptor_head_dim=int(descriptor_encoder.get("head_dim", 64)),
            descriptor_group_head_dim=int(descriptor_encoder.get("group_head_dim", 16)),
            descriptor_group_indices=descriptor_groups,
            task_heads=tuple(manifest["task_heads"]),
            hidden_dims=(hidden_dim, max(32, hidden_dim // 2)),
            dropout=float(config.get("model", {}).get("dropout", 0.15)),
            use_molecular_residual=bool(ablation_features.get("use_molecular_residual", True)),
            use_adapters=bool(ablation_features.get("use_medium_adapter", False)),
            use_mgkg_residual_adapter=bool(mgkg_adapter.get("enabled", False)),
            mgkg_residual_adapter_bottleneck=int(mgkg_adapter.get("bottleneck_dim", 32)),
            mgkg_residual_adapter_heads=tuple(str(value) for value in mgkg_adapter.get("task_heads", [])),
        )
    )
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: EcotoxMultiTaskNetwork, arrays: dict[str, Any], task_head: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        numeric = torch.as_tensor(arrays["numeric"], dtype=torch.float32)
        fingerprint = torch.as_tensor(arrays["fingerprint"], dtype=torch.float32)
        categorical = {
            key: torch.as_tensor(value, dtype=torch.long)
            for key, value in arrays["categorical"].items()
        }
        adapter_ids = torch.as_tensor(
            arrays.get("adapter_id", np.zeros(numeric.shape[0], dtype=np.int64)),
            dtype=torch.long,
        )
        y_scaled = model(numeric, fingerprint, categorical, adapter_ids=adapter_ids)[task_head].detach().cpu().numpy()
    target_scaler = arrays.get("target_scaler")
    if target_scaler is None:
        return y_scaled
    scale_key = str(arrays.get("target_scale_key", "__global__"))
    return np.asarray([target_scaler.inverse_transform(scale_key, float(value)) for value in y_scaled], dtype=np.float32)


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


def permutation_importance(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    *,
    task_head: str,
    baseline_rmse: float,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n = arrays["numeric"].shape[0]
    for idx, name in enumerate(arrays["numeric_names"]):
        deltas = []
        for _ in range(repeats):
            trial = _copy_arrays(arrays)
            trial["numeric"][:, idx] = trial["numeric"][rng.permutation(n), idx]
            rmse = regression_metrics(trial["target"].tolist(), predict(model, trial, task_head).tolist())["rmse"]
            deltas.append(float(rmse) - baseline_rmse)
        rows.append({"feature": name, "feature_group": "numeric", "delta_rmse_mean": np.mean(deltas), "delta_rmse_std": np.std(deltas)})
    for name in arrays["categorical_names"]:
        deltas = []
        for _ in range(repeats):
            trial = _copy_arrays(arrays)
            trial["categorical"][name] = trial["categorical"][name][rng.permutation(n)]
            rmse = regression_metrics(trial["target"].tolist(), predict(model, trial, task_head).tolist())["rmse"]
            deltas.append(float(rmse) - baseline_rmse)
        rows.append({"feature": name, "feature_group": "categorical", "delta_rmse_mean": np.mean(deltas), "delta_rmse_std": np.std(deltas)})
    if int(arrays.get("adapter_cardinality", 0)) > 0:
        deltas = []
        for _ in range(repeats):
            trial = _copy_arrays(arrays)
            trial["adapter_id"] = trial["adapter_id"][rng.permutation(n)]
            rmse = regression_metrics(trial["target"].tolist(), predict(model, trial, task_head).tolist())["rmse"]
            deltas.append(float(rmse) - baseline_rmse)
        rows.append({"feature": "target_scale_medium_adapter", "feature_group": "adapter", "delta_rmse_mean": np.mean(deltas), "delta_rmse_std": np.std(deltas)})
    deltas = []
    for _ in range(repeats):
        trial = _copy_arrays(arrays)
        trial["fingerprint"] = trial["fingerprint"][rng.permutation(n), :]
        rmse = regression_metrics(trial["target"].tolist(), predict(model, trial, task_head).tolist())["rmse"]
        deltas.append(float(rmse) - baseline_rmse)
    rows.append({"feature": "morgan_fingerprint_512bit_group", "feature_group": "fingerprint", "delta_rmse_mean": np.mean(deltas), "delta_rmse_std": np.std(deltas)})
    return pd.DataFrame(rows).sort_values("delta_rmse_mean", ascending=False)


def duration_pdp(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    frame: pd.DataFrame,
    *,
    preprocessing: dict[str, Any],
    task_head: str,
    grid_size: int,
) -> pd.DataFrame:
    duration = pd.to_numeric(frame.get("duration_bin_h"), errors="coerce").dropna()
    if duration.empty:
        return pd.DataFrame(columns=["duration_h", "mean_prediction"])
    grid = np.unique(np.quantile(duration.clip(lower=0), np.linspace(0.02, 0.98, grid_size)))
    stats = preprocessing["numeric_stats"]
    feature_index = {name: int(item["index"]) for name, item in stats.items()}
    rows = []
    for value in grid:
        trial = _copy_arrays(arrays)
        transformed = _duration_features(float(value))
        for name, raw_value in transformed.items():
            if name not in feature_index:
                continue
            item = stats[name]
            trial["numeric"][:, feature_index[name]] = (raw_value - float(item["mean"])) / float(item["std"])
        y_pred = predict(model, trial, task_head)
        rows.append({"duration_h": float(value), "mean_prediction": float(np.mean(y_pred)), "std_prediction": float(np.std(y_pred))})
    return pd.DataFrame(rows)


def shap_importance(
    model: EcotoxMultiTaskNetwork,
    arrays: dict[str, Any],
    *,
    task_head: str,
    max_rows: int,
    max_evals: int,
    seed: int,
) -> pd.DataFrame:
    import shap

    rng = np.random.default_rng(seed)
    n = arrays["numeric"].shape[0]
    indices = np.arange(n) if n <= max_rows else rng.choice(n, size=max_rows, replace=False)
    x = _pack_full_matrix(arrays, indices)
    names = _full_feature_names(arrays)

    def model_fn(values: np.ndarray) -> np.ndarray:
        unpacked = _unpack_full_matrix(values, arrays)
        return predict(model, unpacked, task_head)

    explainer = shap.Explainer(model_fn, x, algorithm="permutation")
    result = explainer(x, max_evals=max(max_evals, 2 * x.shape[1] + 1))
    values = np.asarray(result.values)
    rows = []
    for idx, name in enumerate(names):
        rows.append({"feature": name, "mean_abs_shap": float(np.mean(np.abs(values[:, idx])))})
    frame = pd.DataFrame(rows)
    fingerprint_sum = frame[frame["feature"].str.startswith("fingerprint_bit_")]["mean_abs_shap"].sum()
    frame = frame[~frame["feature"].str.startswith("fingerprint_bit_")].copy()
    frame.loc[len(frame)] = {"feature": "morgan_fingerprint_512bit_group", "mean_abs_shap": float(fingerprint_sum)}
    return frame.sort_values("mean_abs_shap", ascending=False)


def _pack_full_matrix(arrays: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    parts = [arrays["numeric"][indices], arrays["fingerprint"][indices]]
    if arrays["categorical_names"]:
        parts.append(np.column_stack([arrays["categorical"][name][indices] for name in arrays["categorical_names"]]))
    if int(arrays.get("adapter_cardinality", 0)) > 0:
        parts.append(arrays["adapter_id"][indices].reshape(-1, 1))
    return np.column_stack(parts).astype(np.float32)


def _unpack_full_matrix(values: np.ndarray, template: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    n_numeric = template["numeric"].shape[1]
    n_fp = template["fingerprint"].shape[1]
    numeric = values[:, :n_numeric]
    fingerprint = values[:, n_numeric : n_numeric + n_fp]
    offset = n_numeric + n_fp
    categorical = {}
    for idx, name in enumerate(template["categorical_names"]):
        categorical[name] = np.rint(values[:, offset + idx]).clip(min=0).astype(np.int64)
    offset += len(template["categorical_names"])
    if int(template.get("adapter_cardinality", 0)) > 0:
        adapter_id = np.rint(values[:, offset]).clip(min=0, max=int(template["adapter_cardinality"]) - 1).astype(np.int64)
    else:
        adapter_id = np.zeros(values.shape[0], dtype=np.int64)
    return {
        "numeric": numeric,
        "fingerprint": fingerprint,
        "categorical": categorical,
        "adapter_id": adapter_id,
        "target": np.zeros(values.shape[0], dtype=np.float32),
        "target_scaler": template.get("target_scaler"),
        "target_scale_key": template.get("target_scale_key", "__global__"),
        "numeric_names": template["numeric_names"],
        "categorical_names": template["categorical_names"],
        "adapter_cardinality": int(template.get("adapter_cardinality", 0)),
    }


def _full_feature_names(arrays: dict[str, Any]) -> list[str]:
    return (
        list(arrays["numeric_names"])
        + [f"fingerprint_bit_{idx}" for idx in range(arrays["fingerprint"].shape[1])]
        + list(arrays["categorical_names"])
        + (["target_scale_medium_adapter"] if int(arrays.get("adapter_cardinality", 0)) > 0 else [])
    )


def _copy_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric": arrays["numeric"].copy(),
        "fingerprint": arrays["fingerprint"].copy(),
        "categorical": {key: value.copy() for key, value in arrays["categorical"].items()},
        "adapter_id": arrays.get("adapter_id", np.zeros(arrays["numeric"].shape[0], dtype=np.int64)).copy(),
        "target": arrays["target"],
        "target_scaler": arrays.get("target_scaler"),
        "target_scale_key": arrays.get("target_scale_key", "__global__"),
        "numeric_names": arrays["numeric_names"],
        "categorical_names": arrays["categorical_names"],
        "adapter_cardinality": int(arrays.get("adapter_cardinality", 0)),
    }


def _duration_features(duration_h: float) -> dict[str, float]:
    duration = max(duration_h, 0.0)
    log_duration = math.log1p(duration)
    centers = np.array([24.0, 48.0, 96.0, 168.0, 336.0, 720.0])
    log_centers = np.log1p(centers)
    gamma = 0.35
    result = {
        "duration_bin_h": duration,
        "duration_log1p_h": log_duration,
        "duration_sqrt_h": math.sqrt(duration),
        "duration_inv_log1p_h": 1.0 / (1.0 + log_duration),
    }
    for center, log_center in zip(centers, log_centers):
        result[f"duration_rbf_{int(center)}h"] = float(np.exp(-gamma * np.square(log_duration - log_center)))
    return result


def _largest_task(frame: pd.DataFrame, *, split_part: str) -> str:
    counts = frame[frame["split_part"] == split_part]["task_head"].value_counts()
    if counts.empty:
        raise ValueError(f"No task heads found for split_part={split_part!r}.")
    return str(counts.index[0])


def plot_importance(frame: pd.DataFrame, style: dict[str, Any], out_base: Path, *, value_column: str = "delta_rmse_mean") -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=_figsize_inches(style, "single_column_tall", default=(3.35, 4.2)))
    ordered = frame.iloc[::-1]
    color = style.get("colors", {}).get("okabe_ito", {}).get("blue", "#0072B2")
    ax.barh(ordered["feature"], ordered[value_column], color=color)
    ax.set_xlabel(value_column.replace("_", " "))
    ax.set_ylabel("")
    _style_axes(ax, style)
    _save_figure(fig, out_base)


def plot_pdp(frame: pd.DataFrame, style: dict[str, Any], out_base: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=_figsize_inches(style, "single_column_wide", default=(3.35, 2.6)))
    color = style.get("colors", {}).get("okabe_ito", {}).get("vermillion", "#D55E00")
    ax.plot(frame["duration_h"], frame["mean_prediction"], color=color, marker="o", linewidth=1.25, markersize=3)
    ax.set_xscale("log")
    ax.set_xlabel("Exposure duration (h)")
    ax.set_ylabel("Mean predicted toxicity")
    _style_axes(ax, style)
    _save_figure(fig, out_base)


def _load_style(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _apply_matplotlib_style(style: dict[str, Any]) -> None:
    for key, value in (style.get("matplotlib_rc", {}) or {}).items():
        plt.rcParams[key] = value
    regular = (style.get("fonts", {}).get("english", {}) or {}).get("regular")
    if regular:
        font_path = Path(regular)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = style.get("fonts", {}).get("english", {}).get("family_name", "Arial")
    _ensure_available_font()
    sizes = style.get("font_sizes_pt", {}) or {}
    if sizes:
        plt.rcParams["axes.labelsize"] = float(sizes.get("axis_label", 8.0))
        plt.rcParams["xtick.labelsize"] = float(sizes.get("tick_label", 7.0))
        plt.rcParams["ytick.labelsize"] = float(sizes.get("tick_label", 7.0))


def _ensure_available_font() -> None:
    family = plt.rcParams.get("font.family", ["DejaVu Sans"])
    candidates = family if isinstance(family, list) else [family]
    for candidate in candidates:
        try:
            font_manager.findfont(str(candidate), fallback_to_default=False)
            return
        except Exception:
            continue
    plt.rcParams["font.family"] = "DejaVu Sans"


def _style_axes(ax: Any, style: dict[str, Any]) -> None:
    axes = style.get("axes", {}) or {}
    grid = style.get("grid", {}) or {}
    for spine in ax.spines.values():
        spine.set_color(axes.get("spine_color", "#333333"))
        spine.set_linewidth(float(axes.get("spine_linewidth", 0.75)))
    ax.grid(
        bool(grid.get("show", True)),
        color=grid.get("color", "#DDE3EA"),
        linewidth=float(grid.get("linewidth", 0.45)),
        linestyle=grid.get("linestyle", "--"),
        alpha=float(grid.get("alpha", 0.72)),
    )


def _figsize_inches(style: dict[str, Any], key: str, *, default: tuple[float, float]) -> tuple[float, float]:
    size = (style.get("figure_sizes_mm", {}) or {}).get(key)
    if not size:
        return default
    return float(size[0]) / 25.4, float(size[1]) / 25.4


def _save_figure(fig: Any, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
