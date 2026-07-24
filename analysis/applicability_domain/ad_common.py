from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
INPUT_ROOT = ANALYSIS_DIR / "inputs" / "remote_snapshot"
SEEDS = (42, 2042, 3407, 8417)
RUN_PATTERN = re.compile(r"v1\.2\.44_(M00|M10)_.+?种子(42|2042|3407|8417)")
SPLIT_MAP = {
    "finetune_mgkg": "train",
    "finetune_mgkg_validation": "validation",
    "test": "test",
}


def discover_prediction_files(root: Path = INPUT_ROOT) -> dict[tuple[str, int], Path]:
    output: dict[tuple[str, int], Path] = {}
    for path in root.rglob("predictions.csv"):
        match = RUN_PATTERN.search(path.as_posix())
        if not match:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in output:
            raise ValueError(f"Duplicate prediction file for {key}: {path}")
        output[key] = path
    expected = {(route, seed) for route in ("M00", "M10") for seed in SEEDS}
    if set(output) != expected:
        raise ValueError(f"Prediction file set mismatch: missing={sorted(expected-set(output))}")
    return output


def stage_record_ids(frame: pd.DataFrame) -> pd.Series:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from qsar_tl.training.baseline import stage_sample_record_id

    return pd.Series(
        [
            stage_sample_record_id(aggregate_id, medium, target_name, target_family)
            for aggregate_id, medium, target_name, target_family in zip(
                frame["aggregate_id"],
                frame["medium_domain"],
                frame["target_name"],
                frame["target_family"],
            )
        ],
        index=frame.index,
        dtype="string",
    )


def build_route_ensembles(
    files: dict[tuple[str, int], Path] | None = None,
) -> pd.DataFrame:
    files = files or discover_prediction_files()
    static: pd.DataFrame | None = None
    predictions: dict[str, list[pd.Series]] = {"M00": [], "M10": []}
    reference_ids: pd.Series | None = None
    for route in ("M00", "M10"):
        for seed in SEEDS:
            frame = pd.read_csv(files[(route, seed)], low_memory=False)
            frame["record_id"] = stage_record_ids(frame)
            frame = frame.sort_values("record_id").reset_index(drop=True)
            if frame["record_id"].duplicated().any():
                raise ValueError(f"Duplicate record identity for {route}, seed={seed}")
            if reference_ids is None:
                reference_ids = frame["record_id"].copy()
            elif not reference_ids.equals(frame["record_id"]):
                raise ValueError(f"Record identity mismatch for {route}, seed={seed}")
            if static is None and route == "M10" and seed == 42:
                static = frame.drop(columns=["y_pred", "y_pred_scaled", "residual", "abs_error"]).copy()
            predictions[route].append(frame["y_pred"].astype(float).rename(f"{route}_{seed}"))
    if static is None:
        raise ValueError("M10 seed-42 reference frame is unavailable")
    output = static.copy()
    output["analysis_split"] = output["split_part"].map(SPLIT_MAP)
    for route, series in predictions.items():
        matrix = pd.concat(series, axis=1)
        output[f"{route}_prediction"] = matrix.mean(axis=1)
        output[f"{route}_prediction_sd"] = matrix.std(axis=1, ddof=1)
        for seed in SEEDS:
            output[f"{route}_prediction_seed_{seed}"] = matrix[f"{route}_{seed}"]
        output[f"AE_{route}"] = (output["y_true"].astype(float) - output[f"{route}_prediction"]).abs()
    output["delta_AE_M10_minus_M00"] = output["AE_M10"] - output["AE_M00"]
    return output


def regression_metrics(
    frame: pd.DataFrame,
    *,
    truth: str = "y_true",
    prediction: str = "M10_prediction",
) -> dict[str, float | int]:
    pairs = frame[[truth, prediction]].apply(pd.to_numeric, errors="coerce").dropna()
    if pairs.empty:
        return {"n": 0, "r2": math.nan, "rmse": math.nan, "mae": math.nan}
    y = pairs[truth].to_numpy(float)
    p = pairs[prediction].to_numpy(float)
    residual = p - y
    ss_res = float(np.dot(residual, residual))
    centered = y - float(y.mean())
    ss_tot = float(np.dot(centered, centered))
    return {
        "n": int(len(y)),
        "r2": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / len(y)),
        "mae": float(np.abs(residual).mean()),
    }


def within_task_r2(
    frame: pd.DataFrame,
    *,
    truth: str = "y_true",
    prediction: str = "M10_prediction",
    task: str = "model_head",
) -> float:
    selected = frame[[truth, prediction, task]].copy()
    selected[truth] = pd.to_numeric(selected[truth], errors="coerce")
    selected[prediction] = pd.to_numeric(selected[prediction], errors="coerce")
    selected = selected.dropna(subset=[truth, prediction, task])
    residual = selected[prediction].to_numpy(float) - selected[truth].to_numpy(float)
    means = selected.groupby(task, observed=True)[truth].transform("mean").to_numpy(float)
    centered = selected[truth].to_numpy(float) - means
    denominator = float(np.dot(centered, centered))
    return math.nan if denominator <= 0 else 1.0 - float(np.dot(residual, residual)) / denominator


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame.to_csv(path, index=False, encoding="utf-8-sig")
