from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


NUMERIC_FEATURES = (
    "transfer_pred",
    "direct_pred",
    "disagreement",
    "abs_disagreement",
    "transfer_std",
    "direct_std",
    "log10_mw",
    "effect_level",
    "effect_level_present",
)
CATEGORICAL_FEATURES = ("task_family", "taxon_group_l1")
UNKNOWN_CATEGORY = "<unknown>"


@dataclass(frozen=True)
class MetaMetrics:
    n: int
    r2: float
    rmse: float
    mae: float

    def as_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "r2": self.r2, "rmse": self.rmse, "mae": self.mae}


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> MetaMetrics:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("Regression metrics require non-empty aligned one-dimensional arrays.")
    residual = pred - truth
    ss_res = float(np.dot(residual, residual))
    centered = truth - float(np.mean(truth))
    ss_tot = float(np.dot(centered, centered))
    return MetaMetrics(
        n=int(truth.size),
        r2=float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        rmse=math.sqrt(ss_res / truth.size),
        mae=float(np.mean(np.abs(residual))),
    )


def fit_candidate(
    candidate: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    label = str(candidate).strip().upper()
    if label == "E1":
        return fit_constrained_blend(rows)
    if label == "E2":
        return fit_gated_head(rows, seed=seed)
    if label == "E3":
        return fit_residual_head(rows, seed=seed)
    raise ValueError(f"Unsupported E-series candidate: {candidate}")


def predict_candidate(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    candidate = str(model.get("candidate", "")).upper()
    if candidate == "E1":
        direct = _column(rows, "direct_pred")
        transfer = _column(rows, "transfer_pred")
        weight = float(model["transfer_weight"])
        bias = float(model["bias"])
        return direct + weight * (transfer - direct) + bias
    if candidate == "E2":
        features = transform_features(rows, model["feature_transform"])
        beta = np.asarray(model["beta"], dtype=float)
        gate = _sigmoid(features @ beta)
        direct = _column(rows, "direct_pred")
        transfer = _column(rows, "transfer_pred")
        return direct + gate * (transfer - direct)
    if candidate == "E3":
        features = transform_features(rows, model["feature_transform"])
        w1 = np.asarray(model["w1"], dtype=float)
        b1 = np.asarray(model["b1"], dtype=float)
        w2 = np.asarray(model["w2"], dtype=float)
        b2 = float(model["b2"])
        hidden = np.tanh(features @ w1 + b1)
        raw = hidden @ w2 + b2
        correction = float(model["max_correction"]) * np.tanh(raw)
        return _column(rows, "transfer_pred") + correction
    raise ValueError(f"Unsupported fitted candidate: {candidate}")


def fit_constrained_blend(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require_rows(rows)
    truth = _column(rows, "y_true")
    direct = _column(rows, "direct_pred")
    transfer = _column(rows, "transfer_pred")
    best: tuple[float, float, float] | None = None
    for weight in np.linspace(0.0, 1.0, 1001):
        base = direct + float(weight) * (transfer - direct)
        bias = float(np.clip(np.median(truth - base), -0.5, 0.5))
        loss = float(np.mean(_huber_values(base + bias - truth, delta=0.75)))
        item = (loss, -float(weight), bias)
        if best is None or item < best:
            best = item
    assert best is not None
    return {
        "candidate": "E1",
        "schema": "e_series_constrained_blend_v1",
        "transfer_weight": -best[1],
        "direct_weight": 1.0 + best[1],
        "bias": best[2],
        "fit_rows": len(rows),
        "objective": "mean_huber_delta_0p75",
    }


def fit_gated_head(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    max_steps: int = 900,
    learning_rate: float = 0.025,
    l2: float = 0.01,
    batch_size: int = 512,
) -> dict[str, Any]:
    _require_rows(rows)
    fit_rows, monitor_rows = deterministic_meta_split(rows, fraction=0.15, seed=seed)
    transform = fit_feature_transform(fit_rows)
    x_fit = transform_features(fit_rows, transform)
    x_monitor = transform_features(monitor_rows, transform)
    beta = np.zeros(x_fit.shape[1], dtype=float)
    beta[0] = math.log(4.0)

    def loss_and_grad(parameters: np.ndarray, x: np.ndarray, data: Sequence[Mapping[str, Any]]):
        truth = _column(data, "y_true")
        direct = _column(data, "direct_pred")
        transfer = _column(data, "transfer_pred")
        gate = _sigmoid(x @ parameters)
        pred = direct + gate * (transfer - direct)
        derivative = _huber_grad(pred - truth, delta=0.75)
        chain = derivative * gate * (1.0 - gate) * (transfer - direct)
        gradient = x.T @ chain / len(data)
        gradient[1:] += l2 * parameters[1:]
        loss = float(np.mean(_huber_values(pred - truth, delta=0.75)))
        loss += 0.5 * l2 * float(np.dot(parameters[1:], parameters[1:]))
        return loss, gradient

    fit_order = np.random.default_rng(seed + 17).permutation(len(fit_rows))

    def fit_loss_grad(value: np.ndarray, step: int):
        indices = _cyclic_batch_indices(
            fit_order, step=step, batch_size=min(batch_size, len(fit_rows))
        )
        batch_rows = [fit_rows[int(index)] for index in indices]
        return loss_and_grad(value, x_fit[indices], batch_rows)

    beta, best_step = _adam_with_monitor(
        beta,
        fit_loss_grad=fit_loss_grad,
        monitor_loss=lambda value: loss_and_grad(value, x_monitor, monitor_rows)[0],
        max_steps=max_steps,
        learning_rate=learning_rate,
        patience=24,
        monitor_every=10,
    )
    final_transform = fit_feature_transform(rows)
    final_x = transform_features(rows, final_transform)
    final_beta = np.zeros(final_x.shape[1], dtype=float)
    final_beta[0] = math.log(4.0)

    final_order = np.random.default_rng(seed + 23).permutation(len(rows))

    def final_loss_grad(parameters: np.ndarray, step: int):
        indices = _cyclic_batch_indices(
            final_order, step=step, batch_size=min(batch_size, len(rows))
        )
        batch_rows = [rows[int(index)] for index in indices]
        x_batch = final_x[indices]
        truth = _column(batch_rows, "y_true")
        direct = _column(batch_rows, "direct_pred")
        transfer = _column(batch_rows, "transfer_pred")
        gate = _sigmoid(x_batch @ parameters)
        pred = direct + gate * (transfer - direct)
        derivative = _huber_grad(pred - truth, delta=0.75)
        chain = derivative * gate * (1.0 - gate) * (transfer - direct)
        gradient = x_batch.T @ chain / len(batch_rows)
        gradient[1:] += l2 * parameters[1:]
        loss = float(np.mean(_huber_values(pred - truth, delta=0.75)))
        loss += 0.5 * l2 * float(np.dot(parameters[1:], parameters[1:]))
        return loss, gradient

    final_beta = _adam_fixed_steps(
        final_beta,
        loss_grad=final_loss_grad,
        steps=best_step,
        learning_rate=learning_rate,
    )
    return {
        "candidate": "E2",
        "schema": "e_series_context_gate_v1",
        "feature_transform": final_transform,
        "beta": final_beta.tolist(),
        "fit_rows": len(rows),
        "best_steps": best_step,
        "learning_rate": learning_rate,
        "l2": l2,
        "objective": "mean_huber_delta_0p75",
    }


def fit_residual_head(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    hidden_dim: int = 12,
    max_correction: float = 0.75,
    max_steps: int = 1000,
    learning_rate: float = 0.0125,
    l2: float = 0.01,
    batch_size: int = 512,
) -> dict[str, Any]:
    _require_rows(rows)
    fit_rows, monitor_rows = deterministic_meta_split(rows, fraction=0.15, seed=seed + 101)
    transform = fit_feature_transform(fit_rows)
    x_fit = transform_features(fit_rows, transform)
    x_monitor = transform_features(monitor_rows, transform)
    rng = np.random.default_rng(seed)
    parameters = _initial_residual_parameters(x_fit.shape[1], hidden_dim, rng)

    def loss_and_grad(params: np.ndarray, x: np.ndarray, data: Sequence[Mapping[str, Any]]):
        return _residual_loss_grad(
            params,
            x,
            data,
            hidden_dim=hidden_dim,
            max_correction=max_correction,
            l2=l2,
        )

    fit_order = np.random.default_rng(seed + 29).permutation(len(fit_rows))

    def fit_loss_grad(value: np.ndarray, step: int):
        indices = _cyclic_batch_indices(
            fit_order, step=step, batch_size=min(batch_size, len(fit_rows))
        )
        batch_rows = [fit_rows[int(index)] for index in indices]
        return loss_and_grad(value, x_fit[indices], batch_rows)

    parameters, best_step = _adam_with_monitor(
        parameters,
        fit_loss_grad=fit_loss_grad,
        monitor_loss=lambda value: loss_and_grad(value, x_monitor, monitor_rows)[0],
        max_steps=max_steps,
        learning_rate=learning_rate,
        patience=26,
        monitor_every=10,
    )
    final_transform = fit_feature_transform(rows)
    final_x = transform_features(rows, final_transform)
    final_rng = np.random.default_rng(seed)
    final_parameters = _initial_residual_parameters(final_x.shape[1], hidden_dim, final_rng)
    final_order = np.random.default_rng(seed + 31).permutation(len(rows))

    def final_loss_grad(value: np.ndarray, step: int):
        indices = _cyclic_batch_indices(
            final_order, step=step, batch_size=min(batch_size, len(rows))
        )
        batch_rows = [rows[int(index)] for index in indices]
        return _residual_loss_grad(
            value,
            final_x[indices],
            batch_rows,
            hidden_dim=hidden_dim,
            max_correction=max_correction,
            l2=l2,
        )

    final_parameters = _adam_fixed_steps(
        final_parameters,
        loss_grad=final_loss_grad,
        steps=best_step,
        learning_rate=learning_rate,
    )
    w1, b1, w2, b2 = _unpack_residual_parameters(
        final_parameters, final_x.shape[1], hidden_dim
    )
    return {
        "candidate": "E3",
        "schema": "e_series_oof_residual_head_v1",
        "feature_transform": final_transform,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": float(b2),
        "hidden_dim": hidden_dim,
        "max_correction": max_correction,
        "fit_rows": len(rows),
        "best_steps": best_step,
        "learning_rate": learning_rate,
        "l2": l2,
        "objective": "mean_huber_delta_0p75",
    }


def fit_feature_transform(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require_rows(rows)
    numeric = np.asarray([_numeric_feature_row(row) for row in rows], dtype=float)
    means = np.mean(numeric, axis=0)
    stds = np.std(numeric, axis=0)
    stds = np.where(stds > 1.0e-8, stds, 1.0)
    categories: dict[str, list[str]] = {}
    for field in CATEGORICAL_FEATURES:
        values = sorted({_category(row.get(field)) for row in rows})
        categories[field] = values
    return {
        "schema": "e_series_meta_features_v1",
        "numeric_features": list(NUMERIC_FEATURES),
        "numeric_means": means.tolist(),
        "numeric_stds": stds.tolist(),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories": categories,
    }


def transform_features(
    rows: Sequence[Mapping[str, Any]],
    transform: Mapping[str, Any],
) -> np.ndarray:
    numeric = np.asarray([_numeric_feature_row(row) for row in rows], dtype=float)
    means = np.asarray(transform["numeric_means"], dtype=float)
    stds = np.asarray(transform["numeric_stds"], dtype=float)
    blocks = [np.ones((len(rows), 1), dtype=float), (numeric - means) / stds]
    categories = transform["categories"]
    for field in transform["categorical_features"]:
        vocabulary = list(categories[field])
        lookup = {value: idx for idx, value in enumerate(vocabulary)}
        block = np.zeros((len(rows), len(vocabulary) + 1), dtype=float)
        for row_idx, row in enumerate(rows):
            value = _category(row.get(field))
            block[row_idx, lookup.get(value, len(vocabulary))] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def deterministic_meta_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("Meta monitor fraction must be between zero and 0.5.")
    fit: list[Mapping[str, Any]] = []
    monitor: list[Mapping[str, Any]] = []
    threshold = int(round(fraction * 10_000))
    for row in rows:
        identity = str(row.get("aggregate_id", ""))
        digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10_000
        (monitor if bucket < threshold else fit).append(row)
    if len(fit) < 2 or len(monitor) < 2:
        raise ValueError("Deterministic meta split produced an empty or undersized partition.")
    return fit, monitor


def _numeric_feature_row(row: Mapping[str, Any]) -> list[float]:
    transfer = _finite(row.get("transfer_pred"))
    direct = _finite(row.get("direct_pred"))
    disagreement = transfer - direct
    mw = max(_finite(row.get("molecular_weight_g_mol_used"), default=1.0), 1.0e-6)
    effect_raw = row.get("effect_level_x")
    effect_present = 0.0 if effect_raw in (None, "") else 1.0
    effect = _finite(effect_raw)
    return [
        transfer,
        direct,
        disagreement,
        abs(disagreement),
        max(_finite(row.get("transfer_std")), 0.0),
        max(_finite(row.get("direct_std")), 0.0),
        math.log10(mw),
        effect,
        effect_present,
    ]


def _residual_loss_grad(
    parameters: np.ndarray,
    x: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    hidden_dim: int,
    max_correction: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    w1, b1, w2, b2 = _unpack_residual_parameters(parameters, x.shape[1], hidden_dim)
    hidden = np.tanh(x @ w1 + b1)
    raw = hidden @ w2 + b2
    tanh_raw = np.tanh(raw)
    pred = _column(rows, "transfer_pred") + max_correction * tanh_raw
    truth = _column(rows, "y_true")
    derivative = _huber_grad(pred - truth, delta=0.75) / len(rows)
    grad_raw = derivative * max_correction * (1.0 - tanh_raw**2)
    grad_w2 = hidden.T @ grad_raw + l2 * w2
    grad_b2 = float(np.sum(grad_raw))
    grad_hidden = np.outer(grad_raw, w2)
    grad_z1 = grad_hidden * (1.0 - hidden**2)
    grad_w1 = x.T @ grad_z1 + l2 * w1
    grad_b1 = np.sum(grad_z1, axis=0)
    gradient = _pack_residual_parameters(grad_w1, grad_b1, grad_w2, grad_b2)
    loss = float(np.mean(_huber_values(pred - truth, delta=0.75)))
    loss += 0.5 * l2 * (float(np.sum(w1**2)) + float(np.sum(w2**2)))
    return loss, gradient


def _initial_residual_parameters(
    input_dim: int,
    hidden_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    w1 = rng.normal(0.0, 0.02, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=float)
    w2 = rng.normal(0.0, 0.02, size=hidden_dim)
    return _pack_residual_parameters(w1, b1, w2, 0.0)


def _pack_residual_parameters(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> np.ndarray:
    return np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), np.asarray([b2])])


def _unpack_residual_parameters(
    parameters: np.ndarray,
    input_dim: int,
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cursor = 0
    w1_size = input_dim * hidden_dim
    w1 = parameters[cursor : cursor + w1_size].reshape(input_dim, hidden_dim)
    cursor += w1_size
    b1 = parameters[cursor : cursor + hidden_dim]
    cursor += hidden_dim
    w2 = parameters[cursor : cursor + hidden_dim]
    cursor += hidden_dim
    return w1, b1, w2, float(parameters[cursor])


def _adam_with_monitor(
    parameters: np.ndarray,
    *,
    fit_loss_grad: Any,
    monitor_loss: Any,
    max_steps: int,
    learning_rate: float,
    patience: int,
    monitor_every: int,
) -> tuple[np.ndarray, int]:
    value = parameters.copy()
    first = np.zeros_like(value)
    second = np.zeros_like(value)
    best = value.copy()
    best_loss = float("inf")
    best_step = 1
    stale = 0
    for step in range(1, max_steps + 1):
        _, gradient = fit_loss_grad(value, step)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        value -= learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
        if step % max(int(monitor_every), 1) != 0 and step < max_steps:
            continue
        current = float(monitor_loss(value))
        if current < best_loss - 1.0e-8:
            best_loss = current
            best = value.copy()
            best_step = step
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, best_step


def _adam_fixed_steps(
    parameters: np.ndarray,
    *,
    loss_grad: Any,
    steps: int,
    learning_rate: float,
) -> np.ndarray:
    value = parameters.copy()
    first = np.zeros_like(value)
    second = np.zeros_like(value)
    for step in range(1, max(int(steps), 1) + 1):
        _, gradient = loss_grad(value, step)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        value -= learning_rate * first_hat / (np.sqrt(second_hat) + 1.0e-8)
    return value


def _cyclic_batch_indices(
    order: np.ndarray,
    *,
    step: int,
    batch_size: int,
) -> np.ndarray:
    if order.ndim != 1 or order.size == 0:
        raise ValueError("Mini-batch order must be a non-empty one-dimensional array.")
    size = min(max(int(batch_size), 1), int(order.size))
    start = ((max(int(step), 1) - 1) * size) % int(order.size)
    end = start + size
    if end <= order.size:
        return order[start:end]
    return np.concatenate([order[start:], order[: end - order.size]])


def _huber_values(residual: np.ndarray, *, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual**2, delta * (absolute - 0.5 * delta))


def _huber_grad(residual: np.ndarray, *, delta: float) -> np.ndarray:
    return np.where(np.abs(residual) <= delta, residual, delta * np.sign(residual))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_finite(row.get(key)) for row in rows], dtype=float)


def _category(value: Any) -> str:
    normalized = str(value or "").strip()
    return normalized if normalized else UNKNOWN_CATEGORY


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _require_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) < 8:
        raise ValueError("E-series meta fitting requires at least eight aligned OOF rows.")
    for key in ("aggregate_id", "y_true", "direct_pred", "transfer_pred"):
        if any(row.get(key) in (None, "") for row in rows):
            raise ValueError(f"E-series meta rows are missing required field {key!r}.")
