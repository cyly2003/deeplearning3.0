from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for qsar_tl.training.deep_train. "
        "Install the optional ML dependencies, for example: pip install -e .[ml]"
    ) from exc


@dataclass(frozen=True)
class DeepTrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    huber_delta: float = 1.0
    task_weights: Mapping[str, float] = field(default_factory=dict)
    device: str = "cpu"
    seed: int = 42
    num_workers: int = 0


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    mean_loss: float
    samples: int
    task_loss: dict[str, float]


@dataclass(frozen=True)
class TrainingHistory:
    epochs: tuple[EpochMetrics, ...]


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_aggregated_task_batch(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    molecular_numeric = _stack_float_matrix(batch, "molecular_numeric")
    fingerprint = _stack_float_matrix(batch, "fingerprint")
    target_value = torch.tensor(
        [float(sample["target_value"]) for sample in batch],
        dtype=torch.float32,
    )
    task_head = [str(sample["task_head"]) for sample in batch]

    categorical_fields = sorted(
        {
            field_name
            for sample in batch
            for field_name in dict(sample.get("categorical_ids", {}))
        }
    )
    categorical_ids = {
        field_name: torch.tensor(
            [
                int(dict(sample.get("categorical_ids", {})).get(field_name, 0))
                for sample in batch
            ],
            dtype=torch.long,
        )
        for field_name in categorical_fields
    }

    return {
        "molecular_numeric": molecular_numeric,
        "fingerprint": fingerprint,
        "categorical_ids": categorical_ids,
        "task_head": task_head,
        "target_value": target_value,
    }


def train_model(
    model: nn.Module,
    dataset: Any,
    config: DeepTrainingConfig | None = None,
) -> TrainingHistory:
    train_config = config or DeepTrainingConfig()
    if train_config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if train_config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    set_torch_seed(train_config.seed)
    device = torch.device(train_config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_aggregated_task_batch,
    )

    metrics: list[EpochMetrics] = []
    for epoch in range(1, train_config.epochs + 1):
        metrics.append(
            train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                config=train_config,
                epoch=epoch,
                device=device,
            )
        )
    return TrainingHistory(epochs=tuple(metrics))


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    config: DeepTrainingConfig,
    epoch: int = 1,
    device: torch.device | None = None,
) -> EpochMetrics:
    model.train()
    target_device = device or torch.device(config.device)
    loss_fn = nn.HuberLoss(delta=config.huber_delta, reduction="mean")

    total_weighted_loss = 0.0
    total_samples = 0
    task_loss_sum: dict[str, float] = {}
    task_counts: dict[str, int] = {}

    for batch in dataloader:
        molecular_numeric = batch["molecular_numeric"].to(target_device)
        fingerprint = batch["fingerprint"].to(target_device)
        categorical_ids = {
            field_name: ids.to(target_device)
            for field_name, ids in batch["categorical_ids"].items()
        }
        targets = batch["target_value"].to(target_device)
        task_heads = list(batch["task_head"])

        optimizer.zero_grad()
        outputs = model(
            molecular_numeric=molecular_numeric,
            fingerprint=fingerprint,
            categorical_ids=categorical_ids,
        )
        loss = masked_multitask_huber_loss(
            outputs=outputs,
            targets=targets,
            task_heads=task_heads,
            task_weights=config.task_weights,
            loss_fn=loss_fn,
        )
        loss.backward()
        optimizer.step()

        batch_size = len(task_heads)
        total_weighted_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        _accumulate_task_losses(
            outputs=outputs,
            targets=targets,
            task_heads=task_heads,
            task_loss_sum=task_loss_sum,
            task_counts=task_counts,
            loss_fn=loss_fn,
        )

    mean_loss = total_weighted_loss / max(total_samples, 1)
    task_loss = {
        task_head: task_loss_sum[task_head] / task_counts[task_head]
        for task_head in sorted(task_loss_sum)
    }
    return EpochMetrics(
        epoch=epoch,
        mean_loss=mean_loss,
        samples=total_samples,
        task_loss=task_loss,
    )


def masked_multitask_huber_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    task_heads: Sequence[str],
    task_weights: Mapping[str, float],
    loss_fn: nn.Module,
) -> torch.Tensor:
    device = targets.device
    losses: list[torch.Tensor] = []
    for task_head in sorted(set(task_heads)):
        if task_head not in outputs:
            raise KeyError(f"Model did not return prediction head '{task_head}'.")
        mask = torch.tensor(
            [head == task_head for head in task_heads],
            dtype=torch.bool,
            device=device,
        )
        if not bool(mask.any()):
            continue
        weight = float(task_weights.get(task_head, 1.0))
        losses.append(weight * loss_fn(outputs[task_head][mask], targets[mask]))
    if not losses:
        raise ValueError("No task losses were computed for this batch.")
    return torch.stack(losses).sum()


def _accumulate_task_losses(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    task_heads: Sequence[str],
    task_loss_sum: dict[str, float],
    task_counts: dict[str, int],
    loss_fn: nn.Module,
) -> None:
    for task_head in sorted(set(task_heads)):
        mask = torch.tensor(
            [head == task_head for head in task_heads],
            dtype=torch.bool,
            device=targets.device,
        )
        count = int(mask.sum().detach().cpu())
        if count == 0:
            continue
        value = float(loss_fn(outputs[task_head][mask].detach(), targets[mask]).cpu())
        task_loss_sum[task_head] = task_loss_sum.get(task_head, 0.0) + value * count
        task_counts[task_head] = task_counts.get(task_head, 0) + count


def _stack_float_matrix(batch: Sequence[Mapping[str, Any]], field_name: str) -> torch.Tensor:
    rows = [list(sample.get(field_name, [])) for sample in batch]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"All '{field_name}' rows in a batch must have the same width.")
    if width == 0:
        return torch.empty((len(rows), 0), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)
