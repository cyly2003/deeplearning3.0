from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Mapping, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "PyTorch is required for qsar_tl.modeling.network. "
        "Install the optional ML dependencies, for example: pip install -e .[ml]"
    ) from exc


@dataclass(frozen=True)
class DeepModelConfig:
    numeric_dim: int
    fingerprint_dim: int
    task_heads: tuple[str, ...]
    categorical_cardinalities: Mapping[str, int] = field(default_factory=dict)
    categorical_embedding_dims: Mapping[str, int] = field(default_factory=dict)
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1


class EcotoxMultiTaskNetwork(nn.Module):
    """Shared MLP trunk with one regression head per ECOTOX task family."""

    def __init__(self, config: DeepModelConfig) -> None:
        super().__init__()
        if config.numeric_dim < 0 or config.fingerprint_dim < 0:
            raise ValueError("Input dimensions must be non-negative.")
        if not config.task_heads:
            raise ValueError("At least one task head is required.")

        self.config = config
        self.categorical_fields = tuple(sorted(config.categorical_cardinalities))
        self.embeddings = nn.ModuleDict()
        embedding_width = 0
        for field_name in self.categorical_fields:
            cardinality = int(config.categorical_cardinalities[field_name])
            if cardinality <= 0:
                raise ValueError(f"Categorical cardinality for {field_name} must be positive.")
            embedding_dim = int(
                config.categorical_embedding_dims.get(
                    field_name,
                    default_embedding_dim(cardinality),
                )
            )
            self.embeddings[field_name] = nn.Embedding(cardinality, embedding_dim)
            embedding_width += embedding_dim

        trunk_input_dim = config.numeric_dim + config.fingerprint_dim + embedding_width
        if trunk_input_dim <= 0:
            raise ValueError("The model needs at least one input feature.")

        self.trunk = build_mlp(
            input_dim=trunk_input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        head_input_dim = config.hidden_dims[-1] if config.hidden_dims else trunk_input_dim
        self.heads = nn.ModuleDict(
            {task_head: nn.Linear(head_input_dim, 1) for task_head in config.task_heads}
        )

    def forward(
        self,
        molecular_numeric: torch.Tensor,
        fingerprint: torch.Tensor,
        categorical_ids: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        numeric = _ensure_2d_float(molecular_numeric)
        fp = _ensure_2d_float(fingerprint)
        if numeric.shape[0] != fp.shape[0]:
            raise ValueError("molecular_numeric and fingerprint batch sizes differ.")

        parts = [numeric, fp]
        for field_name in self.categorical_fields:
            ids = _categorical_tensor(
                field_name=field_name,
                batch_size=numeric.shape[0],
                categorical_ids=categorical_ids,
                device=numeric.device,
            )
            parts.append(self.embeddings[field_name](ids))

        shared = self.trunk(torch.cat(parts, dim=1))
        return {task_head: head(shared).squeeze(-1) for task_head, head in self.heads.items()}


def default_embedding_dim(cardinality: int) -> int:
    return min(32, max(2, int(sqrt(cardinality)) + 1))


def build_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("Hidden dimensions must be positive.")
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    return nn.Sequential(*layers)


def _ensure_2d_float(values: torch.Tensor) -> torch.Tensor:
    tensor = values.float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _categorical_tensor(
    *,
    field_name: str,
    batch_size: int,
    categorical_ids: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    if categorical_ids is None or field_name not in categorical_ids:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    ids = categorical_ids[field_name].to(device=device, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.repeat(batch_size)
    if ids.ndim != 1:
        raise ValueError(f"Categorical ids for {field_name} must be a 1D tensor.")
    if ids.shape[0] != batch_size:
        raise ValueError(
            f"Categorical ids for {field_name} have batch size {ids.shape[0]}, "
            f"expected {batch_size}."
        )
    return ids
