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


TOXICITY_BIN_LOGITS_KEY = "__toxicity_bin_logits__"


@dataclass(frozen=True)
class DeepModelConfig:
    numeric_dim: int
    fingerprint_dim: int
    task_heads: tuple[str, ...]
    categorical_cardinalities: Mapping[str, int] = field(default_factory=dict)
    categorical_embedding_dims: Mapping[str, int] = field(default_factory=dict)
    effect_level_numeric_indices: tuple[int, ...] = ()
    adapter_count: int = 0
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    use_molecular_residual: bool = True
    use_adapters: bool = True
    toxicity_bin_count: int = 0
    toxicity_binning_mode: str = "none"


class EcotoxMultiTaskNetwork(nn.Module):
    """Shared multitask trunk with molecular-signal residual integration."""

    def __init__(self, config: DeepModelConfig) -> None:
        super().__init__()
        if config.numeric_dim < 0 or config.fingerprint_dim < 0:
            raise ValueError("Input dimensions must be non-negative.")
        if not config.task_heads:
            raise ValueError("At least one task head is required.")

        self.config = config
        self.effect_level_numeric_indices = tuple(int(index) for index in config.effect_level_numeric_indices)
        invalid_effect_indices = [
            index
            for index in self.effect_level_numeric_indices
            if index < 0 or index >= int(config.numeric_dim)
        ]
        if invalid_effect_indices:
            raise ValueError(
                "effect_level_numeric_indices must point inside the numeric feature matrix: "
                f"{invalid_effect_indices}"
            )
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

        molecular_input_dim = config.numeric_dim + config.fingerprint_dim
        trunk_input_dim = molecular_input_dim + embedding_width
        if trunk_input_dim <= 0:
            raise ValueError("The model needs at least one input feature.")

        self.trunk = build_mlp(
            input_dim=trunk_input_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        head_input_dim = config.hidden_dims[-1] if config.hidden_dims else trunk_input_dim
        self.molecular_residual = (
            nn.Linear(molecular_input_dim, head_input_dim)
            if config.use_molecular_residual and molecular_input_dim > 0
            else None
        )
        self.effect_level_encoder = (
            nn.Sequential(
                nn.Linear(len(self.effect_level_numeric_indices), head_input_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(head_input_dim, head_input_dim),
            )
            if self.effect_level_numeric_indices
            else None
        )
        self.adapters = nn.ModuleList()
        if config.use_adapters and config.adapter_count > 0:
            self.adapters.extend(
                build_adapter(head_input_dim, dropout=config.dropout)
                for _ in range(int(config.adapter_count))
            )
        self.heads = nn.ModuleDict(
            {task_head: nn.Linear(head_input_dim, 1) for task_head in config.task_heads}
        )
        toxicity_bin_count = max(0, int(config.toxicity_bin_count))
        toxicity_mode = str(config.toxicity_binning_mode or "none").strip().lower()
        self.toxicity_bin_classifier = (
            nn.Linear(head_input_dim, toxicity_bin_count)
            if toxicity_bin_count > 0 and toxicity_mode in {"aux_classification", "ordinal", "soft_expert"}
            else None
        )

    def forward(
        self,
        molecular_numeric: torch.Tensor,
        fingerprint: torch.Tensor,
        categorical_ids: Mapping[str, torch.Tensor] | None = None,
        adapter_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shared = self.encode_shared(
            molecular_numeric=molecular_numeric,
            fingerprint=fingerprint,
            categorical_ids=categorical_ids,
            adapter_ids=adapter_ids,
        )
        outputs = {task_head: head(shared).squeeze(-1) for task_head, head in self.heads.items()}
        if self.toxicity_bin_classifier is not None:
            outputs[TOXICITY_BIN_LOGITS_KEY] = self.toxicity_bin_classifier(shared)
        return outputs

    def encode_shared(
        self,
        molecular_numeric: torch.Tensor,
        fingerprint: torch.Tensor,
        categorical_ids: Mapping[str, torch.Tensor] | None = None,
        adapter_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numeric = _ensure_2d_float(molecular_numeric)
        fp = _ensure_2d_float(fingerprint)
        if numeric.shape[0] != fp.shape[0]:
            raise ValueError("molecular_numeric and fingerprint batch sizes differ.")

        molecular = torch.cat([numeric, fp], dim=1)
        parts = [molecular]
        for field_name in self.categorical_fields:
            ids = _categorical_tensor(
                field_name=field_name,
                batch_size=numeric.shape[0],
                categorical_ids=categorical_ids,
                device=numeric.device,
            )
            parts.append(self.embeddings[field_name](ids))

        shared = self.trunk(torch.cat(parts, dim=1))
        if self.molecular_residual is not None:
            shared = shared + self.molecular_residual(molecular)
        if self.effect_level_encoder is not None:
            indices = torch.tensor(self.effect_level_numeric_indices, dtype=torch.long, device=numeric.device)
            shared = shared + self.effect_level_encoder(numeric.index_select(1, indices))
        return self._apply_adapters(shared, adapter_ids)

    def _apply_adapters(self, shared: torch.Tensor, adapter_ids: torch.Tensor | None) -> torch.Tensor:
        if not self.adapters:
            return shared
        ids = _adapter_tensor(
            batch_size=shared.shape[0],
            adapter_ids=adapter_ids,
            device=shared.device,
            adapter_count=len(self.adapters),
        )
        adapted = shared.clone()
        for adapter_idx, adapter in enumerate(self.adapters):
            mask = ids == adapter_idx
            if bool(mask.any()):
                adapted[mask] = shared[mask] + adapter(shared[mask])
        return adapted


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


def build_adapter(hidden_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, hidden_dim))
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


def _adapter_tensor(
    *,
    batch_size: int,
    adapter_ids: torch.Tensor | None,
    device: torch.device,
    adapter_count: int,
) -> torch.Tensor:
    if adapter_ids is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    ids = adapter_ids.to(device=device, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.repeat(batch_size)
    if ids.ndim != 1:
        raise ValueError("adapter_ids must be a 1D tensor.")
    if ids.shape[0] != batch_size:
        raise ValueError(f"adapter_ids have batch size {ids.shape[0]}, expected {batch_size}.")
    return ids.clamp(min=0, max=max(adapter_count - 1, 0))
