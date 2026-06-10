from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from qsar_tl.modeling.dataset import AggregatedTaskDataset
from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork
from qsar_tl.training.deep_train import (
    DeepTrainingConfig,
    collate_aggregated_task_batch,
    train_model,
)


def _synthetic_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "s1",
            "molecular_numeric": [0.1, 1.2, -0.3],
            "fingerprint": [1, 0, 1, 0],
            "categorical_ids": {"species_id": 1, "primary_medium_id": 1},
            "task_head": "ECx_Mortality",
            "target_value": 2.0,
        },
        {
            "sample_id": "s2",
            "molecular_numeric": [0.2, 1.0, -0.1],
            "fingerprint": [0, 1, 1, 0],
            "categorical_ids": {"species_id": 2, "primary_medium_id": 1},
            "task_head": "NOEC_Growth",
            "target_value": 1.4,
        },
        {
            "sample_id": "s3",
            "molecular_numeric": [0.4, 0.8, 0.2],
            "fingerprint": [1, 1, 0, 0],
            "categorical_ids": {"species_id": 1, "primary_medium_id": 2},
            "task_head": "ECx_Mortality",
            "target_value": 2.3,
        },
        {
            "sample_id": "s4",
            "molecular_numeric": [0.5, 0.6, 0.4],
            "fingerprint": [0, 0, 1, 1],
            "categorical_ids": {"species_id": 3, "primary_medium_id": 2},
            "task_head": "NOEC_Growth",
            "target_value": 1.1,
        },
    ]


def _model(dataset: AggregatedTaskDataset) -> EcotoxMultiTaskNetwork:
    return EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=dataset.numeric_dim(),
            fingerprint_dim=dataset.fingerprint_dim(),
            categorical_cardinalities={"primary_medium_id": 3, "species_id": 4},
            task_heads=("ECx_Mortality", "NOEC_Growth"),
            hidden_dims=(16, 8),
            dropout=0.0,
        )
    )


def test_aggregated_task_dataset_normalizes_synthetic_rows() -> None:
    dataset = AggregatedTaskDataset(_synthetic_samples())

    assert len(dataset) == 4
    assert dataset.numeric_dim() == 3
    assert dataset.fingerprint_dim() == 4
    assert dataset.task_heads() == ("ECx_Mortality", "NOEC_Growth")
    assert dataset[0]["categorical_ids"] == {"species_id": 1, "primary_medium_id": 1}


def test_multitask_network_forward_shapes() -> None:
    dataset = AggregatedTaskDataset(_synthetic_samples())
    batch = collate_aggregated_task_batch([dataset[index] for index in range(3)])
    model = _model(dataset)

    outputs = model(
        molecular_numeric=batch["molecular_numeric"],
        fingerprint=batch["fingerprint"],
        categorical_ids=batch["categorical_ids"],
    )

    assert set(outputs) == {"ECx_Mortality", "NOEC_Growth"}
    assert outputs["ECx_Mortality"].shape == torch.Size([3])
    assert outputs["NOEC_Growth"].shape == torch.Size([3])


def test_deep_train_cpu_smoke() -> None:
    dataset = AggregatedTaskDataset(_synthetic_samples())
    model = _model(dataset)

    history = train_model(
        model,
        dataset,
        DeepTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-2,
            task_weights={"ECx_Mortality": 1.0, "NOEC_Growth": 0.8},
            device="cpu",
            seed=7,
        ),
    )

    assert len(history.epochs) == 2
    assert history.epochs[-1].samples == 4
    assert math.isfinite(history.epochs[-1].mean_loss)
    assert set(history.epochs[-1].task_loss) == {"ECx_Mortality", "NOEC_Growth"}
