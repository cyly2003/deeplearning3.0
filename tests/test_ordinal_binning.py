from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from qsar_tl.modeling.network import TOXICITY_BIN_LOGITS_KEY
from qsar_tl.training.deep_experiment import toxicity_bin_auxiliary_loss
from qsar_tl.training.ordinal_binning import ordinal_softmax_loss


def test_ordinal_softmax_loss_penalizes_distant_bins_more_than_adjacent_bins() -> None:
    target = torch.tensor([2], dtype=torch.long)
    adjacent_logits = torch.tensor([[0.0, 0.0, 4.0, 3.0]], dtype=torch.float32)
    distant_logits = torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    adjacent_loss = ordinal_softmax_loss(adjacent_logits, target)
    distant_loss = ordinal_softmax_loss(distant_logits, target)

    assert float(distant_loss) > float(adjacent_loss)


def test_toxicity_bin_auxiliary_loss_uses_ordinal_mode() -> None:
    outputs = {TOXICITY_BIN_LOGITS_KEY: torch.tensor([[0.0, 0.0, 4.0, 3.0]], dtype=torch.float32)}
    targets = torch.tensor([2], dtype=torch.long)

    loss, count = toxicity_bin_auxiliary_loss(outputs, targets, mode="ordinal")

    assert count == 1
    assert torch.isfinite(loss)
