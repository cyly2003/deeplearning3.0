from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("M10", "test"): (0.7154353368, 0.7551770292, 0.5457618347),
    ("M00", "test"): (0.6801113449, 0.8006776498, 0.5792099757),
}


def test_locked_test_metrics_reproduced_within_numerical_tolerance() -> None:
    metrics = pd.read_csv(ROOT / "locked_metric_reproduction.csv")
    for (route, split), expected in EXPECTED.items():
        row = metrics.loc[(metrics["route"] == route) & (metrics["split"] == split)].iloc[0]
        actual = np.array([row["r2"], row["rmse"], row["mae"]], dtype=float)
        np.testing.assert_allclose(actual, np.array(expected), rtol=0, atol=1e-9)


def test_ensemble_is_rowwise_four_seed_mean_with_sample_sd() -> None:
    records = pd.read_parquet(ROOT / "ad_record_level.parquet")
    seed_columns = [f"M10_prediction_seed_{seed}" for seed in (42, 2042, 3407, 8417)]
    values = records[seed_columns].to_numpy(float)
    np.testing.assert_allclose(records["M10_prediction"], values.mean(axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(records["M10_prediction_sd"], values.std(axis=1, ddof=1), rtol=0, atol=1e-12)
