from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine deep ablation metrics from multiple experiment roots.")
    parser.add_argument("--root", action="append", required=True, help="Experiment root containing deep/<ablation>/<split>/metrics.csv")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [collect_metrics(Path(root)) for root in args.root]
    summary = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if summary.empty:
        raise ValueError("No metrics found.")
    key = ["split_name", "ablation", "split_part", "task_head"]
    summary = summary.drop_duplicates(subset=key, keep="last").sort_values(key)
    test = summary[summary["split_part"] == "test"].copy()
    delta = compute_delta_vs_full(summary)

    summary.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out_dir / "summary_test_metrics.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(out_dir / "summary_ablation_delta_vs_full.csv", index=False, encoding="utf-8-sig")
    print(f"summary={out_dir / 'summary_metrics.csv'}")
    print(f"test={out_dir / 'summary_test_metrics.csv'}")
    print(f"delta={out_dir / 'summary_ablation_delta_vs_full.csv'}")
    print(f"rows={len(summary)}")


def collect_metrics(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(root.glob("deep/*/*/metrics.csv")):
        split_name = metrics_path.parent.name
        ablation = metrics_path.parent.parent.name
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "source_root", str(root))
        metrics.insert(0, "split_name", split_name)
        metrics.insert(0, "ablation", ablation)
        manifest = _read_json(metrics_path.with_name("manifest.json"))
        metrics["encoder_source"] = manifest.get("encoder_source", "")
        metrics["epochs"] = manifest.get("epochs", "")
        metrics["learning_rate"] = manifest.get("learning_rate", "")
        metrics["best_epoch"] = manifest.get("best_epoch", "")
        metrics["early_stopping_enabled"] = (manifest.get("early_stopping", {}) or {}).get("enabled", "")
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_delta_vs_full(summary: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["split_name", "split_part", "task_head"]
    full = summary[summary["ablation"] == "full"][
        key_columns + ["r2", "rmse", "mae", "huber_loss"]
    ].rename(
        columns={
            "r2": "full_r2",
            "rmse": "full_rmse",
            "mae": "full_mae",
            "huber_loss": "full_huber_loss",
        }
    )
    merged = summary.merge(full, on=key_columns, how="left")
    merged["delta_r2_vs_full"] = merged["r2"] - merged["full_r2"]
    merged["delta_rmse_vs_full"] = merged["rmse"] - merged["full_rmse"]
    merged["delta_mae_vs_full"] = merged["mae"] - merged["full_mae"]
    merged["delta_huber_vs_full"] = merged["huber_loss"] - merged["full_huber_loss"]
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
