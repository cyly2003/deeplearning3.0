from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent


def load_evaluation_module():
    path = ANALYSIS_DIR / "06_evaluate_ad_on_test.py"
    spec = importlib.util.spec_from_file_location("ad_test_evaluation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate M10-vs-M00 transfer gain strata from locked test records.")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    evaluation = load_evaluation_module()
    records = pd.read_parquet(args.output_dir / "ad_record_level.parquet")
    test = records.loc[records["analysis_split"].eq("test")].copy()
    output = evaluation.summarize_transfer_gain(
        test,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output.to_csv(args.output_dir / "transfer_gain_by_support.csv", index=False, encoding="utf-8-sig")
    print(args.output_dir / "transfer_gain_by_support.csv")


if __name__ == "__main__":
    main()
