from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scikit-learn baseline models on a saved split.")
    parser.add_argument("--db", required=True, type=Path, help="SQLite derived modeling database.")
    parser.add_argument("--split-name", required=True, help="Name in split_assignments.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "rf", "hist_gradient_boosting", "hgb"],
        help="Baseline model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    parser.add_argument("--out", required=True, type=Path, help="Output metrics CSV or JSON.")
    parser.add_argument("--source-table", default=None, help="Override modeling source table.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from qsar_tl.training.baseline import run_baseline

    result = run_baseline(
        args.db,
        split_name=args.split_name,
        model_name=args.model,
        out_path=args.out,
        limit=args.limit,
        source_table=args.source_table,
        seed=args.seed,
    )
    print(f"Wrote baseline report: {result.report_path.resolve()}")
    print(f"Predictions: {result.prediction_count}")
    print(f"Task heads: {', '.join(result.task_heads)}")
    print(f"Max feature count: {result.feature_count}")


if __name__ == "__main__":
    main()
