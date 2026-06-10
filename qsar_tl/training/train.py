from __future__ import annotations

import argparse
from pathlib import Path

from qsar_tl.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ECOTOX-QSAR transfer model")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("paths", {}).get("output_dir", "outputs/experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded config: {Path(args.config).resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print("Training implementation placeholder: data pipeline and model will be filled next.")


if __name__ == "__main__":
    main()

