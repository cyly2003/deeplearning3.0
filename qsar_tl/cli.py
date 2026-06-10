from __future__ import annotations

import argparse
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.training.runner import build_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECOTOX-QSAR transfer learning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate an experiment config")
    validate.add_argument("--config", required=True, help="Path to YAML or JSON config")

    run = subparsers.add_parser("run", help="Run an experiment through the configured executor")
    run.add_argument("--config", required=True, help="Path to YAML or JSON config")
    run.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "validate-config":
        print(f"Config OK: {Path(args.config).resolve()}")
        return

    if args.command == "run":
        runner = build_runner(config)
        runner.run(config_path=Path(args.config), dry_run=args.dry_run)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

