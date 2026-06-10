from __future__ import annotations

import argparse
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.data.modeling_tables import build_modeling_tables, summarize_target_table
from qsar_tl.training.runner import build_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECOTOX-QSAR transfer learning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Validate an experiment config")
    validate.add_argument("--config", required=True, help="Path to YAML or JSON config")

    run = subparsers.add_parser("run", help="Run an experiment through the configured executor")
    run.add_argument("--config", required=True, help="Path to YAML or JSON config")
    run.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    build_tables = subparsers.add_parser(
        "build-modeling-tables",
        help="Build wide_records and target_records from the configured SQLite database",
    )
    build_tables.add_argument("--config", required=True, help="Path to YAML or JSON config")
    build_tables.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")

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

    if args.command == "build-modeling-tables":
        paths = config.get("paths", {})
        data = config.get("data", {})
        targets = config.get("targets", {}).get("toxicity", {})
        output_db = data.get("modeling_tables_db", "outputs/derived/modeling_dataset.sqlite")
        stats = build_modeling_tables(
            source_db=paths.get("sqlite_db", "ecotox_clean.sqlite"),
            output_db=output_db,
            limit=args.limit,
            batch_size=int(data.get("build_batch_size", 10000)),
            min_dose_groups_for_midpoint=int(targets.get("min_dose_groups_for_midpoint", 3)),
        )
        print(f"Wrote modeling tables: {Path(output_db).resolve()}")
        print(stats)
        for row in summarize_target_table(output_db):
            print(row)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
