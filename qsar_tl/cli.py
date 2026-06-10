from __future__ import annotations

import argparse
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.data.modeling_tables import build_modeling_tables, summarize_target_table
from qsar_tl.data.task_tables import build_task_tables, summarize_task_table
from qsar_tl.evaluation.splits import generate_and_write_split
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

    build_tasks = subparsers.add_parser(
        "build-task-tables",
        help="Build task_records and aggregated_task_records in the configured derived SQLite database",
    )
    build_tasks.add_argument("--config", required=True, help="Path to YAML or JSON config")
    build_tasks.add_argument("--db", default=None, help="Override derived SQLite database path")
    build_tasks.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")

    split = subparsers.add_parser("generate-split", help="Generate split_assignments in a derived SQLite database")
    split.add_argument("--config", required=True, help="Path to YAML or JSON config")
    split.add_argument("--db", default=None, help="Override derived SQLite database path")
    split.add_argument("--split-name", required=True, help="Name for this split assignment set")
    split.add_argument(
        "--split-type",
        required=True,
        choices=[
            "random_split",
            "chemical_group_split",
            "species_group_split",
            "chemical_species_group_split",
            "medium_transfer_split",
        ],
    )
    split.add_argument("--seed", type=int, default=None)
    split.add_argument("--limit", type=int, default=None)

    baseline = subparsers.add_parser("run-baseline", help="Run a scikit-learn baseline on a saved split")
    baseline.add_argument("--config", required=True, help="Path to YAML or JSON config")
    baseline.add_argument("--db", default=None, help="Override derived SQLite database path")
    baseline.add_argument("--split-name", required=True)
    baseline.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "rf", "hist_gradient_boosting", "hgb"],
    )
    baseline.add_argument("--out", required=True, help="Output metrics CSV or JSON path")
    baseline.add_argument("--limit", type=int, default=None)
    baseline.add_argument("--seed", type=int, default=None)

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
        source_db = _required_config_value(paths, "sqlite_db", "paths.sqlite_db")
        output_db = _required_config_value(data, "modeling_tables_db", "data.modeling_tables_db")
        stats = build_modeling_tables(
            source_db=source_db,
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

    if args.command == "build-task-tables":
        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        stats = build_task_tables(
            db,
            limit=args.limit,
            batch_size=int(config.get("data", {}).get("build_batch_size", 10000)),
        )
        print(f"Wrote task tables: {Path(db).resolve()}")
        print(stats)
        for row in summarize_task_table(db):
            print(row)
        return

    if args.command == "generate-split":
        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        summary = generate_and_write_split(
            db,
            split_name=args.split_name,
            split_type=args.split_type,
            seed=seed,
            limit=args.limit,
        )
        print(f"Wrote split assignments: {Path(db).resolve()}")
        print(summary)
        return

    if args.command == "run-baseline":
        from qsar_tl.training.baseline import run_baseline

        db = args.db or _required_config_value(config.get("data", {}), "modeling_tables_db", "data.modeling_tables_db")
        seed = args.seed if args.seed is not None else int(config.get("project", {}).get("seed", 42))
        result = run_baseline(
            db,
            split_name=args.split_name,
            model_name=args.model,
            out_path=args.out,
            limit=args.limit,
            seed=seed,
        )
        print(f"Wrote baseline report: {result.report_path.resolve()}")
        print(f"Predictions: {result.prediction_count}")
        print(f"Task heads: {', '.join(result.task_heads)}")
        print(f"Max feature count: {result.feature_count}")
        return

    parser.error(f"Unsupported command: {args.command}")


def _required_config_value(section: dict, key: str, dotted_name: str) -> str:
    value = section.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required config value: {dotted_name}")
    return str(value)


if __name__ == "__main__":
    main()
