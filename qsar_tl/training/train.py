from __future__ import annotations

import argparse
import re
from pathlib import Path

from qsar_tl.config import load_config
from qsar_tl.training.deep_experiment import run_deep_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ECOTOX-QSAR transfer model")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--db", default=None, help="Override derived SQLite database path")
    parser.add_argument("--split-name", default=None, help="Split assignment name")
    parser.add_argument("--source-table", default=None, help="Override modeling source table")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke training")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override optimizer weight decay")
    parser.add_argument("--scheduler", default=None, choices=["none", "cosine", "reduce_on_plateau"])
    parser.add_argument("--dropout", type=float, default=None, help="Override model dropout")
    parser.add_argument(
        "--target-standardization",
        default=None,
        choices=[
            "none",
            "identity",
            "global",
            "per_task",
            "per_target",
            "per_task_target",
            "per_adapter",
            "per_task_adapter",
        ],
    )
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--run-version", default=None, help="Override experiment.version for the output folder")
    parser.add_argument("--run-name-zh", default=None, help="Override experiment.name_zh for the output folder")
    parser.add_argument("--ablation", default="full", help="Deep model ablation name, default: full")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=None)
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--monitor-split", default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--finetune-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-batch-size", type=int, default=None)
    parser.add_argument("--finetune-scheduler", default=None, choices=["none", "cosine", "reduce_on_plateau"])
    parser.add_argument("--finetune-freeze", default=None, choices=["none", "heads_only", "heads_embeddings"])
    parser.add_argument("--finetune-validation-fraction", type=float, default=None)
    parser.add_argument("--augment-train-replicates", type=int, default=None)
    parser.add_argument("--augment-finetune-replicates", type=int, default=None)
    parser.add_argument("--augment-numeric-noise-std", type=float, default=None)
    parser.add_argument("--augment-target-noise-std", type=float, default=None)
    parser.add_argument("--test-noise-replicates", type=int, default=None)
    parser.add_argument("--test-noise-numeric-std", type=float, default=None)
    parser.add_argument("--feature-zscore-correction", dest="feature_zscore_correction", action="store_true", default=None)
    parser.add_argument("--no-feature-zscore-correction", dest="feature_zscore_correction", action="store_false")
    parser.add_argument("--feature-zscore-threshold", type=float, default=None)
    parser.add_argument("--metric-min-n", type=int, default=None)
    parser.add_argument(
        "--source-weighting-method",
        default=None,
        choices=["none", "tanimoto", "tanimoto_to_target", "tanimoto_to_finetune"],
    )
    parser.add_argument("--source-weighting-alpha", type=float, default=None)
    parser.add_argument("--effect-level-weighting", dest="effect_level_weighting_enabled", action="store_true", default=None)
    parser.add_argument("--no-effect-level-weighting", dest="effect_level_weighting_enabled", action="store_false")
    parser.add_argument("--effect-level-weighting-beta", type=float, default=None)
    parser.add_argument("--domain-alignment-method", default=None, choices=["none", "coral"])
    parser.add_argument("--domain-alignment-weight", type=float, default=None)
    parser.add_argument("--swa", dest="swa_enabled", action="store_true", default=None)
    parser.add_argument("--no-swa", dest="swa_enabled", action="store_false")
    parser.add_argument("--swa-start-epoch", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_dir = Path(args.out_dir or config.get("paths", {}).get("output_dir", "outputs/experiments/default"))
    split_name = args.split_name or config.get("experiment", {}).get("deep_training", {}).get("split_name", "B_random_8_2")
    db_path = args.db or config.get("data", {}).get("modeling_tables_db")
    if not db_path:
        raise ValueError("Missing data.modeling_tables_db or --db.")
    if args.run_version is not None or args.run_name_zh is not None:
        config = dict(config)
        experiment_cfg = dict(config.get("experiment", {}))
        if args.run_version is not None:
            experiment_cfg["version"] = args.run_version
        if args.run_name_zh is not None:
            experiment_cfg["name_zh"] = args.run_name_zh
        config["experiment"] = experiment_cfg
    run_dir = build_run_dir(output_dir, config=config, ablation=args.ablation, split_name=split_name)
    print(f"Loaded config: {Path(args.config).resolve()}")
    print(f"Output directory: {run_dir.resolve()}")
    result = run_deep_experiment(
        db_path,
        split_name=split_name,
        out_dir=run_dir,
        config=config,
        limit=args.limit,
        seed=int(config.get("project", {}).get("seed", 42)),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        dropout=args.dropout,
        target_standardization=args.target_standardization,
        device=args.device,
        ablation=args.ablation,
        source_table=args.source_table,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        validation_fraction=args.validation_fraction,
        monitor_split=args.monitor_split,
        finetune_epochs=args.finetune_epochs,
        finetune_learning_rate=args.finetune_learning_rate,
        finetune_batch_size=args.finetune_batch_size,
        finetune_scheduler=args.finetune_scheduler,
        finetune_freeze=args.finetune_freeze,
        finetune_validation_fraction=args.finetune_validation_fraction,
        augment_train_replicates=args.augment_train_replicates,
        augment_finetune_replicates=args.augment_finetune_replicates,
        augment_numeric_noise_std=args.augment_numeric_noise_std,
        augment_target_noise_std=args.augment_target_noise_std,
        test_noise_replicates=args.test_noise_replicates,
        test_noise_numeric_std=args.test_noise_numeric_std,
        feature_zscore_correction=args.feature_zscore_correction,
        feature_zscore_threshold=args.feature_zscore_threshold,
        metric_min_n=args.metric_min_n,
        source_weighting_method=args.source_weighting_method,
        source_weighting_alpha=args.source_weighting_alpha,
        effect_level_weighting_enabled=args.effect_level_weighting_enabled,
        effect_level_weighting_beta=args.effect_level_weighting_beta,
        domain_alignment_method=args.domain_alignment_method,
        domain_alignment_weight=args.domain_alignment_weight,
        swa_enabled=args.swa_enabled,
        swa_start_epoch=args.swa_start_epoch,
    )
    print(f"Deep training complete: {result.out_dir.resolve()}")
    print(f"Metrics: {result.metrics_path.resolve()}")
    print(f"History: {result.history_path.resolve()}")
    print(f"Encoder source: {result.encoder_source}")
    print(f"Ablation: {result.ablation}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Early stopping: {result.early_stopping_enabled}")
    print(f"Tasks: {', '.join(result.trained_tasks)}")

def build_run_dir(
    output_dir: Path,
    *,
    config: dict,
    ablation: str,
    split_name: str,
) -> Path:
    experiment_cfg = config.get("experiment", {})
    version = str(experiment_cfg.get("version", "v1.0.0")).strip() or "v1.0.0"
    run_name = str(
        experiment_cfg.get(
            "name_zh",
            experiment_cfg.get("run_name_zh", "训练优化重构_默认实验"),
        )
    ).strip()
    group_name = sanitize_path_part(f"{version}_{run_name}")
    return output_dir / group_name / "deep" / sanitize_path_part(ablation) / sanitize_path_part(split_name)


def sanitize_path_part(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip(" ._")
    return text or "unnamed"


if __name__ == "__main__":
    main()
