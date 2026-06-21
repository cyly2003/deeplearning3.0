from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialConfig:
    trial_id: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    target_standardization: str
    finetune_epochs: int
    finetune_learning_rate: float
    finetune_scheduler: str
    finetune_validation_fraction: float
    augment_finetune_replicates: int
    augment_numeric_noise_std: float


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for trial_id in range(1, args.trials + 1):
        trial = sample_trial(args, rng, trial_id)
        row = run_trial(args, out_root, trial)
        rows.append(row)
        write_csv(out_root / "hpo_trials.csv", rows)
    if args.dry_run:
        print(f"dry_run_trials={len(rows)} manifest={out_root / 'hpo_trials.csv'}")
        return
    successful = [row for row in rows if row.get("status") == "ok" and row.get("objective") not in {"", None}]
    if successful:
        best = min(successful, key=lambda row: float(row["objective"]))
        (out_root / "best_trial.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"best_trial={best['trial_id']} objective={best['objective']} metrics={best['metrics_path']}")
    else:
        print("No successful trials with objective rows.", file=sys.stderr)
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dependency-free HPO for ECOTOX transfer experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--db", default=None)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--source-table", default=None)
    parser.add_argument("--out-root", default="outputs/experiments/hpo_transfer")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--run-version", default="v1.2.0")
    parser.add_argument("--run-name-zh", default="自动超参数优化_迁移验证")
    parser.add_argument("--objective-splits", default="finetune_validation,validation")
    parser.add_argument("--objective-families", default="ECx,NOEC,LOEC")
    parser.add_argument("--objective-metric", default="mae", choices=["mae", "rmse", "huber_loss"])
    parser.add_argument("--metric-min-n", type=int, default=5)
    parser.add_argument("--feature-zscore-threshold", type=float, default=None)
    parser.add_argument("--finetune-validation-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def sample_trial(args: argparse.Namespace, rng: random.Random, trial_id: int) -> TrialConfig:
    return TrialConfig(
        trial_id=trial_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=rng.choice([3e-4, 5e-4, 7e-4]),
        weight_decay=log_uniform(rng, 1e-6, 3e-4),
        dropout=rng.choice([0.10, 0.15, 0.20, 0.25]),
        target_standardization=rng.choice(["per_task_target", "per_task_adapter"]),
        finetune_epochs=rng.choice([20, 40, 60]),
        finetune_learning_rate=log_uniform(rng, 1e-5, 5e-4),
        finetune_scheduler=rng.choice(["cosine", "reduce_on_plateau", "none"]),
        finetune_validation_fraction=args.finetune_validation_fraction,
        augment_finetune_replicates=rng.choice([1, 2, 3]),
        augment_numeric_noise_std=rng.choice([0.0, 0.01, 0.03, 0.05]),
    )


def run_trial(args: argparse.Namespace, out_root: Path, trial: TrialConfig) -> dict[str, Any]:
    trial_dir = out_root / f"trial_{trial.trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(args, trial, trial_dir)
    log_path = trial_dir / "train.log"
    trial_row: dict[str, Any] = {
        **asdict(trial),
        "trial_dir": str(trial_dir),
        "log_path": str(log_path),
        "command": " ".join(command),
        "status": "dry_run" if args.dry_run else "running",
        "objective": "",
        "metrics_path": "",
    }
    if args.dry_run:
        print(trial_row["command"])
        return trial_row
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        trial_row["status"] = f"failed:{completed.returncode}"
        return trial_row
    metrics_path = newest_metrics_path(trial_dir)
    if metrics_path is None:
        trial_row["status"] = "failed:no_metrics"
        return trial_row
    try:
        objective = compute_objective(
            metrics_path,
            splits=tuple(item.strip() for item in args.objective_splits.split(",") if item.strip()),
            families=tuple(item.strip() for item in args.objective_families.split(",") if item.strip()),
            metric=args.objective_metric,
        )
    except ValueError as exc:
        trial_row["status"] = f"failed:{exc}"
        trial_row["metrics_path"] = str(metrics_path)
        return trial_row
    trial_row["status"] = "ok"
    trial_row["objective"] = objective
    trial_row["metrics_path"] = str(metrics_path)
    print(f"trial={trial.trial_id:03d} objective={objective:.6f} metrics={metrics_path}")
    return trial_row


def build_train_command(args: argparse.Namespace, trial: TrialConfig, trial_dir: Path) -> list[str]:
    command = [
        args.python,
        "-m",
        "qsar_tl.training.train",
        "--config",
        args.config,
        "--split-name",
        args.split_name,
        "--out-dir",
        str(trial_dir),
        "--run-version",
        args.run_version,
        "--run-name-zh",
        f"{args.run_name_zh}_trial{trial.trial_id:03d}",
        "--ablation",
        args.ablation,
        "--epochs",
        str(trial.epochs),
        "--batch-size",
        str(trial.batch_size),
        "--learning-rate",
        str(trial.learning_rate),
        "--weight-decay",
        str(trial.weight_decay),
        "--dropout",
        str(trial.dropout),
        "--target-standardization",
        trial.target_standardization,
        "--monitor-split",
        "internal_train_fraction",
        "--finetune-epochs",
        str(trial.finetune_epochs),
        "--finetune-learning-rate",
        str(trial.finetune_learning_rate),
        "--finetune-scheduler",
        trial.finetune_scheduler,
        "--finetune-validation-fraction",
        str(trial.finetune_validation_fraction),
        "--augment-finetune-replicates",
        str(trial.augment_finetune_replicates),
        "--augment-numeric-noise-std",
        str(trial.augment_numeric_noise_std),
        "--metric-min-n",
        str(args.metric_min_n),
    ]
    if args.feature_zscore_threshold is not None:
        command.extend(["--feature-zscore-threshold", str(args.feature_zscore_threshold)])
    if args.db:
        command.extend(["--db", args.db])
    if args.source_table:
        command.extend(["--source-table", args.source_table])
    if args.device:
        command.extend(["--device", args.device])
    return command


def compute_objective(
    metrics_path: Path,
    *,
    splits: tuple[str, ...],
    families: tuple[str, ...],
    metric: str,
) -> float:
    split_set = {item.lower() for item in splits}
    family_set = {item.lower() for item in families}
    numerator = 0.0
    denominator = 0.0
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split_part = str(row.get("split_part", "")).lower()
            task_family = str(row.get("task_head", "")).split("_", 1)[0].lower()
            if split_part not in split_set or task_family not in family_set:
                continue
            if str(row.get("metric_valid_for_summary", "1")).strip() in {"0", "false", "False"}:
                continue
            value = row.get(metric, "")
            if value in {"", None}:
                continue
            n = float(row.get("n", 0) or 0)
            numerator += float(value) * n
            denominator += n
    if denominator <= 0:
        raise ValueError(f"No objective rows in {metrics_path} for splits={splits}, families={families}.")
    return numerator / denominator


def newest_metrics_path(trial_dir: Path) -> Path | None:
    candidates = list(trial_dir.rglob("metrics_filtered.csv"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    candidates = list(trial_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
