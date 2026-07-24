from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.training.traditional_comparison import (
    BOUNDARY_SHA256,
    CONTRACT_DIRS,
    MODEL_DIRS,
    MODEL_ORDER,
    SEEDS,
    prepare_comparison_data,
    run_model_contract,
)


DEFAULT_ROOT = Path("三阶段版") / "与传统机器学习算法对比"
DEFAULT_SNAPSHOT = Path("analysis/applicability_domain/stage3_discovery_context_snapshot.parquet")
DEFAULT_PREPROCESSING = Path(
    "analysis/applicability_domain/inputs/remote_snapshot/"
    "v1.2.44_M00_从头训练_固定评价边界_种子42/deep/full/"
    "M_v1_2_44_M00_仅Stage3从头训练_固定评价边界/preprocessing.json"
)
DEFAULT_CACHE = Path("outputs/features/molecular_features_rdkit_morgan512.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RF/XGBoost/LightGBM/PLS/KNN on the immutable v1.2.44 Stage-3 "
            "boundary with molecule-only and matched molecule+context input contracts."
        )
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--molecular-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--contracts",
        nargs="+",
        choices=sorted(CONTRACT_DIRS),
        default=list(CONTRACT_DIRS),
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER), default=list(MODEL_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--min-train", type=int, default=35)
    parser.add_argument("--min-validation", type=int, default=10)
    parser.add_argument("--min-test", type=int, default=10)
    parser.add_argument(
        "--max-subtasks",
        type=int,
        default=None,
        help="Optional debug cap. Formal runs should leave this unset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    for name in ("min_train", "min_validation", "min_test"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_subtasks is not None and args.max_subtasks < 1:
        raise ValueError("--max-subtasks must be positive when provided.")
    run_root = (
        args.out_root / "冒烟测试_物种终点独立模型"
        if args.smoke
        else args.out_root
    )
    run_root.mkdir(parents=True, exist_ok=True)
    shared = run_root / "00_共同协议与审计"
    shared.mkdir(parents=True, exist_ok=True)
    data = prepare_comparison_data(
        args.snapshot,
        args.preprocessing,
        args.molecular_cache,
    )
    audit_path = shared / "固定边界与特征合同审计.json"
    audit_path.write_text(
        json.dumps(data.audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (shared / "分子信息特征名.txt").write_text(
        "\n".join(data.molecule_feature_names) + "\n", encoding="utf-8"
    )
    (shared / "输入一致特征名.txt").write_text(
        "\n".join(data.full_feature_names) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "boundary_sha256": BOUNDARY_SHA256,
                    "audit": str(audit_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    n_trials = min(args.n_trials, 2) if args.smoke else args.n_trials
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    max_subtasks = min(args.max_subtasks or 2, 2) if args.smoke else args.max_subtasks
    completed = []
    for contract in args.contracts:
        for model in args.models:
            output_dir = run_root / CONTRACT_DIRS[contract] / MODEL_DIRS[model]
            result = run_model_contract(
                data,
                contract=contract,
                model_name=model,
                output_dir=output_dir,
                n_trials=n_trials,
                seeds=seeds,
                n_jobs=args.n_jobs,
                min_train=args.min_train,
                min_validation=args.min_validation,
                min_test=args.min_test,
                max_subtasks=max_subtasks,
                overwrite=args.overwrite,
            )
            completed.append(result)
            print(
                json.dumps(
                    {
                        "event": "model_complete",
                        "contract": contract,
                        "model": model,
                        "output_dir": str(output_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(json.dumps({"status": "complete", "cells": completed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
