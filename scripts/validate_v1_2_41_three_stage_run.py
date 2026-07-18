from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed protocol audit for one v1.2.41 run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=["S1", "S2", "S3"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    required = ("manifest.json", "history.csv", "predictions.csv", "best_model.pt")
    missing = [name for name in required if not (args.run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing required run artifacts: {missing}")
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage3 = manifest.get("finetune_mgkg", {})
    swa = manifest.get("swa", {})
    expected_mse = 0.3 if args.candidate == "S3" else 0.0
    expected_swa = args.candidate in {"S2", "S3"}
    checks: dict[str, tuple[Any, Any]] = {
        "stage1_epochs": (manifest.get("epochs"), 30),
        "stage2_epochs": (manifest.get("finetune_epochs"), 20),
        "stage3_epochs": (manifest.get("finetune_mgkg_epochs"), 40),
        "stage3_epochs_ran": (manifest.get("finetune_mgkg_epochs_ran"), 40),
        "stage3_early_stopping": (stage3.get("early_stopping"), False),
        "stage3_head_lr": (stage3.get("head_learning_rate"), 0.0005),
        "stage3_trunk_lr": (stage3.get("trunk_learning_rate"), 0.0001),
        "stage3_freeze": (stage3.get("freeze"), "none"),
        "stage3_mse_weight": (stage3.get("regression_loss", {}).get("mse_weight"), expected_mse),
        "swa_enabled": (swa.get("enabled"), expected_swa),
    }
    if expected_swa:
        checks.update(
            {
                "swa_phase": (swa.get("phase"), "finetune_mgkg"),
                "swa_start_epoch": (swa.get("start_epoch"), 31),
                "swa_updates": (swa.get("updates"), 10),
                "swa_applied": (swa.get("applied"), True),
            }
        )
    failures = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if not equal(actual, expected)
    }
    if failures:
        raise ValueError(json.dumps(failures, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"status": "ok", "candidate": args.candidate, "run_dir": str(args.run_dir)}))


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-12
        except (TypeError, ValueError):
            return False
    return actual == expected


if __name__ == "__main__":
    main()
