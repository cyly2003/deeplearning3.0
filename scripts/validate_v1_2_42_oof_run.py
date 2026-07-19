from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TARGET_NAME = "neg_log10_mol_kg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one leakage-safe v1.2.42 OOF base run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=["D", "X"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.run_dir / "manifest.json"
    predictions_path = args.run_dir / "predictions.csv"
    model_path = args.run_dir / "best_model.pt"
    for path in (manifest_path, predictions_path, model_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Required OOF artifact is missing or empty: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("prediction_output_split_parts", [])) != {"valid"}:
        raise ValueError("OOF predictions must be restricted to the untouched 'valid' fold.")
    if args.architecture == "D":
        if manifest.get("validation_source") != "internal_train_fraction":
            raise ValueError("Direct OOF early stopping must use an internal training fraction.")
    else:
        if manifest.get("finetune_mgkg_validation_source") != "internal_finetune_fraction":
            raise ValueError("Transfer OOF stage-3 early stopping must stay inside its training fold.")
        if float(manifest.get("finetune_mgkg", {}).get("trunk_learning_rate") or 0.0) != 0.0:
            raise ValueError("Transfer OOF anchor must retain the frozen-trunk S0 contract.")
    rows = []
    with predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if str(row.get("split_part", "")).strip().lower() != "valid":
                raise ValueError("OOF prediction file includes a row outside the untouched fold.")
            if str(row.get("target_name", "")) != TARGET_NAME:
                raise ValueError("OOF prediction file includes a non-mol/kg target row.")
    if not rows:
        raise ValueError("OOF prediction file is empty.")
    identities = [str(row.get("aggregate_id", "")) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("OOF prediction file contains duplicate aggregate identities.")
    print(
        json.dumps(
            {
                "status": "ok",
                "architecture": args.architecture,
                "rows": len(rows),
                "run_dir": str(args.run_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
