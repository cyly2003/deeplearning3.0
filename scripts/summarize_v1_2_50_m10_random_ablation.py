from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_v1_2_44_second_layer_matrix import (
    assert_aligned_prediction_sets,
    build_ensembles,
    metric_row,
    seed_from_text,
    stream_final_predictions,
)


SEEDS = (42, 2042, 3407, 8417)
M10_SPLIT = "M_v1_2_44_M10_仅水相预训练_固定评价边界"

CELL_SPECS = {
    "M10-Full": {
        "root": Path("outputs/experiments/第二层核心因果实验矩阵_v1_2_44"),
        "run_glob": "v1.2.44_M10_水相预训练后直接迁移_种子*",
        "ablation": "full",
    },
    "M10-Context-only": {
        "root": Path("outputs/experiments/v1_2_50_m10_random_ablation"),
        "run_glob": "v1.2.50_M10_ContextOnly_R8_2_固定评价边界_种子*",
        "ablation": "no_molecular_input",
    },
    "M10-Molecule-only": {
        "root": Path("outputs/experiments/v1_2_50_m10_random_ablation"),
        "run_glob": "v1.2.50_M10_MoleculeOnly_R8_2_固定评价边界_种子*",
        "ablation": "no_context",
    },
    "M10-Descriptor+Context": {
        "root": Path("outputs/experiments/v1_2_50_m10_random_ablation"),
        "run_glob": "v1.2.50_M10_DescriptorContext_R8_2_固定评价边界_种子*",
        "ablation": "descriptors_with_context",
    },
    "M10-Fingerprint+Context": {
        "root": Path("outputs/experiments/v1_2_50_m10_random_ablation"),
        "run_glob": "v1.2.50_M10_FingerprintContext_R8_2_固定评价边界_种子*",
        "ablation": "fingerprint_with_context",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the protocol-matched M10 input ablations on the locked "
            "v1.2.44 random 8:2 outer-test boundary."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/v1_2_50_m10_random_ablation/summary"),
    )
    return parser.parse_args()


def require_source_weighting_none(manifest: dict[str, Any], path: Path) -> None:
    weighting = manifest.get("source_weighting")
    if not isinstance(weighting, dict):
        raise ValueError(f"Manifest lacks source_weighting audit: {path}")
    if str(weighting.get("method", "")).lower() != "none":
        raise ValueError(f"Expected source weighting=none: {path}")
    if bool(weighting.get("applied", False)):
        raise ValueError(f"Source weighting must not be applied: {path}")


def discover_cell(
    repo_root: Path,
    cell: str,
    spec: dict[str, Any],
) -> tuple[dict[tuple[str, int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    cell_root = repo_root / Path(spec["root"])
    runs = sorted(cell_root.glob(str(spec["run_glob"])))
    by_seed: dict[int, Path] = {}
    manifests: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for run_dir in runs:
        seed = seed_from_text(run_dir.name)
        if seed is None or seed not in SEEDS:
            continue
        if seed in by_seed:
            raise ValueError(f"Duplicate run for {cell}, seed={seed}")
        pred_path = run_dir / "deep" / str(spec["ablation"]) / M10_SPLIT / "predictions.csv"
        manifest_path = pred_path.with_name("manifest.json")
        if not pred_path.is_file() or pred_path.stat().st_size <= 0:
            raise ValueError(f"Missing predictions for {cell}, seed={seed}: {pred_path}")
        if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
            raise ValueError(f"Missing manifest for {cell}, seed={seed}: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_seed = int(manifest.get("seed", -1))
        if manifest_seed != seed:
            raise ValueError(
                f"Manifest/path seed mismatch for {cell}: path={seed}, manifest={manifest_seed}"
            )
        if str(manifest.get("ablation", "")) != str(spec["ablation"]):
            raise ValueError(f"Ablation mismatch for {cell}, seed={seed}")
        if str(manifest.get("split_name", "")) != M10_SPLIT:
            raise ValueError(f"Split mismatch for {cell}, seed={seed}")
        require_source_weighting_none(manifest, manifest_path)
        rows = stream_final_predictions(pred_path, scale="molar")
        test_rows = [row for row in rows if row["evaluation_part"] == "test"]
        if not test_rows:
            raise ValueError(f"No final soil test rows for {cell}, seed={seed}")
        predictions[(cell, seed, "test")] = test_rows
        by_seed[seed] = run_dir
        manifests.append(
            {
                "cell": cell,
                "seed": seed,
                "run_dir": str(run_dir.relative_to(repo_root)),
                "predictions": str(pred_path.relative_to(repo_root)),
                "split_name": M10_SPLIT,
                "ablation": str(spec["ablation"]),
                "source_weighting": "none",
                "n_test": len(test_rows),
            }
        )
    missing = sorted(set(SEEDS) - set(by_seed))
    if missing:
        raise ValueError(f"Incomplete four-seed matrix for {cell}; missing={missing}")
    return predictions, manifests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    prediction_sets: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    manifest_audit: list[dict[str, Any]] = []
    for cell, spec in CELL_SPECS.items():
        cell_predictions, cell_audit = discover_cell(repo_root, cell, spec)
        prediction_sets.update(cell_predictions)
        manifest_audit.extend(cell_audit)

    cells = tuple(CELL_SPECS)
    assert_aligned_prediction_sets(
        prediction_sets,
        cells=cells,
        seeds=SEEDS,
        parts=("test",),
    )

    seed_metrics: list[dict[str, Any]] = []
    for cell in cells:
        for seed in SEEDS:
            seed_metrics.append(
                metric_row(
                    cell,
                    "test",
                    prediction_sets[(cell, seed, "test")],
                    aggregation="single_seed",
                    seed=seed,
                )
            )

    ensembles = build_ensembles(prediction_sets, cells=cells, seeds=SEEDS)
    ensemble_metrics = [
        metric_row(
            cell,
            "test",
            ensembles[cell],
            aggregation="prediction_ensemble",
            seeds=SEEDS,
        )
        for cell in cells
    ]
    full = next(row for row in ensemble_metrics if row["cell"] == "M10-Full")
    for row in ensemble_metrics:
        row["delta_r2_vs_full"] = float(row["common_molkg_r2"]) - float(
            full["common_molkg_r2"]
        )
        row["delta_rmse_vs_full"] = float(row["common_molkg_rmse"]) - float(
            full["common_molkg_rmse"]
        )
        row["delta_mae_vs_full"] = float(row["common_molkg_mae"]) - float(
            full["common_molkg_mae"]
        )

    mean_sd: list[dict[str, Any]] = []
    metric_fields = (
        "common_molkg_r2",
        "common_molkg_rmse",
        "common_molkg_mae",
        "within_r2_molkg",
    )
    for cell in cells:
        group = [row for row in seed_metrics if row["cell"] == cell]
        item: dict[str, Any] = {
            "cell": cell,
            "evaluation_part": "test",
            "seed_count": len(group),
            "seeds": ",".join(str(seed) for seed in SEEDS),
            "n": group[0]["n"],
        }
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_sd"] = statistics.stdev(values)
        mean_sd.append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = output_dir / "m10_random8_2_four_seed_prediction_ensemble.csv"
    seed_path = output_dir / "m10_random8_2_single_seed_metrics.csv"
    mean_sd_path = output_dir / "m10_random8_2_seed_mean_sd.csv"
    audit_path = output_dir / "m10_random8_2_manifest_audit.csv"
    write_csv(ensemble_path, ensemble_metrics)
    write_csv(seed_path, seed_metrics)
    write_csv(mean_sd_path, mean_sd)
    write_csv(audit_path, manifest_audit)

    identity_hashes = sorted(
        {
            (
                str(row["aggregate_id_sha256"]),
                str(row["result_ids_sha256"]),
                int(row["n"]),
            )
            for row in ensemble_metrics
        }
    )
    if len(identity_hashes) != 1:
        raise ValueError(f"Ensemble identity hashes are not identical: {identity_hashes}")
    summary = {
        "schema": "v1_2_50_m10_random_ablation_summary_v1",
        "status": "complete",
        "evaluation_boundary": "locked v1.2.44 row-random 8:2 outer test",
        "is_scaffold_split": False,
        "split_name": M10_SPLIT,
        "target": "neg_log10_mol_kg; final soil; 18 task heads",
        "source_weighting": "none",
        "seeds": list(SEEDS),
        "aggregation": "rowwise mean prediction across four seeds, then recompute metrics",
        "identity_hash": {
            "aggregate_id_sha256": identity_hashes[0][0],
            "result_ids_sha256": identity_hashes[0][1],
            "n": identity_hashes[0][2],
        },
        "outputs": {
            "ensemble_metrics": str(ensemble_path.relative_to(repo_root)),
            "single_seed_metrics": str(seed_path.relative_to(repo_root)),
            "seed_mean_sd": str(mean_sd_path.relative_to(repo_root)),
            "manifest_audit": str(audit_path.relative_to(repo_root)),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "metrics": ensemble_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
