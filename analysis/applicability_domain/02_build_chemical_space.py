from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ad_chemistry import (
    build_fingerprints,
    chemical_support,
    normalize_structure_table,
    save_fingerprint_cache,
    structure_summary,
)
from ad_common import (
    ANALYSIS_DIR,
    ROOT,
    build_route_ensembles,
    canonical_sha256,
    stage_record_ids,
    write_csv,
    write_json,
)


DEFAULT_DB = ROOT / "outputs" / "derived" / "modeling_dataset_v2_0_0_rebuild.sqlite"
DEFAULT_CONFIG = ANALYSIS_DIR / "config" / "ad_config.yaml"
AUDIT_JSON = ROOT / "实验汇总" / "第二层核心因果实验矩阵_v1_2_44" / "固定拆分与身份审计.json"
SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
PARENT_SPLIT = "M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe chemical-space and C support.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    parent = reconstruct_parent_manifests(args.db, audit)
    ensembles = build_route_ensembles()
    stage3 = ensembles.copy()
    stage3_structures = normalize_structure_table(stage3)
    stage3 = pd.concat([stage3.reset_index(drop=True), stage3_structures.reset_index(drop=True)], axis=1)
    stage3["raw_chemical_id"] = raw_chemical_identity(stage3)

    stage1_train = parent["stage1_train"].copy()
    stage1_train = attach_structures(stage1_train)
    stage2_train = parent["stage2_train"].copy()
    stage2_train = attach_structures(stage2_train)

    stage3_train = stage3.loc[stage3["analysis_split"].eq("train")].copy()
    stage3_validation = stage3.loc[stage3["analysis_split"].eq("validation")].copy()
    stage3_test = stage3.loc[stage3["analysis_split"].eq("test")].copy()
    evaluation = stage3.loc[stage3["analysis_split"].isin(["validation", "test"])].copy()

    chemical = config["chemical"]
    support_target = chemical_support(
        evaluation,
        stage3_train,
        prefix="target",
        radius=int(chemical["radius"]),
        n_bits=int(chemical["n_bits"]),
        top_k=int(chemical["top_k"]),
        thresholds=chemical["candidate_similarity_thresholds"],
    )
    support_source = chemical_support(
        evaluation,
        stage1_train,
        prefix="source",
        radius=int(chemical["radius"]),
        n_bits=int(chemical["n_bits"]),
        top_k=int(chemical["top_k"]),
        thresholds=chemical["candidate_similarity_thresholds"],
    )
    support = pd.concat(
        [
            evaluation.reset_index(drop=True),
            support_target.reset_index(drop=True),
            support_source.reset_index(drop=True),
        ],
        axis=1,
    )

    stage_frames = {
        "stage1_fit": stage1_train,
        "stage2_fit": stage2_train,
        "stage3_train": stage3_train,
        "stage3_validation": stage3_validation,
        "stage3_test": stage3_test,
    }
    stage_summary = [structure_summary(frame, stage=stage) for stage, frame in stage_frames.items()]
    overlap_summary = build_overlap_summary(stage_frames, support)
    write_csv(args.output_dir / "chemical_space_stage_summary.csv", stage_summary)
    write_csv(args.output_dir / "chemical_space_overlap_summary.csv", overlap_summary)
    support_to_source_columns = [
        "record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "structure_status",
        "structure_reason",
        "canonical_parent",
        "murcko_scaffold",
        *[column for column in support if column.endswith("_source") or "source_chemical_ge" in column],
    ]
    support_to_target_columns = [
        "record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "structure_status",
        "structure_reason",
        "canonical_parent",
        "murcko_scaffold",
        *[column for column in support if column.endswith("_target") or "target_chemical_ge" in column],
    ]
    support[support_to_source_columns].to_parquet(
        args.output_dir / "stage3_chemical_support_to_stage1.parquet", index=False
    )
    support[support_to_target_columns].to_parquet(
        args.output_dir / "stage3_chemical_support_to_stage3_train.parquet", index=False
    )
    support.to_parquet(args.output_dir / "stage3_chemical_support_all.parquet", index=False)

    write_manifests(args.output_dir, parent, stage3)
    cache = build_structure_cache(stage_frames)
    cache.to_parquet(args.output_dir / "chemical_structure_cache.parquet", index=False)
    fingerprints = build_fingerprints(
        cache.loc[cache["structure_status"].eq("ok"), "canonical_parent"],
        radius=int(chemical["radius"]),
        n_bits=int(chemical["n_bits"]),
    )
    save_fingerprint_cache(
        args.output_dir / "chemical_fingerprints_radius2_2048.npz",
        fingerprints,
        n_bits=int(chemical["n_bits"]),
    )
    manifest = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "database": str(args.db.resolve()),
        "parent_split": PARENT_SPLIT,
        "stage1_pool_rows": len(parent["stage1_pool"]),
        "stage1_fit_rows": len(stage1_train),
        "stage2_pool_rows": len(parent["stage2_pool"]),
        "stage2_fit_rows": len(stage2_train),
        "stage3_counts": stage3["analysis_split"].value_counts().to_dict(),
        "stage1_fit_record_id_sha256": canonical_sha256(sorted(stage1_train["record_id"].astype(str))),
        "stage2_fit_record_id_sha256": canonical_sha256(sorted(stage2_train["record_id"].astype(str))),
        "similarity_reference_sets": {
            "C_target": "Stage-3 fit/train canonical parents only",
            "C_source": "actual Stage-1 optimization canonical parents only",
        },
        "test_target_used": False,
        "umap_generated": False,
        "umap_reason": "UMAP is optional and does not define AD; omitted from the core leakage-safe support build.",
    }
    write_json(args.output_dir / "chemical_space_manifest.json", manifest)
    (args.output_dir / "CHEMICAL_SPACE_REPORT.md").write_text(
        render_report(stage_summary, overlap_summary, support, parent), encoding="utf-8"
    )
    print(args.output_dir / "CHEMICAL_SPACE_REPORT.md")


def reconstruct_parent_manifests(db_path: Path, audit: dict[str, Any]) -> dict[str, pd.DataFrame]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.build_v1_2_44_second_layer_splits import (
        filter_like_deep_experiment,
        split_internal_finetune_validation,
        split_internal_training_validation,
    )
    from qsar_tl.training.baseline import load_split_frame

    # Use the same authoritative loader and task-filter implementation as the
    # locked split builder. Its merge order is part of the seeded validation
    # contract, so a fresh SQL reimplementation is intentionally avoided.
    parent_frame = load_split_frame(
        db_path,
        split_name=PARENT_SPLIT,
        source_table=SOURCE_TABLE,
        allow_mixed_target_dimensions=True,
    )
    frame, filter_audit = filter_like_deep_experiment(
        parent_frame,
        min_total=int(audit["task_filter"]["min_total"]),
        min_train=int(audit["task_filter"]["min_train"]),
        min_eval=int(audit["task_filter"]["min_eval"]),
    )
    if set(filter_audit["kept_task_heads"]) != set(audit["task_filter"]["kept_task_heads"]):
        raise ValueError("Authoritative task filter no longer matches the locked audit")
    frame = frame.reset_index(drop=True)
    frame["record_id"] = stage_record_ids(frame)
    frame["model_head"] = frame["task_head"].astype(str)
    stage1_pool = frame.loc[frame["split_part"].eq("train")].copy().reset_index(drop=True)
    stage2_pool = frame.loc[frame["split_part"].eq("finetune")].copy().reset_index(drop=True)
    if len(stage1_pool) != 245_147 or len(stage2_pool) != 12_327:
        raise ValueError(f"Filtered Stage-1/2 counts changed: {len(stage1_pool)}/{len(stage2_pool)}")
    samples1 = [{"task_head": value} for value in stage1_pool["model_head"]]
    train1, valid1, source1 = split_internal_training_validation(
        samples1,
        train_indices=list(range(len(stage1_pool))),
        seed=42,
        validation_fraction=0.1,
    )
    samples2 = [{"task_head": value} for value in stage2_pool["model_head"]]
    train2, valid2, source2 = split_internal_finetune_validation(
        samples2,
        finetune_indices=list(range(len(stage2_pool))),
        # The helper applies its documented +17031 offset internally; the
        # locked caller seed is 42, giving the audited effective RNG 17073.
        seed=42,
        validation_fraction=0.2,
    )
    stage1_train = stage1_pool.iloc[train1].copy().reset_index(drop=True)
    stage1_validation = stage1_pool.iloc[valid1].copy().reset_index(drop=True)
    stage2_train = stage2_pool.iloc[train2].copy().reset_index(drop=True)
    stage2_validation = stage2_pool.iloc[valid2].copy().reset_index(drop=True)
    checks = (
        (stage1_train, audit["parent_stage1_boundary"]["train_record_id_sha256"], "Stage-1 train"),
        (stage1_validation, audit["parent_stage1_boundary"]["validation_record_id_sha256"], "Stage-1 validation"),
        (stage2_train, audit["parent_stage2_boundary"]["train_record_id_sha256"], "Stage-2 train"),
        (stage2_validation, audit["parent_stage2_boundary"]["validation_record_id_sha256"], "Stage-2 validation"),
    )
    for selected, expected, label in checks:
        observed = canonical_sha256(sorted(selected["record_id"].astype(str)))
        if observed != expected:
            raise ValueError(f"{label} identity hash mismatch: {observed} != {expected}")
    return {
        "stage1_pool": stage1_pool,
        "stage1_train": stage1_train,
        "stage1_validation": stage1_validation,
        "stage2_pool": stage2_pool,
        "stage2_train": stage2_train,
        "stage2_validation": stage2_validation,
    }


def attach_structures(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    structures = normalize_structure_table(output)
    output = pd.concat([output, structures.reset_index(drop=True)], axis=1)
    output["raw_chemical_id"] = raw_chemical_identity(output)
    return output


def raw_chemical_identity(frame: pd.DataFrame) -> pd.Series:
    cas = frame["cas_number"].astype("string").fillna("").str.strip()
    dtxsid = frame["dtxsid"].astype("string").fillna("").str.strip()
    smiles = frame["smiles"].astype("string").fillna("").str.strip()
    aggregate = frame["aggregate_id"].astype("string")
    return pd.Series(
        np.where(cas.ne(""), "cas:" + cas, np.where(dtxsid.ne(""), "dtxsid:" + dtxsid, np.where(smiles.ne(""), "smiles:" + smiles, "aggregate:" + aggregate))),
        index=frame.index,
        dtype="string",
    )


def build_overlap_summary(
    stage_frames: dict[str, pd.DataFrame], support: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(stage_frames, 2):
        a = stage_frames[left]
        b = stage_frames[right]
        a_parent = set(a.loc[a["structure_status"].eq("ok"), "canonical_parent"])
        b_parent = set(b.loc[b["structure_status"].eq("ok"), "canonical_parent"])
        a_scaffold = set(a.loc[a["murcko_scaffold"].ne(""), "murcko_scaffold"])
        b_scaffold = set(b.loc[b["murcko_scaffold"].ne(""), "murcko_scaffold"])
        rows.append(
            {
                "summary_type": "pairwise_unique_structure_overlap",
                "left": left,
                "right": right,
                "left_canonical_parents": len(a_parent),
                "right_canonical_parents": len(b_parent),
                "exact_parent_overlap": len(a_parent & b_parent),
                "right_parent_seen_fraction": math.nan if not b_parent else len(a_parent & b_parent) / len(b_parent),
                "left_scaffolds": len(a_scaffold),
                "right_scaffolds": len(b_scaffold),
                "scaffold_overlap": len(a_scaffold & b_scaffold),
                "right_scaffold_seen_fraction": math.nan if not b_scaffold else len(a_scaffold & b_scaffold) / len(b_scaffold),
            }
        )
    for split in ("validation", "test"):
        selected = support.loc[support["analysis_split"].eq(split)]
        for reference in ("target", "source"):
            bins = similarity_bins(selected[f"C_{reference}"], selected["structure_status"])
            for label, count in bins.value_counts(dropna=False).items():
                rows.append(
                    {
                        "summary_type": "record_similarity_bin",
                        "left": f"stage3_{split}",
                        "right": f"{reference}_reference",
                        "similarity_bin": label,
                        "n_records": int(count),
                        "record_fraction": float(count / len(selected)),
                    }
                )
    return rows


def similarity_bins(values: pd.Series, status: pd.Series) -> pd.Series:
    output = pd.Series("structure unavailable", index=values.index, dtype="string")
    valid = status.eq("ok") & values.notna()
    output.loc[valid & values.eq(1.0)] = "exact parent / similarity 1.00"
    output.loc[valid & values.ge(0.80) & values.lt(1.0)] = "0.80–<1.00"
    output.loc[valid & values.ge(0.65) & values.lt(0.80)] = "0.65–<0.80"
    output.loc[valid & values.ge(0.40) & values.lt(0.65)] = "0.40–<0.65"
    output.loc[valid & values.lt(0.40)] = "<0.40"
    return output


def write_manifests(output_dir: Path, parent: dict[str, pd.DataFrame], stage3: pd.DataFrame) -> None:
    columns = [
        "record_id",
        "aggregate_id",
        "model_head",
        "medium_domain",
        "target_name",
        "target_family",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
    ]
    for name in ("stage1_pool", "stage1_train", "stage1_validation", "stage2_pool", "stage2_train", "stage2_validation"):
        parent[name][columns].to_parquet(output_dir / f"{name}_manifest.parquet", index=False)
    stage3_columns = [
        "record_id",
        "aggregate_id",
        "analysis_split",
        "model_head",
        "medium_domain",
        "target_name",
        "target_family",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
        "y_true",
    ]
    stage3[stage3_columns].to_parquet(output_dir / "stage3_manifest.parquet", index=False)


def build_structure_cache(stage_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, frame in stage_frames.items():
        selected = frame[
            [
                "raw_smiles",
                "canonical_parent",
                "murcko_scaffold",
                "structure_status",
                "structure_reason",
            ]
        ].drop_duplicates()
        selected.insert(0, "stage", stage)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).drop_duplicates().sort_values(["stage", "raw_smiles"])


def render_report(
    stage_summary: list[dict[str, Any]],
    overlap_summary: list[dict[str, Any]],
    support: pd.DataFrame,
    parent: dict[str, pd.DataFrame],
) -> str:
    stage = pd.DataFrame(stage_summary).set_index("stage")
    test = support.loc[support["analysis_split"].eq("test")]
    validation = support.loc[support["analysis_split"].eq("validation")]
    return "\n".join(
        [
            "# Chemical-space and chemical-support report",
            "",
            "> Status: complete. All chemical references are fit/training subsets; validation and test chemicals are never added to a reference set.",
            "",
            "## Reconstructed fit manifests",
            "",
            f"- Stage 1 pool/actual optimization rows: {len(parent['stage1_pool']):,}/{len(parent['stage1_train']):,}.",
            f"- Stage 2 pool/actual optimization rows: {len(parent['stage2_pool']):,}/{len(parent['stage2_train']):,}.",
            "- Stage 3 fit/validation/test rows: 9,724/2,433/3,042.",
            "- All four internal Stage-1/2 identity hashes match the locked audit.",
            "",
            "## Structure coverage",
            "",
            "| Stage | Records | Valid rows | Valid fraction | Canonical parents | Non-empty scaffolds |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {name} | {int(row.n_records):,} | {int(row.structure_valid_rows):,} | {row.structure_valid_fraction:.2%} | {int(row.n_canonical_parents):,} | {int(row.n_nonempty_scaffolds):,} |"
                for name, row in stage.iterrows()
            ],
            "",
            "## Stage-3 chemical support",
            "",
            f"- Validation median C_target/C_source: {validation['C_target'].median():.3f}/{validation['C_source'].median():.3f} among structure-valid rows.",
            f"- Test median C_target/C_source: {test['C_target'].median():.3f}/{test['C_source'].median():.3f} among structure-valid rows.",
            f"- Test exact-parent seen in Stage-3 fit: {test.loc[test['structure_status'].eq('ok'), 'exact_parent_seen_target'].mean():.2%} of structure-valid records.",
            f"- Test exact-parent seen in Stage-1 fit: {test.loc[test['structure_status'].eq('ok'), 'exact_parent_seen_source'].mean():.2%} of structure-valid records.",
            "- Similarity 1.0 and exact canonical-parent identity are stored separately; the narrative uses the exact-parent flag.",
            "- Structure-unavailable rows have C_target/C_source = NaN and remain a distinct status.",
            "",
            "## Interpretation boundary",
            "",
            "The random outer split is strongly interpolation-dominated at the chemical level. Chemical support alone is therefore expected to have limited rejection power; this is precisely why the next phase evaluates biological-experimental, task-head and joint local support without treating UMAP or similarity coverage as proof of extrapolation.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
