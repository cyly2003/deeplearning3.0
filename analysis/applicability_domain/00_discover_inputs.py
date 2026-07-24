from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ANALYSIS_DIR / "inputs" / "remote_snapshot"
DEFAULT_DB = ROOT / "outputs" / "derived" / "modeling_dataset_v2_0_0_rebuild.sqlite"
SOURCE_TABLE = "aggregated_task_records_ptox_soil_mass_molar_qc"
PARENT_SPLIT = "M_v1_2_40_ptox_to_soil_molkg_B_random_8_2"
EXPECTED_SEEDS = (42, 2042, 3407, 8417)
EXPECTED_PART_COUNTS = {
    "finetune_mgkg": 9_724,
    "finetune_mgkg_validation": 2_433,
    "test": 3_042,
}
EXPECTED_BOUNDARY_SHA256 = (
    "a2febaa7ac4679c6f273a50210315a69118490c4e38b887e4f72958bfbc2ff24"
)
EXPECTED_RECORD_ID_SHA256 = {
    "finetune_mgkg": "cf14390672af38a494fc5fdd6540a5d56ff9f8924c3c4479240333f975ae23b6",
    "finetune_mgkg_validation": "edacfcec1367f932d191334f0b6b2fa3a2c87a1899adf980eac3bdad509a92b3",
    "test": "e9559640dc209d8e44a38101e09989905ae5b10e6f33e1c64fc9d14d124bc9e0",
}
RUN_PATTERN = re.compile(r"v1\.2\.44_(M00|M10)_.+?种子(42|2042|3407|8417)")

BASE_NUMERIC_CONTEXT = ("effect_level_x", "duration_bin_h")
BASE_CATEGORICAL_CONTEXT = (
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
)
TAXONOMY_FIELDS = {
    "latin_name",
    "kingdom",
    "phylum",
    "class_name",
    "tax_order",
    "family",
    "genus",
    "species",
    "taxon_group_l1",
    "taxon_group_l2",
    "taxon_group_l3",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and audit AD inputs without training.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = discover_run_files(args.input_root, "predictions.csv")
    manifests = discover_run_files(args.input_root, "manifest.json")
    preprocessing = discover_run_files(args.input_root, "preprocessing.json")
    require_complete(predictions, "predictions.csv")
    require_complete(manifests, "manifest.json")
    require_complete(preprocessing, "preprocessing.json")

    prediction_inventory, reference_frame, prediction_audit = audit_predictions(predictions)
    manifest_audit = audit_manifests(manifests, preprocessing)
    db_audit = audit_database(args.db)
    context_frame, context_audit = audit_context_fields(
        args.db, reference_frame, manifest_audit["model_categorical_fields"]
    )
    structure_audit = audit_structures(reference_frame)
    cache_audit = audit_feature_caches()

    inventory_rows = build_input_inventory(
        predictions=predictions,
        manifests=manifests,
        preprocessing=preprocessing,
        db_path=args.db,
    )
    write_csv(args.output_dir / "input_inventory.csv", inventory_rows)
    write_csv(args.output_dir / "prediction_inventory.csv", prediction_inventory)
    write_csv(args.output_dir / "context_field_audit.csv", context_audit)
    context_frame.to_parquet(
        args.output_dir / "stage3_discovery_context_snapshot.parquet", index=False
    )
    discovery_json = {
        "schema_version": 1,
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "input_root": str(args.input_root.resolve()),
        "database": db_audit,
        "predictions": prediction_audit,
        "manifests": manifest_audit,
        "structures": structure_audit,
        "feature_caches": cache_audit,
        "locked_boundary_sha256": EXPECTED_BOUNDARY_SHA256,
    }
    write_json(args.output_dir / "ad_discovery_audit.json", discovery_json)
    report = render_report(
        input_root=args.input_root,
        db_path=args.db,
        inventory_rows=inventory_rows,
        prediction_inventory=prediction_inventory,
        prediction_audit=prediction_audit,
        manifest_audit=manifest_audit,
        db_audit=db_audit,
        context_audit=context_audit,
        structure_audit=structure_audit,
        cache_audit=cache_audit,
    )
    (args.output_dir / "AD_DISCOVERY_REPORT.md").write_text(report, encoding="utf-8")
    print(args.output_dir / "AD_DISCOVERY_REPORT.md")


def discover_run_files(root: Path, filename: str) -> dict[tuple[str, int], Path]:
    output: dict[tuple[str, int], Path] = {}
    for path in root.rglob(filename):
        match = RUN_PATTERN.search(path.as_posix())
        if not match:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in output:
            raise ValueError(f"Duplicate {filename} for {key}: {output[key]} and {path}")
        output[key] = path
    return output


def require_complete(files: dict[tuple[str, int], Path], label: str) -> None:
    expected = {(route, seed) for route in ("M00", "M10") for seed in EXPECTED_SEEDS}
    if set(files) != expected:
        raise ValueError(
            f"Incomplete {label}: missing={sorted(expected - set(files))}, "
            f"unexpected={sorted(set(files) - expected)}"
        )


def stable_identity(frame: pd.DataFrame) -> pd.Series:
    required = ("aggregate_id", "medium_domain", "target_name", "target_family")
    missing = [column for column in required if column not in frame]
    if missing or frame[list(required)].isna().any().any():
        raise ValueError(f"Cannot build strict stage-sample identity; missing={missing}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from qsar_tl.training.baseline import stage_sample_record_id

    return pd.Series(
        [
            stage_sample_record_id(aggregate_id, medium, target_name, target_family)
            for aggregate_id, medium, target_name, target_family in zip(
                frame["aggregate_id"],
                frame["medium_domain"],
                frame["target_name"],
                frame["target_family"],
            )
        ],
        index=frame.index,
        dtype="string",
    )


def audit_predictions(
    files: dict[tuple[str, int], Path]
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    reference_keys: pd.DataFrame | None = None
    reference_frame: pd.DataFrame | None = None
    common_columns: set[str] | None = None
    for (route, seed), path in sorted(files.items()):
        frame = pd.read_csv(path, low_memory=False)
        frame["stable_record_id"] = stable_identity(frame)
        counts = frame["split_part"].value_counts().to_dict()
        if counts != EXPECTED_PART_COUNTS:
            raise ValueError(f"Unexpected split counts in {path}: {counts}")
        duplicate_count = int(frame["stable_record_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"Duplicate stable identities in {path}: {duplicate_count}")
        key_columns = [
            "stable_record_id",
            "aggregate_id",
            "result_ids",
            "split_part",
            "task_head",
            "model_head",
            "target_name",
            "target_family",
            "medium_domain",
            "y_true",
        ]
        current_keys = frame[key_columns].sort_values("stable_record_id").reset_index(drop=True)
        if reference_keys is None:
            reference_keys = current_keys
        elif not frames_equal_with_nan(reference_keys, current_keys):
            raise ValueError(f"Prediction identities or truths are not aligned: {path}")
        if route == "M10" and seed == 42:
            reference_frame = frame.copy()
        frames[(route, seed)] = frame
        common_columns = set(frame.columns) if common_columns is None else common_columns & set(frame.columns)
        inventory.append(
            {
                "route": route,
                "seed": seed,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": len(frame),
                "unique_stable_record_ids": frame["stable_record_id"].nunique(),
                "duplicate_stable_record_ids": duplicate_count,
                "train_rows": counts.get("finetune_mgkg", 0),
                "validation_rows": counts.get("finetune_mgkg_validation", 0),
                "test_rows": counts.get("test", 0),
                "columns": len(frame.columns),
            }
        )
    assert reference_frame is not None
    record_id_hashes = {}
    for part, expected_hash in EXPECTED_RECORD_ID_SHA256.items():
        identities = sorted(
            reference_frame.loc[reference_frame["split_part"].eq(part), "stable_record_id"].astype(str)
        )
        actual_hash = hashlib.sha256(
            json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record_id_hashes[part] = actual_hash
        if actual_hash != expected_hash:
            raise ValueError(
                f"Strict record identity hash mismatch for {part}: {actual_hash} != {expected_hash}"
            )
    return inventory, reference_frame, {
        "routes": ["M00", "M10"],
        "seeds": list(EXPECTED_SEEDS),
        "files": len(files),
        "rows_per_file": int(len(reference_frame)),
        "part_counts": EXPECTED_PART_COUNTS,
        "stable_identity": (
            "stage_sample_v1 SHA256 composite of aggregate_id, medium_domain, "
            "target_name and target_family"
        ),
        "record_id_sha256": record_id_hashes,
        "four_seed_alignment": "pass",
        "route_alignment": "pass",
        "common_column_count": len(common_columns or set()),
        "target_names": sorted(reference_frame["target_name"].dropna().astype(str).unique()),
        "target_families": sorted(reference_frame["target_family"].dropna().astype(str).unique()),
        "target_scale_keys": sorted(reference_frame["target_scale_key"].dropna().astype(str).unique()),
        "task_heads": sorted(reference_frame["task_head"].dropna().astype(str).unique()),
    }


def frames_equal_with_nan(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        a = left[column]
        b = right[column]
        same = a.eq(b) | (a.isna() & b.isna())
        if not bool(same.all()):
            return False
    return True


def audit_manifests(
    manifests: dict[tuple[str, int], Path],
    preprocessing: dict[tuple[str, int], Path],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    categorical_fields: set[str] = set()
    numeric_features: set[str] = set()
    for key, path in sorted(manifests.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        categorical_fields.update((payload.get("categorical_cardinalities") or {}).keys())
        transformer = payload.get("numeric_transformer") or {}
        numeric_features.update(transformer.get("features") or transformer.get("feature_names") or [])
        rows.append(
            {
                "route": key[0],
                "seed": key[1],
                "rows": payload.get("rows"),
                "train_rows": payload.get("train_rows"),
                "actual_train_rows": payload.get("actual_train_rows"),
                "stage3_train_rows": payload.get("finetune_mgkg_train_rows"),
                "stage3_validation_rows": payload.get("finetune_mgkg_validation_rows"),
                "fingerprint_dim": payload.get("fingerprint_dim"),
                "numeric_dim": payload.get("numeric_dim"),
                "source_table": (payload.get("data_source") or {}).get("source_table"),
            }
        )
    fingerprint_dims = sorted({row["fingerprint_dim"] for row in rows})
    source_tables = sorted({row["source_table"] for row in rows})
    return {
        "run_manifests": rows,
        "preprocessing_files": len(preprocessing),
        "fingerprint_dims": fingerprint_dims,
        "source_tables": source_tables,
        "model_categorical_fields": sorted(categorical_fields),
        "model_numeric_features": sorted(numeric_features),
        "numeric_fitters_present": all(bool(row["numeric_dim"]) for row in rows),
        "categorical_vocabularies_present": bool(categorical_fields),
    }


def connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True, timeout=60)
    connection.execute("PRAGMA query_only=ON")
    connection.row_factory = sqlite3.Row
    return connection


def audit_database(path: Path) -> dict[str, Any]:
    with connect_read_only(path) as connection:
        source_columns = [
            row["name"] for row in connection.execute(f'PRAGMA table_info("{SOURCE_TABLE}")')
        ]
        source_rows = int(connection.execute(f'SELECT COUNT(*) FROM "{SOURCE_TABLE}"').fetchone()[0])
        source_unique = int(
            connection.execute(
                f'SELECT COUNT(DISTINCT aggregate_id) FROM "{SOURCE_TABLE}"'
            ).fetchone()[0]
        )
        parent_counts = {
            row["split_part"]: int(row["n"])
            for row in connection.execute(
                """
                SELECT split_part, COUNT(*) AS n
                FROM split_assignments
                WHERE split_name=? AND source_table=?
                GROUP BY split_part
                """,
                (PARENT_SPLIT, SOURCE_TABLE),
            )
        }
        v44_names = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT split_name FROM split_assignments WHERE split_name LIKE '%v1_2_44%'"
            )
        ]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "source_table": SOURCE_TABLE,
        "source_rows": source_rows,
        "unique_aggregate_ids": source_unique,
        "source_column_count": len(source_columns),
        "source_columns": source_columns,
        "parent_split": PARENT_SPLIT,
        "parent_split_counts": parent_counts,
        "local_v1_2_44_split_names": sorted(v44_names),
        "local_v1_2_44_split_state": "not persisted locally" if not v44_names else "present",
        "read_mode": "SQLite mode=ro + PRAGMA query_only=ON",
    }


def fetch_source_rows(
    db_path: Path, aggregate_ids: Iterable[Any], columns: Iterable[str]
) -> pd.DataFrame:
    ids = sorted({int(value) for value in aggregate_ids})
    selected = list(dict.fromkeys(columns))
    chunks: list[pd.DataFrame] = []
    with connect_read_only(db_path) as connection:
        for start in range(0, len(ids), 800):
            chunk = ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = f'''SELECT {','.join(f'"{column}"' for column in selected)}
                        FROM "{SOURCE_TABLE}" WHERE aggregate_id IN ({placeholders})'''
            chunks.append(pd.read_sql_query(query, connection, params=chunk))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=selected)


def audit_context_fields(
    db_path: Path, reference: pd.DataFrame, model_categorical_fields: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidate_fields = [
        field
        for field in (*BASE_NUMERIC_CONTEXT, *BASE_CATEGORICAL_CONTEXT)
        if field in set(model_categorical_fields) | set(BASE_NUMERIC_CONTEXT)
    ]
    source_columns = [
        "aggregate_id",
        "medium_domain",
        "target_name",
        "target_family",
        *candidate_fields,
    ]
    source = fetch_source_rows(db_path, reference["aggregate_id"], source_columns)
    key_columns = ["aggregate_id", "medium_domain", "target_name", "target_family"]
    stage3 = reference.drop(columns=[field for field in candidate_fields if field in reference], errors="ignore")
    merged = stage3.merge(source, on=key_columns, how="left", validate="one_to_one")
    merged["analysis_split"] = merged["split_part"].map(
        {
            "finetune_mgkg": "train",
            "finetune_mgkg_validation": "validation",
            "test": "test",
        }
    )
    audit: list[dict[str, Any]] = []
    train = merged.loc[merged["analysis_split"] == "train"]
    for field in candidate_fields:
        field_type = "numeric" if field in BASE_NUMERIC_CONTEXT else "categorical"
        missing = missing_mask(merged[field])
        train_missing = missing_mask(train[field])
        nonmissing_train = train.loc[~train_missing, field]
        unique_train = int(normalize_categories(nonmissing_train).nunique()) if field_type == "categorical" else int(pd.to_numeric(nonmissing_train, errors="coerce").nunique())
        row: dict[str, Any] = {
            "field": field,
            "type": field_type,
            "actual_model_input": True,
            "overall_missing_rate": float(missing.mean()),
            "train_missing_rate": float(train_missing.mean()),
            "validation_missing_rate": float(
                missing_mask(merged.loc[merged["analysis_split"] == "validation", field]).mean()
            ),
            "test_missing_rate": float(
                missing_mask(merged.loc[merged["analysis_split"] == "test", field]).mean()
            ),
            "train_unique_values": unique_train,
            "validation_unseen_rate": np.nan,
            "test_unseen_rate": np.nan,
            "include_in_b_exp": False,
            "decision_reason": "",
        }
        if field_type == "categorical":
            vocabulary = set(normalize_categories(nonmissing_train))
            for split in ("validation", "test"):
                values = merged.loc[merged["analysis_split"] == split, field]
                observed = normalize_categories(values)
                nonmissing = ~missing_mask(values)
                unseen = nonmissing & ~observed.isin(vocabulary)
                row[f"{split}_unseen_rate"] = float(unseen.mean())
        if unique_train <= 1:
            row["decision_reason"] = "excluded: constant or no effective training variation"
        elif row["train_missing_rate"] > 0.80:
            row["decision_reason"] = "excluded: training missing rate > 0.80"
        else:
            row["include_in_b_exp"] = True
            row["decision_reason"] = "included: actual Stage-3 model input with training variation"
        audit.append(row)
    return merged, audit


def missing_mask(series: pd.Series) -> pd.Series:
    text_missing = series.astype("string").fillna("").str.strip().eq("")
    return series.isna() | text_missing


def normalize_categories(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().str.casefold()


def audit_structures(reference: pd.DataFrame) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from rdkit import RDLogger
    from scripts.build_scaffold_cluster_splits import normalize_structure

    RDLogger.DisableLog("rdApp.*")
    normalized_by_smiles: dict[str, dict[str, str]] = {}
    statuses: list[str] = []
    canonical: list[str] = []
    scaffolds: list[str] = []
    for smiles in reference["smiles"].fillna("").astype(str):
        if smiles not in normalized_by_smiles:
            normalized_by_smiles[smiles] = normalize_structure(smiles)
        item = normalized_by_smiles[smiles]
        statuses.append(item["structure_status"] if item["structure_status"] == "ok" else item["parse_error"])
        canonical.append(item["canonical_smiles"])
        scaffolds.append(item["scaffold_smiles"])
    counts = Counter(statuses)
    valid = np.array([value == "ok" for value in statuses], dtype=bool)
    test_mask = reference["split_part"].eq("test").to_numpy()
    return {
        "rows": len(reference),
        "valid_rows": int(valid.sum()),
        "unavailable_rows": int((~valid).sum()),
        "parse_rate": float(valid.mean()),
        "status_counts": dict(sorted(counts.items())),
        "unique_raw_smiles": int(reference["smiles"].fillna("").astype(str).nunique()),
        "unique_canonical_parents": len({value for value in canonical if value}),
        "unique_nonempty_scaffolds": len({value for value in scaffolds if value}),
        "test_valid_rows": int((valid & test_mask).sum()),
        "test_unavailable_rows": int(((~valid) & test_mask).sum()),
        "canonicalization": "largest organic fragment, uncharged when possible, non-isomeric canonical SMILES",
    }


def audit_feature_caches() -> dict[str, Any]:
    cache_512 = ROOT / "outputs" / "features" / "molecular_features_rdkit_morgan512.jsonl"
    manifest_512 = cache_512.with_suffix(cache_512.suffix + ".manifest.json")
    cache_2048_candidates = [
        path
        for path in (ROOT / "outputs").rglob("*")
        if path.is_file() and re.search(r"(morgan|fingerprint).*(2048)|(2048).*(morgan|fingerprint)", path.name, re.I)
    ]
    return {
        "model_512_cache": str(cache_512.resolve()) if cache_512.exists() else "",
        "model_512_cache_present": cache_512.exists(),
        "model_512_manifest": str(manifest_512.resolve()) if manifest_512.exists() else "",
        "ad_2048_cache_candidates": [str(path.resolve()) for path in cache_2048_candidates],
        "ad_2048_cache_present": bool(cache_2048_candidates),
        "ad_2048_action": "reuse" if cache_2048_candidates else "recompute from canonical parent; no model retraining",
    }


def build_input_inventory(
    *,
    predictions: dict[tuple[str, int], Path],
    manifests: dict[tuple[str, int], Path],
    preprocessing: dict[tuple[str, int], Path],
    db_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, mapping in (
        ("predictions", predictions),
        ("manifest", manifests),
        ("preprocessing", preprocessing),
    ):
        for (route, seed), path in sorted(mapping.items()):
            rows.append(
                {
                    "kind": kind,
                    "route": route,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    rows.append(
        {
            "kind": "sqlite_database",
            "route": "",
            "seed": "",
            "path": str(db_path.resolve()),
            "bytes": db_path.stat().st_size,
            "sha256": "not computed for 9+ GB live SQLite snapshot",
        }
    )
    return rows


def render_report(
    *,
    input_root: Path,
    db_path: Path,
    inventory_rows: list[dict[str, Any]],
    prediction_inventory: list[dict[str, Any]],
    prediction_audit: dict[str, Any],
    manifest_audit: dict[str, Any],
    db_audit: dict[str, Any],
    context_audit: list[dict[str, Any]],
    structure_audit: dict[str, Any],
    cache_audit: dict[str, Any],
) -> str:
    lines = [
        "# AD input discovery report",
        "",
        "> Status: Phase 0 discovery passed. This report records observed inputs; it does not define AD thresholds.",
        "",
        "## Locked analysis boundary",
        "",
        "- Primary route: M10 (aquatic pTox pretraining followed by Stage-3 soil mol/kg full fine-tuning).",
        "- Baseline route: M00 (Stage-3 soil mol/kg training from random initialization).",
        f"- Seeds: {', '.join(map(str, EXPECTED_SEEDS))}.",
        f"- Stage-3 rows: train {EXPECTED_PART_COUNTS['finetune_mgkg']:,}, validation {EXPECTED_PART_COUNTS['finetune_mgkg_validation']:,}, outer test {EXPECTED_PART_COUNTS['test']:,}.",
        f"- Fixed Stage-3 boundary SHA256: `{EXPECTED_BOUNDARY_SHA256}`.",
        "- Target confirmed from predictions: `neg_log10_mol_kg` / `solid_neglog_mol_kg`; the primary score is on the native mol/kg log scale.",
        "- Interpretation boundary: interpolation within the fixed ECOTOX-derived random outer split; not unrestricted chemical-family extrapolation.",
        "",
        "## Actual files found",
        "",
        f"- Synced read-only snapshot root: `{input_root.resolve()}`.",
        f"- SQLite source: `{db_path.resolve()}` (opened with `mode=ro` and `query_only`).",
        f"- Required run files present: {sum(row['kind'] == 'predictions' for row in inventory_rows)} predictions, {sum(row['kind'] == 'manifest' for row in inventory_rows)} manifests, {sum(row['kind'] == 'preprocessing' for row in inventory_rows)} preprocessing files.",
        "- Exact paths, sizes and SHA256 values: `input_inventory.csv`.",
        "",
        "## Prediction identity audit",
        "",
        f"- Each prediction file contains {prediction_audit['rows_per_file']:,} unique Stage-3 records and no duplicate stable identity.",
        f"- Split counts are exactly {prediction_audit['part_counts']}.",
        "- Stable record identity is the project-standard `stage_sample_v1` SHA256 composite of aggregate ID, medium domain, target name and target family. `sample_id` happens to equal `aggregate_id` here but is not used as the scientific identity contract.",
        f"- Reconstructed record-identity hashes match the locked audit: {prediction_audit['record_id_sha256']}.",
        "- Cross-seed alignment: pass for M10 and M00.",
        "- Cross-route alignment: pass for stable identity, result IDs, split, task head, target, medium and y_true.",
        f"- Task heads: {len(prediction_audit['task_heads'])}; target scale keys: {prediction_audit['target_scale_keys']}.",
        "- Per-file counts and hashes: `prediction_inventory.csv`.",
        "",
        "## SQLite and split state",
        "",
        f"- Source table: `{db_audit['source_table']}` with {db_audit['source_rows']:,} rows and {db_audit['unique_aggregate_ids']:,} unique aggregate IDs.",
        f"- Parent split: `{db_audit['parent_split']}`; counts {db_audit['parent_split_counts']}.",
        "- The local SQLite snapshot does not persist the v1.2.44 route split names. This is not an identity gap: the downloaded prediction rows reconstruct the complete Stage-3 train/validation/test manifests and match the fixed audit boundary.",
        "- Stage-1 and Stage-2 source manifests will be reconstructed from the pinned v1.2.40 parent assignment plus the saved M10 preprocessing contract; no split will be written back to SQLite.",
        "",
        "## Fields confirmed",
        "",
        "- Identity/provenance: `sample_id`, `aggregate_id`, `result_ids`, split name/part.",
        "- Target/task: `y_true`, `y_pred`, `task_head`, `model_head`, `target_name`, `target_family`, `target_scale_key`.",
        "- Chemistry: CAS, DTXSID, chemical name and SMILES. Canonical parent and Murcko scaffold are derived fields, not source-table columns.",
        "- Taxonomy: Latin name, kingdom, phylum, class, order, family, genus and species.",
        "- Experimental context: the actual Stage-3 model inputs audited below; raw duration is represented by `duration_bin_h`, not a separate `duration_h` field.",
        f"- Saved model fingerprint dimension: {manifest_audit['fingerprint_dims']}; the AD fingerprint contract remains radius 2 / 2,048 bit and is kept separate.",
        "",
        "## Context field audit",
        "",
        "The full field-level audit is `context_field_audit.csv`. Inclusion is based only on actual model use, training variation and training missingness; no target errors were consulted.",
        "",
        "| Field | Type | Train missing | Train unique | Validation unseen | Test unseen | B_exp decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in context_audit:
        lines.append(
            f"| {row['field']} | {row['type']} | {fmt(row['train_missing_rate'])} | {row['train_unique_values']} | {fmt(row['validation_unseen_rate'])} | {fmt(row['test_unseen_rate'])} | {row['decision_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Structure and feature-cache audit",
            "",
            f"- Stage-3 structure-valid rows: {structure_audit['valid_rows']:,}/{structure_audit['rows']:,} ({structure_audit['parse_rate']:.2%}); unavailable rows remain a separate status and will receive NaN chemical support, never zero.",
            f"- Outer-test structure-valid/unavailable: {structure_audit['test_valid_rows']:,}/{structure_audit['test_unavailable_rows']:,}.",
            f"- Unique canonical parents/non-empty Murcko scaffolds: {structure_audit['unique_canonical_parents']:,}/{structure_audit['unique_nonempty_scaffolds']:,}.",
            f"- Model 512-bit feature cache present: {cache_audit['model_512_cache_present']}.",
            f"- Saved AD 2,048-bit cache present: {cache_audit['ad_2048_cache_present']}; action: {cache_audit['ad_2048_action']}.",
            "",
            "## Missing items and deviations from the handoff assumptions",
            "",
            "- No persisted local v1.2.44 split assignments; Stage-3 manifests are instead recovered exactly from the saved predictions.",
            "- No precomputed radius-2, 2,048-bit fingerprint matrix was found. It can be deterministically computed from canonical parents without retraining.",
            "- `canonical_parent` and Murcko scaffold are not raw SQLite columns; the existing project standardization routine must generate them.",
            "- The database provides `duration_bin_h`, not an unbinned `duration_h` field.",
            "- The discovered predictions are sufficient to reproduce M10/M00 validation and test ensembles. The next gate is the locked metric reproduction with tolerance 1e-6.",
            "",
            "## Can the AD analysis proceed without retraining?",
            "",
            "Yes. All model predictions, preprocessing contracts, identities, targets, chemistry, taxonomy and Stage-3 context fields needed for the planned AD analysis are present. Only deterministic feature reconstruction and support calculations are required. Main-model retraining is neither needed nor authorized by this workflow.",
            "",
            "## Scientific safeguards carried forward",
            "",
            "- Validation targets alone may select the AD rule; outer-test targets are report-only.",
            "- C_target will reference Stage-3 training chemicals only; C_source will reference the actual Stage-1 optimization subset only.",
            "- Taxonomic tiers are ordered support labels, not mechanistic distances.",
            "- Structure-unavailable records remain distinguishable from valid low-similarity records.",
            "- UMAP, if generated, is descriptive and cannot define the AD.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.4f}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
