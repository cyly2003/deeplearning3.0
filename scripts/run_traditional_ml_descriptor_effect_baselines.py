from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qsar_tl.evaluation.metrics import regression_metrics


RUN_VERSION = "v1.2.28"
DEFAULT_SEED = 42
TARGET_COLUMN = "target_value_median"
SOURCE_TABLES = {
    "aquatic": "aggregated_task_records_aquatic_ptox_qc",
    "soil": "aggregated_task_records_soil_ptox_qc",
}
DOMAIN_LABELS = {"aquatic": "水生", "soil": "土壤"}
DEFAULT_MODELS = ("xgboost", "lightgbm", "random_forest", "knn", "pls")
DEFAULT_SCOPES = ("species_endpoint",)
SUPPORTED_SCOPES = ("domain", "endpoint", "species", "species_endpoint")
MAX_ABS_DESCRIPTOR_VALUE = 1e12
MAX_ABS_PREDICTION_VALUE = 1e6
EFFECT_FEATURE_COLUMNS = [
    "effect_level_x",
    "effect_level_x_fraction",
    "effect_level_x_log1p",
    "effect_level_x_present",
]
QSAR_DESCRIPTOR_RAW_NAMES = {
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumValenceElectrons",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "FractionCSP3",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
    "NumAromaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles",
    "BalabanJ",
    "BertzCT",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxAbsEStateIndex",
    "MinAbsEStateIndex",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
}
QSAR_DESCRIPTOR_PREFIXES = (
    "BCUT2D_",
    "Chi",
    "EState_VSA",
    "VSA_EState",
    "PEOE_VSA",
    "SMR_VSA",
    "SlogP_VSA",
)
MOLECULAR_SIZE_RELATED_RAW_NAMES = {
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumValenceElectrons",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
    "NumAromaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedCarbocycles",
    "FractionCSP3",
    "Kappa1",
    "Kappa2",
    "Kappa3",
}
MOLECULAR_SIZE_RELATED_PREFIXES = (
    "BCUT2D_",
    "Chi",
    "EState_VSA",
    "VSA_EState",
    "PEOE_VSA",
    "SMR_VSA",
    "SlogP_VSA",
)


@dataclass(frozen=True)
class SubtaskSpec:
    domain: str
    scope: str
    name: str
    task_head: str | None = None
    latin_name: str | None = None

    @property
    def slug(self) -> str:
        return safe_slug(f"{self.domain}_{self.scope}_{self.name}")


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    features: pd.DataFrame
    feature_columns: list[str]
    descriptor_cache_path: Path
    invalid_smiles_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local traditional-ML QSAR baselines using only RDKit molecular "
            "descriptors and effect-level features."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/experiments/v1_2_28_species_endpoint_ml_descriptor_effect_baselines"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("实验汇总") / "机器学习基线_分子描述符效应水平",
    )
    parser.add_argument("--descriptor-cache", type=Path, default=Path("outputs/derived/rdkit_2d_descriptor_cache_v1.csv.gz"))
    parser.add_argument(
        "--descriptor-sensitivity",
        choices=["full", "drop_molecular_size_related"],
        default="full",
        help="Optionally remove molecular-weight/size correlated descriptor families for sensitivity analysis.",
    )
    parser.add_argument("--style", type=Path, default=Path("style_journal_clean_v1.yaml"))
    parser.add_argument("--domains", nargs="+", default=list(SOURCE_TABLES), choices=sorted(SOURCE_TABLES))
    parser.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES), choices=list(SUPPORTED_SCOPES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--include-species",
        nargs="*",
        default=[],
        help="Latin species names whose species-endpoint tasks should be audited even when low-sample.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--hpo-objective", choices=["rmse", "mae"], default="rmse")
    parser.add_argument("--min-total", type=int, default=50)
    parser.add_argument("--min-train", type=int, default=35)
    parser.add_argument("--min-val", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-endpoint-tasks", type=int, default=24)
    parser.add_argument("--max-species", type=int, default=12)
    parser.add_argument("--max-species-endpoint-tasks", type=int, default=30)
    parser.add_argument("--max-hpo-rows", type=int, default=3000)
    parser.add_argument("--max-fit-rows", type=int, default=12000)
    parser.add_argument("--max-train-prediction-rows", type=int, default=5000)
    parser.add_argument("--max-plots", type=int, default=100000)
    parser.add_argument("--plot-bins", type=int, default=36)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_started = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_root / "tables"
    figures_dir = args.out_root / "figures"
    task_outputs_dir = args.out_root / "species_endpoint_outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    task_outputs_dir.mkdir(parents=True, exist_ok=True)

    models = [normalize_model_name(model) for model in args.models]
    include_species = normalize_species_names(args.include_species)
    skipped_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    descriptor_table_paths: list[Path] = []
    bundles: dict[str, DatasetBundle] = {}

    for domain in args.domains:
        print(f"[load] domain={domain}", flush=True)
        bundles[domain] = load_domain_bundle(
            args.db,
            domain=domain,
            descriptor_cache_path=args.descriptor_cache,
            seed=args.seed,
            descriptor_sensitivity=args.descriptor_sensitivity,
        )
        print(
            f"[loaded] domain={domain} rows={len(bundles[domain].frame)} "
            f"features={len(bundles[domain].feature_columns)} invalid_smiles={bundles[domain].invalid_smiles_count}",
            flush=True,
        )

    for domain in args.domains:
        bundle = bundles[domain]
        subtasks = build_subtasks(
            bundle.frame,
            domain=domain,
            scopes=args.scopes,
            min_total=args.min_total,
            max_endpoint_tasks=args.max_endpoint_tasks,
            max_species=args.max_species,
            max_species_endpoint_tasks=args.max_species_endpoint_tasks,
            include_species=include_species,
            skipped_rows=skipped_rows,
        )
        print(f"[subtasks] domain={domain} count={len(subtasks)}", flush=True)
        for subtask in subtasks:
            subtask_frame, subtask_features = select_subtask(bundle, subtask)
            split_parts = assign_outer_split(len(subtask_frame), seed=args.seed, validation_fraction=args.validation_fraction)
            train_idx = np.flatnonzero(split_parts == "train")
            val_idx = np.flatnonzero(split_parts == "validation")
            if subtask.scope == "species_endpoint":
                descriptor_table_paths.append(
                    write_subtask_descriptor_table(
                        subtask=subtask,
                        frame=subtask_frame,
                        features=subtask_features,
                        split_parts=split_parts,
                        task_outputs_dir=task_outputs_dir,
                    )
                )
            if len(train_idx) < args.min_train or len(val_idx) < args.min_val:
                skipped_rows.append(
                    skipped_subtask_row(
                        subtask,
                        reason="train_or_validation_below_threshold",
                        n_total=len(subtask_frame),
                        n_train=len(train_idx),
                        n_val=len(val_idx),
                    )
                )
                continue

            for model_name in models:
                result = run_model_for_subtask(
                    subtask=subtask,
                    frame=subtask_frame,
                    features=subtask_features,
                    feature_columns=bundle.feature_columns,
                    split_parts=split_parts,
                    model_name=model_name,
                    seed=args.seed,
                    n_trials=args.n_trials,
                    hpo_objective=args.hpo_objective,
                    max_hpo_rows=args.max_hpo_rows,
                    max_fit_rows=args.max_fit_rows,
                    max_train_prediction_rows=args.max_train_prediction_rows,
                    n_jobs=args.n_jobs,
                )
                if result["skip"] is not None:
                    skipped_rows.append(result["skip"])
                    continue
                metric_rows.append(result["metrics"])
                trial_rows.extend(result["trials"])
                if result["predictions"] is not None:
                    prediction_frames.append(result["predictions"])
                print(
                    "[done] "
                    f"{subtask.slug} model={model_name} "
                    f"val_r2={result['metrics']['val_r2']:.4f} "
                    f"val_rmse={result['metrics']['val_rmse']:.4f} "
                    f"val_mae={result['metrics']['val_mae']:.4f}",
                    flush=True,
                )

    metrics = pd.DataFrame(metric_rows)
    trials = pd.DataFrame(trial_rows)
    skipped = pd.DataFrame(skipped_rows)
    best = select_best_by_subtask(metrics)

    all_metrics_path = tables_dir / "traditional_ml_descriptor_effect_all_metrics.csv"
    best_path = tables_dir / "traditional_ml_descriptor_effect_best_by_subtask.csv"
    trial_path = tables_dir / "traditional_ml_descriptor_effect_hpo_trials.csv"
    skipped_path = tables_dir / "traditional_ml_descriptor_effect_skipped_subtasks.csv"
    prediction_path = tables_dir / "traditional_ml_descriptor_effect_prediction_rows.csv.gz"
    descriptor_reference_paths = write_descriptor_selection_files(args.out_root, bundles)
    summary_paths = write_summary_tables(metrics, best, tables_dir)

    write_csv(metrics, all_metrics_path)
    write_csv(best, best_path)
    write_csv(trials, trial_path)
    write_csv(skipped, skipped_path)

    prediction_rows = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not prediction_rows.empty:
        prediction_rows.to_csv(prediction_path, index=False, encoding="utf-8-sig", compression="gzip")
    else:
        prediction_path = Path("")

    structured_paths = write_structured_task_tables(
        metrics=metrics,
        best=best,
        trials=trials,
        prediction_rows=prediction_rows,
        task_outputs_dir=task_outputs_dir,
    )

    if not args.no_plots and not prediction_rows.empty and not metrics.empty:
        plot_count = write_plots(
            prediction_rows=prediction_rows,
            metrics=metrics,
            figures_dir=task_outputs_dir,
            style_path=args.style,
            bins=args.plot_bins,
            max_plots=args.max_plots,
        )
    else:
        plot_count = 0

    manifest = {
        "run_version": RUN_VERSION,
        "started_at_unix": run_started,
        "ended_at_unix": time.time(),
        "duration_seconds": time.time() - run_started,
        "db": str(args.db),
        "source_tables": {domain: SOURCE_TABLES[domain] for domain in args.domains},
        "domains": args.domains,
        "scopes": args.scopes,
        "models": models,
        "seed": args.seed,
        "outer_split": "row_random_8_2",
        "validation_fraction": args.validation_fraction,
        "hpo": {
            "library": "optuna",
            "sampler": "TPESampler",
            "n_trials": args.n_trials,
            "objective": args.hpo_objective,
            "max_hpo_rows": args.max_hpo_rows,
        },
        "feature_policy": {
            "descriptor_sensitivity": args.descriptor_sensitivity,
            "excluded_molecular_size_related_descriptors": (
                sorted(MOLECULAR_SIZE_RELATED_RAW_NAMES)
                if args.descriptor_sensitivity == "drop_molecular_size_related"
                else []
            ),
            "excluded_molecular_size_related_prefixes": (
                list(MOLECULAR_SIZE_RELATED_PREFIXES)
                if args.descriptor_sensitivity == "drop_molecular_size_related"
                else []
            ),
            "included": [
                "literature-guided union of RDKit 2D QSAR descriptors computed from smiles",
                "effect_level_x",
                "effect_level_x_fraction",
                "effect_level_x_log1p",
                "effect_level_x_present",
            ],
            "numeric_preprocessing": [
                f"descriptor values with absolute value > {MAX_ABS_DESCRIPTOR_VALUE:g} are treated as missing",
                "missing numeric features are imputed with train-only medians",
                "zero-variance features are removed inside each train-only preprocessing fit",
            ],
            "excluded": [
                "latin_name and all taxonomy/species context fields",
                "medium/domain context fields",
                "task_head/endpoint labels as features",
                "target/converted concentration fields",
            ],
        },
        "outputs": {
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
            "all_metrics": str(all_metrics_path),
            "best_by_subtask": str(best_path),
            "hpo_trials": str(trial_path),
            "skipped_subtasks": str(skipped_path),
            "best_prediction_rows": str(prediction_path) if str(prediction_path) else None,
            "summary_tables": {name: str(path) for name, path in summary_paths.items()},
            "plot_count": plot_count,
            "plot_style": "single_task_observed_vs_predicted_scatter",
            "task_outputs_dir": str(task_outputs_dir),
            "descriptor_table_count": len(descriptor_table_paths),
            "structured_tables": {name: str(path) for name, path in structured_paths.items()},
            "descriptor_selection_files": {name: str(path) for name, path in descriptor_reference_paths.items()},
        },
    }
    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    copy_to_summary(args.out_root, args.summary_dir, manifest)
    print(json.dumps({"out_root": str(args.out_root), "summary_dir": str(args.summary_dir), "plots": plot_count}, ensure_ascii=False, indent=2))


def load_domain_bundle(
    db_path: Path,
    *,
    domain: str,
    descriptor_cache_path: Path,
    seed: int,
    descriptor_sensitivity: str = "full",
) -> DatasetBundle:
    table = SOURCE_TABLES[domain]
    query = f"""
        SELECT
            aggregate_id,
            cas_number,
            dtxsid,
            chemical_name,
            smiles,
            species_number,
            latin_name,
            task_head,
            task_family,
            effect_family,
            effect_level_x,
            medium_domain,
            {TARGET_COLUMN}
        FROM "{table}"
        WHERE {TARGET_COLUMN} IS NOT NULL
          AND smiles IS NOT NULL
          AND TRIM(smiles) <> ''
          AND task_head IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(query, conn)
    frame["domain"] = domain
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN, "smiles", "task_head"]).reset_index(drop=True)
    descriptor_frame, invalid_smiles = build_descriptor_frame(
        frame["smiles"],
        descriptor_cache_path=descriptor_cache_path,
        seed=seed,
    )
    descriptor_features = descriptor_frame.reindex(frame["smiles"].astype(str).tolist()).reset_index(drop=True)
    descriptor_features = descriptor_features.mask(descriptor_features.abs() > MAX_ABS_DESCRIPTOR_VALUE)
    selected_descriptor_columns = select_qsar_descriptor_columns(
        descriptor_features.columns,
        exclude_molecular_size_related=descriptor_sensitivity == "drop_molecular_size_related",
    )
    descriptor_features = descriptor_features.loc[:, selected_descriptor_columns]
    effect_features = effect_level_feature_frame(frame["effect_level_x"])
    features = pd.concat(
        [
            descriptor_features,
            effect_features.reset_index(drop=True),
        ],
        axis=1,
    )
    valid_mask = descriptor_features.notna().any(axis=1)
    frame = frame.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    feature_columns = list(features.columns)
    return DatasetBundle(
        frame=frame,
        features=features,
        feature_columns=feature_columns,
        descriptor_cache_path=descriptor_cache_path,
        invalid_smiles_count=invalid_smiles,
    )


def build_descriptor_frame(
    smiles: Iterable[Any],
    *,
    descriptor_cache_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    del seed
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    descriptor_cache_path.parent.mkdir(parents=True, exist_ok=True)
    unique_smiles = sorted({str(value).strip() for value in smiles if str(value).strip()})
    descriptor_names = [name for name, _ in Descriptors.descList]
    cached = read_descriptor_cache(descriptor_cache_path)
    missing = [item for item in unique_smiles if item not in cached.index]
    invalid_count = 0
    if missing:
        rows: list[dict[str, Any]] = []
        for idx, smi in enumerate(missing, start=1):
            mol = Chem.MolFromSmiles(smi)
            row: dict[str, Any] = {"smiles": smi, "descriptor_valid": 0 if mol is None else 1}
            if mol is None:
                invalid_count += 1
                for name in descriptor_names:
                    row[f"rdkit_{safe_descriptor_name(name)}"] = np.nan
            else:
                for name, func in Descriptors.descList:
                    try:
                        value = func(mol)
                    except Exception:
                        value = np.nan
                    row[f"rdkit_{safe_descriptor_name(name)}"] = clean_float(value)
            rows.append(row)
            if idx % 1000 == 0:
                print(f"[descriptor-cache] computed {idx}/{len(missing)} missing molecules", flush=True)
        new_cache = pd.DataFrame(rows).set_index("smiles")
        cached = pd.concat([cached, new_cache], axis=0)
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        cached.to_csv(descriptor_cache_path, encoding="utf-8-sig", compression="gzip")
    if cached.empty:
        raise ValueError("No descriptors were generated.")
    cached = cached.replace([np.inf, -np.inf], np.nan)
    descriptor_columns = [column for column in cached.columns if column.startswith("rdkit_")]
    cached = cached[descriptor_columns]
    return cached, invalid_count


def read_descriptor_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=pd.Index([], name="smiles"))
    try:
        frame = pd.read_csv(path, index_col="smiles")
    except (OSError, ValueError):
        return pd.DataFrame(index=pd.Index([], name="smiles"))
    frame.index = frame.index.astype(str)
    return frame


def select_qsar_descriptor_columns(
    columns: Iterable[str],
    *,
    exclude_molecular_size_related: bool = False,
) -> list[str]:
    selected: list[str] = []
    for column in columns:
        if not str(column).startswith("rdkit_"):
            continue
        raw_name = str(column).removeprefix("rdkit_")
        if raw_name in QSAR_DESCRIPTOR_RAW_NAMES or raw_name.startswith(QSAR_DESCRIPTOR_PREFIXES):
            if exclude_molecular_size_related and is_molecular_size_related_descriptor(raw_name):
                continue
            selected.append(str(column))
    if not selected:
        raise ValueError("No RDKit columns matched the QSAR descriptor union.")
    return selected


def is_molecular_size_related_descriptor(raw_name: str) -> bool:
    return raw_name in MOLECULAR_SIZE_RELATED_RAW_NAMES or raw_name.startswith(MOLECULAR_SIZE_RELATED_PREFIXES)


def descriptor_category(column: str) -> str:
    raw_name = str(column).removeprefix("rdkit_")
    if raw_name in {"MolLogP", "MolMR"} or raw_name.startswith(("SlogP_VSA", "SMR_VSA")):
        return "hydrophobicity_refractivity"
    if raw_name in {"TPSA", "LabuteASA", "NumHAcceptors", "NumHDonors", "NHOHCount", "NOCount"}:
        return "polarity_hbond_surface"
    if raw_name.startswith(("PEOE_VSA", "EState_VSA", "VSA_EState")) or "EState" in raw_name:
        return "electronic_estate_surface"
    if raw_name.startswith("BCUT2D_") or "PartialCharge" in raw_name:
        return "charge_bcut"
    if raw_name.startswith("Chi") or raw_name in {"BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3"}:
        return "topology_connectivity_shape"
    if raw_name in {
        "NumRotatableBonds",
        "RingCount",
        "NumAromaticRings",
        "NumAliphaticRings",
        "NumSaturatedRings",
        "NumAromaticHeterocycles",
        "NumAromaticCarbocycles",
        "NumAliphaticHeterocycles",
        "NumAliphaticCarbocycles",
        "NumSaturatedHeterocycles",
        "NumSaturatedCarbocycles",
        "FractionCSP3",
    }:
        return "rings_flexibility_saturation"
    return "constitution_size_composition"


def descriptor_reference_key(category: str) -> str:
    mapping = {
        "hydrophobicity_refractivity": "wildman_crippen_1999",
        "polarity_hbond_surface": "ertl_2000_labute_2000",
        "electronic_estate_surface": "kier_hall_1999",
        "charge_bcut": "burden_pearlman_1999_rdkit",
        "topology_connectivity_shape": "randic_1975_kier_hall_1976_balaban_1982",
        "rings_flexibility_saturation": "todeschini_consonni_2009_rdkit",
        "constitution_size_composition": "todeschini_consonni_2009_rdkit",
    }
    return mapping.get(category, "todeschini_consonni_2009_rdkit")


def effect_level_feature_frame(series: pd.Series) -> pd.DataFrame:
    level = pd.to_numeric(series, errors="coerce")
    present = level.notna().astype(float)
    filled = level.fillna(0.0).clip(lower=0.0)
    return pd.DataFrame(
        {
            "effect_level_x": filled.astype(float),
            "effect_level_x_fraction": (filled / 100.0).astype(float),
            "effect_level_x_log1p": np.log1p(filled).astype(float),
            "effect_level_x_present": present.astype(float),
        },
        index=series.index,
    )


def build_subtasks(
    frame: pd.DataFrame,
    *,
    domain: str,
    scopes: list[str],
    min_total: int,
    max_endpoint_tasks: int,
    max_species: int,
    max_species_endpoint_tasks: int,
    skipped_rows: list[dict[str, Any]],
    include_species: set[str] | None = None,
) -> list[SubtaskSpec]:
    subtasks: list[SubtaskSpec] = []
    include_species = include_species or set()
    if "domain" in scopes:
        subtasks.append(SubtaskSpec(domain=domain, scope="domain", name=f"{domain}_all"))
    if "endpoint" in scopes:
        counts = frame["task_head"].value_counts(dropna=True)
        for task_head, n in counts.head(max_endpoint_tasks).items():
            if int(n) < min_total:
                skipped_rows.append(skipped_candidate_row(domain, "endpoint", str(task_head), "total_below_min_total", int(n)))
                continue
            subtasks.append(SubtaskSpec(domain=domain, scope="endpoint", name=str(task_head), task_head=str(task_head)))
    if "species" in scopes:
        species_counts = normalized_text(frame["latin_name"]).value_counts()
        for latin_name, n in species_counts.head(max_species).items():
            if latin_name == "<missing>":
                continue
            if int(n) < min_total:
                skipped_rows.append(skipped_candidate_row(domain, "species", latin_name, "total_below_min_total", int(n)))
                continue
            subtasks.append(SubtaskSpec(domain=domain, scope="species", name=latin_name, latin_name=latin_name))
    if "species_endpoint" in scopes:
        tmp = frame.assign(_latin_name=normalized_text(frame["latin_name"]))
        all_grouped = (
            tmp[tmp["_latin_name"] != "<missing>"]
            .groupby(["_latin_name", "task_head"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        grouped = all_grouped.head(max_species_endpoint_tasks).copy()
        if include_species:
            include_grouped = all_grouped[all_grouped["_latin_name"].isin(include_species)].copy()
            grouped = (
                pd.concat([grouped, include_grouped], ignore_index=True)
                .drop_duplicates(subset=["_latin_name", "task_head"], keep="first")
                .sort_values("n", ascending=False)
            )
        for row in grouped.to_dict(orient="records"):
            latin_name = str(row["_latin_name"])
            task_head = str(row["task_head"])
            n = int(row["n"])
            forced_low_sample = latin_name in include_species
            if n < min_total and not forced_low_sample:
                skipped_rows.append(
                    skipped_candidate_row(
                        domain,
                        "species_endpoint",
                        f"{latin_name}|{task_head}",
                        "total_below_min_total",
                        n,
                    )
                )
                continue
            subtasks.append(
                SubtaskSpec(
                    domain=domain,
                    scope="species_endpoint",
                    name=f"{latin_name}__{task_head}",
                    latin_name=latin_name,
                    task_head=task_head,
                )
            )
    return subtasks


def select_subtask(bundle: DatasetBundle, subtask: SubtaskSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = np.ones(len(bundle.frame), dtype=bool)
    if subtask.task_head is not None:
        mask &= bundle.frame["task_head"].astype(str).to_numpy() == subtask.task_head
    if subtask.latin_name is not None:
        mask &= normalized_text(bundle.frame["latin_name"]).to_numpy() == subtask.latin_name
    frame = bundle.frame.loc[mask].reset_index(drop=True)
    features = bundle.features.loc[mask].reset_index(drop=True)
    return frame, features


def assign_outer_split(n_rows: int, *, seed: int, validation_fraction: float) -> np.ndarray:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_rows * validation_fraction)))
    parts = np.full(n_rows, "train", dtype=object)
    parts[indices[:n_val]] = "validation"
    return parts


def task_output_dir(root: Path, subtask_or_row: Any) -> Path:
    domain = str(getattr(subtask_or_row, "domain", ""))
    latin_name = str(getattr(subtask_or_row, "latin_name", "") or getattr(subtask_or_row, "subtask_name", "") or "unknown_species")
    task_head = str(getattr(subtask_or_row, "task_head", "") or "unknown_endpoint")
    return root / domain_label(domain) / safe_slug(latin_name) / safe_slug(task_head)


def domain_label(domain: str) -> str:
    return DOMAIN_LABELS.get(str(domain), safe_slug(str(domain)))


def write_subtask_descriptor_table(
    *,
    subtask: SubtaskSpec,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    split_parts: np.ndarray,
    task_outputs_dir: Path,
) -> Path:
    descriptor_columns = [column for column in features.columns if str(column).startswith("rdkit_")]
    selected_columns = descriptor_columns + [column for column in EFFECT_FEATURE_COLUMNS if column in features.columns]
    metadata_columns = [
        "aggregate_id",
        "cas_number",
        "dtxsid",
        "chemical_name",
        "smiles",
        "latin_name",
        "task_head",
        "task_family",
        "effect_family",
        "effect_level_x",
        TARGET_COLUMN,
    ]
    available_metadata = [column for column in metadata_columns if column in frame.columns]
    descriptor_table = pd.concat(
        [
            frame.loc[:, available_metadata].reset_index(drop=True),
            pd.Series(split_parts, name="split_part"),
            features.loc[:, selected_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    for key, value in subtask_columns(subtask).items():
        descriptor_table[key] = value
    front = [
        "domain",
        "subtask_scope",
        "subtask_name",
        "subtask_slug",
        "split_part",
    ]
    descriptor_table = descriptor_table[front + [column for column in descriptor_table.columns if column not in set(front)]]
    out_dir = task_output_dir(task_outputs_dir, subtask) / "描述符表"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{subtask.slug}_compound_descriptor_table.csv"
    write_csv(descriptor_table, path)
    return path


def write_structured_task_tables(
    *,
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    trials: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    task_outputs_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if metrics.empty:
        return paths
    for row in metrics.drop_duplicates("subtask_slug").itertuples(index=False):
        task_dir = task_output_dir(task_outputs_dir, row)
        metric_dir = task_dir / "指标表"
        prediction_dir = task_dir / "预测值表"
        metric_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)

        slug = str(row.subtask_slug)
        task_metrics = metrics[metrics["subtask_slug"] == slug].copy()
        metric_path = metric_dir / f"{slug}_model_metrics.csv"
        write_csv(task_metrics, metric_path)
        paths[f"{slug}:metrics"] = metric_path

        task_best = best[best["subtask_slug"] == slug].copy() if not best.empty else pd.DataFrame()
        if not task_best.empty:
            best_path = metric_dir / f"{slug}_best_model.csv"
            write_csv(task_best, best_path)
            paths[f"{slug}:best"] = best_path

        task_trials = trials[trials["subtask_slug"] == slug].copy() if not trials.empty else pd.DataFrame()
        if not task_trials.empty:
            trial_path = metric_dir / f"{slug}_hpo_trials.csv"
            write_csv(task_trials, trial_path)
            paths[f"{slug}:trials"] = trial_path

        task_predictions = (
            prediction_rows[prediction_rows["subtask_slug"] == slug].copy()
            if not prediction_rows.empty
            else pd.DataFrame()
        )
        if not task_predictions.empty:
            prediction_path = prediction_dir / f"{slug}_prediction_rows.csv.gz"
            task_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig", compression="gzip")
            paths[f"{slug}:predictions"] = prediction_path
    return paths


def write_descriptor_selection_files(out_root: Path, bundles: dict[str, DatasetBundle]) -> dict[str, Path]:
    all_columns: set[str] = set()
    for bundle in bundles.values():
        all_columns.update(column for column in bundle.feature_columns if str(column).startswith("rdkit_"))
    rows = []
    for column in sorted(all_columns):
        category = descriptor_category(column)
        rows.append(
            {
                "descriptor_column": column,
                "rdkit_descriptor": column.removeprefix("rdkit_"),
                "category": category,
                "reference_key": descriptor_reference_key(category),
            }
        )
    reference_dir = out_root / "descriptor_selection"
    reference_dir.mkdir(parents=True, exist_ok=True)
    descriptor_list_path = reference_dir / "descriptor_selection_reference.csv"
    pd.DataFrame(rows).to_csv(descriptor_list_path, index=False, encoding="utf-8-sig")

    rationale_path = reference_dir / "descriptor_selection_rationale.md"
    rationale_path.write_text(build_descriptor_selection_rationale(rows), encoding="utf-8")
    return {"descriptor_list": descriptor_list_path, "rationale": rationale_path}


def build_descriptor_selection_rationale(rows: list[dict[str, str]]) -> str:
    category_counts = (
        pd.DataFrame(rows)
        .groupby(["category", "reference_key"], as_index=False)
        .size()
        .sort_values(["category", "reference_key"])
        if rows
        else pd.DataFrame(columns=["category", "reference_key", "size"])
    )
    lines = [
        "# Descriptor Selection Rationale",
        "",
        "This baseline uses one fixed literature-guided union of 2D molecular descriptors for every species-endpoint subtask.",
        "The goal is not to tune descriptors per subtask, but to make model comparisons reproducible and comparable across aquatic and soil endpoints.",
        "",
        "Species, taxonomy, medium, endpoint labels, concentration/target fields, and any biological context variables are excluded from model inputs.",
        "Only the selected RDKit descriptor columns plus effect-level numeric features are used.",
        "",
        "## Selected Descriptor Families",
        "",
    ]
    for row in category_counts.to_dict(orient="records"):
        lines.append(f"- `{row['category']}`: {int(row['size'])} descriptors; reference key `{row['reference_key']}`.")
    lines.extend(
        [
            "",
            "## Literature Basis",
            "",
            "- Hansch, C.; Fujita, T. Rho-Sigma-Pi Analysis. A Method for the Correlation of Biological Activity and Chemical Structure. J. Am. Chem. Soc. 1964, 86, 1616-1626. DOI: 10.1021/ja01062a035.",
            "- Randic, M. On Characterization of Molecular Branching. J. Am. Chem. Soc. 1975, 97, 6609-6615. DOI: 10.1021/ja00856a001.",
            "- Kier, L. B.; Hall, L. H. Molecular Connectivity in Chemistry and Drug Research. Academic Press, 1976.",
            "- Balaban, A. T. Highly Discriminating Distance-Based Topological Index. Chem. Phys. Lett. 1982, 89, 399-404. DOI: 10.1016/0009-2614(82)80009-2.",
            "- Kier, L. B.; Hall, L. H. Molecular Structure Description: The Electrotopological State. Academic Press, 1999.",
            "- Wildman, S. A.; Crippen, G. M. Prediction of Physicochemical Parameters by Atomic Contributions. J. Chem. Inf. Comput. Sci. 1999, 39, 868-873. DOI: 10.1021/ci990307l.",
            "- Ertl, P.; Rohde, B.; Selzer, P. Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions and Its Application to the Prediction of Drug Transport Properties. J. Med. Chem. 2000, 43, 3714-3717. DOI: 10.1021/jm000942e.",
            "- Labute, P. A Widely Applicable Set of Descriptors. J. Mol. Graph. Model. 2000, 18, 464-477. DOI: 10.1016/S1093-3263(00)00068-1.",
            "- Todeschini, R.; Consonni, V. Molecular Descriptors for Chemoinformatics. Wiley-VCH, 2009. DOI: 10.1002/9783527628766.",
            "- RDKit descriptor implementations are used for reproducible calculation of these 2D descriptor families.",
            "",
            "## Notes for Interpretation",
            "",
            "- The selected descriptors cover hydrophobicity, refractivity, polarity, H-bonding, molecular size, topology, branching, ring/flexibility, electronic state, charge, and VSA descriptor families that are common in QSAR.",
            "- These descriptors are chemically interpretable for ecotoxicity because they approximate membrane partitioning, steric size, polarity, ionization/charge distribution, and structural complexity.",
            "- No descriptor is selected by looking at validation performance of a specific subtask, so this avoids feature-selection leakage across train/validation rows.",
            "",
        ]
    )
    return "\n".join(lines)


def run_model_for_subtask(
    *,
    subtask: SubtaskSpec,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    split_parts: np.ndarray,
    model_name: str,
    seed: int,
    n_trials: int,
    hpo_objective: str,
    max_hpo_rows: int,
    max_fit_rows: int,
    max_train_prediction_rows: int,
    n_jobs: int,
) -> dict[str, Any]:
    start = time.time()
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        return {
            "metrics": None,
            "trials": [],
            "predictions": None,
            "skip": skipped_subtask_row(subtask, reason="optuna_not_installed", n_total=len(frame)),
        }
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_idx = np.flatnonzero(split_parts == "train")
    val_idx = np.flatnonzero(split_parts == "validation")
    x_all = features.astype(np.float32)
    y_all = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)

    train_idx = train_idx[np.isfinite(y_all[train_idx])]
    val_idx = val_idx[np.isfinite(y_all[val_idx])]
    if len(train_idx) < 2 or len(val_idx) < 2:
        return {
            "metrics": None,
            "trials": [],
            "predictions": None,
            "skip": skipped_subtask_row(
                subtask,
                reason="finite_target_rows_below_threshold",
                n_total=len(frame),
                n_train=len(train_idx),
                n_val=len(val_idx),
            ),
        }

    train_idx_for_hpo = sample_indices(train_idx, max_hpo_rows, seed=seed + 101)
    inner_train, inner_val = split_train_for_hpo(train_idx_for_hpo, seed=seed + 202)
    if len(inner_train) < 2 or len(inner_val) < 2:
        return {
            "metrics": None,
            "trials": [],
            "predictions": None,
            "skip": skipped_subtask_row(subtask, reason="inner_hpo_split_too_small", n_total=len(frame)),
        }

    preprocessor = NumericPreprocessor.fit(x_all.iloc[inner_train])
    x_inner_train = preprocessor.transform(x_all.iloc[inner_train])
    y_inner_train = y_all[inner_train]
    x_inner_val = preprocessor.transform(x_all.iloc[inner_val])
    y_inner_val = y_all[inner_val]

    trial_records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = suggest_params(
            trial,
            model_name,
            n_jobs=n_jobs,
            seed=seed,
            n_features=x_inner_train.shape[1],
            n_train=len(y_inner_train),
        )
        model = build_model(model_name, params=params, seed=seed, n_jobs=n_jobs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_inner_train, y_inner_train)
                pred = np.asarray(model.predict(x_inner_val), dtype=float).reshape(-1)
        except Exception as exc:
            trial_records.append(
                {
                    **subtask_columns(subtask),
                    "model": model_name,
                    "trial_number": int(trial.number),
                    "objective": hpo_objective,
                    "objective_value": float(MAX_ABS_PREDICTION_VALUE),
                    "trial_rmse": float("nan"),
                    "trial_mae": float("nan"),
                    "trial_r2": float("nan"),
                    "trial_status": "failed",
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
            return float(MAX_ABS_PREDICTION_VALUE)
        if not valid_prediction_vector(pred):
            trial_records.append(
                {
                    **subtask_columns(subtask),
                    "model": model_name,
                    "trial_number": int(trial.number),
                    "objective": hpo_objective,
                    "objective_value": float(MAX_ABS_PREDICTION_VALUE),
                    "trial_rmse": float("nan"),
                    "trial_mae": float("nan"),
                    "trial_r2": float("nan"),
                    "trial_status": "invalid_prediction",
                    "failure_reason": f"nonfinite_or_abs_gt_{MAX_ABS_PREDICTION_VALUE:g}",
                    "params_json": json.dumps(params, sort_keys=True),
                }
            )
            return float(MAX_ABS_PREDICTION_VALUE)
        metrics = regression_metrics(y_inner_val.tolist(), pred.tolist())
        value = float(metrics[hpo_objective])
        trial_records.append(
            {
                **subtask_columns(subtask),
                "model": model_name,
                "trial_number": int(trial.number),
                "objective": hpo_objective,
                "objective_value": value,
                "trial_rmse": float(metrics["rmse"]),
                "trial_mae": float(metrics["mae"]),
                "trial_r2": float(metrics["r2"]),
                "trial_status": "ok",
                "failure_reason": "",
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        return value

    try:
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as exc:
        return {
            "metrics": None,
            "trials": trial_records,
            "predictions": None,
            "skip": skipped_subtask_row(subtask, reason=f"hpo_failed:{type(exc).__name__}:{exc}", n_total=len(frame)),
        }

    best_params = suggest_params_from_study(study, model_name, n_jobs=n_jobs, seed=seed)
    fit_train_idx = sample_indices(train_idx, max_fit_rows, seed=seed + 303)
    final_preprocessor = NumericPreprocessor.fit(x_all.iloc[fit_train_idx])
    x_fit = final_preprocessor.transform(x_all.iloc[fit_train_idx])
    y_fit = y_all[fit_train_idx]
    best_params = coerce_model_params_for_fit(best_params, model_name, n_features=x_fit.shape[1], n_train=len(y_fit))
    model = build_model(model_name, params=best_params, seed=seed, n_jobs=n_jobs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_fit, y_fit)
    except Exception as exc:
        if model_name == "pls" and int(best_params.get("n_components", 1)) > 1:
            fallback_params = dict(best_params)
            fallback_params["n_components"] = 1
            fallback_model = build_model(model_name, params=fallback_params, seed=seed, n_jobs=n_jobs)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fallback_model.fit(x_fit, y_fit)
                model = fallback_model
                best_params = fallback_params
            except Exception as fallback_exc:
                return {
                    "metrics": None,
                    "trials": trial_records,
                    "predictions": None,
                    "skip": skipped_subtask_row(
                        subtask,
                        reason=(
                            f"final_fit_failed:{type(exc).__name__}:{exc};"
                            f"pls_fallback_failed:{type(fallback_exc).__name__}:{fallback_exc}"
                        ),
                        n_total=len(frame),
                    ),
                }
        else:
            return {
                "metrics": None,
                "trials": trial_records,
                "predictions": None,
                "skip": skipped_subtask_row(subtask, reason=f"final_fit_failed:{type(exc).__name__}:{exc}", n_total=len(frame)),
            }

    train_pred_idx = sample_indices(train_idx, max_train_prediction_rows, seed=seed + 404)
    x_train_pred = final_preprocessor.transform(x_all.iloc[train_pred_idx])
    x_val = final_preprocessor.transform(x_all.iloc[val_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pred = np.asarray(model.predict(x_train_pred), dtype=float).reshape(-1)
        val_pred = np.asarray(model.predict(x_val), dtype=float).reshape(-1)
    if not valid_prediction_vector(train_pred) or not valid_prediction_vector(val_pred):
        return {
            "metrics": None,
            "trials": trial_records,
            "predictions": None,
            "skip": skipped_subtask_row(
                subtask,
                reason=f"final_prediction_invalid_or_abs_gt_{MAX_ABS_PREDICTION_VALUE:g}",
                n_total=len(frame),
                n_train=len(train_idx),
                n_val=len(val_idx),
            ),
        }
    train_metrics = regression_metrics(y_all[train_pred_idx].tolist(), train_pred.tolist())
    val_metrics = regression_metrics(y_all[val_idx].tolist(), val_pred.tolist())
    metric_row = {
        **subtask_columns(subtask),
        "model": model_name,
        "n_total": int(len(frame)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_fit_train": int(len(fit_train_idx)),
        "n_train_metric": int(len(train_pred_idx)),
        "n_features": int(len(feature_columns)),
        "n_trials": int(n_trials),
        "hpo_objective": hpo_objective,
        "best_objective_value": float(study.best_value),
        "train_r2": float(train_metrics["r2"]),
        "train_rmse": float(train_metrics["rmse"]),
        "train_mae": float(train_metrics["mae"]),
        "train_huber_loss": float(train_metrics["huber_loss"]),
        "val_r2": float(val_metrics["r2"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_mae": float(val_metrics["mae"]),
        "val_huber_loss": float(val_metrics["huber_loss"]),
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "duration_seconds": float(time.time() - start),
    }
    predictions = build_prediction_rows(
        subtask=subtask,
        model_name=model_name,
        frame=frame,
        train_idx=train_pred_idx,
        val_idx=val_idx,
        y_all=y_all,
        train_pred=train_pred,
        val_pred=val_pred,
    )
    return {"metrics": metric_row, "trials": trial_records, "predictions": predictions, "skip": None}


class NumericPreprocessor:
    def __init__(self, columns: list[str], medians: pd.Series, keep_mask: pd.Series) -> None:
        self.columns = columns
        self.medians = medians
        self.keep_mask = keep_mask

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "NumericPreprocessor":
        numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        medians = numeric.median(axis=0).fillna(0.0)
        filled = numeric.fillna(medians)
        variance = filled.var(axis=0)
        keep_mask = variance.fillna(0.0) > 0.0
        if not bool(keep_mask.any()):
            keep_mask[:] = True
        return cls(columns=list(numeric.columns), medians=medians, keep_mask=keep_mask)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        filled = numeric.fillna(self.medians)
        selected = filled.loc[:, self.keep_mask]
        return selected.to_numpy(dtype=np.float32)


def split_train_for_hpo(indices: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * 0.2)))
    return shuffled[n_val:], shuffled[:n_val]


def sample_indices(indices: np.ndarray, max_rows: int, *, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(sampled)


def valid_prediction_vector(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(np.isfinite(values).all() and np.nanmax(np.abs(values)) <= MAX_ABS_PREDICTION_VALUE)


def suggest_params(
    trial: Any,
    model_name: str,
    *,
    n_jobs: int,
    seed: int,
    n_features: int | None = None,
    n_train: int | None = None,
) -> dict[str, Any]:
    del n_jobs, seed
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 160),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.35, 1.0),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 18, 24, 32]),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.35, 1.0),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 18, 24, 32]),
        }
    if model_name == "knn":
        max_neighbors = max(2, min(35, int((n_train or 20) - 1)))
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, max_neighbors),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if model_name == "pls":
        max_components = max(1, min(8, int(n_features or 8), int((n_train or 8) - 1)))
        return {
            "n_components": trial.suggest_int("n_components", 1, max_components),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 120, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if model_name == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.95),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params_from_study(study: Any, model_name: str, *, n_jobs: int, seed: int) -> dict[str, Any]:
    del n_jobs, seed
    params = dict(study.best_params)
    if model_name in {"extra_trees", "random_forest"} and "max_depth" in params and params["max_depth"] == "None":
        params["max_depth"] = None
    return params


def coerce_model_params_for_fit(params: dict[str, Any], model_name: str, *, n_features: int, n_train: int) -> dict[str, Any]:
    coerced = dict(params)
    if model_name == "knn" and "n_neighbors" in coerced:
        coerced["n_neighbors"] = int(max(1, min(int(coerced["n_neighbors"]), max(1, n_train))))
    if model_name == "pls" and "n_components" in coerced:
        upper = max(1, min(int(n_features), max(1, int(n_train) - 1)))
        coerced["n_components"] = int(max(1, min(int(coerced["n_components"]), upper)))
    return coerced


def build_model(model_name: str, *, params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            **params,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
            **params,
        )
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **params)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(**params),
        )
    if model_name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            PLSRegression(scale=False, max_iter=1000, tol=1e-06, **params),
        )
    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed, **params)
    if model_name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            ElasticNet(random_state=seed, max_iter=8000, **params),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_prediction_rows(
    *,
    subtask: SubtaskSpec,
    model_name: str,
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_all: np.ndarray,
    train_pred: np.ndarray,
    val_pred: np.ndarray,
) -> pd.DataFrame:
    train = frame.iloc[train_idx][
        ["aggregate_id", "cas_number", "dtxsid", "smiles", "latin_name", "task_head", "task_family", "effect_level_x"]
    ].copy()
    train["split_part"] = "train"
    train["y_true"] = y_all[train_idx]
    train["y_pred"] = train_pred
    val = frame.iloc[val_idx][
        ["aggregate_id", "cas_number", "dtxsid", "smiles", "latin_name", "task_head", "task_family", "effect_level_x"]
    ].copy()
    val["split_part"] = "validation"
    val["y_true"] = y_all[val_idx]
    val["y_pred"] = val_pred
    predictions = pd.concat([train, val], ignore_index=True)
    for key, value in subtask_columns(subtask).items():
        predictions[key] = value
    predictions["model"] = model_name
    predictions["residual"] = predictions["y_pred"] - predictions["y_true"]
    front = [
        "domain",
        "subtask_scope",
        "subtask_name",
        "subtask_slug",
        "model",
        "split_part",
        "y_true",
        "y_pred",
        "residual",
    ]
    return predictions[front + [column for column in predictions.columns if column not in set(front)]]


def select_best_by_subtask(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ranked = metrics.sort_values(
        ["subtask_slug", "val_r2", "val_mae", "val_rmse"],
        ascending=[True, False, True, True],
    ).copy()
    return ranked.groupby("subtask_slug", as_index=False).head(1).reset_index(drop=True)


def write_summary_tables(metrics: pd.DataFrame, best: pd.DataFrame, tables_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if metrics.empty:
        return paths
    for scope in ("domain", "endpoint", "species", "species_endpoint"):
        subset = metrics[metrics["subtask_scope"] == scope].copy()
        if subset.empty:
            continue
        path = tables_dir / f"traditional_ml_descriptor_effect_{scope}_metrics.csv"
        write_csv(subset, path)
        paths[f"{scope}_metrics"] = path
    model_comparison = (
        metrics.groupby(["domain", "model"], as_index=False)
        .agg(
            subtask_count=("subtask_slug", "nunique"),
            total_validation_n=("n_validation", "sum"),
            mean_val_r2=("val_r2", "mean"),
            median_val_r2=("val_r2", "median"),
            weighted_val_rmse=("val_rmse", lambda x: weighted_metric(x, metrics.loc[x.index, "n_validation"])),
            weighted_val_mae=("val_mae", lambda x: weighted_metric(x, metrics.loc[x.index, "n_validation"])),
        )
        .sort_values(["domain", "weighted_val_rmse", "weighted_val_mae"], ascending=[True, True, True])
    )
    model_comparison_path = tables_dir / "traditional_ml_descriptor_effect_model_comparison_summary.csv"
    write_csv(model_comparison, model_comparison_path)
    paths["model_comparison_summary"] = model_comparison_path
    if not best.empty:
        best_by_scope = (
            best.groupby(["domain", "subtask_scope", "model"], as_index=False)
            .agg(
                subtask_count=("subtask_slug", "count"),
                total_validation_n=("n_validation", "sum"),
                mean_val_r2=("val_r2", "mean"),
                weighted_val_rmse=("val_rmse", lambda x: weighted_metric(x, best.loc[x.index, "n_validation"])),
                weighted_val_mae=("val_mae", lambda x: weighted_metric(x, best.loc[x.index, "n_validation"])),
            )
            .sort_values(["domain", "subtask_scope", "mean_val_r2"], ascending=[True, True, False])
        )
        path = tables_dir / "traditional_ml_descriptor_effect_best_scope_summary.csv"
        write_csv(best_by_scope, path)
        paths["best_scope_summary"] = path
    return paths


def weighted_metric(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not bool(mask.any()):
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def write_plots(
    *,
    prediction_rows: pd.DataFrame,
    metrics: pd.DataFrame,
    figures_dir: Path,
    style_path: Path,
    bins: int,
    max_plots: int,
) -> int:
    import matplotlib.pyplot as plt
    import yaml
    from matplotlib import font_manager

    del bins
    style = {}
    if style_path.exists():
        style = yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
    apply_plot_style(plt, font_manager, style)

    ranked = metrics.sort_values(["domain", "latin_name", "task_head", "model"]).head(max_plots)
    plot_count = 0
    for row in ranked.itertuples(index=False):
        subset = prediction_rows[
            (prediction_rows["subtask_slug"] == row.subtask_slug) & (prediction_rows["model"] == row.model)
        ].copy()
        if subset.empty:
            continue
        figure_dir = task_output_dir(figures_dir, row) / "图表"
        figure_dir.mkdir(parents=True, exist_ok=True)
        out_base = figure_dir / f"{safe_slug(str(row.model))}_observed_vs_predicted_scatter"
        plot_observed_predicted_scatter(subset, out_base=out_base, style=style)
        plot_count += 1
    return plot_count


def plot_observed_predicted_scatter(
    frame: pd.DataFrame,
    *,
    out_base: Path,
    style: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])].copy()
    if frame.empty:
        return
    colors = style.get("colors", {}).get("data_split", {})
    train_color = colors.get("holdout", "#C44E52")
    validation_color = "#5B5FD6"

    fig, ax = plt.subplots(figsize=figsize_inches(style, "single_column_square", default=(4.0, 4.0)))
    train = frame[frame["split_part"] == "train"]
    validation = frame[frame["split_part"] == "validation"]
    if not train.empty:
        ax.scatter(
            train["y_pred"],
            train["y_true"],
            s=23,
            marker="o",
            color=train_color,
            alpha=0.62,
            edgecolors="none",
            label="Training set",
            rasterized=True,
        )
    if not validation.empty:
        ax.scatter(
            validation["y_pred"],
            validation["y_true"],
            s=34,
            marker="^",
            color=validation_color,
            alpha=0.82,
            edgecolors="none",
            label="Validation set",
            rasterized=True,
        )

    axis_values = pd.concat([frame["y_true"], frame["y_pred"]], ignore_index=True)
    axis_min = float(axis_values.quantile(0.005))
    axis_max = float(axis_values.quantile(0.995))
    if not np.isfinite(axis_min) or not np.isfinite(axis_max) or axis_max <= axis_min:
        axis_min = float(np.nanmin([frame["y_true"].min(), frame["y_pred"].min()]))
        axis_max = float(np.nanmax([frame["y_true"].max(), frame["y_pred"].max()]))
    padding = max((axis_max - axis_min) * 0.05, 0.5)
    axis_min -= padding
    axis_max += padding
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#8A8A8A", linewidth=1.35, linestyle="-", zorder=0)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted pTox", fontweight="bold")
    ax.set_ylabel("Observed pTox", fontweight="bold")
    ax.set_title(scatter_title(frame), fontsize=10.5, fontweight="bold", pad=5)
    if len(validation) >= 2:
        metrics = regression_metrics(validation["y_true"].tolist(), validation["y_pred"].tolist())
        ax.text(
            0.03,
            0.97,
            f"Val R\u00b2={metrics['r2']:.2f}\nRMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            color="#333333",
        )
    legend = ax.legend(frameon=False, fontsize=8.0, loc="lower right", handletextpad=0.35, labelspacing=0.25)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    style_axes(ax, style)
    ax.grid(False)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix), dpi=450 if suffix == ".png" else None)
    plt.close(fig)


def scatter_title(frame: pd.DataFrame) -> str:
    first = frame.iloc[0]
    scope = str(first.get("subtask_scope", ""))
    task_head = str(first.get("task_head", "") or "")
    latin_name = str(first.get("latin_name", "") or "")
    if task_head:
        return task_head
    if scope == "domain":
        return "All tasks"
    if scope == "species" and latin_name:
        return latin_name
    return str(first.get("subtask_name", ""))


def figure_task_folder(row: Any) -> Path:
    task_head = str(getattr(row, "task_head", "") or "")
    latin_name = str(getattr(row, "latin_name", "") or "")
    if task_head:
        return Path(safe_slug(task_head))
    if latin_name:
        return Path(safe_slug(latin_name))
    return Path("all_tasks")


def plot_binned_observed_predicted(
    frame: pd.DataFrame,
    *,
    out_base: Path,
    style: dict[str, Any],
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    frame = frame[np.isfinite(frame["y_true"]) & np.isfinite(frame["y_pred"])].copy()
    if frame.empty:
        return
    low, high = frame["y_true"].quantile([0.005, 0.995]).tolist()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = frame["y_true"].min(), frame["y_true"].max()
    if high <= low:
        return
    edges = np.linspace(low, high, bins + 1)
    frame["_bin"] = pd.cut(frame["y_true"], bins=edges, include_lowest=True, labels=False)
    summary = (
        frame.dropna(subset=["_bin"])
        .groupby(["split_part", "_bin"], as_index=False)
        .agg(
            observed_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
            pred_q25=("y_pred", lambda x: float(np.quantile(x, 0.25))),
            pred_q75=("y_pred", lambda x: float(np.quantile(x, 0.75))),
            residual_median=("residual", "median"),
            residual_q25=("residual", lambda x: float(np.quantile(x, 0.25))),
            residual_q75=("residual", lambda x: float(np.quantile(x, 0.75))),
            n=("y_true", "size"),
        )
    )
    train = frame[frame["split_part"] == "train"]
    val = frame[frame["split_part"] == "validation"]
    train_metrics = regression_metrics(train["y_true"].tolist(), train["y_pred"].tolist()) if len(train) >= 2 else {}
    val_metrics = regression_metrics(val["y_true"].tolist(), val["y_pred"].tolist()) if len(val) >= 2 else {}
    colors = style.get("colors", {}).get("data_split", {})
    train_color = colors.get("train", "#4C78A8")
    val_color = colors.get("holdout", "#C44E52")

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize_inches(style, "calibration_residual_double", default=(7.0, 5.3)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    for split_part, color, label in (
        ("train", train_color, "Train data"),
        ("validation", val_color, "Validation data"),
    ):
        part = summary[summary["split_part"] == split_part].sort_values("observed_mean")
        if part.empty:
            continue
        ax_top.fill_between(
            part["observed_mean"],
            part["pred_q25"],
            part["pred_q75"],
            color=color,
            alpha=0.14,
            linewidth=0,
            label=f"{label} IQR",
        )
        ax_top.plot(part["observed_mean"], part["pred_mean"], color=color, marker="o", linewidth=1.25, markersize=3.0, label=f"{label} mean")
        ax_bottom.fill_between(
            part["observed_mean"],
            part["residual_q25"],
            part["residual_q75"],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax_bottom.plot(
            part["observed_mean"],
            part["residual_median"],
            color=color,
            marker="o",
            linewidth=1.05,
            markersize=2.7,
            label=f"{label} median residual",
        )
    min_axis = min(frame["y_true"].min(), frame["y_pred"].min())
    max_axis = max(frame["y_true"].max(), frame["y_pred"].max())
    ax_top.plot([min_axis, max_axis], [min_axis, max_axis], color="#222222", linestyle="--", linewidth=0.9, label="1:1 line")
    ax_bottom.axhline(0.0, color="#333333", linewidth=0.8)
    metric_lines = [
        format_metric_text("Train", train_metrics),
        format_metric_text("Validation", val_metrics),
    ]
    ax_top.text(0.035, 0.94, "\n".join(metric_lines), transform=ax_top.transAxes, va="top", ha="left", color="#AA2A2A", fontsize=7.2)
    if "r2" in val_metrics:
        ax_top.text(
            0.055,
            0.69,
            f"Val R\u00b2={val_metrics['r2']:.2f}",
            transform=ax_top.transAxes,
            color="#B40000",
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="top",
        )
    ax_top.set_ylabel("Binned predicted pTox")
    ax_bottom.set_ylabel("Residuals\n(Pred. - Obs.)")
    ax_bottom.set_xlabel("Observed pTox bin mean")
    ax_top.legend(frameon=False, fontsize=7, loc="lower right")
    ax_bottom.legend(frameon=False, fontsize=7, loc="lower left")
    for axis in (ax_top, ax_bottom):
        style_axes(axis, style)
    note = f"{bins} equal-width bins over central 99% observed pTox; metrics use all plotted train/validation records."
    fig.text(0.015, 0.01, note, ha="left", va="bottom", fontsize=6.5, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for suffix in (".png", ".svg"):
        fig.savefig(out_base.with_suffix(suffix))
    plt.close(fig)


def format_metric_text(label: str, metrics: dict[str, float]) -> str:
    if not metrics:
        return f"{label} metrics unavailable"
    return f"{label} R\u00b2 = {metrics['r2']:.2f}, RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}"


def apply_plot_style(plt: Any, font_manager: Any, style: dict[str, Any]) -> None:
    rc = style.get("matplotlib_rc", {})
    if isinstance(rc, dict):
        plt.rcParams.update(rc)
    font_path = Path(style.get("fonts", {}).get("english", {}).get("regular", ""))
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = style.get("fonts", {}).get("english", {}).get("family_name", "Arial")
    font_sizes = style.get("font_sizes_pt", {})
    if font_sizes:
        plt.rcParams["axes.labelsize"] = float(font_sizes.get("axis_label", 8.0))
        plt.rcParams["xtick.labelsize"] = float(font_sizes.get("tick_label", 7.0))
        plt.rcParams["ytick.labelsize"] = float(font_sizes.get("tick_label", 7.0))
        plt.rcParams["legend.fontsize"] = float(font_sizes.get("legend", 7.0))


def style_axes(axis: Any, style: dict[str, Any]) -> None:
    axes = style.get("axes", {})
    ticks = style.get("ticks", {})
    grid = style.get("grid", {})
    for spine in axis.spines.values():
        spine.set_color(axes.get("spine_color", "#333333"))
        spine.set_linewidth(max(1.35, float(axes.get("spine_linewidth", 0.75))))
    axis.tick_params(
        direction=ticks.get("direction", "out"),
        length=float(ticks.get("major_size", 3.0)),
        width=max(1.15, float(ticks.get("major_width", 0.65))),
        colors=ticks.get("color", "#333333"),
        pad=float(ticks.get("pad", 2.5)),
    )
    axis.grid(
        bool(grid.get("show", True)),
        color=grid.get("color", "#DDE3EA"),
        linewidth=float(grid.get("linewidth", 0.45)),
        linestyle=grid.get("linestyle", "--"),
        alpha=float(grid.get("alpha", 0.72)),
    )


def figsize_inches(style: dict[str, Any], key: str, *, default: tuple[float, float]) -> tuple[float, float]:
    sizes = style.get("figure_sizes_mm", {})
    if key not in sizes:
        return default
    width_mm, height_mm = sizes[key]
    return float(width_mm) / 25.4, float(height_mm) / 25.4


def copy_to_summary(out_root: Path, summary_dir: Path, manifest: dict[str, Any]) -> None:
    import shutil

    summary_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    tables_src = out_root / "tables"
    task_outputs_src = out_root / "species_endpoint_outputs"
    descriptor_selection_src = out_root / "descriptor_selection"
    tables_dst = summary_dir / "指标表"
    tables_dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(tables_src.glob("*.csv*")):
        dst = tables_dst / src.name
        shutil.copy2(src, dst)
        copied.append({"category": "table", "source_path": str(src), "summary_path": str(dst)})
    for src in sorted(task_outputs_src.rglob("*")):
        if src.is_dir():
            continue
        dst = summary_dir / src.relative_to(task_outputs_src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"category": "task_output", "source_path": str(src), "summary_path": str(dst)})
    for src in sorted(descriptor_selection_src.glob("*")):
        if src.is_dir():
            continue
        dst = summary_dir / "特征选择依据" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"category": "descriptor_selection", "source_path": str(src), "summary_path": str(dst)})
    manifest_src = out_root / "manifest.json"
    if manifest_src.exists():
        dst = summary_dir / "manifest.json"
        shutil.copy2(manifest_src, dst)
        copied.append({"category": "manifest", "source_path": str(manifest_src), "summary_path": str(dst)})
    pd.DataFrame(copied).to_csv(summary_dir / "source_path_manifest.csv", index=False, encoding="utf-8-sig")
    readme = build_summary_readme(manifest)
    (summary_dir / "README.md").write_text(readme, encoding="utf-8")


def build_summary_readme(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 机器学习基线_分子描述符效应水平",
            "",
            f"Run version: `{manifest['run_version']}`.",
            "",
            "## Scope",
            "",
            "- Local traditional machine-learning baselines for aquatic and soil pTox tables, split by species and toxicity endpoint.",
            "- Models: XGBoost, LightGBM, random forest, KNN, and PLS unless overridden by the command line.",
            "- Features are restricted to a literature-guided union of RDKit 2D QSAR descriptors plus effect-level numeric features.",
            "- Species identity, taxonomy, medium labels, endpoint labels, and target/concentration fields are not used as model inputs.",
            "- The split is a reproducible row-random 8:2 train/validation split, so this is an interpolation baseline rather than chemical-family extrapolation evidence.",
            "",
            "## Files",
            "",
            "- `指标表/traditional_ml_descriptor_effect_all_metrics.csv`: all model and subtask metrics.",
            "- `指标表/traditional_ml_descriptor_effect_best_by_subtask.csv`: best model per subtask by validation R2, with MAE/RMSE tie-breaks.",
            "- `指标表/traditional_ml_descriptor_effect_hpo_trials.csv`: Optuna TPE trial history.",
            "- `指标表/traditional_ml_descriptor_effect_skipped_subtasks.csv`: low-sample or failed subtask audit.",
            "- `指标表/traditional_ml_descriptor_effect_prediction_rows.csv.gz`: train/validation prediction rows for every completed model.",
            "- `水生/<物种>/<终点>/图表/`: one simple observed-vs-predicted scatter plot per model.",
            "- `水生/<物种>/<终点>/指标表/` and `土壤/<物种>/<终点>/指标表/`: per-task metrics, best model, and HPO trial history.",
            "- `水生/<物种>/<终点>/描述符表/` and `土壤/<物种>/<终点>/描述符表/`: compound descriptor tables used by each subtask.",
            "- `特征选择依据/`: descriptor list and literature rationale.",
            "",
            "## Interpretation Notes",
            "",
            "- Strong validation performance here means same-distribution interpolation is easy for a descriptor-only baseline.",
            "- It should not be used to replace chemical/scaffold holdout conclusions.",
            "- Species-level results are trained after filtering to a species, but the species name itself is not a feature.",
            "",
        ]
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def subtask_columns(subtask: SubtaskSpec) -> dict[str, Any]:
    return {
        "domain": subtask.domain,
        "subtask_scope": subtask.scope,
        "subtask_name": subtask.name,
        "subtask_slug": subtask.slug,
        "task_head": subtask.task_head or "",
        "latin_name": subtask.latin_name or "",
    }


def skipped_subtask_row(
    subtask: SubtaskSpec,
    *,
    reason: str,
    n_total: int,
    n_train: int | None = None,
    n_val: int | None = None,
) -> dict[str, Any]:
    return {
        **subtask_columns(subtask),
        "reason": reason,
        "n_total": int(n_total),
        "n_train": "" if n_train is None else int(n_train),
        "n_validation": "" if n_val is None else int(n_val),
    }


def skipped_candidate_row(domain: str, scope: str, name: str, reason: str, n_total: int) -> dict[str, Any]:
    subtask = SubtaskSpec(domain=domain, scope=scope, name=name)
    return skipped_subtask_row(subtask, reason=reason, n_total=n_total)


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "lgbm": "lightgbm",
        "light_gbm": "lightgbm",
        "xgb": "xgboost",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "k_neighbors": "knn",
        "k_nearest_neighbors": "knn",
        "kneighbors": "knn",
        "partial_least_squares": "pls",
        "pls_regression": "pls",
        "extratrees": "extra_trees",
        "hgb": "hist_gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
        "elasticnet": "elastic_net",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {
        "lightgbm",
        "xgboost",
        "extra_trees",
        "random_forest",
        "knn",
        "pls",
        "hist_gradient_boosting",
        "elastic_net",
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(supported)}")
    return normalized


def normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().replace("", "<missing>")


def normalize_species_names(values: Iterable[str]) -> set[str]:
    return {
        str(value).strip()
        for value in values
        if str(value).strip() and str(value).strip() != "<missing>"
    }


def safe_descriptor_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
    return slug[:160] if slug else "subtask"


def clean_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    if not math.isfinite(numeric):
        return float("nan")
    if abs(numeric) > MAX_ABS_DESCRIPTOR_VALUE:
        return float("nan")
    return numeric


if __name__ == "__main__":
    main()
