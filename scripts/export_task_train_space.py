from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsar_tl.training.baseline import add_duration_nonlinear_features, load_split_frame
from qsar_tl.training.deep_experiment import encode_category_id


DEFAULT_ID_COLUMNS = (
    "split_name",
    "split_part",
    "task_head",
    "task_family",
    "effect_family",
    "effect_level_x",
    "target_name",
    "target_family",
    "target_basis",
    "medium_domain",
    "cas_number",
    "dtxsid",
    "chemical_name",
    "smiles",
    "species_number",
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
    "primary_medium",
    "habitat_labels",
    "organism_habitat",
    "media_type",
    "organism_lifestage",
    "duration_bin_h",
    "duration_log1p_h",
    "duration_sqrt_h",
)
DEFAULT_TARGET_COLUMNS = (
    "target_value_mean",
    "target_value_median",
    "target_value_weighted_mean",
    "target_value_count",
    "target_value_min",
    "target_value_max",
)
DEFAULT_EMBEDDING_FIELDS = (
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
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-task chemical/species training-space tables for plotting QSAR transfer-learning "
            "application-domain or data-coverage figures."
        )
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-table", default="aggregated_task_records_aquatic_soil_ptox_qc")
    parser.add_argument("--split-name", default="M_v2_aquatic_to_soil_ptox_adapt_C_f100")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-parts", nargs="+", default=["train", "finetune"])
    parser.add_argument("--target-column", default="target_value_mean")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional trained run directory containing preprocessing.json and best_model.pt.",
    )
    parser.add_argument(
        "--embedding-fields",
        default=",".join(DEFAULT_EMBEDDING_FIELDS),
        help="Comma-separated categorical fields to concatenate for species embedding export.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_split_frame(args.db, split_name=args.split_name, source_table=args.source_table)
    frame = add_duration_nonlinear_features(frame)
    split_parts = {str(part).strip().lower() for part in args.split_parts if str(part).strip()}
    if split_parts:
        frame = frame[frame["split_part"].astype("string").str.lower().isin(split_parts)].copy()
    if frame.empty:
        raise ValueError(f"No rows remain for split_parts={sorted(split_parts)!r}.")

    row_columns = existing_columns(frame, DEFAULT_ID_COLUMNS + DEFAULT_TARGET_COLUMNS)
    row_path = out_dir / "task_train_space_rows.csv.gz"
    frame.loc[:, row_columns].to_csv(row_path, index=False, encoding="utf-8-sig", compression="gzip")

    chemical_path = out_dir / "task_train_chemical_space.csv"
    chemical_space = summarize_chemical_space(frame, target_column=args.target_column)
    chemical_space.to_csv(chemical_path, index=False, encoding="utf-8-sig")

    species_path = out_dir / "task_train_species_space.csv"
    species_space = summarize_species_space(frame, target_column=args.target_column)
    species_space.to_csv(species_path, index=False, encoding="utf-8-sig")

    embedding_outputs: dict[str, str] = {}
    if args.model_dir:
        embedding_fields = tuple(
            field.strip()
            for field in str(args.embedding_fields).split(",")
            if field.strip()
        )
        embedding_frame, embedding_manifest = build_species_embedding_lookup(
            frame,
            model_dir=Path(args.model_dir),
            embedding_fields=embedding_fields,
        )
        embedding_path = out_dir / "species_embedding_lookup.csv.gz"
        embedding_frame.to_csv(embedding_path, index=False, encoding="utf-8-sig", compression="gzip")
        embedding_outputs["species_embedding_lookup"] = str(embedding_path)
        field_manifest_path = out_dir / "species_embedding_field_manifest.csv"
        pd.DataFrame(embedding_manifest).to_csv(field_manifest_path, index=False, encoding="utf-8-sig")
        embedding_outputs["species_embedding_field_manifest"] = str(field_manifest_path)

        joined = species_space.merge(
            embedding_frame,
            on=existing_columns(species_space, species_identity_columns()),
            how="left",
        )
        joined_path = out_dir / "task_train_species_embedding_space.csv.gz"
        joined.to_csv(joined_path, index=False, encoding="utf-8-sig", compression="gzip")
        embedding_outputs["task_train_species_embedding_space"] = str(joined_path)

    manifest = {
        "db": str(args.db),
        "source_table": args.source_table,
        "split_name": args.split_name,
        "split_parts": sorted(split_parts),
        "rows": int(len(frame)),
        "task_count": int(frame["task_head"].nunique(dropna=True)) if "task_head" in frame.columns else 0,
        "chemical_count": int(frame["smiles"].nunique(dropna=True)) if "smiles" in frame.columns else 0,
        "species_count": int(frame["species_number"].nunique(dropna=True)) if "species_number" in frame.columns else 0,
        "outputs": {
            "task_train_space_rows": str(row_path),
            "task_train_chemical_space": str(chemical_path),
            "task_train_species_space": str(species_path),
            **embedding_outputs,
        },
        "notes": [
            "split_part=train is the aquatic source-domain pretraining set for the current transfer split.",
            "split_part=finetune is the soil target-domain adaptation pool; seed-specific finetune_validation is not separated here.",
            "Embedding vectors are exported from one trained model directory, not from the seed-mean prediction ensemble.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def summarize_chemical_space(frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    group_cols = existing_columns(
        frame,
        (
            "split_name",
            "task_head",
            "task_family",
            "split_part",
            "cas_number",
            "dtxsid",
            "chemical_name",
            "smiles",
        ),
    )
    agg = grouped_base_summary(frame, group_cols, target_column=target_column)
    if "species_number" in frame.columns:
        agg["unique_species_count"] = (
            frame.groupby(group_cols, dropna=False)["species_number"].nunique(dropna=True).to_numpy()
        )
    return agg.sort_values(group_cols).reset_index(drop=True)


def summarize_species_space(frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    group_cols = existing_columns(frame, species_identity_columns(prefix=("split_name", "task_head", "task_family", "split_part")))
    agg = grouped_base_summary(frame, group_cols, target_column=target_column)
    if "smiles" in frame.columns:
        agg["unique_chemical_count"] = frame.groupby(group_cols, dropna=False)["smiles"].nunique(dropna=True).to_numpy()
    return agg.sort_values(group_cols).reset_index(drop=True)


def grouped_base_summary(frame: pd.DataFrame, group_cols: list[str], *, target_column: str) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False, sort=False)
    summary = grouped.size().reset_index(name="n_rows")
    if "medium_domain" in frame.columns:
        summary["medium_domains"] = grouped["medium_domain"].agg(join_unique_text).to_numpy()
    if target_column in frame.columns:
        target = grouped[target_column].agg(["mean", "min", "max"]).reset_index(drop=True)
        summary[f"{target_column}_mean"] = target["mean"].to_numpy()
        summary[f"{target_column}_min"] = target["min"].to_numpy()
        summary[f"{target_column}_max"] = target["max"].to_numpy()
    return summary


def build_species_embedding_lookup(
    frame: pd.DataFrame,
    *,
    model_dir: Path,
    embedding_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import torch

    preprocessing_path = model_dir / "preprocessing.json"
    state_path = model_dir / "best_model.pt"
    if not preprocessing_path.exists():
        raise FileNotFoundError(preprocessing_path)
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    categorical_maps: dict[str, dict[str, int]] = preprocessing.get("categorical_maps", {})
    state = torch.load(state_path, map_location="cpu")

    identity_cols = existing_columns(frame, species_identity_columns())
    profiles = frame.loc[:, identity_cols].drop_duplicates().sort_values(identity_cols).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    field_manifest: list[dict[str, Any]] = []
    offset = 0
    for field in embedding_fields:
        weight = state.get(f"embeddings.{field}.weight")
        if weight is None:
            continue
        width = int(weight.shape[1])
        field_manifest.append(
            {
                "field": field,
                "start_dim": offset,
                "end_dim_exclusive": offset + width,
                "width": width,
                "cardinality": int(weight.shape[0]),
            }
        )
        offset += width

    for _, profile in profiles.iterrows():
        payload = {column: profile.get(column, "") for column in identity_cols}
        vector: list[float] = []
        for item in field_manifest:
            field = str(item["field"])
            weight = state[f"embeddings.{field}.weight"]
            mapping = categorical_maps.get(field, {})
            category_id = encode_category_id(profile.get(field), mapping)
            if category_id >= int(weight.shape[0]):
                category_id = int(mapping.get("<unknown>", 0))
            values = weight[int(category_id)].detach().cpu().numpy().astype(float).tolist()
            vector.extend(values)
            payload[f"{field}_category_id"] = int(category_id)
        for idx, value in enumerate(vector):
            payload[f"emb_{idx:03d}"] = float(value)
        rows.append(payload)
    return pd.DataFrame(rows), field_manifest


def species_identity_columns(prefix: tuple[str, ...] = ()) -> tuple[str, ...]:
    return (
        *prefix,
        "species_number",
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
    )


def existing_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def join_unique_text(values: Any) -> str:
    unique = sorted({str(value).strip() for value in values if str(value).strip() and str(value) != "nan"})
    return ";".join(unique)


if __name__ == "__main__":
    main()
