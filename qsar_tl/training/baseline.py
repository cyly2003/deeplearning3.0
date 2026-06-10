from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from qsar_tl.evaluation.reporting import regression_report_rows, write_regression_report
from qsar_tl.evaluation.splits import resolve_id_column, resolve_source_table, table_columns, table_exists


TARGET_COLUMN_CANDIDATES = ("target_value_median", "target_value", "tox_value")
TASK_HEAD_CANDIDATES = ("task_head", "target_name", "endpoint")
MODEL_NAMES = ("random_forest", "hist_gradient_boosting")

EXCLUDED_FEATURE_COLUMNS = {
    "aggregate_id",
    "record_id",
    "result_id",
    "test_id",
    "reference_number",
    "target_value",
    "target_value_median",
    "target_value_mean",
    "target_value_std",
    "target_value_count",
    "target_value_min",
    "target_value_max",
    "tox_value",
    "tox_value_source",
    "tox_value_imputed",
    "target_name",
    "task_head",
    "target_basis",
    "target_status",
    "excluded_reason",
    "split_name",
    "split_part",
    "split_type",
    "seed",
    "created_at",
    "conc1_type",
    "conc1_mean_op",
    "conc1_mean_standardized",
    "conc1_min_op",
    "conc1_min_standardized",
    "conc1_max_op",
    "conc1_max_standardized",
    "conc1_standard_unit",
    "conc1_unit_family",
    "conc1_standardization_status",
}

CATEGORICAL_HINT_COLUMNS = {
    "cas_number",
    "species_number",
    "chemical_name",
    "dtxsid",
    "smiles",
    "chemical_class_l1",
    "chemical_class_l2",
    "chemical_class_l3",
    "latin_name",
    "common_name",
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
    "organism_lifestage",
    "media_type",
    "endpoint",
    "effect",
    "measurement",
    "trend",
}


@dataclass(frozen=True)
class BaselineResult:
    report_path: Path
    metrics: list[dict[str, Any]]
    prediction_count: int
    task_heads: list[str]
    feature_count: int


@dataclass
class TabularPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    onehot_categories: dict[str, list[str]]
    hash_columns: list[str]
    numeric_medians: dict[str, float]
    hash_bins: int = 16

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        max_onehot_categories: int = 30,
        hash_bins: int = 16,
    ) -> "TabularPreprocessor":
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        numeric_medians: dict[str, float] = {}

        for column in frame.columns:
            if column in CATEGORICAL_HINT_COLUMNS:
                categorical_columns.append(column)
                continue
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().any():
                numeric_columns.append(column)
                median = converted.median()
                numeric_medians[column] = 0.0 if pd.isna(median) else float(median)
            else:
                categorical_columns.append(column)

        onehot_categories: dict[str, list[str]] = {}
        hash_columns: list[str] = []
        for column in categorical_columns:
            values = normalized_category_series(frame[column])
            categories = sorted(values.dropna().unique().tolist())
            if len(categories) <= max_onehot_categories:
                onehot_categories[column] = categories
            else:
                hash_columns.append(column)

        return cls(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            onehot_categories=onehot_categories,
            hash_columns=hash_columns,
            numeric_medians=numeric_medians,
            hash_bins=hash_bins,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        features: dict[str, Any] = {}
        for column in self.numeric_columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            features[column] = converted.fillna(self.numeric_medians[column]).astype(float)

        for column, categories in self.onehot_categories.items():
            values = normalized_category_series(frame[column])
            for category in categories:
                features[f"{column}={category}"] = (values == category).astype(float)

        for column in self.hash_columns:
            values = normalized_category_series(frame[column])
            hashed = np.zeros((len(frame), self.hash_bins), dtype=float)
            for row_idx, value in enumerate(values):
                if value == "<missing>":
                    continue
                bin_idx = stable_hash_bin(f"{column}={value}", self.hash_bins)
                hashed[row_idx, bin_idx] = 1.0
            for bin_idx in range(self.hash_bins):
                features[f"{column}#hash{bin_idx}"] = hashed[:, bin_idx]

        if not features:
            return pd.DataFrame({"bias": np.ones(len(frame), dtype=float)}, index=frame.index)
        return pd.DataFrame(features, index=frame.index)


def normalized_category_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().replace("", "<missing>")


def stable_hash_bin(value: str, bins: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bins


def run_baseline(
    db_path: str | Path,
    *,
    split_name: str,
    out_path: str | Path,
    model_name: str = "random_forest",
    limit: int | None = None,
    source_table: str | None = None,
    seed: int = 42,
) -> BaselineResult:
    model_name = normalize_model_name(model_name)
    frame = load_split_frame(
        db_path,
        split_name=split_name,
        source_table=source_table,
        limit=limit,
    )
    target_column = resolve_target_column(frame)
    task_column = resolve_task_column(frame)
    feature_columns = select_feature_columns(frame, target_column=target_column, task_column=task_column)

    predictions: list[dict[str, Any]] = []
    max_feature_count = 0
    task_heads: list[str] = []
    for task_head, task_frame in frame.groupby(task_column, dropna=False):
        task_label = "default" if pd.isna(task_head) else str(task_head)
        train_frame = task_frame[task_frame["split_part"] == "train"].copy()
        eval_frame = task_frame[task_frame["split_part"].isin(["train", "valid", "test"])].copy()
        if train_frame.empty or eval_frame.empty:
            continue

        y_train = pd.to_numeric(train_frame[target_column], errors="coerce")
        train_mask = y_train.notna()
        train_frame = train_frame.loc[train_mask]
        y_train = y_train.loc[train_mask].astype(float)
        if train_frame.empty:
            continue

        eval_y = pd.to_numeric(eval_frame[target_column], errors="coerce")
        eval_frame = eval_frame.loc[eval_y.notna()]
        eval_y = eval_y.loc[eval_y.notna()].astype(float)
        if eval_frame.empty:
            continue

        preprocessor = TabularPreprocessor.fit(train_frame[feature_columns])
        x_train = preprocessor.transform(train_frame[feature_columns])
        x_eval = preprocessor.transform(eval_frame[feature_columns])
        max_feature_count = max(max_feature_count, x_train.shape[1])

        model = build_model(model_name, seed=seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_eval)
        task_heads.append(task_label)

        for idx, (_, row) in enumerate(eval_frame.iterrows()):
            predictions.append(
                {
                    "split_name": split_name,
                    "split_part": row["split_part"],
                    "task_head": task_label,
                    "y_true": float(eval_y.iloc[idx]),
                    "y_pred": float(y_pred[idx]),
                }
            )

    if not predictions:
        raise ValueError("No predictions were produced. Check split assignments, target values, and train rows.")

    metrics = regression_report_rows(predictions)
    report_path = write_regression_report(metrics, out_path)
    return BaselineResult(
        report_path=report_path,
        metrics=metrics,
        prediction_count=len(predictions),
        task_heads=sorted(set(task_heads)),
        feature_count=max_feature_count,
    )


def load_split_frame(
    db_path: str | Path,
    *,
    split_name: str,
    source_table: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        if not table_exists(conn, "split_assignments"):
            raise ValueError("split_assignments table does not exist. Generate a split before running baseline.")
        resolved_source = resolve_split_source_table(conn, split_name, source_table)
        columns = table_columns(conn, resolved_source)
        id_column = resolve_id_column(columns)
        split_id_column = "aggregate_id" if id_column == "aggregate_id" else "record_id"
        limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
        query = f"""
            SELECT r.*, s.split_name, s.split_part
            FROM "{resolved_source}" AS r
            JOIN split_assignments AS s
              ON s.split_name = ?
             AND s.source_table = ?
             AND s.{split_id_column} = CAST(r."{id_column}" AS TEXT)
            WHERE s.split_part IN ('train', 'valid', 'test')
            ORDER BY r."{id_column}"
            {limit_clause}
        """
        frame = pd.read_sql_query(query, conn, params=(split_name, resolved_source))

    if frame.empty:
        raise ValueError(f"No rows found for split_name={split_name!r}.")
    if "target_status" in frame.columns:
        frame = frame[(frame["target_status"].isna()) | (frame["target_status"] == "included")]
    return frame


def resolve_split_source_table(
    conn: sqlite3.Connection,
    split_name: str,
    source_table: str | None,
) -> str:
    if source_table:
        return resolve_source_table(conn, source_table)
    row = conn.execute(
        """
        SELECT source_table
        FROM split_assignments
        WHERE split_name = ?
        GROUP BY source_table
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """,
        (split_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No split assignments found for split_name={split_name!r}.")
    return resolve_source_table(conn, row[0])


def resolve_target_column(frame: pd.DataFrame) -> str:
    for column in TARGET_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(f"No target column found. Expected one of: {', '.join(TARGET_COLUMN_CANDIDATES)}")


def resolve_task_column(frame: pd.DataFrame) -> str:
    for column in TASK_HEAD_CANDIDATES:
        if column in frame.columns:
            return column
    frame["task_head"] = "default"
    return "task_head"


def select_feature_columns(frame: pd.DataFrame, *, target_column: str, task_column: str) -> list[str]:
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    excluded.add(target_column)
    excluded.add(task_column)
    return [column for column in frame.columns if column not in excluded]


def build_model(model_name: str, *, seed: int) -> Any:
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=100,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower().replace("-", "_")
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest_regressor": "random_forest",
        "hgb": "hist_gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
        "hist_gradient_boosting_regressor": "hist_gradient_boosting",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from: {', '.join(MODEL_NAMES)}")
    return normalized
