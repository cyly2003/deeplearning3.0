from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qsar_tl.evaluation.reporting import regression_report_rows, write_regression_report
from qsar_tl.evaluation.splits import resolve_id_column, resolve_source_table, table_columns, table_exists


TARGET_COLUMN_CANDIDATES = ("target_value_median", "target_value", "tox_value")
TASK_HEAD_CANDIDATES = ("task_head", "target_name", "endpoint")
MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "pls",
    "extra_trees",
    "elastic_net",
    "mlp",
    "hist_gradient_boosting",
)
EVAL_SPLIT_PARTS = ("train", "finetune", "valid", "test")
DURATION_COLUMN_CANDIDATES = (
    "duration_bin_h",
    "exposure_duration_mean_h",
    "duration_h",
)
TARGET_DIMENSION_COLUMNS = ("target_family", "target_name")

EXCLUDED_FEATURE_COLUMNS = {
    "aggregate_id",
    "record_id",
    "result_id",
    "test_id",
    "reference_number",
    "publication_year",
    "reference_numbers",
    "publication_years",
    "test_ids",
    "result_ids",
    "species_number",
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
    "value_quality",
    "target_name",
    "target_family",
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
    "conc1_mean",
    "conc1_mean_standardized",
    "conc1_min_op",
    "conc1_min",
    "conc1_min_standardized",
    "conc1_max_op",
    "conc1_max",
    "conc1_max_standardized",
    "conc1_unit",
    "conc1_standard_unit",
    "conc1_unit_family",
    "conc1_standardization_status",
    "unit_family_v2",
    "standard_unit_v2",
    "standard_value_mg_l",
    "standard_value_mol_l",
    "standard_value_mg_kg",
    "standard_value_g_ha",
    "standard_value_mg_kg_diet",
    "standard_value_mg_kg_bw_day",
    "unit_conversion_source",
    "unit_conversion_confidence",
    "unit_conversion_note",
    "conversion_path",
    "active_ingredient_basis",
    "acid_equivalent_basis",
    "medium_domains",
    "medium_domain_reason",
    "medium_conflict_flag",
}

CATEGORICAL_HINT_COLUMNS = {
    "cas_number",
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
    skipped_tasks: dict[str, str] = field(default_factory=dict)


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
            features[safe_feature_name("num", column)] = converted.fillna(self.numeric_medians[column]).astype(float)

        for column, categories in self.onehot_categories.items():
            values = normalized_category_series(frame[column])
            for category in categories:
                features[safe_feature_name("cat", column, category)] = (values == category).astype(float)

        for column in self.hash_columns:
            values = normalized_category_series(frame[column])
            hashed = np.zeros((len(frame), self.hash_bins), dtype=float)
            for row_idx, value in enumerate(values):
                if value == "<missing>":
                    continue
                bin_idx = stable_hash_bin(f"{column}={value}", self.hash_bins)
                hashed[row_idx, bin_idx] = 1.0
            for bin_idx in range(self.hash_bins):
                features[safe_feature_name("hash", column, str(bin_idx))] = hashed[:, bin_idx]

        if not features:
            return pd.DataFrame({"bias": np.ones(len(frame), dtype=float)}, index=frame.index)
        return pd.DataFrame(features, index=frame.index)


def normalized_category_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().replace("", "<missing>")


def stable_hash_bin(value: str, bins: int) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % bins


def safe_feature_name(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    readable = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")[:48]
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    if readable:
        return f"{prefix}_{readable}_{digest}"
    return f"{prefix}_{digest}"


def run_baseline(
    db_path: str | Path,
    *,
    split_name: str,
    out_path: str | Path,
    model_name: str = "random_forest",
    limit: int | None = None,
    source_table: str | None = None,
    seed: int = 42,
    huber_delta: float = 1.0,
    min_total: int = 0,
    min_train: int = 1,
    min_eval: int = 1,
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
    frame = add_duration_nonlinear_features(frame)
    feature_columns = select_feature_columns(frame, target_column=target_column, task_column=task_column)

    predictions: list[dict[str, Any]] = []
    max_feature_count = 0
    task_heads: list[str] = []
    skipped_tasks: dict[str, str] = {}
    for task_head, task_frame in frame.groupby(task_column, dropna=False):
        task_label = "default" if pd.isna(task_head) else str(task_head)
        skip_reason = task_skip_reason(
            task_frame,
            min_total=min_total,
            min_train=min_train,
            min_eval=min_eval,
        )
        if skip_reason is not None:
            skipped_tasks[task_label] = skip_reason
            continue
        train_frame = task_frame[task_frame["split_part"] == "train"].copy()
        eval_frame = task_frame[task_frame["split_part"].isin(EVAL_SPLIT_PARTS)].copy()
        if train_frame.empty or eval_frame.empty:
            skipped_tasks[task_label] = "empty_train_or_eval_after_split_filter"
            continue

        y_train = pd.to_numeric(train_frame[target_column], errors="coerce")
        train_mask = y_train.notna()
        train_frame = train_frame.loc[train_mask]
        y_train = y_train.loc[train_mask].astype(float)
        if train_frame.empty:
            skipped_tasks[task_label] = "empty_train_after_target_filter"
            continue

        eval_y = pd.to_numeric(eval_frame[target_column], errors="coerce")
        eval_frame = eval_frame.loc[eval_y.notna()]
        eval_y = eval_y.loc[eval_y.notna()].astype(float)
        if eval_frame.empty:
            skipped_tasks[task_label] = "empty_eval_after_target_filter"
            continue

        preprocessor = TabularPreprocessor.fit(train_frame[feature_columns])
        x_train = preprocessor.transform(train_frame[feature_columns])
        x_eval = preprocessor.transform(eval_frame[feature_columns])
        max_feature_count = max(max_feature_count, x_train.shape[1])

        model = build_model(model_name, seed=seed, n_features=x_train.shape[1])
        model.fit(x_train, y_train)
        y_pred = model.predict(x_eval)
        y_pred = np.asarray(y_pred).reshape(-1)
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

    metrics = regression_report_rows(predictions, huber_delta=huber_delta)
    report_path = write_regression_report(metrics, out_path)
    return BaselineResult(
        report_path=report_path,
        metrics=metrics,
        prediction_count=len(predictions),
        task_heads=sorted(set(task_heads)),
        feature_count=max_feature_count,
        skipped_tasks=skipped_tasks,
    )


def task_skip_reason(
    task_frame: pd.DataFrame,
    *,
    min_total: int,
    min_train: int,
    min_eval: int,
) -> str | None:
    total = len(task_frame)
    train = int((task_frame["split_part"] == "train").sum())
    eval_count = int(task_frame["split_part"].isin([part for part in EVAL_SPLIT_PARTS if part != "train"]).sum())
    if total < min_total:
        return f"total_samples_lt_{min_total}"
    if train < min_train:
        return f"train_samples_lt_{min_train}"
    if eval_count < min_eval:
        return f"eval_samples_lt_{min_eval}"
    return None


def add_duration_nonlinear_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    duration_column = next((column for column in DURATION_COLUMN_CANDIDATES if column in result.columns), None)
    if duration_column is None:
        return result
    duration = pd.to_numeric(result[duration_column], errors="coerce").clip(lower=0)
    result["duration_log1p_h"] = np.log1p(duration)
    result["duration_sqrt_h"] = np.sqrt(duration)
    result["duration_inv_log1p_h"] = 1.0 / (1.0 + np.log1p(duration))
    centers = np.array([24.0, 48.0, 96.0, 168.0, 336.0, 720.0])
    log_duration = np.log1p(duration.to_numpy(dtype=float))
    log_centers = np.log1p(centers)
    gamma = 0.35
    for center, log_center in zip(centers, log_centers):
        values = np.exp(-gamma * np.square(log_duration - log_center))
        result[f"duration_rbf_{int(center)}h"] = np.nan_to_num(values, nan=0.0)
    return result


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
        assignment_limit = "" if limit is None else f" LIMIT {int(limit)}"
        assignments = pd.read_sql_query(
            f"""
            SELECT
                {split_id_column} AS source_id,
                split_name,
                split_part,
                split_type AS _assignment_split_type,
                group_key AS _assignment_group_key
            FROM split_assignments
            WHERE split_name = ?
              AND source_table = ?
              AND split_part IN ({','.join('?' for _ in EVAL_SPLIT_PARTS)})
            ORDER BY rowid
            {assignment_limit}
            """,
            conn,
            params=(split_name, resolved_source, *EVAL_SPLIT_PARTS),
        )
        if assignments.empty:
            raise ValueError(f"No split assignments found for split_name={split_name!r}.")
        assignments["source_key"] = assignments["source_id"].astype(str)
        frame = load_source_rows_by_ids(
            conn,
            resolved_source,
            id_column,
            assignments["source_id"].tolist(),
        )
        frame["source_key"] = frame[id_column].astype(str)
        frame = frame.merge(
            assignments[
                [
                    "source_key",
                    "split_name",
                    "split_part",
                    "_assignment_split_type",
                    "_assignment_group_key",
                ]
            ],
            on="source_key",
            how="inner",
        )
        frame, split_join_audit = enforce_split_medium_contract(
            frame,
            split_name=split_name,
            source_table=resolved_source,
        )
        frame = frame.drop(
            columns=[
                column
                for column in ("source_key", "_assignment_split_type", "_assignment_group_key")
                if column in frame.columns
            ]
        )
        frame.attrs["split_join_audit"] = split_join_audit

    if frame.empty:
        raise ValueError(f"No rows found for split_name={split_name!r}.")
    if "target_status" in frame.columns:
        frame = frame[(frame["target_status"].isna()) | (frame["target_status"] == "included")]
    validate_single_target_dimension(frame, split_name=split_name, source_table=resolved_source)
    return frame


def enforce_split_medium_contract(
    frame: pd.DataFrame,
    *,
    split_name: str,
    source_table: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit: dict[str, Any] = {
        "split_name": split_name,
        "source_table": source_table,
        "input_rows": int(len(frame)),
        "removed_rows_count": 0,
        "removed_rows": [],
    }
    required_columns = {"medium_domain", "_assignment_group_key", "split_part"}
    if frame.empty or not required_columns.issubset(frame.columns):
        audit["output_rows"] = int(len(frame))
        return frame, audit

    expected = frame["_assignment_group_key"].map(expected_medium_from_group_key)
    observed = frame["medium_domain"].astype("string").str.strip().str.lower()
    checked = expected.notna()
    mismatch = checked & observed.ne(expected)
    audit["checked_rows"] = int(checked.sum())
    if not bool(mismatch.any()):
        audit["output_rows"] = int(len(frame))
        return frame, audit

    removed = (
        frame.loc[mismatch]
        .assign(_expected_medium_domain=expected[mismatch], _observed_medium_domain=observed[mismatch])
        .groupby(["split_part", "_assignment_group_key", "_expected_medium_domain", "_observed_medium_domain"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    audit["removed_rows_count"] = int(mismatch.sum())
    audit["removed_rows"] = [
        {
            "split_part": clean_audit_value(row["split_part"]),
            "assignment_group_key": clean_audit_value(row["_assignment_group_key"]),
            "expected_medium_domain": clean_audit_value(row["_expected_medium_domain"]),
            "observed_medium_domain": clean_audit_value(row["_observed_medium_domain"]),
            "n": int(row["n"]),
        }
        for _, row in removed.iterrows()
    ]
    filtered = frame.loc[~mismatch].copy()
    audit["output_rows"] = int(len(filtered))
    return filtered, audit


def expected_medium_from_group_key(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text == "aquatic" or text.startswith("aquatic_"):
        return "aquatic"
    if text == "soil" or text.startswith("soil_"):
        return "soil"
    if text == "sediment" or text.startswith("sediment_"):
        return "sediment"
    return None


def clean_audit_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def validate_single_target_dimension(
    frame: pd.DataFrame,
    *,
    split_name: str | None = None,
    source_table: str | None = None,
) -> None:
    target_dimension_column = next((column for column in TARGET_DIMENSION_COLUMNS if column in frame.columns), None)
    if target_dimension_column is None or frame.empty:
        return
    target_values = frame[target_dimension_column].astype("string").str.strip()
    target_values = target_values[target_values.notna() & (target_values != "")]
    unique_targets = sorted(target_values.unique().tolist())
    if len(unique_targets) <= 1:
        return

    counts = target_values.value_counts(dropna=False).head(8)
    count_text = ", ".join(f"{target}={int(count)}" for target, count in counts.items())
    context = []
    if source_table:
        context.append(f"source_table={source_table!r}")
    if split_name:
        context.append(f"split_name={split_name!r}")
    context_text = " ".join(context)
    raise ValueError(
        "Mixed target dimensions are not allowed for modeling. "
        f"{context_text} contains multiple {target_dimension_column} values: {count_text}. "
        "Use a dimension-specific source table such as *_ptox, *_mg_kg, *_g_ha, *_l_ha, "
        "or another table with one convertible target scale."
    )


def load_source_rows_by_ids(
    conn: sqlite3.Connection,
    table_name: str,
    id_column: str,
    ids: list[Any],
    *,
    chunk_size: int = 900,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        if id_column == "aggregate_id":
            params = [coerce_sqlite_id(value) for value in chunk]
        else:
            params = [str(value) for value in chunk]
        placeholders = ", ".join("?" for _ in params)
        chunk_frame = pd.read_sql_query(
            f'SELECT * FROM "{table_name}" WHERE "{id_column}" IN ({placeholders})',
            conn,
            params=params,
        )
        if not chunk_frame.empty:
            frames.append(chunk_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def coerce_sqlite_id(value: Any) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


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


def build_model(model_name: str, *, seed: int, n_features: int | None = None) -> Any:
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=5000),
        )
    if model_name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_components = max(1, min(10, int(n_features or 1)))
        return make_pipeline(StandardScaler(with_mean=False), PLSRegression(n_components=n_components))
    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                random_state=seed,
            ),
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for model_name='xgboost'.") from exc
        # XGBoost 3.2.0's pseudohuber objective produced a constant prediction
        # around 49 on pTox targets in the remote environment. Keep Huber as
        # the shared evaluation metric and use the stable squared-error
        # objective for this baseline.
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=8,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required for model_name='lightgbm'.") from exc
        return LGBMRegressor(
            objective="huber",
            alpha=1.0,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

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
        "extratrees": "extra_trees",
        "extra_tress": "extra_trees",
        "extratress": "extra_trees",
        "extra_trees_regressor": "extra_trees",
        "plsregression": "pls",
        "partial_least_squares": "pls",
        "elasticnet": "elastic_net",
        "mlpregressor": "mlp",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
        "hgb": "hist_gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
        "hist_gradient_boosting_regressor": "hist_gradient_boosting",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_NAMES:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from: {', '.join(MODEL_NAMES)}")
    return normalized
