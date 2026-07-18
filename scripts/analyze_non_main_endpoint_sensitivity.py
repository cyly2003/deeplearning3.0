from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MAIN_TASK_FAMILIES = {"ECx", "LOEC", "NOEC"}
ANALYSIS_VERSION = "20260708"

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / f"non_main_endpoint_sensitivity_{ANALYSIS_VERSION}"

NO_METAL_DB = ROOT / "outputs" / "derived" / "modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite"
NO_METAL_TABLE = "aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic"

REPRESENTATIVE_MANIFEST = (
    ROOT
    / "outputs"
    / "experiments"
    / "v1_2_22_no_metal_random_split_remote"
    / "v1.2.22_no_metal_random5fold_fold3_seed2042_cebin_lw0025_censored_w0p01"
    / "deep"
    / "full"
    / "M_v2_aquatic_to_soil_ptox_no_metal_adapt_E_random_5fold_fold3_f100"
    / "manifest.json"
)


@dataclass(frozen=True)
class PredictionScenario:
    label: str
    split_policy: str
    source_result: str
    paths: tuple[Path, ...]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_json(REPRESENTATIVE_MANIFEST)
    trained = build_trained_endpoint_table(manifest)
    source_counts = load_source_counts()
    trained = trained.merge(source_counts, on=["task_family", "task_head"], how="left")
    trained.to_csv(OUT_DIR / "trained_endpoint_list.csv", index=False, encoding="utf-8-sig")

    non_main_trained = trained[~trained["is_main_result_task"]].copy()
    non_main_trained.to_csv(OUT_DIR / "non_main_trained_endpoint_list.csv", index=False, encoding="utf-8-sig")

    skipped = build_skipped_endpoint_table(manifest, source_counts)
    skipped.to_csv(OUT_DIR / "skipped_endpoint_candidates.csv", index=False, encoding="utf-8-sig")

    task_metrics = []
    family_metrics = []
    scope_metrics = []
    for scenario in prediction_scenarios():
        frame = read_prediction_rows(scenario.paths)
        task_metrics.append(add_scenario(metric_by(frame, ["task_family", "task_head"]), scenario))
        family_metrics.append(add_scenario(metric_by(frame, ["task_family"]), scenario))
        frame = frame.copy()
        frame["task_scope"] = frame["task_family"].astype(str).map(
            lambda value: "main_ECx_LOEC_NOEC" if value in MAIN_TASK_FAMILIES else "non_main_toxicity_aux"
        )
        scope_metrics.append(add_scenario(metric_by(frame, ["task_scope"]), scenario))

    task_metrics_frame = pd.concat(task_metrics, ignore_index=True)
    family_metrics_frame = pd.concat(family_metrics, ignore_index=True)
    scope_metrics_frame = pd.concat(scope_metrics, ignore_index=True)
    task_metrics_frame.to_csv(OUT_DIR / "prediction_metrics_by_task.csv", index=False, encoding="utf-8-sig")
    family_metrics_frame.to_csv(OUT_DIR / "prediction_metrics_by_family.csv", index=False, encoding="utf-8-sig")
    scope_metrics_frame.to_csv(OUT_DIR / "prediction_metrics_main_vs_non_main.csv", index=False, encoding="utf-8-sig")

    evidence = build_evidence_summary(trained, non_main_trained, scope_metrics_frame)
    evidence.to_csv(OUT_DIR / "multitask_auxiliary_evidence_summary.csv", index=False, encoding="utf-8-sig")

    write_readme(
        trained=trained,
        non_main_trained=non_main_trained,
        skipped=skipped,
        scope_metrics=scope_metrics_frame,
        evidence=evidence,
    )

    print(json.dumps({"out_dir": str(OUT_DIR), "files": sorted(p.name for p in OUT_DIR.iterdir())}, ensure_ascii=False, indent=2))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_trained_endpoint_table(manifest: dict) -> pd.DataFrame:
    stats = manifest.get("target_standardization", {}).get("stats", {})
    task_weights = manifest.get("task_weights", {})
    task_heads = list(manifest.get("task_heads", []))
    rows = []
    for task_head in task_heads:
        task_family = task_head.split("_", 1)[0]
        stat_key = f"{task_head}|aquatic_pTox_mol_L"
        fit_count = float(stats.get(stat_key, {}).get("count", 0.0) or 0.0)
        base_group = "main_toxicity" if task_family in MAIN_TASK_FAMILIES else "toxicity_aux"
        weight = float(task_weights.get(task_head, 1.0))
        rows.append(
            {
                "task_family": task_family,
                "task_head": task_head,
                "task_group": base_group,
                "is_main_result_task": task_family in MAIN_TASK_FAMILIES,
                "manifest_fit_count_train_plus_finetune": int(round(fit_count)),
                "task_weight_after_group_and_balance": weight,
                "approx_weighted_fit_mass": fit_count * weight,
            }
        )
    frame = pd.DataFrame(rows)
    total_fit = frame["manifest_fit_count_train_plus_finetune"].sum()
    total_weighted = frame["approx_weighted_fit_mass"].sum()
    frame["fit_count_fraction"] = frame["manifest_fit_count_train_plus_finetune"] / max(total_fit, 1)
    frame["weighted_fit_mass_fraction"] = frame["approx_weighted_fit_mass"] / max(total_weighted, 1e-12)
    return frame.sort_values(["is_main_result_task", "task_family", "task_head"]).reset_index(drop=True)


def load_source_counts() -> pd.DataFrame:
    query = f"""
        SELECT
            task_family,
            task_head,
            COUNT(*) AS source_rows,
            SUM(CASE WHEN medium_domain = 'aquatic' THEN 1 ELSE 0 END) AS source_rows_aquatic,
            SUM(CASE WHEN medium_domain = 'soil' THEN 1 ELSE 0 END) AS source_rows_soil,
            COUNT(DISTINCT COALESCE(cas_number, dtxsid, chemical_name, smiles)) AS source_chemical_count,
            COUNT(DISTINCT latin_name) AS source_species_count
        FROM {NO_METAL_TABLE}
        GROUP BY task_family, task_head
    """
    with sqlite3.connect(NO_METAL_DB) as con:
        return pd.read_sql_query(query, con)


def build_skipped_endpoint_table(manifest: dict, source_counts: pd.DataFrame) -> pd.DataFrame:
    skipped_tasks = manifest.get("skipped_tasks", {})
    rows = []
    for task_head, reason in sorted(skipped_tasks.items()):
        task_family = task_head.split("_", 1)[0]
        rows.append(
            {
                "task_family": task_family,
                "task_head": task_head,
                "task_group": "main_toxicity" if task_family in MAIN_TASK_FAMILIES else "toxicity_aux",
                "is_main_result_task": task_family in MAIN_TASK_FAMILIES,
                "skipped_reason": reason,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.merge(source_counts, on=["task_family", "task_head"], how="left").sort_values(
        ["is_main_result_task", "task_family", "task_head"]
    )


def prediction_scenarios() -> tuple[PredictionScenario, ...]:
    v122 = ROOT / "outputs" / "experiments" / "v1_2_22_no_metal_random_split_remote_summary"
    v124 = ROOT / "outputs" / "experiments" / "v1_2_24_scaffold_cluster_holdout_mainline_remote_summary"
    return (
        PredictionScenario(
            label="v1.2.22_no_metal_random_8_2_5seed",
            split_policy="random_8_2",
            source_result="v1.2.22",
            paths=(v122 / "split_policy_ensemble_random_8_2_holdout_prediction_rows.csv",),
        ),
        PredictionScenario(
            label="v1.2.22_no_metal_random_5fold_5seed",
            split_policy="random_5fold",
            source_result="v1.2.22",
            paths=tuple(v122 / f"split_policy_ensemble_random_5fold_fold{i}_prediction_rows.csv" for i in range(1, 6)),
        ),
        PredictionScenario(
            label="v1.2.24_scaffold_cluster_8_2_5seed",
            split_policy="scaffold_cluster_8_2",
            source_result="v1.2.24",
            paths=(v124 / "split_policy_ensemble_scaffold_cluster_8_2_holdout_prediction_rows.csv",),
        ),
        PredictionScenario(
            label="v1.2.24_scaffold_cluster_5fold_5seed",
            split_policy="scaffold_cluster_5fold",
            source_result="v1.2.24",
            paths=tuple(v124 / f"split_policy_ensemble_scaffold_cluster_5fold_fold{i}_prediction_rows.csv" for i in range(1, 6)),
        ),
    )


def read_prediction_rows(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    usecols = ["task_family", "task_head", "task_group", "y_true", "y_pred", "abs_error"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, usecols=usecols))
    frame = pd.concat(frames, ignore_index=True)
    frame["task_family"] = frame["task_family"].astype(str)
    frame["task_head"] = frame["task_head"].astype(str)
    return frame


def metric_by(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        y_true = group["y_true"].astype(float).to_numpy()
        y_pred = group["y_pred"].astype(float).to_numpy()
        residual = y_true - y_pred
        abs_error = abs(residual)
        n = len(group)
        sse = float((residual**2).sum())
        centered = y_true - float(y_true.mean()) if n else y_true
        sst = float((centered**2).sum())
        r2 = math.nan if n < 2 or sst <= 0 else 1.0 - sse / sst
        rmse = math.sqrt(sse / n) if n else math.nan
        mae = float(abs_error.mean()) if n else math.nan
        huber = float(huber_loss(residual).mean()) if n else math.nan
        row = {column: value for column, value in zip(columns, key)}
        row.update({"n": n, "r2": r2, "rmse": rmse, "mae": mae, "huber_loss": huber})
        rows.append(row)
    return pd.DataFrame(rows)


def huber_loss(residual) -> pd.Series:
    values = pd.Series(residual, dtype="float64").abs()
    return values.where(values > 1.0, 0.5 * values**2).where(values <= 1.0, values - 0.5)


def add_scenario(frame: pd.DataFrame, scenario: PredictionScenario) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "source_result", scenario.source_result)
    frame.insert(1, "evaluation_scenario", scenario.label)
    frame.insert(2, "split_policy", scenario.split_policy)
    return frame


def build_evidence_summary(
    trained: pd.DataFrame,
    non_main_trained: pd.DataFrame,
    scope_metrics: pd.DataFrame,
) -> pd.DataFrame:
    total_tasks = len(trained)
    non_main_tasks = len(non_main_trained)
    total_fit = int(trained["manifest_fit_count_train_plus_finetune"].sum())
    non_main_fit = int(non_main_trained["manifest_fit_count_train_plus_finetune"].sum())
    total_weighted = float(trained["approx_weighted_fit_mass"].sum())
    non_main_weighted = float(non_main_trained["approx_weighted_fit_mass"].sum())
    rows = [
        {
            "evidence_type": "training_task_coverage",
            "metric": "non_main_trained_task_heads",
            "value": non_main_tasks,
            "denominator": total_tasks,
            "fraction": non_main_tasks / max(total_tasks, 1),
            "interpretation": "Non-main toxicity heads present in the shared multitask model.",
        },
        {
            "evidence_type": "training_sample_coverage",
            "metric": "non_main_fit_rows_train_plus_finetune",
            "value": non_main_fit,
            "denominator": total_fit,
            "fraction": non_main_fit / max(total_fit, 1),
            "interpretation": "Approximate rows used to fit target standardization for trained non-main heads in the representative run.",
        },
        {
            "evidence_type": "training_loss_mass_proxy",
            "metric": "non_main_weighted_fit_mass",
            "value": non_main_weighted,
            "denominator": total_weighted,
            "fraction": non_main_weighted / max(total_weighted, 1e-12),
            "interpretation": "Approximate task-weighted fit mass; this is a proxy, not a causal ablation effect.",
        },
    ]
    for scenario, group in scope_metrics.groupby("evaluation_scenario"):
        total_n = int(group["n"].sum())
        for _, row in group.iterrows():
            rows.append(
                {
                    "evidence_type": "prediction_scope_metrics",
                    "metric": f"{scenario}:{row['task_scope']}",
                    "value": float(row["mae"]),
                    "denominator": total_n,
                    "fraction": int(row["n"]) / max(total_n, 1),
                    "interpretation": f"MAE for {row['task_scope']} rows under the trained multitask full model; not a no-auxiliary retrain delta.",
                }
            )
    rows.append(
        {
            "evidence_type": "causal_gain",
            "metric": "no_auxiliary_retrain_delta",
            "value": math.nan,
            "denominator": math.nan,
            "fraction": math.nan,
            "interpretation": "No completed no_auxiliary_tasks or single_task_heads retraining summary was found; direct causal gain is not yet quantified.",
        }
    )
    return pd.DataFrame(rows)


def write_readme(
    *,
    trained: pd.DataFrame,
    non_main_trained: pd.DataFrame,
    skipped: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    evidence: pd.DataFrame,
) -> None:
    non_main_names = ", ".join(non_main_trained["task_head"].tolist())
    current = scope_metrics[
        scope_metrics["evaluation_scenario"].isin(
            [
                "v1.2.22_no_metal_random_8_2_5seed",
                "v1.2.24_scaffold_cluster_8_2_5seed",
            ]
        )
    ][["evaluation_scenario", "task_scope", "n", "r2", "rmse", "mae", "huber_loss"]]
    lines = [
        "# Non-main endpoint sensitivity analysis",
        "",
        f"Analysis version: `{ANALYSIS_VERSION}`.",
        "",
        "## Scope",
        "",
        "- Main result task families are `ECx`, `LOEC`, and `NOEC`.",
        "- Non-main trained toxicity endpoints are task families outside that set but present in the trained multitask heads.",
        f"- Non-main trained task heads in the representative current no-metal run: {non_main_names}.",
        "- This analysis reads existing artifacts only and does not retrain any model.",
        "",
        "## Key files",
        "",
        "- `non_main_trained_endpoint_list.csv`: endpoints requested by the user.",
        "- `trained_endpoint_list.csv`: all trained task heads and approximate training support.",
        "- `skipped_endpoint_candidates.csv`: non-main endpoint candidates present in data but skipped by task filters.",
        "- `prediction_metrics_by_task.csv`: task-level performance from ensemble prediction rows.",
        "- `prediction_metrics_by_family.csv`: endpoint-family performance from ensemble prediction rows.",
        "- `prediction_metrics_main_vs_non_main.csv`: main vs non-main comparison.",
        "- `multitask_auxiliary_evidence_summary.csv`: what can and cannot be claimed as a multitask gain.",
        "",
        "## Current direct limitation",
        "",
        "No completed `no_auxiliary_tasks` or `single_task_heads` retraining result was found in the current summary set. Therefore, this bundle reports sensitivity and support under the full multitask model, but does not claim a causal performance gain from auxiliary endpoints.",
        "",
        "## Main vs non-main snapshot",
        "",
        markdown_table(current),
        "",
        "## Training support snapshot",
        "",
        markdown_table(
            non_main_trained[
                [
                    "task_family",
                    "task_head",
                    "task_group",
                    "manifest_fit_count_train_plus_finetune",
                    "task_weight_after_group_and_balance",
                    "source_rows",
                    "source_rows_soil",
                ]
            ]
        ),
        "",
        "## Counts",
        "",
        f"- Trained task heads: {len(trained)}.",
        f"- Non-main trained task heads: {len(non_main_trained)}.",
        f"- Skipped endpoint candidates: {len(skipped)}.",
        f"- Evidence rows: {len(evidence)}.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([format_cell(value) for value in row])
    columns = list(frame.columns)
    widths = [
        max(len(str(columns[idx])), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(str(columns[idx]).ljust(widths[idx]) for idx in range(len(columns))) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def format_cell(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
