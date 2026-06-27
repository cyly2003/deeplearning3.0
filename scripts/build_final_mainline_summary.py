from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MAIN_TASK_FAMILIES = ("ECx", "LOEC", "NOEC")
CHEMICAL_HOLDOUT_RUN = "anchor_tanimoto_a1_5seed_ensemble"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build final comparison tables from existing mainline QSAR summaries."
    )
    parser.add_argument(
        "--chemical-summary-dir",
        default="outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary",
        help="Directory containing v1.2.12 chemical-holdout 5-seed ensemble summaries.",
    )
    parser.add_argument(
        "--random-summary-dir",
        default="outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary",
        help="Directory containing v1.2.15/v1.2.16 random split policy summaries.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/final_mainline_comparison",
        help="Output directory for harmonized comparison tables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    chemical_dir = Path(args.chemical_summary_dir)
    random_dir = Path(args.random_summary_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_summary_tables(chemical_dir=chemical_dir, random_dir=random_dir, out_dir=out_dir)
    print(json.dumps({"out_dir": str(out_dir), "outputs": outputs}, ensure_ascii=False, indent=2))


def build_summary_tables(*, chemical_dir: Path, random_dir: Path, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chemical_overall = load_chemical_overall(chemical_dir)
    random_overall = load_random_overall(random_dir)
    overall = pd.concat([chemical_overall, random_overall], ignore_index=True)
    overall_path = out_dir / "existing_final_overall_summary.csv"
    write_csv(overall, overall_path)

    chemical_family = load_chemical_family(chemical_dir)
    random_family = load_random_family(random_dir)
    family = pd.concat([chemical_family, random_family], ignore_index=True)
    family_path = out_dir / "existing_final_family_summary_30task.csv"
    write_csv(family, family_path)

    chemical_task_all = load_chemical_task(chemical_dir)
    random_task_all = load_random_task(random_dir, main_only=False)
    task_all = pd.concat([chemical_task_all, random_task_all], ignore_index=True)
    task_all_path = out_dir / "existing_final_task_summary_35task.csv"
    write_csv(task_all, task_all_path)

    chemical_task_main = filter_main_tasks(chemical_task_all)
    random_task_main = load_random_task(random_dir, main_only=True)
    task_main = pd.concat([chemical_task_main, random_task_main], ignore_index=True)
    task_main_path = out_dir / "existing_final_task_summary_30task.csv"
    write_csv(task_main, task_main_path)

    single_seed_path = out_dir / "chemical_holdout_single_seed_summary.csv"
    single_seed = load_optional_single_seed_summary()
    if single_seed is not None:
        write_csv(single_seed, single_seed_path)

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        build_readme(
            overall=overall,
            task_main=task_main,
            single_seed_written=single_seed is not None,
            chemical_dir=chemical_dir,
            random_dir=random_dir,
        ),
        encoding="utf-8",
    )

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "chemical_summary_dir": str(chemical_dir),
        "random_summary_dir": str(random_dir),
        "main_task_families": list(MAIN_TASK_FAMILIES),
        "chemical_holdout_run": CHEMICAL_HOLDOUT_RUN,
        "outputs": {
            "overall": str(overall_path),
            "family_30task": str(family_path),
            "task_35task": str(task_all_path),
            "task_30task": str(task_main_path),
            "single_seed": str(single_seed_path) if single_seed is not None else None,
            "readme": str(readme_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest["outputs"] | {"manifest": str(manifest_path)}


def load_chemical_overall(chemical_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(chemical_dir / "seed_mean_ensemble_focus_summary.csv")
    frame = frame[(frame["run"] == CHEMICAL_HOLDOUT_RUN) & (frame["tier"] == "all_test")].copy()
    frame["evaluation_policy"] = "chemical_holdout_f100_5seed"
    frame["split_strategy"] = "fixed_chemical_holdout"
    frame["split_name"] = "M_v2_aquatic_to_soil_ptox_adapt_C_f100"
    frame["folds_combined"] = "fixed_test"
    frame["seeds"] = "42;1042;2042;3042;4042"
    frame["ensemble_seed_count"] = 5
    frame["task_scope"] = "ECx_LOEC_NOEC_30task"
    frame["source_result"] = "v1.2.12_anchor"
    return select_overall_columns(frame)


def load_random_overall(random_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(random_dir / "split_policy_ensemble_combined_summary.csv").copy()
    frame["evaluation_policy"] = frame["split_policy"].map(
        {
            "random_8_2": "random_8_2_3seed",
            "random_5fold": "random_5fold_3seed",
        }
    ).fillna(frame["split_policy"])
    frame["split_strategy"] = frame["split_policy"]
    frame["split_name"] = frame["split_policy"].map(
        {
            "random_8_2": "M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100",
            "random_5fold": "M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold1-5_f100",
        }
    ).fillna("")
    frame["ensemble_seed_count"] = frame["seeds"].astype(str).apply(lambda value: len([x for x in value.split(";") if x]))
    frame["task_scope"] = "ECx_LOEC_NOEC_30task"
    frame["source_result"] = "v1.2.16_random_split_3seed"
    return select_overall_columns(frame)


def load_chemical_family(chemical_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(chemical_dir / "seed_mean_ensemble_family_summary.csv")
    frame = frame[frame["run"] == CHEMICAL_HOLDOUT_RUN].copy()
    frame["evaluation_policy"] = "chemical_holdout_f100_5seed"
    frame["split_strategy"] = "fixed_chemical_holdout"
    frame["folds_combined"] = "fixed_test"
    frame["seeds"] = "42;1042;2042;3042;4042"
    frame["ensemble_seed_count"] = 5
    frame["source_result"] = "v1.2.12_anchor"
    return select_family_columns(frame)


def load_random_family(random_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(random_dir / "split_policy_ensemble_family_summary.csv").copy()
    frame["evaluation_policy"] = frame["split_policy"].map(
        {
            "random_8_2": "random_8_2_3seed",
            "random_5fold": "random_5fold_3seed",
        }
    ).fillna(frame["split_policy"])
    frame["split_strategy"] = frame["split_policy"]
    frame["ensemble_seed_count"] = frame["seeds"].astype(str).apply(lambda value: len([x for x in value.split(";") if x]))
    frame["source_result"] = "v1.2.16_random_split_3seed"
    return select_family_columns(frame)


def load_chemical_task(chemical_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(chemical_dir / "seed_mean_ensemble_task_summary.csv")
    frame = frame[frame["run"] == CHEMICAL_HOLDOUT_RUN].copy()
    frame["evaluation_policy"] = "chemical_holdout_f100_5seed"
    frame["split_strategy"] = "fixed_chemical_holdout"
    frame["folds_combined"] = "fixed_test"
    frame["seeds"] = "42;1042;2042;3042;4042"
    frame["ensemble_seed_count"] = 5
    frame["source_result"] = "v1.2.12_anchor"
    return select_task_columns(frame)


def load_random_task(random_dir: Path, *, main_only: bool) -> pd.DataFrame:
    filename = "split_policy_ensemble_main_task_summary.csv" if main_only else "split_policy_ensemble_task_summary.csv"
    frame = pd.read_csv(random_dir / filename).copy()
    frame["evaluation_policy"] = frame["split_policy"].map(
        {
            "random_8_2": "random_8_2_3seed",
            "random_5fold": "random_5fold_3seed",
        }
    ).fillna(frame["split_policy"])
    frame["split_strategy"] = frame["split_policy"]
    frame["ensemble_seed_count"] = frame["seeds"].astype(str).apply(lambda value: len([x for x in value.split(";") if x]))
    frame["source_result"] = "v1.2.16_random_split_3seed"
    return select_task_columns(frame)


def load_optional_single_seed_summary() -> pd.DataFrame | None:
    path = Path("outputs/experiments/v1_2_11_f100_seed_stability_remote_summary/focus_summary.csv")
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    mask = frame["run"].astype(str).str.startswith("transfer_f100_anchor_tanimoto_a1_seed")
    mask &= frame["prediction_split_part"].astype(str).eq("test")
    frame = frame[mask].copy()
    if frame.empty:
        return None
    frame["evaluation_policy"] = "chemical_holdout_f100_single_seed"
    frame["split_strategy"] = "fixed_chemical_holdout"
    frame["task_scope"] = "ECx_LOEC_NOEC_30task"
    frame["seed"] = frame["run"].astype(str).str.extract(r"seed(\d+)")[0]
    return frame[
        [
            "evaluation_policy",
            "split_strategy",
            "split_name",
            "seed",
            "prediction_split_part",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
        ]
    ].sort_values(["seed", "prediction_split_part"])


def filter_main_tasks(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["task_family"].astype(str).isin(MAIN_TASK_FAMILIES)].copy()


def select_overall_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "evaluation_policy",
            "split_strategy",
            "split_name",
            "folds_combined",
            "seeds",
            "ensemble_seed_count",
            "task_scope",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
            "source_result",
        ]
    ]


def select_family_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "evaluation_policy",
            "split_strategy",
            "task_family",
            "folds_combined",
            "seeds",
            "ensemble_seed_count",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "task_count",
            "tasks",
            "source_result",
        ]
    ].sort_values(["evaluation_policy", "task_family"])


def select_task_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "evaluation_policy",
            "split_strategy",
            "task_family",
            "task_head",
            "folds_combined",
            "seeds",
            "ensemble_seed_count",
            "n",
            "r2",
            "rmse",
            "mae",
            "huber_loss",
            "source_result",
        ]
    ].sort_values(["evaluation_policy", "task_family", "task_head"])


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def build_readme(
    *,
    overall: pd.DataFrame,
    task_main: pd.DataFrame,
    single_seed_written: bool,
    chemical_dir: Path,
    random_dir: Path,
) -> str:
    lines = [
        "# Final Mainline Comparison Summary",
        "",
        "This directory harmonizes the existing final result tables for the current QSAR transfer mainline.",
        "",
        "## Sources",
        "",
        f"- Chemical-holdout mainline: `{chemical_dir}`.",
        f"- Random split reference: `{random_dir}`.",
        "",
        "## Files",
        "",
        "- `existing_final_overall_summary.csv`: overall ECx/LOEC/NOEC 30-task metrics.",
        "- `existing_final_family_summary_30task.csv`: endpoint-family metrics for ECx, LOEC, and NOEC.",
        "- `existing_final_task_summary_30task.csv`: task-level main-scope metrics.",
        "- `existing_final_task_summary_35task.csv`: task-level full trained-head metrics, including ICx/LDx when present.",
    ]
    if single_seed_written:
        lines.append("- `chemical_holdout_single_seed_summary.csv`: fixed-test single-seed anchor metrics for comparison with the 5-seed ensemble.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `chemical_holdout_f100_5seed` is the current mainline external-generalization result.",
            "- `random_8_2_3seed` and `random_5fold_3seed` are same-distribution/interpolation references and are pending 5-seed refresh.",
            "- Ensemble metrics are computed by averaging aligned prediction rows across seeds; they should be reported separately from single-model mean +/- SD.",
            "",
            "## Current Overall Metrics",
            "",
            markdown_table(
                overall.loc[
                    :,
                    ["evaluation_policy", "n", "r2", "rmse", "mae", "huber_loss", "ensemble_seed_count"],
                ]
            ),
            "",
            "## Main Task Count",
            "",
            f"- Main-scope task rows: {len(task_main)}.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[format_markdown_value(value) for value in row] for row in frame.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
