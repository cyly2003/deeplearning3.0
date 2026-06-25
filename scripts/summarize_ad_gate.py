from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_deep_runs import metrics_from_prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AD gate/confidence tiers from AD prediction rows.")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-part", default="test")
    args = parser.parse_args()

    rows = []
    audit_root = Path(args.audit_root)
    for path in sorted(audit_root.glob("*/ad_prediction_rows.csv")):
        run_name = path.parent.name
        prediction_rows = [
            row for row in read_csv(path)
            if str(row.get("split_part", "")) == str(args.split_part)
        ]
        rows.extend(gate_rows(run_name, prediction_rows, split_part=str(args.split_part)))
    write_csv(Path(args.out), rows)


def gate_rows(run_name: str, rows: list[dict[str, str]], *, split_part: str) -> list[dict[str, Any]]:
    total = len(rows)
    tiers = [
        ("all_test", rows),
        ("overall_in_domain", [row for row in rows if truthy(row.get("ad_overall_in_domain"))]),
        (
            "not_species_extrapolation",
            [row for row in rows if row.get("ad_ad_warning") != "species_extrapolation"],
        ),
        (
            "species_task_family_seen_train",
            [row for row in rows if truthy(row.get("ad_species_task_family_seen_train"))],
        ),
        ("species_seen_train", [row for row in rows if truthy(row.get("ad_species_seen_train"))]),
        ("family_seen_train", [row for row in rows if truthy(row.get("ad_family_seen_train"))]),
        (
            "species_extrapolation_only",
            [row for row in rows if row.get("ad_ad_warning") == "species_extrapolation"],
        ),
        (
            "species_task_family_unseen",
            [row for row in rows if not truthy(row.get("ad_species_task_family_seen_train"))],
        ),
    ]
    output = []
    for tier, tier_rows in tiers:
        metrics = metrics_from_prediction_rows(tier_rows) if tier_rows else empty_metrics()
        output.append(
            {
                "run": run_name,
                "split_part": split_part,
                "tier": tier,
                "coverage_n": len(tier_rows),
                "coverage_fraction": len(tier_rows) / max(total, 1),
                **metrics,
            }
        )
    return output


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def empty_metrics() -> dict[str, Any]:
    return {
        "n": 0,
        "r2": "",
        "rmse": "",
        "mae": "",
        "huber_loss": "",
        "task_count": 0,
        "tasks": "",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["n"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
