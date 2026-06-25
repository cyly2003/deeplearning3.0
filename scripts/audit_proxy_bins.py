from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsar_tl.training.baseline import load_split_frame
from qsar_tl.training.deep_experiment import MOLECULAR_DESCRIPTOR_NAMES, MolecularFeatureBuilder


DESCRIPTOR_INDEX = {name: idx for idx, name in enumerate(MOLECULAR_DESCRIPTOR_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit environment-behavior proxy bins for a saved split.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--source-table", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--molecular-cache", default=None)
    parser.add_argument("--fingerprint-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    frame = load_split_frame(
        args.db,
        split_name=args.split_name,
        source_table=args.source_table,
        limit=args.limit,
    )
    encoder = MolecularFeatureBuilder(
        fingerprint_size=int(args.fingerprint_size),
        cache_path=args.molecular_cache,
    )
    record_rows = build_proxy_rows(frame, encoder)
    summary_rows = summarize_proxy_rows(record_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "proxy_audit_records.csv", record_rows)
    write_csv(out_dir / "proxy_audit_summary.csv", summary_rows)
    manifest = {
        "db": str(args.db),
        "source_table": args.source_table,
        "split_name": args.split_name,
        "rows": int(len(record_rows)),
        "summary_rows": int(len(summary_rows)),
        "encoder_source": encoder.source,
    }
    (out_dir / "proxy_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def build_proxy_rows(frame: Any, encoder: MolecularFeatureBuilder) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        descriptors, _ = encoder.encode(row.get("smiles"))
        molwt = descriptor(descriptors, "MolWt")
        tpsa = descriptor(descriptors, "TPSA")
        logp = descriptor(descriptors, "MolLogP")
        h_acceptors = descriptor(descriptors, "NumHAcceptors")
        h_donors = descriptor(descriptors, "NumHDonors")
        rings = descriptor(descriptors, "RingCount")
        rotatable = descriptor(descriptors, "RotatableBonds")
        logkoc_proxy = 0.54 * logp + 1.47 if np.isfinite(logp) else float("nan")
        rows.append(
            {
                "aggregate_id": row.get("aggregate_id", ""),
                "split_part": row.get("split_part", ""),
                "medium_domain": row.get("medium_domain", ""),
                "task_family": row.get("task_family", ""),
                "task_head": row.get("task_head", ""),
                "toxicity_bin_label": row.get("toxicity_bin_label", ""),
                "cas_number": row.get("cas_number", ""),
                "chemical_name": row.get("chemical_name", ""),
                "MolWt": molwt,
                "TPSA": tpsa,
                "MolLogP": logp,
                "HBA": h_acceptors,
                "HBD": h_donors,
                "RingCount": rings,
                "RotatableBonds": rotatable,
                "logKoc_proxy": logkoc_proxy,
                "logp_bin": logp_bin(logp),
                "tpsa_bin": tpsa_bin(tpsa),
                "molwt_bin": molwt_bin(molwt),
                "koc_proxy_bin": koc_proxy_bin(logkoc_proxy),
                "hbond_bin": hbond_bin(h_acceptors, h_donors),
                "flexibility_bin": flexibility_bin(rotatable),
            }
        )
    return rows


def descriptor(values: list[float], name: str) -> float:
    idx = DESCRIPTOR_INDEX[name]
    if idx >= len(values):
        return float("nan")
    value = float(values[idx])
    return value if np.isfinite(value) else float("nan")


def logp_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 1:
        return "low_logp_lt1"
    if value < 3:
        return "mid_logp_1_3"
    if value < 5:
        return "high_logp_3_5"
    return "very_high_logp_ge5"


def tpsa_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 40:
        return "low_tpsa_lt40"
    if value <= 90:
        return "mid_tpsa_40_90"
    return "high_tpsa_gt90"


def molwt_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 200:
        return "low_molwt_lt200"
    if value <= 500:
        return "mid_molwt_200_500"
    return "high_molwt_gt500"


def koc_proxy_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < 2:
        return "low_logkoc_proxy_lt2"
    if value < 3:
        return "mid_logkoc_proxy_2_3"
    return "high_logkoc_proxy_ge3"


def hbond_bin(h_acceptors: float, h_donors: float) -> str:
    total = h_acceptors + h_donors
    if not np.isfinite(total):
        return "missing"
    if total <= 2:
        return "low_hbond_0_2"
    if total <= 8:
        return "mid_hbond_3_8"
    return "high_hbond_gt8"


def flexibility_bin(rotatable: float) -> str:
    if not np.isfinite(rotatable):
        return "missing"
    if rotatable <= 2:
        return "low_rotatable_0_2"
    if rotatable <= 8:
        return "mid_rotatable_3_8"
    return "high_rotatable_gt8"


def summarize_proxy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_columns = ("split_part", "medium_domain", "task_family", "logp_bin", "tpsa_bin", "koc_proxy_bin")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(column, "")) for column in group_columns)].append(row)
    summary = []
    for key, items in sorted(grouped.items()):
        summary.append(
            {
                **{column: value for column, value in zip(group_columns, key)},
                "n": len(items),
                "unique_chemicals": len({str(item.get("cas_number", "")) for item in items if item.get("cas_number")}),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
