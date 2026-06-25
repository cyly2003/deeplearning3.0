from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from qsar_tl.evaluation.application_domain import (
    ApplicationDomainConfig,
    build_application_domain_report,
    taxon_prefix_similarity,
    williams_leverage,
)


def test_taxon_prefix_similarity_uses_hierarchical_distance() -> None:
    query = ("animalia", "chordata", "actinopterygii", "cypriniformes", "leuciscidae")
    same_family = ("animalia", "chordata", "actinopterygii", "cypriniformes", "leuciscidae")
    same_order = ("animalia", "chordata", "actinopterygii", "cypriniformes", "cyprinidae")
    different_phylum = ("animalia", "arthropoda", "branchiopoda", "diplostraca", "daphniidae")

    assert taxon_prefix_similarity(query, same_family) == 1.0
    assert taxon_prefix_similarity(query, same_order) == 0.8
    assert taxon_prefix_similarity(query, different_phylum) == 0.2


def test_williams_leverage_standardizer_fits_train_only() -> None:
    descriptors = np.asarray([[0.0], [1.0], [1000.0]], dtype=float)
    fingerprints = np.zeros((3, 1), dtype=float)
    train_mask = np.asarray([True, True, False])

    leverage, critical_h, components_used = williams_leverage(
        descriptors,
        fingerprints,
        train_mask,
        pca_components=0,
    )

    assert components_used == 0
    assert critical_h == 3.0
    assert leverage[2] > 100000.0


def test_build_application_domain_report_outputs_chemical_and_species_ad(tmp_path: Path) -> None:
    db_path = tmp_path / "ad.sqlite"
    with closing(sqlite3.connect(db_path)) as conn:
        create_ad_fixture(conn)

    cache_path = tmp_path / "molecular_cache.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                json.dumps({"smiles": "CCO", "descriptors": [1.0, 2.0], "fingerprint": [1.0, 0.0, 1.0, 0.0]}),
                json.dumps({"smiles": "CCN", "descriptors": [1.2, 2.2], "fingerprint": [1.0, 1.0, 0.0, 0.0]}),
                json.dumps({"smiles": "O", "descriptors": [0.5, 1.0], "fingerprint": [0.0, 0.0, 1.0, 0.0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "ad_report.csv"

    result = build_application_domain_report(
        db_path,
        split_name="B_random_8_2",
        source_table="aggregated_task_records_soil_mg_kg_qc",
        out_path=out_path,
        config=ApplicationDomainConfig(
            fingerprint_size=4,
            molecular_cache_path=cache_path,
            pca_components=2,
            tanimoto_threshold=0.5,
            taxon_similarity_threshold=0.8,
        ),
    )

    report = pd.read_csv(out_path)
    test_row = report[report["split_part"] == "test"].iloc[0]
    assert result.rows == 3
    assert result.train_rows == 2
    assert result.pca_components_used == 1
    assert test_row["max_tanimoto_to_train"] == 0.5
    assert test_row["chemical_in_domain_tanimoto"]
    assert test_row["max_taxon_similarity_to_train"] == 0.2
    assert not test_row["species_in_domain_taxon"]
    assert not test_row["species_seen_train"]
    assert not test_row["family_seen_train"]
    assert test_row["ad_warning"] == "species_extrapolation"
    assert out_path.with_suffix(".csv.manifest.json").exists()


def create_ad_fixture(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE aggregated_task_records_soil_mg_kg_qc (
            aggregate_id INTEGER,
            cas_number TEXT,
            dtxsid TEXT,
            chemical_name TEXT,
            smiles TEXT,
            species_number TEXT,
            latin_name TEXT,
            kingdom TEXT,
            phylum TEXT,
            class_name TEXT,
            tax_order TEXT,
            family TEXT,
            task_head TEXT,
            target_name TEXT,
            target_basis TEXT,
            medium_domain TEXT,
            target_value_median REAL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO aggregated_task_records_soil_mg_kg_qc VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            (
                1,
                "64-17-5",
                "DTXSID-1",
                "Ethanol",
                "CCO",
                "sp1",
                "Daphnia magna",
                "Animalia",
                "Arthropoda",
                "Branchiopoda",
                "Diplostraca",
                "Daphniidae",
                "ECx_Mortality",
                "neg_log10_mg_kg",
                "mg/kg:NAT",
                "soil",
                1.0,
            ),
            (
                2,
                "75-04-7",
                "DTXSID-2",
                "Ethylamine",
                "CCN",
                "sp2",
                "Daphnia magna",
                "Animalia",
                "Arthropoda",
                "Branchiopoda",
                "Diplostraca",
                "Daphniidae",
                "ECx_Mortality",
                "neg_log10_mg_kg",
                "mg/kg:NAT",
                "soil",
                1.2,
            ),
            (
                3,
                "7732-18-5",
                "DTXSID-3",
                "Water",
                "O",
                "sp3",
                "Pimephales promelas",
                "Animalia",
                "Chordata",
                "Actinopterygii",
                "Cypriniformes",
                "Leuciscidae",
                "ECx_Mortality",
                "neg_log10_mg_kg",
                "mg/kg:NAT",
                "soil",
                2.0,
            ),
        ],
    )
    conn.execute(
        """
        CREATE TABLE split_assignments (
            split_name TEXT,
            record_id TEXT,
            aggregate_id TEXT,
            split_part TEXT,
            seed INTEGER,
            split_type TEXT,
            source_table TEXT,
            group_key TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("B_random_8_2", None, "1", "train", 42, "random_holdout_split", "aggregated_task_records_soil_mg_kg_qc", None),
            ("B_random_8_2", None, "2", "train", 42, "random_holdout_split", "aggregated_task_records_soil_mg_kg_qc", None),
            ("B_random_8_2", None, "3", "test", 42, "random_holdout_split", "aggregated_task_records_soil_mg_kg_qc", None),
        ],
    )
    conn.commit()
