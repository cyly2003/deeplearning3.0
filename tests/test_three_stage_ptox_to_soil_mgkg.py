from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from scripts.build_three_stage_ptox_to_soil_mgkg_split import build_three_stage_split
from qsar_tl.training.baseline import load_split_frame
from qsar_tl.training.deep_experiment import apply_finetune_freeze, apply_head_routing, get_ablation_spec


def test_three_stage_split_keeps_ptox_and_mgkg_roles_separate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "three_stage.sqlite"
        with closing(sqlite3.connect(db_path)) as conn:
            columns = "aggregate_id TEXT, medium_domain TEXT, target_name TEXT, target_family TEXT, task_head TEXT, target_value REAL"
            conn.execute(f"CREATE TABLE combined_records ({columns})")
            conn.execute(f"CREATE TABLE soil_ptox_records ({columns})")
            conn.execute(f"CREATE TABLE soil_mgkg_records ({columns})")
            conn.execute(
                """
                CREATE TABLE split_assignments (
                    split_name TEXT NOT NULL,
                    record_id TEXT,
                    aggregate_id TEXT,
                    split_part TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    split_type TEXT NOT NULL,
                    source_table TEXT NOT NULL,
                    group_key TEXT
                )
                """
            )
            aquatic = [
                ("SHARED", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.0),
                ("A2", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.1),
            ]
            soil_ptox = [
                ("SHARED", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.2),
                ("P2", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.3),
                ("P3", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.4),
            ]
            soil_mgkg = [
                ("SHARED", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.0),
                ("M2", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.1),
                ("M3", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.2),
            ]
            conn.executemany("INSERT INTO combined_records VALUES (?, ?, ?, ?, ?, ?)", aquatic + soil_ptox + soil_mgkg)
            conn.executemany("INSERT INTO soil_ptox_records VALUES (?, ?, ?, ?, ?, ?)", soil_ptox)
            conn.executemany("INSERT INTO soil_mgkg_records VALUES (?, ?, ?, ?, ?, ?)", soil_mgkg)
            conn.executemany(
                "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("SoilPtox_B", None, "SHARED", "train", 42, "random", "soil_ptox_records", "SHARED"),
                    ("SoilPtox_B", None, "P2", "train", 42, "random", "soil_ptox_records", "P2"),
                    ("SoilPtox_B", None, "P3", "test", 42, "random", "soil_ptox_records", "P3"),
                    ("SoilMgkg_B", None, "SHARED", "train", 42, "random", "soil_mgkg_records", "SHARED"),
                    ("SoilMgkg_B", None, "M2", "train", 42, "random", "soil_mgkg_records", "M2"),
                    ("SoilMgkg_B", None, "M3", "test", 42, "random", "soil_mgkg_records", "M3"),
                ],
            )
            conn.commit()

        summary = build_three_stage_split(
            db_path=db_path,
            source_table="combined_records",
            soil_ptox_source_table="soil_ptox_records",
            soil_ptox_split_name="SoilPtox_B",
            soil_mgkg_source_table="soil_mgkg_records",
            soil_mgkg_split_name="SoilMgkg_B",
            split_name="M_three_stage",
        )

        assert summary == {"finetune": 2, "finetune_mgkg": 2, "test": 1, "train": 2}
        frame = load_split_frame(
            db_path,
            split_name="M_three_stage",
            source_table="combined_records",
            allow_mixed_target_dimensions=True,
        )
        assert len(frame) == sum(summary.values())
        assert frame.attrs["split_join_audit"]["assignment_rows"] == len(frame)
        assert frame.attrs["split_join_audit"]["loaded_rows"] == len(frame)
        assert frame.attrs["split_join_audit"]["removed_rows_count"] == 0
        assert set(frame["split_part"]) == {"train", "finetune", "finetune_mgkg", "test"}
        assert not ((frame["target_name"] == "ptox_mol_l") & (frame["split_part"] == "test")).any()
        stage3_test = frame.loc[frame["split_part"] == "test"]
        assert set(stage3_test["medium_domain"]) == {"soil"}
        assert set(stage3_test["target_name"]) == {"neg_log10_mg_kg"}
        assert set(stage3_test["target_family"]) == {"solid_neglog_mg_kg"}

        with closing(sqlite3.connect(db_path)) as conn:
            assigned = conn.execute(
                "SELECT record_id, group_key FROM split_assignments WHERE split_name = 'M_three_stage'"
            ).fetchall()
            assert len(assigned) == len(frame)
            assert len({record_id for record_id, _ in assigned}) == len(assigned)
            assert all(record_id.startswith("stage_sample_v1:") for record_id, _ in assigned)
            assert all(
                "medium_domain=" in group_key
                and "target_name=" in group_key
                and "target_family=" in group_key
                for _, group_key in assigned
            )

        routed = apply_head_routing(frame, mode="task_target")
        ptox_heads = set(routed.loc[routed["target_name"] == "ptox_mol_l", "model_head"])
        mgkg_heads = set(routed.loc[routed["target_name"] == "neg_log10_mg_kg", "model_head"])
        assert ptox_heads == {"ECx_Mortality__aquatic_pTox_mol_L"}
        assert mgkg_heads == {"ECx_Mortality__solid_neglog_mg_kg"}

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                """
                UPDATE split_assignments
                SET group_key = REPLACE(
                    group_key,
                    'target_name=neg_log10_mg_kg',
                    'target_name=ptox_mol_l'
                )
                WHERE split_name = 'M_three_stage' AND aggregate_id = 'M3'
                """
            )
            conn.commit()
        with pytest.raises(ValueError, match="Strict stage contract mismatch"):
            load_split_frame(
                db_path,
                split_name="M_three_stage",
                source_table="combined_records",
                allow_mixed_target_dimensions=True,
            )


def test_three_stage_split_rejects_duplicate_strict_composite_identity() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "duplicate_identity.sqlite"
        with closing(sqlite3.connect(db_path)) as conn:
            columns = "aggregate_id TEXT, medium_domain TEXT, target_name TEXT, target_family TEXT"
            conn.execute(f"CREATE TABLE combined_records ({columns})")
            conn.execute(f"CREATE TABLE soil_ptox_records ({columns})")
            conn.execute(f"CREATE TABLE soil_mgkg_records ({columns})")
            conn.execute(
                """
                CREATE TABLE split_assignments (
                    split_name TEXT NOT NULL,
                    record_id TEXT,
                    aggregate_id TEXT,
                    split_part TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    split_type TEXT NOT NULL,
                    source_table TEXT NOT NULL,
                    group_key TEXT
                )
                """
            )
            aquatic_duplicate = (
                "A1",
                "aquatic",
                "ptox_mol_l",
                "aquatic_pTox_mol_L",
            )
            soil_ptox = [
                ("P1", "soil", "ptox_mol_l", "aquatic_pTox_mol_L"),
                ("P2", "soil", "ptox_mol_l", "aquatic_pTox_mol_L"),
            ]
            soil_mgkg = [
                ("M1", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg"),
                ("M2", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg"),
            ]
            conn.executemany(
                "INSERT INTO combined_records VALUES (?, ?, ?, ?)",
                [aquatic_duplicate, aquatic_duplicate, *soil_ptox, *soil_mgkg],
            )
            conn.executemany("INSERT INTO soil_ptox_records VALUES (?, ?, ?, ?)", soil_ptox)
            conn.executemany("INSERT INTO soil_mgkg_records VALUES (?, ?, ?, ?)", soil_mgkg)
            conn.executemany(
                "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("SoilPtox_B", None, "P1", "train", 42, "random", "soil_ptox_records", "P1"),
                    ("SoilPtox_B", None, "P2", "test", 42, "random", "soil_ptox_records", "P2"),
                    ("SoilMgkg_B", None, "M1", "train", 42, "random", "soil_mgkg_records", "M1"),
                    ("SoilMgkg_B", None, "M2", "test", 42, "random", "soil_mgkg_records", "M2"),
                ],
            )
            conn.commit()

        with pytest.raises(ValueError, match="duplicate strict composite identities"):
            build_three_stage_split(
                db_path=db_path,
                source_table="combined_records",
                soil_ptox_source_table="soil_ptox_records",
                soil_ptox_split_name="SoilPtox_B",
                soil_mgkg_source_table="soil_mgkg_records",
                soil_mgkg_split_name="SoilMgkg_B",
                split_name="M_three_stage",
            )


def test_molecular_input_ablations_keep_the_requested_context_contract() -> None:
    ms1 = get_ablation_spec("descriptors_with_context")
    ms2 = get_ablation_spec("fingerprint_with_context")
    no_molecular = get_ablation_spec("no_molecular_input")
    ms3 = get_ablation_spec("no_context")

    assert ms1.use_descriptors and not ms1.use_fingerprint
    assert ms1.use_context_numeric and ms1.use_species_lifestage
    assert not ms2.use_descriptors and ms2.use_fingerprint
    assert ms2.use_context_numeric and ms2.use_species_lifestage
    assert not no_molecular.use_descriptors and not no_molecular.use_fingerprint
    assert no_molecular.use_context_numeric and no_molecular.use_species_lifestage
    assert ms3.use_descriptors and ms3.use_fingerprint
    assert not ms3.use_context_numeric and not ms3.use_species_lifestage


def test_mgkg_final_stage_allows_shared_representation_and_new_head_to_train() -> None:
    import torch.nn as nn

    class TinyNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Linear(2, 2)
            self.embeddings = nn.ModuleDict({"species": nn.Embedding(3, 2)})
            self.adapters = nn.ModuleList()
            self.heads = nn.ModuleDict({"ptox": nn.Linear(2, 1), "mgkg": nn.Linear(2, 1)})

    model = TinyNetwork()
    apply_finetune_freeze(model, "none")

    assert all(parameter.requires_grad for parameter in model.heads.parameters())
    assert all(parameter.requires_grad for parameter in model.trunk.parameters())
    assert all(parameter.requires_grad for parameter in model.embeddings.parameters())
