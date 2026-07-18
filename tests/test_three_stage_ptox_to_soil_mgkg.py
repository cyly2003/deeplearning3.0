from __future__ import annotations

import csv
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pytest

from scripts.build_three_stage_ptox_to_soil_mgkg_split import build_three_stage_split
from qsar_tl.training.baseline import load_split_frame
from qsar_tl.training.deep_experiment import (
    _TargetReplayBatchSampler,
    apply_finetune_freeze,
    apply_head_routing,
    build_finetune_parameter_groups,
    get_ablation_spec,
)


def test_three_stage_split_keeps_ptox_and_mgkg_roles_separate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "three_stage.sqlite"
        audit_path = Path(tmpdir) / "routing_audit.csv"
        with closing(sqlite3.connect(db_path)) as conn:
            columns = (
                "aggregate_id TEXT, medium_domain TEXT, target_name TEXT, target_family TEXT, "
                "task_head TEXT, target_value REAL, result_ids TEXT, cas_number TEXT, "
                "test_ids TEXT, reference_numbers TEXT"
            )
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
                ("SHARED_AGG", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.0, "[100]", "50-00-0", "[1]", "[10]"),
                ("A_SHARED_RAW", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.1, "[300]", "51-00-0", "[2]", "[20]"),
                # Same CAS/test/reference as P2 is not an exclusion key when the
                # aggregate and exact raw result source remain distinct.
                ("A_NARROW", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.2, "[400]", "52-00-0", "[7]", "[70]"),
                ("A2", "aquatic", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.3, "[401]", "53-00-0", "[8]", "[80]"),
            ]
            soil_ptox = [
                ("SHARED_AGG", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.4, "[100]", "50-00-0", "[1]", "[10]"),
                ("P2", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.5, "[200]", "52-00-0", "[7]", "[70]"),
                # This test-part candidate is not used for stage-2 fitting, but
                # its exact source still has priority over aquatic stage 1.
                ("P3", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "ECx_Mortality", 1.6, "[300]", "54-00-0", "[9]", "[90]"),
            ]
            soil_mgkg = [
                ("M1", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.0, "[500]", "60-00-0", "[11]", "[110]"),
                ("M2", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.1, "[501]", "61-00-0", "[12]", "[120]"),
                ("M3", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "ECx_Mortality", 2.2, "[502]", "62-00-0", "[13]", "[130]"),
            ]
            placeholders = ", ".join("?" for _ in range(10))
            conn.executemany(f"INSERT INTO combined_records VALUES ({placeholders})", aquatic + soil_ptox + soil_mgkg)
            conn.executemany(f"INSERT INTO soil_ptox_records VALUES ({placeholders})", soil_ptox)
            conn.executemany(f"INSERT INTO soil_mgkg_records VALUES ({placeholders})", soil_mgkg)
            conn.executemany(
                "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ("SoilPtox_B", None, "SHARED_AGG", "train", 42, "random", "soil_ptox_records", "SHARED_AGG"),
                    ("SoilPtox_B", None, "P2", "train", 42, "random", "soil_ptox_records", "P2"),
                    ("SoilPtox_B", None, "P3", "test", 42, "random", "soil_ptox_records", "P3"),
                    ("SoilMgkg_B", None, "M1", "train", 42, "random", "soil_mgkg_records", "M1"),
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
            audit_csv=audit_path,
        )

        assert summary == {"finetune": 2, "finetune_mgkg": 2, "test": 1, "train": 2}
        with audit_path.open(encoding="utf-8-sig", newline="") as handle:
            audit = next(csv.DictReader(handle))
        assert audit["aquatic_ptox_candidates_before"] == "4"
        assert audit["routing_rule"] == "exact_aggregate_or_raw_result_v1"
        assert audit["aquatic_ptox_excluded_total"] == "2"
        assert audit["aquatic_ptox_excluded_shared_result_only"] == "1"
        assert audit["aquatic_ptox_excluded_both"] == "1"
        assert audit["aquatic_ptox_stage1_after"] == "2"
        assert audit["soil_ptox_candidates_overlapping_aquatic_before"] == "2"
        assert audit["residual_aggregate_id_overlap"] == "0"
        assert audit["residual_result_id_overlap"] == "0"
        assert len(audit["aquatic_ptox_excluded_identity_sha256"]) == 64
        assert len(audit["aquatic_ptox_stage1_identity_sha256"]) == 64
        assert len(audit["soil_ptox_candidates_identity_sha256"]) == 64
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
        assert set(frame.loc[frame["split_part"] == "train", "aggregate_id"].astype(str)) == {"A_NARROW", "A2"}
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
            columns = "aggregate_id TEXT, medium_domain TEXT, target_name TEXT, target_family TEXT, result_ids TEXT"
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
                "[1]",
            )
            soil_ptox = [
                ("P1", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "[2]"),
                ("P2", "soil", "ptox_mol_l", "aquatic_pTox_mol_L", "[3]"),
            ]
            soil_mgkg = [
                ("M1", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "[4]"),
                ("M2", "soil", "neg_log10_mg_kg", "solid_neglog_mg_kg", "[5]"),
            ]
            conn.executemany(
                "INSERT INTO combined_records VALUES (?, ?, ?, ?, ?)",
                [aquatic_duplicate, aquatic_duplicate, *soil_ptox, *soil_mgkg],
            )
            conn.executemany("INSERT INTO soil_ptox_records VALUES (?, ?, ?, ?, ?)", soil_ptox)
            conn.executemany("INSERT INTO soil_mgkg_records VALUES (?, ?, ?, ?, ?)", soil_mgkg)
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


def test_last_trunk_freeze_and_discriminative_parameter_groups() -> None:
    import torch.nn as nn

    class TinyNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
            self.embeddings = nn.ModuleDict({"species": nn.Embedding(3, 2)})
            self.adapters = nn.ModuleList()
            self.heads = nn.ModuleDict({"ptox": nn.Linear(2, 1), "mgkg": nn.Linear(2, 1)})
            self.mgkg_residual_adapter = None

    model = TinyNetwork()
    groups = build_finetune_parameter_groups(
        model,
        freeze_mode="last_trunk",
        head_learning_rate=5e-4,
        trunk_learning_rate=3e-5,
    )

    assert [group["name"] for group in groups] == ["head", "trunk"]
    assert [group["lr"] for group in groups] == [5e-4, 3e-5]
    assert all(parameter.requires_grad for parameter in model.heads.parameters())
    assert not any(parameter.requires_grad for parameter in model.trunk[0].parameters())
    assert all(parameter.requires_grad for parameter in model.trunk[2].parameters())
    assert not any(parameter.requires_grad for parameter in model.embeddings.parameters())
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert grouped == trainable


def test_target_replay_batch_sampler_has_fixed_ratio_and_epoch_rotation() -> None:
    sampler = _TargetReplayBatchSampler(
        target_count=12,
        replay_count=5,
        batch_size=8,
        replay_fraction=0.25,
        seed=42,
    )
    epoch_zero = list(iter(sampler))
    target_rows = [index for batch in epoch_zero for index in batch if index < 12]
    replay_rows = [index for batch in epoch_zero for index in batch if index >= 12]

    assert sorted(target_rows) == list(range(12))
    assert all(len(batch) == 8 for batch in epoch_zero)
    assert all(sum(index >= 12 for index in batch) == 2 for batch in epoch_zero)
    assert len(replay_rows) == 4
    assert len(set(replay_rows)) == 4

    sampler.set_epoch(1)
    epoch_one = list(iter(sampler))
    assert epoch_one != epoch_zero
    sampler.set_epoch(0)
    assert list(iter(sampler)) == epoch_zero
