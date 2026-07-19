from __future__ import annotations

import csv
import copy
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
    build_mgkg_hierarchical_head_spec,
    build_task_equal_width_target_bin_sampling_spec,
    export_stage3_init_checkpoint,
    get_ablation_spec,
    load_stage3_init_checkpoint,
    canonical_sha256,
    seal_stage3_init_contract,
    stage3_init_checkpoint_contract,
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


def test_stage3_equal_width_target_bin_sampling_is_train_only_bounded_and_preserves_task_mass() -> None:
    samples = []
    for task, values in {
        "A__solid_neglog_mol_kg": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
        "B__solid_neglog_mol_kg": [1.0, 1.0, 2.0, 3.0],
    }.items():
        samples.extend(
            {
                "task_head": task,
                "target_value_raw": value,
                "target_family": "solid_neglog_mol_kg",
            }
            for value in values
        )
    validation_index = len(samples)
    samples.append(
        {
            "task_head": "A__solid_neglog_mol_kg",
            "target_value_raw": 99.0,
            "target_family": "solid_neglog_mol_kg",
        }
    )
    train_indices = list(range(validation_index))

    weights, audit = build_task_equal_width_target_bin_sampling_spec(
        samples,
        train_indices=train_indices,
        enabled=True,
        bins=10,
        min_weight=0.5,
        max_weight=2.0,
    )

    assert validation_index not in weights
    assert audit["kind"] == "within_task_equal_width_target_bin_inverse_frequency"
    assert min(weights.values()) >= 0.5
    assert max(weights.values()) <= 2.0
    assert len({round(value, 8) for value in weights.values()}) > 1
    for task in ("A__solid_neglog_mol_kg", "B__solid_neglog_mol_kg"):
        task_indices = [
            index for index in train_indices if samples[index]["task_head"] == task
        ]
        assert sum(weights[index] for index in task_indices) == pytest.approx(
            len(task_indices), abs=1e-7
        )
        assert audit["tasks"][task]["expected_mass_error"] == pytest.approx(0.0, abs=1e-7)


def test_hierarchical_head_counts_and_residual_scales_use_stage3_training_only() -> None:
    samples = [
        {
            "task_head": "ECx_Mortality__solid_neglog_mol_kg",
            "task_family": "ECx",
            "target_family": "solid_neglog_mol_kg",
        },
        {
            "task_head": "ECx_Growth__solid_neglog_mol_kg",
            "task_family": "ECx",
            "target_family": "solid_neglog_mol_kg",
        },
        {
            "task_head": "ECx_Mortality__solid_neglog_mol_kg",
            "task_family": "ECx",
            "target_family": "solid_neglog_mol_kg",
        },
        {  # Explicit validation row must not affect n or alpha.
            "task_head": "ECx_Mortality__solid_neglog_mol_kg",
            "task_family": "ECx",
            "target_family": "solid_neglog_mol_kg",
        },
        {  # Non-molar targets are never routed through G3.
            "task_head": "ECx_Mortality__solid_neglog_mg_kg",
            "task_family": "ECx",
            "target_family": "solid_neglog_mg_kg",
        },
    ]

    spec = build_mgkg_hierarchical_head_spec(
        samples,
        train_indices=[0, 1, 2, 4],
        enabled=True,
        family_tau=3.0,
        task_tau=2.0,
    )

    mortality = "ECx_Mortality__solid_neglog_mol_kg"
    growth = "ECx_Growth__solid_neglog_mol_kg"
    family = spec["head_families"][mortality]
    assert set(spec["heads"]) == {mortality, growth}
    assert spec["family_counts"][family] == 3
    assert spec["task_counts"][mortality] == 2
    assert spec["family_scales"][family] == pytest.approx(3.0 / 6.0)
    assert spec["task_scales"][mortality] == pytest.approx(2.0 / 4.0)


def test_stage3_checkpoint_partial_load_resets_g3_residuals(tmp_path: Path) -> None:
    import torch

    from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork

    ptox_head = "ECx_Mortality__aquatic_pTox_mol_L"
    molkg_head = "ECx_Mortality__solid_neglog_mol_kg"
    common = {
        "numeric_dim": 2,
        "fingerprint_dim": 2,
        "task_heads": (ptox_head, molkg_head),
        "hidden_dims": (4,),
        "dropout": 0.0,
    }
    source = EcotoxMultiTaskNetwork(DeepModelConfig(**common))
    with torch.no_grad():
        source.trunk[0].weight.fill_(0.25)
        source.heads[ptox_head].bias.fill_(0.75)
        source.heads[molkg_head].weight.fill_(3.0)
    contract = seal_stage3_init_contract(
        {
            "schema_version": 2,
            "preprocessing": {
                "manifest": {
                    "categorical_maps": {"family": {"<missing>": 0, "A": 1}},
                    "numeric_feature_names": ["MolWt", "TPSA"],
                }
            },
            "stage12_protocol": {
                "seed": 42,
                "stage1": {
                    "early_stopping": {"patience": 5},
                    "train_samples": {
                        "scientific_identity_sha256": canonical_sha256(
                            [{"aggregate_id": "a", "target_value_raw": 1.0}]
                        )
                    },
                },
            },
        }
    )
    checkpoint = tmp_path / "stage2.pt"
    export_stage3_init_checkpoint(source, checkpoint, contract=contract)

    target = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            **common,
            mgkg_hierarchical_heads=(molkg_head,),
            mgkg_hierarchical_head_families={molkg_head: "ecx"},
            mgkg_hierarchical_family_scales={"ecx": 0.9},
            mgkg_hierarchical_task_scales={molkg_head: 0.8},
        )
    )
    audit = load_stage3_init_checkpoint(
        target,
        checkpoint,
        expected_contract=contract,
        reset_hierarchical_heads=(molkg_head,),
    )

    assert audit["loaded"]
    assert torch.equal(target.trunk[0].weight, source.trunk[0].weight)
    assert torch.equal(target.heads[ptox_head].bias, source.heads[ptox_head].bias)
    assert torch.count_nonzero(target.heads[molkg_head].weight) == 0
    assert torch.count_nonzero(target.mgkg_hierarchical_family_heads["ecx"].weight) == 0
    apply_finetune_freeze(target, "heads_only")
    assert all(
        parameter.requires_grad
        for parameter in target.mgkg_hierarchical_shared_head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in target.mgkg_hierarchical_family_heads.parameters()
    )
    assert not any(parameter.requires_grad for parameter in target.trunk.parameters())


def test_stage3_contract_matches_cold_and_warm_source_weight_cache_but_not_science_changes() -> None:
    cold_summary = {
        "enabled": True,
        "method": "tanimoto_to_target",
        "alpha": 0.75,
        "min_weight": 0.5,
        "max_weight": 2.0,
        "applied": True,
        "weighted_samples": 12,
        "target_reference_samples": 4,
        "weight_min": 0.61,
        "weight_mean": 1.0,
        "weight_max": 1.44,
        "cache_enabled": True,
        "cache_hit": False,
        "cache_created": True,
        "cache_schema": "source_similarity_weights_v2",
        "cache_key": "deterministic-key",
        "cache_path": "/cold/cache/deterministic-key.npz",
    }
    warm_summary = {
        **cold_summary,
        "cache_hit": True,
        "cache_created": False,
        "cache_path": "/warm/cache/deterministic-key.npz",
    }

    def build_contract(summary: dict[str, object], weight_sha256: str = "weights-a") -> dict[str, object]:
        return stage3_init_checkpoint_contract(
            data_identity={"split": "fixed"},
            preprocessing={"schema": "fixed"},
            base_architecture={"model": "fixed"},
            weighting_and_auxiliary={
                # Pass the raw runtime summary: the canonical contract
                # constructor itself must remove cache provenance.
                "source_weighting": summary,
                "actual_training_sample_weights": {
                    "stage1": {"rows": 12, "sha256": weight_sha256},
                    "stage2": {"rows": 4, "sha256": "weights-stage2"},
                },
            },
            stage12_protocol={"seed": 42},
        )

    cold_contract = build_contract(cold_summary)
    warm_contract = build_contract(warm_summary)
    assert cold_contract == warm_contract
    assert cold_contract["sha256"] == warm_contract["sha256"]

    for key, value in (
        ("alpha", 0.9),
        ("method", "proxy_distance_to_finetune"),
    ):
        changed = copy.deepcopy(warm_summary)
        changed[key] = value
        assert build_contract(changed)["sha256"] != cold_contract["sha256"]
    assert build_contract(warm_summary, "weights-b")["sha256"] != cold_contract["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "category_mapping",
        "feature_order",
        "early_stop",
        "target_value",
    ],
)
def test_stage3_checkpoint_contract_rejects_preprocessing_protocol_and_target_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork

    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=2,
            fingerprint_dim=2,
            task_heads=("ptox", "molkg"),
            hidden_dims=(4,),
            dropout=0.0,
        )
    )
    payload = {
        "schema_version": 2,
        "preprocessing": {
            "manifest": {
                "categorical_maps": {"family": {"<missing>": 0, "A": 1}},
                "numeric_feature_names": ["MolWt", "TPSA"],
            }
        },
        "stage12_protocol": {
            "seed": 42,
            "stage1": {
                "early_stopping": {"patience": 5},
                "train_samples": {
                    "scientific_identity_sha256": canonical_sha256(
                        [{"aggregate_id": "a", "target_value_raw": 1.0}]
                    )
                },
            },
        },
    }
    contract = seal_stage3_init_contract(payload)
    checkpoint = tmp_path / f"stage2_{mutation}.pt"
    export_stage3_init_checkpoint(model, checkpoint, contract=contract)

    changed = copy.deepcopy(payload)
    if mutation == "category_mapping":
        changed["preprocessing"]["manifest"]["categorical_maps"]["family"]["A"] = 2
    elif mutation == "feature_order":
        changed["preprocessing"]["manifest"]["numeric_feature_names"] = [
            "TPSA",
            "MolWt",
        ]
    elif mutation == "early_stop":
        changed["stage12_protocol"]["stage1"]["early_stopping"]["patience"] = 6
    elif mutation == "target_value":
        changed["stage12_protocol"]["stage1"]["train_samples"][
            "scientific_identity_sha256"
        ] = canonical_sha256([{"aggregate_id": "a", "target_value_raw": 2.0}])
    expected = seal_stage3_init_contract(changed)

    with pytest.raises(ValueError, match="contract mismatch"):
        load_stage3_init_checkpoint(
            model,
            checkpoint,
            expected_contract=expected,
        )


def test_stage3_checkpoint_recomputes_payload_hash_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    import torch

    from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork

    model = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            numeric_dim=2,
            fingerprint_dim=2,
            task_heads=("ptox",),
            hidden_dims=(4,),
            dropout=0.0,
        )
    )
    contract = seal_stage3_init_contract(
        {"schema_version": 2, "stage12_protocol": {"seed": 42}}
    )
    checkpoint = tmp_path / "tampered.pt"
    export_stage3_init_checkpoint(model, checkpoint, contract=contract)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded["contract"]["stage12_protocol"]["seed"] = 43
    # Deliberately retain the stale self-reported sha256.
    torch.save(loaded, checkpoint)

    with pytest.raises(ValueError, match="payload.*hash is invalid"):
        load_stage3_init_checkpoint(
            model,
            checkpoint,
            expected_contract=contract,
        )


def test_g3_checkpoint_allows_only_registered_hierarchical_missing_keys(
    tmp_path: Path,
) -> None:
    import torch

    from qsar_tl.modeling.network import DeepModelConfig, EcotoxMultiTaskNetwork

    molkg_head = "ECx_Mortality__solid_neglog_mol_kg"
    common = {
        "numeric_dim": 2,
        "fingerprint_dim": 2,
        "task_heads": ("ptox", molkg_head),
        "hidden_dims": (4,),
        "dropout": 0.0,
    }
    source = EcotoxMultiTaskNetwork(DeepModelConfig(**common))
    target = EcotoxMultiTaskNetwork(
        DeepModelConfig(
            **common,
            mgkg_hierarchical_heads=(molkg_head,),
            mgkg_hierarchical_head_families={molkg_head: "ecx"},
        )
    )
    contract = seal_stage3_init_contract(
        {"schema_version": 2, "stage12_protocol": {"seed": 42}}
    )
    checkpoint = tmp_path / "missing_base_parameter.pt"
    export_stage3_init_checkpoint(source, checkpoint, contract=contract)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded["state_dict"].pop("trunk.0.weight")
    torch.save(loaded, checkpoint)

    with pytest.raises(ValueError, match="missing non-hierarchical"):
        load_stage3_init_checkpoint(
            target,
            checkpoint,
            expected_contract=contract,
            reset_hierarchical_heads=(molkg_head,),
        )
