from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

from scripts.summarize_v1_2_44_second_layer_matrix import (
    B1_CELLS,
    B1_CONTRASTS,
    REQUIRED_CELLS,
    assert_aligned_prediction_sets,
    audit_run_manifests,
    build_b1_heatmap_data,
    fit_task_mw_baseline,
    identify_cell,
    main,
    plot_b1_heatmap,
    task_mean_predictions,
    within_r2,
)


def prediction(
    aggregate_id: str,
    *,
    task: str = "ECx_Growth",
    y_molkg: float = 2.0,
    pred_molkg: float = 2.1,
    mw: float = 100.0,
) -> dict:
    offset = 5.0  # log10(1000 * 100)
    return {
        "aggregate_id": aggregate_id,
        "task_head": task,
        "result_ids": [f"result-{aggregate_id}"],
        "y_molkg": y_molkg,
        "pred_molkg": pred_molkg,
        "y_mgkg": y_molkg - offset,
        "pred_mgkg": pred_molkg - offset,
        "molecular_weight": mw,
        "log10_mw": 2.0,
        "evaluation_part": "test",
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("M10_仅水相预训练_种子42/deep/full/manifest.json", "M10"),
        ("M01_仅土壤pTox适配_种子2042/deep/full/manifest.json", "M01"),
        ("M11F_冻结主干_种子3407/deep/full/manifest.json", "M11F"),
        ("M11U复现_全参数微调_种子8417/deep/full/manifest.json", "M11U"),
        ("B2C_仅上下文输入_种子42/deep/full/manifest.json", "B2C"),
        ("B2M_仅分子输入_种子42/deep/full/manifest.json", "B2M"),
        ("B2_仅上下文输入_种子42/deep/full/manifest.json", "B2C"),
        ("B2_仅分子输入_种子42/deep/full/manifest.json", "B2M"),
        ("M00_从头训练_固定评价边界_种子42/deep/full/manifest.json", "M00"),
        ("v1.2.40_X0_molar_seed42/deep/full/manifest.json", "M11_v140"),
        ("v1.2.40_X0_mass_seed42/deep/full/manifest.json", "ScaleMass"),
    ],
)
def test_identify_cell_accepts_english_ids_and_chinese_run_folders(
    text: str, expected: str
) -> None:
    assert identify_cell(text) == expected


def test_alignment_fails_closed_on_row_identity_mismatch() -> None:
    sets = {
        ("M00", 42, "test"): [prediction("a")],
        ("M10", 42, "test"): [prediction("b")],
    }
    with pytest.raises(ValueError, match="Row identity mismatch"):
        assert_aligned_prediction_sets(
            sets,
            cells=("M00", "M10"),
            seeds=(42,),
            parts=("test",),
        )


def test_alignment_fails_closed_on_truth_mismatch() -> None:
    sets = {
        ("M00", 42, "test"): [prediction("a", y_molkg=2.0)],
        ("M10", 42, "test"): [prediction("a", y_molkg=2.2)],
    }
    with pytest.raises(ValueError, match="Truth/MW mismatch"):
        assert_aligned_prediction_sets(
            sets,
            cells=("M00", "M10"),
            seeds=(42,),
            parts=("test",),
        )


def test_within_r2_uses_evaluation_task_centering() -> None:
    rows = [
        prediction("a", task="A", y_molkg=0.0, pred_molkg=0.0),
        prediction("b", task="A", y_molkg=2.0, pred_molkg=1.0),
        prediction("c", task="B", y_molkg=10.0, pred_molkg=10.0),
        prediction("d", task="B", y_molkg=12.0, pred_molkg=11.0),
    ]
    assert within_r2(rows, truth="y_molkg", prediction="pred_molkg") == pytest.approx(0.5)


def test_mean_baseline_uses_train_only_task_mean() -> None:
    train = [
        prediction("train-a", task="A", y_molkg=1.0),
        prediction("train-b", task="A", y_molkg=3.0),
    ]
    test = [prediction("test", task="A", y_molkg=100.0)]
    predicted = task_mean_predictions(train, test)
    assert predicted[0]["pred_molkg"] == pytest.approx(2.0)
    assert predicted[0]["pred_mgkg"] == pytest.approx(-3.0)


def test_mean_baseline_fails_when_test_task_is_absent_from_train() -> None:
    train = [prediction("train", task="A")]
    test = [prediction("test", task="B")]
    with pytest.raises(ValueError, match="lacks train-only task means"):
        task_mean_predictions(train, test)


def test_mw_only_fails_when_within_task_mw_variation_is_absent() -> None:
    rows = [
        prediction("a", task="A", mw=100.0),
        prediction("b", task="A", mw=100.0),
    ]
    with pytest.raises(ValueError, match="lacks within-task logMW variation"):
        fit_task_mw_baseline(rows, target="y_molkg")


def test_manifest_audit_fails_when_context_only_hides_tanimoto_weighting() -> None:
    run_map = {}
    for cell in REQUIRED_CELLS:
        ablation = "no_molecular_input" if cell == "B2C" else "no_context" if cell == "B2M" else "full"
        features = {
            "use_descriptors": cell != "B2C",
            "use_fingerprint": cell != "B2C",
            "use_molecular_graph": False,
            "use_context_numeric": cell != "B2M",
            "use_duration_features": cell != "B2M",
            "use_species_lifestage": cell != "B2M",
            "use_other_categorical_context": cell != "B2M",
            "use_medium_adapter": False,
        }
        run_map[(cell, 42)] = {
            "manifest": {
                "ablation": ablation,
                "ablation_features": features,
                "head_routing": "task_target",
                "source_weighting": {
                    "method": "tanimoto_to_finetune",
                    "applied": cell != "B2C",
                },
            }
        }
    with pytest.raises(ValueError, match="context-only protocol"):
        audit_run_manifests(run_map, seeds=(42,))


def test_b1_delta_mae_sign_is_positive_when_m11u_is_better() -> None:
    seeds = (42, 2042, 3407, 8417)
    task = "ECx_Growth"
    comparator_mae = {"M10": 0.60, "M01": 0.65, "M00": 0.70, "M11F": 0.55}
    seed_rows = []
    ensemble_rows = []
    for cell in B1_CELLS:
        mae = 0.50 if cell == "M11U" else comparator_mae[cell]
        for seed in seeds:
            seed_rows.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "task_head": task,
                    "common_molkg_mae": mae,
                }
            )
        ensemble_rows.append(
            {"cell": cell, "task_head": task, "common_molkg_mae": mae, "n": 30}
        )
    rows = build_b1_heatmap_data(
        seed_rows,
        ensemble_rows,
        seeds=seeds,
        tasks=(task,),
    )
    by_comparator = {row["comparator"]: row for row in rows}
    assert by_comparator["M00"]["delta_mae_ensemble"] == pytest.approx(0.20)
    assert all(row["delta_mae_ensemble"] > 0 for row in rows)


def test_heatmap_exports_png_and_editable_svg(tmp_path: Path) -> None:
    tasks = tuple(f"ECx_Task{i:02d}" for i in range(18))
    rows = []
    for task_index, task in enumerate(tasks):
        for contrast_index, (_, _, label) in enumerate(B1_CONTRASTS):
            rows.append(
                {
                    "task_head": task,
                    "contrast": label,
                    "delta_mae_ensemble": (task_index - 8.5) * 0.002
                    + (contrast_index - 1.5) * 0.005,
                }
            )
    output_base = tmp_path / "B1_18任务迁移增益热图"
    plot_b1_heatmap(rows, tasks=tasks, output_base=output_base)
    assert output_base.with_suffix(".png").stat().st_size > 0
    svg = output_base.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "土壤pTox适配" in svg


def test_end_to_end_writes_complete_chinese_summary_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_root = tmp_path / "legacy"
    matrix_root = tmp_path / "第二层核心因果实验矩阵_v1_2_44"
    output_dir = matrix_root / "统一汇总_中文"
    seeds = (42, 2042, 3407, 8417)
    legacy_names = {
        "M11_v140": "v1.2.40_X0_molar_seed{seed}",
        "ScaleMass": "v1.2.40_X0_mass_seed{seed}",
    }
    new_names = {
        "M00": "M00_从头训练_固定评价边界_种子{seed}",
        "M10": "M10_仅水相预训练_种子{seed}",
        "M01": "M01_仅土壤pTox适配_种子{seed}",
        "M11F": "M11F_冻结主干_种子{seed}",
        "M11U": "M11U复现_全参数微调_种子{seed}",
        "B2C": "B2C_仅上下文输入_种子{seed}",
        "B2M": "B2M_仅分子输入_种子{seed}",
    }
    cell_error = {
        "M00": 0.30,
        "M10": 0.22,
        "M01": 0.24,
        "M11F": 0.19,
        "M11U": 0.15,
        "M11_v140": 0.155,
        "ScaleMass": 0.18,
        "B2C": 0.28,
        "B2M": 0.21,
    }
    for cell, template in {**legacy_names, **new_names}.items():
        root = legacy_root if cell in legacy_names else matrix_root
        for seed in seeds:
            run_dir = root / template.format(seed=seed) / "deep" / "full" / "split"
            run_dir.mkdir(parents=True)
            ablation = (
                "no_molecular_input"
                if cell == "B2C"
                else "no_context"
                if cell == "B2M"
                else "full"
            )
            features = {
                "use_descriptors": cell != "B2C",
                "use_fingerprint": cell != "B2C",
                "use_molecular_graph": False,
                "use_context_numeric": cell != "B2M",
                "use_duration_features": cell != "B2M",
                "use_species_lifestage": cell != "B2M",
                "use_other_categorical_context": cell != "B2M",
                "use_medium_adapter": False,
            }
            weighting_applied = cell != "M00"
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "run_name_zh": template.format(seed=seed),
                        "ablation": ablation,
                        "ablation_features": features,
                        "head_routing": "task_target",
                        "source_weighting": {
                            "method": (
                                "tanimoto_to_finetune" if weighting_applied else "none"
                            ),
                            "applied": weighting_applied,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rows = []
            for task_index in range(18):
                task = f"ECx_Task{task_index:02d}"
                for part, count in (("train", 2), ("validation", 2), ("test", 2)):
                    for index in range(count):
                        aggregate_id = f"{part}-{task_index}-{index}"
                        mw = 80.0 + task_index * 6.0 + index * (25.0 + task_index)
                        y_molkg = 1.0 + task_index * 0.08 + index * 0.25
                        offset = math.log10(1000.0 * mw)
                        seed_shift = (seed % 13) * 0.0005
                        error = cell_error[cell] + task_index * 0.001 + seed_shift
                        y_native = y_molkg - offset if cell == "ScaleMass" else y_molkg
                        pred_native = y_native + error
                        if part == "train":
                            split_part = "finetune_mgkg"
                        elif part == "validation":
                            split_part = (
                                "valid"
                                if cell in new_names
                                else "validation"
                                if cell == "M00"
                                else "finetune_mgkg_validation"
                            )
                        else:
                            split_part = "test"
                        rows.append(
                            {
                                "aggregate_id": aggregate_id,
                                "task_head": task,
                                "split_part": split_part,
                                "target_name": (
                                    "neg_log10_mg_kg"
                                    if cell == "ScaleMass"
                                    else "neg_log10_mol_kg"
                                ),
                                "target_family": (
                                    "solid_neglog_mg_kg"
                                    if cell == "ScaleMass"
                                    else "solid_neglog_mol_kg"
                                ),
                                "medium_domain": "soil",
                                "y_true": y_native,
                                "y_pred": pred_native,
                                "molecular_weight_g_mol_used": mw,
                                "result_ids": json.dumps([f"result-{aggregate_id}"]),
                                "smiles": "CCO",
                            }
                        )
            # A non-final aquatic row with no MW proves that the loader filters
            # before parsing/retaining large stage-1/2 prediction content.
            rows.append(
                {
                    "aggregate_id": "aquatic-source-row",
                    "task_head": "ECx_Growth",
                    "split_part": "train",
                    "target_name": "ptox_mol_l",
                    "target_family": "aquatic_pTox_mol_L",
                    "medium_domain": "aquatic",
                    "y_true": 4.0,
                    "y_pred": 4.1,
                    "molecular_weight_g_mol_used": "",
                    "result_ids": "[]",
                    "smiles": "",
                }
            )
            with (run_dir / "predictions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_v1_2_44_second_layer_matrix.py",
            "--legacy-root",
            str(legacy_root),
            "--matrix-root",
            str(matrix_root),
            "--output-dir",
            str(output_dir),
        ],
    )
    main()
    expected = (
        "01_逐种子整体指标.csv",
        "03_四种子逐行集成主指标.csv",
        "06_B1_18任务迁移增益热图数据.csv",
        "07_B2_任务均值与输入信号对照.csv",
        "08_B3_尺度对照指标.csv",
        "12_B3_误差分子量相关性.csv",
        "B1_18任务迁移增益热图.png",
        "B1_18任务迁移增益热图.svg",
        "汇总说明.json",
    )
    for filename in expected:
        assert (output_dir / filename).stat().st_size > 0
    summary = json.loads((output_dir / "汇总说明.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["task_count"] == 18
    assert summary["primary_reporting"] == "four_seed_rowwise_prediction_ensemble_on_test"
