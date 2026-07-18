# Mainline Remote Inventory and Cleanup Candidates 2026-07-10

更新时间：2026-07-12 12:49 (+08:00)

## 0. Server Expiry Preservation Update 2026-07-12

After a fresh remote/local comparison, the following missing or incomplete evidence-bearing artifacts were pulled and packaged before the remote server expires:

- Pulled to `outputs/experiments`: `runtime_summaries`, `v1_2_7_censored_ordinal_ad_first_batch_remote_summary`, `v1_2_15_random_split_policy_remote_summary`.
- Pulled or refreshed in `outputs/logs`: v1.2.24 complete scaffold-cluster log and runtime CSV, v1.2.31/v1.2.33 molecular-signal logs and runtime CSV, and v1.2.34-v1.2.36 queue nohup logs.
- Packaged to `实验汇总/13_结构骨架聚类外推审计_v1_2_24/远端训练结果_summary`: v1.2.24 complete summary, 20 files.
- Packaged to `实验汇总/分子信号强度探索_20260707/远端摘要明细`: v1.2.31-v1.2.36 summary details, 40 files.
- Packaged to `实验汇总/18_历史策略与方法筛选_正式摘要_v1_2_7_v1_2_15`: v1.2.7 and v1.2.15 formal historical strategy summaries, 20 files including README.
- Packaged to `实验汇总/10_运行时间记录`: strategy runtime summary, v1.2.24/v1.2.31/v1.2.33-v1.2.36 runtime CSVs, and key completion logs under `服务器到期补拉日志_20260712`.

Raw training trees and model checkpoints were still not bulk-pulled. The current conclusion is preserved by summary, audit, runtime, and selected prediction-row files; raw directories should be pulled only if a future figure or aggregation needs them.

## 1. Remote Status

- Remote project: `/home/easyai/DL1/ecotox_qsar_transfer`
- Remote host: `easyai@i.easy-ai.cloud:32136`
- GPU state at check time: RTX 4060 Ti, about 4% GPU utilization, 586/16380 MiB used.
- Running state: no active QSAR training process was found. `pgrep` only returned the query process, NVIDIA queue threads, and system Python/network services.
- SSH warning observed: `remote port forwarding failed for listen port 7897`. This is known SSH noise and did not block read-only listing or `scp`.

## 2. Results Pulled Back This Turn

The following compact summary/runtime artifacts were missing locally and were pulled from the remote:

- `outputs/experiments/v1_2_34_padel_descriptor_remote_summary`
- `outputs/experiments/v1_2_35_padel_prior_clustered_remote_summary`
- `outputs/experiments/v1_2_36_graph_only_remote_summary`
- `outputs/logs/run_v1_2_34_padel_descriptor_times.csv`
- `outputs/logs/run_v1_2_35_padel_prior_clustered_times.csv`
- `outputs/logs/run_v1_2_36_graph_only_times.csv`

Audit status for the newly pulled summaries:

| Version | Runs | Required files | Exit code | Aquatic eval rows | Runtime |
|---|---:|---|---|---:|---:|
| v1.2.34 PaDEL raw | 2 | present | 0 | 0 | 5493 s |
| v1.2.35 PaDEL prior clustered | 2 | present | 0 | 0 | 5109 s |
| v1.2.36 graph-only | 2 | present | 0 | 0 | 3419 s |

Raw training directories were not bulk-pulled because they contain large `predictions.csv` files and checkpoints. Approximate remote sizes:

| Remote raw directory | Size | Decision |
|---|---:|---|
| `v1_2_31_molecular_size_sensitivity_remote` | 684 MB | keep remote; summary is enough for current conclusion |
| `v1_2_33_molecular_signal_ablation_remote` | 1.4 GB | keep remote; summary is enough for current conclusion |
| `v1_2_34_padel_descriptor_remote` | 349 MB | keep remote; summary pulled |
| `v1_2_35_padel_prior_clustered_remote` | 346 MB | keep remote; summary pulled |
| `v1_2_36_graph_only_remote` | 343 MB | keep remote; summary pulled |
| `v1_2_26_no_metal_random_split_ablation_remote` | 6.1 GB | keep remote; summary already local |

Pull raw directories only if we need to rebuild prediction-row figures or rerun custom aggregation that cannot be done from the existing summary tables.

## 3. Current Version Boundary

The best current version is boundary-specific:

| Use case | Current version | Evidence directory | Key metric |
|---|---|---|---|
| Primary random-interpolation reporting after removing metal/metalloid and inorganic chemicals | `v1.2.22` | `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary` | random 8:2 5-seed ensemble: n=2608, R2=0.7895, RMSE=0.8865, MAE=0.6243; random 5-fold 5-seed ensemble: n=13063, R2=0.7920, RMSE=0.8899, MAE=0.6200 |
| Chemical-family extrapolation pressure test | `v1.2.24` | `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary` | scaffold holdout 5-seed ensemble: n=2493, R2=0.1810, RMSE=1.4063, MAE=1.0664; scaffold 5-fold: n=11459, R2=0.3664, RMSE=1.4808, MAE=1.1178 |
| Molecular signal diagnostic | `v1.2.33-v1.2.36` | `outputs/experiments/v1_2_33..._summary` through `v1_2_36..._summary` | RDKit full remains the main molecular-input reference; PaDEL prior clustered recovers random performance but not scaffold extrapolation; graph-only retains signal but does not match full |
| Molecular-size/unit-conversion sensitivity | `v1.2.31` | `outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary` | removing molecular-size descriptors does not hurt random 8:2 and only mildly worsens scaffold holdout |
| Historical fixed-CAS anchor | `v1.2.12` / `v1.2.17` | `outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary`; `outputs/experiments/final_mainline_comparison` | historical fixed-CAS 5-seed ensemble: n=2594, MAE=0.9277, RMSE=1.2368, R2=0.5388 |

Recommended wording:

> The current primary reporting model is v1.2.22 for no-metal random interpolation. Structural extrapolation should be reported separately using v1.2.24 scaffold/similarity-cluster results. v1.2.33-v1.2.36 are molecular-signal diagnostics and should not be promoted to mainline replacements.

## 4. Keep as Active Evidence

Keep these local outputs in the active result layer:

- `outputs/experiments/v1_2_22_no_metal_random_split_remote_summary`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary`
- `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_audit`
- `outputs/experiments/v1_2_23_random_mainline_cst_ad`
- `outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary`
- `outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary`
- `outputs/experiments/v1_2_33_molecular_signal_ablation_remote_summary`
- `outputs/experiments/v1_2_34_padel_descriptor_remote_summary`
- `outputs/experiments/v1_2_35_padel_prior_clustered_remote_summary`
- `outputs/experiments/v1_2_36_graph_only_remote_summary`
- `outputs/experiments/final_mainline_comparison`
- `outputs/experiments/final_mainline_train_space`
- `实验汇总/00_主线结果总览与追溯文档`
- `实验汇总/01_最终指标汇总_论文主表`
- `实验汇总/03_随机划分主线_8比2与5折`
- `实验汇总/07_去金属无机敏感性_随机主线`
- `实验汇总/08_随机主线CST应用域`
- `实验汇总/11_子任务表现汇总`
- `实验汇总/13_结构骨架聚类外推审计_v1_2_24`
- `实验汇总/分子信号强度探索_20260707`
- `实验汇总/机器学习基线_分子描述符效应水平`
- `实验汇总/机器学习基线_分子描述符效应水平_土壤扩展n30`
- `实验汇总/机器学习基线_分子描述符效应水平_水相扩展n200`

## 5. Archive Candidates, Pending User Confirmation

No cleanup move has been performed yet. Recommended archive target:

- Local: `outputs/_archive/cleanup_20260710/`
- Remote: `/home/easyai/DL1/ecotox_qsar_transfer/outputs/_archive/cleanup_20260710/`

Archive-first candidates:

| Scope | Candidate pattern/path | Reason |
|---|---|---|
| Local smoke/debug | `outputs/experiments/_smoke*` | link checks only; not formal evidence |
| Local CST smoke | `outputs/experiments/v1_2_23_random_mainline_cst_ad_smoke*` | superseded by formal `v1_2_23_random_mainline_cst_ad` |
| Remote smoke/probe | `v1_2_21_random_split_ablation_remote_smoke`, `v1_2_21_random_split_ablation_remote_summary_smoke`, `v1_2_24_*_smoke`, `v1_2_10_*_probe`, `v1_2_8_*_probe`, `v1_2_9_*_probe` | smoke/probe outputs do not support current conclusions |
| Remote interim summaries | `v1_2_18_mainline_ablation_remote_summary_interim`, `v1_2_19_single_domain_bce_remote_summary_interim` | superseded by completed formal summaries |
| Remote raw diagnostic runs | `v1_2_31_molecular_size_sensitivity_remote`, `v1_2_33_molecular_signal_ablation_remote`, `v1_2_34_padel_descriptor_remote`, `v1_2_35_padel_prior_clustered_remote`, `v1_2_36_graph_only_remote` | large raw artifacts; compact summaries are local and sufficient for current interpretation |
| Remote old pilot/baseline roots | `baseline_matrix`, `easyai_remote`, `deep_ablation_pilot_remote`, `deep_medium_transfer_water_to_solid` | early pipeline/pilot outputs; not part of current no-metal mainline |
| Local duplicate traditional ML branch | `outputs/experiments/v1_2_28_species_endpoint_ml_descriptor_effect_baselines` if `v1_2_28_species_endpoint_ml_descriptor_effect_baselines_stable_pls` is complete | earlier version superseded by stable PLS run |
| Local ASCII duplicate summaries | `v1_2_29_*_summary_ascii`, `v1_2_30_*_summary_ascii` | only keep if referenced by a script or report; otherwise redundant next to Chinese-packaged result trees |

Before executing cleanup, verify each candidate still has a completed formal summary or a Chinese-packaged copy. Do not delete; move to archive with a manifest.

## 6. Next Cleanup Procedure

1. Recheck remote processes and GPU before any remote move.
2. Create a manifest listing source path, destination path, size, and reason.
3. Move only the confirmed candidates into archive directories.
4. Re-run quick presence checks for active evidence directories.
5. Update `PROJECT_STATUS.md`, this inventory, and `docs/experiment_registry.csv` after the archive step.
