# 工作区保守整理清单 2026-06-22

## 整理原则

- 本次整理采用保守归档策略：移动到归档区，不直接删除实验结果或派生数据库。
- 代码、配置、脚本、测试和科研说明文档先纳入 GitHub 备份；`outputs/`、`*.sqlite`、`USER.txt`、缓存目录继续保持忽略。
- 当前主派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 保留在原位置；远端也保留对应 v2 数据库。
- 远端目录 `/home/easyai/DL1/ecotox_qsar_transfer` 不是 Git 工作树，远端整理通过本地同步脚本和远端 `mv` 归档完成。

## 整理前本地概况

| 路径 | 文件数 | 体积 | 说明 |
| --- | ---: | ---: | --- |
| `outputs/derived` | 8 | 10.924 GB | 派生 SQLite 与 WAL/SHM 文件 |
| `outputs/experiments` | 1907 | 1.558 GB | 本地拉回或本地生成的实验结果 |
| `outputs/audits` | 1149 | 0.421 GB | 数据审计中间结果 |
| `outputs/features` | 3 | 0.017 GB | 分子特征缓存，保留 |
| `outputs/logs` | 192 | 0.001 GB | 训练和同步日志，保留 |

## 整理前远端概况

| 路径 | 体积 | 说明 |
| --- | ---: | --- |
| `outputs/derived` | 18 GB | 远端派生 SQLite |
| `outputs/experiments` | 6.5 GB | 远端训练结果 |
| `outputs/features` | 18 MB | 远端特征缓存，保留 |
| `outputs/ad` | 12 MB | 应用域结果，保留 |
| `outputs/logs` | 1.7 MB | 远端日志，保留 |
| `outputs/audits` | 20 KB | 审计输出，保留 |

## 本地保留项

- `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 及其 `-wal` / `-shm` 伴随文件。
- `outputs/features/`、`outputs/logs/`、`outputs/audits/`。
- 当前主线或近期待比较结果：`v1_2_2*`、`v1_2_3*`、`v1_2_4*`、`v2_0_0_main_retrain`、`v2_0_0_transfer_low_soil`、`v2_0_0_transfer_diagnostics` 等仍保留原位；本地不存在的远端主结果不强制拉回。

## 本地归档候选

归档目标：`outputs/_archive/cleanup_20260622/`。

- 旧派生库：`modeling_dataset.sqlite`、`modeling_dataset.sqlite-wal`、`modeling_dataset.sqlite-shm`、`modeling_dataset_v1_0_0_rebuild.sqlite`、`modeling_dataset_smoke_v1.sqlite`。
- smoke/debug 目录：`deep_ablation_smoke_remote`、`deep_aquatic_ptox_smoke`、`easyai_remote`、`hpo_transfer`、`smoke_v1`、`v1_2_1_smoke_remote`、`v2_0_0_smoke`。
- legacy v1.0/v1.1/v1.2.0 原始实验目录：`v1_0_0_comparison`、`v1_0_0_same_budget`、`v1_1_0_transfer_step_diagnostics`、`v1_1_0_transfer_step_diagnostics_ascii`、`v1_2_0_hpo_v2_f20_remote_summary`。

## 远端归档候选

归档目标：`/home/easyai/DL1/ecotox_qsar_transfer/outputs/_archive/cleanup_20260622/`。

- 旧派生库：`outputs/derived/modeling_dataset.sqlite`、`outputs/derived/modeling_dataset_v1_0_0_rebuild.sqlite`。
- smoke/debug 目录：`baseline_aquatic_ptox_v2_smoke`、`deep_ablation_smoke_remote`、`deep_epoch30_code_smoke_remote`、`deep_pilot_remote`、`deep_smoke_remote`、`deep_aquatic_ptox_smoke`、`smoke_qc_low_soil_transfer`、`smoke_qc_soil_ptox`、`v1_2_0_hpo_smoke`、`v1_2_1_smoke_remote`、`v1_2_2_matrix_smoke_remote`、`v1_2_3_effect_level_weighting_smoke_remote`、`v2_0_0_smoke`。
- legacy v1.0/v1.1/v1.2.0 原始实验目录：`v1_0_0_same_budget`、`v1_1_0_transfer_step_diagnostics`、`v1_2_0_自动超参数优化_迁移验证_f20`、`v1_2_0_自动超参数优化_迁移验证_v2_f20`。

## 执行后核验项

- 本地 Git 状态只剩忽略项：`outputs/`、SQLite、缓存、`USER.txt`。
- 本地测试：`E:\TOOLS\anaconda\python.exe -m pytest`。
- 配置核验：`E:\TOOLS\anaconda\python.exe -m qsar_tl.cli validate-config configs/experiment.remote.easyai.yaml`。
- 远端核验：`pgrep` 不显示实际训练进程，`du -sh outputs/*` 和实验目录列表显示归档生效。
