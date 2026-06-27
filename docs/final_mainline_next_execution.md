# Final Mainline Next Execution Notes

更新时间：2026-06-27

本文档记录在 `v1.2.17` 方法文档与已有指标汇总完成后的下一批执行入口。它不是最终结果报告；真正训练完成后仍需更新 `PROJECT_STATUS.md`、`docs/experiment_registry.csv` 和对应 summary 目录。

## 1. 已完成的优先交付物

- 方法文档：`docs/final_mainline_methods_materials.md`
- 已有最终指标汇总脚本：`scripts/build_final_mainline_summary.py`
- 已有最终指标包：`outputs/experiments/final_mainline_comparison`

刷新已有指标包：

```bash
python scripts/build_final_mainline_summary.py
```

Windows 本地推荐解释器：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_final_mainline_summary.py
```

## 2. 随机划分 5-seed refresh

目标：把当前 random 8:2 和 random 5-fold 从 3-seed ensemble 扩展到 5-seed ensemble。

已完成 seed：`42 1042 2042`

只需补跑 seed：`3042 4042`

远端命令：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
SEEDS="3042 4042" ENSEMBLE_SEEDS="42 1042 2042 3042 4042" \
  bash scripts/run_v1_2_15_random_split_policy_remote.sh ensemble_all
```

当前状态：

- 已在远端后台启动，日志为 `outputs/logs/run_v1_2_15_random_split_policy_5seed_refresh_20260627_154528.log`。
- 2026-06-27 16:22 核验：第一项 `random8_2_seed3042` 已完成并落盘，脚本已自动进入 `random5fold_fold1_seed3042`。
- 完成全部 12 个新增训练后会自动重建 summary 和 5-seed ensemble 表。

本地只读状态检查：

```powershell
pwsh .\scripts\check_v1_2_15_random_refresh.ps1 `
  -Config configs\experiment.remote.easyai.yaml `
  -LocalPython E:\TOOLS\anaconda\envs\qsar-ph3\python.exe
```

判定标准：`[refresh-completion] complete=12 missing=0 expected=12` 且 `[summary-seeds] five_seed_ready` 同时出现。

完成后需要同步/检查：

- `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary/split_policy_ensemble_combined_summary.csv`
- `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary/split_policy_ensemble_family_summary.csv`
- `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary/split_policy_ensemble_main_task_summary.csv`
- `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary/split_policy_ensemble_task_summary.csv`

然后重新运行：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_final_mainline_summary.py
```

## 3. 主线训练空间导出

目标：为后续应用域、覆盖空间、化合物-物种联合空间图准备可直接作图的数据矩阵。

已新增脚本：

- `scripts/export_task_train_space.py`

已完成远端导出并同步到本地：

- `outputs/experiments/final_mainline_train_space/task_train_space_rows.csv.gz`
- `outputs/experiments/final_mainline_train_space/task_train_chemical_space.csv`
- `outputs/experiments/final_mainline_train_space/task_train_species_space.csv`
- `outputs/experiments/final_mainline_train_space/species_embedding_lookup.csv.gz`
- `outputs/experiments/final_mainline_train_space/task_train_species_embedding_space.csv.gz`
- `outputs/experiments/final_mainline_train_space/species_embedding_field_manifest.csv`

导出口径：

- split：`M_v2_aquatic_to_soil_ptox_adapt_C_f100`
- split parts：`train + finetune`
- 规模：296,368 行、6,610 个化合物、5,393 个物种、116 个训练 task head。
- embedding 来源：主线 seed42 的 `preprocessing.json` 和 `best_model.pt`；5-seed ensemble 没有单一共享 embedding 空间。

## 4. 主线 5-seed 消融

目标：在固定 `M_v2_aquatic_to_soil_ptox_adapt_C_f100` 与主线参数下，解释最终模型性能来自哪些输入模块和训练策略。

建议固定项：

- source table：`aggregated_task_records_aquatic_soil_ptox_qc`
- split：`M_v2_aquatic_to_soil_ptox_adapt_C_f100`
- seeds：`42 1042 2042 3042 4042`
- pretrain epochs：30
- finetune epochs：60
- finetune validation fraction：0.2
- target standardization：`per_task_target`
- batch size：512
- pretrain LR：0.0005
- finetune LR：0.0003082636455810776
- scheduler：pretrain cosine，finetune reduce_on_plateau

模块消融：

- `no_fingerprint`
- `no_descriptors`
- `no_species_lifestage`
- `no_duration`
- `no_context`
- `no_medium_adapter`
- `no_molecular_residual`

策略消融：

- no source weighting：`--source-weighting-method none`
- no toxicity binning：`--no-toxicity-binning`
- no censored loss：`--no-censored-loss`

已新增版本化 launcher，而不是复用旧脚本覆盖：

- `scripts/run_v1_2_18_mainline_ablation_remote.sh`
- 可选顺序队列入口：`scripts/run_v1_2_20_followup_queue_remote.sh`
- 输出根目录：`outputs/experiments/v1_2_18_mainline_ablation_remote`
- 汇总目录：`outputs/experiments/v1_2_18_mainline_ablation_remote_summary`

远端 smoke：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
bash scripts/run_v1_2_18_mainline_ablation_remote.sh smoke
```

远端正式矩阵：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
bash scripts/run_v1_2_18_mainline_ablation_remote.sh matrix
```

当前状态：脚本已同步远端并通过 `bash -n`，但完整矩阵尚未启动，建议等待 random 5-seed refresh 完成。

如果希望在 random refresh 达到完成判据后自动接上消融和单域矩阵，可在远端后台启动：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
stamp=$(date +%Y%m%d_%H%M%S)
log=outputs/logs/run_v1_2_20_followup_queue_${stamp}.log
pidfile=outputs/logs/run_v1_2_20_followup_queue_${stamp}.pid
setsid bash scripts/run_v1_2_20_followup_queue_remote.sh wait_then_followup \
  > "$log" 2>&1 < /dev/null &
echo $! > "$pidfile"
```

当前已启动该队列：

- log：`outputs/logs/run_v1_2_20_followup_queue_20260627_163936.log`
- pidfile：`outputs/logs/run_v1_2_20_followup_queue_20260627_163936.pid`
- 初始状态：`complete=1 missing=11 expected=12`，队列正在等待 random refresh 完成。

## 5. 水相-only 与土壤-only B/C/E 基线

目标：证明当前深度框架在单独水相和单独土壤域内也有建模价值，再与迁移线比较。

数据表：

- aquatic-only：`aggregated_task_records_aquatic_ptox_qc`
- soil-only：`aggregated_task_records_soil_ptox_qc`

划分：

- B：random 8:2
- C：chemical-holdout 8:2
- E：random 5-fold

建议每个域跑 5-seed ensemble：

- seeds：`42 1042 2042 3042 4042`
- 模型：`full`
- 训练参数尽量沿用主线框架；
- 不启用 `tanimoto_to_finetune`，因为单域基线没有 aquatic source -> soil finetune 的迁移权重定义。

已新增版本化 launcher：

- `scripts/run_v1_2_19_single_domain_bce_remote.sh`
- 可选顺序队列入口：`scripts/run_v1_2_20_followup_queue_remote.sh`
- 输出根目录：`outputs/experiments/v1_2_19_single_domain_bce_remote`
- 汇总目录：`outputs/experiments/v1_2_19_single_domain_bce_remote_summary`

远端 smoke：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
bash scripts/run_v1_2_19_single_domain_bce_remote.sh smoke
```

远端正式矩阵：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
bash scripts/run_v1_2_19_single_domain_bce_remote.sh matrix
```

当前状态：脚本已同步远端并通过 `bash -n`，但完整矩阵尚未启动，建议等待 random 5-seed refresh 完成。

## 6. 完成判据

每一批实验完成后至少检查：

- 所有 run 的 `manifest.json`、`metrics.csv`、`predictions.csv`、`history.csv` 存在；
- `audit_summary.csv` 或等价完整性检查无缺失；
- chemical-holdout / chemical-group split 中 CAS train/test overlap 为 0；
- summary 表同时输出 overall、family、30-task 和 35-task；
- `PROJECT_STATUS.md` 与 `docs/experiment_registry.csv` 更新。
