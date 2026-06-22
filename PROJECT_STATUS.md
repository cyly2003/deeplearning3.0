# Project Status

更新时间：2026-06-22

## 2026-06-22 v1.2.6 authority-based toxicity binning 矩阵

- 目标：在当前框架内加入默认关闭的权威阈值毒性分箱辅助任务，并围绕迁移/不迁移、不同土壤样本比例和最佳既有超参组合设计新矩阵。
- 分支/代码状态：
  - 当前分支：`codex/authority-toxicity-binning-matrix`。
  - 新增 `configs/toxicity_bins.authority_v1.yaml`，按 EPA aquatic、OECD TG 207 soil screening scaffold、EPA terrestrial oral 分类记录阈值来源。
  - 新增 `qsar_tl/training/toxicity_binning.py`：为每个样本写入 `toxicity_bin_*` 字段；`water_mol_l` 优先用 `standard_value_mg_l`，否则用 RDKit/cache 的 `MolWt` 将 `standard_value_mol_l` 转为 `mg/L` 后分箱。未能分箱样本仍参与主回归，只从辅助 CE loss 中 mask。
  - `EcotoxMultiTaskNetwork` 新增默认关闭的 toxicity-bin auxiliary logits；训练 loss 为原 weighted Huber regression + `loss_weight * CE(toxicity_bin)`，batch 无 eligible bin 时 CE 跳过。
  - 训练输出新增 `toxicity_bin_metrics.csv` 与 `toxicity_bin_boundary_audit.csv`；`manifest.json` 记录 eligible、boundary、conversion/status counts；`predictions.csv` 保留分箱和单位换算字段。
  - 新增远端矩阵脚本：`scripts/run_v1_2_6_authority_binning_matrix_remote.sh`，支持 `smoke/core/hpo/all`；新增汇总脚本：`scripts/summarize_deep_runs.py`。
- 本地/远端验证：
  - 本地：`E:\TOOLS\anaconda\python.exe -m pytest tests\test_toxicity_binning.py tests\test_modeling_shapes.py tests\test_deep_experiment_cache.py -q`，`38 passed`。
  - 远端：`bash -n scripts/run_v1_2_6_authority_binning_matrix_remote.sh` + 同一 pytest，`38 passed`。
- 远端 smoke 已通过：
  - 输出根目录：`outputs/experiments/v1_2_6_authority_binning_matrix_remote`。
  - soil smoke：`v1.2.6_smoke_soil_f100_authority_bin_aux_lw005/deep/full/SoilPtoxQC2_C_low_f100`，1 epoch，必需文件无缺失，eligible bin 样本 11,587，boundary 样本 1,701，eval aquatic=0。
  - transfer smoke：`v1.2.6_smoke_transfer_f20_source_alpha1_authority_bin_aux_lw005/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`，1 epoch pretrain + 1 epoch finetune，必需文件无缺失，eligible bin 样本 158,728，boundary 样本 17,933，eval aquatic=0。
  - `water_mol_l -> mg/L` 转换已实际发生：soil smoke 中 5,137 行，transfer smoke 中 22,240 行。
- 已启动 core 矩阵后台队列：
  - 远端脚本：`scripts/run_v1_2_6_authority_binning_matrix_remote.sh core`。
  - 运行状态记录：`outputs/logs/run_v1_2_6_authority_binning_matrix_times.csv`。
  - core 队列 7 个正式 run 均已完成，exit=0；另有 2 个 smoke run，exit=0。
  - 远端汇总目录：`outputs/experiments/v1_2_6_authority_binning_matrix_remote_summary`。
  - 汇总文件：`focus_summary.csv`、`family_summary.csv`、`effect_level_summary.csv`、`toxicity_bin_summary.csv`、`audit_summary.csv`、`best_by_split.csv`、`common_task_summary.csv`、`common_task_comparison.csv`。
- core audit：
  - 9 个 smoke/core run 的 `required_files_present=True`，`aquatic_eval_rows=0`。
  - eligible bin 覆盖：soil f20 2,623；soil f100/fullC 11,587；transfer f20 158,728；transfer f50 166,751；transfer f100 172,773。
- core ECx/NOEC/LOEC focus test 指标：
  - `transfer_f20_source_alpha1_authority_bin_aux_lw005`：n=2,557，任务头=25，R2=0.2978，RMSE=1.5081，MAE=1.1478。
  - `transfer_f20_source_alpha1_effect025_authority_bin_aux_lw005`：n=2,557，任务头=25，R2=0.2119，MAE=1.2009；说明 authority-bin 与 effect-level beta0.25 叠加不稳，会拖累 NOEC/LOEC。
  - `transfer_f50_source_alpha1_authority_bin_aux_lw005`：n=2,591，任务头=29，R2=0.3971，MAE=1.0713。
  - `transfer_f100_source_alpha1_authority_bin_aux_lw005`：n=2,594，任务头=30，R2=0.4770，MAE=0.9993。
  - `soil_f20_authority_bin_aux_lw005_augN001`：n=1,407，任务头=6，R2=0.0342，MAE=1.2815；明显弱于既有 f20 soil-only/transfer。
  - `soil_f100_authority_bin_aux_lw005_eff050`：n=2,266，任务头=17，R2=0.4500，MAE=1.0200；弱于 v1.2.5 f100 最佳 MAE=0.9897。
  - `soil_fullC_authority_bin_aux_lw005_core`：n=2,266，任务头=17，R2=0.4677，MAE=1.0036；与 v1.2.5 fullC 最佳 MAE=1.0013 基本持平。
- 共同任务头公平对比：
  - f20 common 6-task：soil f20 MAE=1.2815/R2=0.0342；transfer f20 authority-bin MAE=1.1637/R2=0.2467。
  - f20 common 6-task + effect0.25：transfer MAE=1.2252/R2=0.1278，差于不加 effect0.25。
  - f100 common 17-task：soil f100 MAE=1.0200/R2=0.4500；transfer f100 authority-bin MAE=0.9972/R2=0.4705。
  - fullC soil vs transfer f100 common 17-task：soil fullC MAE=1.0036/R2=0.4677；transfer f100 authority-bin MAE=0.9972/R2=0.4705。
- 当前判断：
  - authority-bin auxiliary loss 对 `source_weighting=tanimoto_to_finetune alpha=1.0` 的迁移路线有实质收益：f20 test MAE 从上一轮 `matrix_source_tanimoto_alpha1` 的 1.1634 降到 1.1478，R2 从 0.2785 提到 0.2978。
  - authority-bin 对 soil-only fullC 基本中性，对 soil-only f20/f100 不如上一轮 best；因此它更像是迁移阶段的辅助结构正则，而不是 soil-only 的通用提升。
  - 下一步 HPO 不应继续强化 effect-level beta，而应围绕 `toxicity_binning.loss_weight` 做轻量敏感性。
- HPO 队列：
  - 已完成：`bash scripts/run_v1_2_6_authority_binning_matrix_remote.sh hpo`，完成时间 2026-06-22 21:02:27 +08:00。
  - 日志：`outputs/logs/run_v1_2_6_authority_binning_matrix_hpo_20260622_181645.log`。
  - HPO + core 总计 19 个 run；`audit_summary.csv` 核验 `required_files_present=True`、`aquatic_eval_rows=0`、`exit_code=0`。
  - HPO 内容：loss_weight `0.025/0.10` 对 soil f100/fullC、transfer f20/f100 做敏感性，并补跑 transfer f20 + effect0.25 的 lw0025/lw010 联合对照。
- HPO 最终结论：
  - f20 transfer 最优仍是默认 `toxicity_binning.loss_weight=0.05`：test n=2,557，R2=0.2978，MAE=1.1478；`0.025` 和 `0.10` 分别为 MAE=1.1812/R2=0.2588、MAE=1.1827/R2=0.2534。
  - f100 transfer 最优为 `toxicity_binning.loss_weight=0.025`：test n=2,594，R2=0.4978，MAE=0.9768；优于默认 0.05 的 MAE=0.9993/R2=0.4770，也略优于 v1.2.5 soil-only f100 最佳 MAE=0.9897/R2=0.4851。
  - soil-only 中 authority-bin HPO 未超过 v1.2.5 soil-only 上限：soil f100 最好为 lw005/lw010 约 MAE=1.020-1.022，fullC 最好仍是 lw005 MAE=1.0036/R2=0.4677。
  - effect-level beta0.25 与 authority-bin 叠加仍不建议作为主策略：f20 + effect0.25 最好是 lw010，test MAE=1.1532/R2=0.2766；MAE 接近但仍差于不加 effect-level 的 1.1478/R2=0.2978，且默认 lw005 叠加会明显退化到 MAE=1.2009/R2=0.2119。
  - 共同任务公平对比：f20 common 6-task 最优仍是 transfer f20 authority lw005，MAE=1.1637/R2=0.2467；f100 common 17-task 最优为 transfer f100 authority lw0025，MAE=0.9827/R2=0.4820；fullC soil vs transfer f100 common 17-task 中，transfer f100 lw0025 也优于 soil fullC authority lw005（MAE=0.9827 vs 1.0036）。
  - family 分层中，transfer f100 lw0025 的 ECx/LOEC/NOEC test MAE 分别为 0.8491/1.0098/1.0065，R2 分别为 0.5654/0.5088/0.4557；相对 f20 lw005 的 1.0183/1.1704/1.1891 和 0.3707/0.3226/0.2375 明显改善。
  - toxicity-bin 分层显示当前 test 样本主要按水相 mg/L 阈值进入 `water_*` bin；bin 内 R2 因硬分箱后目标范围变窄常为负，分层诊断应以 MAE 和边界样本数量为主。`water_very_high_toxicity` 仍是高误差层，f100 lw0025 non-boundary MAE=1.5061，f20 lw005 non-boundary MAE=2.2165。
  - 下一轮建议：迁移主线保留 `source_weighting=tanimoto_to_finetune alpha=1.0` + authority-bin aux；f20 用 loss_weight=0.05，f100/fullC 方向优先试 loss_weight=0.025；暂不继续 Phase 3 soft-expert 或 effect-level 叠加，除非先做针对高毒 bin 的专门误差修正。

## 2026-06-22 v1.2.5 soil-only upper-bound 矩阵

- 目标：不使用水相预训练，将 `aggregated_task_records_soil_ptox_qc` 土壤样本带入当前深度框架，放大 soil-only 实验矩阵以估计当前框架下的土壤单域性能上限，并报告运行时间。
- 分支/代码状态：
  - 当前分支：`codex/workspace-cleanup-20260622`。
  - 发现传统模型基线存在目标字段泄漏风险：新版聚合字段 `target_value_weighted_mean`、`target_value_unweighted_median`、`target_value_weighted_std` 未被排除，导致初次传统基线出现近零 MAE/近 1 R2 的不可信结果。
  - 已修补 `qsar_tl/training/baseline.py`：排除所有 `target_value*` 与 `tox_value*` 前缀字段；已补充 `tests/test_baseline_models.py` 定向测试。
  - 测试：本地 `E:\TOOLS\anaconda\python.exe -m pytest tests\test_baseline_models.py -q` 通过，远端同一测试也通过，均为 `7 passed`。
- 运行矩阵：
  - 深度模型 99 个 run：`SoilPtoxQC2_C_low_f20`、`SoilPtoxQC2_C_low_f100`、`SoilPtoxQC2_C_chemical_holdout_8_2` 三个 split。
  - 核心网格：learning rate `0.0003/0.0005/0.0008` × dropout `0.05/0.10/0.20` × weight decay `1e-6/9.856751793848817e-06/3e-5`，`epochs=50`，early stopping patience 15。
  - 附加策略：effect-level weighting beta `0.25/0.50`、数值噪声增强 `0.01/0.03`、80 epoch 长跑对照。
  - 修补后 no-leakage 传统基线 18 个 run：RandomForest、ExtraTrees、XGBoost、LightGBM、HistGradientBoosting、ElasticNet × 3 个 split；MLP 因 sklearn `_best_coefs` 异常未纳入 no-leakage 重跑。
- 运行时间：
  - 主矩阵日志：`outputs/logs/run_v1_2_5_soil_only_upper_bound_20260622.log`。
  - 主矩阵墙钟：2026-06-22 08:35:51 到 11:16:43，约 2:40:52；`run_times` 中 120 个计时任务，117 个成功，成功任务累计 9627 秒。其中深度模型 99 个成功，初次传统基线 18 个成功、3 个 MLP 失败。
  - no-leakage 传统基线日志：`outputs/logs/run_v1_2_5_soil_only_traditional_baselines_noleakage_20260622.nohup.log`。
  - no-leakage 传统基线墙钟：2026-06-22 11:35:57 到 11:43:35，约 7:38；18/18 成功，累计 457 秒。
- 输出与本地回收：
  - 远端深度完整结果：`outputs/experiments/v1_2_5_soil_only_upper_bound_remote`，约 746 MB；模型权重约 157 MB。本地仅回收 summary 与日志，完整 artifact 保留远端，避免无必要同步大文件。
  - 本地深度 summary：`outputs/experiments/v1_2_5_soil_only_upper_bound_remote_summary`。
  - no-leakage 传统基线完整小目录：`outputs/experiments/v1_2_5_soil_only_traditional_baselines_noleakage_remote`。
  - 关键新增汇总：`v1_2_5_soil_only_upper_bound_comparison.csv`、`v1_2_5_best_family_summary.csv`、`v1_2_5_best_effect_level_summary.csv`、`traditional_baseline_focus_comparable.csv`。
- 审计：
  - `deep_audit_summary.csv` 中所有 v1.2.5 soil-only run 的 `required_files_present=True`，`aquatic_eval_rows=0`。
  - 代表性 prediction medium counts：f20 为 `{"test|soil": 1407, "train|soil": 1603}`；f100/fullC 为 `{"test|soil": 2322, "train|soil": 11254}`。
- 深度模型最佳结果，均为 ECx/LOEC/NOEC focus test：
  - f20 最低 MAE：`augN001_f20_lr5e4_do010_wd1e5_e50`，n=1407，任务头=6，R2=0.1145，RMSE=1.6232，MAE=1.2310，Huber=0.8279。
  - f20 最高 R2：`core_f20_lr3e4_do005_wd1e5_e50`，n=1407，任务头=6，R2=0.1421，RMSE=1.5976，MAE=1.2407，Huber=0.8275。
  - f100 最佳：`eff050_f100_lr5e4_do010_wd1e5_e50`，n=2266，任务头=17，R2=0.4851，RMSE=1.3001，MAE=0.9897，Huber=0.6041。
  - full_C 最佳：`core_fullC_lr3e4_do020_wd1e5_e50`，n=2266，任务头=17，R2=0.4617，RMSE=1.3293，MAE=1.0013，Huber=0.6166。
- 与 v1.2.4/current transfer 对照：
  - f20：v1.2.5 最低 MAE=1.2310，未超过 v1.2.4 `soilonly_current_low_f20` 的 MAE=1.2084；也弱于迁移 common-task 最好结果，例如 `source_tanimoto_alpha1_effect_beta0p5` common f20 MAE=1.1762。说明低土壤样本下，水相预训练/迁移仍主要提供小样本补偿。
  - f100：v1.2.5 最佳 MAE=0.9897，优于 v1.2.4 `soilonly_current_low_f100` 的 MAE=1.0486，也明显优于 `matrix_source_tanimoto_alpha1` common f100 MAE=1.1680。
  - full_C：v1.2.5 最佳 MAE=1.0013，略优于 v1.2.4 `soilonly_current_full_C` 的 MAE=1.0095，R2 从 0.4456 提升到 0.4617。
- 分层结果：
  - f100 最佳 run：ECx MAE=0.8050、LOEC MAE=1.0331、NOEC MAE=1.0324；ECx/LOEC/NOEC R2 分别为 0.5994/0.4861/0.4377。
  - full_C 最佳 run：ECx MAE=0.8148、LOEC MAE=1.0377、NOEC MAE=1.0511；ECx/LOEC/NOEC R2 分别为 0.5985/0.4777/0.3904。
  - f20 最佳 run：ECx 只覆盖 `ECx_Growth`，MAE=1.0721；LOEC MAE=1.2127；NOEC MAE=1.2977。NOEC 仍是低样本下最难稳定的族。
  - ECx effect-level：f100/full_C 的 EC50 层 n=375，MAE 约 0.82-0.84，R2 约 0.53；EC10/EC25 样本量很小，单层 R2 波动较大，不能过度解释。
- no-leakage 传统基线可比口径：
  - f20 最佳为 HistGradientBoosting：R2=-0.0247，MAE=1.3250。
  - f100/fullC 最佳为 RandomForest：R2=0.3314，MAE=1.0796。
  - 传统模型不再出现不合理近零误差；在相同任务集合下低于深度模型，说明当前深度框架的上限主要来自多任务深度表征而不是树模型基线。
- 当前结论：
  - 土壤样本足够时，soil-only 在当前框架内的上限约为 MAE 1.0 log unit、R2 0.46-0.49；轻量 HPO 与 effect-level beta0.5 能进一步改善 f100。
  - 低样本 f20 未因更大 soil-only 网格而提升，反而略弱于 v1.2.4 和迁移共同任务结果；因此迁移学习的价值目前集中在低土壤样本/任务覆盖不足场景，而不是足量土壤样本下超过 soil-only。
  - 后续若继续探索上限，优先做 soil-only f20 的数据增强/任务重采样和按任务族的验证集调参；不建议再盲目扩大全局 LR/dropout/weight decay 网格。

## 2026-06-21 v1.2.4 soil-only current-framework 对照

- 目标：按当前深度框架单独训练土壤 pTox，不使用水相预训练、不传 finetune 参数、不启用 source/effect-level weighting，用于判断水相迁移相对土壤单域训练的真实增益。
- 已同步当前代码/配置到远端 `/home/easyai/DL1/ecotox_qsar_transfer`，并完成 3 个远端 run：
  - `soilonly_current_low_f20`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_low_f20`。
  - `soilonly_current_low_f100`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_low_f100`。
  - `soilonly_current_full_C`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_chemical_holdout_8_2`。
- 运行参数：`epochs=30`，`batch_size=512`，`learning_rate=0.0005`，`weight_decay=9.856751793848817e-06`，`scheduler=cosine`，`dropout=0.1`，`target_standardization=per_task_target`，`metric_min_n=5`，`source_weighting=none`，`effect_level_weighting=false`，`swa=false`。
- 运行脚本和日志：
  - 本地脚本：`outputs/logs/run_v1_2_4_soil_only_current_framework_20260621.sh`
  - 远端/本地日志：`outputs/logs/run_v1_2_4_soil_only_current_framework_20260621.log`
  - 远端完成时间：`[done] 2026-06-21T19:16:25+08:00 v1.2.4 soil-only current-framework matrix`
- 输出与本地回收：
  - 远端完整结果：`outputs/experiments/v1_2_4_soil_only_current_framework_remote`
  - 本地 summary：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_summary`
  - 本地无模型 artifact 包：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_artifacts_no_model.tar.gz`
  - 本地完整解包副本：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_artifacts_extracted/v1_2_4_soil_only_current_framework_remote`
  - 汇总文件：`focus_summary.csv`、`common_task_comparison.csv`、`family_summary.csv`、`effect_level_summary.csv`、`audit_summary.csv`、`summary_manifest.json`
- 产物和介质审计：
  - 3 个 run 均生成 `predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `soilonly_current_low_f20` 预测行：test/soil 1,407；train/soil 1,603；评估集 aquatic=0。
  - `soilonly_current_low_f100` 预测行：test/soil 2,322；train/soil 11,254；评估集 aquatic=0。
  - `soilonly_current_full_C` 预测行：test/soil 2,322；train/soil 11,254；评估集 aquatic=0。
  - 3 个 run 的 `source_weighting.applied=false`，`effect_level_weighting.applied=false`。
- ECx/NOEC/LOEC 合并 test 指标，按 `predictions.csv` 重新计算：
  - `soilonly_current_low_f20`：n=1,407，任务头=6，R2=0.1848，RMSE=1.5573，MAE=1.2084。
  - `soilonly_current_low_f100`：n=2,266，任务头=17，R2=0.4202，RMSE=1.3796，MAE=1.0486。
  - `soilonly_current_full_C`：n=2,266，任务头=17，R2=0.4456，RMSE=1.3490，MAE=1.0095。
  - 对照：`matrix_source_tanimoto_alpha1` 全量 focus test 为 n=2,557，任务头=25，R2=0.2785，RMSE=1.5286，MAE=1.1634；`source_tanimoto_alpha1_effect_beta0p25` 为 R2=0.2778，RMSE=1.5293，MAE=1.1594。
- 共同任务头公平对比：
  - 与 `soilonly_current_low_f20` 的 6 个任务头对齐时，`matrix_source_tanimoto_alpha1`：R2=0.1935，RMSE=1.5483，MAE=1.1966；`source_tanimoto_alpha1_effect_beta0p25`：R2=0.1928，RMSE=1.5490，MAE=1.1913；`soilonly_current_low_f20`：R2=0.1848，RMSE=1.5573，MAE=1.2084。说明 f20 低样本下 soil-only 接近但仍略弱于迁移，且任务覆盖明显少于迁移模型。
  - 与 `soilonly_current_low_f100` 的 17 个任务头对齐时，`matrix_source_tanimoto_alpha1`：R2=0.2746，RMSE=1.5335，MAE=1.1680；`source_tanimoto_alpha1_effect_beta0p25`：R2=0.2720，RMSE=1.5363，MAE=1.1664；`soilonly_current_low_f100`：R2=0.4202，RMSE=1.3796，MAE=1.0486；`soilonly_current_full_C`：R2=0.4456，RMSE=1.3490，MAE=1.0095。
- family 分层 test：
  - `soilonly_current_full_C`：ECx MAE=0.8112，LOEC MAE=1.0547，NOEC MAE=1.0565。
  - `soilonly_current_low_f100`：ECx MAE=0.8607，LOEC MAE=1.0777，NOEC MAE=1.1057。
  - `matrix_source_tanimoto_alpha1`：ECx MAE=0.9814，LOEC MAE=1.2069，NOEC MAE=1.2098。
  - `source_tanimoto_alpha1_effect_beta0p25`：ECx MAE=1.0308，LOEC MAE=1.1681，NOEC MAE=1.2136。
- 当前判断：
  - f20 小土壤样本下，水相迁移仍有价值：它保留更多任务头，并在共同 6 个任务头上略优于 soil-only。
  - 一旦土壤训练样本提高到 f100/full_C，同一 current-framework 的 soil-only 明显优于当前水相预训练迁移策略，说明负迁移仍存在；水相预训练的主收益更像是低样本任务覆盖和小样本补偿，而不是在足量土壤 pTox 上提供更强表征。
  - 下一步若继续提升，应优先做更严格 source filtering / curriculum annealing，而不是继续增大 effect beta、source alpha 或单纯延长 finetune。

## 2026-06-21 v1.2.3 source Tanimoto + effect-level weighting 矩阵

- 已在分支 `codex/effect-level-weighting-source-tanimoto` 上新增默认关闭训练开关：
  - `training.effect_level_weighting` / CLI `--effect-level-weighting --effect-level-weighting-beta <beta>`。
  - 仅对实际训练索引 `train + finetune_train` 中 `ECx/LCx/ICx/LDx` 且存在 `effect_level_x` 的样本计算 `frequency^(-beta)` 权重；在 `0.5-3.0` 约束内归一化到均值 1，并与已有 `source_weighting` 样本权重相乘。
  - validation、finetune_validation、test 不参与频率统计，也不在 loss 评估中使用样本权重。
  - `manifest.json` 新增 `effect_level_weighting`；`history.csv` 新增 source/effect-level/final sample weight 审计列。
- 本地验证：
  - `E:\TOOLS\anaconda\python.exe -m compileall qsar_tl\training\deep_experiment.py qsar_tl\training\train.py tests\test_deep_experiment_cache.py` 通过。
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_deep_experiment_cache.py tests\test_remote_execution_config.py -q` 通过，32 passed；本地 Torch 仍有既有 NumPy DLL 初始化 warning。
  - `configs\experiment.remote.easyai.yaml` 配置校验通过。
- 远端 smoke 已通过：
  - 输出目录：`outputs/experiments/v1_2_3_effect_level_weighting_smoke_remote/v1.2.3_smoke_source_tanimoto_effect_beta0p5/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 参数：1 epoch pretrain + 1 epoch finetune，`source_weighting=tanimoto_to_finetune alpha=1.0`，`effect_level_weighting beta=0.5`。
  - 必需文件均生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `effect_level_weighting` 实际加权 139,736 个训练样本，effect weight 范围约 0.6712-3.0，均值 1.0；与 source weighting 相乘后的 final sample weight 均值约 0.9966。
  - 介质审计：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
- 已完成同预算远端正式矩阵并拉回本地汇总：
  - 远端/本地汇总目录：`outputs/experiments/v1_2_3_effect_level_weighting_matrix_remote_summary`
  - 队列脚本/日志：`outputs/logs/run_v1_2_3_effect_level_weighting_matrix_20260621_151249.sh`、`outputs/logs/run_v1_2_3_effect_level_weighting_matrix_20260621_151249.log`
  - 汇总文件：`focus_summary.csv`、`focus_comparison_vs_baselines.csv`、`family_summary.csv`、`effect_level_summary.csv`、`manifest_summary.csv`、`audit_summary.json`
  - 5 个矩阵 run 必需文件均无缺失，`finetune`、`finetune_validation`、`test` 的 aquatic 混入均为 0。
- ECx/NOEC/LOEC 合并指标，按 `predictions.csv` 重新计算：
  - `source_tanimoto_alpha1_effect_beta0p25` test 最好：R2=0.2778，RMSE=1.5293，MAE=1.1594；相对 `matrix_source_tanimoto_alpha1` 的 test MAE=1.1634 小幅改善约 0.0040 log unit，但 R2 略低 0.0007。
  - `matrix_source_tanimoto_alpha1` 仍是 R2/RMSE 最稳参照：test R2=0.2785，RMSE=1.5286，MAE=1.1634。
  - `source_tanimoto_alpha1_effect_beta0p75` test：R2=0.2767，RMSE=1.5306，MAE=1.1677，整体接近 alpha1 原策略。
  - `source_tanimoto_alpha1_effect_beta0p5` test：R2=0.2405，RMSE=1.5683，MAE=1.1938，验证集不优且 test 明显退化。
  - `source_tanimoto_alpha1_effect_beta0p5_swa_finetune15` test：R2=0.2294，RMSE=1.5798，MAE=1.1874；SWA 改善 ECx 但牺牲 NOEC/LOEC，combined test 不建议保留。
  - 可选 `source_tanimoto_alpha1p5_effect_beta0p5` finetune_validation 最好：R2=0.7284，RMSE=0.8641，MAE=0.6215；但 test R2=0.2489，MAE=1.1934，提示更强 source alpha 更贴合微调验证集而非化学 holdout test。
- family/effect-level 层面 test：
  - beta0.25 改善 LOEC：MAE=1.1681，相对 `matrix_source_tanimoto_alpha1` 的 1.2069 改善；NOEC 基本持平略差：MAE=1.2136 vs 1.2098；ECx 明显变差：MAE=1.0308 vs 0.9814。
  - beta0.75 改善 ECx 和 LOEC：ECx MAE=0.9891，LOEC MAE=1.1826；但 NOEC 退化：MAE=1.2403。
  - SWA 组合 ECx 最好：MAE=0.9484；但 LOEC/NOEC 退化到 1.2290/1.2636，是 combined test 变差主因。
  - effect-level：beta0.25 对小样本 EC10/EC25 明显改善（EC10 MAE=1.7943、EC25 MAE=1.4874），但 EC50 退化到 MAE=0.9995；beta0.75/beta0.5 对 EC50 更好（约 0.942/0.940），但 EC10/EC25 更差。
- 当前判断：
  - 新的候选最佳可记为 `source_weighting=tanimoto_to_finetune alpha=1.0 + effect_level_weighting beta=0.25`，但收益很小，属于 MAE 方向的细微改善而非全面胜出。
  - 若主目标优先 chemical-holdout combined MAE，可保留 beta0.25 作为下一轮候选；若优先稳健 R2/RMSE 和 ECx 单族，上一轮 `matrix_source_tanimoto_alpha1` 仍更均衡。
  - 不建议继续增大 effect beta 或 source alpha；这些设置会改善 finetune_validation 或 EC50，但削弱 NOEC/LOEC 和 test 泛化。

## 2026-06-20 v1.2.2 第一批迁移优化矩阵启动

- 已在分支 `codex/transfer-matrix-coral-sourceweight-swa` 上实现第一批矩阵的默认关闭训练开关：
  - `training.source_weighting` / CLI `--source-weighting-method tanimoto_to_finetune --source-weighting-alpha <alpha>`：按土壤 finetune 训练样本的最大 Tanimoto 相似度重加权水相 source 训练样本，权重均值归一到 1。
  - `training.domain_alignment` / CLI `--domain-alignment-method coral --domain-alignment-weight <weight>`：在 shared representation 上加入 CORAL 域对齐损失，默认 pretrain 阶段用 water source batch 对齐 soil finetune reference batch。
  - `training.swa` / CLI `--swa --swa-start-epoch <epoch>`：在指定阶段后半程累积 averaged weights，当前默认用于 finetune 阶段。
  - `history.csv` 新增 `mean_task_loss`、`mean_alignment_loss`、`alignment_steps`、`swa_updates`；`manifest.json` 新增 `source_weighting`、`domain_alignment`、`swa` 记录。
- 本地验证：
  - `compileall qsar_tl/modeling/network.py qsar_tl/training/deep_train.py qsar_tl/training/deep_experiment.py qsar_tl/training/train.py tests/test_deep_experiment_cache.py` 通过。
  - `pytest tests/test_deep_experiment_cache.py tests/test_modeling_shapes.py -q` 通过，28 passed；本地 Torch 仍有既有 NumPy DLL 初始化 warning。
- 已同步代码到远端 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端组合 smoke 已通过：
  - 输出目录：`outputs/experiments/v1_2_2_matrix_smoke_remote/v1.2.2_smoke_coral_sourceweight_swa/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 参数：1 epoch pretrain + 1 epoch finetune，`CORAL=0.01`、`source_weighting=tanimoto_to_finetune alpha=1.0`、`SWA start_epoch=1`。
  - 必需文件均生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `source_weighting` 实际应用到 241,876 个 aquatic train 样本，soil finetune reference 为 2,193；权重范围约 0.6716-1.3431，均值 1.0。
  - CORAL pretrain reference 为 2,193 个 soil finetune 样本，pretrain alignment steps=945；SWA finetune updates=1 且 `applied=true`。
  - `predictions.csv` 介质计数：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
  - `split_medium_audit.csv` 仍记录剔除错配行：finetune aquatic 238、test aquatic 300、train soil 1,111。
- 已启动同预算远端正式矩阵后台队列：
  - 队列脚本：`outputs/logs/run_v1_2_2_transfer_matrix_20260620_115110.sh`。
  - 队列总日志：`outputs/logs/v1_2_2_transfer_matrix_20260620_115110.log`。
  - 队列进程：bash PID `1948322`；当前第一项为 `matrix_coral_0p01`。
  - 公共预算：pretrain epochs=30，finetune epochs=60，batch_size=512，learning_rate=0.0005，weight_decay=9.856751793848817e-06，dropout=0.1，target_standardization=`per_task_target`，finetune_lr=0.0003082636455810776，finetune_scheduler=`reduce_on_plateau`，finetune_validation_fraction=0.2。
  - 输出根目录：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote`。
  - 队列顺序：`matrix_coral_0p01`、`matrix_coral_0p05`、`matrix_source_tanimoto_alpha1`、`matrix_swa_finetune15`、`matrix_coral_0p01_source_alpha1`。
  - `GradNorm/PCGrad` 暂未进入本队列，因为它会明显改动多任务梯度路径，需单独分支或后续小步实现。

## 2026-06-21 v1.2.2 第一批迁移优化矩阵结果

- 远端正式矩阵队列已全部完成；汇总目录已拉回本地：
  - 远端：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote_summary`
  - 本地：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote_summary`
  - 关键文件：`focus_summary.csv`、`family_summary.csv`、`effect_level_summary.csv`、`manifest_summary.csv`、`audit_summary.json`
- split audit：
  - 所有 v1.2.2 run 的 `finetune`、`finetune_validation`、`test` 均为 soil，`aquatic_in_soil_eval=0`。
  - 介质计数一致：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685。
- ECx/NOEC/LOEC 合并指标，按 `predictions.csv` 重新计算：
  - 上一轮 `v1.2.1_best_rerun` test：R2=0.1700，RMSE=1.6396，MAE=1.2311；finetune_validation：R2=0.6735，RMSE=0.9475，MAE=0.6844。
  - `matrix_source_tanimoto_alpha1` test 最好：R2=0.2785，RMSE=1.5286，MAE=1.1634；相对 v1.2.1 best，test MAE 改善约 0.0677 log unit，R2 提升约 0.1085。
  - `matrix_swa_finetune15` test 次优：R2=0.2340，RMSE=1.5750，MAE=1.1967。
  - `matrix_coral_0p01_source_alpha1` test：R2=0.2086，RMSE=1.6010，MAE=1.2196，小幅优于 v1.2.1 best。
  - `matrix_coral_0p01` test：R2=0.1696，RMSE=1.6399，MAE=1.2560；与 v1.2.1 best 基本持平或略差。
  - `matrix_coral_0p05` test：R2=0.1912，RMSE=1.6185，MAE=1.2577；R2/RMSE 略改善但 MAE 变差。
  - finetune_validation 最好是 `matrix_coral_0p01`：R2=0.7170，RMSE=0.8822，MAE=0.6219；但该改善未能转化到 test，提示 CORAL 单独使用可能更贴合微调验证集而非化学 holdout test。
- family 层面 test：
  - ECx：`matrix_source_tanimoto_alpha1` 最好，R2=0.4359，RMSE=1.2974，MAE=0.9814；比 v1.2.1 best 的 MAE=1.0404 改善。
  - LOEC：`matrix_source_tanimoto_alpha1` 最好，R2=0.2695，RMSE=1.5891，MAE=1.2069；比 v1.2.1 best 的 MAE=1.2598 改善。
  - NOEC：`matrix_source_tanimoto_alpha1` 最好，R2=0.2152，RMSE=1.5723，MAE=1.2098；比 v1.2.1 best 的 MAE=1.2960 改善，是本轮最重要收益。
- effect-level 层面 test：
  - EC50：`matrix_swa_finetune15` MAE=0.9221、R2=0.4414；`matrix_source_tanimoto_alpha1` MAE=0.9332、R2=0.4448；二者均优于 v1.2.1 best 的 MAE=0.9795。
  - EC10/EC25 样本量小且仍不稳定；`matrix_source_tanimoto_alpha1` 对 EC10 MAE=1.8702，相比 v1.2.1 best 的 2.0121 略改善，但 R2 仍为负。
  - NOEC/LOEC 的 `effect_level=none` 层由 `matrix_source_tanimoto_alpha1` 同时最佳。
- 当前判断：
  - 第一批里应优先保留 `source_weighting=tanimoto_to_finetune, alpha=1.0` 作为新的候选最佳迁移策略。
  - SWA 对 EC50 有帮助，但对 EC10/EC25 和 NOEC/LOEC 不如 source weighting 稳。
  - CORAL 单独或与 source weighting 组合在 finetune_validation 上较好，但 test 泛化不足，下一步不建议继续单纯增大 CORAL weight。

## 2026-06-17 v1.2.1 效应水平、介质切分和特征 z-score 纠偏修正

- 已修正 `effect_level_x` 的建模与报告逻辑：
  - `ECx/LCx/ICx/LDx` 的效应水平数字继续作为输入，但新增 `effect_level_x_fraction`、`effect_level_x_log1p`、`effect_level_x_present` 派生通道。
  - 深度网络新增 effect-level 专门分支，避免效应水平只被混在普通数值特征中。
  - `predictions.csv` 保留 `effect_level_x`，并新增 `effect_level_metrics.csv` 与 `effect_level_metrics_filtered.csv`，用于按 EC10/EC50/LC50 等水平分层诊断。
- 已修正迁移 split 加载时的介质混入风险：
  - `load_split_frame` 会按 split assignment 的 `group_key` 校验介质域，`soil_*` 只允许 soil 行，`aquatic` 只允许 aquatic 行。
  - 训练输出新增 `split_medium_audit.csv`，记录各 split 的 aquatic/soil 数量以及被剔除的跨介质 join 行。
- 已新增核心指标过滤：
  - 原始 `metrics.csv` 保留所有分组。
  - 新增 `metrics_filtered.csv`，默认只保留 `n >= 5` 且 R2 有定义的分组，避免 `n=1/2` 的 NaN R2 子任务进入主结论和 HPO 目标。
- 已新增特征 z-score 自动纠偏：
  - `training.feature_zscore_correction.enabled=true` 默认开启，阈值默认 6。
  - 仅使用训练/微调训练样本拟合均值、标准差和截断边界，验证集、测试集和扰动评估复用同一纠偏参数。
- 本地验证：
  - `compileall qsar_tl scripts tests` 通过。
  - `pytest tests/test_deep_experiment_cache.py tests/test_modeling_shapes.py tests/test_task_mapping.py tests/test_medium_domain_tables.py tests/test_remote_execution_config.py tests/test_aquatic_soil_adaptation_split.py` 通过，49 passed。
  - 本地根目录 `ecotox_clean.sqlite` 无 `split_assignments`，真实训练 smoke 需在远端派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 同步代码后执行。

## 2026-06-17 v1.2.1 远端 smoke 与 best trial 复跑

- 已同步最新代码到远端 `/home/easyai/DL1/ecotox_qsar_transfer`，使用远端派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
- 远端 smoke 已通过：
  - source table：`aggregated_task_records_aquatic_soil_ptox_qc`。
  - split：`M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 预算：1 epoch pretrain + 1 epoch finetune，`batch_size=256`，`device=cuda:0`。
  - 输出目录：`outputs/experiments/v1_2_1_smoke_remote/v1.2.1_smoke_medium_audit_zscore/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 必需文件均在远端生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`。
  - `predictions.csv` 介质计数：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
  - `split_medium_audit.csv` 显示被校验剔除的错配行：finetune 中 aquatic 238、test 中 aquatic 300、train 中 soil 1,111。
- 已按上一轮 HPO best trial 010 同预算复跑当前最佳迁移实验：
  - 参数：pretrain epochs=30，finetune epochs=60，batch_size=512，learning_rate=0.0005，weight_decay=9.856751793848817e-06，dropout=0.1，target_standardization=`per_task_target`，finetune_lr=0.0003082636455810776，finetune_scheduler=`reduce_on_plateau`，finetune_validation_fraction=0.2。
  - 输出目录：`outputs/experiments/v1_2_1_best_retrain_remote/v1.2.1_best_trial_010_effect_level_medium_zscore_rerun/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 远端日志：`outputs/logs/v1_2_1_best_retrain_20260617_212406.log`。
  - 训练结果：pretrain 完成 30 epoch；finetune 在第 30 个 finetune epoch 早停；最佳全局 epoch=50，即 finetune 第 20 epoch，monitor loss=0.093797。
  - 必需文件均生成，且 `predictions.csv` 介质计数同 smoke：finetune/finetune_validation/test 全为 soil，aquatic=0。
- 已生成本地对比汇总目录：`outputs/experiments/v1_2_1_best_retrain_remote_summary`。
  - `comparison_summary_v1_2_1.csv`：上一轮 baseline、上一轮 HPO best trial 010 与本轮复跑的 split 汇总。
  - `comparison_delta_vs_hpo_best_trial_010.csv`：本轮复跑相对上一轮 HPO best 的差值。
  - `comparison_family_v1_2_1.csv` 与 `comparison_family_delta_vs_hpo_best_trial_010.csv`：ECx/NOEC/LOEC 在 test 与 finetune_validation 的家族级对比。
  - `effect_level_focus_v1_2_1.csv`：ECx/NOEC/LOEC 的 effect-level 分层汇总。
- 关键对比结论：
  - 主族 `ECx/NOEC/LOEC` 的 `finetune_validation`：R2 从 0.6186 升至 0.6735，但 MAE 从 0.6324 升至 0.6844，RMSE 从 0.8556 升至 0.9475，说明验证集方差解释略好但绝对误差更差。
  - 主族 `ECx/NOEC/LOEC` 的 `test`：R2 从 0.1849 降至 0.1700，MAE 从 1.1095 升至 1.2311，RMSE 从 1.4365 升至 1.6396；本轮修正后测试误差高于上一轮 HPO best。
  - family 层面：test ECx R2 从 0.2673 升至 0.4044，但 MAE 从 0.9480 升至 1.0404；test LOEC/NOEC 均下降，LOEC MAE 1.1373 -> 1.2598，NOEC MAE 1.1727 -> 1.2960。
  - effect-level 层面：test EC50 为主要 ECx 层级，n=441，R2=0.4174，MAE=0.9795；EC10 与 EC25 样本量较小且误差较高，分别为 n=29/MAE=2.0121、n=15/MAE=1.9282。NOEC/LOEC 无独立 effect level 数值，按 `none` 层汇总。

## 2026-06-16 v2.0.0 新清洗数据远端重训启动

- 已按新清洗派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 启动远端重训队列。
- 新版 RDKit/Morgan512 分子特征缓存已重建：
  - 文件：`outputs/features/molecular_features_rdkit_morgan512.jsonl`
  - `source_table=aggregated_task_records_qc`
  - `unique_smiles=7837`
  - `fallback_count=0`
- 远端同步已完成，远端项目目录仍为 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端预检已通过：
  - `PRAGMA quick_check` 返回 `ok`。
  - GPU 可用：RTX 4060 Ti，`cuda:0`。
  - v2 关键表规模：`aggregated_task_records_qc=477656`，`aggregated_task_records_aquatic_ptox_qc=281435`，`aggregated_task_records_soil_ptox_qc=16020`，`aggregated_task_records_soil_mg_kg_qc=17506`，`aggregated_task_records_aquatic_soil_ptox_qc=297455`。
- 已生成新版 split：
  - `AquaticPtoxQC2_` A-F splits。
  - `SoilPtoxQC2_` A-F splits。
  - `SoilMgkgQC2_` A-F splits。
  - 低土壤样本迁移 split：`M_v2_aquatic_to_soil_ptox_adapt_C_f10/f20/f50/f100`。
  - 配对 soil-only split：`SoilPtoxQC2_C_low_f10/f20/f50/f100`。
- 泄漏检查：
  - `AquaticPtoxQC2_C_chemical_holdout_8_2`、`SoilPtoxQC2_C_chemical_holdout_8_2`、`SoilMgkgQC2_C_chemical_holdout_8_2` 的 train/test 化合物分组重叠均为 0。
  - `AquaticPtoxQC2_F_chemical_adapt_7_2_1`、`SoilPtoxQC2_F_chemical_adapt_7_2_1`、`SoilMgkgQC2_F_chemical_adapt_7_2_1` 的 train 与 finetune/test 化合物分组重叠均为 0。
- Phase 0 smoke 已通过：
  - 水相 pTox B split，1 epoch，输出完整。
  - 土壤 pTox C split，1 epoch，输出完整。
  - 水相预训练到土壤 pTox f20，1 epoch pretrain + 1 epoch finetune，输出完整。
- 后台正式队列已进入 Phase 1：
  - 当前日志：`outputs/logs/v2_0_0_retrain_20260616_195035.log`。
  - 已完成：`v2.0.0_reclean_v2_aquatic_ptox_B_random/deep/full/AquaticPtoxQC2_B_random_8_2`，30 epoch，`best_epoch=30`。
  - 当前运行：`v2.0.0_reclean_v2_aquatic_ptox_C_chemical/deep/full/AquaticPtoxQC2_C_chemical_holdout_8_2`，30 epoch。
  - 调度脚本：`scripts/run_v2_0_0_retrain_remote.sh`。
- 队列后续自动执行：
  - Phase 1：水相 pTox、土壤 pTox、土壤 mg/kg 的 B/C full 30 epoch。
  - Phase 2：`f10/f20/f50/f100` 水相预训练 + 土壤 pTox 微调，以及配对 soil-only 对照。
  - Phase 3：按 Phase 2 test 主终点加权 MAE 自动选择最佳 f，并补跑 `finetune40_cosine` 与 `finetune20_constant` 诊断，同时输出 ECx/NOEC/LOEC 解释和 AD 分层指标。
- 待队列完成后：
  - 拉回 `outputs/experiments/v2_0_0_*`、`outputs/ad/v2_0_0` 和日志。
  - 汇总 R2、RMSE、MAE、Huber loss，并与 v1.0.0/v1.1.0 结果对比。
  - 生成正式 v2.0.0 实验总结文档。

## 2026-06-13 v1.1.0 迁移步长诊断与主终点解释

- 已完成远端 v1.1.0 迁移步长诊断，日志：`outputs/logs/v1_1_0_transfer_step_diagnostics_20260613_204650.log`。
- 实验目录：`outputs/experiments/v1_1_0_transfer_step_diagnostics`；三组诊断为：
  - `v1.1.0_迁移步长诊断_预训练30微调40_cosine`
  - `v1.1.0_迁移步长诊断_预训练30微调20_微调恒定学习率`
  - `v1.1.0_迁移步长诊断_预训练60微调20_cosine`
- 已新增训练调度参数：
  - `python -m qsar_tl.training.train --scheduler {none,cosine,reduce_on_plateau}`
  - `python -m qsar_tl.training.train --finetune-scheduler {none,cosine,reduce_on_plateau}`
  - `scripts/train_remote.ps1` 同步支持 `-Scheduler` 和 `-FinetuneScheduler`。
- 已新增解释汇总链路：
  - `scripts/analyze_endpoint_family_explanations.py`：按 ECx/NOEC/LOEC 输出测试误差分布，并选择代表任务运行 SHAP。
  - `scripts/summarize_transfer_step_diagnostics.py`：汇总步长诊断、终点分组误差和 SHAP Top 特征。
  - `scripts/explain_deep_model.py` 已修正 target standardization 后的解释尺度，现在 SHAP 解释的是反标准化后的原始 pTox 预测。
- 关键结果：
  - 基线 `30预训练+20微调 cosine` 的微调 loss：final 0.093646，min 0.091554。
  - `30预训练+40微调 cosine` 的微调 loss：final 0.051203，min 0.047837。
  - `30预训练+20微调 恒定LR` 的微调 loss：final/min 0.052589。
  - `60预训练+20微调 cosine` 的预训练 loss 从第 30 轮约 0.050387 继续降到第 60 轮 0.035730，但微调 final 仍为 0.091088。
- 主终点合并测试表现：
  - 基线 ECx/LOEC/NOEC 样本加权 MAE：1.1709。
  - `30预训练+40微调 cosine`：MAE 1.1327，当前最优。
  - `30预训练+20微调 恒定LR`：MAE 1.1405，接近最优。
  - `60预训练+20微调 cosine`：MAE 1.1591，仅小幅优于基线。
- 当前结论：微调阶段确实存在有效步长不足或 cosine 衰减过快的问题，但测试集误差不随训练 loss 单调改善；下一轮应优先比较微调调度、验证集早停和更稳健的迁移策略，而不是简单继续堆 epoch。
- 报告：`docs/v1.1.0_transfer_step_diagnostics.md`。

## 2026-06-12 土壤毒性预测与迁移学习重设计方案

- 已完成正式方案文档：`docs/soil_toxicity_transfer_redesign_plan.md`。
- 方案已固化以下共识：
  - 土壤毒性预测为主线，水相迁移用于学习分子构效知识并支撑土壤小样本预测。
  - 重新设计 QC 与聚合：endpoint 数字保留为 `effect_level_x`，robust z-score > 4 的明显离群值排除，跨文献采用最新文献优先、0.3 log unit 冲突阈值和年份权重加权平均。
  - 模型结构采用 shared molecular/taxon/context trunk + target-scale/medium adapter + endpoint task head。
  - 物种上下文改为 taxon 层级；`species_number` 只作为审计和消融字段，不作为正式主输入。
  - 化学应用域以 Williams leverage 为主、Tanimoto 为补充；物种应用域使用 taxon distance。
  - 不确定度第一版只输出模型不确定度和外推/应用域不确定度。
  - 数据划分同时覆盖 random、chemical holdout、AD-aware、low-soil-data transfer、target-scale-specific，并保留 A-F 兼容对照。
- 已在 `qsar_tl/data/task_tables.py` 的聚合键和聚合函数处加入显著注释，说明当前聚合不会混合 EC10/EC50 或不同量纲，但存在跨文献/跨 test batch 聚合风险。

## 2026-06-12 重设计方案第一批代码落地

- 已新增正式 QC 聚合链路：
  - `qsar_tl/data/reference_weighting.py`
  - `qsar_tl/data/qc_aggregation.py`
  - CLI：`python -m qsar_tl.cli build-qc-task-tables --config configs/experiment.example.yaml --db outputs/derived/modeling_dataset.sqlite`
- `wide_records -> target_records -> task_records` 已透传 `publication_year`，用于跨文献聚合时按最新文献优先加权。
- `task_records_qc` 输出每条任务记录的 `medium_domain`、`tox_qc_status`、`tox_qc_reason`、`robust_z`、`qc_group_n`、`reference_weight`。
- `aggregated_task_records_qc` 输出正式训练目标：
  - `target_value_median` 兼容旧训练入口，但在 `_qc` 表内实际承载文献/测试单元加权平均。
  - `target_value_unweighted_median` 保留原始中位数。
  - `cross_reference_conflict_flag` 和 `cross_reference_range` 用于审计跨文献冲突。
  - 同一 `reference_number + test_id` 先内部取中位数，再进行跨文献年份权重加权，避免同一文献多条记录放大权重。
- `scripts/build_medium_domain_tables.py` 已支持 `--table-suffix _qc`，可从 `aggregated_task_records_qc` 生成：
  - `aggregated_task_records_aquatic_ptox_qc`
  - `aggregated_task_records_soil_mg_kg_qc`
  - `aggregated_task_records_soil_ptox_qc`
  - 以及其他同尺度 `_qc` 子表。
- 正式深度模型输入已同步：
  - 不再把 `species_number` 放入正式 categorical embedding。
  - 使用 `latin_name + kingdom/phylum/class_name/tax_order/family/genus/species + taxon_group_l1/l2/l3 + organism_lifestage` 作为物种上下文。
  - 新增 `medium_domain + target_name` 的 target-scale/medium adapter，shared trunk 输出先经 adapter 残差校正，再进入 endpoint task head。
  - 新增 `no_medium_adapter` 消融项；`no_context` 和 `descriptors_only` 自动关闭 adapter。
  - `heads_embeddings` 微调模式现在会同时微调 task heads、categorical embeddings 和 adapters。
- 正式传统 baseline 输入已同步：
  - `species_number`、`publication_year`、`reference_numbers`、`publication_years`、`test_ids`、`result_ids` 等审计字段不进入特征。
  - taxon 层级字段保留为可用 categorical context。
- 已新增应用域报告模块：
  - `qsar_tl/evaluation/application_domain.py`
  - CLI：`python -m qsar_tl.cli build-ad-report --config configs/experiment.example.yaml --db outputs/derived/modeling_dataset.sqlite --source-table aggregated_task_records_soil_mg_kg_qc --split-name <split> --out outputs/ad/<split>.csv`
  - 输出 Williams leverage、Tanimoto 最大相似度、taxon distance、化学/物种/总体 in-domain 标记和 `ad_warning`。
- 已同步 `scripts/explain_deep_model.py`，解释训练好的 adapter 模型时会传入 adapter id，并把 `target_scale_medium_adapter` 纳入 permutation/SHAP 的候选上下文字段。
- 已更新 `configs/experiment.example.yaml`、`configs/experiment.remote.easyai.yaml` 和 `docs/config_schema.md`：
  - 正式推荐源表改为 `_qc` 子表。
  - 新增 QC 聚合、应用域和 `model.use_medium_adapters` 配置说明。
- 本轮验证：
  - `pytest tests/test_modeling_shapes.py tests/test_deep_experiment_cache.py tests/test_application_domain.py tests/test_baseline_models.py tests/test_qc_aggregation.py tests/test_medium_domain_tables.py tests/test_task_tables.py tests/test_task_mapping.py tests/test_splits.py`，48 passed。
  - `pytest tests`，61 passed。
  - `py_compile` 已覆盖新增/修改的核心 Python 模块和解释脚本。

## 2026-06-12 `_qc` 低土壤样本迁移远端运行

- 已扩展 `scripts/build_aquatic_soil_adaptation_split.py`：
  - 支持 `--soil-finetune-fraction`。
  - 支持同时生成 transfer split 和共享 test set 的 soil-only split。
  - 当前 paired split：
    - `M_qc_aquatic_to_soil_ptox_adapt_C_f10/f20/f50/f100`
    - `SoilPtoxQC_C_low_f10/f20/f50/f100`
- 已扩展 `python -m qsar_tl.cli build-molecule-feature-cache`：
  - 新增 `--source-table`，可直接为 `_qc` 源表构建分子特征缓存。
  - cache manifest 记录 `source_table`。
- 本地验证：
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest tests -q`，63 passed。
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m compileall qsar_tl scripts tests` 通过。
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m qsar_tl.cli validate-config --config configs\experiment.remote.easyai.yaml` 通过。
- 远端 `_qc` 数据链已重建：
  - `task_records_qc` / `aggregated_task_records_qc` 已生成。
  - QC 后 `aggregated_task_records_qc`：120,753。
  - `aggregated_task_records_aquatic_ptox_qc`：89,471。
  - `aggregated_task_records_soil_ptox_qc`：2,847。
  - `aggregated_task_records_soil_mg_kg_qc`：4,999。
  - `aggregated_task_records_aquatic_soil_ptox_qc`：92,318。
- 远端已生成 `_qc` A-F split：
  - `AquaticPtoxQC_`
  - `SoilPtoxQC_`
  - `SoilMgkgQC_`
- paired low-soil split 规模：
  - f10：transfer finetune 239 / test 532；soil-only train 239 / test 532。
  - f20：transfer finetune 520 / test 532；soil-only train 520 / test 532。
  - f50：transfer finetune 1,179 / test 532；soil-only train 1,179 / test 532。
  - f100：transfer finetune 2,315 / test 532；soil-only train 2,315 / test 532。
- 远端验证：
  - `_qc` pTox 表目标量纲单一。
  - paired split 的 test set 完全一致。
  - `SoilPtoxQC_` 和 `SoilMgkgQC_` 的 C/F 化合物泄漏为 0。
  - 本机 RDKit 生成并同步 `outputs/features/molecular_features_rdkit_morgan512.jsonl`，覆盖 `_qc` 水相+土壤 pTox 5,523 个唯一 SMILES，fallback_count=0。
- 远端 smoke：
  - `SoilPtoxQC_B_random_8_2` 使用 `limit=2000`、1 epoch 通过；`limit=500` 被任务阈值保护拒绝，原因是所有 task head 均低于 `min_total=200`。
  - `M_qc_aquatic_to_soil_ptox_adapt_C_f10` 使用 1 epoch pretrain + 1 epoch finetune 通过。
- 已启动远端正式批处理：
  - job script：`outputs/logs/qc_low_soil_batch_job.sh`
  - log：`outputs/logs/qc_low_soil_batch_20260612.log`
  - job PID：`3108179`
  - 当前队列：f10/f20/f50/f100，每个比例先跑 transfer，再跑 soil-only。
  - transfer 输出目录：`outputs/experiments/qc_low_soil_transfer_epoch30_20_lr3e4`
  - soil-only 输出目录：`outputs/experiments/qc_low_soil_soilonly_epoch30`

## 2026-06-12 量纲审计与建模源表约束（当前窗口）

- 当前窗口目标已切换为：数据量纲统一、量纲审计、排除样本过少的量纲、禁止不同不可换算单位混用建模；后续正式实验另开窗口推进。
- 已新增 `scripts/audit_medium_retention_targeted.py`，用于一次性审计水相、沉积物、土壤在 `target_records -> task_records -> aggregated_task_records` 的保留数量、目标尺度、单位族、排除原因和介质子表规模。
- 最新远端审计报告已保存并拉回：`outputs/audits/medium_retention_targeted_v2.txt`。
- 当前建模源表选择规则：
  - 禁止直接使用混合目标尺度宽表建模，例如 `aggregated_task_records_aquatic`、`aggregated_task_records_soil` 只能用于审计或对照说明。
  - 正式建模必须使用同量纲子表，例如 `aggregated_task_records_aquatic_ptox`、`aggregated_task_records_soil_mg_kg`、`aggregated_task_records_soil_ptox`。
  - `qsar_tl.training.baseline.load_split_frame` 已加入保护：同一 source table/split 中若出现多个 `target_name`，传统基线和深度模型都会直接报错，防止不可换算量纲混用。
- 当前按 `minimum_modeling_rows=100` 的量纲状态：
  - 水相：`aquatic_ptox` 89,679、`aquatic_mg_kg` 11,564、`aquatic_percent` 770 为可建模；`aquatic_mg_per_organism` 34、`aquatic_mg_per_experimental_unit` 17 标记为样本不足。
  - 土壤：`soil_mg_kg` 5,171、`soil_ptox` 2,867、`soil_g_ha` 3,146、`soil_percent` 159 为可建模；`soil_l_ha` 48、`soil_seed` 13 标记为样本不足。
  - 沉积物：总聚合 36、`sediment_ptox` 34、`sediment_mg_kg` 1，均低于 100；不建议单独训练，只适合作为迁移外部观察或与水相同 pTox 尺度合并的参考。
- 已更新 `configs/experiment.remote.easyai.yaml` 和 `configs/experiment.example.yaml`：
  - 水相推荐源表：`aggregated_task_records_aquatic_ptox`。
  - 土壤推荐源表：`aggregated_task_records_soil_mg_kg`，并保留 `aggregated_task_records_soil_ptox` 作为 pTox 迁移参照。
  - 沉积物标记 `modeling_status: too_few_rows`，默认不推荐单独建模。
- 已在 A-F split 生成入口加入正式建模阈值保护：
  - `generate-experiment-splits` 在未指定 `--limit` 时读取 `targets.medium_units.minimum_modeling_rows`。
  - 远端已验证 `aggregated_task_records_sediment_ptox` 因 34 条低于 100 被拒绝。
  - 远端已验证 `aggregated_task_records_aquatic_ptox` 带 `--limit 200` 的 smoke split 可正常生成；临时 `GuardSmoke_` split 已清理。
- 已验证：
  - 本地：`pytest tests/test_baseline_models.py tests/test_splits.py tests/test_medium_domain_tables.py tests/test_unit_normalizer.py`，28 passed。
  - 本地/远端：`py_compile` 通过，远端 `configs/experiment.remote.easyai.yaml` 配置校验通过。

## 当前阶段

已完成第一版 `A -> B -> C` 起步交付：

1. 技术蓝图文档。
2. 配置文件规范。
3. 示例实验配置。
4. 第一版 Python 包骨架。
5. CLI 入口。
6. 本机/远程训练执行器接口。
7. PySide6 GUI 训练控制台空壳。

## 2026-06-12 介质筛选与单位 v2 校正进度

- 已删除目标值构建中“剂量组数不少于 3 才允许 min/max 中点”的要求；当前规则为优先 mean，mean 缺失且 min/max 完整时使用中点，并记录 `tox_value_source=min_max_midpoint`。
- 已加入原始单位 v2 编码，覆盖水相质量/摩尔浓度、土壤/沉积物 mg/kg、面积施用量、体积施用量、种子处理、百分比、按个体剂量和按实验单元剂量。
- 远端全量重建后，`included_targets` 从旧流程约 54.6 万提升到 1,040,655；`aggregated_task_records` 为 121,240。
- 介质聚合表最新规模：
  - 水相：`aggregated_task_records_aquatic` 102,652；水相主实验推荐使用 `aggregated_task_records_aquatic_ptox` 89,679。
  - 土壤：`aggregated_task_records_soil` 11,438；其中 `soil_mg_kg` 5,171、`soil_ptox` 2,867、`soil_g_ha` 3,146。
  - 沉积物：`target_records` 289、目标可用 209、任务映射纳入 69、聚合后 36；当前主要限制是数据库中明确沉积物证据记录本身少，不是单位编码导致的大量损失。
- 已补充介质分表：水相和沉积物按 `ptox`、`mg/kg`、百分比、按个体剂量、按实验单元剂量等目标尺度输出，避免后续把不同目标尺度混入同一主实验。
- 已新增审计脚本：
  - `scripts/audit_medium_domain_summary.py`
  - `scripts/audit_medium_sample_flow_fast.py`
- 已修正传统 ML baseline 特征选择，排除原始浓度、标准化浓度和 v2 单位元数据，避免目标/剂量泄漏。
- 后续介质实验入口建议：
  - “水相”主实验：`aggregated_task_records_aquatic_ptox`。
  - “水相 + 沉积物”可比 pTox：`aggregated_task_records_aquatic_sediment_ptox`。
  - “土壤”按问题选择 `aggregated_task_records_soil` 或同尺度子表；迁移实验优先用 `soil_ptox` 与水相 pTox 对齐。
  - “沉积物”不建议单独完整训练，可作为小样本外部验证或与水相合并的 pTox 参照。

## 2026-06-12 校正后远端重跑队列

- 已确认远端 `AquaticPtox_` 与 `AquaticSedimentPtox_` 各 14 个 A-F/fold split 均已生成，source table 分别为：
  - `aggregated_task_records_aquatic_ptox`，每个 split 89,679 条。
  - `aggregated_task_records_aquatic_sediment_ptox`，每个 split 89,713 条。
- 远端 smoke 已通过：
  - baseline smoke：`AquaticPtox_B_random_8_2` × 7 个传统模型 × `limit=500`。
  - deep smoke：`AquaticPtox_B_random_8_2` × `full` × 1 epoch × `limit=500`，`encoder_source=rdkit_cache`。
- 当前正在远端后台运行：
  - `baseline_aquatic_ptox_v2`，PID `442143`，日志 `outputs/logs/baseline_aquatic_ptox_v2_20260612_011447.log`，范围为 `AquaticPtox_` A-F/fold × 7 个传统模型 × `limit=10000`。
  - `deep_aquatic_sediment_ptox_full_epoch30`，PID `470111` 等待脚本已接上并开始运行，日志 `outputs/logs/deep_aquatic_sediment_ptox_full_epoch30_20260612_011951.log`，范围为 `AquaticSedimentPtox_` A-F/fold × `full` × 30 epoch × `limit=10000`。
- 当前已排队等待：
  - `baseline_aquatic_sediment_ptox_v2`，等待 PID `442143` 完成后启动，日志 `outputs/logs/baseline_aquatic_sediment_ptox_v2_20260612_011859.log`。
- 已拉回本机的轻量结果：
  - `outputs/experiments/baseline_aquatic_ptox_v2`
  - `outputs/logs/baseline_aquatic_ptox_v2_20260612_011447.log`
  - `outputs/logs/deep_aquatic_ptox_full_epoch30_20260612_011616.log`
- 已完成并汇总：
  - `deep_aquatic_ptox_full_epoch30`，14 个 A-F/fold split 均完成。
  - 汇总文件：`outputs/experiments/deep_aquatic_ptox_full_epoch30/summary_test_metrics.csv`、`summary_training_convergence.csv`、`figures/`。
  - 按 split family 的测试均值：A R2 0.6768；B R2 0.6470；C R2 0.4336；D R2 0.3608；E R2 0.5956；F R2 0.3070。随机/适应划分高于化合物严格划分，符合外推难度预期。
  - `deep_aquatic_sediment_ptox_full_epoch30`，14 个 A-F/fold split 均完成并汇总。
  - 按 split family 的测试均值：A R2 0.6580；B R2 0.6390；C R2 0.4555；D R2 0.3595；E R2 0.5995；F R2 0.2878。与水相单独结果接近，符合沉积物 pTox 仅新增 34 条样本的审计结论。
- 已发现并修正 XGBoost baseline 配置问题：
  - 远端 `xgboost 3.2.0` 的 `reg:pseudohubererror` 在 pTox 任务上输出约 49 的常数预测，导致 R2 极端负值，不能作为有效基线。
  - 已将 XGBoost 训练目标改为稳定的 `reg:squarederror`；Huber loss 仍作为统一评价指标输出。
  - 已启动 `baseline_aquatic_ptox_v2` 的 XGBoost 全 split 补跑，日志 `outputs/logs/baseline_aquatic_ptox_v2_xgboost_rerun_20260612_014328.log`。
- 当前仍在远端运行：
  - `baseline_soil_ptox_v2`，PID `888067`，日志 `outputs/logs/baseline_soil_ptox_v2_20260612_023008.log`，范围为 `SoilPtox_` A-F/fold × 7 个传统模型 × `limit=10000`。
  - `deep_soil_ptox_full_epoch30_v2`，PID `891131`，日志 `outputs/logs/deep_soil_ptox_full_epoch30_v2_20260612_023038.log`，范围为 `SoilPtox_` A-F/fold × `full` × 30 epoch × `limit=10000`。
  - `baseline_soil_mgkg_v2`，PID `896453`，等待 `baseline_soil_ptox_v2` 完成后自动运行，日志 `outputs/logs/baseline_soil_mgkg_v2_20260612_023125.log`。
  - `deep_soil_mgkg_full_epoch30_v2`，PID `899111`，等待 `deep_soil_ptox_full_epoch30_v2` 完成后自动运行，日志 `outputs/logs/deep_soil_mgkg_full_epoch30_v2_20260612_023151.log`。
- 已完成并重新汇总 baseline：
  - `baseline_aquatic_ptox_v2`：98/98 metrics，XGBoost 已按 `reg:squarederror` 重新补跑并覆盖旧结果。
  - `baseline_aquatic_sediment_ptox_v2`：98/98 metrics，XGBoost 已按 `reg:squarederror` 重新补跑并覆盖旧结果。
  - 已拉回本机：`outputs/experiments/baseline_aquatic_ptox_v2`、`outputs/experiments/baseline_aquatic_sediment_ptox_v2`。
- 已生成土壤相关 A-F split：
  - `Soil_`：source table `aggregated_task_records_soil`，总样本 11,438。
  - `SoilPtox_`：source table `aggregated_task_records_soil_ptox`，总样本 2,867。
  - `SoilMgkg_`：source table `aggregated_task_records_soil_mg_kg`，总样本 5,171。
- 已完成并汇总土壤 deep full：
  - `deep_soil_full_epoch30_v2`，14 个 A-F/fold split 均完成。
  - 汇总文件：`outputs/experiments/deep_soil_full_epoch30_v2/summary_test_metrics.csv`、`summary_training_convergence.csv`、`figures/`。
  - 按 split family 的测试均值：A R2 0.9184；B R2 0.9137；C R2 0.7726；D R2 0.7567；E R2 0.9079；F R2 0.7071。
  - 注意：`Soil_` 是广义土壤表，包含 mg/kg、g/ha、pTox、percent 等不同目标尺度；该结果适合作为广义土壤任务对照，但同尺度结论仍需 `SoilPtox_` 和 `SoilMgkg_`。
- 已完成并汇总土壤 baseline：
  - `baseline_soil_v2`，98/98 metrics 完成。
  - 已拉回本机：`outputs/experiments/baseline_soil_v2`。

## 已确认架构调整

- 工具需要能在本机运行。
- 训练主路径按远程 GPU 训练设计。
- 本机执行器用于配置验证、小样本调试和冒烟测试。
- 远程执行器第一版已经预留，当前通过 SSH 命令占位。

## 已验证命令

```powershell
python -m qsar_tl.cli validate-config --config configs\experiment.example.yaml
```

结果：配置读取成功。

```powershell
python -m qsar_tl.cli run --config configs\experiment.example.yaml --dry-run
```

结果：成功生成远程训练 dry-run 命令。

## 下一步建议

优先进行数据库字段扫描和字段映射落地：

1. SQLite 数据库路径已确认：`G:\QSAR迁移学习\ecotox_clean.sqlite`。
2. 数据库结构扫描报告已生成：`docs/database_schema_scan.md`。
3. 正式字段映射已生成：`configs/field_mapping.ecotox_clean.yaml`。
4. 下一步实现建模宽表构建模块。
5. 下一步实现目标变量单位换算和剔除原因记录。

## 本轮继续目标

- 已新增建模宽表和目标变量构建模块：`qsar_tl/data/modeling_tables.py`。
- 已新增 CLI 命令：`python -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml --limit 10000`。
- 第一版目标构建只纳入已标准化单位：
  - `water_mg_l` 转换为 `ptox_mol_l`
  - `water_mol_l` 转换为 `ptox_mol_l`
  - `soil_mg_kg` 转换为 `neg_log10_mg_kg`
  - `oral_mg_kg_d` 转换为 `neg_log10_mg_kg_bw_day`
- 其他单位暂时进入 `excluded_reason`，避免不可靠数据进入训练。

## 完整建模表构建结果

已运行：

```powershell
python -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml
```

结果：

- `wide_records`: 1,234,077
- `target_records`: 1,234,077
- `included_targets`: 510,934
- `excluded_targets`: 723,143

详细报告见：`docs/modeling_table_build_report.md`。

## 并行框架补全进度

- 已新增任务头映射与聚合样本框架：`qsar_tl/data/task_mapping.py`、`qsar_tl/data/task_tables.py`。
- 已新增 split 与传统基线框架：`qsar_tl/evaluation/splits.py`、`qsar_tl/training/baseline.py`。
- 已新增深度模型骨架：`qsar_tl/modeling/dataset.py`、`qsar_tl/modeling/network.py`、`qsar_tl/training/deep_train.py`。
- 已重构 GUI 控制台框架：`qsar_tl/gui/app.py`、`qsar_tl/gui/widgets.py`、`qsar_tl/gui/job_model.py`、`qsar_tl/gui/process_runner.py`。
- 路径约束：代码不写死本机盘符和解释器路径；数据库、输出和远程路径通过配置或 CLI 参数传入。

### 框架验证结果

已通过：

```powershell
python -m compileall qsar_tl scripts tests
python -m pytest tests -q
python -m qsar_tl.cli --help
```

测试结果：`17 passed, 1 skipped`。跳过项来自当前 base 环境的可选依赖差异；`qsar-ph3` 环境可运行 baseline 冒烟。

已完成 CLI 冒烟：

```powershell
python -m qsar_tl.cli build-task-tables --config configs\experiment.example.yaml --db <derived_db> --limit 2000
python -m qsar_tl.cli generate-split --config configs\experiment.example.yaml --db <derived_db> --split-name random_smoke --split-type random_split --limit 2000
python -m qsar_tl.cli run-baseline --config configs\experiment.example.yaml --db <derived_db> --split-name random_smoke --model random_forest --limit 500 --out <metrics_csv>
```

真实全量聚合结果：

- `task_records`: 510,934
- 第一批严格纳入任务记录：202,678
- `aggregated_task_records`: 80,725
- 主要任务头：`ECx_Mortality`、`ECx_Population`、`NOEC_Growth`、`NOEC_Mortality`、`NOEC_Population`

## GitHub 备份策略

- 本地数据库 `ecotox_clean.sqlite` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 原始对话记录 `USER.txt` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 技术蓝图、配置、代码骨架、数据库结构报告和数据画像报告可以提交。

## 远端训练迁移进度

- 已新增远端配置：`configs/experiment.remote.easyai.yaml`。
- 已补充 `execution.remote.port` 解析和 SSH 命令生成，支持非默认 SSH 端口。
- 已新增 PowerShell 远端流程脚本：
  - `scripts/check_remote_gpu.ps1`
  - `scripts/sync_to_server.ps1`
  - `scripts/train_remote.ps1`
  - `scripts/sync_from_server.ps1`
  - `scripts/remote_common.ps1`
- 已新增流程文档：`docs/remote_training_workflow.md`。
- 推荐长期同步策略：GitHub 管理代码和配置，`rsync` 管理大型 SQLite 数据和输出；当前脚本使用 Windows/OpenSSH `scp` 作为不依赖额外工具的基线方案。

## SSH 与实验计划进度

- 本机已有 `id_ed25519` 密钥对，公钥指纹：`SHA256:JDw1/1LbcUFpufq2B+la1Et3hsTUWzTGLeC60ULw6AU`。
- 已配置 SSH 免密别名：
  - `qsar-gpu` -> `i.easy-ai.cloud:32136`
  - `qsar-gpu-backup` -> `b.easy-ai.cloud:32136`
- 主/备用别名均已通过 `BatchMode=yes` 免密验证。
- 当前远端 GPU 状态异常：`nvidia-smi` 返回退出码 9，提示无法与 NVIDIA driver 通信。
- 当前远端 `/opt/anaconda3/bin/python` 是 Python 3.12.7，但未安装 `torch`。
- 已新增实验计划文档：`docs/experiment_plan_ablation_splits.md`。
- 已确认模型架构文档：`docs/model_architecture_agreed.md`。
- 已将 Morgan 指纹位数调整为 512 bit。
- 已新增 A-F 实验划分生成入口：`python -m qsar_tl.cli generate-experiment-splits`。
- 已新增传统机器学习 baseline matrix 入口：`python -m qsar_tl.cli run-baseline-matrix`。
- 已新增远端实验序列脚本：`scripts/run_remote_sequence.ps1`。
- 已在真实派生库 `outputs/derived/modeling_dataset.sqlite` 生成 A-F split；C/F/D 化合物严格划分泄漏检查为 0。
- 已完成本机小样本 baseline matrix 冒烟：RF、XGBoost、LightGBM、PLS、ExtraTrees、ElasticNet、MLP 均可调度。
- 已同步代码和 `outputs/derived/modeling_dataset.sqlite` 到远端 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端 GPU 已恢复可用：RTX 4060 Ti，NVIDIA driver 575.57.08，CUDA 12.9。
- 已安装远端 PyTorch CUDA 版，验证结果：`torch 2.11.0+cu128`，`torch.cuda.is_available() == True`，GPU 数量 1。
- 远端 XGBoost、LightGBM、SHAP 安装因 PyPI 下载超时暂未完成；RF、ExtraTrees、PLS、ElasticNet、MLP 已可用于远端 CPU baseline。
- 已完成远端小样本 baseline matrix 冒烟：`B_random_8_2` + `random_forest/extra_trees/pls/elastic_net/mlp`，`limit=500`。
- 已完成远端训练入口冒烟：`scripts/train_remote.ps1 -SmokeTest`。当前 `qsar_tl.training.train` 仍为 placeholder，下一步需接入真实深度训练数据加载与多任务网络训练。
- 已将 `qsar_tl.training.train` 接入真实深度训练流程：
  - 从 `split_assignments` 和 `aggregated_task_records` 读取数据。
  - 按任务样本阈值过滤任务头。
  - 构建 512 bit 指纹、分子/上下文数值特征、物种/life stage/介质等 categorical embedding。
  - 输出 `history.csv`、`metrics.csv`、`predictions.csv`、`manifest.json`。
- 已完成本机深度 smoke：`B_random_8_2`，`limit=500`，`epochs=1`，使用 RDKit 编码。
- 已完成远端 CUDA 深度 smoke：`B_random_8_2`，`limit=500`，`epochs=1`，使用 `stable_smiles_fallback` 编码。
- 已完成远端 CUDA 深度 pilot：`B_random_8_2`，`limit=10000`，`epochs=5`，训练任务头 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality`。
  - 训练 loss 从 `9.740133` 降至 `1.675708`。
  - 测试 R2：`ECx_Growth=0.249`，`ECx_Immobilization=0.452`，`ECx_Mortality=0.411`。
  - 输出目录：`outputs/experiments/deep_pilot_remote/deep/B_random_8_2`。
- 当前远端未安装 RDKit，正式 RDKit 描述符训练前需要安装 RDKit 或先在本机构建分子特征缓存并同步到远端。
- 已新增 RDKit 分子特征缓存流程：
  - CLI：`python -m qsar_tl.cli build-molecule-feature-cache`
  - 缓存文件：`outputs/features/molecular_features_rdkit_morgan512.jsonl`
  - 当前缓存包含 4,831 个唯一 SMILES，RDKit fallback 计数为 0。
  - 远端训练已能读取缓存，`encoder_source=rdkit_cache`。
- 已完成远端 RDKit-cache CUDA pilot：
  - `B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1`
  - 每个 split 使用 `limit=10000`、`epochs=5`、`batch_size=256`、`device=cuda:0`
  - 汇总文件：`outputs/experiments/deep_pilot_rdkit_cache_remote/summary_test_metrics.csv`
  - 结果显示从随机划分到化合物严格划分，测试/验证表现整体下降，符合新化合物外推难度上升预期。
- 已新增深度模型消融开关：
  - `full`
  - `no_fingerprint`
  - `no_duration`
  - `no_species_lifestage`
  - `no_context`
  - `no_descriptors`
  - `descriptors_only`
  - `no_molecular_residual`
- 已新增远端批量深度训练脚本：`scripts/train_remote_batch.ps1`。
- 已新增消融结果汇总脚本：`scripts/summarize_deep_ablation.py`，可输出指标表、相对 full 的 delta 表、训练收敛曲线和符合 `style_journal_clean_v1.yaml` 的 PNG/SVG 图。
- 已完成远端消融 pilot：
  - 目录：`outputs/experiments/deep_ablation_pilot_remote`
  - 范围：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1` × `full/no_fingerprint/no_duration/no_species_lifestage/no_context`
  - 参数：`limit=10000`、`epochs=5`、`batch_size=256`、`device=cuda:0`
  - 结论：5 epoch 只适合作为链路和代码验证，不足以解释模块贡献。
- 已完成远端 30 epoch 消融收敛检查：
  - 目录：`outputs/experiments/deep_ablation_epoch30_remote`
  - 范围同上，缺失组合已补跑完整。
  - 结论：相较 5 epoch 指标明显改善；结合用户后续判断，30 epoch 作为当前主实验训练长度更合理。
- 已完成远端 80 epoch 消融收敛检查：
  - 目录：`outputs/experiments/deep_ablation_epoch80_remote`
  - 汇总：`outputs/experiments/deep_ablation_epoch80_remote/summary_test_metrics.csv`
  - 收敛曲线：`outputs/experiments/deep_ablation_epoch80_remote/summary_training_convergence.csv`
  - 图表：`outputs/experiments/deep_ablation_epoch80_remote/figures`
  - 文档：`docs/deep_ablation_epoch80_summary.md`
  - 结论：80 epoch 比 30 epoch 训练 loss 仍有小幅下降，但部分迁移 split 的消融项后段 loss 有反弹；该结果仅作为收敛诊断，正式主实验仍按用户确认的约 30 epoch 策略执行，必要时再显式启用 validation/finetune early stopping。
- 用户随后确认：30 epoch 左右比较合理，后续实验按约 30 epoch 训练。
- 已将远端配置 `configs/experiment.remote.easyai.yaml` 的默认 `training.epochs` 调整为 `30`。
- 已将远端批量训练脚本 `scripts/train_remote_batch.ps1` 默认 `Epochs` 调整为 `30`。
- 已新增可选 early stopping 与 best epoch 输出：
  - 默认关闭，不影响固定 30 epoch 策略。
  - 训练输出会记录 `best_epoch`、`best_model.pt`、`history.csv` 中的 monitor 字段。
  - 如果后续需要验证集早停，可通过 `--early-stopping` 或 PowerShell `-EarlyStopping` 显式启用。
- 已完成 A-F 全划分 full 模型远端 30 epoch 训练：
  - 目录：`outputs/experiments/deep_full_af_epoch30_remote`
  - 范围：A、B、C、D 5 折、E 5 折、F
  - 模型：`full`
  - 参数：`limit=10000`、`epochs=30`、`batch_size=256`、`device=cuda:0`
  - 汇总：`outputs/experiments/deep_full_af_epoch30_remote/summary_test_metrics.csv`
  - 文档：`docs/deep_full_af_epoch30_summary.md`
  - 初步结果：随机 5 折 E 平均 R2 最高，化合物严格/迁移划分 C/D/F 明显更难，符合新化合物外推难度更高的预期。
- 已补齐 A/D/E 的 30 epoch 关键消融项，并与既有 B/C/F 消融结果合并：
  - 新消融目录：`outputs/experiments/deep_ablation_af_epoch30_remote`
  - 合并目录：`outputs/experiments/deep_ablation_af_combined_epoch30`
  - 合并脚本：`scripts/combine_deep_ablation_results.py`
  - 文档：`docs/deep_ablation_af_epoch30_summary.md`
  - 总组合数：70 个 split-ablation 组合。
  - 初步结果：
    - 512 bit Morgan fingerprint 对 A/B/E 随机或适应划分贡献最稳定。
    - 上下文模块与物种/life stage 在 C/D/F 化合物严格或迁移划分中贡献更明显。
    - 当前 duration 模块整体增益较小，需进一步做 raw/log/RBF 的细粒度消融。
- 已完成远端传统机器学习可用基线 A-F 矩阵：
  - 目录：`outputs/experiments/baseline_af_available_limit10000`
  - 范围：A、B、C、D 5 折、E 5 折、F
  - 模型：`random_forest`、`extra_trees`、`pls`、`elastic_net`、`mlp`
  - 暂缺：远端尚未安装 `xgboost`、`lightgbm`，需后续补跑。
  - 汇总脚本：`scripts/summarize_baseline_matrix.py`
  - 文档：`docs/baseline_af_available_summary.md`
  - 初步结果：Random Forest 是当前可用传统模型中的最强基线，但 A-F 全部划分族上均低于 30 epoch deep full；C/D/F 严格化合物或迁移划分差距最大。
- 已补齐远端 boosting 传统基线：
  - 已安装并验证：`xgboost 3.2.0`、`lightgbm 4.6.0`、`shap 0.48.0`、`numpy 1.26.4`。
  - 目录：`outputs/experiments/baseline_af_boosting_limit10000`
  - 合并目录：`outputs/experiments/baseline_af_all_models_limit10000`
  - 文档：`docs/baseline_af_all_models_summary.md`
  - 结果：7 个传统模型、14 个 split/fold、630 行指标已汇总；最强传统基线在 A-F 全部划分族上仍低于 30 epoch deep full。
  - 修复：LightGBM 不接受特殊字符特征名，已在 `qsar_tl/training/baseline.py` 中加入安全特征名映射。
- 已完成深度模型学习率敏感性测试：
  - 目录：`outputs/experiments/deep_lr_sensitivity`
  - 范围：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1` × `full`
  - 学习率：`1e-4`、`3e-4`、`5e-4`、`1e-3`
  - 参数：`limit=10000`、`epochs=30`、`batch_size=256`、`device=cuda:0`
  - 汇总脚本：`scripts/summarize_learning_rate_sweep.py`
  - 文档：`docs/deep_learning_rate_sensitivity_summary.md`
  - 结论：`1e-4` 明显偏小；`5e-4` 在 B/C/F 总体均值上最佳，已将远端配置默认 `training.learning_rate` 调整为 `0.0005`；C-only 严格化合物外推可保留 `0.0003` 作为保守对照。
- 已启动介质迁移实验：
  - 已扩展 `medium_transfer_split` 的介质代码识别，支持 FW/SW/AQU 等水相代码和 ART/NAT/UKS/LIT/MIN 等固相代码。
  - `generate-split` 已支持 `--medium-train-domains`、`--medium-test-domains`、`--medium-unknown-part` 参数。
  - 已生成并运行 `M_water_to_solid` full 深度模型：
    - split 规模：train 73,164；test 1,548；valid 6,013
    - 目录：`outputs/experiments/deep_medium_transfer_water_to_solid`
    - 参数：`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`
    - 文档：`docs/deep_medium_transfer_water_to_solid_summary.md`
  - 初步结果：测试集 R2 大幅为负，主要风险是水相 `pTox mol/L` 与固相 `-log10 mg/kg` 目标尺度不等价；该结果应视为迁移链路验证和目标定义风险提示，不能直接作为模型最终迁移能力结论。
- 已修复介质域表与远端 SQLite 同步问题：
  - `aggregated_task_records` 聚合表已保留 `primary_medium`、`habitat_labels`、`organism_habitat`。
  - 已新增介质子表构建脚本：`scripts/build_medium_domain_tables.py`。
  - 当前介质子表规模：水相 76,133；水相+沉积物 76,163；土壤 3,125；沉积物 30；固相 3,155。
  - `scripts/sync_to_server.ps1` 已在上传 SQLite 前生成稳定 backup 快照，避免 WAL 模式下只传主库导致远端 `malformed database schema`。
- 已完成迁移前介质来源对比的 full 模型 30 epoch 训练：
  - 目录：`outputs/experiments/deep_medium_pretransfer_epoch30`
  - 范围：水相、水相+沉积物、土壤、固相 × B 随机 8:2 / C 化合物严格 8:2。
  - 参数：`limit=10000`、`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`。
  - 文档：`docs/deep_medium_pretransfer_epoch30_summary.md`
  - 初步结果：水相 B 平均 R2 约 0.669，水相 C 约 0.486；水相+沉积物与水相接近，因沉积物样本仅 30 条；土壤/固相 R2 较高但测试样本量较低且任务头结构不同，不能直接和水相均值做强结论比较。
- 已补齐深度模型解释分析链路：
  - 深度训练输出已新增 `preprocessing.json`，记录数值特征名、标准化参数、类别映射、指纹维度和消融配置。
  - 已新增解释脚本：`scripts/explain_deep_model.py`。
  - 解释输出包括 permutation importance、暴露时长 PDP、小样本 SHAP permutation explainer，并将 512 个 Morgan 指纹位聚合为一个可读的指纹组。
  - 已完成首轮正式解释：`Aquatic_C_chemical_holdout_8_2` + `ECx_Mortality`。
  - 输出目录：`outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Aquatic_C_ECx_Mortality`
  - 初步解释：Morgan 指纹组贡献最大；`MolWt`、`MolLogP`、`latin_name`、`species_number` 和 `duration_bin_h` 均进入重要特征前列；暴露时长 PDP 呈现先升高后平台/回落的非线性趋势，符合暴露-毒性关系可能存在平台期的模型假设。
  - 已扩展解释到土壤/固相 C 划分 `ECx_Mortality`：
    - `outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Soil_C_ECx_Mortality`
    - `outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Solid_C_ECx_Mortality`
  - 跨介质解释对比显示：水相更明显依赖分子结构和 RDKit 描述符；土壤/固相中 `target_basis`、物种、life stage、介质和短时长 RBF 贡献更突出。土壤/固相暴露时长 PDP 与水相方向相反，提示目标单位、介质实验设计和样本组成差异可能强烈影响迁移解释。
- 已完成同目标尺度的水相到土壤 pTox 正式迁移实验：
  - 已新增 pTox 可比子表：
    - `aggregated_task_records_aquatic_soil_ptox`: 77,859
    - `aggregated_task_records_soil_ptox`: 1,884
    - `aggregated_task_records_solid_ptox`: 1,914
  - 已修复 `medium_transfer_split`：存在审计后的 `medium_domain` 字段时优先使用该字段，避免 `primary_medium/media_type` 冲突导致水相行进入土壤 test。
  - split：`M_aquatic_to_soil_ptox`
    - train：水相 `ptox_mol_l` 75,975
    - test：土壤 `ptox_mol_l` 1,884
  - 训练目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30`
  - 参数：`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`、不使用 `limit`。
  - 测试任务头平均 R2 为 -0.1027；`ECx_Mortality` R2 为 -0.4079。
  - 已新增土壤 pTox 内部参照：`outputs/experiments/deep_soil_ptox_reference_epoch30`
    - `SoilPtox_B_random_8_2` 平均 R2 约 0.4516。
    - `SoilPtox_C_chemical_holdout_8_2` 平均 R2 约 -0.1120。
  - 已完成迁移模型 `ECx_Mortality` 的 SHAP/PDP/Permutation 解释：
    - 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30/explainability/M_aquatic_to_soil_ptox_ECx_Mortality`
    - 解释显示迁移模型主要依赖 Morgan 指纹、`MolWt`、`HeavyAtomCount`、`MolLogP` 等分子结构特征，土壤上下文校正较弱，提示水相训练偏置明显。
  - 文档：`docs/deep_transfer_aquatic_to_soil_ptox_summary.md`
- 已新增 aquatic pretrain + soil finetune 二阶段训练能力：
  - `qsar_tl.training.train` 新增 `--finetune-epochs`、`--finetune-learning-rate`、`--finetune-batch-size`。
  - `scripts/train_remote.ps1` 和 `scripts/train_remote_batch.ps1` 已支持对应远端参数。
  - 已新增 split 构建脚本：`scripts/build_aquatic_soil_adaptation_split.py`。
  - adaptation split：`M_aquatic_to_soil_ptox_adapt_C`
    - train：水相 pTox 75,975
    - finetune：土壤 pTox 1,589
    - test：土壤 pTox 295
  - 已完成同一 test split 的 zero-shot C 对照：
    - 目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_adaptC_zeroshot_epoch30`
    - 共同任务 `ECx_Growth/ECx_Mortality` 平均 R2：-3.0625
  - 已完成 pretrain 30 epoch + finetune 10 epoch：
    - 目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10`
    - finetune 参数：`epochs=10`、`learning_rate=0.0001`、`batch_size=512`
    - 共同任务平均 R2：-0.4047
    - 相比 zero-shot C 明显改善，但仍弱于 soil-only C reference 平均 R2 -0.1120。
  - 已完成 finetune 模型 `ECx_Mortality` 的 SHAP/PDP/Permutation 解释：
    - 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10/explainability/M_aquatic_to_soil_ptox_adapt_C_ECx_Mortality`
    - 微调后 `species_number`、`latin_name`、`media_type` 进入 SHAP 前列，但 Permutation 仍以 `MolWt`、`HeavyAtomCount` 和 Morgan 指纹组为主，说明微调开始利用土壤上下文但分子结构偏置仍强。
- 已完成 aquatic pretrain + soil finetune 学习率/步数诊断：
  - 诊断目录：
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10_lr3e4`
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20`
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20_lr3e4`
  - 汇总目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity`
  - 图表：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity/finetune_sensitivity_common_tasks.png`
  - 文档：`docs/deep_transfer_aquatic_to_soil_ptox_summary.md`
  - 共同任务 `ECx_Growth/ECx_Mortality` 上，`full finetune 3e-4/20` 当前最佳：
    - 平均 R2：0.0833
    - 平均 RMSE：1.5170
    - 平均 MAE：1.2083
    - 平均 Huber loss：0.7966
  - 对照：`soil-only C reference` 平均 R2 为 -0.1120，平均 Huber loss 为 0.8623；`zero-shot C` 平均 R2 为 -3.0625，平均 Huber loss 为 2.1831。
  - 结论：原 `1e-4/10` 微调学习率和步数均偏保守；`3e-4/20` 更适合作为当前二阶段迁移主配置。冻结到 `heads_embeddings` 明显欠拟合，不建议作为主策略。
  - 风险提示：本轮诊断使用同一土壤 holdout 比较超参数，正式结论应再配独立验证划分或嵌套验证，避免测试集调参偏乐观。

## 2026-06-22 工作区与远端保守整理

- 整理原则：本次只做保守归档，不直接删除实验结果或派生数据库；归档目录可回溯。
- GitHub 备份：
  - 分支：`codex/workspace-cleanup-20260622`
  - 整理前源码/文档安全提交：`bc3040c Record QSAR cleanup baseline and project updates`
  - 已新增 `.gitattributes`，固定文本换行策略，减少 Windows LF/CRLF 噪声。
- 本地整理：
  - 清单文档：`docs/cleanup_inventory_20260622.md`
  - 本地归档目录：`outputs/_archive/cleanup_20260622/`
  - 已移动 17 项旧中间文件/目录，包括旧派生库、smoke/debug 目录、legacy v1.0/v1.1/v1.2.0 原始实验目录。
  - 当前主派生库仍保留在 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
  - 整理后 `outputs/derived` 约 6.210 GB，`outputs/experiments` 约 0.319 GB，`outputs/_archive/cleanup_20260622` 约 5.954 GB。
- 远端整理：
  - 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
  - 整理前确认远端不是 Git 工作树，代码已通过 `scripts/sync_to_server.ps1` 从本地同步，不重传大 SQLite。
  - 远端归档目录：`/home/easyai/DL1/ecotox_qsar_transfer/outputs/_archive/cleanup_20260622/`
  - 已移动 19 项旧中间文件/目录；保留当前 v2 主结果、v1.2.2-v1.2.4 当前结果、features、logs、audits 和应用域结果。
  - 远端 `outputs/derived` 仅保留 `modeling_dataset_v2_0_0_rebuild.sqlite`，约 7.7 GB；`outputs/experiments` 约 3.4 GB；远端归档约 13 GB。
- 核验结果：
  - 本地测试：`E:\TOOLS\anaconda\python.exe -m pytest` 通过，95 passed、1 skipped、1 warning。
  - 配置核验：`E:\TOOLS\anaconda\python.exe -m qsar_tl.cli validate-config --config configs/experiment.remote.easyai.yaml` 通过。
  - 远端训练进程检查未发现实际训练进程。
