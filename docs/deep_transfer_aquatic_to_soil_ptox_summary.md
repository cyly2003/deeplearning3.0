# Deep Transfer: Aquatic to Soil pTox Epoch-30 Summary

更新时间：2026-06-11

## 实验目的

此前 `M_water_to_solid` 直接混合水相 `pTox mol/L` 与固相 `-log10 mg/kg`，测试 R2 大幅为负，主要风险是目标尺度不等价。本轮改为只保留同一目标尺度：

- 训练域：水相 `target_name = ptox_mol_l`
- 测试域：土壤 `target_name = ptox_mol_l`
- split：`M_aquatic_to_soil_ptox`

该实验回答的是：在目标尺度一致时，水相 QSAR 知识能否直接迁移到土壤 pTox 样本。

## 数据与划分

新增可复现子表：

| 子表 | 记录数 |
|---|---:|
| `aggregated_task_records_aquatic_soil_ptox` | 77,859 |
| `aggregated_task_records_soil_ptox` | 1,884 |
| `aggregated_task_records_solid_ptox` | 1,914 |

严格迁移 split：

| split part | medium domain | target_name | n |
|---|---|---|---:|
| train | aquatic | `ptox_mol_l` | 75,975 |
| test | soil | `ptox_mol_l` | 1,884 |

为避免字段冲突，`medium_transfer_split` 已调整为优先使用审计后的 `medium_domain` 字段，而不是重新从 `primary_medium/media_type` 混合推断。

## 训练配置

- 远端目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30`
- 模型：`full`
- 训练轮数：30 epoch
- 学习率：`0.0005`
- batch size：512
- 分子编码：`rdkit_cache`
- 设备：`cuda:0`
- 训练不使用 `limit`，避免截断 test 域。

任务过滤后实际纳入：

- 总样本：61,561
- train 样本：59,725
- task heads：`ECx_Growth`、`ECx_Mortality`、`ECx_Population`、`ECx_Reproduction`、`LOEC_Growth`、`NOEC_Growth`、`NOEC_Mortality`、`NOEC_Reproduction`

因土壤 test 样本不足跳过的任务包括：`ECx_Immobilization`、多个 LOEC/NOEC 小样本任务。

## 迁移测试表现

| task_head | test n | R2 | RMSE | MAE | Huber loss |
|---|---:|---:|---:|---:|---:|
| `ECx_Growth` | 657 | 0.1063 | 1.2836 | 0.9694 | 0.5858 |
| `ECx_Mortality` | 426 | -0.4079 | 2.2476 | 1.8373 | 1.3918 |
| `ECx_Population` | 84 | -0.0954 | 1.2562 | 1.0213 | 0.6077 |
| `ECx_Reproduction` | 252 | -0.0790 | 1.3716 | 1.0543 | 0.6599 |
| `LOEC_Growth` | 83 | 0.2450 | 1.4721 | 1.1410 | 0.7334 |
| `NOEC_Growth` | 193 | 0.0383 | 1.2593 | 0.9507 | 0.5746 |
| `NOEC_Mortality` | 59 | -0.2699 | 1.5003 | 1.3040 | 0.8499 |
| `NOEC_Reproduction` | 82 | -0.3593 | 1.3362 | 1.0226 | 0.6320 |

测试任务头平均：

- R2：-0.1027
- RMSE：1.4659
- MAE：1.1626
- Huber loss：0.7544

## 土壤 pTox 内部参照

为判断迁移失败是跨介质问题还是土壤新化合物外推本身困难，新增土壤 pTox 内部参照：

- `SoilPtox_B_random_8_2`
- `SoilPtox_C_chemical_holdout_8_2`

| 参照 split | test task heads | 平均 R2 | 平均 RMSE | 平均 MAE | 平均 Huber loss |
|---|---:|---:|---:|---:|---:|
| `SoilPtox_B_random_8_2` | 3 | 0.4516 | 1.1342 | 0.8369 | 0.5044 |
| `SoilPtox_C_chemical_holdout_8_2` | 2 | -0.1120 | 1.6423 | 1.2825 | 0.8623 |
| `M_aquatic_to_soil_ptox` | 8 | -0.1027 | 1.4659 | 1.1626 | 0.7544 |

解释：

1. 土壤随机划分表现明显更好，说明同介质、相似化合物分布下模型能学习土壤 pTox。
2. 土壤化合物严格划分平均 R2 也接近 0 以下，说明土壤 pTox 新化合物外推本身很难。
3. 水相到土壤 pTox 迁移并未带来稳定增益；尤其 `ECx_Mortality` 为负，说明水相急性死亡毒性知识不能直接外推到土壤死亡毒性。

## 迁移模型解释分析

解释对象：

- run：`M_aquatic_to_soil_ptox`
- task head：`ECx_Mortality`
- 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30/explainability/M_aquatic_to_soil_ptox_ECx_Mortality`

测试指标：

- n：426
- R2：-0.4079
- RMSE：2.2476
- MAE：1.8373
- Huber loss：1.3918

主要解释结果：

- Permutation importance 前列：`MolWt`、`HeavyAtomCount`、Morgan 指纹组、`RingCount`、`TPSA`、`organism_habitat`。
- SHAP 前列：Morgan 指纹组、`MolWt`、`HeavyAtomCount`、`MolLogP`、`RingCount`、`TPSA`。
- 暴露时长 PDP 从 24 h 到高时长总体上升，接近水相模型趋势，而不是此前土壤/固相内模型的下降趋势。

科学含义：

迁移模型主要依赖分子结构和分子大小类特征，土壤上下文、物种和生命阶段校正较弱。这提示模型仍保留水相训练偏置，未充分学习土壤介质下暴露-毒性关系。土壤 pTox 迁移后续不宜直接采用“水相预训练后零样本土壤测试”作为最终方案，应进入少量土壤样本微调或域适配实验。

## 后续建议

1. 进行 aquatic pretrain + soil finetune：使用水相 pTox 预训练，再用土壤 pTox 训练子集微调，测试土壤 holdout。
2. 只在共同任务头上比较迁移增益，优先 `ECx_Growth`、`ECx_Mortality`、`ECx_Reproduction`。
3. 对 `target_basis` 和介质上下文字段做更严格分层，避免实验基准差异主导迁移结论。
4. 保留 `neg_log10_mg_kg` 固相任务作为独立土壤剂量标尺模型，不与水相 pTox 直接混合训练。

## Aquatic Pretrain + Soil Finetune

已新增二阶段训练能力：

- 训练入口：`python -m qsar_tl.training.train`
- 新参数：`--finetune-epochs`、`--finetune-learning-rate`、`--finetune-batch-size`
- 远端脚本：`scripts/train_remote.ps1`、`scripts/train_remote_batch.ps1`
- split 构建脚本：`scripts/build_aquatic_soil_adaptation_split.py`

adaptation split：

| split | medium domain | target_name | n |
|---|---|---|---:|
| train | aquatic | `ptox_mol_l` | 75,975 |
| finetune | soil | `ptox_mol_l` | 1,589 |
| test | soil | `ptox_mol_l` | 295 |

训练配置：

- pretrain：水相 30 epoch，learning rate `0.0005`
- finetune：土壤 10 epoch，learning rate `0.0001`
- batch size：512
- 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10`

为了公平比较，另补跑同一 adaptation split 的 zero-shot C 对照：

- 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_adaptC_zeroshot_epoch30`

### 同一土壤 holdout 上的结果

| run | 共同任务 | 平均 R2 | 平均 RMSE | 平均 MAE | 平均 Huber loss |
|---|---:|---:|---:|---:|---:|
| aquatic zero-shot C | 2 | -3.0625 | 3.2193 | 2.6522 | 2.1831 |
| aquatic pretrain + soil finetune | 2 | -0.4047 | 1.8963 | 1.4487 | 1.0285 |
| soil-only C reference | 2 | -0.1120 | 1.6423 | 1.2825 | 0.8623 |

共同任务为 `ECx_Growth` 和 `ECx_Mortality`。

解释：

1. soil finetune 相比同一 test split 的 zero-shot 明显改善，说明少量土壤样本能够校正水相预训练偏置。
2. finetune 仍弱于 soil-only C reference，说明当前二阶段策略还没有优于直接土壤训练，可能需要更好的微调策略、冻结策略或域适配损失。
3. 小样本任务如 `LOEC_Growth` 的 R2 极端负值会显著拉低全任务平均，应以共同任务头和样本数加权指标为主。

### Finetune 模型解释

解释对象：

- run：`M_aquatic_to_soil_ptox_adapt_C`
- task head：`ECx_Mortality`
- 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10/explainability/M_aquatic_to_soil_ptox_adapt_C_ECx_Mortality`

测试指标：

- n：91
- R2：-0.1391
- RMSE：2.1836
- MAE：1.6049
- Huber loss：1.1817

解释结果：

- Permutation importance：`MolWt`、`HeavyAtomCount`、Morgan 指纹组、`effect_level_x`、`RingCount` 靠前。
- SHAP：Morgan 指纹组最高，其后为 `MolWt`、`species_number`、`HeavyAtomCount`、`MolLogP`、`RingCount`、`latin_name`、`media_type`。
- PDP：暴露时长从 24 h 到高时长总体上升，仍保留水相预训练模型的时长响应趋势。

科学含义：

微调后 `species_number`、`latin_name`、`media_type` 进入 SHAP 前列，说明模型开始利用土壤样本的生物和介质上下文；但 Permutation 仍以分子大小和结构特征为主，死亡毒性测试 R2 尚未转正。当前结果支持继续做更细的微调策略，而不是把 10 epoch 全模型微调作为最终迁移方案。

## Finetune 学习率与步数诊断

针对“微调学习步长是否太短”的问题，补跑了同一 adaptation split 上的三组诊断实验：

- `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10_lr3e4`
- `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20`
- `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20_lr3e4`

汇总输出：

- CSV：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity/common_task_summary.csv`
- 图表：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity/finetune_sensitivity_common_tasks.png`
- SVG：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity/finetune_sensitivity_common_tasks.svg`

共同任务仍限定为 `ECx_Growth` 和 `ECx_Mortality`，以保证 zero-shot、finetune 和 soil-only C reference 在同一土壤 holdout 上可比。

| run | 共同任务 | 平均 R2 | 平均 RMSE | 平均 MAE | 平均 Huber loss |
|---|---:|---:|---:|---:|---:|
| aquatic zero-shot C | 2 | -3.0625 | 3.2193 | 2.6522 | 2.1831 |
| heads+embeddings finetune `1e-4/10` | 2 | -0.8053 | 2.1303 | 1.7000 | 1.2724 |
| full finetune `1e-4/10` | 2 | -0.4047 | 1.8963 | 1.4487 | 1.0285 |
| full finetune `1e-4/20` | 2 | -0.2500 | 1.7543 | 1.3838 | 0.9657 |
| full finetune `3e-4/10` | 2 | -0.1910 | 1.7376 | 1.3195 | 0.9148 |
| soil-only C reference | 2 | -0.1120 | 1.6423 | 1.2825 | 0.8623 |
| full finetune `3e-4/20` | 2 | 0.0833 | 1.5170 | 1.2083 | 0.7966 |

按任务拆分：

| task head | full finetune `3e-4/20` R2 | soil-only C R2 | full finetune `3e-4/20` Huber | soil-only C Huber |
|---|---:|---:|---:|---:|
| `ECx_Growth` | -0.1382 | -0.4722 | 0.6465 | 0.7731 |
| `ECx_Mortality` | 0.3049 | 0.2482 | 0.9466 | 0.9515 |

训练收敛对比显示：

| run | finetune epochs | finetune loss first -> last |
|---|---:|---:|
| full finetune `1e-4/10` | 10 | 7.0442 -> 4.0886 |
| full finetune `3e-4/10` | 10 | 7.3279 -> 2.8073 |
| full finetune `1e-4/20` | 20 | 6.5277 -> 2.7591 |
| full finetune `3e-4/20` | 20 | 5.1870 -> 1.7352 |
| heads+embeddings finetune `1e-4/10` | 10 | 7.0342 -> 6.6478 |

结论：

1. 原 `1e-4/10` 微调确实偏保守；问题同时来自学习率偏低和微调步数偏少。
2. `3e-4/20` 是当前同一 holdout 上的最佳二阶段迁移配置，平均 Huber loss 低于 soil-only C reference，平均 R2 转正。
3. 冻结到 heads+embeddings 明显欠拟合，当前不宜作为主迁移策略。
4. 由于这些诊断实验都在同一土壤 holdout 上比较，后续正式报告中应把它们表述为超参数诊断；最终结论最好再配一个独立验证划分或嵌套验证，避免反复依据同一 test split 调参造成乐观偏差。
