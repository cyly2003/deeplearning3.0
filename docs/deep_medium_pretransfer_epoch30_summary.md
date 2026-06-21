# Deep Medium Pre-Transfer Epoch-30 Summary

更新时间：2026-06-11

## 实验目的

在正式做水相到土壤/固相迁移之前，先比较不同介质来源数据在迁移前的同框架表现：

- 水相：`aggregated_task_records_aquatic`
- 水相 + 沉积物：`aggregated_task_records_aquatic_plus_sediment`
- 土壤：`aggregated_task_records_soil`
- 固相：`aggregated_task_records_solid`

本轮只训练 `full` 深度框架，用作介质来源对比的基线。

## 运行配置

- 远端目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 输出目录：`outputs/experiments/deep_medium_pretransfer_epoch30`
- 模型：`full`
- 划分：B 随机 8:2；C 化合物严格 8:2
- 训练轮数：30 epoch
- 学习率：`0.0005`
- batch size：512
- 设备：`cuda:0`
- 分子编码：`rdkit_cache`
- 单次训练行数上限：`limit=10000`
- 图表样式：`style_journal_clean_v1.yaml`

## 数据规模

远端 SQLite 完整性已验证为 `ok`。本轮使用的介质子表规模：

| 子表 | 记录数 |
|---|---:|
| `aggregated_task_records_aquatic` | 76,133 |
| `aggregated_task_records_aquatic_plus_sediment` | 76,163 |
| `aggregated_task_records_soil` | 3,125 |
| `aggregated_task_records_solid` | 3,155 |

沉积物单独子表仅 30 条，暂不作为独立深度训练介质。

## 测试集汇总

| split | 测试 n | 平均 R2 | 平均 RMSE | 平均 MAE | 平均 Huber loss |
|---|---:|---:|---:|---:|---:|
| `Aquatic_B_random_8_2` | 1,770 | 0.6689 | 1.0052 | 0.7554 | 0.4048 |
| `Aquatic_C_chemical_holdout_8_2` | 1,759 | 0.4860 | 1.3260 | 1.0397 | 0.6404 |
| `AquaticSediment_B_random_8_2` | 1,772 | 0.5997 | 1.0552 | 0.7649 | 0.4226 |
| `AquaticSediment_C_chemical_holdout_8_2` | 1,758 | 0.4953 | 1.3060 | 1.0367 | 0.6335 |
| `Soil_B_random_8_2` | 495 | 0.7870 | 1.5134 | 1.0849 | 0.7274 |
| `Soil_C_chemical_holdout_8_2` | 560 | 0.7597 | 1.4842 | 1.1193 | 0.7351 |
| `Solid_B_random_8_2` | 482 | 0.8543 | 1.2211 | 0.8556 | 0.5203 |
| `Solid_C_chemical_holdout_8_2` | 423 | 0.7790 | 1.5644 | 1.1730 | 0.7726 |

## 初步解释

1. 水相随机划分明显高于水相化合物严格划分，说明新化合物外推仍是主要难点。
2. 水相 + 沉积物与水相非常接近，因为沉积物样本仅 30 条，对总体水相训练分布影响很小。
3. 土壤和固相平均 R2 较高，但测试样本量只有约 400-560，且任务头结构与水相不同，不能直接把均值解释为“固相比水相更容易”。
4. 固相/土壤的 `ECx_Mortality` 误差明显偏高，是后续迁移和解释分析需要重点检查的 endpoint。
5. 本轮结果适合作为迁移前参照；正式讨论介质迁移能力时，应同时报告任务头级别样本数、目标单位/尺度和化合物严格划分结果。

## 输出文件

- 汇总指标：`outputs/experiments/deep_medium_pretransfer_epoch30/summary_metrics.csv`
- 测试集汇总：`outputs/experiments/deep_medium_pretransfer_epoch30/summary_test_metrics.csv`
- 收敛曲线汇总：`outputs/experiments/deep_medium_pretransfer_epoch30/summary_training_convergence.csv`
- 图表目录：`outputs/experiments/deep_medium_pretransfer_epoch30/figures`
- 每个模型目录已补充 `preprocessing.json`，记录数值特征名、标准化参数、类别映射和指纹维度，便于后续 SHAP/PDP/Permutation 解释复现。

## 解释分析首轮结果

已新增深度模型解释脚本：

```powershell
python scripts/explain_deep_model.py --config configs/experiment.remote.easyai.yaml --run-dir <run_dir> --source-table <source_table> --task-head <task_head>
```

首轮正式解释对象：

- split：`Aquatic_C_chemical_holdout_8_2`
- task head：`ECx_Mortality`
- 测试样本：512
- 输出目录：`outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Aquatic_C_ECx_Mortality`

输出文件：

- `baseline_metrics.json`
- `permutation_importance.csv`
- `shap_feature_importance.csv`
- `pdp_duration.csv`
- `permutation_importance_top20.png/.svg`
- `shap_importance_top20.png/.svg`
- `pdp_exposure_duration.png/.svg`

首轮解释结果：

- 512 bit Morgan 指纹组是最主要贡献来源，Permutation `delta_rmse_mean=0.8153`，SHAP 聚合贡献也最高。
- RDKit 描述符中 `MolWt`、`MolLogP`、`NumHAcceptors` 等靠前，说明分子大小、疏水性和氢键受体能力对急性死亡毒性预测有明显贡献。
- `latin_name`、`species_number`、`organism_lifestage` 进入重要特征前列，说明物种分类学和生命阶段上下文在严格化合物外推中仍有信息增益。
- 暴露时长 PDP 显示预测毒性从 24 h 到约 400 h 上升，随后有平台或回落迹象；这与暴露时长-毒性关系可能非线性、存在平台期的假设一致，但当前只是模型解释结果，需结合 endpoint、物种和实验设计进一步验证。

注意：SHAP 当前对深度模型采用小样本 permutation explainer，并将 512 个指纹位聚合成一个指纹组，适合做方向性解释，不应解释为逐 bit 的稳定机制结论。

## ECx_Mortality 跨介质解释对比

已进一步对化合物严格划分 C 下的土壤和固相 `ECx_Mortality` 运行同一解释流程：

| 解释对象 | 测试 n | R2 | RMSE | MAE | Huber loss |
|---|---:|---:|---:|---:|---:|
| `Aquatic_C_ECx_Mortality` | 512 | 0.4724 | 1.5420 | 1.1901 | 0.7789 |
| `Soil_C_ECx_Mortality` | 162 | 0.7347 | 1.8262 | 1.5007 | 1.0638 |
| `Solid_C_ECx_Mortality` | 179 | 0.5334 | 2.4431 | 1.9606 | 1.5119 |

### 主要解释差异

- 水相：Permutation 和 SHAP 均显示 Morgan 指纹组贡献最高，其后是 `MolWt`、`duration_bin_h`、`MolLogP`、`latin_name`、`species_number`。水相急性死亡毒性在该模型中更明显依赖分子结构、分子大小、疏水性和物种分类信息。
- 土壤：Permutation 中 `target_basis` 排名最高，其后是 Morgan 指纹组、`species_number`、`organism_lifestage`；SHAP 中 Morgan 指纹组仍最高，但 `target_basis` 和生命阶段贡献明显。说明土壤数据中实验目标基准、物种和生命阶段对预测影响较大。
- 固相：Permutation 中 `target_basis`、Morgan 指纹组、`species_number`、`media_type`、`organism_lifestage` 和 `duration_rbf_48h` 靠前；SHAP 中 Morgan 指纹组最高，短时长 RBF、生命阶段和目标基准也较突出。固相模型对实验上下文更敏感。
- 暴露时长 PDP：水相随 24 h 到数百小时总体上升后出现平台/回落；土壤和固相则从 24 h 到高时长总体下降。该方向差异不能直接解释为真实毒理机制差异，可能混合了目标单位、实验介质、endpoint 结构和样本组成差异。

### 科研解释注意事项

1. 土壤/固相的 `ECx_Mortality` 测试样本数明显小于水相，解释稳定性较弱。
2. `target_basis` 在土壤/固相中排名很高，提示不同目标基准或单位体系可能正在影响模型，而不只是分子毒性机制。
3. 在正式水相到土壤迁移前，应优先处理目标尺度可比性：水相 `pTox mol/L` 与土壤/固相 `-log10 mg/kg` 不能直接视为同一毒性标尺。
4. 这些结果支持后续将迁移实验拆成“同单位/同目标尺度”和“跨介质上下文迁移”两个层次，而不是一次性混合解释。

## 同步修复

本轮发现派生 SQLite 使用 WAL 模式，单独上传主库会导致远端读取时报：

```text
malformed database schema
```

已修复 `scripts/sync_to_server.ps1`：上传 `.sqlite/.sqlite3/.db` 文件前先用 SQLite backup API 生成稳定快照，再按原文件名上传到远端。远端重新同步后 `PRAGMA integrity_check` 已返回 `ok`。
