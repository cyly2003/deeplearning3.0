# Deep Ablation Pilot Summary

更新时间：2026-06-11

## 运行范围

本轮为远端 CUDA 消融 pilot，不作为最终论文结论，仅用于验证训练链路和判断后续全量实验优先级。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 本机结果目录：`outputs/experiments/deep_ablation_pilot_remote`
- 分子编码：RDKit descriptor + 512 bit Morgan fingerprint cache
- 训练设备：`cuda:0`
- 样本限制：`limit=10000`
- 训练轮数：`epochs=5`
- batch size：`256`
- 代表性划分：
  - `B_random_8_2`
  - `C_chemical_holdout_8_2`
  - `F_chemical_adapt_7_2_1`
- 消融项：
  - `full`
  - `no_fingerprint`
  - `no_duration`
  - `no_species_lifestage`
  - `no_context`

## 输出文件

- 总指标：`outputs/experiments/deep_ablation_pilot_remote/summary_metrics.csv`
- 测试集指标：`outputs/experiments/deep_ablation_pilot_remote/summary_test_metrics.csv`
- 相对完整模型变化：`outputs/experiments/deep_ablation_pilot_remote/summary_ablation_delta_vs_full.csv`
- 图表：
  - `outputs/experiments/deep_ablation_pilot_remote/figures/mean_test_r2_by_ablation.png`
  - `outputs/experiments/deep_ablation_pilot_remote/figures/mean_test_r2_by_ablation.svg`
  - `outputs/experiments/deep_ablation_pilot_remote/figures/delta_test_r2_vs_full.png`
  - `outputs/experiments/deep_ablation_pilot_remote/figures/delta_test_r2_vs_full.svg`

## 测试集平均表现

下表为 3 个任务头 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality` 的 test 平均值。

| split | ablation | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---:|---:|---:|---:|---:|
| B_random_8_2 | full | 0.3622 | 1.3305 | 0.9896 | 0.6184 |
| B_random_8_2 | no_context | 0.3143 | 1.3861 | 1.0560 | 0.6669 |
| B_random_8_2 | no_duration | 0.4004 | 1.2948 | 0.9648 | 0.5949 |
| B_random_8_2 | no_fingerprint | 0.1957 | 1.4920 | 1.1395 | 0.7411 |
| B_random_8_2 | no_species_lifestage | 0.3963 | 1.3038 | 0.9857 | 0.6083 |
| C_chemical_holdout_8_2 | full | 0.2854 | 1.5264 | 1.2545 | 0.8326 |
| C_chemical_holdout_8_2 | no_context | 0.1382 | 1.6733 | 1.3578 | 0.9351 |
| C_chemical_holdout_8_2 | no_duration | 0.3327 | 1.4728 | 1.2060 | 0.7893 |
| C_chemical_holdout_8_2 | no_fingerprint | 0.2249 | 1.5894 | 1.2893 | 0.8614 |
| C_chemical_holdout_8_2 | no_species_lifestage | 0.2978 | 1.5103 | 1.2356 | 0.8183 |
| F_chemical_adapt_7_2_1 | full | 0.2119 | 1.6166 | 1.3326 | 0.9053 |
| F_chemical_adapt_7_2_1 | no_context | -0.0870 | 1.8952 | 1.5628 | 1.1297 |
| F_chemical_adapt_7_2_1 | no_duration | 0.1378 | 1.6906 | 1.3945 | 0.9565 |
| F_chemical_adapt_7_2_1 | no_fingerprint | 0.1755 | 1.6495 | 1.3649 | 0.9391 |
| F_chemical_adapt_7_2_1 | no_species_lifestage | 0.0745 | 1.7439 | 1.4507 | 1.0171 |

## 初步判断

1. 512 bit Morgan fingerprint 对随机划分和化合物严格划分均有稳定贡献，尤其在 `B_random_8_2` 中去掉指纹后平均 R2 从 0.3622 降到 0.1957。
2. 上下文模块在更接近迁移外推的 `F_chemical_adapt_7_2_1` 中最关键，去掉全部上下文后平均 R2 从 0.2119 降到 -0.0870，说明物种、LifeStage、介质和暴露条件对新化合物/新条件泛化有重要调节作用。
3. 物种和 LifeStage 在随机划分中贡献不明显，但在 `F_chemical_adapt_7_2_1` 中去掉后平均 R2 明显下降，提示它们更可能提升外推稳定性，而不是简单提高同分布拟合。
4. 暴露时长消融结果需要谨慎解释：`B` 和 `C` 中短程 pilot 去掉 duration 后略好，但 `F` 中变差。这可能来自短训练轮数、时间特征和其他上下文的共线性、或严格迁移划分下暴露时间分布差异。后续应在更长 epoch、全 A-F split 和任务分层下复核。

## 下一步

建议下一批训练优先执行：

1. 扩展到 A-F 全部 split，保留当前 5 个关键消融项。
2. 将 `full`、`no_fingerprint`、`no_context`、`no_species_lifestage` 作为优先项增加 epoch，观察外推稳定性。
3. 对 duration 模块增加更细消融：只保留原始连续时长、只保留 log/sqrt、只保留 RBF 时间基函数，以判断平台期/S 型假设是否真正有增益。
4. 在 full 和关键消融项完成后，再进入介质迁移实验与 SHAP/PDP 分析。
