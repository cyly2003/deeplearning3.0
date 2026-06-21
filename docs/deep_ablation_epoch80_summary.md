# Deep Ablation Epoch80 Summary

更新时间：2026-06-11

## 运行范围

本轮用于回应“5 epoch/30 epoch 训练不充分”的问题，是更长训练的远端收敛性检查。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 本机结果目录：`outputs/experiments/deep_ablation_epoch80_remote`
- 分子编码：RDKit descriptor + 512 bit Morgan fingerprint cache
- 训练设备：`cuda:0`
- 样本限制：`limit=10000`
- 训练轮数：`epochs=80`
- batch size：`256`
- 划分：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1`
- 消融项：`full`、`no_fingerprint`、`no_duration`、`no_species_lifestage`、`no_context`

## 输出文件

- 总指标：`outputs/experiments/deep_ablation_epoch80_remote/summary_metrics.csv`
- 测试集指标：`outputs/experiments/deep_ablation_epoch80_remote/summary_test_metrics.csv`
- 相对完整模型变化：`outputs/experiments/deep_ablation_epoch80_remote/summary_ablation_delta_vs_full.csv`
- 训练收敛曲线数据：`outputs/experiments/deep_ablation_epoch80_remote/summary_training_convergence.csv`
- 图表目录：`outputs/experiments/deep_ablation_epoch80_remote/figures`

## 测试集平均表现

下表为 3 个任务头 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality` 的 test 平均值。

| split | ablation | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---:|---:|---:|---:|---:|
| B_random_8_2 | full | 0.6017 | 1.0326 | 0.7562 | 0.4077 |
| B_random_8_2 | no_context | 0.6612 | 0.9746 | 0.7118 | 0.3776 |
| B_random_8_2 | no_duration | 0.5978 | 1.0255 | 0.7565 | 0.4081 |
| B_random_8_2 | no_fingerprint | 0.3755 | 1.3130 | 0.9593 | 0.5860 |
| B_random_8_2 | no_species_lifestage | 0.6603 | 0.9774 | 0.7133 | 0.3771 |
| C_chemical_holdout_8_2 | full | 0.5419 | 1.2226 | 0.9800 | 0.5847 |
| C_chemical_holdout_8_2 | no_context | 0.4336 | 1.3585 | 1.0908 | 0.6869 |
| C_chemical_holdout_8_2 | no_duration | 0.5373 | 1.2287 | 0.9852 | 0.5905 |
| C_chemical_holdout_8_2 | no_fingerprint | 0.5146 | 1.2579 | 0.9585 | 0.5758 |
| C_chemical_holdout_8_2 | no_species_lifestage | 0.4131 | 1.3813 | 1.1102 | 0.7035 |
| F_chemical_adapt_7_2_1 | full | 0.4711 | 1.3179 | 1.0572 | 0.6478 |
| F_chemical_adapt_7_2_1 | no_context | 0.4284 | 1.3736 | 1.1293 | 0.7121 |
| F_chemical_adapt_7_2_1 | no_duration | 0.5208 | 1.2569 | 1.0243 | 0.6165 |
| F_chemical_adapt_7_2_1 | no_fingerprint | 0.4587 | 1.3337 | 1.0536 | 0.6549 |
| F_chemical_adapt_7_2_1 | no_species_lifestage | 0.4104 | 1.3942 | 1.1356 | 0.7237 |

## 30 到 80 epoch 的变化

相比 30 epoch，80 epoch 大多数 test 指标仍有改善，说明 30 epoch 仍偏短，但收益已经变小。

- `B_random_8_2/full`：mean R2 +0.0178，RMSE -0.0351。
- `C_chemical_holdout_8_2/full`：mean R2 +0.0217，RMSE -0.0292。
- `F_chemical_adapt_7_2_1/full`：mean R2 +0.0097，RMSE -0.0166。
- `F_chemical_adapt_7_2_1/no_context`：mean R2 +0.1523，说明 30 epoch 对该消融项明显未充分训练。
- `F_chemical_adapt_7_2_1/no_species_lifestage`：mean R2 +0.0863，但后段训练 loss 有反弹，应避免继续无约束增加 epoch。

## 初步判断

1. 5 epoch 明显只是链路 pilot，不足以解释模块贡献。
2. 30 epoch 仍偏短，尤其是上下文消融和迁移压力较大的 `F` split。
3. 80 epoch 更接近可解释的 pilot 结果，但仍不能替代正式实验，因为当前训练只记录最终 test 指标，缺少按 epoch 的验证集 early stopping。
4. Morgan 512 fingerprint 在随机划分和化合物严格划分中仍有明确贡献；去掉指纹后 `B_random_8_2` 的 mean R2 从 0.6017 降到 0.3755。
5. 上下文信息在化合物严格划分和迁移划分中仍有贡献，但在随机划分中 no_context 表现更好，可能说明随机划分下上下文信息带来噪声、共线性或过拟合风险。
6. duration 模块的结果仍需单独拆分验证。`F` 中 `no_duration` 高于 full，提示当前时间 RBF/连续时长特征可能需要重新调参、正则化或用更细的 duration-only 消融验证。

## 建议

正式实验不建议固定单一 `epochs=80` 后直接结束。建议下一步实现：

1. `epochs=120` 上限。
2. 基于 validation/finetune split 的 early stopping，patience 建议 15。
3. 保存 best epoch 和 best model manifest。
4. duration 更细消融：
   - only raw duration
   - raw + log/sqrt
   - raw + RBF
   - no duration
5. 在 early stopping 完成后再扩展到 A-F 全 split。
