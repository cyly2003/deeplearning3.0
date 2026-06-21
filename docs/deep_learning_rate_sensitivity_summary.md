# Deep Learning Rate Sensitivity Summary

更新时间：2026-06-11

## 运行范围

本轮用于回答“当前学习步长是否太短”。按用户确认后的约 30 epoch 策略，固定深度完整模型 `full`，在 B/C/F 三个代表性划分上比较 4 个学习率。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 本机结果目录：`outputs/experiments/deep_lr_sensitivity`
- 划分：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1`
- 模型：`full`
- 学习率：`1e-4`、`3e-4`、`5e-4`、`1e-3`
- 训练轮数：`epochs=30`
- batch size：`256`
- 设备：远端 `cuda:0`

## 输出文件

- 总指标：`outputs/experiments/deep_lr_sensitivity/summary_metrics.csv`
- 测试集指标：`outputs/experiments/deep_lr_sensitivity/summary_test_metrics.csv`
- split × learning rate 汇总：`outputs/experiments/deep_lr_sensitivity/summary_split_learning_rate_metrics.csv`
- family × learning rate 汇总：`outputs/experiments/deep_lr_sensitivity/summary_family_learning_rate_metrics.csv`
- 收敛曲线数据：`outputs/experiments/deep_lr_sensitivity/summary_training_convergence.csv`
- 图表目录：`outputs/experiments/deep_lr_sensitivity/figures`

## 测试集结果

下表为每个 split family 下 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality` 的 test 平均值。

| family | learning rate | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---:|---:|---:|---:|---:|
| B | 0.0001 | 0.4932 | 1.1814 | 0.8854 | 0.5159 |
| B | 0.0003 | 0.5885 | 1.0564 | 0.7623 | 0.4193 |
| B | 0.0005 | 0.6082 | 1.0300 | 0.7319 | 0.3996 |
| B | 0.0010 | 0.6010 | 1.0288 | 0.7510 | 0.4083 |
| C | 0.0001 | 0.4696 | 1.3161 | 1.0764 | 0.6606 |
| C | 0.0003 | 0.5150 | 1.2585 | 1.0125 | 0.6123 |
| C | 0.0005 | 0.5033 | 1.2735 | 1.0282 | 0.6266 |
| C | 0.0010 | 0.4690 | 1.3135 | 1.0476 | 0.6480 |
| F | 0.0001 | 0.3485 | 1.4682 | 1.2196 | 0.7874 |
| F | 0.0003 | 0.4411 | 1.3577 | 1.1014 | 0.6930 |
| F | 0.0005 | 0.4900 | 1.2947 | 1.0622 | 0.6462 |
| F | 0.0010 | 0.4963 | 1.2886 | 1.0411 | 0.6276 |

## 总体均值

| learning rate | mean R2 | mean RMSE | mean MAE | mean Huber |
|---:|---:|---:|---:|---:|
| 0.0001 | 0.4371 | 1.3219 | 1.0605 | 0.6546 |
| 0.0003 | 0.5149 | 1.2242 | 0.9587 | 0.5749 |
| 0.0005 | 0.5338 | 1.1994 | 0.9408 | 0.5575 |
| 0.0010 | 0.5221 | 1.2103 | 0.9466 | 0.5613 |

## 结论

1. `1e-4` 明显偏小。三个 split 的第 30 轮训练 loss 仍在约 1.09-1.20，测试指标也最低，说明 30 epoch 内收敛不足。
2. `3e-4` 是稳定基准，但不是总体最优。它在 C 化合物严格 holdout 上最稳，mean R2 为 0.5150。
3. `5e-4` 是当前推荐默认值。它在 B 和总体均值上最好，F 迁移划分也明显优于 `3e-4`，总体 mean R2 从 0.5149 提升到 0.5338，Huber loss 从 0.5749 降到 0.5575。
4. `1e-3` 训练 loss 最低，但测试指标没有稳定超过 `5e-4`，且 C 严格化合物划分下降明显，说明开始出现更强的过拟合或外推不稳定风险。

## 后续执行建议

- 后续主实验默认使用 `learning_rate=5e-4`、`epochs≈30`。
- 如果单独做 C 类严格化合物外推保守验证，可额外保留 `3e-4` 对照。
- 不建议把默认学习率直接提高到 `1e-3`，除非启用 validation/finetune early stopping 并确认 C/D/F 外推不下降。
