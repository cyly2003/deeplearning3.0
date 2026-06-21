# Deep Full Model A-F Epoch30 Summary

更新时间：2026-06-11

## 运行范围

本轮按用户确认后的策略，将深度完整模型固定为约 30 epoch，在 A-F 全部数据划分策略上运行。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 本机结果目录：`outputs/experiments/deep_full_af_epoch30_remote`
- 模型：`full`
- 分子编码：RDKit descriptors + 512 bit Morgan fingerprint cache
- 训练设备：`cuda:0`
- 样本限制：`limit=10000`
- 训练轮数：`epochs=30`
- batch size：`256`
- early stopping：关闭；当前按用户确认的固定 30 epoch 策略执行

## 输出文件

- 总指标：`outputs/experiments/deep_full_af_epoch30_remote/summary_metrics.csv`
- 测试集指标：`outputs/experiments/deep_full_af_epoch30_remote/summary_test_metrics.csv`
- 收敛曲线数据：`outputs/experiments/deep_full_af_epoch30_remote/summary_training_convergence.csv`
- 图表目录：`outputs/experiments/deep_full_af_epoch30_remote/figures`

## Split 平均测试表现

下表为 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality` 的 test 平均值。

| split | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---:|---:|---:|---:|
| A_random_adapt_7_2_1 | 0.4455 | 1.1573 | 0.8390 | 0.4817 |
| B_random_8_2 | 0.5547 | 1.0933 | 0.8012 | 0.4446 |
| C_chemical_holdout_8_2 | 0.4917 | 1.2862 | 1.0458 | 0.6397 |
| D_chemical_group_5fold_fold1 | 0.5121 | 1.3173 | 1.0001 | 0.6145 |
| D_chemical_group_5fold_fold2 | 0.5521 | 1.1913 | 0.9119 | 0.5304 |
| D_chemical_group_5fold_fold3 | 0.3944 | 1.2907 | 1.0092 | 0.6085 |
| D_chemical_group_5fold_fold4 | 0.4449 | 1.2987 | 1.0089 | 0.6222 |
| D_chemical_group_5fold_fold5 | 0.1905 | 1.3560 | 1.0121 | 0.6217 |
| E_random_5fold_fold1 | 0.5594 | 1.0944 | 0.7475 | 0.4138 |
| E_random_5fold_fold2 | 0.6274 | 1.0707 | 0.7923 | 0.4382 |
| E_random_5fold_fold3 | 0.6478 | 1.0010 | 0.7710 | 0.4107 |
| E_random_5fold_fold4 | 0.6342 | 1.0727 | 0.7690 | 0.4202 |
| E_random_5fold_fold5 | 0.5880 | 1.0672 | 0.7925 | 0.4368 |
| F_chemical_adapt_7_2_1 | 0.4029 | 1.4003 | 1.1336 | 0.7183 |

## Split Family 平均表现

| split family | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---:|---:|---:|---:|
| A | 0.4455 | 1.1573 | 0.8390 | 0.4817 |
| B | 0.5547 | 1.0933 | 0.8012 | 0.4446 |
| C | 0.4917 | 1.2862 | 1.0458 | 0.6397 |
| D | 0.4188 | 1.2908 | 0.9884 | 0.5995 |
| E | 0.6114 | 1.0612 | 0.7745 | 0.4239 |
| F | 0.4029 | 1.4003 | 1.1336 | 0.7183 |

## 初步解释

1. 随机划分表现最好：`E_random_5fold` 平均 R2 为 0.6114，`B_random_8_2` 平均 R2 为 0.5547。该结果说明同分布或近同分布插值任务对模型相对容易。
2. 化合物严格划分表现明显下降：`C`、`D`、`F` 的 RMSE 和 Huber loss 普遍高于随机划分，说明新化合物外推仍是主要难点。
3. `F_chemical_adapt_7_2_1` 表现最差，mean R2 为 0.4029，RMSE 为 1.4003。这符合 train/fine-tune/test 三段化合物隔离带来的迁移压力。
4. `D_chemical_group_5fold` fold 间波动较大，fold5 的 mean R2 只有 0.1905，提示某些化合物组可能在分子结构或毒性终点分布上与训练集差异较大，需要后续做 fold-level 分布诊断。
5. 当前 full 模型以 30 epoch 固定训练，所有 split 的 `best_epoch` 均为 30，说明代码按用户确认后的训练长度执行，没有使用 early stopping 中途截断。

## 下一步

1. 按 30 epoch 扩展 A-F 的关键消融矩阵：`no_fingerprint`、`no_duration`、`no_species_lifestage`、`no_context`。
2. 对 `D_chemical_group_5fold_fold5` 做样本分布和化合物空间诊断，判断是否存在特别困难的外推组。
3. 完成传统机器学习 baseline 的 A-F 对比。
4. 进入介质迁移实验与 SHAP/PDP 分析。
