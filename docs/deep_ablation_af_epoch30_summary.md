# Deep Ablation A-F Epoch30 Summary

更新时间：2026-06-11

## 运行范围

本轮按用户确认后的固定约 30 epoch 策略，合并 A-F 全划分下的完整模型与关键模块消融结果。

- 合并结果目录：`outputs/experiments/deep_ablation_af_combined_epoch30`
- full 结果来源：`outputs/experiments/deep_full_af_epoch30_remote`
- B/C/F 消融来源：`outputs/experiments/deep_ablation_epoch30_remote`
- A/D/E 消融来源：`outputs/experiments/deep_ablation_af_epoch30_remote`
- 总组合数：70 个 split-ablation 组合
- 训练轮数：`epochs=30`
- 分子编码：RDKit descriptors + 512 bit Morgan fingerprint cache
- 设备：远端 `cuda:0`

## 输出文件

- 总指标：`outputs/experiments/deep_ablation_af_combined_epoch30/summary_metrics.csv`
- 测试集指标：`outputs/experiments/deep_ablation_af_combined_epoch30/summary_test_metrics.csv`
- 相对 full 的变化：`outputs/experiments/deep_ablation_af_combined_epoch30/summary_ablation_delta_vs_full.csv`

## Split Family 平均测试表现

下表为每个 split family 下 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality` 的 test 平均值。

| family | ablation | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---|---:|---:|---:|---:|
| A | full | 0.4455 | 1.1573 | 0.8390 | 0.4817 |
| A | no_context | 0.5773 | 1.0572 | 0.7661 | 0.4212 |
| A | no_duration | 0.4207 | 1.1796 | 0.8527 | 0.4944 |
| A | no_fingerprint | 0.2716 | 1.3878 | 1.0183 | 0.6406 |
| A | no_species_lifestage | 0.5427 | 1.0937 | 0.8074 | 0.4498 |
| B | full | 0.5839 | 1.0677 | 0.7678 | 0.4276 |
| B | no_context | 0.6348 | 1.0165 | 0.7445 | 0.4049 |
| B | no_duration | 0.5846 | 1.0635 | 0.7695 | 0.4278 |
| B | no_fingerprint | 0.3767 | 1.3219 | 0.9869 | 0.6078 |
| B | no_species_lifestage | 0.6190 | 1.0363 | 0.7554 | 0.4144 |
| C | full | 0.5202 | 1.2518 | 1.0129 | 0.6078 |
| C | no_context | 0.3968 | 1.4017 | 1.1465 | 0.7336 |
| C | no_duration | 0.5079 | 1.2672 | 1.0230 | 0.6176 |
| C | no_fingerprint | 0.4588 | 1.3297 | 1.0151 | 0.6250 |
| C | no_species_lifestage | 0.4326 | 1.3588 | 1.1022 | 0.6897 |
| D | full | 0.4188 | 1.2908 | 0.9884 | 0.5995 |
| D | no_context | 0.3042 | 1.4133 | 1.0865 | 0.6863 |
| D | no_duration | 0.3873 | 1.3172 | 1.0147 | 0.6227 |
| D | no_fingerprint | 0.3940 | 1.3377 | 1.0142 | 0.6269 |
| D | no_species_lifestage | 0.3304 | 1.3862 | 1.0672 | 0.6690 |
| E | full | 0.6114 | 1.0612 | 0.7745 | 0.4239 |
| E | no_context | 0.5656 | 1.1174 | 0.8038 | 0.4523 |
| E | no_duration | 0.6051 | 1.0658 | 0.7708 | 0.4239 |
| E | no_fingerprint | 0.4590 | 1.2632 | 0.9663 | 0.5843 |
| E | no_species_lifestage | 0.5540 | 1.1319 | 0.8113 | 0.4616 |
| F | full | 0.4614 | 1.3345 | 1.1051 | 0.6866 |
| F | no_context | 0.2761 | 1.5398 | 1.2704 | 0.8465 |
| F | no_duration | 0.4424 | 1.3550 | 1.1125 | 0.7008 |
| F | no_fingerprint | 0.4791 | 1.3143 | 1.0453 | 0.6379 |
| F | no_species_lifestage | 0.3241 | 1.4871 | 1.2312 | 0.8074 |

## 相对 full 的模块贡献

负的 `delta_r2_vs_full` 表示去掉该模块后性能下降，即该模块对 full 模型有正贡献。

| family | ablation | mean ΔR2 vs full | mean ΔRMSE vs full |
|---|---|---:|---:|
| A | no_context | 0.1318 | -0.1001 |
| A | no_duration | -0.0248 | 0.0223 |
| A | no_fingerprint | -0.1739 | 0.2305 |
| A | no_species_lifestage | 0.0972 | -0.0636 |
| B | no_context | 0.0509 | -0.0513 |
| B | no_duration | 0.0007 | -0.0042 |
| B | no_fingerprint | -0.2072 | 0.2542 |
| B | no_species_lifestage | 0.0351 | -0.0314 |
| C | no_context | -0.1234 | 0.1499 |
| C | no_duration | -0.0122 | 0.0154 |
| C | no_fingerprint | -0.0614 | 0.0779 |
| C | no_species_lifestage | -0.0875 | 0.1070 |
| D | no_context | -0.1146 | 0.1225 |
| D | no_duration | -0.0315 | 0.0264 |
| D | no_fingerprint | -0.0248 | 0.0469 |
| D | no_species_lifestage | -0.0884 | 0.0954 |
| E | no_context | -0.0458 | 0.0563 |
| E | no_duration | -0.0063 | 0.0046 |
| E | no_fingerprint | -0.1523 | 0.2020 |
| E | no_species_lifestage | -0.0574 | 0.0707 |
| F | no_context | -0.1852 | 0.2054 |
| F | no_duration | -0.0190 | 0.0206 |
| F | no_fingerprint | 0.0177 | -0.0202 |
| F | no_species_lifestage | -0.1373 | 0.1526 |

## 初步解释

1. 512 bit Morgan fingerprint 对随机与适应划分贡献最稳定。A、B、E 中去掉 fingerprint 后 R2 分别下降 0.1739、0.2072、0.1523，说明分子结构位指纹对同分布或近同分布预测有明显贡献。
2. 上下文模块在化合物严格和迁移划分中更关键。C、D、F 中去掉全部上下文后 R2 分别下降 0.1234、0.1146、0.1852，说明物种、life stage、介质和暴露条件对新化合物外推有重要调节作用。
3. 物种和 life stage 在严格划分中有稳定贡献。C、D、F 中去掉该模块后 R2 分别下降 0.0875、0.0884、0.1373，提示分类学和生命阶段信息对生态毒性差异建模有科学意义。
4. duration 当前整体增益较小。多数 family 中去掉 duration 的 R2 变化接近 0，仅 C/D/F 有小幅下降。这不等于暴露时长无意义，更可能说明当前 duration 表征和其他上下文存在共线性，或时间非线性基函数还需要更细消融。
5. A/B 中 no_context 和 no_species_lifestage 反而优于 full，提示随机或非严格划分下上下文特征可能引入噪声或过拟合；这部分不应解释为上下文无效，而应结合严格划分结果判断其外推价值。

## 下一步

1. 对 duration 做细粒度消融：raw duration、log/sqrt、RBF basis、no duration。
2. 对 `D_chemical_group_5fold_fold5` 做化合物空间和任务分布诊断。
3. 运行传统机器学习 baseline 的 A-F 全矩阵。
4. 进入水相、水相+沉积物、土壤、沉积物迁移前的介质迁移实验。
5. 在最终候选模型上做 SHAP 和 PDP。
