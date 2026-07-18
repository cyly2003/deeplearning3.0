# v1.2.21 Random Split Targeted Ablation Summary

更新时间：2026-07-02 13:45 (+08:00)

## 实验范围

本实验用于检验固定 chemical-holdout 主线消融结论是否也能在随机插值划分下复现。实验不重跑 full baseline，而是以 v1.2.15 的 full 单模型与 5-seed ensemble 作为参照。

- 数据表：`aggregated_task_records_aquatic_soil_ptox_qc`
- 随机 8:2：`M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100`，5 seeds
- 随机 5-fold：`M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold1-5_f100`，seed42
- targeted 消融：`no_context`、`no_species_lifestage`、`no_molecular_residual`、`no_source_weighting`、`no_toxicity_binning`、`no_censored_loss`
- 远端输出：`outputs/experiments/v1_2_21_random_split_ablation_remote_summary`

## 运行时间

| runs | 总耗时 h | 平均 min/run | 最短 min | 最长 min | 失败数 |
|---:|---:|---:|---:|---:|---:|
| 60 | 33.56 | 33.56 | 22.75 | 38.92 | 0 |

## test 集整体指标

参照 full 单模型基线来自 v1.2.15：

- random 8:2 full 单模型：MAE 0.6370，R2 0.7570
- random 5-fold full 单模型：MAE 0.6397，R2 0.7576
- random 8:2 5-seed ensemble：MAE 0.5925，R2 0.7853
- random 5-fold 5-seed ensemble：MAE 0.5939，R2 0.7878

| split policy | 消融项 | runs | MAE mean | MAE sd | R2 mean | R2 sd | ΔMAE vs full 单模型 |
|---|---|---:|---:|---:|---:|---:|---:|
| random 8:2 | no_molecular_residual | 5 | 0.6202 | 0.0119 | 0.7631 | 0.0059 | -0.0168 |
| random 8:2 | no_source_weighting | 5 | 0.6361 | 0.0106 | 0.7579 | 0.0069 | -0.0010 |
| random 8:2 | no_censored_loss | 5 | 0.6366 | 0.0092 | 0.7584 | 0.0062 | -0.0005 |
| random 8:2 | no_toxicity_binning | 5 | 0.6557 | 0.0063 | 0.7445 | 0.0054 | +0.0187 |
| random 8:2 | no_species_lifestage | 5 | 0.7154 | 0.0071 | 0.6847 | 0.0063 | +0.0784 |
| random 8:2 | no_context | 5 | 0.8650 | 0.0115 | 0.5617 | 0.0078 | +0.2280 |
| random 5-fold | no_molecular_residual | 5 | 0.6262 | 0.0045 | 0.7616 | 0.0043 | -0.0134 |
| random 5-fold | no_censored_loss | 5 | 0.6335 | 0.0125 | 0.7590 | 0.0100 | -0.0062 |
| random 5-fold | no_source_weighting | 5 | 0.6400 | 0.0084 | 0.7545 | 0.0110 | +0.0004 |
| random 5-fold | no_toxicity_binning | 5 | 0.6699 | 0.0123 | 0.7357 | 0.0040 | +0.0303 |
| random 5-fold | no_species_lifestage | 5 | 0.7168 | 0.0192 | 0.6844 | 0.0226 | +0.0771 |
| random 5-fold | no_context | 5 | 0.8555 | 0.0133 | 0.5756 | 0.0183 | +0.2159 |

## 结论

随机 8:2 与随机 5-fold 得到了一致的消融排序：`context embedding` 是随机插值预测中最关键的信息来源，去除后 MAE 增加约 0.22 log units；`species/lifestage` 信息次之，MAE 增加约 0.08 log units；`toxicity binning` 有稳定但较小的贡献，MAE 增加约 0.02-0.03 log units。

`source weighting`、`censored loss` 与 `molecular residual` 在随机划分下没有表现出明显增益，说明它们更可能服务于固定 chemical-holdout 或迁移外推边界，而不是同分布插值精度。论文叙事中应将 random split 结果定位为插值参照和稳定性验证，不能替代 fixed chemical-holdout 的外推泛化证据。
