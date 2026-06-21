# Traditional Baseline A-F Available Models Summary

更新时间：2026-06-11

## 运行范围

本轮在远端完成 A-F 全划分下可用传统机器学习基线矩阵。远端当前未安装 `xgboost` 和 `lightgbm`，因此本轮先汇总已可运行的 sklearn 系列模型。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 本机结果目录：`outputs/experiments/baseline_af_available_limit10000`
- 划分范围：A、B、C、D 5 折、E 5 折、F
- 模型范围：`random_forest`、`extra_trees`、`pls`、`elastic_net`、`mlp`
- 损失设定：Huber loss，`delta=1.0`
- 样本限制：`limit=10000`
- 指标任务头：`ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality`

## 输出文件

- 总指标：`outputs/experiments/baseline_af_available_limit10000/summary_metrics.csv`
- 测试集指标：`outputs/experiments/baseline_af_available_limit10000/summary_test_metrics.csv`
- split × model 汇总：`outputs/experiments/baseline_af_available_limit10000/summary_split_model_metrics.csv`
- family × model 汇总：`outputs/experiments/baseline_af_available_limit10000/summary_family_model_metrics.csv`
- 相对 30 epoch deep full 的差值：`outputs/experiments/baseline_af_available_limit10000/summary_family_model_delta_vs_deep_full.csv`
- 图表目录：`outputs/experiments/baseline_af_available_limit10000/figures`

## 最优传统基线

下表为每个 split family 中 mean test R2 最高的传统模型。当前 5 个可用模型里，`random_forest` 在 A-F 全部划分族中均为最优。

| family | best baseline | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---|---:|---:|---:|---:|
| A | random_forest | 0.2770 | 1.3671 | 0.9949 | 0.6181 |
| B | random_forest | 0.4190 | 1.2656 | 0.9294 | 0.5659 |
| C | random_forest | -0.0005 | 1.8098 | 1.4311 | 1.0121 |
| D | random_forest | -0.0159 | 1.7432 | 1.3706 | 0.9516 |
| E | random_forest | 0.4104 | 1.3120 | 0.9778 | 0.6014 |
| F | random_forest | 0.0358 | 1.7834 | 1.4150 | 0.9971 |

## 与 30 epoch Deep Full 对比

负的 `ΔR2 vs deep full` 表示传统基线低于深度完整模型；正的 `ΔRMSE/ΔMAE/ΔHuber` 表示传统基线误差更大。

| family | best baseline | ΔR2 vs deep full | ΔRMSE vs deep full | ΔMAE vs deep full | ΔHuber vs deep full |
|---|---|---:|---:|---:|---:|
| A | random_forest | -0.1684 | 0.2098 | 0.1558 | 0.1364 |
| B | random_forest | -0.1357 | 0.1723 | 0.1282 | 0.1213 |
| C | random_forest | -0.4922 | 0.5236 | 0.3852 | 0.3724 |
| D | random_forest | -0.4347 | 0.4524 | 0.3822 | 0.3521 |
| E | random_forest | -0.2010 | 0.2508 | 0.2033 | 0.1775 |
| F | random_forest | -0.3671 | 0.3831 | 0.2814 | 0.2787 |

## 初步解释

1. Random Forest 是当前传统基线中的最强模型，但在 A-F 全部划分族上均低于 deep full。说明当前深度模型的分子描述符、512 bit Morgan fingerprint、物种/生命周期/介质/暴露上下文残差集成确实提供了传统树模型未充分利用的跨任务表示能力。
2. C/D/F 化合物严格或迁移划分中，Random Forest 的 mean R2 接近 0 或略高于 0，而 deep full 仍保持 0.40-0.49 左右。这是目前最有科学价值的差异，说明多任务共享主干与上下文建模主要提升的是新化合物外推，而不只是随机划分插值。
3. PLS、ElasticNet 等线性模型整体偏弱，提示 ECx 端点与分子结构/物种上下文之间存在明显非线性。MLP 在当前传统特征管线中未超过 Random Forest，可能与样本规模、特征标准化、单任务训练和缺少多任务共享有关。
4. 当前结论尚未包含 XGBoost 和 LightGBM。由于它们通常是强树模型基线，后续应在远端依赖安装稳定后补跑，再最终确认传统基线排名。

## 下一步

1. 远端安装或离线同步 `xgboost`、`lightgbm` 后补跑两类梯度提升树基线。
2. 做学习率敏感性小矩阵：`full` × B/C/F × `1e-4, 3e-4, 5e-4, 1e-3` × 30 epoch，用验证/测试指标判断是否将默认学习率从 `3e-4` 调至 `5e-4`。
3. 进入介质迁移实验：水相、水相+沉积物、土壤、沉积物迁移前对比，并在候选框架上做 SHAP/PDP。
