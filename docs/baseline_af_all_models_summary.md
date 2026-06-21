# Traditional Baseline A-F All Models Summary

更新时间：2026-06-11

## 运行范围

本轮补齐远端缺失依赖后，完成 A-F 全划分下 7 类传统机器学习基线汇总。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 已合并结果目录：`outputs/experiments/baseline_af_all_models_limit10000`
- 可用模型结果来源：`outputs/experiments/baseline_af_available_limit10000`
- Boosting 结果来源：`outputs/experiments/baseline_af_boosting_limit10000`
- 模型：`random_forest`、`xgboost`、`lightgbm`、`pls`、`extra_trees`、`elastic_net`、`mlp`
- 划分范围：A、B、C、D 5 折、E 5 折、F
- 损失设定：Huber loss，`delta=1.0`
- 样本限制：`limit=10000`

## 远端依赖状态

已在远端用户目录安装并验证：

- `xgboost 3.2.0`
- `lightgbm 4.6.0`
- `shap 0.48.0`
- `numpy 1.26.4`

安装过程中曾出现 `shap 0.52.0` 拉起 `numpy 2.0.2` 的不兼容风险，已降级 SHAP 并将 NumPy 固定回 1.26.4。

## 输出文件

- 总指标：`outputs/experiments/baseline_af_all_models_limit10000/summary_metrics.csv`
- 测试集指标：`outputs/experiments/baseline_af_all_models_limit10000/summary_test_metrics.csv`
- split × model 汇总：`outputs/experiments/baseline_af_all_models_limit10000/summary_split_model_metrics.csv`
- family × model 汇总：`outputs/experiments/baseline_af_all_models_limit10000/summary_family_model_metrics.csv`
- 相对 30 epoch deep full 的差值：`outputs/experiments/baseline_af_all_models_limit10000/summary_family_model_delta_vs_deep_full.csv`
- 图表目录：`outputs/experiments/baseline_af_all_models_limit10000/figures`

## 最优传统基线

| family | best baseline | mean R2 | mean RMSE | mean MAE | mean Huber |
|---|---|---:|---:|---:|---:|
| A | random_forest | 0.2770 | 1.3671 | 0.9949 | 0.6181 |
| B | random_forest | 0.4190 | 1.2656 | 0.9294 | 0.5659 |
| C | lightgbm | 0.0197 | 1.7836 | 1.4052 | 0.9912 |
| D | lightgbm | -0.0100 | 1.7371 | 1.3493 | 0.9374 |
| E | random_forest | 0.4104 | 1.3120 | 0.9778 | 0.6014 |
| F | lightgbm | 0.0629 | 1.7570 | 1.3929 | 0.9692 |

## 与 Deep Full 对比

下表为每个 family 中最强传统基线相对 30 epoch deep full 的差值。负的 `ΔR2` 表示传统基线低于 deep full。

| family | best baseline | ΔR2 vs deep full | ΔRMSE vs deep full | ΔMAE vs deep full | ΔHuber vs deep full |
|---|---|---:|---:|---:|---:|
| A | random_forest | -0.1684 | 0.2098 | 0.1558 | 0.1364 |
| B | random_forest | -0.1357 | 0.1723 | 0.1282 | 0.1213 |
| C | lightgbm | -0.4721 | 0.4974 | 0.3594 | 0.3515 |
| D | lightgbm | -0.4288 | 0.4463 | 0.3609 | 0.3379 |
| E | random_forest | -0.2010 | 0.2508 | 0.2033 | 0.1775 |
| F | lightgbm | -0.3400 | 0.3567 | 0.2592 | 0.2509 |

## 初步解释

1. 补齐 XGBoost/LightGBM 后，传统基线仍未超过 deep full。A/B/E 中 Random Forest 最强；C/D/F 中 LightGBM 略优于 Random Forest，但 R2 仍接近 0。
2. C/D/F 的差距最大，说明传统单任务表格模型在新化合物或迁移划分下外推能力有限。深度多任务模型的共享主干和上下文残差集成仍有明显优势。
3. XGBoost 未进入任何 family 的前三强，提示当前 XGBoost 超参数或 pseudo-Huber 目标并不适合该特征表。后续若要把 XGBoost 作为强基线，应单独做树深、学习率、采样率和目标函数调参。
4. LightGBM 首次运行失败是因为特征名含特殊字符。已在 `TabularPreprocessor` 中加入安全特征名映射，修复后全矩阵运行成功。

## 下一步

1. 若要发表级传统基线，需要对 RF/LightGBM/XGBoost 做小规模超参数搜索。
2. 进入介质迁移实验时，应避免把水相 `pTox mol/L` 与固相 `-log10 mg/kg` 的结果简单等价解释。
