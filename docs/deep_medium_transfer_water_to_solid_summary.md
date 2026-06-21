# Deep Medium Transfer Water To Solid Summary

更新时间：2026-06-11

## 运行范围

本轮作为介质迁移实验的链路验证，生成水相到固相的 medium transfer split，并运行 full 深度模型。

- 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
- 结果目录：`outputs/experiments/deep_medium_transfer_water_to_solid`
- split：`M_water_to_solid`
- split 规则：水相/FW/SW/AQU 等进入 train，土壤/沉积物/ART/NAT/UKS/LIT/MIN 等进入 test，未知介质进入 valid
- split 规模：train 73,164；test 1,548；valid 6,013
- 模型：`full`
- 训练轮数：`epochs=30`
- 学习率：`0.0005`
- batch size：`512`
- 设备：远端 `cuda:0`

## 输出文件

- 指标：`outputs/experiments/deep_medium_transfer_water_to_solid/summary_test_metrics.csv`
- 收敛曲线：`outputs/experiments/deep_medium_transfer_water_to_solid/summary_training_convergence.csv`
- 图表目录：`outputs/experiments/deep_medium_transfer_water_to_solid/figures`

## 测试集主要结果

| task_head | n | R2 | RMSE | MAE | Huber |
|---|---:|---:|---:|---:|---:|
| ECx_Mortality | 575 | -0.6996 | 4.4756 | 3.5939 | 3.1282 |
| ECx_Growth | 437 | -3.3334 | 4.9485 | 4.5295 | 4.0445 |
| ECx_Reproduction | 107 | -11.4384 | 5.8245 | 5.6124 | 5.1125 |
| NOEC_Mortality | 101 | -16.2964 | 5.4777 | 5.1846 | 4.6852 |
| ECx_Population | 85 | -1.5174 | 4.1868 | 2.9904 | 2.5383 |
| NOEC_Growth | 81 | -3.4173 | 4.3511 | 3.8472 | 3.3643 |

## 解释与风险

1. 该迁移结果很差，不能直接解释为模型结构完全失败。当前训练集主要是水相 `pTox mol/L`，测试集主要是固相 `-log10 mg/kg`，两者不是严格同一物理目标尺度。
2. 当前结果首先证明介质迁移链路可以运行，并揭示跨介质目标定义不统一会造成严重分布偏移。
3. 后续正式介质迁移应先做“迁移前对比”：水相内、固相内、土壤/沉积物内分别建模，确认各介质自身可预测性，再考虑跨介质迁移。
4. 若要做水相到土壤迁移，建议引入介质相关转换或辅助环境行为特征，例如 Koc、logKow、溶解度、有机碳归一化浓度等，否则模型只能把两个单位体系硬拟合到同一输出空间。

## 下一步

1. 生成介质内随机/化合物严格 split，用于水相、水相+固相、固相迁移前对比。
2. 将 soil 与 sediment 从 `target_basis` 或原始 `media_type` 中尽可能拆分；当前聚合表对这两类的语义仍不够明确。
3. 在正式 SHAP/PDP 前，先完成介质目标定义审计，避免解释跨单位伪相关。
