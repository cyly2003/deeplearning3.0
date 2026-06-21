# v1.0.0 终点任务分配方案

## 核心原则

本项目不把“化合物 scaffold 外推”作为主线，而是以 ECOTOX 全量生态毒性数据为基础，建立覆盖更宽物种、化合物和介质范围的 QSAR 迁移学习框架。主任务应集中在毒性浓度终点，辅助任务用于提供毒性机制、效应类型和暴露场景知识，BAF/BCF 等累积因子用于解释与未来辅助分支，不直接混入毒性浓度目标。

## 主任务

主任务用于模型主要优化、论文核心指标报告和土壤小样本迁移对比。

| 终点组 | 纳入规则 | 科研含义 | 任务权重 |
| --- | --- | --- | --- |
| EC/LC | ECx、LCx 及等价浓度效应终点 | 急慢性毒性强度，是 QSAR 毒性建模的主轴 | 高 |
| NOEC | NOEC、NOEL | 无观察效应浓度，偏生态风险阈值 | 高 |
| LOEC | LOEC、LOEL | 最低观察效应浓度，连接毒性阈值与风险判定 | 高 |

说明：NOEL/LOEL 在当前数据结构中与 NOEC/LOEC 同为阈值浓度信息，因此并入主任务，而不是丢弃。

## 辅助毒性任务

辅助任务不作为最终论证的主要性能指标，但参与多任务学习，为主任务提供结构-效应知识。

| 终点组 | 当前处理 | 原因 |
| --- | --- | --- |
| ICx | 辅助毒性任务 | 常见抑制浓度，能补充生长、繁殖、生理抑制信息 |
| ACx | 辅助毒性任务 | 活性/效应浓度，补充机制相关效应 |
| EDx/LDx | 辅助毒性任务 | 与毒性效应有关，但定义和暴露语义较杂 |
| BMDx/BMCx | 辅助毒性任务 | 剂量-反应建模相关终点，可提供剂量敏感性知识 |
| MATC | 辅助毒性任务 | 介于 NOEC/LOEC 的慢性毒性浓度摘要 |

这些任务默认低于主任务权重，避免“数据多但定义弱”的辅助终点稀释 EC/LC、NOEC、LOEC 的主线目标。

## BAF/BCF 累积因子

BAF、BCF、BCFD 当前不进入毒性浓度目标。原因是当前 target builder 主要从暴露浓度字段构建 pTox 或 mg/kg 目标，而 BAF/BCF 本质是生物富集/生物浓缩因子，不是外部暴露浓度。把它们直接转为 pTox 会造成目标语义错误。

推荐方案：

1. 当前 v1.0.0：保留 BAF/BCF 记录的识别和排除原因，用于数据审计和解释。
2. 下一阶段：单独建立 `log_baf`、`log_bcf` 或 `bioaccumulation_factor` 辅助分支。
3. 论文解释：将 BAF/BCF 作为“污染物环境行为、生物有效性和食物链累积风险”的解释支撑，而不是毒性浓度预测主指标。

## 暂缓任务

| 终点组 | 当前处理 | 后续方向 |
| --- | --- | --- |
| NR | 暂不进入回归 | 更适合做删失数据、无响应分类或阈值分类 |
| LT/ET | 暂不进入浓度回归 | 更适合做 time-to-event 或 survival 任务 |
| 口服剂量类 | 暂不进入当前生态浓度回归 | 需要剂量单位和暴露路径单独建模 |

## 效应家族

当前放宽后纳入的效应家族包括：Mortality、Growth、Reproduction、Population、Immobilization、Development、Behavior、Morphology、Physiology、Biochemical、GeneticDamage、Injury、Feeding、Molting、Accumulation。

这比只保留死亡/生长/繁殖更宽，但仍保留清晰的 effect_family 字段，便于后续按生态效应解释模型误差和 SHAP 结果。

## 样本保留变化

基于 `outputs/derived/modeling_dataset_v1_0_0_rebuild.sqlite` 的重建结果：

| 阶段 | 放宽前 | 放宽后 |
| --- | ---: | ---: |
| included_task_records | 291,760 | 792,765 |
| aggregated_task_records | 121,240 | 329,470 |
| qc_included_records | - | 789,823 |
| aggregated_task_records_qc | - | 326,612 |
| aquatic pTox QC | - | 155,066 |
| soil pTox QC | - | 11,737 |
| aquatic+soil pTox QC | - | 166,803 |

主要仍排除 NR、BAF/BCF、口服剂量和 LT50 等不适合当前浓度回归主线的记录。

## 配置建议

当前推荐权重：

| 任务组 | 建议权重 |
| --- | ---: |
| main_toxicity | 1.00 |
| toxicity_aux | 0.35 |
| bioaccumulation_aux | 0.20 |

当前代码中 BAF/BCF 仍未进入 bioaccumulation_aux 训练分支；该权重是为下一阶段单独构建富集因子目标预留。

