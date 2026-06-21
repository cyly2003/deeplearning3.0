# 土壤毒性预测与水相迁移学习重设计方案

更新时间：2026-06-12

## 1. 研究定位

本项目的核心目标是建立可用于土壤有机污染物生态风险阈值推导的毒性预测框架。水相数据不是最终目标，而是用于大样本预训练，帮助模型学习较稳定的分子构效关系，再迁移到土壤小样本场景中。

当前设计需要回答两个关键科学问题：

1. 如何在实测土壤毒性数据不足条件下形成可用于土壤有机生态风险阈值推导的毒性证据？
2. 模型补全的土壤毒性证据如何转化为具有不确定度约束的有机污染物目标限值？

因此，正式实验不能只比较随机 R2，也不能只用化合物严格 holdout 作为唯一评价。随机划分代表应用域内预测能力，化合物 holdout 代表新化合物压力测试，迁移学习的核心证据应来自低土壤样本条件下是否优于 soil-only 模型。

## 2. 当前主要问题

### 2.1 聚合逻辑需要重审

当前聚合逻辑位于 `qsar_tl/data/task_tables.py`。当前行为是：只有聚合键完全相同的记录才会被合并，聚合后默认以 `target_value_median` 作为模型目标。

当前聚合键包含：

- 化合物标识：`cas_number`、`dtxsid`、`chemical_name`、`smiles`
- 物种标识：`species_number`、`latin_name`、`common_name`
- 任务定义：`task_head`、`task_family`、`effect_family`、`effect_level_x`
- 目标量纲：`target_name`、`target_basis`、`unit_family_v2`、`standard_unit_v2`
- 介质和实验条件：`primary_medium`、`habitat_labels`、`organism_habitat`、`media_type`、`organism_lifestage`、`duration_bin_h`

这保证了 `EC10`、`EC20`、`EC50` 不会被聚到一起，因为 `effect_level_x` 已进入聚合键；也保证了不同目标量纲不会在聚合阶段混合。

但当前聚合键不包含：

- `reference_number`
- `test_id`
- `publication_year`

因此，不同文献或不同实验批次中，只要化合物、物种、终点、效应水平、量纲、介质和时长一致，就可能被跨文献合并。这个行为可能降低噪声，但也可能掩盖跨研究差异并造成 R2 偏高。

### 2.2 物种上下文尚未符合预期外推逻辑

当前深度模型主要使用 `species_number`、`latin_name` 等字段作为 categorical embedding。`species_number` 本质是数据库 ID，更适合记忆已见物种，不适合支撑未见物种的分类学外推。

正式模型应使用分类学层级作为主输入：

- kingdom
- phylum
- class
- order
- family
- genus
- species

`species_number` 只保留用于审计、消融和错误回查，不进入正式主模型训练特征。

### 2.3 broad soil 不适合作为最终主模型

`aggregated_task_records_soil` 是土壤介质 broad 表，包含多个目标尺度：

- `soil_mg_kg`
- `soil_ptox`
- `soil_g_ha`
- `soil_percent`
- 少量 `soil_l_ha`、`soil_seed`

这个表可用于审计、对照和多量纲共享学习框架探索，但不能作为单输出回归主结论。正式模型必须通过 target-scale adapter 或独立输出头避免不同不可换算量纲混用。

## 3. 数据清洗与 QC 规则

### 3.1 目标量纲纳入范围

第一阶段正式纳入：

- `aquatic_ptox`
- `soil_ptox`
- `soil_mg_kg`
- `soil_g_ha`

第一阶段排除：

- sediment
- percent
- L/ha
- seed treatment
- mg/organism
- mg/experimental unit
- 其他样本量过少或科学解释弱的量纲

这些排除不是永久删除，而是保留在审计表中，避免低样本或难解释目标进入主模型。

### 3.2 ECx/LCx 处理

`EC10`、`EC20`、`EC50`、`LC50` 继续映射为统一任务族，例如：

- `ECx_Mortality`
- `ECx_Growth`
- `ECx_Reproduction`

数字部分作为 `effect_level_x`：

- `EC10` -> `effect_level_x = 10`
- `LC50` -> `effect_level_x = 50`

`effect_level_x` 是预测条件，不是目标值。它必须：

- 作为数值上下文输入模型；
- 参与聚合键；
- 在绘图和解释报告中保留；
- 不用于拆分成过多稀疏任务头。

NOEC 和 LOEC 与 ECx 的毒理含义不同，保留为独立任务族：

- `NOEC_*`
- `LOEC_*`

### 3.3 离群值处理

在 `target_name + medium_domain + task_head` 分组内进行 robust outlier 检查：

- 组内样本数 `n >= 50`：robust z-score > 4 的样本直接排除；
- 组内样本数 `n < 50`：不自动删除，只进入人工审计列表；
- 非正毒性值、明显单位错误、无法解释的极端值可直接排除并记录原因。

建议新增字段：

- `tox_qc_status`
- `tox_qc_reason`
- `robust_z`
- `qc_group_n`

### 3.4 跨文献冲突与加权聚合

采用“最新文献优先 + 冲突标记 + 加权平均”的聚合策略。

默认规则：

- 冲突阈值：0.3 log unit；
- 最新发表年份权重为 1.0；
- 每早 5 年权重乘 0.8；
- 最低权重不低于 0.3；
- 聚合使用加权平均，不是加权求和；
- 新旧文献差异小于等于 0.3 log unit 时，可合并；
- 差异大于 0.3 log unit 时，保留旧文献但降权，并标记 `cross_reference_conflict`。

建议新增字段：

- `publication_year`
- `reference_number`
- `test_id`
- `reference_weight`
- `cross_reference_conflict_flag`
- `cross_reference_range`
- `weighted_target_value`
- `weighted_target_value_std`
- `source_reference_count`
- `source_test_count`

这些字段进入审计和预测报告，但 `reference_number/test_id/publication_year` 不进入训练特征。

## 4. 特征工程设计

### 4.1 分子特征

分子特征分四套对比：

- D0：当前 8 个 RDKit descriptor + Morgan 512 bit；
- D1：扩展 30-80 个筛选 RDKit 2D descriptor + Morgan 512 bit；
- D2：特征选择后的少量 RDKit descriptor + Morgan 512 bit；
- D3：全量 RDKit descriptor + Morgan 512 bit。

第二阶段再加入先验描述符聚类头。

### 4.2 先验描述符聚类

描述符聚类不应按相关性随意聚类，而应按化学含义预分组，例如：

- 分子大小/质量：MolWt、HeavyAtomCount、原子数、键数；
- 疏水性/分配行为：MolLogP、MolMR；
- 极性/氢键：TPSA、HBD、HBA；
- 拓扑复杂度：RingCount、RotatableBonds、FractionCSP3；
- 芳香性/不饱和度：芳香环、脂肪环、双键特征；
- 电荷/电离相关：formal charge、酸碱性描述符，如可计算；
- 结构片段：Morgan 或 MACCS 相关表示。

使用方式：

- 固定簇级表示用于 Williams leverage 应用域；
- 神经网络 cluster head 作为第二阶段消融项；
- 不把 cluster head 作为第一版主模型必需项，避免一开始组合爆炸。

### 4.3 物种与分类学特征

正式主模型使用 taxon 层级：

- kingdom
- phylum
- class
- order
- family
- genus
- species

同时保留：

- latin_name
- organism_lifestage
- habitat_labels
- organism_habitat
- primary_medium
- media_type

`species_number` 不作为正式主输入，只用于：

- 审计；
- 消融；
- 高残差样本回查；
- 对比 ID 记忆与 taxon 外推的差异。

### 4.4 暴露和任务上下文

上下文包括：

- `effect_level_x`
- `duration_bin_h`
- `duration_log1p_h`
- `duration_sqrt_h`
- `duration_inv_log1p_h`
- duration RBF 特征
- lifestage
- medium/domain
- target scale

`effect_level_x` 是终点条件变量，不是目标值。它用于让同一 `ECx_*` 任务头学习 EC10/EC20/EC50 的连续响应关系。

## 5. 模型架构

### 5.1 第一阶段主架构

采用：

```text
molecular encoder
taxon encoder
context encoder
        |
shared trunk
        |
target-scale / medium adapter
        |
endpoint task head
```

核心原则：

- 分子、分类学、上下文共享主干；
- 不同介质和目标尺度通过 adapter 校正；
- endpoint/effect family 作为 task head；
- 不同不可换算目标不能直接共用同一个输出头。

### 5.2 输出头设计

推荐结构：

```text
shared trunk -> scale/medium adapter -> endpoint head
```

第一阶段 target scale/medium 包括：

- aquatic_ptox
- soil_ptox
- soil_mg_kg
- soil_g_ha

endpoint head 包括：

- ECx_Mortality
- ECx_Growth
- ECx_Reproduction
- NOEC_Growth
- LOEC_Reproduction
- 其他满足样本阈值的任务头

### 5.3 水相到土壤迁移

迁移不是强行共享水相和土壤输出头，而是迁移分子构效知识。

训练流程：

1. 水相 pTox 大样本预训练 shared molecular encoder / shared trunk；
2. 土壤阶段初始化 shared trunk；
3. 冻结 molecular encoder，训练 soil adapter + endpoint head；
4. 解冻全模型，小学习率微调；
5. 与 soil-only from scratch 在相同 soil test set 上比较。

## 6. 应用域与不确定度

### 6.1 不确定度范围

第一版只输出：

- 模型不确定度；
- 外推/应用域不确定度。

不把数据冲突不确定度作为最终预测不确定度。跨文献冲突只用于 QC、聚合降权和审计。

### 6.2 模型不确定度

采用 5-seed ensemble：

- `y_pred_mean`
- `y_pred_std_ensemble`
- `y_pred_min`
- `y_pred_max`

该不确定度反映不同随机初始化和训练过程导致的模型预测分歧。

### 6.3 化学应用域

主方法：Williams leverage。

Williams 特征空间：

- 先验描述符聚类固定表示；
- Morgan fingerprint PCA；
- 不直接使用神经网络 embedding。

补充方法：

- Morgan fingerprint 最近邻 Tanimoto similarity。

主文以 Williams 为主，补充材料比较 Williams 和 Tanimoto 的覆盖率与误差。如果 Tanimoto 覆盖更宽且验证误差没有恶化，可作为扩展 AD。

### 6.4 物种应用域

物种 AD 基于 taxon distance。

定义：

- 同 species：`taxon_distance = 0`
- 同 genus：`taxon_distance = 1`
- 同 family：`taxon_distance = 2`
- 同 order：`taxon_distance = 3`
- 超过 order 或 taxon 缺失严重：`taxon_distance >= 4`

标签：

- species/genus/family：`in-domain`
- order：`near-domain`
- 超过 order：`out-of-domain`

输出字段：

- `taxon_distance`
- `nearest_train_taxon_rank`
- `nearest_train_taxon_name`
- `species_AD_flag`

### 6.5 综合应用域

综合 AD：

- chemical in-domain 且 species in-domain：`overall_AD = in-domain`
- 任一 near-domain：`overall_AD = near-domain`
- 任一 out-of-domain：`overall_AD = out-of-domain`

训练评估预测报告应保留 AD 字段，用于绘制：

- AD 分组下的 R2/RMSE/MAE；
- Williams plot；
- taxon distance vs absolute error；
- ensemble std vs absolute error；
- Tanimoto similarity vs residual。

## 7. 数据划分体系

所有主要模型都需要在多种划分方式下比较，不再只依赖单一 holdout。

### 7.1 保留 A-F 兼容划分

继续保留：

- A：7:2:1 random train/finetune/test；
- B：8:2 random train/test；
- C：8:2 chemical holdout；
- D：5-fold chemical group CV；
- E：5-fold random CV；
- F：7:2:1 chemical adapt。

A/B/E 代表应用域内或近同分布性能；C/D/F 是化合物外推压力测试。

### 7.2 新增 AD-aware 划分

测试集按应用域分层：

- chemical in-domain
- chemical near-domain
- chemical out-of-domain
- species in-domain
- species near-domain
- species out-of-domain

目标是把“应用域内预测”和“外推预测”分开解释。

### 7.3 low-soil-data transfer 划分

固定 soil test set，土壤训练/微调样本比例设为：

- 10%
- 20%
- 50%
- 100%

比较：

- soil-only from scratch；
- aquatic-pretrained + soil adapter/head finetune。

迁移价值的核心证据是：在低土壤样本比例下，迁移模型显著优于 soil-only。

### 7.4 target-scale-specific 划分

分别对以下目标尺度输出结果：

- soil_mg_kg
- soil_ptox
- soil_g_ha
- aquatic_ptox

多量纲共享模型可以共同训练，但评估必须按 target scale 单独汇总。

## 8. 实验矩阵

### 8.1 baseline 模型

传统模型：

- Random Forest
- XGBoost
- LightGBM
- ExtraTrees
- PLS
- ElasticNet
- MLP

评价指标：

- R2
- RMSE
- MAE
- Huber loss
- AD 分层指标

### 8.2 深度模型

主模型：

- shared molecular/taxon/context trunk
- scale/medium adapter
- endpoint heads

特征版本：

- D0 当前 8 RDKit + Morgan 512
- D1 扩展 30-80 RDKit + Morgan 512
- D2 特征选择少量 RDKit + Morgan 512
- D3 全量 RDKit + Morgan 512
- 第二阶段：descriptor cluster head

### 8.3 消融实验

必做消融：

- no fingerprint
- no descriptors
- no taxon
- no lifestage
- no duration
- no medium adapter
- no target-scale adapter
- no molecular residual
- species_number 替代 taxon
- taxon + species_number

重点回答：

- 分子构效知识是否主要来自 molecular encoder？
- taxon 层级是否优于 species ID？
- soil adapter 是否捕捉介质差异？
- 水相预训练是否真正改善土壤低样本预测？

## 9. 预测报告与绘图输出

### 9.1 训练评估预测报告

每个样本一行，建议字段：

- split_name
- split_part
- model_name
- seed
- target_name
- medium_domain
- task_head
- effect_level_x
- duration_bin_h
- aggregate_id
- result_ids
- reference_number
- publication_year
- cas_number
- chemical_name
- smiles
- taxon fields
- y_true
- y_pred
- residual
- abs_error
- ensemble_mean
- ensemble_std
- williams_leverage
- williams_h_star
- williams_AD_flag
- tanimoto_nn
- tanimoto_AD_flag
- taxon_distance
- species_AD_flag
- overall_AD_flag

审计字段不进入模型训练特征，只用于回查、绘图和解释。

### 9.2 图表清单

论文和学位论文建议输出：

- 目标值分布图，按 target scale 和 medium 分面；
- QC 前后样本量流图；
- 离群值审计图；
- cross-reference conflict 分布；
- true vs predicted，按 AD flag 着色；
- residual distribution，按 split 和 AD 分组；
- Williams plot；
- Tanimoto similarity vs abs error；
- taxon distance vs abs error；
- ensemble std vs abs error；
- low-soil-data transfer 曲线；
- feature version 对比图；
- adapter 消融图；
- SHAP/PDP/ALE 解释图。

## 10. 代码改造清单

### 10.1 数据层

新增或重构：

- `qsar_tl/data/toxicity_qc.py`
- `qsar_tl/data/reference_weighting.py`
- `qsar_tl/data/qc_aggregation.py`
- `scripts/audit_toxicity_outliers.py`
- `scripts/audit_reference_conflicts.py`

输出表：

- `target_records_qc`
- `task_records_qc`
- `aggregated_task_records_qc`
- `aggregated_task_records_aquatic_ptox_qc`
- `aggregated_task_records_soil_ptox_qc`
- `aggregated_task_records_soil_mg_kg_qc`
- `aggregated_task_records_soil_g_ha_qc`

### 10.2 特征层

新增：

- descriptor set registry；
- RDKit descriptor expansion；
- descriptor feature selection；
- descriptor cluster definitions；
- Morgan PCA cache；
- taxon hierarchy encoder；
- taxon distance calculator。

### 10.3 模型层

新增：

- molecular encoder variants；
- taxon encoder；
- target-scale/medium adapter；
- endpoint head registry；
- two-stage finetune runner；
- low-soil-data split generator。

### 10.4 应用域与不确定度

新增：

- Williams leverage calculator；
- Tanimoto nearest-neighbor calculator；
- species AD calculator；
- ensemble prediction combiner；
- AD-aware metric summary。

### 10.5 报告层

新增：

- sample-level prediction report；
- AD-stratified metrics；
- high residual audit table；
- figure generation scripts；
- QC and aggregation summary document。

## 11. 实施优先级

第一阶段：数据可信度

1. 加入 `publication_year/reference_number/test_id` 到派生表。
2. 审计 endpoint 数字、量纲、介质和目标分布。
3. 实现 outlier QC。
4. 实现 reference-aware weighted aggregation。
5. 生成 QC 后介质/量纲子表。

第二阶段：模型架构修正

1. taxon 层级编码替换 species ID 主输入。
2. scale/medium adapter + endpoint head。
3. D0/D1/D2/D3 分子特征版本。
4. sample-level prediction report。

第三阶段：实验矩阵

1. random / A-F 兼容实验。
2. chemical holdout 压力测试。
3. AD-aware 实验。
4. low-soil-data transfer 实验。
5. baseline 和 deep 对比。

第四阶段：解释与阈值应用

1. SHAP/PDP/ALE。
2. AD-stratified residual analysis。
3. ensemble uncertainty。
4. 预测毒性证据到 SSD/HC5/PNEC/目标限值的传导框架。

## 12. 当前不应继续沿用的做法

- 不再把 broad soil 单输出结果作为最终主结论。
- 不再把 `species_number` 作为正式主模型的物种外推输入。
- 不再把水相 pTox 和土壤 mg/kg 放进同一个输出头。
- 不再只报告随机划分 R2。
- 不再把 Huber loss 当作离群值清洗的替代方案。
- 不再静默跨文献聚合。

## 13. 推荐的主结论逻辑

正式论文或学位论文中，建议按以下逻辑组织结果：

1. QC 后土壤毒性数据质量显著改善，聚合规则更符合文献证据权重。
2. 在应用域内，土壤模型可以给出稳定预测。
3. 化合物 holdout 下性能下降，说明新化合物外推仍是难点，但这是模型边界而非失败。
4. 水相预训练在低土壤样本比例下改善土壤预测，证明水相大样本可提供可迁移的分子构效知识。
5. taxon 层级物种上下文和物种 AD 能解释不同物种覆盖范围下的预测可靠性。
6. Williams chemical AD + taxon species AD + ensemble uncertainty 可以为土壤毒性证据补全和风险阈值推导提供可靠性约束。
