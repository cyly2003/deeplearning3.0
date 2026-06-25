# 当前工作树整理与后续实验方案 2026-06-25

本文用于回答三个问题：

1. 当前分支、脚本和实验主线到底是什么关系。
2. 当前最优策略是怎么一步步来的。
3. 下一步是否应先跑新消融实验，以及更合理的后续实验方案是什么。

## 1. 当前工作树状态

当前本地分支：

- `codex/authority-toxicity-binning-matrix`

远端同名分支最后提交停在：

- `e24b7bf Record authority binning HPO results`

这意味着：Git 历史里正式提交到远端的内容主要到 v1.2.6 authority-bin HPO；但本地工作区已经继续推进到 v1.2.13，包括 censored loss、ordinal、AD audit、proxy、seed stability、5-seed ensemble 和 validation-policy 检查。因此现在分支名不能准确代表当前工作内容。

当前未提交内容大致分成四类：

1. 核心训练/评估代码改动
   - `qsar_tl/training/train.py`
   - `qsar_tl/training/deep_experiment.py`
   - `qsar_tl/training/deep_train.py`
   - `qsar_tl/training/toxicity_binning.py`
   - `qsar_tl/training/censored_loss.py`
   - `qsar_tl/training/ordinal_binning.py`
   - `qsar_tl/evaluation/application_domain.py`
   - `qsar_tl/modeling/network.py`

2. 版本化实验启动脚本
   - `scripts/run_v1_2_7_censored_ordinal_ad_first_batch_remote.sh`
   - `scripts/run_v1_2_8_censored_loss_matrix_remote.sh`
   - `scripts/run_v1_2_9_proxy_source_rule_remote.sh`
   - `scripts/run_v1_2_10_f100_proxy_alpha_remote.sh`
   - `scripts/run_v1_2_11_f100_seed_stability_remote.sh`
   - `scripts/run_v1_2_12_f100_5seed_confirmation_remote.sh`
   - `scripts/run_v1_2_13_anchor_validation_policy_remote.sh`

3. 汇总、审计、可视化脚本
   - `scripts/summarize_deep_runs.py`
   - `scripts/summarize_ad_gate.py`
   - `scripts/summarize_seed_ensembles.py`
   - `scripts/audit_censored_records.py`
   - `scripts/audit_prediction_application_domain.py`
   - `scripts/audit_proxy_bins.py`
   - `scripts/plot_experiment_progress.py`

4. 追溯与说明文档
   - `docs/experiment_registry.csv`
   - `docs/experiment_decision_log.md`
   - `docs/code_organization_and_experiment_scripts.md`
   - `docs/experiment_progress_visualization_20260625.md`
   - 本文档

建议不要继续在这个分支上无限追加新实验。更合理的做法是：

1. 先把当前 v1.2.7-v1.2.13 相关代码、脚本和文档整理提交。
2. 当前分支可以保留为 `codex/authority-toxicity-binning-matrix` 的延伸提交，但提交信息必须明确写明已经包含 v1.2.13。
3. 下一阶段新开分支，例如 `codex/v1-2-14-ad-censored-analysis`。

## 2. 当前最优策略是怎么来的

当前主线不是一次实验直接得到的，而是一条递进筛选链。

### 2.1 v1.2.1 到 v1.2.2：发现 source Tanimoto 是第一条有效迁移线索

v1.2.1 修正了 medium audit、effect-level、z-score 等问题后，发现 test 表现仍弱，说明仅靠原始迁移流程不足。

v1.2.2 的第一批迁移优化矩阵显示：

- `source_tanimoto alpha=1.0` 明显优于前一轮 corrected baseline。
- CORAL 等方向没有成为主线。

因此，`source_tanimoto alpha=1.0` 成为后续迁移路线的基础。

### 2.2 v1.2.3：effect-level weighting 没有成为主线

effect-level weighting 有局部信号，但不够稳定。

结论：

- 作为诊断保留。
- 不作为默认策略。

### 2.3 v1.2.4-v1.2.5：建立 soil-only 对照

这两轮说明一个关键事实：

- 低土壤样本下，迁移学习有价值，尤其在任务覆盖不足时。
- 土壤样本足够时，soil-only 很强，甚至能超过早期迁移策略。

因此，后续不能简单说“transfer 一定优于 soil-only”，必须做 shared-task 公平比较。

### 2.4 v1.2.6：authority CE-bin 成为迁移正则

authority-based toxicity binning 带来了关键改进：

- f20 transfer 最优：CE-bin loss weight `0.05`。
- f100 transfer 最优：CE-bin loss weight `0.025`。

解释：

- 它不是 soil-only 的通用增强器。
- 它更像是迁移阶段的风险等级正则化，让模型在水相到土壤微调时不只拟合连续数值，也吸收毒性等级先验。

### 2.5 v1.2.7：ordinal 不替代 CE-bin，AD audit 变重要

ordinal loss 比 no-bin 稍好，但没有超过普通 CE authority-bin。

AD audit 显示：

- 不能只看化学相似度。
- 物种、life stage、task-family 是否覆盖，是模型可靠性的重要层。

### 2.6 v1.2.8：censored loss 对 f100 有明确帮助

f100 中 `censored_weight=0.01` 明确改善了主线。

但更大 censored weight 会改变 ECx/LOEC/NOEC 之间的平衡，不能盲目加权。

### 2.7 v1.2.9-v1.2.10：proxy 有信号，但不够干净

proxy-distance 和 Tanimoto+proxy 在 f100 上有局部收益，但：

- 不同指标间有冲突。
- endpoint-family 间有冲突。
- species extrapolation 层可能变差。
- alpha 敏感性没有解决问题。

因此 proxy 不作为主线，只作为诊断线索。

### 2.8 v1.2.11-v1.2.12：多 seed 证明单 seed 不可靠，ensemble 稳定性最好

v1.2.11 说明 seed42 的强结果不能代表单模型稳定表现。

v1.2.12 的 5-seed ensemble 成为当前最稳结果：

- anchor Tanimoto 5-seed ensemble：MAE `0.9277`，RMSE `1.2368`，R2 `0.5388`。

关键解释：

- test 集没有换。
- 提升来自 seed averaging/ensemble 降低随机训练波动。

### 2.9 v1.2.13：释放 finetune_validation 不值得作为主策略

`finetune_validation_fraction=0.0` 与 `0.2` control 在 R2/RMSE 上几乎持平，但 MAE 与 Huber loss 略差。

结论：

- 保留 `finetune_validation_fraction=0.2`。
- val0 只作为敏感性分析。

## 3. 对你提出的几个判断的回应

### 3.1 是否多做几次 test 轮换就行

可以做，但不能理解为“多抽几个 test，选一个好看的结果”。

更合理的形式是外层 repeated split 或 repeated chemical holdout：

- 每一次 split 都是一个独立外层测试。
- 每一次 split 内部只能用 train/validation 选模型。
- 最后报告所有外层 test 的均值、标准差、最差分位数和 AD 分层。

这样可以回答：

- 当前主线是否只对某一个固定 test 偶然有效。
- 哪些 split 类型下稳定。
- 哪些化合物/物种/任务族外推下失败。

推荐叫法：

- `v1.2.14 locked repeated outer split confirmation`

不建议把它叫做“test 轮换调参”。它应该是最终确认，不是继续调参。

### 3.2 当前物种特征是不是 taxonomy 层级 embedding

是的，当前模型训练输入已经包含 taxonomy/context 类别特征。

配置中 `features.species.encoder=taxonomy_context`，包含：

- `kingdom`
- `phylum`
- `class`
- `order`
- `family`
- `genus`
- `species`
- `life_stage`
- `habitat`
- `primary_medium`
- `exposure_route`
- `duration_h`
- `log1p_duration_h`

训练实现中也包含：

- `latin_name`
- `kingdom`
- `phylum`
- `class_name`
- `tax_order`
- `family`
- `genus`
- `species`
- `taxon_group_l1/l2/l3`
- `primary_medium`
- `habitat_labels`
- `organism_habitat`
- `media_type`
- `organism_lifestage`
- `target_basis`
- `effect_family`

所以，后续不应该简单说“加入 taxonomy embedding”。已经有了。

更准确的新方向是：

1. 基于当前 taxonomy embedding，构建更适合本模型的 AD 评估方法。
2. 区分模型输入特征和 AD 可靠性判断。
3. 用 taxonomy 距离、species seen、life-stage seen、task-family seen、chemical similarity、prediction uncertainty 共同构建模型适用域。

### 3.3 ECx/LOEC/NOEC 是否应在训练中加权

你说得对：ECx、LOEC、NOEC 本身的实验生成机制和不确定性不同，直接在训练中强行加权可能会偏离 QSAR 主线，也可能引入主观性。

更合适的做法是：

- 训练阶段暂时保持当前主线。
- 评价阶段分开报告 ECx/LOEC/NOEC。
- 风险评估阶段再对不同 endpoint/effect 设定应用层权重。

也就是说：

- QSAR 主线：连续毒性预测与迁移泛化。
- 风险评估扩展：不同 endpoint/effect 对生态风险贡献不同，可以后续建权重。

## 4. 适合当前模型的新 AD 评估方法建议

建议命名为：

`Chem-Species-Task Applicability Domain (CST-AD)`

或中文：

“化学-物种-任务联合适用域评估”。

它不是单一阈值，而是多维分层。

### 4.1 化学 AD

已有基础：

- Morgan Tanimoto 到训练集最大相似度。
- Williams leverage。
- proxy descriptor distance。

建议保留，但不要单独决定模型可靠性。

### 4.2 物种 AD

已有基础：

- taxonomy prefix similarity。
- species seen train。
- life stage seen train。

建议增强为分级：

- Tier S0：同物种 + 同 task-family 见过。
- Tier S1：同物种见过，但 task-family 未见。
- Tier S2：同属/同科见过。
- Tier S3：只在更高分类阶元相近。
- Tier S4：taxonomy 远离或缺失。

这比简单 `species_in_domain=True/False` 更适合生态毒理解释。

### 4.3 任务 AD

任务维度包括：

- task_family：ECx/LOEC/NOEC。
- task_head：ECx_Mortality、NOEC_Growth 等。
- effect_level_x 是否可用。
- target_basis、unit family、medium_domain 是否一致。

建议构建：

- task_head seen/unseen。
- task_family seen/unseen。
- effect endpoint seen/unseen。
- target basis seen/unseen。

### 4.4 模型不确定性 AD

利用 5-seed ensemble：

- 预测均值。
- 预测标准差。
- seed 间最大差值。
- prediction interval。

这一步不改变模型，只改变可靠性判断。

### 4.5 CST-AD 最终输出

每条预测给出：

- chemical_AD_tier
- species_AD_tier
- task_AD_tier
- ensemble_uncertainty_tier
- final_AD_label

例如：

- `AD-A`: chemical in-domain + species/task seen + low uncertainty
- `AD-B`: chemical in-domain + taxonomy nearby + moderate uncertainty
- `AD-C`: chemical ok but species/task extrapolation
- `AD-D`: chemical extrapolation or high uncertainty

这样可以形成论文中的一个方法亮点，而不是只说“用了 Tanimoto 和 Williams plot”。

## 5. censored 数据：先分析分布，再决定策略

你提出“先单独分析 censored 样本分布”是正确的。

下一步不应直接上 Tobit 或复杂 likelihood，而应先回答：

1. censored 样本主要来自哪些 endpoint family？
2. `>`、`>=`、`<`、`<=` 分别有多少？
3. censored 是否集中在高毒性区间？
4. censored 是否主要来自水相、土壤、某些物种或某些单位？
5. censored 样本是否更容易出现在 LOEC/NOEC？
6. 当前模型在 censored 样本附近是否更容易高估或低估？

### 5.1 删失方向是什么意思

如果记录是：

- `> 100 mg/L`：真实值大于 100，说明只知道下界。
- `< 0.1 mg/L`：真实值小于 0.1，说明只知道上界。

转换到毒性强度时要特别小心。

如果目标是 `pTox = -log10(concentration)`：

- 浓度越小，pTox 越大，毒性越强。
- 原始浓度 `< 0.1` 表示真实浓度更小，pTox 更大，是高毒方向的下界。
- 原始浓度 `> 100` 表示真实浓度更大，pTox 更小，是低毒方向的上界。

所以 `>` 和 `<` 不能一概而论，必须结合目标变换方向解释。

### 5.2 为什么不是直接用普通 MAE

普通 MAE 假设标签是精确点值。

但 censored 标签不是点值，而是一个区间或半区间：

- 真实值只知道大于某个值。
- 或只知道小于某个值。

因此，用 midpoint 或当作精确值会引入误差。

### 5.3 Tobit / interval-censored likelihood 是什么

它们是专门处理“真实值只知道在某个范围内”的统计方法。

简单理解：

- 普通回归：预测值应该接近一个点。
- censored 回归：预测值只要落在合理区间或满足方向约束，就不应被过度惩罚。

但这些方法会改变训练目标，解释成本较高。因此当前建议先做分布分析和误差诊断，再决定是否升级。

## 6. 性能上限与预测区间分两步做

这个思路合理。

### Step A：探索性能上限

目标：知道当前框架在不改变科学解释边界的情况下能到什么程度。

建议：

- 固定当前主线结构。
- 使用 5-seed 或更多 seed ensemble。
- 可测试轻量模型平均、checkpoint averaging、SWA、不同 early-stopping seed。
- 不新增 DANN/MMD/CORAL。

输出：

- 最佳 ensemble MAE/RMSE/R2。
- 单模型均值与标准差。
- AD 分层后的性能上限。

### Step B：构建预测标准差和预测区间

目标：从“点预测模型”变成“可用于风险判断的可靠性模型”。

建议：

- 从 5-seed ensemble 计算预测均值和标准差。
- 构建经验 prediction interval。
- 用 conformal calibration 校准覆盖率。
- 对高不确定性、阈值附近样本标注。

输出：

- prediction_mean
- prediction_sd
- lower/upper interval
- uncertainty tier
- near-threshold flag
- AD label

## 7. 是否现在先跑一个新消融实验

不建议立刻跑新的训练消融。

原因：

1. 现有 v1.2.1-v1.2.13 已经形成了足够的策略消融链：
   - source Tanimoto
   - effect-level weighting
   - soil-only 对照
   - authority CE-bin
   - ordinal
   - censored loss
   - proxy source rule
   - proxy alpha
   - seed stability
   - validation policy
2. 当前主要混乱不是缺少一个消融，而是缺少主线整理和 final evaluation。
3. 如果现在再跑一个新消融，很可能增加一个新文件夹，但不能解决论文主线问题。

更建议先做一个“回顾性消融整合表”，即用已有结果整理策略贡献，而不是重新训练。

## 8. 推荐后续实验方案

### v1.2.14：CST-AD 应用域方法构建，不改训练模型

目标：

- 基于当前 v1.2.12 anchor ensemble 构建模型专属 AD 方法。

输入：

- v1.2.12 5-seed prediction rows。
- AD audit rows。
- taxonomy/context columns。
- ensemble prediction variance。

输出：

- `cst_ad_prediction_rows.csv`
- `cst_ad_stratified_metrics.csv`
- `cst_ad_failure_cases.csv`
- `cst_ad_summary.md`

核心分析：

- chemical AD tier。
- species AD tier。
- task AD tier。
- uncertainty tier。
- combined CST-AD label。

判定规则：

- 如果 AD-A/AD-B 显著优于 AD-C/AD-D，说明该 AD 方法有解释价值。
- 如果 AD label 能识别高误差样本，则可作为论文方法亮点。

### v1.2.15：censored 样本分布与误差诊断，不改训练模型

目标：

- 弄清 censored 数据到底在哪些任务、介质、毒性区间、物种和删失方向上集中。

输出：

- censored direction summary。
- endpoint-family summary。
- medium/unit summary。
- toxicity-bin summary。
- high-error censored cases。

判定规则：

- 如果 censored 集中在特定方向或 endpoint，再设计方向性 loss。
- 如果 censored 分布广泛且噪声大，继续保留当前小权重 censored hinge loss。

### v1.2.16：repeated outer split final confirmation

目标：

- 验证当前主线不是依赖单一固定 test。

设计：

- 不新增模型机制。
- 重复 3-5 个 chemical holdout / species-aware outer splits。
- 每个 outer split 用当前主线训练。
- 报告均值、标准差、最差 split、AD 分层。

判定规则：

- 如果各 split 的 MAE/R2 稳定，则当前主线可作为论文主结果。
- 如果波动大，则论文重点转向“AD 方法识别可靠适用范围”，而不是宣称整体泛化强。

### v1.2.17：性能上限探索

目标：

- 在已锁定评价体系下探索上限。

候选：

- 5-seed vs 10-seed ensemble。
- checkpoint averaging。
- SWA。
- ensemble distillation。

不建议此阶段加入：

- DANN
- MMD
- CORAL
- 大 proxy 网格

## 9. 当前最建议执行的下一步

建议下一步不是训练消融，而是执行：

`v1.2.14 CST-AD 应用域方法构建`

理由：

- 不会改变训练主线。
- 不需要先大规模远端训练。
- 能解释你最关心的模型适用范围。
- 能为论文形成区别于普通 QSAR 性能叙事的方法亮点。
- 能为后续 repeated outer split 和风险评估提供分层框架。
