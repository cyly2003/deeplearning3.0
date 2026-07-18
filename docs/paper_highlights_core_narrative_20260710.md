# 论文 Highlights 与核心叙事优化稿

日期：2026-07-10

## 1. 推荐核心定位

本研究不应写成“单纯构建了一个更高精度的 QSAR 模型”，而应定位为：

> 面向 ECOTOX 异质生态毒性记录，构建并验证一个应用域约束的 context-aware multi-task transfer QSAR 框架，用于在土壤毒性样本稀缺、物种/终点高度异质和实验噪声较高的条件下进行多物种 pTox 预测，并定量揭示同分布插值与化学结构族外推之间的性能落差。

这个定位比“解决生态系统互作导致的预测困难”更稳妥。当前模型确实使用了物种分类学、life stage、暴露时长、效应水平、终点和介质/目标上下文，但并未直接建模真实生态互作网络、种间关系或群落动力学，因此“生态互作规律”应降调为“生态毒理上下文信息”或“species-/endpoint-aware context”。

## 2. 可用于投稿的 Highlights

### 中文版

1. 基于 ECOTOX 构建了去金属/无机物的水相-土壤 pTox 建模数据集，并保留了可审计的数据筛选与单位换算流程。
2. 提出 context-aware 多任务迁移学习 QSAR 框架，将水相毒性知识迁移到土壤多物种毒性预测。
3. 在随机 8:2 与随机 5-fold 验证中，5-seed 集成模型获得稳定的土壤 pTox 插值性能。
4. 消融结果显示，实验上下文与物种/life-stage 信息是模型性能提升的主要来源。
5. 通过 scaffold/similarity-cluster 压力测试和 CST-AD，明确区分可靠插值预测与结构族外推风险。

### 英文版

1. A metal- and inorganic-excluded ECOTOX aquatic-soil pTox dataset was curated with auditable preprocessing.
2. A context-aware multi-task transfer QSAR framework was developed for sparse soil ecotoxicity prediction.
3. Five-seed ensembles achieved stable interpolation performance under random 80:20 and five-fold validation.
4. Ablation analysis identified exposure context and species/life-stage information as dominant contributors.
5. Scaffold-cluster validation and CST-AD separated reliable interpolation from chemical-family extrapolation risk.

## 3. 主线结果应如何写

### 推荐强主张

- 模型在去金属/无机物后的随机插值边界下表现稳定：random 8:2 5-seed ensemble 为 R2=0.7895、RMSE=0.8865、MAE=0.6243；random 5-fold 5-seed ensemble 为 R2=0.7920、RMSE=0.8899、MAE=0.6200。
- 消融实验支持“生态毒理上下文”比单纯分子描述符更关键：去除 context 后 MAE 增加最大，去除 species/lifestage 后次之，toxicity binning 贡献较小但稳定。
- 分子信号没有被 species/context embedding 完全遮蔽。RDKit full、PaDEL prior-clustered 和 graph-only 旁支共同说明，模型性能不是单一分子量或单一描述符耦合造成的。
- CST-AD 可作为应用域透明报告工具，说明随机主线主要是高覆盖插值预测，同时标记少量中等覆盖样本。

### 需要降调的表述

- “土壤物种-化合物外推 QSAR”应改为“土壤多物种 pTox 预测，并通过结构族划分评估外推压力”。
- “辅助环境修复决策”应改为“辅助污染物生态风险筛查、优先级排序和补充实验设计”。
- “生态位信息隐性嵌入”可改为“物种分类学、life stage、暴露时长和效应水平等生态毒理上下文被纳入模型表征”。
- “解决生态系统互作导致的预测困难”应改为“提供一种处理多物种、多终点、跨介质毒性数据异质性的建模范式”。

### 不建议写成结论

- 不建议声称模型已经可靠解决新化合物结构族外推。当前 scaffold/similarity-cluster 结果显示外推明显更难：holdout R2=0.1810、MAE=1.0664；5-fold R2=0.3664、MAE=1.1178。
- 不建议将 CST-AD 写成强错误分类器。当前随机主线下 AD-A 覆盖约 96%，高错误识别 AUROC 约 0.50，更适合定位为覆盖性/可靠性分层工具。
- 不建议把传统 ML baseline 与迁移模型做无条件强弱比较。传统 ML baseline 多为固定物种-终点内 row-random 插值，验证边界不同。

## 4. 论文核心叙事

### 第一层：问题提出

环境 QSAR 面临的数据条件比药物毒性 benchmark 更复杂：ECOTOX 记录覆盖大量化合物、物种、介质、暴露时间和毒性终点，但土壤样本相对稀缺，且记录噪声、单位不一致、删失值和终点异质性明显。传统单任务或化学描述符主导的 QSAR 难以同时利用水相大样本、土壤小样本、物种差异和实验上下文。

### 第二层：方法创新

本研究将水相 pTox 作为源域、土壤 pTox 作为目标域，构建 aquatic pretrain + soil finetune 的二阶段迁移学习流程。模型同时融合 Morgan fingerprint、RDKit 描述符、物种分类学/life-stage embedding、暴露时长、效应水平、介质/目标上下文和多任务 task head，并引入 Tanimoto-to-finetune source weighting、authority-based toxicity bin auxiliary task 与 censored hinge loss。

### 第三层：验证逻辑

评价不只报告一个总体性能，而是分为三层：

1. 随机 8:2 与随机 5-fold：作为当前论文主性能边界，回答模型在同分布插值和高覆盖场景下能否稳定预测。
2. scaffold/similarity-cluster holdout 与 5-fold：作为结构族外推压力测试，回答当测试化合物远离训练化学空间时性能如何下降。
3. CST-AD 与消融：回答模型为什么有效、哪些信息源最重要、哪些预测位于较可靠应用域。

### 第四层：科学意义

本研究的核心贡献不是声称完全替代实验毒理，而是提供一种可审计、可解释、边界明确的生态毒性数据补全和风险筛查工具。它可用于优先识别需要进一步实验验证的化合物-物种-终点组合，辅助土壤生态风险评估中的数据缺口填补，并为后续构建更严格的新化合物外推模型提供基准和失败边界。

## 5. 摘要/引言主句模板

### 中文

ECOTOX 等生态毒性数据库为环境 QSAR 建模提供了大规模数据基础，但其跨物种、跨终点、跨介质和高噪声特征使传统化学描述符 QSAR 难以直接获得可靠泛化。本研究构建了一个 context-aware 多任务迁移学习框架，将水相 pTox 数据中学习到的化合物-物种-效应表征迁移到土壤 pTox 预测，并通过随机验证、结构族外推压力测试、消融实验和 CST-AD 应用域分析系统评估模型性能与适用边界。

### English

Large ecotoxicity resources such as ECOTOX provide an opportunity for data-driven environmental QSAR, but their multi-species, multi-endpoint, cross-medium and noisy nature challenges conventional descriptor-only models. Here, we developed a context-aware multi-task transfer-learning QSAR framework that transfers aquatic pTox knowledge to soil pTox prediction and evaluates its performance boundaries using random validation, scaffold/similarity-cluster stress testing, targeted ablation and CST-AD applicability-domain analysis.

## 6. 文献定位与可引用方向

- ECOTOX 官方说明：ECOTOX 是 EPA 提供的单一化学物质对水生和陆生物种环境毒性数据知识库，可作为本研究数据来源合理性的基础引用。
- QSAR/AD 验证逻辑：应强调 defined endpoint、model validation、applicability domain 与 mechanistic interpretation，避免仅报告随机划分性能。
- 多任务/深度毒性预测：DeepTox、MoleculeNet 和多任务毒性预测文献可用于说明深度学习和多任务表征在毒性预测中的方法背景，但要突出本研究面向生态毒性、物种上下文和土壤迁移，而不是药物/体外毒性 benchmark。
- 跨物种毒性预测：已有工作显示 taxonomy 和 experimental setup 可提升跨 taxa 化学危害预测，这可支撑本研究 species/lifestage/context embedding 的合理性。
- 模型解释：SHAP、PDP、消融和应用域分析应作为互补证据；SHAP/PDP 只能解释模型依赖，不应被写成因果机制证明。

## 7. 推荐论文标题方向

1. Context-aware multi-task transfer QSAR for soil ecotoxicity prediction from ECOTOX
2. Applicability-domain-aware transfer learning for multi-species soil pTox prediction
3. Transferring aquatic ecotoxicity knowledge to sparse soil toxicity prediction with multi-task QSAR
4. Boundary-aware ecological QSAR for multi-species soil toxicity prediction using ECOTOX
