# v1.0.0 文献依据：AD 主线、scaffold 压力测试与小样本迁移

## 关键依据

| 来源 | URL | 对本项目的支撑 |
|---|---|---|
| OECD. Guidance Document on the Validation of (Q)SAR Models. | https://one.oecd.org/document/env/jm/mono%282007%292/en/pdf | OECD 五原则要求明确 endpoint、算法、AD、拟合/稳健性/预测性和机制解释，支持把 AD 作为 QSAR 主评价框架。 |
| Sahigara et al. 2012. Comparison of Different Approaches to Define the Applicability Domain of QSAR Models. | https://www.mdpi.com/1420-3049/17/5/4791 | AD 可由 descriptor space、similarity、distance、leverage、density 等多种方法定义，AD 内预测更适合作为可靠结论。 |
| Ruusmann et al. 2015. QSAR DataBank repository. | https://link.springer.com/article/10.1186/s13321-015-0082-6 | 支持保存模型元数据、端点、描述符、算法、验证流程、AD 和解释信息，作为可追溯 QSAR 工作流。 |
| Wu et al. 2018. MoleculeNet: a benchmark for molecular machine learning. | https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a | MoleculeNet 使用多种 split；scaffold split 可作为结构外推压力测试，但不应机械替代所有毒性任务的主评价。 |
| Wallach and Heifets 2018. Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization. | https://arxiv.org/abs/1706.06619 | 结构冗余会抬高随机划分性能，必须报告相似性、AD 或冗余风险，而不是只看单一 split 分数。 |
| Guo et al. 2024. Scaffold Splits Overestimate Virtual Screening Performance. | https://arxiv.org/abs/2406.00873 | scaffold split 本身也可能高估外推，适合作为敏感性分析。该来源为预印本，证据权重低于 OECD 和同行评议文献。 |
| US EPA ECOSAR official page. | https://www.epa.gov/tsca-screening-tools/ecological-structure-activity-relationships-ecosar-predictive-model | 生态毒性 QSAR/SAR 可用于筛查和类比推断，但需要专业判断和适用性判断，支持 AD 外仅作为风险提示。 |
| Kirschbaum and Bande 2024. Transfer Learning for Molecular Property Predictions from Small Data Sets. | https://arxiv.org/abs/2404.13393 | 小样本分子性质建模可以使用迁移学习，但需检查源任务与目标任务相似性，并不能替代 AD 与外部验证。 |

## 推荐论文表述

中文：

> 本研究以 applicability domain, AD 为 QSAR 模型可靠性评价主框架。模型主结论限定于 AD 内样本的预测性能、稳健性和机制一致性；AD 外预测作为外推风险提示，并结合不确定度和相似性信息解释。

英文：

> Following OECD QSAR validation principles, the reconstructed workflow emphasizes defined endpoint, transparent algorithm, applicability domain, predictive performance, and mechanistic interpretation. Therefore, the primary conclusions are restricted to predictions within the model AD, while out-of-domain predictions are reported as extrapolation warnings with uncertainty and similarity evidence.

关于 scaffold split：

> Scaffold split is treated as a structural extrapolation stress test rather than the primary decision criterion. Its role is to reveal performance decay on new chemical frameworks, not to replace AD-based reliability assessment, external validation, or mechanistic interpretation.

## 对实验设计的直接约束

- 主表必须输出 AD 内/AD 外分层指标。
- scaffold 类验证只能作为补充压力测试。
- 水相预训练到土壤小样本迁移应报告源域-目标域差异、target scale、介质 adapter 和 AD 覆盖。
- AD 外预测需要保留不确定度/相似性提示，不作为定量生态毒性结论。
