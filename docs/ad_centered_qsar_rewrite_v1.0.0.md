# v1.0.0 训练优化重构：AD 扩展 ECOTOX-QSAR 与土壤小样本迁移

## 核心定位

本项目的主线不是追求极端 scaffold 外推，而是基于 ECOTOX 全量数据库构建覆盖多物种、多介质、多化合物类别的 QSAR 模型，并以 applicability domain, AD 为主评价框架。外推能力是亮点，但主结论应限制在 AD 内预测、外部验证和机制解释可以支撑的范围内。

土壤小样本任务的创新点是：利用水相生态毒性大样本预训练形成更宽的化合物-物种-暴露条件表征，再通过土壤少量样本适配，放大传统单一类别 QSAR 的应用域，同时保持土壤目标尺度本身的物理含义。

## 目标尺度原则

- 水相 `ptox_mol_l` 作为水相生态毒性主目标。
- 土壤环境浓度可以建模，默认保留 `neg_log10_mg_kg` 作为土壤主目标。
- 不把 `mol/L` 和 `mol/kg` 直接混为同一物理目标；通过 `target_name`、`medium_domain`、adapter 和 target standardization 区分。
- 如后续有可靠分子量字段，可派生 `soil_mol_kg` 或 `soil_pTox_mol_kg`，但必须显式记录换算和缺失原因。

## v1.0.0 代码重构点

- 训练目标默认使用 `per_task_target` 标准化，并在预测表中同时输出原始尺度和标准化尺度。
- 多任务 loss 从直接加和改为按任务权重归一化平均，避免任务数和样本数差异导致训练尺度失控。
- 默认优化器改为 AdamW，加入 weight decay、梯度裁剪、可选 cosine scheduler。
- 类别变量加入 `<missing>`、`<unknown>`、`<rare>`，区分训练低频类别和测试未见类别。
- 预测输出保留样本级元数据，便于与 AD 报告、化合物、物种、介质和 target scale 合并审计。
- Williams AD 标准化和 PCA 仅在 train 上 fit，再 transform 全部样本，避免测试集信息泄漏。
- 实验输出目录改为 `版本号_中文一句话实验名/deep/ablation/split_name`。

## 配套文档

- 终点和任务权重分配：`docs/endpoint_task_allocation_v1.0.0.md`
- 重构后 smoke 与初始对照：`docs/v1.0.0_optimization_initial_comparison.md`
- 同训练预算复跑对比：`docs/v1.0.0_same_budget_comparison.md`
- 文献依据与表述建议：`docs/literature_support_ad_qsar_v1.0.0.md`

## 后续实验命名

版本号使用 `vn.n.n` 格式，例如：

- `v1.0.0_训练优化重构_水相pTox随机划分对照`
- `v1.0.0_训练优化重构_水相预训练土壤mgkg小样本迁移`
- `v1.0.0_训练优化重构_AD内预测性能主分析`

命名要求：

- 中文一句话讲清楚实验目的。
- split 名保留在下级目录，避免实验名过长。
- 同一科学问题只递增版本号，不重复造相近英文缩写。

## 优化前后对比任务

smoke 跑通后，至少复跑下面两组历史任务做直接对照：

- `AquaticPtox_B_random_8_2`
- `M_qc_aquatic_to_soil_ptox_adapt_C_f20`

对比表至少保留：

- 训练配置版本和中文实验名。
- `metrics.csv` 的 test/finetune 指标。
- `history.csv` 的 loss、学习率、最大梯度范数。
- `predictions.csv` 的样本级误差分布。
- AD 合并后的 AD 内/AD 外分层表现。
