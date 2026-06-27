# 最终主线模型材料与方法草稿

更新时间：2026-06-27

本文档面向硕博论文“材料与方法”章节撰写，描述当前主线最优模型：`v1.2.12` aquatic-to-soil pTox transfer，固定 chemical-holdout `f100`，5-seed ensemble。文中所有参数优先依据当前仓库代码、配置文件、远端 run `manifest.json` 和已生成 summary 表；对尚未完成的后续实验单独标注，不凭记忆补写。

## 1. 研究对象与建模目标

本研究以 ECOTOX 来源的化合物-物种-毒性终点聚合记录为建模对象，目标是在可比的 pTox 尺度上预测不同化合物对不同生物分类单元和毒性终点的毒性响应。当前主线模型聚焦于水相到土壤相的迁移学习：首先利用大规模水相 pTox 数据学习化合物结构、物种分类学和实验上下文对毒性的共同表征，然后使用土壤 pTox 数据进行二阶段微调，并在独立土壤 chemical-holdout test 上评估外推能力。

当前主线使用的数据表为 `aggregated_task_records_aquatic_soil_ptox_qc`，来自派生数据库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。该表同时包含水相与土壤 pTox 聚合记录，关键字段包括 `aggregate_id`、`cas_number`、`dtxsid`、`smiles`、`species_number`、`latin_name`、`task_head`、`task_family`、`effect_family`、`effect_level_x`、`target_name`、`target_basis`、`medium_domain`、分类学字段、暴露时长字段以及聚合后的目标值统计量。当前本地数据库没有 `split_assignments` 表，训练时的 split assignments 与完整模型 artifacts 以远端 `/home/easyai/DL1/ecotox_qsar_transfer` 为准。

主线训练实际包含 35 个任务头，包括 ECx、LOEC、NOEC 以及少量 ICx/LDx 任务。论文主结果和当前最终指标汇总采用 ECx/LOEC/NOEC 30-task focus，ICx/LDx 保留在补充表中。这一处理避免训练中可利用的辅助任务与论文主线评价范围混淆。

## 2. 数据预处理与质量控制

原始毒性记录经过单位标准化、介质域审计、目标尺度统一和条件聚合后进入建模表。当前主线使用 pTox 目标，目标值字段为 `target_value_median`，并在模型训练中按 `per_task_target` 进行目标标准化。该标准化方式按任务头与目标尺度组合拟合均值和标准差，使不同终点及目标尺度在训练损失中具有更可比的数值范围。远端主线 run manifest 显示，`target_standardization.mode = per_task_target`，拟合分割部分为 `train` 与 `finetune`。

聚合记录保留了多种数据质量相关字段，包括 `value_quality`、`cross_reference_conflict_flag`、`cross_reference_range`、`target_value_count`、`target_value_std`、`source_reference_count` 等。当前主线未在训练阶段进一步手动删除这些记录，而是使用已构建的 QC pTox 表作为输入。对删失数据，模型启用了 censored hinge loss 的辅助约束，但主预测指标仍基于可评估的数值型 `y_true` 与 `y_pred`。

训练前对数值特征进行了标准化与异常 z-score 裁剪。配置文件中 `training.feature_zscore_correction.enabled = true`，阈值为 6.0；远端 `preprocessing.json` 记录了每个数值特征的均值、标准差、裁剪上下界和被裁剪样本比例。该处理只改变特征输入尺度，不改变原始数据库记录。

依据文件：

- `configs/experiment.remote.easyai.yaml`
- `qsar_tl/training/deep_experiment.py`
- 远端主线 run 的 `manifest.json` 与 `preprocessing.json`

## 3. 数据切分策略

### 3.1 固定 chemical-holdout 主线 split

当前主线 split 名称为 `M_v2_aquatic_to_soil_ptox_adapt_C_f100`。该 split 是 aquatic-to-soil adaptation split：水相 pTox 样本作为预训练 `train`，土壤 pTox 中 chemical-holdout 训练部分作为 `finetune`，土壤 pTox chemical-holdout 测试部分作为 `test`。

主线 run manifest 显示，模型训练阶段实际使用：

- 预训练 train 样本：250,694 行；
- 土壤 finetune 样本：13,526 行；
- finetune 内部训练：10,823 行；
- finetune 内部验证：2,703 行；
- 固定 test focus 评价行：2,594 行 ECx/LOEC/NOEC soil test prediction rows。

这里需要区分“数据库 split assignments 原始行数”和“模型实际训练行数”。模型训练会经过任务头过滤、可训练目标过滤、内部 finetune-validation 划分等步骤，因此材料与方法中应优先报告 run manifest 中记录的训练计数；数据库当前状态只作为辅助核对。

### 3.2 chemical-holdout 的切分标准

仓库中的 A-F split 生成逻辑显示，chemical-holdout 并不是按 Tanimoto 阈值切分，而是按 `cas_number` 分组切分。对 C 类 split，先以 `cas_number` 构建化合物组，再将化合物组按随机种子分配到 train/test，比例为 8:2。所有属于同一 CAS 的记录被分配到同一 split part，因此 train 与 test 不应共享同一 CAS 化合物。

对应代码逻辑：

- `qsar_tl/evaluation/splits.py`
  - `C_chemical_holdout_8_2` 使用 `assign_group_named_parts(..., ("cas_number",), fractions={"train": 0.8, "test": 0.2})`；
  - `D_chemical_group_5fold` 使用 `cas_number` 分组 5-fold；
  - `F_chemical_adapt_7_2_1` 使用 `cas_number` 分组 7:2:1。

因此，chemical-holdout 的切分标准是“同一 CAS 不跨训练与测试”，不是“最大 Tanimoto 小于某个阈值”。Tanimoto 相似度用于训练加权和应用域分析，而不是用于主线 split 的样本划分。

### 3.3 Tanimoto 相似度阈值与应用域

应用域审计中使用 Morgan fingerprint 计算 test 化合物到训练化合物的最大 Tanimoto 相似度。`scripts/audit_prediction_application_domain.py` 的默认阈值为 0.5；当 test 化合物对训练化合物的最大 Tanimoto 相似度不低于 0.5，或 Williams leverage 判定在域内时，化合物维度可被视为 in-domain。该阈值服务于应用域分层，不参与 chemical-holdout 划分。

物种应用域使用分类学层级前缀匹配。默认 taxon columns 为 `kingdom, phylum, class_name, tax_order, family`，最大分类学相似度等于 test 物种与训练集中可匹配的最深层级数除以总层级数。默认 taxon similarity 阈值为 0.8，即至少达到较深的分类层级相似时才被视作 species/taxon in-domain。

## 4. 特征工程

### 4.1 化合物结构特征

化合物结构由 SMILES 编码。当前配置使用 `rdkit_descriptor_morgan`，并优先读取预先构建的 RDKit 分子特征缓存 `outputs/features/molecular_features_rdkit_morgan512.jsonl`。主线模型输入包括：

- 512-bit Morgan fingerprint；
- RDKit 分子描述符：`MolWt`、`TPSA`、`MolLogP`、`HeavyAtomCount`、`NumHAcceptors`、`NumHDonors`、`RingCount`、`RotatableBonds`。

Morgan fingerprint 半径为 2，位数为 512。指纹用于模型输入，也用于 source weighting 与 AD Tanimoto 计算。

### 4.2 效应水平与暴露时长特征

数值上下文特征包括效应水平与暴露时长两部分。主线 `preprocessing.json` 显示，模型实际数值特征共 22 个，其中 8 个为 RDKit 分子描述符，14 个为上下文数值特征：

- 效应水平：`effect_level_x`、`effect_level_x_fraction`、`effect_level_x_log1p`、`effect_level_x_present`；
- 暴露时长：`duration_bin_h`、`duration_log1p_h`、`duration_sqrt_h`、`duration_inv_log1p_h`；
- 暴露时长 RBF：`duration_rbf_24h`、`duration_rbf_48h`、`duration_rbf_96h`、`duration_rbf_168h`、`duration_rbf_336h`、`duration_rbf_720h`。

网络中还包含一个 effect-level encoder。该模块从数值特征中选择 `effect_level_x` 相关列，经过小型 MLP 后以残差形式加到共享表征上，从而使 ECx 中的 x 水平、NOEC/LOEC 效应类型差异等信息参与预测。

### 4.3 物种分类学与上下文 embedding

当前模型不是把整条 taxonomy path 合成为一个 embedding，而是将每个分类学层级和上下文字段分别建立 vocabulary 与 embedding，然后将各字段 embedding 拼接进入共享主干网络。远端 `preprocessing.json` 和 `manifest.json` 显示 categorical columns 包括：

- 物种与分类学：`latin_name`、`kingdom`、`phylum`、`class_name`、`tax_order`、`family`、`genus`、`species`；
- 物种粗分组：`taxon_group_l1`、`taxon_group_l2`、`taxon_group_l3`；
- 实验与介质上下文：`primary_medium`、`habitat_labels`、`organism_habitat`、`media_type`、`organism_lifestage`、`target_basis`、`effect_family`。

默认 embedding 维度由 `default_embedding_dim(cardinality) = min(32, max(2, int(sqrt(cardinality)) + 1))` 决定。低频类别按 `categorical_min_count = 2` 归入 `<rare>`，缺失和未知类别保留专门 token。

## 5. 模型架构

当前模型为多任务深度神经网络 `EcotoxMultiTaskNetwork`，核心结构如下：

1. 输入层接收三类输入：标准化数值特征、Morgan fingerprint、各 categorical 字段的 embedding id。
2. 将数值特征与 fingerprint 拼接为 molecular input，再与 taxonomy/context embedding 拼接形成 trunk input。
3. 共享 MLP trunk 对拼接后的全特征进行表征学习。主线训练脚本覆盖使用 `--dropout 0.10`，配置基础 hidden dim 为 256；训练代码将其解析为两层 hidden dims：`(256, 128)`。
4. 分子残差分支 `molecular_residual` 将 molecular input 线性投影到 trunk 输出维度后与共享表征相加，以保留直接的化合物结构信号。
5. effect-level encoder 将效应水平相关数值特征编码后以残差形式加入共享表征。
6. medium adapter 根据 adapter id 对不同介质/目标域进行残差校正。主线 manifest 中 adapter map 包括 `aquatic|aquatic_pTox_mol_L` 与 `soil|aquatic_pTox_mol_L`。
7. 每个 task head 使用独立线性层输出一个标量预测值。主线训练实际包含 35 个 task head。
8. 当启用 toxicity binning 时，模型额外包含 toxicity bin classifier，用于辅助分类损失。

该架构的核心思想是共享化合物-物种-上下文表征，同时保留任务头差异和介质适配能力；相比单任务模型，它可以利用跨终点、跨物种和跨介质的数据相关性。

## 6. 训练策略与损失函数

### 6.1 二阶段迁移学习

主线采用 aquatic pretrain + soil finetune 的二阶段训练：

- 预训练：在水相 pTox train 样本上训练 30 epoch；
- 微调：在土壤 pTox finetune 样本上训练最多 60 epoch；
- finetune 内部 20% 样本作为 `finetune_validation`，用于微调阶段监控与 early stopping；
- finetune 冻结策略为 `none`，即全模型参与微调。

主线训练参数来自 `scripts/run_v1_2_11_f100_seed_stability_remote.sh` 和远端 run manifest：

- batch size：512；
- pretrain learning rate：0.0005；
- finetune learning rate：0.0003082636455810776；
- weight decay：0.000009856751793848817；
- pretrain scheduler：cosine；
- finetune scheduler：reduce_on_plateau；
- early stopping patience：pretrain 15，finetune 10；
- dropout：0.10；
- target standardization：per_task_target；
- device：cuda:0。

5 个 seed 为 `42, 1042, 2042, 3042, 4042`。最终报告中，单模型结果与 5-seed ensemble 结果必须分开呈现。

### 6.2 回归损失与多任务输出

主回归损失为 Huber loss，delta 为 1.0。每个 batch 中按 task head 选择对应输出，只对该样本所属 task head 计算回归误差。模型评估指标包括 R2、RMSE、MAE 和 Huber loss。

由于 pTox 为负对数尺度，MAE 的单位可理解为 log10 尺度误差。例如 MAE 约 1 表示平均误差约为 1 个数量级。该指标适合整体性能比较，但在风险分类场景中仍需后续加入预测区间、阈值风险分类和应用域判断。

### 6.3 source weighting

主线启用 `source_weighting_method = tanimoto_to_finetune`，`alpha = 1.0`。该策略只对 source domain 的水相 train 样本赋权，目标参考集为土壤 finetune 样本。实现逻辑为：

1. 取每个水相 source 样本的 Morgan fingerprint；
2. 计算其到土壤 finetune fingerprint 集合的最大 Tanimoto 相似度；
3. 设原始权重 `w_raw = 1 + alpha * max_tanimoto`；
4. 将权重均值归一化为 1；
5. 裁剪到配置范围内，主线 manifest 中 min/max 分别为 0.25/2.0。

远端主线 manifest 显示，seed3042 run 的 source weighting 已应用于 250,694 个 source samples，target reference samples 为 10,823；相似度范围约为 0.0476 到 1.0，平均约 0.7962。该策略可解释为迁移正则化：与土壤 finetune 化学空间更接近的水相样本在预训练中权重更高。

### 6.4 toxicity bin auxiliary loss

主线启用 authority-based toxicity binning，模式为 `aux_classification`，方案为 `authority_v1`，loss weight 为 0.025。该分支不是替代回归目标，而是在共享表征上增加一个 toxicity bin 分类辅助任务，以帮助模型学习毒性等级边界和高/低毒性区间结构。`authority_v1` 的规则文件为 `configs/toxicity_bins.authority_v1.yaml`，其中水相规则按 mg/L 阈值或 mol/L 经分子量换算到 mg/L 后分箱，土壤规则使用 mg/kg screening bins；该土壤分箱是方法阈值筛查框架，不应表述为通用 GHS 风险分类。

此前实验显示 authority-bin 对迁移路线更像一种迁移正则化，而非对所有土壤-only 模型都普遍提升。因此在论文中应描述为“辅助任务/正则化策略”，不宜写成独立毒性分类模型。

### 6.5 censored loss

主线启用 censored hinge loss，权重为 0.01，margin 为 0.0，处理符号包括 `<`、`<=`、`>`、`>=`。该损失对存在上下界含义的删失记录施加方向性约束。实现中浓度右删失符号 `>`/`>=` 表示真实浓度高于观测界限，因此 pTox 真值低于对应 pTox bound，预测值高于 bound 时被惩罚；浓度左删失 `<`/`<=` 方向相反。远端 manifest 显示，seed3042 run 中 candidate rows 为 56,345，可用 censored rows 为 984；这些样本进入训练损失，但不改变主评估中基于明确数值预测行的指标计算。

当前 censored 数据还没有完成单独分布分析，因此材料与方法中应避免过度解释其风险分类意义，只说明其作为方向性约束辅助训练。

## 7. 5-seed ensemble 与结果汇总

最终主线采用 5-seed ensemble。ensemble 不是重新训练一个新模型，而是对同一 fixed test 样本在 5 个 seed 模型下的预测值进行样本级对齐并求平均。对齐键包括样本 id、aggregate id、split、task、target 与真实值等字段。ensemble 后重新计算 R2、RMSE、MAE 和 Huber loss。

当前最终指标包已整理到：

`outputs/experiments/final_mainline_comparison`

关键结果如下：

- fixed chemical-holdout f100 5-seed ensemble：n=2,594，R2=0.5388，RMSE=1.2368，MAE=0.9277，Huber=0.5563；
- endpoint family：
  - ECx：n=510，R2=0.6295，RMSE=1.0487，MAE=0.7918；
  - LOEC：n=1,027，R2=0.5484，RMSE=1.2721，MAE=0.9591；
  - NOEC：n=1,057，R2=0.4877，RMSE=1.2850，MAE=0.9628。

随机 8:2 与随机 5-fold 当前为 3-seed ensemble 同分布参考，尚待补跑 `3042/4042` 后刷新为 5-seed：

- random 8:2 3-seed：n=3,165，R2=0.7775，RMSE=0.8676，MAE=0.6053；
- random 5-fold 3-seed：n=15,630，R2=0.7833，RMSE=0.8601，MAE=0.6019。

这些随机划分结果用于说明同分布插值上限，不替代 fixed chemical-holdout 主线。

## 8. 后续补充实验接口

为支撑论文完整性，后续需要补齐三类实验：

1. 随机划分 5-seed refresh：在现有 `42/1042/2042` 基础上只补 `3042/4042`，然后用相同 ensemble 汇总脚本刷新随机 8:2 与 random 5-fold。
2. 主线 5-seed 消融：固定 `M_v2_aquatic_to_soil_ptox_adapt_C_f100`，对输入模块与训练策略进行 5-seed 全消融。
3. 单域基线：使用同一深度框架分别训练 aquatic-only 与 soil-only 的 B/C/E split，证明框架在水相与土壤域内均有建模价值，并与迁移线进行横向比较。

上述实验应继续写入 `PROJECT_STATUS.md`、`docs/experiment_registry.csv` 和对应 summary 目录，避免结果只保存在对话记录中。

## 9. 当前不确定与不应过度表述的部分

- 当前方法文档没有把 random split 写成最终外推性能，只作为同分布参考。
- 当前 chemical-holdout 不包含 Tanimoto 阈值切分；若论文中写“相似度阈值”，应放在应用域分析，而不是数据划分。
- learned embedding 可用于补充可视化，但不同 seed 的 embedding 空间不具备天然一一对齐关系，因此正式化合物-物种空间图应优先使用 SMILES/Morgan/Tanimoto 与 taxonomy 层级相似度。
- Huber loss 是训练与评价指标之一，但不能单独解释中等毒性区间的风险误判问题；风险评估需要后续构建预测区间、风险阈值分类和应用域联合判断。
