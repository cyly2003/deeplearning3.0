# v1.2.47 Scaffold/Similarity-Family 化合物外推实验协议

日期：2026-07-21

## 1. 科学问题

检验在目标化合物所属结构家族未出现在任何训练阶段时，水相 pTox 预训练是否仍能提高
土壤 `neg_log10_mol_kg` 预测，以及土壤 pTox 中间阶段是否产生额外贡献。

本实验只回答化学结构族外推，不同时要求 reference、物种或任务不重叠。其结果必须与
v1.2.44 随机插值边界和 v1.2.46 reference-group 研究来源边界分开报告。

## 2. 数据母集与目标任务

- 从 v1.2.44 锁定的 M11 路由重新构建，不从 v1.2.46 reference-group split 派生。
- Stage 3 候选母集：15,199 条土壤 `neg_log10_mol_kg` 记录、18 个固定任务。
- Stage 1：水相 pTox；Stage 2：土壤 pTox；Stage 3：土壤 mol/kg。
- 保留 v1.2.44 已有的精确 aggregate/result 来源隔离规则。
- 有效碳骨架无法确定、SMILES 无法解析或没有有机母体的记录从本严格结构外推实验中排除，
  但必须输出逐条排除表、各任务覆盖率和总体保留率；不得静默删除。
- 合法的无环有机物不因 Murcko scaffold 为空而排除，仍通过 canonical parent 和 Morgan
  相似性分组。

## 3. 结构标准化与结构家族定义

先对 Stage 3 目标母集中的唯一结构执行：

1. RDKit 解析 SMILES；
2. 选择最大的含碳母体片段；
3. 在可行时去电荷；
4. 生成非异构 canonical parent SMILES，使盐型和立体异构差异不会形成虚假独立组；
5. 计算非空 Bemis-Murcko scaffold；
6. 计算 Morgan/ECFP 指纹，radius=2、2,048 bits；
7. 使用并查集构建目标域结构连通分量：相同 canonical parent、相同非空 Murcko scaffold，或
   Morgan Tanimoto `>=0.65` 的任一对化合物均连接；所有传递连接合并为同一 component。

Stage 1/2 的唯一结构使用完全相同的标准化与指纹版本，但不需要用 source-only 分子桥接并
扩大 Stage 3 component。目标划分锁定后，源域记录按其与 Stage 3 validation/test 化合物的
canonical、scaffold 和直接 Morgan 相似度关系进行过滤，并通过最终跨阶段硬审计确认隔离。

不能直接复用 v1.2.24 的 Butina centroid 分组。v1.2.47 必须使用相似度图连通分量，或在
Butina 初分组后迭代合并所有仍存在 `Tanimoto>=0.65` 跨组边的组件，直到硬审计通过。

阈值 `0.65` 在查看模型结果前锁定。若该阈值导致巨型连通分量而无法满足样本门槛，本实验
应停止并报告不可行，不得依据测试性能临时调整阈值。任何 `0.70/0.80` 阈值只能作为以后
独立登记的敏感性分析。

CAS 和 DTXSID 仅用于追溯审计，不作为主分组键；reference、物种和任务允许跨 split，
因为本版本只隔离化学结构族。

## 4. Stage 3 划分

- 目标比例：train/validation/test=`64/16/20`。
- 一个结构 component 只能位于一侧。
- 在固定 base seed `20260721` 下生成至少 4,096 个 prediction-blind 候选划分。
- 候选评分仅使用行数比例、每任务行数、每任务 component 数和任务内目标值五分位平衡；
  禁止读取任何模型预测或模型指标。
- 若 4,096 个原始哈希候选均未满足硬门槛，则仅对排序最前的 64 个候选执行确定性的
  structure-component 整组约束修复：每次只移动一个完整 component，并仅依据任务支持
  缺口、全局比例、身份隔离和上述 prediction-blind 平衡分数接受改进；不得降低门槛、
  拆分 component 或读取模型结果。修复路径和全部移动必须写入锁定审计。
- 选定后锁定 assignment hash、评价身份 hash、结构 component hash 和完整审计文件。

数据可行性预审在读取任何模型结果之前发现，结构可解析后 `ECx_Population` 仅 85 行、
`ICx_Growth` 仅 51 行，无法同时满足 train 100、test 30 和 validation 非空。因此使用
以下按结构可解析总支持量预注册的两级门槛，而不是删除任务或事后调整：

- 保留全部 18 个任务；不得通过训练时动态阈值删除任务；
- 对结构可解析总量 `>=131` 的标准支持任务：train `>=100` 行且 `>=5` 个 component，
  validation `>=1` 行且 `>=1` 个 component，test `>=30` 行且 `>=5` 个 component；
- 对结构可解析总量 `<131` 的预注册低支持任务：train `>=20` 行、validation `>=1`
  行、test `>=10` 行，三侧各 `>=1` 个 component；当前仅
  `ECx_Population` 和 `ICx_Growth` 属于该层级；
- 优先选择每任务 validation `>=20` 行且 `>=3` 个 component 的候选；未达到时标记
  task-level validation 支持不足，但不因这一优先条件改变已锁定硬门槛；
- train、validation、test 之间 canonical parent、Murcko scaffold、structure component、
  aggregate、result 和 test 标识交集均为 0；
- 三侧任意跨侧最大 Morgan Tanimoto 必须 `<0.65`。
- 全局行数比例应位于 train `60–68%`、validation `12–20%`、test `17–23%`；若最大
  不可拆 component 导致越界，则构建失败并人工审计，不得按性能重新挑 seed。

## 5. 源域结构泄漏隔离

这是相对旧 v1.2.24 必须新增的关键规则。

- Stage 3 validation/test 所属结构 component 中的全部 Stage 1 水相和 Stage 2 土壤 pTox
  记录必须从源域训练中排除。
- Stage 1/2 缺少有效结构的记录从本严格实验中排除并单独报告。
- Stage 1/2 与 Stage 3 validation/test 之间的 canonical parent、Murcko scaffold 和
  structure component 交集必须为 0，最大 Morgan Tanimoto 必须 `<0.65`。
- 不复用 v1.2.44、v1.2.46 的 checkpoint。任何依赖源记录身份的 source-weight cache
  必须重新生成并将过滤后身份 hash 写入 cache key。
- reference 可以跨侧；不能把 v1.2.47 描述为同时实现研究来源外推。

## 6. 正式实验矩阵

| Cell | Stage 1 水相 pTox | Stage 2 土壤 pTox | Stage 3 土壤 mol/kg | 科学问题 |
|---|---|---|---|---|
| SF-M00 | 无 | 无 | 从头全参数训练 | 严格结构外推基线 |
| SF-M10 | 有，过滤 held-out 结构族 | 无 | 全参数微调 | 水相预训练是否支持新结构族 |
| SF-M11U | 有，过滤 held-out 结构族 | 有，过滤 held-out 结构族 | 全参数微调 | 土壤 pTox 是否有额外结构外推价值 |

- 正式种子：`42/2042/3407/8417`，共 12 个训练单元。
- 复用 v1.2.40/v1.2.46 的模型架构与训练超参数；`medium_adapters=false`。
- Stage 3 必须为 `freeze=none` 的全参数微调，不能冻结 trunk。
- Stage 3 early stopping 只监控显式 validation：
  `finetune_mgkg_monitor_split=valid`、`finetune_mgkg_validation_fraction=0`。
- 外层 test 为 report-only，不参与 early stopping、模型选择、split 选择或运行中监控。

本轮只做一个预注册 64/16/20 holdout，不同时扩展 5-fold 或多个相似度阈值。

## 7. Smoke 到正式运行门禁

1. 远端 `py_compile`、`bash -n` 和定向 pytest 全部通过；
2. 构建一次 split 并锁定 contract/hash，随后正式训练必须 `REBUILD_SPLITS=0`；
3. seed 42 的 SF-M00、SF-M10、SF-M11U 各运行 1 epoch smoke；
4. validator 核对结构隔离、有效配置、显式 validation、完整 prediction parts 和无外部
   checkpoint；
5. 3/3 smoke 通过后，以 `PARALLEL_JOBS=2` 启动 12 个正式单元；
6. 只有 12 个 manifest、predictions 和 validator 均通过并输出 `[matrix_complete]` 后，
   才能读取外层 test 并汇总。

## 8. 统计与报告

主要评价尺度：`neg_log10_mol_kg`；同时报告共同 mg/kg 尺度，但不得混写。

主要输出：

- 四种子单模型 mean+SD；
- 四种子逐行 prediction ensemble 的 R2、RMSE、MAE；
- within-task centered R2；
- 18 个任务的 n、component 数、R2、RMSE、MAE；任务级 R2 仅在 test `n>=30` 且
  `component>=5` 时标为 supported，其他任务只解释 MAE/RMSE 与支持度；
- SF-M10−SF-M00 和 SF-M11U−SF-M10 的逐行配对差异；
- 以 held-out structure component 为重采样单位的 20,000 次 paired cluster bootstrap；
- SF-M11U 与 SF-M10 使用 `±0.01` MAE 作为实际等效界限。

跨边界比较时并列展示 v1.2.44 random、v1.2.46 reference-group 和 v1.2.47
scaffold/similarity-family，但由于测试身份不同，不把三者差值解释为配对因果效应。

## 9. 预注册判断

- SF-M10 显著优于 SF-M00：水相预训练提供可迁移到新结构族的初始化；
- SF-M10 与 SF-M00 接近：现有迁移收益主要限于已知或相近化学空间；
- SF-M11U 与 SF-M10 实际等效：正式主模型继续采用更简洁的 M10；
- SF-M11U 稳定优于 SF-M10：土壤 pTox 只在结构外推边界下表现出有限补充价值，但不能
  反向改写 v1.2.44/v1.2.46 已经得到的边界内结论；
- 三者性能均明显下降：将其解释为化学结构家族外推困难，而非训练失败。

## 10. 与旧实验的区别

v1.2.24 证明旧 no-metal soil-pTox 分支在 scaffold/cluster 压力测试下性能明显下降，
但其 runner 会将全部水相记录放入预训练，缺少当前严格的源域 held-out 结构族隔离、显式
Stage 3 validation、fail-closed validator 和当前 mol/kg 18任务边界。因此 v1.2.47 只复用
其结构标准化思想，不复用其正式 split 或训练结果作为当前矩阵对照。

对旧 holdout 的结构复核显示，土壤 finetune 与 test 的 canonical/scaffold 隔离及
`max Tanimoto<0.65` 成立；但 302 个土壤 test 化合物中，248 个与水相 Stage 1 存在
canonical parent 重叠，264 个对水相 Stage 1 的最大 Tanimoto `>=0.65`，其中 249 个
`>=0.80`。因此旧实验只能描述为“相对土壤 finetune 的结构外推压力测试”，不能声称
held-out 化合物及其近邻在完整迁移链中从未出现。
