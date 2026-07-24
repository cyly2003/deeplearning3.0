# v1.2.46 reference-group 边界实验协议

更新时间：2026-07-20（Asia/Shanghai）

## 科学问题

在不要求 CAS、SMILES 或 scaffold 互斥的前提下，检验水相 pTox 预训练的收益能否扩展到
新的研究来源（reference），并检验土壤 pTox 中间阶段在该边界下是否仍与仅水相预训练
实际等效。本实验是对随机 8:2 主结果适用范围的边界验证，不等同于新化学骨架外推。

## 固定分析宇宙

- 数据表：`aggregated_task_records_ptox_soil_mass_molar_qc`；
- 父边界：v1.2.44 严格 M11 路线；
- Stage 3：固定 18 个 soil `neg_log10_mol_kg` 任务，共 15,199 条记录；
- 模型单元：M00、M10、M11U；
- 训练种子：42、2042、3407、8417；
- Stage 3 均为 `freeze=none` 的全参数统一学习率微调，学习率 `5e-4`；
- medium adapter 关闭，不增加新损失、新head或新超参数搜索。

## reference-group 划分

1. 解析每条 Stage 3 聚合记录的 `reference_numbers` JSON 数组。
2. 对多reference记录构建传递闭包连通分量；一个连通分量只能进入一个集合。
3. 目标比例为 train/validation/test = 64%/16%/20%。
4. 在固定候选种子范围内，仅依据逐任务行数、reference component数及任务内目标五分位
   平衡选择拆分，不读取任何模型预测或性能指标。
5. 硬门禁为每个任务 train不少于100条、test不少于30条、validation至少1条；test至少
   包含5个reference component。validation不承诺每个任务均有30条，因此小支持任务只
   用于early stopping，不据此报告稳定的任务级R²。
6. Stage 3 train、validation、test之间的reference、reference component、aggregate_id、
   result_id和test_id均必须互斥。

ICx_Growth存在单一reference约占绝大多数记录的结构，因此同时要求validation和test均
不少于30条在数学上不可行；本协议保留该任务并将validation门槛设为至少1条，而不是
删除任务或放宽test边界。

## 全阶段reference隔离

- Stage 3 validation/test中的所有reference同时从aquatic Stage 1和soil-pTox Stage 2
  训练记录中排除；
- reference缺失的Stage 1/2聚合记录也从本次严格边界训练中排除并计入审计；
- v1.2.44中已执行的aggregate_id/result_id精确来源去重继续保留；
- 不因不同reference共享化合物、物种、终点或相似条件而进一步删除记录。

## 实验矩阵

| 单元 | Stage 1 水相pTox | Stage 2 土壤pTox | Stage 3 soil mol/kg | 目的 |
|---|---|---|---|---|
| RG-M00 | 无 | 无 | 从头全参数训练 | reference边界直接基线 |
| RG-M10 | 有 | 无 | 全参数微调 | 检验水相迁移在新研究来源下是否有效 |
| RG-M11U | 有 | 有 | 全参数微调 | 检验土壤pTox是否产生附加贡献 |

共 `3 × 4 = 12` 个正式训练单元。所有模型共用完全相同的Stage 3身份和拆分hash。

## 执行门禁

1. 本地定向测试和Python语法检查通过；
2. 远端只同步明确allowlist文件，不同步未跟踪文档、Web、附件、features或数据库；
3. 远端完成v1.2.45 reference-cluster bootstrap与数据/配置审计；
4. 远端构建并锁定reference-group拆分及contract hash；
5. seed42、每阶段1 epoch的M00/M10/M11U smoke全部通过manifest和prediction身份验证；
6. smoke输出与formal输出隔离；
7. smoke通过后，formal只读取已锁拆分，不重新选择candidate；
8. 运行中不读取外层test预测进行模型选择。

## 判定

- RG-M10稳定优于RG-M00：支持水相预训练跨研究来源迁移；
- RG-M10未优于RG-M00：将水相迁移结论限定为随机切分插值边界；
- RG-M11U与RG-M10的MAE差异90%置信区间位于±0.01 log单位：采用更简洁的M10主模型；
- reference-group结果只证明新研究来源边界，不证明新结构、新物种或新终点外推。
