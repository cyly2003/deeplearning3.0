# QSAR 远端训练实验计划：模块消融与划分策略评估

更新时间：2026-06-11

## 1. 当前前置状态

SSH 免密已经验证：

- 主地址别名：`qsar-gpu`
- 备用地址别名：`qsar-gpu-backup`
- 主地址和备用地址均可通过 `BatchMode=yes` 免密返回远端主机名。

远端训练环境当前仍需修复：

- `nvidia-smi` 当前退出码为 9，提示无法与 NVIDIA driver 通信。
- `/opt/anaconda3/bin/python` 为 Python 3.12.7，但当前未安装 `torch`。
- 正式 GPU 训练前需要先恢复 NVIDIA driver，并安装项目训练依赖。

建议远端环境准备顺序：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
/opt/anaconda3/bin/python -m pip install -e ".[ml]"
/opt/anaconda3/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

如后续需要 RDKit 描述符，应优先在 conda 环境中通过 conda-forge 安装 RDKit，避免 pip 轮子兼容性问题。

## 2. 科研目标

本轮实验主要回答两个问题：

1. 模型中各输入模块和辅助任务对毒性预测是否有独立贡献。
2. 这些贡献在不同外推难度的数据划分策略下是否稳定。

模型架构以 `docs/model_architecture_agreed.md` 为准。

重点不是只追求随机划分高分，而是判断模型在以下场景中的泛化能力：

- 新化合物外推。
- 新物种外推。
- 新化合物和新物种同时外推。
- 不同环境介质之间迁移。

## 3. 固定实验原则

- 固定随机种子，第一批使用 `42`，关键结论复现实验增加 `2026` 和 `3407`。
- 所有数据删除、单位换算、目标值填补、暴露时长填补和聚合规则沿用当前配置，并保留追溯字段。
- 先使用相同任务头、相同 split、相同训练轮数比较模块贡献，不在消融实验中同时改动超参数。
- 所有结果至少按 `task_head`、`split_type`、`medium` 输出分组指标。
- 随机划分结果只作为上限参考，不作为模型真实外推能力的唯一依据。

## 4. 数据划分策略矩阵

第一版保留当前五类 split：

| split | 科研含义 | 预期难度 | 主要解释 |
| --- | --- | --- | --- |
| `random_split` | 同分布随机抽样 | 低 | 检查模型拟合能力和管线是否正常 |
| `chemical_group_split` | 测试集化合物未在训练集中出现 | 中高 | 评估新污染物/新化合物外推能力 |
| `species_group_split` | 测试集物种未在训练集中出现 | 中高 | 评估跨物种毒性外推能力 |
| `chemical_species_group_split` | 化合物和物种同时外推 | 高 | 最接近真实生态风险预测中的双重外推 |
| `medium_transfer_split` | 按介质进行迁移评估 | 高 | 评估水相、土壤、沉积物等介质间迁移稳定性 |

## 5. 模块消融矩阵

建议先从下表中的核心实验开始，避免一次性运行过多组合。

| run group | 实验名 | 模块设置 | 目的 |
| --- | --- | --- | --- |
| full | `full_multitask` | 分子描述符 + Morgan 指纹 + 物种分类学 + 实验上下文 + 辅助任务 | 完整模型基线 |
| molecule | `desc_only` | 仅 RDKit 描述符 | 判断连续分子描述符贡献 |
| molecule | `morgan_only` | 仅 Morgan 指纹 | 判断结构指纹贡献 |
| context | `molecule_no_species_context` | 分子模块保留，移除物种和实验上下文 | 判断非分子信息总体贡献 |
| context | `molecule_taxonomy_only` | 分子模块 + 物种分类学，移除 life stage、介质、路线、时长等上下文 | 判断分类学层级贡献 |
| context | `molecule_context_no_taxonomy` | 分子模块 + 实验上下文，移除物种分类学 | 判断实验条件和生态上下文贡献 |
| duration | `no_duration_features` | 移除 `duration_h` 和 `log1p_duration_h` | 判断暴露时长是否提供稳定信息 |
| auxiliary | `no_auxiliary_tasks` | 移除 BAF/BCF 和 oral 辅助任务 | 判断辅助学习总体贡献 |
| auxiliary | `no_bioaccum_aux` | 仅移除 BAF/BCF 辅助任务 | 判断生物富集辅助信号贡献 |
| auxiliary | `no_oral_aux` | 仅移除经口辅助任务 | 判断系统毒性辅助信号贡献 |
| multitask | `single_task_heads` | 每个任务头单独训练，不共享主干 | 判断多任务共享是否真正带来迁移收益 |

推荐第一阶段只运行：

- `full_multitask`
- `molecule_no_species_context`
- `molecule_taxonomy_only`
- `no_duration_features`
- `no_auxiliary_tasks`
- `single_task_heads`

如果第一阶段结论清晰，再展开到完整矩阵。

## 6. 分阶段运行计划

### 阶段 A：远端环境和管线冒烟

目的：确认同步、配置、远端 Python、日志和输出拉回可用。

建议命令：

```powershell
pwsh .\scripts\sync_to_server.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -IncludeDerivedDb
pwsh .\scripts\train_remote.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -SmokeTest
pwsh .\scripts\sync_from_server.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径>
```

通过条件：

- 远端能 `validate-config`。
- 训练入口能运行并写日志。
- 本机能拉回 `outputs/logs`。

### 阶段 B：小样本调试矩阵

目的：验证每个消融配置都能正常构建模型、训练和写指标。

建议设置：

- 每个 run 使用小样本限制，例如 2,000 到 10,000 条聚合样本。
- 训练 3 到 5 个 epoch。
- 只跑 `random_split` 和 `chemical_species_group_split`。
- 只用 seed `42`。

判断标准：

- 所有 run 均完成。
- 指标文件结构一致。
- full model 在 random split 上应能明显优于简单均值基线。
- group split 分数下降是正常现象，但不应出现系统性崩溃或 NaN。

### 阶段 C：核心消融实验

目的：估计主要模块贡献。

建议设置：

- 使用第一阶段筛选出的 6 个核心消融组。
- 跑全部 5 种 split。
- seed 先固定为 `42`。
- epoch 使用正式设置，例如 100，加入早停后可改为最大 200。

输出核心表：

- `metrics_by_run_split_task.csv`
- `metrics_by_run_split_medium.csv`
- `ablation_delta_vs_full.csv`
- `generalization_gap_vs_random.csv`

### 阶段 D：关键结论复现实验

目的：评估结论是否受随机种子影响。

建议设置：

- 只保留 `full_multitask` 和阶段 C 中最有解释价值的 2 到 3 个消融组。
- seed 使用 `42`、`2026`、`3407`。
- 跑全部 split。

结论以均值和标准差报告，不只报告单次最优结果。

## 7. 指标与解释方式

基础指标：

- `R2`
- `RMSE`
- `MAE`
- `Huber loss`

推荐派生指标：

- `delta_R2_vs_full`
- `delta_RMSE_vs_full`
- `generalization_gap = metric(random_split) - metric(group_split)`
- 按任务头的 macro average。
- 按样本数加权的 weighted average。

科研解释规则：

- 如果某模块只提升 `random_split`，但不提升 `chemical_group_split` 或 `chemical_species_group_split`，说明它可能主要增强了同分布拟合，不一定增强真实外推。
- 如果物种分类学模块显著提升 `species_group_split`，说明分类学层级可能捕捉了跨物种敏感性差异。
- 如果暴露时长模块在 NOEC/LOEC 任务上贡献更大，说明慢性毒性和暴露时间依赖可能被模型利用。
- 如果辅助任务提升主任务且 group split 也提升，说明 BAF/BCF 或 oral 任务可能提供了可迁移的污染物性质或生物有效性信息。
- 如果辅助任务降低主任务表现，需检查任务量纲差异、loss 权重过高、辅助标签质量或数据分布偏移。

## 8. 推荐图表

论文和学位论文建议优先输出：

- 不同 split 下 full model 与消融组性能柱状图。
- `delta_R2_vs_full` 热图，行为消融组，列为 split。
- 真实值-预测值图，按任务头或介质分面。
- 残差图，按化合物类别、物种类群、介质分组。
- generalization gap 图。
- 模块贡献雷达图或森林图。
- SHAP/permutation importance 图，解释分子描述符、物种和暴露条件贡献。

## 9. 当前需要确认的问题

1. 第一批正式实验是否以当前配置中的 5 个主任务头为准，还是要加入当前数据中样本量较大的 `ECx_Population`、`NOEC_Mortality`、`NOEC_Population`？
2. 第一轮消融是否先聚焦水相毒性，还是水相、土壤、沉积物一起进入多介质模型？
3. 远端 RTX 4060 Ti 16 GB 的训练预算希望控制在什么范围：每轮少于 2 小时、过夜运行，还是可以连续多天批量跑？
4. 传统机器学习基线是否要纳入同一实验矩阵，还是先只比较深度多任务模型的模块消融？
5. 最终论文图表优先中文学位论文风格，还是英文期刊风格？

## 10. 推荐下一步

1. 指纹位数沿用旧模型设定为 512 bit。
2. BAF/BCF 作为辅助任务标签，不作为主模型普通输入特征。
3. A/F 中 7:2:1 的 2 作为 fine-tune/adaptation 集，1 作为最终验证/测试集。
4. 主任务仅使用 EC/LC、NOEC、LOEC 下各效应终点，低样本任务头按阈值排除。
5. 土壤目标保持 `-log10(mg/kg)`。
6. 修复远端 GPU driver 状态并安装 `torch`。
7. 同步当前代码、配置和派生数据到远端。
8. 为 6 个核心消融组生成独立配置文件。
9. 新增批量远端运行脚本，按 run group 和 split 顺序执行。
10. 新增结果汇总脚本，统一生成指标总表和消融贡献图。
