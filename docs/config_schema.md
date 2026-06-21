# 配置文件规范

第一版使用 YAML 作为实验配置主格式。GUI 负责读写 YAML，CLI 和训练脚本只读取配置，不直接依赖 GUI。

## 1. 顶层结构

```yaml
project:
  name: ecotox_qsar_transfer
  seed: 42

paths:
  sqlite_db: data/raw/ecotox_clean.sqlite
  output_dir: outputs/experiments/example

execution:
  mode: remote

data:
  field_mapping: configs/field_mapping.example.yaml
  aggregation:
    strategy: strict_duration_tolerance
  qc_aggregation:
    source_table: task_records
    qc_task_table: task_records_qc
    aggregate_table: aggregated_task_records_qc

targets:
  toxicity:
    value_rule: mean_then_min_max_midpoint

model:
  molecule_encoder: rdkit_descriptor_morgan
  species_encoder: taxonomy_context
  use_medium_adapters: true

training:
  loss: huber
  task_weights:
    main: 1.0
    bioaccumulation_aux: 0.3
    oral_aux: 0.3

evaluation:
  splits:
    - random_split
    - chemical_group_split
    - species_group_split
```

## 2. execution

`execution.mode` 控制训练运行位置：

- `local`: 本机运行，用于调试、冒烟测试和小样本试跑。
- `remote`: 远程运行，作为主要训练路径。

远程配置应包含：

```yaml
execution:
  mode: remote
  local_python: E:/TOOLS/anaconda/envs/qsar-ph3/python.exe
  remote:
    host: your.server
    port: 22
    user: your_user
    project_dir: /home/your_user/ecotox_qsar_transfer
    python: /home/your_user/miniconda/envs/qsar/bin/python
    sync_data: true
    pull_outputs: true
```

## 3. data

`field_mapping` 用于把不同 SQLite 表和字段映射到标准内部字段。第一版不在代码中硬编码数据库字段。

`data.qc_aggregation` 用于生成正式建模前的 QC 聚合表：

- `task_records_qc`: 保留每条任务记录的 QC 状态、robust z-score、文献权重和介质域。
- `aggregated_task_records_qc`: 同一文献/测试先内部汇总，再按出版年份权重进行跨文献加权平均。
- 明显离群值默认只在 `target_name + medium_domain + task_head` 分组样本数不少于 50 时自动排除，避免小样本任务被过度清洗。

## 4. targets

毒性值规则：

- `mean_then_min_max_midpoint`: 优先 mean；mean 缺失且 min/max 完整时用 min/max 中点，不再按剂量组数过滤。

单位策略：

- 水相：`mol/L` 和 `mg/L`。
- 土壤：主任务 `-log10(mg/kg)`。
- 沉积物：保留 `mg/kg`，后续视数据情况决定主目标。
- 经口：`mg/kg bw` 或 `mg/kg bw/day`，作为辅助任务。

介质实验推荐入口：

- 水相主实验使用 `aggregated_task_records_aquatic_ptox_qc`，只保留可比的 `ptox_mol_l` 目标；`aggregated_task_records_aquatic` 保留为水生生物广义暴露数据探索表，包含 `mg/kg`、`g/ha`、百分比等非水相浓度尺度。
- 土壤主实验按研究问题选择 `aggregated_task_records_soil_mg_kg_qc` 或 `aggregated_task_records_soil_ptox_qc`；`aggregated_task_records_soil` 和非 `_qc` 表只作为审计或旧流程对照。
- 沉积物当前样本量很小，`aggregated_task_records_sediment_mg_kg` 和 `aggregated_task_records_sediment_ptox` 更适合作为小样本迁移/外部验证入口，不建议单独训练完整主模型。

## 5. model

第一版正式模型采用共享主干 + target-scale/medium adapter + 多任务预测头：

- `molecule_encoder`: `rdkit_descriptor_morgan`
- `species_encoder`: `taxonomy_context`
- `use_medium_adapters`: 默认开启，用 `medium_domain + target_name` 形成 adapter 路由，隔离水相、土壤、沉积物和目标量纲的系统偏移。
- `task_heads`: 按“终点家族 + 效应类型”定义；`effect_level_x` 不混入任务头名称，而是作为独立效应水平变量进入模型，并通过专门的 effect-level 分支参与共享表示。
- `species_number`: 只保留为审计字段，不作为正式模型的物种上下文输入。

## 6. training

第一版使用：

- Huber Loss。
- 固定任务权重。
- 固定随机种子。
- `feature_zscore_correction`: 默认开启，仅用训练/微调训练样本拟合数值特征均值和标准差，对标准化后的极端 z-score 做截断，并将同一纠偏参数迁移到验证集、测试集和扰动评估。
- `effect_level_weighting`: 默认关闭。开启后仅用实际训练样本（`train` + finetune_train，不含 validation/test）中 `ECx/LCx/ICx/LDx` 的 `effect_level_x` 频率计算 `frequency^(-beta)` 样本权重，归一化到均值 1 后截断到 `min_weight`-`max_weight`，并与已有 `source_weighting` 样本权重相乘。
- `reporting.min_metric_group_n`: 默认 5；低于阈值或 R2 未定义的子任务保留在原始指标表中，但不进入过滤后的核心汇总和 HPO 目标。

## 7. evaluation

第一版指标：

- R2
- RMSE
- MAE
- Huber loss

第一版 split：

- random_split
- chemical_group_split
- species_group_split
- chemical_species_group_split
- medium_transfer_split

应用域报告：

- 化学应用域：Williams leverage 为主，Tanimoto 最大相似度作为补充。
- 物种应用域：默认使用 `kingdom/phylum/class_name/tax_order/family` 的 taxon prefix distance。
- 输出字段用于绘图和预测可靠性分层，不混入训练表。
