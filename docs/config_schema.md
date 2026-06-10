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

targets:
  toxicity:
    value_rule: mean_then_min_max_midpoint

model:
  molecule_encoder: rdkit_descriptor_morgan
  species_encoder: taxonomy_context

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
    user: your_user
    project_dir: /home/your_user/ecotox_qsar_transfer
    python: /home/your_user/miniconda/envs/qsar/bin/python
    sync_data: true
    pull_outputs: true
```

## 3. data

`field_mapping` 用于把不同 SQLite 表和字段映射到标准内部字段。第一版不在代码中硬编码数据库字段。

## 4. targets

毒性值规则：

- `mean_then_min_max_midpoint`: 优先 mean；mean 缺失且剂量组数不少于 3 时用 min/max 中点。

单位策略：

- 水相：`mol/L` 和 `mg/L`。
- 土壤：主任务 `-log10(mg/kg)`。
- 沉积物：保留 `mg/kg`，后续视数据情况决定主目标。
- 经口：`mg/kg bw` 或 `mg/kg bw/day`，作为辅助任务。

## 5. model

第一版模型采用共享主干 + 多任务预测头：

- `molecule_encoder`: `rdkit_descriptor_morgan`
- `species_encoder`: `taxonomy_context`
- `task_heads`: 按“终点家族 + 效应类型”定义。

## 6. training

第一版使用：

- Huber Loss。
- 固定任务权重。
- 固定随机种子。

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

