# Project Status

更新时间：2026-06-10

## 当前阶段

已完成第一版 `A -> B -> C` 起步交付：

1. 技术蓝图文档。
2. 配置文件规范。
3. 示例实验配置。
4. 第一版 Python 包骨架。
5. CLI 入口。
6. 本机/远程训练执行器接口。
7. PySide6 GUI 训练控制台空壳。

## 已确认架构调整

- 工具需要能在本机运行。
- 训练主路径按远程 GPU 训练设计。
- 本机执行器用于配置验证、小样本调试和冒烟测试。
- 远程执行器第一版已经预留，当前通过 SSH 命令占位。

## 已验证命令

```powershell
python -m qsar_tl.cli validate-config --config configs\experiment.example.yaml
```

结果：配置读取成功。

```powershell
python -m qsar_tl.cli run --config configs\experiment.example.yaml --dry-run
```

结果：成功生成远程训练 dry-run 命令。

## 下一步建议

优先进行数据库字段扫描和字段映射落地：

1. SQLite 数据库路径已确认：`G:\QSAR迁移学习\ecotox_clean.sqlite`。
2. 数据库结构扫描报告已生成：`docs/database_schema_scan.md`。
3. 正式字段映射已生成：`configs/field_mapping.ecotox_clean.yaml`。
4. 下一步实现建模宽表构建模块。
5. 下一步实现目标变量单位换算和剔除原因记录。

## 本轮继续目标

- 已新增建模宽表和目标变量构建模块：`qsar_tl/data/modeling_tables.py`。
- 已新增 CLI 命令：`python -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml --limit 10000`。
- 第一版目标构建只纳入已标准化单位：
  - `water_mg_l` 转换为 `ptox_mol_l`
  - `water_mol_l` 转换为 `ptox_mol_l`
  - `soil_mg_kg` 转换为 `neg_log10_mg_kg`
  - `oral_mg_kg_d` 转换为 `neg_log10_mg_kg_bw_day`
- 其他单位暂时进入 `excluded_reason`，避免不可靠数据进入训练。

## 完整建模表构建结果

已运行：

```powershell
python -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml
```

结果：

- `wide_records`: 1,234,077
- `target_records`: 1,234,077
- `included_targets`: 510,934
- `excluded_targets`: 723,143

详细报告见：`docs/modeling_table_build_report.md`。

## 并行框架补全进度

- 已新增任务头映射与聚合样本框架：`qsar_tl/data/task_mapping.py`、`qsar_tl/data/task_tables.py`。
- 已新增 split 与传统基线框架：`qsar_tl/evaluation/splits.py`、`qsar_tl/training/baseline.py`。
- 已新增深度模型骨架：`qsar_tl/modeling/dataset.py`、`qsar_tl/modeling/network.py`、`qsar_tl/training/deep_train.py`。
- 已重构 GUI 控制台框架：`qsar_tl/gui/app.py`、`qsar_tl/gui/widgets.py`、`qsar_tl/gui/job_model.py`、`qsar_tl/gui/process_runner.py`。
- 路径约束：代码不写死本机盘符和解释器路径；数据库、输出和远程路径通过配置或 CLI 参数传入。

### 框架验证结果

已通过：

```powershell
python -m compileall qsar_tl scripts tests
python -m pytest tests -q
python -m qsar_tl.cli --help
```

测试结果：`17 passed, 1 skipped`。跳过项来自当前 base 环境的可选依赖差异；`qsar-ph3` 环境可运行 baseline 冒烟。

已完成 CLI 冒烟：

```powershell
python -m qsar_tl.cli build-task-tables --config configs\experiment.example.yaml --db <derived_db> --limit 2000
python -m qsar_tl.cli generate-split --config configs\experiment.example.yaml --db <derived_db> --split-name random_smoke --split-type random_split --limit 2000
python -m qsar_tl.cli run-baseline --config configs\experiment.example.yaml --db <derived_db> --split-name random_smoke --model random_forest --limit 500 --out <metrics_csv>
```

真实全量聚合结果：

- `task_records`: 510,934
- 第一批严格纳入任务记录：202,678
- `aggregated_task_records`: 80,725
- 主要任务头：`ECx_Mortality`、`ECx_Population`、`NOEC_Growth`、`NOEC_Mortality`、`NOEC_Population`

## GitHub 备份策略

- 本地数据库 `ecotox_clean.sqlite` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 原始对话记录 `USER.txt` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 技术蓝图、配置、代码骨架、数据库结构报告和数据画像报告可以提交。
