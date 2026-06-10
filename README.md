# QSAR Transfer Learning Workspace

本项目用于重构 ECOTOX-QSAR 迁移学习流程，目标是形成一个可复现、可扩展、适合科研论文和学位论文方法描述的建模系统。

当前阶段是第一版骨架：

- 从已初步清洗的 SQLite 关系数据库开始。
- 先构建建模宽表和条件聚合样本。
- 使用 RDKit 描述符与 Morgan 指纹作为第一版分子主信号。
- 使用物种分类学 embedding 与生态/实验上下文作为辅助输入。
- 主任务预测 EC/LC、NOEC、LOEC。
- BAF/BCF 与经口暴露作为辅助任务共享知识。
- GUI 使用 PySide6，工具可本机运行，但训练主路径按远程 GPU 训练设计。

## Recommended Entry Points

- 技术蓝图：[docs/technical_blueprint.md](docs/technical_blueprint.md)
- 配置规范：[docs/config_schema.md](docs/config_schema.md)
- 示例实验配置：[configs/experiment.example.yaml](configs/experiment.example.yaml)
- CLI 入口：`python -m qsar_tl.cli`
- GUI 入口：`python -m qsar_tl.gui.app`

## Current Data Build Commands

数据库结构扫描：

```powershell
python scripts\scan_sqlite_schema.py --db ecotox_clean.sqlite --out docs\database_schema_scan.md
```

数据库关键字段画像：

```powershell
python scripts\profile_ecotox_clean.py --db ecotox_clean.sqlite --out docs\database_profile.md
```

构建小规模建模宽表与目标变量表：

```powershell
python -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml --limit 10000
```

构建任务头和条件聚合样本：

```powershell
python -m qsar_tl.cli build-task-tables --config configs\experiment.example.yaml
```

生成 split 并运行基线：

```powershell
python -m qsar_tl.cli generate-split --config configs\experiment.example.yaml --split-name random_v1 --split-type random_split
python -m qsar_tl.cli run-baseline --config configs\experiment.example.yaml --split-name random_v1 --model random_forest --out outputs\experiments\baseline_random_v1_metrics.csv
```

所有路径均可通过配置文件或 CLI 参数覆盖，远程训练时不依赖本机盘符路径。

## Environment

用户当前常用 Python：

```powershell
E:\TOOLS\anaconda\python.exe --version
```

常用 conda 环境路径：

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3
```

第一版运行前建议确认：

- conda 环境名称
- Python 版本
- 虚拟环境路径
- 项目根目录路径
- SQLite 数据库路径
- 远程服务器 SSH 主机名、用户名、项目目录和训练环境
