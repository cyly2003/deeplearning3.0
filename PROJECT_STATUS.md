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
E:\TOOLS\anaconda\python.exe -m qsar_tl.cli validate-config --config configs\experiment.example.yaml
```

结果：配置读取成功。

```powershell
E:\TOOLS\anaconda\python.exe -m qsar_tl.cli run --config configs\experiment.example.yaml --dry-run
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
E:\TOOLS\anaconda\python.exe -m qsar_tl.cli build-modeling-tables --config configs\experiment.example.yaml
```

结果：

- `wide_records`: 1,234,077
- `target_records`: 1,234,077
- `included_targets`: 510,934
- `excluded_targets`: 723,143

详细报告见：`docs/modeling_table_build_report.md`。

## GitHub 备份策略

- 本地数据库 `ecotox_clean.sqlite` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 原始对话记录 `USER.txt` 已通过 `.gitignore` 排除，不提交到 GitHub。
- 技术蓝图、配置、代码骨架、数据库结构报告和数据画像报告可以提交。
