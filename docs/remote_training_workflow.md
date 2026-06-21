# 远端训练同步与拉回流程

本文档记录当前远端 GPU 训练的第一版操作流程。远端配置文件为
`configs/experiment.remote.easyai.yaml`，远端项目目录为
`/home/easyai/DL1/ecotox_qsar_transfer`。

## 推荐同步方式

长期推荐采用“Git 同步代码 + rsync 同步大数据和结果”的方式：

- 代码、配置、文档进入 GitHub，便于版本追踪和复现实验。
- SQLite 数据库、派生建模表和训练输出不进 Git，由 `rsync` 或脚本同步。
- 如果本机没有 `rsync`，先使用本项目提供的 PowerShell + OpenSSH `scp` 脚本，后续再切换到 `rsync`。

当前脚本默认使用 Windows/OpenSSH 自带的 `ssh` 和 `scp`，不依赖本机固定盘符。
本机已配置 SSH 免密别名：`qsar-gpu` 和 `qsar-gpu-backup`。
如主地址不可用，可在任一脚本中临时加入 `-RemoteHost b.easy-ai.cloud`。

如果当前 PowerShell 里的 `python` 不是 Python 3.12，请在脚本中显式传入
`-LocalPython <python3.12解释器路径>`。

## 1. 检查远端 GPU 和 Python

```powershell
pwsh .\scripts\check_remote_gpu.ps1 -Config configs\experiment.remote.easyai.yaml
```

备用地址检查：

```powershell
pwsh .\scripts\check_remote_gpu.ps1 -Config configs\experiment.remote.easyai.yaml -RemoteHost b.easy-ai.cloud
```

该命令会检查远端系统、磁盘、`nvidia-smi`、Python 路径和 conda 版本，并确保远端项目目录存在。

## 2. 同步代码和配置

只同步项目代码、配置、文档和脚本：

```powershell
pwsh .\scripts\sync_to_server.ps1 -Config configs\experiment.remote.easyai.yaml
```

如果远端需要从原始 SQLite 数据库重新构建建模表：

```powershell
pwsh .\scripts\sync_to_server.ps1 -Config configs\experiment.remote.easyai.yaml -IncludeSourceDb
```

如果本机已经构建了 `outputs/derived/modeling_dataset.sqlite`，并希望远端直接训练或基线验证：

```powershell
pwsh .\scripts\sync_to_server.ps1 -Config configs\experiment.remote.easyai.yaml -IncludeDerivedDb
```

两个数据开关可以同时使用。数据路径来自配置文件，不写死本机绝对路径。

## 3. 远端冒烟测试

先执行配置验证和训练入口冒烟：

```powershell
pwsh .\scripts\train_remote.ps1 -Config configs\experiment.remote.easyai.yaml -SmokeTest
```

当前 `qsar_tl.training.train` 仍是训练入口占位实现，因此冒烟测试主要验证：

- SSH 端口与账号可用；
- 远端项目目录、配置文件和包导入可用；
- 远端 Python 能运行训练入口；
- 日志能写入 `outputs/logs`。

如需只查看将执行的远端命令：

```powershell
pwsh .\scripts\train_remote.ps1 -Config configs\experiment.remote.easyai.yaml -SmokeTest -DryRun
```

## 4. 远端执行训练

训练实现补齐后，使用同一脚本启动正式训练：

```powershell
pwsh .\scripts\train_remote.ps1 -Config configs\experiment.remote.easyai.yaml
```

训练日志会写入远端 `outputs/logs/remote_train_<timestamp>.log`。

受控深度训练预跑：

```powershell
pwsh .\scripts\train_remote.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -SplitName B_random_8_2 -Limit 10000 -Epochs 5 -BatchSize 256 -Device cuda:0 -OutDir outputs/experiments/deep_pilot_remote
```

## 4.1 远端实验序列

生成 A-F 划分并运行传统机器学习 baseline matrix：

```powershell
pwsh .\scripts\run_remote_sequence.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -GenerateSplits -RunBaselineMatrix
```

小样本冒烟：

```powershell
pwsh .\scripts\run_remote_sequence.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -GenerateSplits -RunBaselineMatrix -Limit 500
```

只查看远端命令：

```powershell
pwsh .\scripts\run_remote_sequence.ps1 -Config configs\experiment.remote.easyai.yaml -LocalPython <python3.12路径> -GenerateSplits -RunBaselineMatrix -DryRun
```

## 5. 拉回结果

拉回远端实验输出、日志和派生数据：

```powershell
pwsh .\scripts\sync_from_server.ps1 -Config configs\experiment.remote.easyai.yaml
```

默认拉回：

- `outputs/experiments`
- `outputs/logs`
- `outputs/derived`

可按需指定子目录：

```powershell
pwsh .\scripts\sync_from_server.ps1 -Config configs\experiment.remote.easyai.yaml -RemoteSubdirs outputs/logs,outputs/experiments/easyai_remote
```

## 6. 远端环境准备

首次同步后，建议在远端项目目录内执行一次可编辑安装：

```bash
cd /home/easyai/DL1/ecotox_qsar_transfer
/opt/anaconda3/bin/python -m pip install -e ".[ml]"
```

如果后续需要 RDKit 分子描述符，建议在远端 conda 环境中通过 conda-forge 安装 RDKit，再运行特征构建或训练流程。
