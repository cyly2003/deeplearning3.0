# Runtime Summary 2026-06-30

本文件汇总远端 `/home/easyai/DL1/ecotox_qsar_transfer/outputs/logs` 中已完成 run 的运行时间。时间来自各 launcher 写入的 `run_times.csv`。

## 总体耗时

| 实验 | runs | 总耗时 h | 平均 min/run | 中位 min/run | 最短 min | 最长 min |
|---|---:|---:|---:|---:|---:|---:|
| v1.2.15 random transfer | 32 | 17.31 | 32.45 | 33.05 | 13.00 | 37.07 |
| v1.2.18 fixed C ablation | 50 | 26.50 | 31.80 | 31.90 | 22.37 | 38.12 |
| v1.2.19 single-domain BCE | 70 | 20.03 | 17.17 | 16.10 | 1.62 | 34.73 |

## v1.2.18 固定 Chemical-Holdout 消融耗时

| 策略/模块 | runs | 总耗时 h | 平均 min/run | 最短 min | 最长 min |
|---|---:|---:|---:|---:|---:|
| no_context | 5 | 2.05 | 24.64 | 22.37 | 26.92 |
| no_medium_adapter | 5 | 2.22 | 26.62 | 24.38 | 28.13 |
| no_species_lifestage | 5 | 2.58 | 30.93 | 27.77 | 35.23 |
| no_duration | 5 | 2.63 | 31.56 | 28.02 | 37.03 |
| no_descriptors | 5 | 2.76 | 33.09 | 28.55 | 37.03 |
| no_molecular_residual | 5 | 2.85 | 34.26 | 27.80 | 37.32 |
| no_fingerprint | 5 | 2.96 | 35.52 | 32.18 | 38.12 |
| no_censored_loss | 5 | 2.72 | 32.65 | 26.87 | 35.32 |
| no_source_weighting | 5 | 2.78 | 33.32 | 31.20 | 36.58 |
| no_toxicity_binning | 5 | 2.95 | 35.37 | 28.97 | 37.02 |

## v1.2.19 单域 BCE 耗时

| domain:split | runs | 总耗时 h | 平均 min/run | 最短 min | 最长 min |
|---|---:|---:|---:|---:|---:|
| aquatic:B | 5 | 2.71 | 32.55 | 32.23 | 32.95 |
| aquatic:C | 5 | 2.58 | 30.95 | 30.43 | 31.63 |
| aquatic:E fold-runs | 25 | 13.75 | 32.99 | 32.35 | 34.73 |
| soil:B | 5 | 0.14 | 1.70 | 1.68 | 1.72 |
| soil:C | 5 | 0.14 | 1.63 | 1.62 | 1.65 |
| soil:E fold-runs | 25 | 0.71 | 1.70 | 1.67 | 1.77 |

## v1.2.21 随机划分消融

新增脚本：`scripts/run_v1_2_21_random_split_ablation_remote.sh`。

默认 targeted 矩阵：

- random 8:2：`42/1042/2042/3042/4042` 五个 seed。
- random 5-fold：先跑 `seed42` 的 fold1-5。
- 模块消融：`no_context`、`no_species_lifestage`、`no_molecular_residual`。
- 策略消融：`no_source_weighting`、`no_toxicity_binning`、`no_censored_loss`。

启动状态：

- 启动时间：2026-06-30 16:55 (+08:00)。
- log：`outputs/logs/run_v1_2_21_random_split_ablation_priority_20260630_165543.log`。
- pidfile：`outputs/logs/run_v1_2_21_random_split_ablation_priority_20260630_165543.pid`。
- 当前首个正式 run：`random8_2_ablation_no_context_seed42_cebin_lw0025_censored_w0p01`。

注意：前台 smoke 因 SSH 超时中断了命令会话；已完成的 smoke 产物已迁移到 `_smoke` 输出根目录，正式 `v1_2_21_random_split_ablation_remote` 根目录不包含 smoke run。
