# Project Status

更新时间：2026-07-19 10:20 (+08:00)

## 2026-07-19 v1.2.41 完成，v1.2.42 E 系列 OOF 融合准备启动

- v1.2.41 已以 `[matrix_complete_no_winner]` 结束。两种子固定验证集上，S0 为
  R2 `0.6907`、MAE `0.5536`；S1/S2/S3 的 R2 分别为 `0.6171/0.6164/0.6175`，
  MAE 分别为 `0.6339/0.6346/0.6381`。三项均未同时改善 R2 与 MAE，因此没有
  扩展种子，也没有用测试集挑选候选。
- 该结果说明当前瓶颈不是 stage 3 轮数不足；将 trunk 从冻结改为全量解冻会明显
  遗忘前两阶段表征，SWA 与 0.7 Huber + 0.3 MSE 未能修复。后续保留 v1.2.40
  `X0_molar` 冻结 trunk 的三阶段模型，不再继续同类 full-unfreeze 调参。
- v1.2.42 E 系列只在已完成的 Direct 与 Transfer 基模型预测上学习轻量组合：
  E0 为 Transfer ensemble 锚点，E1 为受约束线性融合，E2 为上下文门控，E3 为
  有界残差头。基模型结构、前两阶段路由、目标尺度和外层 random 8:2 测试集均不变。
- 元模型采用 5-fold OOF：每个外层 stage-3 训练记录恰好被一个未见过该记录的基模型
  预测；外层 test 不写入任何 OOF split。每个 OOF 基模型的 early stopping 只使用其
  训练折内部划分，不能查看 OOF 留出折。
- 筛选仍使用 seeds `42, 3407`。v1.2.40 seed42 的固定 stage-3 validation identities
  被保留为选择集，并从元模型拟合行中完全排除；候选必须在该选择集上同时提高 native
  mol/kg R2 并降低 MAE。只有锁定 winner 后才补跑 seeds `2042, 8417`，用四种子 OOF
  重新拟合小头，并一次性报告外层 test。
- 新增：`scripts/build_v1_2_42_e_series_oof_splits.py`、
  `scripts/run_v1_2_42_e_series_remote.sh`、`scripts/summarize_v1_2_42_e_series.py`、
  `scripts/validate_v1_2_42_oof_run.py` 和 `qsar_tl/training/meta_ensemble.py`。
- 本地纯数据/元模型定向测试 `3 passed`；完整训练入口与 shell launcher 将在
  `qsar-gpu-new` 原镜像环境做 smoke 后再进入 formal。

## 2026-07-18 v1.2.38 本地土壤 mg/kg 目标尺度补跑完成

- 远端服务器已不可用，本次改为本地 `.venv-cuda\Scripts\python.exe` 和 `configs\experiment.local.cuda3050ti.yaml` 执行。
- 新增本地 launcher：`scripts/run_v1_2_38_soil_mgkg_random_local.ps1`；源表为 `aggregated_task_records_soil_mg_kg_qc`，目标尺度为 `neg_log10_mg_kg`。
- 已生成独立 split：`SoilMgkgQC2_B_random_8_2` 与 `SoilMgkgQC2_E_random_5fold_fold1-5`，未覆盖土壤 pTox 主线 split。
- smoke 初次使用远端配置时因 Windows DataLoader 多进程权限失败；切换本地配置 `num_workers=0` 后 smoke 通过。
- formal seed42 本地 6/6 run 完成：random 8:2 + random 5-fold。
- all test prediction rows 指标：random 8:2 `n=3054`，R2 `0.6582`，RMSE `0.7888`，MAE `0.5704`；5-fold 合并 `n=15236`，R2 `0.6526`，RMSE `0.7859`，MAE `0.5637`。
- 解释边界：这是土壤 `mg/kg` 独立目标尺度的随机划分可行性证据，不是 pTox-to-mg/kg 换算证据，也不能直接作为土壤风险限值结论。下一步应做 multi-seed 和 scaffold/similarity-cluster 验证。

更新时间：2026-07-12 12:49 (+08:00)

## 2026-07-12 服务器到期前远端结果保全与“实验汇总”补齐

- 远端项目 `/home/easyai/DL1/ecotox_qsar_transfer` 仍可访问；本次重新比对远端 `outputs/experiments` 与本地目录后，补拉了本地缺失的正式轻量结果：`runtime_summaries`、`v1_2_7_censored_ordinal_ad_first_batch_remote_summary`、`v1_2_15_random_split_policy_remote_summary`。
- 同步了关键日志/时间记录：v1.2.24 完整完成日志与 times CSV、v1.2.31/v1.2.33 分子信号日志与 times CSV、v1.2.34-v1.2.36 队列 nohup 日志。v1.2.24 本地旧日志长度较短，已用远端完整日志覆盖。
- 已按中文目录体系复制到 `实验汇总`：
  - `实验汇总/13_结构骨架聚类外推审计_v1_2_24/远端训练结果_summary`：v1.2.24 完整远端 summary，20 个文件，含 ensemble 总表和逐行预测。
  - `实验汇总/分子信号强度探索_20260707/远端摘要明细`：v1.2.31-v1.2.36 summary 明细，共 40 个文件。
  - `实验汇总/18_历史策略与方法筛选_正式摘要_v1_2_7_v1_2_15`：v1.2.7/v1.2.15 早期正式 summary，共 20 个文件（含 README）；仅作方法筛选追溯，不并入当前主线。
  - `实验汇总/10_运行时间记录`：新增 runtime_by_strategy、v1.2.24/v1.2.31/v1.2.33-v1.2.36 times CSV，并新增 `服务器到期补拉日志_20260712`。
- 新增保全说明与清单：`实验汇总/00_主线结果总览与追溯文档/服务器到期结果保全说明_20260712.md` 和 `服务器到期结果保全清单_20260712.csv`。
- 未批量拉回 v1.2.31-v1.2.36 raw 训练树和模型权重；这些目录主要是大型中间文件/权重/完整预测，当前 summary、audit、runtime 与必要逐行表已足够支撑现有结论。若后续需要重建图或重新聚合，可按 `docs/mainline_remote_inventory_20260710.md` 定向拉取。

## 2026-07-10 远端结果盘点、主线版本边界与清理候选

- 远端只读状态：`/home/easyai/DL1/ecotox_qsar_transfer` 可访问；RTX 4060 Ti 当前基本空闲，未发现实际训练进程。`pgrep` 只见系统/查询进程和 NVIDIA 队列线程；`nvidia-smi` 显示 GPU 使用约 4%、显存约 586/16380 MiB。
- 已拉回本地缺失的最新 summary 与 runtime 文件：
  - `outputs/experiments/v1_2_34_padel_descriptor_remote_summary`
  - `outputs/experiments/v1_2_35_padel_prior_clustered_remote_summary`
  - `outputs/experiments/v1_2_36_graph_only_remote_summary`
  - `outputs/logs/run_v1_2_34_padel_descriptor_times.csv`
  - `outputs/logs/run_v1_2_35_padel_prior_clustered_times.csv`
  - `outputs/logs/run_v1_2_36_graph_only_times.csv`
- 新拉回 summary 的审计状态均通过：required files present，exit_code `0`，aquatic_eval_rows `0`。总训练耗时约：v1.2.34 `5493 s`，v1.2.35 `5109 s`，v1.2.36 `3419 s`。
- 当前“最优版本号”必须按评价边界回答，不能合并成单个版本：
  - 随机插值/当前论文主性能边界：`v1.2.22`。no-metal random 8:2 5-seed ensemble：`n=2608`，R2 `0.7895`，RMSE `0.8865`，MAE `0.6243`；no-metal random 5-fold 5-seed ensemble：`n=13063`，R2 `0.7920`，RMSE `0.8899`，MAE `0.6200`。
  - 结构族外推压力测试：`v1.2.24`。scaffold/similarity-cluster holdout 5-seed ensemble：`n=2493`，R2 `0.1810`，RMSE `1.4063`，MAE `1.0664`；scaffold/similarity-cluster 5-fold：`n=11459`，R2 `0.3664`，RMSE `1.4808`，MAE `1.1178`。
  - 分子信号机制旁支：`v1.2.33-v1.2.36` 只作为诊断证据，不替代主线。`v1.2.33` RDKit full 是当前分子输入主模型对照；`v1.2.35` PaDEL prior clustered 在 random8_2 接近 RDKit full，但 scaffold 外推仍弱；`v1.2.34` PaDEL raw 和 `v1.2.36` graph-only 均不能作为主线替代。
- 未批量拉回 raw 训练目录：远端 raw 目录包含大体积 `predictions.csv` 与模型权重，`v1.2.31` 约 684 MB，`v1.2.33` 约 1.4 GB，`v1.2.34-v1.2.36` 各约 0.34 GB，`v1.2.26` 约 6.1 GB。当前主线判定使用 summary、ensemble prediction rows、audit 和 runtime 文件已足够；若后续需要重新聚合预测行或重建图，再定向拉取。
- 清理动作暂未执行。候选清单与保留/归档原则已写入 `docs/mainline_remote_inventory_20260710.md`：smoke/debug、probe/interim、早期 pilot 和不支撑当前主线结论的 raw 训练目录建议归档优先，不直接删除。

## 2026-07-08 v1.2.34-v1.2.36 PaDEL/graph 分子信号旁支完成

- 远端队列状态：`scripts/queue_v1_2_34_35_padel_after_graph_remote.sh` 已完成，GPU 当前空闲；v1.2.34 PaDEL raw 与 v1.2.35 PaDEL prior clustered 均 2/2 priority runs exit 0。v1.2.34 第一次启动在 2026-07-07 16:46 因 PaDEL cache miss descriptor 行宽不一致失败一次，修复后 16:57 重启并完成；该失败未产生有效指标。
- `v1.2.34` PaDEL raw descriptor：
  - random8_2 test：`n=2608`，R2 `0.7157`，RMSE `1.0302`，MAE `0.7270`。
  - scaffold_cluster_8_2 test：`n=2493`，R2 `-0.0731`，RMSE `1.6097`，MAE `1.2350`。
  - 解释：raw PaDEL 1444 维直接输入不优于 RDKit full，scaffold 外推明显变差，不能作为当前主线替代。
- `v1.2.35` PaDEL prior clustered head：
  - random8_2 test：`n=2608`，R2 `0.7646`，RMSE `0.9374`，MAE `0.6670`。
  - scaffold_cluster_8_2 test：`n=2493`，R2 `0.0726`，RMSE `1.4964`，MAE `1.1464`。
  - 解释：先验聚类 head 明显修复 raw PaDEL 的过宽 descriptor 输入问题，random8_2 接近 RDKit full；但 scaffold 外推仍弱于 RDKit full/graph-only，不建议替换主线。
- `v1.2.36` graph-only：
  - random8_2 test：`n=2608`，R2 `0.6717`，RMSE `1.1071`，MAE `0.7987`。
  - scaffold_cluster_8_2 test：`n=2493`，R2 `0.0779`，RMSE `1.4922`，MAE `1.1254`。
  - 解释：仅用 molecular graph（无 descriptor、无 Morgan）仍有分子信号，但性能低于 full；scaffold MAE 与 PaDEL prior 接近且略好。
- 当前结论：分子信号没有被物种/上下文 embedding 完全遮蔽；但主线性能不是由单一 RDKit descriptor 驱动。Morgan fingerprint 在 random 插值中贡献大，RDKit descriptor/graph/prior-clustered PaDEL 在 scaffold 外推中保留一定结构信号。论文表述应定位为 context-aware / species-informed QSAR，而非传统 descriptor-only QSAR。

## 2026-07-07 v1.2.33-v1.2.36 分子信号补跑队列启动

- 目标：按用户确认的“分子信号强度探索”边界补跑当前缺口，并启动 PaDEL/graph 旁支；最佳性能模型仍可 ensemble，机制/消融默认 seed `2042`。
- 已启动 v1.2.33 RDKit 分子信号消融 priority 队列：
  - 脚本：`scripts/run_v1_2_33_molecular_signal_ablation_remote.sh`。
  - 输出根：`outputs/experiments/v1_2_33_molecular_signal_ablation_remote`。
  - summary 根：`outputs/experiments/v1_2_33_molecular_signal_ablation_remote_summary`。
  - 范围：no-metal `random8_2` 与 `scaffold_cluster_8_2`，seed `2042`，`full`、`no_descriptors`、`no_fingerprint`、`descriptors_only`。
  - 远端状态：2026-07-07 15:44 已完成 priority 队列；`random8_2` 与 `scaffold_cluster_8_2` 各 4 个对照共 8/8 run 全部 exit 0，并已生成 `outputs/experiments/v1_2_33_molecular_signal_ablation_remote_summary`。
- PaDEL 分支准备：
  - 本地已安装 `padelpy==0.1.16`。
  - `scripts/build_padel_feature_cache.py` 已支持从 SQLite 读取唯一 SMILES 并用 PaDEL 2D descriptor 生成训练 JSONL cache；smoke 通过，3 个 SMILES 生成 1444 个 descriptor。
  - 全量 PaDEL 2D cache 已完成并同步远端：`outputs/features/molecular_features_padel_morgan512.jsonl`，`rows_written=6353/6353`，`descriptor_count=1444`，`descriptor_generation_failures=0`，`morgan_fingerprint_failures=0`。manifest 记录 `missing_descriptor_values=353986`、`nonfinite_descriptor_values=451`，后续解释 PaDEL 描述符时需保留缺失/非有限值审计。
  - 配置：`configs/experiment.remote.easyai.padel.yaml`（raw descriptor）与 `configs/experiment.remote.easyai.padel_prior_clustered.yaml`（`prior_clustered_heads`）。
  - launcher：`scripts/run_v1_2_34_padel_molecular_signal_remote.sh`，默认可用于 PaDEL raw；prior clustered 通过 `CONFIG`、`RUN_VERSION`、`EXPERIMENT_LABEL` 环境变量切换。
  - 本地自动队列监控已触发同步并启动远端 `scripts/queue_v1_2_34_35_padel_after_graph_remote.sh`；graph-only 完成后 PaDEL raw 已启动。
  - 集成修复：PaDEL cache miss/缺失 SMILES 会按 1444 维 cache schema 补零，不再退回 8 维 RDKit/fallback descriptor；PaDEL 极端 descriptor 原始值在统计与样本构建前统一裁剪到 `±1e12`，避免均值/方差 overflow。`tests/test_deep_experiment_cache.py` 当前 47 passed。
  - 远端状态：第一次 v1.2.34 raw 因 descriptor 行宽不一致退出；修复同步后于 2026-07-07 16:57 重新启动 `random8_2_full_seed2042_padel_descriptor`，当前进程处于 PaDEL 高维特征预处理/训练启动阶段，CPU 100%、RSS 约 7.3 GB，尚未进入 GPU epoch。v1.2.35 prior clustered 仍排在 v1.2.34 raw 后。
- graph-only 分支准备并排队：
  - graph cache 已生成：`outputs/features/molecular_graphs_no_metal_inorganic.jsonl`，6353/6353 unique SMILES written，failures=0。
  - 训练接口已接入纯 PyTorch message-passing graph encoder；新增 ablation `graph_only_molecule`，语义为不用 descriptor、不用 Morgan fingerprint，但保留物种/上下文 embedding。
  - 配置：`configs/experiment.remote.easyai.graph_only.yaml`。
  - launcher：`scripts/run_v1_2_36_graph_only_remote.sh`。
  - 排队脚本：`scripts/queue_v1_2_36_graph_only_after_v1_2_33_remote.sh` 已在远端启动；v1.2.33 完成后已于 2026-07-07 15:47 启动 graph-only priority。`random8_2_graph_only_molecule_seed2042_graph_only_molecule` 于 16:17 exit 0，`scaffold_cluster_8_2_graph_only_molecule_seed2042_graph_only_molecule` 于 16:44 exit 0，priority 队列完成。
- 验证：
  - `py_compile` 覆盖 `network.py`、`deep_train.py`、`deep_experiment.py`、PaDEL/graph feature 脚本，通过。
  - 三份配置 `experiment.remote.easyai.padel.yaml`、`experiment.remote.easyai.padel_prior_clustered.yaml`、`experiment.remote.easyai.graph_only.yaml` 均 `validate-config` 通过。
  - graph encoder 合成 batch 前向/反向通过。

## 2026-07-07 v1.2.32 分子信号强度探索汇总与 PaDEL/graph 旁支准备

- 目标：回应 Fig.3 SHAP 与敏感性结果之间的解释张力，系统整理“物种/上下文 embedding 是否压过分子信号”的当前证据，并为 PaDEL 描述符、PaDEL 先验聚类 head、molecular graph-only 分子输入保留旁支接口。
- 重要边界更新：用户已明确固定化学留出 `v1.2.18`（CAS-number 项目）后续不再考虑。当前汇总中 `v1.2.18` 只作为 `excluded_historical` 来源记录，不能用于当前结论、coverage 或补跑优先级。
- 第一阶段已完成，只读取现有结果，不补跑：
  - 汇总目录：`实验汇总/分子信号强度探索_20260707`。
  - 入口：`实验汇总/分子信号强度探索_20260707/README.md`。
  - 表格：`source_manifest.csv`、`molecular_signal_overall_metrics.csv`、`ablation_coverage_matrix.csv`、`rerun_gap_matrix.csv`。
  - 生成脚本：`scripts/build_molecular_signal_strength_summary.py`。
- 当前证据边界内可用结论：
  - `no_context` 与 `no_species_lifestage` 在随机插值下仍是最大/次大损失项，支持物种/上下文 embedding 信息量很强。
  - `no_molecular_size_descriptors` 在 no-metal random 8:2 下 MAE `-0.0068`、R2 `+0.0023`，在 scaffold-cluster 8:2 下 MAE `+0.0194`、R2 `-0.0217`；不支持“模型主要靠 MolWt/分子大小耦合获得性能”的结论。
  - 排除 `v1.2.18` 后，当前边界缺少 `no_descriptors`、`no_fingerprint`、`descriptors_only` 的有效对照。
- 推荐最小补跑矩阵：seed `2042`，先跑 no-metal `random_8_2` 与 `scaffold_cluster_8_2` 上的 `no_descriptors`、`no_fingerprint`、`descriptors_only`。最佳性能/最终模型仍可使用 ensemble；机制消融和诊断消融默认单 seed2042。
- 第二阶段旁支已完成最小实现，不改变主线默认行为：
  - PaDEL CSV/JAR 到兼容 JSONL 缓存：`qsar_tl/features/padel.py`，命令行入口 `scripts/build_padel_feature_cache.py`。
  - descriptor prior grouping：`qsar_tl/features/descriptor_groups.py`，模板 `configs/padel_descriptor_clusters.example.yaml`。
  - 可选 descriptor head：`EcotoxMultiTaskNetwork` 支持 `raw`（默认不变）、`dense_head`、`prior_clustered_heads`。
  - graph 预留接口：`qsar_tl/features/molecular_graph.py`，命令行入口 `scripts/build_molecular_graph_cache.py`；当前仅生成 graph cache，不接入训练循环。
  - 设计说明：`docs/molecular_signal_strength_padel_graph_branch_design_20260707.md`。
- 兼容性修正：训练路径现在从 molecular feature cache 读取 `descriptor_names`，并在 `preprocessing.json` 记录；`no_molecular_size_descriptors`、toxicity-bin 分子量读取、proxy-distance source weighting 和 descriptor group 解析均改为 name-based，避免 PaDEL 描述符顺序改变造成隐性错误。
- 验证：
  - `E:\TOOLS\anaconda\python.exe -m py_compile qsar_tl\modeling\network.py qsar_tl\training\deep_experiment.py qsar_tl\features\descriptor_groups.py qsar_tl\features\padel.py qsar_tl\features\molecular_graph.py scripts\build_padel_feature_cache.py scripts\build_molecular_graph_cache.py scripts\build_molecular_signal_strength_summary.py scripts\explain_deep_model.py` 通过。
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_deep_experiment_cache.py tests\test_traditional_ml_descriptor_effect_baselines.py` 通过：51 passed，1 个 PyTorch/NumPy warning。
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_molecular_signal_strength_summary.py` 通过并生成汇总包。

## 2026-07-07 v1.2.31 分子量/分子大小描述符敏感性验证 priority 队列完成

- 背景：`mg/L -> mol/L -> pTox` 换算使用分子量，公式上 `pTox = -log10(mg/L) + 3 + log10(MW)`；同时模型输入中包含 `MolWt` 和其他分子大小相关描述符，因此需要验证模型性能是否主要依赖该目标尺度耦合。
- 深度模型新增 ablation：`no_molecular_size_descriptors`，只遮蔽 `MolWt`、`TPSA`、`HeavyAtomCount`、`NumHAcceptors`、`NumHDonors`、`RingCount`、`RotatableBonds`，保留 Morgan fingerprint、`MolLogP`、物种/上下文、source weighting、toxicity binning 和 censored loss。
- 传统 ML baseline 新增参数：`--descriptor-sensitivity drop_molecular_size_related`，用于在 descriptor-only baseline 中删除更宽泛的分子量/分子大小/表面积/环柔性/VSA/Chi/BCUT 代理列；默认仍为 `full`，不改变 v1.2.28-v1.2.30 可复现性。
- 新增 launcher：`scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh`。
  - `priority`：seed2042 下跑 no-metal random 8:2 与 scaffold/similarity-cluster holdout，各自对比 `full` 和 `no_molecular_size_descriptors`。
  - `random` / `scaffold` / `all`：扩展到 random 5-fold、scaffold 5-fold 或完整矩阵。
  - 计划输出根：`outputs/experiments/v1_2_31_molecular_size_sensitivity_remote`；汇总根：`outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary`。
- 远端启动：2026-07-07 01:35 (+08:00) 已同步代码并 detached 启动 `priority` 队列。
  - 主进程：PID `405007`。
  - 日志：`outputs/logs/run_v1_2_31_molecular_size_sensitivity_20260707_013548.log`。
  - 当前首个 run：`random8_2_full_seed2042_molecular_size_sensitivity`，split 为 `M_v2_aquatic_to_soil_ptox_no_metal_adapt_B_random_8_2_f100`。
- 运行完成：2026-07-07 03:15 (+08:00)，4/4 formal runs 完成，exit_code 全部为 0。
  - summary：`outputs/experiments/v1_2_31_molecular_size_sensitivity_remote_summary`。
  - runtime：`outputs/logs/run_v1_2_31_molecular_size_sensitivity_times.csv`。
  - 摘要：`docs/v1_2_31_molecular_size_sensitivity_summary.md`。
- test 指标：
  - no-metal random 8:2 full：`n=2608`，R2 `0.7635`，RMSE `0.9397`，MAE `0.6724`。
  - no-metal random 8:2 no_molecular_size_descriptors：`n=2608`，R2 `0.7658`，RMSE `0.9350`，MAE `0.6656`；相对 full，MAE `-0.0068`、R2 `+0.0023`。
  - scaffold/similarity-cluster holdout full：`n=2493`，R2 `0.1204`，RMSE `1.4574`，MAE `1.1114`。
  - scaffold/similarity-cluster holdout no_molecular_size_descriptors：`n=2493`，R2 `0.0988`，RMSE `1.4752`，MAE `1.1308`；相对 full，MAE `+0.0194`、R2 `-0.0217`。
- 当前结论：随机插值下删掉分子大小描述符没有损失，反而略好；scaffold 结构族外推下有小幅损失，但远小于总体外推误差。该结果不支持“模型主要靠分子量换算耦合取得性能”的担忧；但 `MolWt`/分子大小特征解释仍需降调为“化学结构/分子大小代理，且部分与 pTox 单位换算尺度耦合”，不能写成独立机制证明。
- 后续：内部决策暂不需要立即扩展 full 5-fold；若要作为论文定量敏感性结论，建议补 scaffold 5-fold 或多 seed。

## 2026-07-06 v1.2.24 scaffold/similarity-cluster 外推汇总同步并纳入 Figure 2

- 背景：Figure 2 需要同时展示随机验证边界与 scaffold/similarity-cluster 结构族外推边界，并补充每种验证边界下的子任务表现雷达图。
- 数据同步：已从远端同步 `outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary` 到本地；该目录包含 combined summary、fold summary、holdout/5-fold ensemble prediction rows、family/effect/toxicity-bin/task 汇总。
- 关键 ensemble 指标：
  - scaffold/similarity-cluster holdout：`n=2493`，`R²=0.1810`，`MAE=1.0664`，`Huber loss=0.6746`。
  - scaffold/similarity-cluster cross-validation：`n=11459`，`R²=0.3664`，`MAE=1.1178`，`Huber loss=0.7224`。
- Figure 2 已更新：
  - 组图：`outputs/paper_figures/fig2_selected_model_performance/fig2_selected_model_performance_composite.svg/png`。
  - 独立子图：4 个预测诊断面板 + 4 个任务级雷达图面板，均位于 `outputs/paper_figures/fig2_selected_model_performance/panels`。
  - 任务雷达数据与任务代码映射：`outputs/paper_figures/fig2_selected_model_performance/tables/fig2_task_radar_data.csv` 和 `fig2_task_code_mapping.csv`。
- 解释边界：scaffold/similarity-cluster 结果是结构族外推压力测试，性能显著低于随机验证是预期现象，应解释为新结构族泛化难度增加，而不是直接作为训练失败。随机验证仍用于同分布/近同分布插值表现，二者不能混成同一类验证结论。

## 2026-07-06 v1.2.30 水相物种-终点传统 ML 扩展 baseline 完成

- 背景：用户确认水相 v1.2.28 top 30 物种-终点组合也不是全量。数据库核查显示：水相 `n>=200` 有 199 个物种-终点组合、97 个物种；`n>=100` 有 407 个组合、199 个物种；`n>=50` 有 781 个组合、384 个物种。
- 目标：先补跑计算量可控且验证集较稳的水相 `n>=200` 物种-终点传统机器学习 baseline，保持与 v1.2.28/v1.2.29 相同特征策略和模型集合。
- 数据与划分：
  - 源表：`aggregated_task_records_aquatic_ptox_qc`。
  - 子任务：全部 `n>=200` 的水相物种-终点组合，共 199 个。
  - 物种数：97 个。
  - 划分：每个子任务内 seed=42 row-random 8:2；`min_train=160`、`min_validation=40`。
- 模型：XGBoost、LightGBM、Random Forest、KNN、PLS；Optuna TPE `n_trials=3`。
- 输出：
  - 原始输出根：`outputs/experiments/v1_2_30_aquatic_species_endpoint_ml_descriptor_effect_n200`。
  - 汇总目录：`实验汇总/机器学习基线_分子描述符效应水平_水相扩展n200`。
  - 图表：995 个模型散点图，每个 PNG/SVG，共 1,990 个图文件。
  - 描述符表：199 个子任务 compound descriptor table。
  - 摘要：`docs/v1_2_30_aquatic_species_endpoint_ml_n200_summary.md`。
- 完成情况：995/995 模型完成，skipped=0。
- 水相扩展版总体模型排序（按 weighted validation RMSE）：LightGBM `0.8778`、XGBoost `0.8798`、Random Forest `0.9101`、KNN `0.9762`、PLS `1.3223`。
- best-by-subtask 数：XGBoost 91、LightGBM 64、Random Forest 24、KNN 13、PLS 7。
- 解释边界：v1.2.30 是水相物种-终点内 row-random 插值 baseline，适合补图和描述符-only 传统 ML 对比；仍不代表跨物种、跨终点或新化合物外推能力。

## 2026-07-06 v1.2.29 土壤物种-终点传统 ML 扩展 baseline 完成

- 背景：用户指出 v1.2.28 土壤 top 30 物种-终点组合覆盖不够全。核查后确认 v1.2.28 土壤结果覆盖 30 个组合、12 个物种；若按 `n>=30` 阈值，土壤可扩展到 73 个组合、24 个物种。
- 目标：补跑土壤域更完整的物种-终点传统机器学习 baseline，保持与 v1.2.28 相同特征策略和模型集合。
- 数据与划分：
  - 源表：`aggregated_task_records_soil_ptox_qc`。
  - 子任务：全部 `n>=30` 的土壤物种-终点组合，共 73 个。
  - 物种数：24 个。
  - 划分：每个子任务内 seed=42 row-random 8:2；`min_train=24`、`min_validation=6`。
- 模型：XGBoost、LightGBM、Random Forest、KNN、PLS；Optuna TPE `n_trials=3`。
- 输出：
  - 原始输出根：`outputs/experiments/v1_2_29_soil_species_endpoint_ml_descriptor_effect_n30`。
  - 汇总目录：`实验汇总/机器学习基线_分子描述符效应水平_土壤扩展n30`。
  - 图表：364 个模型散点图，每个 PNG/SVG，共 728 个图文件。
  - 描述符表：73 个子任务 compound descriptor table。
  - 摘要：`docs/v1_2_29_soil_species_endpoint_ml_n30_summary.md`。
- 完成情况：365 个预期模型中 364 个完成；`Oryza sativa / ECx_GeneticDamage` 的 PLS 因 sklearn PLS 内部 NaN loading 退化被 skipped，保留审计记录。
- 扩展版土壤总体模型排序（按 weighted validation RMSE）：XGBoost `1.0195`、Random Forest `1.0334`、KNN `1.0567`、LightGBM `1.1536`、PLS `1.2353`。
- 解释边界：v1.2.29 比 v1.2.28 更适合做土壤物种覆盖和拼图素材；但因阈值降到 `n>=30`，许多子任务验证集较小，应作为扩展/探索性 baseline，正式结论仍优先参考样本更充足的物种-终点组合。

## 2026-07-06 v1.2.28 物种-终点传统机器学习 baseline 完成

- 目标：按“不同物种 × 不同毒性终点”单独建模，补齐 XGBoost、LightGBM、Random Forest、KNN、PLS 的传统机器学习对比；输入仅使用文献驱动筛选的 RDKit 2D 分子描述符并集和 `effect_level_x` 派生特征，不使用物种上下文、分类学、介质、终点标签或目标/浓度字段作为模型输入。
- 执行环境：本地 `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe`；Optuna TPE 贝叶斯调参，每个模型/子任务 `n_trials=3`；固定 seed=42。
- 数据与划分：
  - 水相：`aggregated_task_records_aquatic_ptox_qc`，281,395 条可建模记录。
  - 土壤：`aggregated_task_records_soil_ptox_qc`，16,014 条可建模记录。
  - 子任务：水相 top 30 个物种-终点组合、土壤 top 30 个物种-终点组合；每个组合内 row-random 8:2 train/validation。
  - 特征：121 个输入特征，包括 117 个 RDKit QSAR 常用描述符和 4 个 effect-level 特征；缺失值仅用 train-only median 插补，零方差特征在训练折内移除。
- 输出：
  - 原始输出根：`outputs/experiments/v1_2_28_species_endpoint_ml_descriptor_effect_baselines_stable_pls`。
  - 归档汇总：`实验汇总/机器学习基线_分子描述符效应水平`。
  - 结构：`水生/<物种>/<终点>/图表|指标表|描述符表|预测值表` 与 `土壤/<物种>/<终点>/...`。
  - 图表：299 个模型散点图，每个同时导出 PNG/SVG，共 598 个图文件；坐标轴、刻度、图例和指标文本加粗。
  - 描述符表：60 个子任务各 1 份 compound descriptor table，便于后续相关性矩阵和应用域图。
  - 特征选择依据：`特征选择依据/descriptor_selection_rationale.md` 和 `descriptor_selection_reference.csv`，列出 Hansch/Fujita、Randic、Kier/Hall、Balaban、Wildman/Crippen、Ertl、Labute、Todeschini/Consonni 等原始或经典文献依据。
- 关键结果：
  - 共 300 个预期模型组合中 299 个完成；`Oryza sativa / ECx_GeneticDamage` 的 PLS 因 sklearn PLS 内部数值退化被 skipped，输入矩阵和目标值均无 NaN，因此保留审计记录而不伪造结果。
  - 水相模型总体排序（按加权 validation RMSE）：LightGBM `0.8374`、XGBoost `0.8465`、Random Forest `0.8684`、KNN `0.9428`、PLS `1.4661`。
  - 土壤模型总体排序（按加权 validation RMSE）：XGBoost `0.9084`、Random Forest `0.9416`、KNN `0.9813`、LightGBM `1.0459`、PLS `1.1060`。
  - 赤子爱胜蚓 `Eisenia fetida / ECx_Mortality`：XGBoost 最好，validation `R2=0.5704`、RMSE `1.5956`、MAE `1.1495`；`ECx_Growth` 等其他赤子爱胜蚓终点样本量不足，保留在低样本 skipped/描述符审计中。
- 解释边界：该结果是子任务内 row-random 插值 baseline，适合回答“在固定物种和固定终点内，传统 ML 使用分子描述符+效应水平能拟合到什么程度”；不能作为跨物种、跨终点或新化合物/骨架外推证据。

## 2026-07-03 v1.2.27 传统机器学习 descriptor+effect-level 本地基线完成

- 目标：按用户要求补跑水相、土壤、不同物种、不同毒性终点的传统机器学习基线；输入仅使用 RDKit 2D 分子描述符和 `effect_level_x` 派生特征，不使用物种上下文、分类学、介质标签、终点标签或目标/浓度字段作为模型输入。
- 执行环境：本地 `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe`，Optuna TPE 贝叶斯调参；基础解释器 `E:\TOOLS\anaconda\python.exe` 的 SciPy/sklearn DLL 不可用，因此未使用。
- 数据与划分：
  - 水相：`aggregated_task_records_aquatic_ptox_qc`，281,395 条可建模记录。
  - 土壤：`aggregated_task_records_soil_ptox_qc`，16,014 条可建模记录。
  - 划分：固定 seed=42 的本地 row-random 8:2 train/validation；这是 descriptor-only 同分布插值基线，不作为 chemical/scaffold 外推证据。
  - 子任务：水相 16 个、土壤 15 个；低样本或验证集不足组合写入 skipped audit。
- 模型与调参：`LightGBM`、`ExtraTrees`，每个模型/子任务 4 次 Optuna TPE trial，以 validation RMSE 为目标；极端 RDKit descriptor 值 `abs(value)>1e12` 按缺失处理，缺失值用 train-only median 插补。
- 输出：
  - 原始输出根：`outputs/experiments/v1_2_27_traditional_ml_descriptor_effect_baselines`。
  - 归档汇总：`实验汇总/12_传统机器学习基线_分子描述符效应水平`。
  - 指标表：`traditional_ml_descriptor_effect_all_metrics.csv`、`traditional_ml_descriptor_effect_best_by_subtask.csv`、`traditional_ml_descriptor_effect_best_scope_summary.csv`、`traditional_ml_descriptor_effect_hpo_trials.csv`、`traditional_ml_descriptor_effect_skipped_subtasks.csv`。
  - 图表：29 张 PNG + SVG，按 `图表/<domain>/<scope>/<task>/` 分类；每张为单任务 observed-vs-predicted 散点图，标题只标任务名，如 `ECx_Growth`。
- 关键结果（best-by-subtask validation）：
  - 水相总体：ExtraTrees `R2=0.5542`，RMSE `1.2623`，MAE `0.9601`。
  - 土壤总体：LightGBM `R2=0.6098`，RMSE `1.1303`，MAE `0.8091`。
  - 水相 endpoint 中 `ECx_Immobilization` 表现较强：LightGBM `R2=0.8341`，RMSE `0.8141`，MAE `0.5789`。
  - 土壤 endpoint 中 `ECx_Growth` 表现较强：LightGBM `R2=0.7151`，RMSE `0.8576`，MAE `0.6476`。
  - 物种和物种-终点内插值普遍高于总体，但这是过滤到特定物种/任务后的同分布拟合能力，不应解释为跨物种泛化。

## 2026-07-03 v1.2.26 no-metal/inorganic 随机主线 targeted 消融完成

- 目标：在 `v1.2.22` 去除金属/无机物后的随机划分主线上，复用 `v1.2.21` 的 targeted 消融项，检查模块/训练策略贡献排序是否仍与全数据随机插值场景一致。
- 新增 launcher：`scripts/run_v1_2_26_no_metal_random_split_ablation_remote.sh`。
  - 数据库：`outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`。
  - 源表：`aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic`。
  - split：`M_v2_aquatic_to_soil_ptox_no_metal_adapt_B_random_8_2_f100` 与 `M_v2_aquatic_to_soil_ptox_no_metal_adapt_E_random_5fold_fold1-5_f100`。
  - 消融项：`no_context`、`no_species_lifestage`、`no_molecular_residual`、`no_source_weighting`、`no_toxicity_binning`、`no_censored_loss`。
  - 默认 seed：`2042`；设计为 random 8:2 6 run + random 5-fold 5×6 run，共 36 个 formal run。
- seed 选择依据：`outputs/experiments/v1_2_22_no_metal_random_split_remote_summary/split_policy_run_summary.csv` 中，seed `2042` 同时是 no-metal random 8:2 和 random 5-fold 加权 MAE 最低的单模型 seed。
  - random 8:2 seed2042：n=2608，R2=0.7673，RMSE=0.9320，MAE=0.6597，Huber=0.3376。
  - random 5-fold seed2042：按 fold test n 加权 MAE=0.6554，RMSE=0.9322，Huber=0.3312。
- 运行状态：
  - 远端等待队列在 `v1.2.24` 结束后自动启动，2026-07-03 22:36:54 (+08:00) 完成 summarize。
  - 36/36 formal runs 完成，exit_code 全部为 0。
  - 总训练耗时 15.18 h，平均 25.30 min/run，范围 20.90-29.30 min/run。
  - 等待日志：`outputs/logs/run_v1_2_26_no_metal_random_split_ablation_wait_20260703_010346.log`。
  - run_times：`outputs/logs/run_v1_2_26_no_metal_random_split_ablation_times.csv`。
  - 输出根：`outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote`。
  - 汇总根：`outputs/experiments/v1_2_26_no_metal_random_split_ablation_remote_summary`。
- 结果摘要：`docs/v1_2_26_no_metal_random_split_ablation_summary.md`。
- 远端验证与完整性：
  - `bash -n scripts/run_v1_2_26_no_metal_random_split_ablation_remote.sh` 通过。
  - 6 个 no-metal random transfer split 均已在远端 SQLite 中查到非零 split assignment。
- 核心结果：相对于 v1.2.22 no-metal seed2042 full baseline，MAE 增量为：
  - `no_context`：random 8:2 `+0.2131`，random 5-fold `+0.2366`，损失最大。
  - `no_species_lifestage`：random 8:2 `+0.1040`，random 5-fold `+0.0952`，损失第二。
  - `no_toxicity_binning`：random 8:2 `+0.0315`，random 5-fold `+0.0391`，中等损失。
  - `no_source_weighting`：random 8:2 `+0.0091`，random 5-fold `+0.0133`，影响较小。
  - `no_censored_loss`：random 8:2 `-0.0136`，random 5-fold `+0.0047`，接近 baseline。
  - `no_molecular_residual`：random 8:2 `-0.0092`，random 5-fold `+0.0025`，接近 baseline。
- 解释边界：该结果支持“有机物主导的随机插值场景下，context 与 species/lifestage 仍是主贡献源”；source weighting、censored loss、molecular residual 在随机插值下不是强贡献项。该结论不能替代 `v1.2.24` scaffold/cluster chemical-family 外推证据。

## 2026-07-02 v1.2.24 scaffold/cluster chemical-family holdout 主线已启动

- 目标：替代原 CAS 号 `C_chemical_holdout_8_2` 的科学性不足，用结构骨架与 fingerprint 相似性簇定义新化合物族外推边界，并用当前最优主线策略完整重跑。
- 文献与方法依据：`docs/scaffold_cluster_split_method_20260702.md`。
  - Bemis-Murcko scaffold 用于结构母核分组。
  - Morgan/ECFP radius 2、2048 bits 用于 fingerprint 表征。
  - Butina/Tanimoto similarity cluster 用于补充 scaffold 不能覆盖的高相似结构近邻。
  - 默认阈值：Tanimoto similarity `0.65`；该阈值作为方法参数使用，最终合理性由 `tanimoto_leakage_summary.csv` 审计确认，而不是当作通用常数。
- 新增脚本：
  - `scripts/build_scaffold_cluster_splits.py`：本地 RDKit 构建结构组、写入 `G/H` split、输出审计。
  - `scripts/import_split_assignments_csv.py`：远端无 RDKit 时，从 CSV 导入 split assignments。
  - `scripts/run_v1_2_24_scaffold_cluster_holdout_remote.sh`：完整远端 launcher。
- 本地 split/audit：
  - 数据库：`outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`。
  - 土壤源表：`aggregated_task_records_soil_ptox_qc_no_metal_inorganic`。
  - 审计目录：`outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_audit`。
  - 原始 no-metal soil rows：13,549。
  - 可解析有机母体并纳入 scaffold/cluster split：11,992 rows，1,088 chemical units，550 structure groups。
  - 无有机母体而排除：117 chemical units，1,557 rows，主要为 NaCl、硼酸、SO2、无机盐等残留。
  - `G_scaffold_cluster_8_2`：train 9,136，test 2,856。
  - `H_scaffold_cluster_5fold`：test 为 2,284 或 2,856，结构组不可拆分导致 fold3 较大。
  - `structure_overlap_audit.csv`：所有 split 的 group/canonical parent/scaffold overlap 均为 0。
  - `tanimoto_leakage_summary.csv`：holdout 无 test chemical 达到 `max_tanimoto_to_train >= 0.65`；5-fold 仅 fold4/fold5 有约 0.5% test chemical 略高于 0.65，均无 `>=0.80`。
- 远端处理：
  - 远端 `/opt/anaconda3/bin/python` 及已有 conda 环境均无 RDKit，因此结构 split 在本地生成，导出 `scaffold_cluster_split_assignments.csv` 后导入远端 SQLite。
  - 远端 `bash -n scripts/run_v1_2_24_scaffold_cluster_holdout_remote.sh` 通过。
  - 远端 `splits` 模式通过，生成 6 个 aquatic-to-soil transfer split；aquatic train 固定为 219,849。
  - smoke 通过：`RUN_HOLDOUT=1 RUN_5FOLD=1 FOLDS=1 SMOKE_SEEDS=42`，1 epoch pretrain + 1 epoch finetune，两个 smoke run 均完成并 summarize。
- 正式运行：
  - 2026-07-02 19:17:04 (+08:00) 已启动完整 `ensemble_all` 后台队列。
  - PID：`85458`。
  - 日志：`outputs/logs/run_v1_2_24_scaffold_cluster_holdout_20260702_191704.log`。
  - 输出根：`outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote`。
  - 汇总根：`outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_remote_summary`。
  - 设计：5 seeds × (`scaffold_cluster_8_2` holdout + `scaffold_cluster_5fold` fold1-5) = 30 formal runs。
  - 当前首个正式 run：`no_metal_scaffold8_2_seed42_cebin_lw0025_censored_w0p01`。

## 2026-07-02 v1.2.23 随机主线 CST-AD 完成

- 目标：在不重新训练模型的前提下，直接使用已拉回本地的远端随机划分主线预测结果，构建 Chemical-Species-Task Applicability Domain (CST-AD) 表征。
- 输入结果：
  - 主输入：`outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary` 中的 5-seed ensemble prediction rows。
  - 本轮不使用 `v1_2_21_random_split_ablation_remote` 作为 full baseline；该目录是随机划分 targeted 消融矩阵，适合后续模块贡献分析。
  - 数据库：`outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
- 新增脚本：`scripts/build_random_mainline_cst_ad.py`。
  - 直接重建 random 8:2 与 random 5-fold 的参考训练空间：aquatic source train + soil target finetune pool，排除对应 test fold。
  - Chemical AD：Morgan fingerprint 最大 Tanimoto 相似度 + Williams leverage，默认阈值 `0.5`，强覆盖阈值 `0.7`。
  - Species AD：taxonomy prefix similarity、species/genus/family/order 覆盖、species-task-family 覆盖。
  - Task AD：chemical/species/task head/task family 组合覆盖计数。
  - 明确禁用 ensemble-SD uncertainty tier；输出中不使用 `y_pred_member_*` 或 seed 间标准差做分层。
- 输出目录：
  - 表格：`outputs/experiments/v1_2_23_random_mainline_cst_ad`
  - 图：`outputs/figures/v1_2_23_random_mainline_cst_ad`
- 关键结果：
  - random 8:2 C+S+T：AD-A n=3164，coverage=96.49%，MAE=0.5896；AD-B n=114，MAE=0.6400；AD-C n=1，MAE=1.6373。
  - random 5-fold C+S+T：AD-A n=15648，coverage=96.26%，MAE=0.5904；AD-B n=585，MAE=0.6832；AD-C n=23，MAE=0.5220。
  - 高错误识别能力较弱：C+S+T 的 AUROC 约 0.50，说明随机插值划分下 CST-AD 主要是覆盖性/可靠性分层工具，不是强误差分类器。
  - learned species embedding 最近邻距离可作为补充几何诊断：S0/S1 距离接近 0，S2/S3/S4 距离升高；但与 abs_error 的秩相关约 0.018，不应单独作为误差预测指标。
- 决策：
  - 本轮主线依据改为随机 8:2 与随机 5-fold。
  - CAS 号 chemical holdout 不再作为本轮敏感性/压力测试依据，因为 CAS 分组隔离不等价于结构骨架外推。
  - 后续若要评估新化合物外推，应另做 scaffold/cluster/Tanimoto-distance split，而不是沿用 CAS holdout 作为科学外推依据。
- 本地验证：
  - `E:\TOOLS\anaconda\python.exe -m py_compile scripts\build_random_mainline_cst_ad.py`
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_build_random_mainline_cst_ad.py -q`
  - 结果：4 passed；本机仍出现 PyTorch/NumPy DLL 初始化 warning，但不影响本次脚本完成。

## 2026-07-02 v1.2.21 随机划分 targeted 消融完成

- v1.2.21 随机划分 targeted 消融已完成：60/60 run，exit_code 全部为 0。
- 运行时间：远端 run_times 累计 33.56 h，平均 33.56 min/run，最短 22.75 min，最长 38.92 min；launcher 于 2026-07-02 02:34 (+08:00) 完成 summarize。
- 输出目录：
  - run root：`outputs/experiments/v1_2_21_random_split_ablation_remote`
  - summary root：`outputs/experiments/v1_2_21_random_split_ablation_remote_summary`
  - run_times：`outputs/logs/run_v1_2_21_random_split_ablation_times.csv`
- 随机 8:2 与 5-fold 的消融排序高度一致：
  - `no_context` 损失最大：random 8:2 MAE 0.8650，random 5-fold MAE 0.8555。
  - `no_species_lifestage` 损失第二：random 8:2 MAE 0.7154，random 5-fold MAE 0.7168。
  - `no_toxicity_binning` 有中等损失：random 8:2 MAE 0.6557，random 5-fold MAE 0.6699。
  - `no_source_weighting` 与 `no_censored_loss` 在随机插值划分下接近 full 单模型基线。
  - `no_molecular_residual` 在随机划分下没有表现出必要性，MAE 反而略低于 v1.2.15 full 单模型汇总；该现象应解释为随机插值场景下的冗余/正则化差异，不应直接替代 fixed chemical-holdout 的外推结论。
- 新增结果摘要：`docs/v1_2_21_random_split_ablation_summary.md`。
- 注意：v1.2.22 no-metal/inorganic 随机划分敏感性实验已在 v1.2.21 结束后自动启动，当前远端 GPU 进程显示其正在运行。

## 2026-07-01 v1.2.22 去除无机/含金属后随机划分重跑准备

- 目标：不覆盖旧库和旧输出，另建 no-metal/inorganic 派生 SQLite，并用当前最优随机划分策略重跑随机 8:2 与随机 5-fold。
- 背景：
  - 当前 `aggregated_task_records_aquatic_soil_ptox_qc` 中 metal/metalloid 记录为 64,057/297,455 行，占 21.54%；对应 325/6,815 个 CAS，占 4.77%。
  - 这些记录行数占比较高，且 RDKit 描述符/Morgan fingerprint 对金属、无机物和盐类表征存在适用性风险，因此需要单独做去除后敏感性实验。
- 新增过滤建库脚本：`scripts/build_no_metal_inorganic_dataset.py`。
  - 输入旧库：`outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
  - 输出新库：`outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`。
  - 过滤规则：若某个 CAS/DTXSID 在 `target_records` 中被标记为 `chemical_class_l1=inorganic` 或 `chemical_class_l2=metal_metalloid`，则从新库的 `target_records`、`aggregated_task_records_aquatic_soil_ptox_qc`、`aggregated_task_records_aquatic_ptox_qc`、`aggregated_task_records_soil_ptox_qc` 对应副本中排除。
  - 原始 SQLite 不删行、不改表；新表统一使用 `_no_metal_inorganic` 后缀，便于与旧结果对照。
- 新增远端 launcher：`scripts/run_v1_2_22_no_metal_random_split_remote.sh`。
  - 默认使用当前最优策略：`tanimoto_to_finetune alpha=1.0`、authority CE bin loss weight `0.025`、censored loss weight `0.01`、`finetune_validation_fraction=0.2`。
  - 默认 5-seed：`42 1042 2042 3042 4042`。
  - 随机 8:2：5 个 seed。
  - 随机 5-fold：5 folds × 5 seeds。
  - 输出根：`outputs/experiments/v1_2_22_no_metal_random_split_remote`。
  - 汇总根：`outputs/experiments/v1_2_22_no_metal_random_split_remote_summary`。
- 本地验证：
  - `E:\TOOLS\anaconda\python.exe -m py_compile scripts\build_no_metal_inorganic_dataset.py scripts\build_aquatic_soil_adaptation_split.py`
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_build_no_metal_inorganic_dataset.py tests\test_aquatic_soil_adaptation_split.py tests\test_splits.py -q`
  - 结果：15 passed，1 skipped。
- 本机 `bash -n` 因 WSL/Bash 启动权限失败，需同步后在远端执行 `bash -n scripts/run_v1_2_22_no_metal_random_split_remote.sh`。
- 远端准备状态：
  - 已同步代码到 `/home/easyai/DL1/ecotox_qsar_transfer`。
  - 远端 `bash -n scripts/run_v1_2_22_no_metal_random_split_remote.sh` 通过。
  - 已执行 `bash scripts/run_v1_2_22_no_metal_random_split_remote.sh splits`，完成过滤库和 split 构建。
  - 新过滤库：`outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`。
  - 过滤库规模：
    - `target_records`：1,023,043 / 1,234,077 行，排除 211,034 行。
    - `aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic`：233,398 / 297,455 行，排除 64,057 行。
    - `aggregated_task_records_aquatic_ptox_qc_no_metal_inorganic`：219,849 / 281,435 行，排除 61,586 行。
    - `aggregated_task_records_soil_ptox_qc_no_metal_inorganic`：13,549 / 16,020 行，排除 2,471 行。
  - no-metal random 8:2 soil split：train 10,839，test 2,710；transfer split：aquatic train 219,849，soil finetune 10,839，soil test 2,710。
  - no-metal random 5-fold soil split：每折 train 10,839-10,840，test 2,709-2,710；transfer split 使用同一 aquatic train 219,849。
  - 2026-07-01 16:00 (+08:00) 已提交等待队列，等待当前 v1.2.21 launcher PID `3939959` 退出后自动启动完整 `ensemble_all`。
  - 等待日志：`outputs/logs/run_v1_2_22_no_metal_random_split_wait_20260701_160001.log`；等待 PID：`4089408`。

## 2026-06-30 运行时间汇总与 v1.2.21 随机划分消融启动

- 已新增运行时间汇总文档：`docs/runtime_summary_20260630.md`。
- 已汇总远端 `run_times.csv`：
  - v1.2.15 random transfer：32 runs，总耗时 17.31 h，平均 32.45 min/run。
  - v1.2.18 fixed C ablation：50 runs，总耗时 26.50 h，平均 31.80 min/run。
  - v1.2.19 single-domain BCE：70 runs，总耗时 20.03 h；aquatic 平均约 31-33 min/run，soil 平均约 1.6-1.8 min/run。
- v1.2.18 与 v1.2.19 已完成：
  - v1.2.18：50/50 run 完成。
  - v1.2.19：70/70 run 完成；远端队列已结束。
- 已新增随机划分 targeted 消融 launcher：`scripts/run_v1_2_21_random_split_ablation_remote.sh`。
  - random 8:2：五个 seed。
  - random 5-fold：先跑 seed42 的 fold1-5。
  - targeted 消融：`no_context/no_species_lifestage/no_molecular_residual` 与 `no_source_weighting/no_toxicity_binning/no_censored_loss`。
  - 2026-06-30 16:55 已在远端启动 priority 矩阵；log 为 `outputs/logs/run_v1_2_21_random_split_ablation_priority_20260630_165543.log`。
  - 当前首个正式 run：`random8_2_ablation_no_context_seed42_cebin_lw0025_censored_w0p01`。

## 2026-06-28 random 5-seed refresh 完成并进入主线消融

- random split 5-seed refresh 已完成：
  - 远端完成判据：`complete=12 missing=0 expected=12`，`summary-seeds five_seed_ready`。
  - 结束日志：`outputs/logs/run_v1_2_15_random_split_policy_5seed_refresh_20260627_154528.log`，最后一个 run `random5fold_fold5_seed4042` 于 2026-06-27 22:53 完成，整体 `ensemble_all` 于 2026-06-27 23:01 status=0。
  - 5-seed random 8:2：n=3165，R2=0.7853，RMSE=0.8521，MAE=0.5925，Huber=0.2858。
  - 5-seed random 5-fold：n=15630，R2=0.7878，RMSE=0.8512，MAE=0.5939，Huber=0.2852。
- 本地最终指标包已刷新：
  - `outputs/experiments/final_mainline_comparison/existing_final_overall_summary.csv`
  - `outputs/experiments/final_mainline_comparison/existing_final_family_summary_30task.csv`
  - `outputs/experiments/final_mainline_comparison/existing_final_task_summary_30task.csv`
  - `outputs/experiments/final_mainline_comparison/existing_final_task_summary_35task.csv`
  - random 行已由 `random_*_3seed` 更新为 `random_*_5seed`。
- follow-up 队列已自动进入 v1.2.18 主线消融：
  - 已完成：`ablation_no_fingerprint_seed42_cebin_lw0025_censored_w0p01`，run-done status=0。
  - 正在运行：`ablation_no_descriptors_seed42_cebin_lw0025_censored_w0p01`。
  - v1.2.19 单域 B/C/E 仍在队列后段，等待 v1.2.18 矩阵完成。
- 本地验证：
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_final_mainline_summary.py`
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest tests\test_build_final_mainline_summary.py tests\test_summarize_split_policy_ensembles.py -q`
  - 结果：2 passed。

## 2026-06-27 v1.2.18/v1.2.19 后续矩阵入口准备与主线训练空间导出

- 目标：在 v1.2.17 锁定方法文档与已有最终指标后，继续落实三件事：random split 5-seed refresh、主线模块/策略消融入口、单域 aquatic-only/soil-only B/C/E 对照入口，并导出后续应用域/覆盖空间作图所需数据。
- random split 5-seed refresh：
  - 远端已启动：`SEEDS="3042 4042" ENSEMBLE_SEEDS="42 1042 2042 3042 4042" bash scripts/run_v1_2_15_random_split_policy_remote.sh ensemble_all`。
  - 当前日志：`outputs/logs/run_v1_2_15_random_split_policy_5seed_refresh_20260627_154528.log`。
  - 2026-06-27 16:22：`seed3042 random8_2` 已完成并落盘，`run-done status=0`，duration=2213s；脚本已自动进入 `seed3042 random5fold fold1`。
  - 新增只读状态检查入口：`scripts/check_v1_2_15_random_refresh.ps1`；完成判据为 `complete=12 missing=0 expected=12` 且 summary seeds 为 `42;1042;2042;3042;4042`。
  - 完成后需要同步 `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary`，再重跑 `scripts/build_final_mainline_summary.py` 刷新 random 5-seed 汇总。
- 新增训练空间导出脚本：
  - `scripts/export_task_train_space.py`。
  - 远端真实导出目录：`outputs/experiments/final_mainline_train_space`；本地已同步同名目录。
  - 导出 split：`M_v2_aquatic_to_soil_ptox_adapt_C_f100`；source table：`aggregated_task_records_aquatic_soil_ptox_qc`；split parts：`train + finetune`。
  - 输出文件：
    - `task_train_space_rows.csv.gz`：训练暴露空间逐行表。
    - `task_train_chemical_space.csv`：按 `task_head/split_part/chemical` 汇总的 SMILES 化合物空间。
    - `task_train_species_space.csv`：按 `task_head/split_part/species` 汇总的物种分类学空间。
    - `species_embedding_lookup.csv.gz`：由主线 seed42 模型 learned taxonomy embedding 拼接得到的物种 embedding lookup。
    - `task_train_species_embedding_space.csv.gz`：每个子任务物种空间与 embedding 的合并表。
    - `species_embedding_field_manifest.csv`：各 taxonomy embedding 字段在拼接向量中的维度范围。
  - 实际导出规模：296,368 行训练暴露记录、6,610 个化合物、5,393 个物种、116 个训练 task head。
  - 解释边界：`finetune` 是目标土壤微调池；seed-specific `finetune_validation` 没在该导出中拆出。embedding 来源是单个 seed42 模型，不代表 5-seed ensemble 的单一共享 embedding 空间。
- 新增主线消融 launcher：
  - `scripts/run_v1_2_18_mainline_ablation_remote.sh`。
  - 默认固定主线 split/参数，只改变模块或策略：`no_fingerprint`、`no_descriptors`、`no_species_lifestage`、`no_duration`、`no_context`、`no_medium_adapter`、`no_molecular_residual`，以及 `no_source_weighting`、`no_toxicity_binning`、`no_censored_loss`。
  - 已同步远端并通过 `bash -n`；尚未启动完整矩阵，避免与当前 random refresh 抢 GPU。
- 新增后续队列入口：
  - `scripts/run_v1_2_20_followup_queue_remote.sh`。
  - 作用：只负责等待 random 5-seed completion gate 后串联调用 v1.2.18 消融和 v1.2.19 单域 launcher，不复制训练参数。
  - 2026-06-27 16:36 远端验证：`bash -n` 通过；`check` 模式正确报告 `complete=1 missing=11 expected=12` 与 `summary-seeds stale_or_incomplete`。
  - 2026-06-27 16:39 已启动 `wait_then_followup` 后台队列；log 为 `outputs/logs/run_v1_2_20_followup_queue_20260627_163936.log`，pidfile 为 `outputs/logs/run_v1_2_20_followup_queue_20260627_163936.pid`。
  - 初始状态：`complete=1 missing=11 expected=12`，队列每 600 秒轮询一次；只有 random refresh 完成且 5-seed summary ready 后才会启动 v1.2.18/v1.2.19。
- 新增单域 B/C/E launcher：
  - `scripts/run_v1_2_19_single_domain_bce_remote.sh`。
  - 域：`aquatic` 使用 `aggregated_task_records_aquatic_ptox_qc`；`soil` 使用 `aggregated_task_records_soil_ptox_qc`。
  - 划分：`B_random_8_2`、`C_chemical_holdout_8_2`、`E_random_5fold_fold1-5`。
  - 固定 full 架构，显式 `--finetune-epochs 0` 与 `--source-weighting-method none`；保留主线的 target standardization、toxicity binning 和 censored loss。
  - 已同步远端并通过 `bash -n`；尚未启动完整矩阵。
- 本地验证：
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m py_compile scripts\export_task_train_space.py`
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest tests\test_export_task_train_space.py -q`
  - 结果：新增导出测试 1 passed。

## 2026-06-27 v1.2.17 主线材料方法文档与已有最终指标汇总完成

- 目标：在开始补跑 5-seed random split、主线 5-seed 消融和单域 B/C/E 基线前，优先固定当前主线方法学描述和已有最终指标包，避免后续实验继续扩展后难以追溯。
- 新增材料与方法文档：
  - `docs/final_mainline_methods_materials.md`。
  - 口径：当前主线定义为 `v1.2.12` aquatic-to-soil pTox transfer，固定 `M_v2_aquatic_to_soil_ptox_adapt_C_f100` chemical-holdout，5-seed anchor ensemble。
  - 明确区分：chemical-holdout split 按 `cas_number` 分组隔离，不按 Tanimoto 阈值切分；Tanimoto 0.5 属于 AD audit 阈值，source weighting 的 Tanimoto 属于训练权重。
  - 明确 taxonomy/context 不是整条 path 合并 embedding，而是 `latin_name/kingdom/phylum/class_name/tax_order/family/genus/species` 等字段各自 embedding 后拼接进入主干网络。
- 新增已有最终指标汇总脚本：
  - `scripts/build_final_mainline_summary.py`。
  - 输出目录：`outputs/experiments/final_mainline_comparison`。
  - 输出文件：
    - `existing_final_overall_summary.csv`
    - `existing_final_family_summary_30task.csv`
    - `existing_final_task_summary_30task.csv`
    - `existing_final_task_summary_35task.csv`
    - `chemical_holdout_single_seed_summary.csv`
    - `README.md`
    - `manifest.json`
  - `30task` 主表只含 ECx/LOEC/NOEC；`35task` 补充表保留 ICx/LDx。
- 已补齐本地 `outputs/experiments/v1_2_11_f100_seed_stability_remote_summary/focus_summary.csv` 的远端小 CSV 副本，使 `chemical_holdout_single_seed_summary.csv` 包含 42/1042/2042/3042/4042 五个 seed。
- 当前已有 final overall 指标：
  - fixed chemical-holdout f100 5-seed ensemble：n=2594，R2=0.5388，RMSE=1.2368，MAE=0.9277，Huber=0.5563。
  - random 8:2 3-seed ensemble：n=3165，R2=0.7775，RMSE=0.8676，MAE=0.6053，Huber=0.2955。
  - random 5-fold 3-seed ensemble：n=15630，R2=0.7833，RMSE=0.8601，MAE=0.6019，Huber=0.2909。
- 解释边界：
  - fixed chemical-holdout f100 5-seed ensemble 是当前论文主线外推结果。
  - random 8:2 / random 5-fold 仍是同分布插值参考，当前为 3-seed；后续按计划只需补跑 3042/4042 并刷新为 5-seed。
  - 后续消融和单域 B/C/E 基线将在该方法文档和指标包基础上继续追加，不覆盖当前结果。

## 2026-06-26 v1.2.16 随机划分 3-seed ensemble 扩展完成

- 目标：在 v1.2.15 随机 8:2 / 随机 5-fold split-policy pilot 基础上加入 seed ensemble，检验随机划分下 ensemble 是否继续带来稳定收益。
- 设计：
  - 复用 v1.2.15 正式 seed42 run。
  - 新增 seed：`1042`、`2042`。
  - ensemble 定义：同一 split、同一 test 样本在 3 个 seed 模型下分别预测，然后平均 `y_pred`；5-fold 情况是在每个 fold 内做 3-seed 平均，再把 5 个 fold 的 ensemble test 预测拼接汇总。
  - 仍只统计 ECx/LOEC/NOEC soil test focus 行，避免把 ICx/LDx 混入 v1.2.15 的正式比较口径。
- 新增脚本：
  - `scripts/summarize_split_policy_ensembles.py`：按 `sample_id/aggregate_id/split/task/y_true` 等 key 对齐多个 seed 的 `predictions.csv`，平均预测并输出 ensemble fold、combined 和 family summary。
  - `scripts/run_v1_2_15_random_split_policy_remote.sh` 支持 `SEEDS`、`ENSEMBLE_SEEDS` 和 `ensemble/ensemble_all` 模式。
- 远端运行：
  - 命令：`SEEDS="42 1042 2042" ENSEMBLE_SEEDS="42 1042 2042" bash scripts/run_v1_2_15_random_split_policy_remote.sh ensemble_all`。
  - 输出目录仍为：`outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary`。
  - 12 个新增训练均 `exit_code=0`；新增 seed1042/2042 后正式 run 总数为 18 个（随机 8:2 三个 seed + 5 folds × 三个 seed）。
- 3-seed ensemble test focus 结果：
  - 随机 8:2 ensemble：n=3165，MAE=0.6053，RMSE=0.8676，R2=0.7775，Huber=0.2955。
  - 随机 5-fold ensemble 合并：n=15630，MAE=0.6019，RMSE=0.8601，R2=0.7833，Huber=0.2909。
- 对照 v1.2.15 seed42 单模型：
  - 随机 8:2：MAE 0.6449 -> 0.6053，R2 0.7493 -> 0.7775。
  - 随机 5-fold 合并：MAE 0.6422 -> 0.6019，R2 0.7548 -> 0.7833。
  - 单模型 3-seed mean 也接近 seed42：随机 8:2 MAE=0.6427 +/- 0.0071；随机 5-fold run 级 MAE=0.6389 +/- 0.0123。ensemble 的改进主要来自预测平均，而不是新 seed 单模型本身显著更强。
- endpoint family（3-seed ensemble）：
  - 随机 5-fold：ECx MAE=0.5019/R2=0.8193；LOEC MAE=0.6358/R2=0.7710；NOEC MAE=0.6121/R2=0.7811。
  - 随机 8:2：ECx MAE=0.5007/R2=0.8125；LOEC MAE=0.6289/R2=0.7712；NOEC MAE=0.6287/R2=0.7693。
- 阶段性结论：
  - 随机划分下 ensemble 收益明确，约带来 0.04 log unit 的 MAE 改善和约 0.03 的 R2 提升。
  - 随机 8:2 与随机 5-fold 的 ensemble 结果仍非常接近，说明这两个随机划分策略给出的同分布性能估计基本一致。
  - 该结果应作为随机划分/插值场景下的 ensemble 上限参考；论文主线中关于外推能力的论证仍应以固定 chemical-holdout 和 CST-AD 分层为主。

## 2026-06-26 v1.2.15 随机 8:2 vs 随机 5-fold 划分策略试跑完成

- 目标：在模型与训练策略不变的前提下，只改变目标土壤域划分策略，比较随机 8:2 holdout 与随机 5-fold CV 对当前 f100 transfer anchor 指标估计的影响。
- 策略固定项：
  - source table：`aggregated_task_records_aquatic_soil_ptox_qc`。
  - 迁移结构：水相 pTox `train` + 土壤 pTox `finetune/test`。
  - anchor：`source_weighting=tanimoto_to_finetune alpha=1.0`、authority CE bin loss weight `0.025`、censored loss weight `0.01`、`finetune_validation_fraction=0.2`。
  - 单 seed：`seed=42`；这次是 split-policy pilot，不是多 seed 稳定性实验。
- 新增脚本：
  - `scripts/run_v1_2_15_random_split_policy_remote.sh`：从 `SoilPtoxQC2_B_random_8_2` 和 `SoilPtoxQC2_E_random_5fold_fold1-5` 派生对应 aquatic-to-soil adaptation split，并运行同一 anchor 策略。
  - `scripts/summarize_split_policy_pilot.py`：从正式 run 的 `predictions.csv` 重新计算 test focus 指标，输出 holdout/fold 与 policy 合并摘要。
- 远端正式输出：
  - 根目录：`outputs/experiments/v1_2_15_random_split_policy_formal_remote`。
  - 汇总目录：`outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary`。
  - 运行时间表：`outputs/logs/run_v1_2_15_random_split_policy_times.csv`。
  - 注意：早期入口验证误写入 `outputs/experiments/v1_2_15_random_split_policy_remote`，其预训练仅 1 epoch，不作为正式结果；正式结果只看 `*_formal_remote`。
- Split 构建核验：
  - `M_v2_aquatic_to_soil_ptox_adapt_B_random_8_2_f100`：train aquatic 281435，finetune soil 12816，test soil 3204。
  - 5 个 `M_v2_aquatic_to_soil_ptox_adapt_E_random_5fold_fold*_f100`：每折 train aquatic 281435，finetune soil 12816，test soil 3204。
- 正式 run 完整性：
  - 6 个训练均 `exit_code=0`，`required_files_present=True`。
  - `aquatic_eval_rows=0`，说明 test/finetune_validation 评估中没有水相样本混入。
  - 每个正式 run 均保留 `tanimoto_to_finetune alpha=1.0`、CE bin 与 censored loss 设置。
- test focus 合并结果（ECx/LOEC/NOEC soil）：
  - 随机 8:2 holdout：n=3165，MAE=0.6449，RMSE=0.9207，R2=0.7493，Huber=0.3261。
  - 随机 5-fold 合并：n=15630，MAE=0.6422，RMSE=0.9149，R2=0.7548，Huber=0.3214。
  - 5-fold 单折均值 +/- sd：MAE=0.6422 +/- 0.0149，RMSE=0.9148 +/- 0.0144，R2=0.7546 +/- 0.0070。
- endpoint family：
  - ECx：5-fold MAE=0.5214/R2=0.8025；8:2 MAE=0.5193/R2=0.7948。
  - LOEC：5-fold MAE=0.6824/R2=0.7422；8:2 MAE=0.6707/R2=0.7466。
  - NOEC：5-fold MAE=0.6555/R2=0.7483；8:2 MAE=0.6754/R2=0.7332。
- 阶段性结论：
  - 随机 8:2 与随机 5-fold 给出的性能估计非常接近；5-fold 合并略好，但幅度小，不应解释为模型策略本身改进。
  - 随机划分性能显著高于当前固定 chemical-holdout f100 主线，这是预期现象：随机划分允许相似化合物/物种上下文跨 train/test，更接近插值能力评估；固定 chemical-holdout 仍更适合支撑外推泛化和论文主结果。
  - 这次试跑可作为“随机划分下的上限/同分布性能参考”，不替代 v1.2.12 固定 chemical-holdout 5-seed ensemble 主结果。

## 2026-06-25 CST-AD 应用域初版方案与轻量出图

- 目标：在固定 test 和 v1.2.12 anchor 5-seed ensemble 主线不变的前提下，先构建 Chemical-Species-Task Applicability Domain (CST-AD) 的样本级矩阵和科研图表，不重新训练模型。
- 新增脚本：
  - `scripts/build_cst_ad_matrix.py`：读取 ensemble `*_ad_prediction_rows.csv`，生成 `cst_ad_prediction_rows.csv`、`cst_ad_tier_summary.csv`、`cst_ad_family_summary.csv`、`cst_ad_failure_cases.csv` 和 manifest。
  - `scripts/plot_cst_ad.py`：读取 CST-AD 矩阵，输出 chemical-species 聚合气泡图、AD tier 性能图、真实值-预测值图和 ensemble uncertainty-error 图。
- 新增文稿：`docs/cst_ad_application_domain_method.md`，记录 taxonomy embedding 取舍、CST-AD 四个维度、阈值、输出矩阵和论文表述草案。
- 当前轻量输出：
  - 输入：`outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/anchor_tanimoto_a1_5seed_ensemble_ad_prediction_rows.csv`。
  - 输出矩阵：`outputs/experiments/v1_2_14_cst_ad_initial`。
  - 输出图：`outputs/figures/cst_ad_20260625`。
  - 采用既有 AD audit 阈值：chemical Tanimoto `0.5`、taxonomy similarity `0.8`；ensemble SD 高不确定阈值为 P90。
- 图表修正：应用域主图应按 `task_family` 分面，不应把 ECx、LOEC、NOEC 全部混在一张 chemical-species space 图中。当前 `scripts/plot_cst_ad.py` 已输出 `cst_ad_chemical_species_space_by_task_family.*`，以及 ECx/LOEC/NOEC 三张单独图。
- 初步结果：AD-A 覆盖 972/2594=37.5%，MAE=0.8227；AD-B 覆盖 44.3%，MAE=0.9668；AD-C 覆盖 16.8%，MAE=1.0671；AD-D 覆盖 1.5%，MAE=0.8334。整体上 AD-A 误差更低，但仍存在少数高误差域内样本，后续需要结合 endpoint、censored/data-quality 和 train-reference 组合覆盖解释。
- 当前限制：本地 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 无 `split_assignments` 表，因此本地未补全 `cst_train_n_*` train-reference 组合计数；完整 Task AD 计数需在远端含 split assignments 的派生库上运行同一脚本并传入 `--db/--source-table/--split-name`。

## 2026-06-24 v1.2.13 anchor validation-policy 最小验证完成与结论

- 背景问题：当前 `finetune_validation` 是从 `finetune` 内部按 task head 分层随机抽出的 20%，同一 seed 下固定，换 seed 时随训练随机性一起轮换。v1.2.12 已说明 5-seed ensemble 明显优于单模型均值，但这仍是 ensemble 收益；下一步需要直接检验“释放这 20% finetune 样本、全量 finetune 后固定 test 是否更好”。
- 新增脚本：`scripts/run_v1_2_13_anchor_validation_policy_remote.sh`。
- 实验范围：
  - 只跑当前主线 anchor：`source_weighting=tanimoto_to_finetune alpha=1.0`、authority CE bin loss weight 0.025、censored loss weight 0.01、f100 split。
  - 新增组默认 `FINETUNE_VALIDATION_FRACTION=0.0`，`SEEDS="42 1042 2042 3042 4042"`；与 v1.2.12 既有 `finetune_validation_fraction=0.2` 5-seed anchor ensemble 作固定 test 对照。
  - 输出根目录：`outputs/experiments/v1_2_13_anchor_validation_policy_remote`；汇总目录：`outputs/experiments/v1_2_13_anchor_validation_policy_remote_summary`。
  - 汇总包括单模型 summary、AD gate、`seed_mean_ensemble`、以及 `validation_policy_comparison.csv`。
- 判定规则：
  - 若 val0 单模型 mean +/- sd 与 5-seed ensemble 同时优于 val0p2 control，说明内部验证留出确实带来数据效率损失，可把 val0 作为 final locked model 策略；但仍需说明没有 finetune early-stopping，模型选择依赖锁参后固定 epoch。
  - 若 val0 只在 ensemble 或少数 seed 上改善，报告为“可选部署/集成策略”，不作为单模型泛化增强证据。
  - 若 val0 变差或 AD/species extrapolation 变差，则保留 v1.2.12 anchor 5-seed ensemble 作为稳健主结果，不再继续扩展 finetune-validation policy。
  - 本阶段继续不启动 DANN/MMD/CORAL，也不再扩展 proxy alpha/source-rule 网格。
- 远端已启动：
  - 启动时间：2026-06-24 14:29:36 (+08:00)。
  - 日志：`outputs/logs/run_v1_2_13_anchor_validation_policy_20260624_142936.log`。
  - 当前进程：launch shell PID=`1304279`，脚本 PID=`1304282`，首个训练 PID=`1304292`。
  - 当前首个训练：`transfer_f100_anchor_tanimoto_a1_seed42_cebin_lw0025_censored_w0p01_val0`，split `M_v2_aquatic_to_soil_ptox_adapt_C_f100`。
  - 启动前远端 `bash -n scripts/run_v1_2_13_anchor_validation_policy_remote.sh` 通过，且无旧训练进程；本地 `git diff --check -- PROJECT_STATUS.md scripts/run_v1_2_13_anchor_validation_policy_remote.sh` 通过。
- 2026-06-24 15:10 阶段性核验：
  - `transfer_f100_anchor_tanimoto_a1_seed42_cebin_lw0025_censored_w0p01_val0` 已完成，运行时间 2026-06-24 14:29:36 到 15:05:52，duration=2176s，exit_code=0。
  - manifest 确认该 run 为全量 finetune policy：`finetune_rows=13526`，`finetune_train_rows=13526`，`finetune_validation_rows=0`，`finetune.early_stopping=False`，finetune 跑满 60 epoch，最终 best_epoch=90。
  - 固定 test focus（ECx/LOEC/NOEC soil）结果：n=2594，MAE=0.9867，RMSE=1.3181，R2=0.4762。
  - 对应 control `v1.2.12/v1.2.11 seed42 val0p2`：n=2594，MAE=0.9654，RMSE=1.2784，R2=0.5072；即首个 seed 上 val0 明显弱于保留 20% `finetune_validation` 的 policy。
  - family 层：val0 的 ECx MAE=0.8316/R2=0.5617，略优于 control 的 ECx MAE=0.8410，但 LOEC（MAE=1.0207/R2=0.4893）和 NOEC（MAE=1.0285/R2=0.4238）均弱于 control（LOEC MAE=0.9843/R2=0.5295，NOEC MAE=1.0071/R2=0.4490）。首个 seed 的负信号主要来自 LOEC/NOEC 退化。
  - 当前已自动进入第二个训练 `transfer_f100_anchor_tanimoto_a1_seed1042_cebin_lw0025_censored_w0p01_val0`。单 seed 不能终止矩阵，但目前不支持“释放 finetune_validation 会自动提高固定 test 泛化”的假设；需等 5-seed mean/ensemble 和 AD gate 后最终判定。
- 2026-06-24 15:51 阶段性核验：
  - `transfer_f100_anchor_tanimoto_a1_seed1042_cebin_lw0025_censored_w0p01_val0` 已完成，运行时间 2026-06-24 15:05:52 到 15:42:29，duration=2197s，exit_code=0；manifest 同样确认 `finetune_train_rows=13526`、`finetune_validation_rows=0`、finetune 跑满 60 epoch、最终 best_epoch=90。
  - `seed1042 val0` 固定 test focus：n=2594，MAE=1.0147，RMSE=1.3184，R2=0.4760；对应 `seed1042 val0p2 control`：MAE=1.0142，RMSE=1.3336，R2=0.4638。该 seed 上 val0 的 R2/RMSE 略好，但 MAE 基本持平略差。
  - 当前前 2 个 seed 临时均值：val0 MAE=1.0007，RMSE=1.3182，R2=0.4761；val0p2 control MAE=0.9898，RMSE=1.3060，R2=0.4855。整体仍偏向保留 20% `finetune_validation` 的 control。
  - 当前已自动进入第三个训练 `transfer_f100_anchor_tanimoto_a1_seed2042_cebin_lw0025_censored_w0p01_val0`，截至 2026-06-24 15:53 已到 pretrain epoch 3。继续等待 5-seed 完整结果、AD audit 与 seed-mean ensemble 后再做最终判定。
- 2026-06-24 16:30 阶段性核验：
  - `transfer_f100_anchor_tanimoto_a1_seed2042_cebin_lw0025_censored_w0p01_val0` 已完成，运行时间 2026-06-24 15:42:29 到 16:18:30，duration=2161s，exit_code=0；同样为全量 finetune policy，最终 best_epoch=90。
  - `seed2042 val0` 固定 test focus：n=2594，MAE=0.9914，RMSE=1.3105，R2=0.4822；对应 `seed2042 val0p2 control`：MAE=1.0008，RMSE=1.3190，R2=0.4755。该 seed 上 val0 略优于 control。
  - 当前前 3 个 seed 临时均值：val0 MAE=0.9976 +/- 0.0150，RMSE=1.3156，R2=0.4781；val0p2 control MAE=0.9935 +/- 0.0252，RMSE=1.3103，R2=0.4822。总体仍略偏向 control，但差距小于前 2 seed。
  - family 临时均值：val0 的 ECx MAE 较好（0.8420 vs control 0.8535）但 ECx R2 略弱；LOEC 明显弱于 control（MAE 1.0386 vs 1.0200，R2 0.4850 vs 0.4998）；NOEC 略优于 control（MAE 1.0329 vs 1.0353，R2 0.4276 vs 0.4191）。当前负向差距主要来自 LOEC。
  - 当前已自动进入第四个训练 `transfer_f100_anchor_tanimoto_a1_seed3042_cebin_lw0025_censored_w0p01_val0`，截至 2026-06-24 16:29 已进入 pretrain 初期。继续等待 5-seed 完整结果与 AD/ensemble 后最终判定。
- 2026-06-24 17:03 阶段性核验：
  - `transfer_f100_anchor_tanimoto_a1_seed3042_cebin_lw0025_censored_w0p01_val0` 已完成，运行时间 2026-06-24 16:18:30 到 16:52:24，duration=2034s，exit_code=0。
  - `seed3042 val0` 固定 test focus：n=2594，MAE=0.9978，RMSE=1.3061，R2=0.4857；对应 `seed3042 val0p2 control`：MAE=1.0034，RMSE=1.3302，R2=0.4665。该 seed 上 val0 明显优于 control。
  - 当前前 4 个 seed 临时均值：val0 MAE=0.9977 +/- 0.0122，RMSE=1.3132，R2=0.4800；val0p2 control MAE=0.9960 +/- 0.0212，RMSE=1.3153，R2=0.4783。val0 的 MAE 仍略差，但 RMSE/R2 已略优，说明结论从前 3 seed 的偏负转为“混合、不足以单独判胜”。
  - 当前已自动进入第五个训练 `transfer_f100_anchor_tanimoto_a1_seed4042_cebin_lw0025_censored_w0p01_val0`，截至 2026-06-24 17:02 已进入 pretrain epoch 2。最终判断必须等待 5-seed、AD audit 和 seed-mean ensemble。
- 2026-06-24 17:37 最终完成：
  - `transfer_f100_anchor_tanimoto_a1_seed4042_cebin_lw0025_censored_w0p01_val0` 已完成，运行时间 2026-06-24 16:52:24 到 17:29:17，duration=2212s，exit_code=0；5 个 val0 run 的 AD audit 均完成，summary 与 `validation_policy_comparison.csv` 已生成。
  - 本地已拉回 summary：`outputs/experiments/v1_2_13_anchor_validation_policy_remote_summary`。
  - val0 5-seed 单模型固定 test mean +/- sd：MAE=0.9926 +/- 0.0155，RMSE=1.3083 +/- 0.0122，R2=0.4839 +/- 0.0096。
  - 对照 val0p2 5-seed 单模型固定 test mean +/- sd：MAE=0.9883 +/- 0.0168，RMSE=1.3091 +/- 0.0204，R2=0.4875 +/- 0.0158。
  - val0 5-seed ensemble 固定 test：n=2594，MAE=0.9363，RMSE=1.2365，R2=0.5390；val0p2 control 5-seed ensemble：n=2594，MAE=0.9277，RMSE=1.2368，R2=0.5388。
  - AD gate（val0 ensemble）：overall_in_domain 覆盖 2371/2594=91.4%，MAE=0.9300，RMSE=1.2349，R2=0.5414；species_extrapolation_only n=223，MAE=1.0029，RMSE=1.2532，R2=0.0860。
  - endpoint-family（val0 ensemble）：ECx MAE=0.8049/R2=0.6148，LOEC MAE=0.9705/R2=0.5453，NOEC MAE=0.9665/R2=0.4983。
  - 最终判定：释放 `finetune_validation` 的 20% 样本进入训练后，R2/RMSE 与 val0p2 ensemble 几乎持平，但 MAE 与 Huber loss 略差；单模型均值也没有优于 val0p2。因此不把 val0 作为主模型策略。保留 v1.2.12 anchor 5-seed ensemble 作为当前稳健主结果；val0 可作为“全量微调/无 early-stopping”的敏感性检查，不继续扩展 validation-policy 矩阵。

## 2026-06-24 v1.2.12 f100 5-seed confirmation 完成与结论

- 目标：不扩大新方法矩阵，只把 v1.2.11 的 3-seed 结果扩展到 5 seed，验证 ensemble 收益和 AD 分层结论是否稳定。继续不启用 DANN/MMD/CORAL。
- 新增脚本：`scripts/run_v1_2_12_f100_5seed_confirmation_remote.sh`。
  - 默认只补 `EXTENSION_SEEDS="3042 4042"`。
  - 复用 `scripts/run_v1_2_11_f100_seed_stability_remote.sh` 跑新增 seed 的 anchor、`tanimoto_proxy alpha=0.5`、`proxy_distance alpha=0.5`。
  - 完成后调用 `scripts/summarize_seed_ensembles.py`，对 `ALL_SEEDS="42 1042 2042 3042 4042"` 生成 5-seed ensemble summary。
  - 5-seed ensemble 输出目录：`outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary`。
- 本地/远端启动前检查：
  - 本地 `E:\TOOLS\anaconda\python.exe -m py_compile scripts\summarize_seed_ensembles.py` 通过。
  - 本地 `git diff --check` 通过。
  - 远端 `bash -n scripts/run_v1_2_12_f100_5seed_confirmation_remote.sh` 通过。
  - 远端 `/opt/anaconda3/bin/python -m py_compile scripts/summarize_seed_ensembles.py` 通过。
  - 启动前远端无旧训练进程。
- 远端已启动：
  - 启动时间：2026-06-24 10:15:21 (+08:00)。
  - wrapper PID=`90434`；当前子脚本 PID=`90444`；当前训练 PID=`90451`。
  - 日志：`outputs/logs/run_v1_2_12_f100_5seed_confirmation_20260624_101407.log`。
  - 当前首个训练：`transfer_f100_anchor_tanimoto_a1_seed3042_cebin_lw0025_censored_w0p01`。
- 2026-06-24 10:55 阶段性核验：
  - `transfer_f100_anchor_tanimoto_a1_seed3042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 10:15:21 到 10:50:43，duration=2122s，exit_code=0。
  - 该 run 预训练在 epoch 28 早停，finetune 在 epoch 58 早停，最终 best epoch=76。
  - 该 run 全局 `finetune_validation`：n=2703，MAE=0.6309，RMSE=0.9074，R2=0.7645；固定 test：n=2730，MAE=0.9907，RMSE=1.3183，R2=0.4804。
  - 解释：新增 seed3042 的 anchor test 表现与 v1.2.11 的 3-seed anchor 单模型均值（MAE=0.9935，R2=0.4822）一致，目前没有推翻 3-seed 稳定性结论。
  - `transfer_f100_tanimoto_proxy_a0p5_seed3042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 10:50:43 到 11:22:08，duration=1885s，exit_code=0；预训练在 epoch 22 早停，finetune 在 epoch 59 早停，最终 best epoch=71。
  - 该 run 全局 `finetune_validation`：n=2703，MAE=0.6102，RMSE=0.8878，R2=0.7746；固定 test：n=2730，MAE=0.9611，RMSE=1.2681，R2=0.5192。
  - 同 seed 初步比较：`tanimoto_proxy alpha=0.5 seed3042` 明显优于 `anchor seed3042`（test MAE 低 0.0296，R2 高 0.0389），增强了 secondary candidate 信号；但最终仍需 5-seed ensemble 与 AD gate 确认其收益是否主要来自 in-domain/seen 层。
  - `transfer_f100_anchor_tanimoto_a1_seed4042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 11:22:08 到 11:50:59，duration=1731s，exit_code=0；预训练在 epoch 20 早停，finetune 跑满 60 epoch，最终 best epoch=74。
  - 该 run 全局 `finetune_validation`：n=2703，MAE=0.5972，RMSE=0.8829，R2=0.7742；固定 test：n=2730，MAE=0.9997，RMSE=1.3227，R2=0.4769。
  - 初步解释：新增 anchor seed3042/4042 的固定 test 都贴近 v1.2.11 的 anchor 单模型均值（MAE=0.9935，R2=0.4822），支持 anchor 单模型稳定区间大约在 MAE≈0.99、R2≈0.48，而不是 seed42 的单次强结果。
  - `transfer_f100_tanimoto_proxy_a0p5_seed4042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 11:50:59 到 12:20:27，duration=1768s，exit_code=0；预训练在 epoch 20 早停，finetune 跑满 60 epoch，最终 best epoch=74。
  - 该 run 全局 `finetune_validation`：n=2703，MAE=0.5959，RMSE=0.8667，R2=0.7824；固定 test：n=2730，MAE=1.0319，RMSE=1.3493，R2=0.4557。
  - 同 seed 初步比较：`tanimoto_proxy alpha=0.5 seed4042` 的 `finetune_validation` 略好于 anchor，但固定 test 明显弱于 anchor seed4042（test MAE 高 0.0322，R2 低 0.0212），说明 proxy 的单模型收益仍存在 seed 依赖，不能仅凭 seed3042 判定胜出。
  - 当前 5-seed 单模型 test 暂定均值（anchor/tanimoto_proxy 已完成，proxydist 新增 seed 待跑）：
    - anchor：MAE=0.9883 +/- 0.0168，RMSE=1.3091 +/- 0.0204，R2=0.4875 +/- 0.0158。
    - `tanimoto_proxy alpha=0.5`：MAE=0.9878 +/- 0.0294，RMSE=1.3035 +/- 0.0323，R2=0.4917 +/- 0.0253。
  - 临时解释：5-seed 单模型层面二者几乎打平，proxy 的均值只略优但方差更大；仍不应宣称明确主胜出。最终判断需等 seed-mean ensemble 与 AD gate，尤其看收益是否集中在 in-domain/species-task-family-seen。
  - 4 个新增 anchor/tanimoto_proxy run 的 AD audit 已全部完成，均 exit_code=0：
    - `anchor seed3042`：2026-06-24 12:20:27 到 12:22:01，duration=94s。
    - `tanimoto_proxy seed3042`：2026-06-24 12:22:01 到 12:23:35，duration=94s。
    - `anchor seed4042`：2026-06-24 12:23:35 到 12:25:07，duration=92s。
    - `tanimoto_proxy seed4042`：2026-06-24 12:25:07 到 12:26:41，duration=94s。
  - v1.2.12 已进入第二段 `proxy_distance alpha=0.5` 扩展；由于 anchor seed3042 已存在，脚本跳过 anchor 并启动 `transfer_f100_proxydist_a0p5_seed3042_cebin_lw0025_censored_w0p01`。
  - `transfer_f100_proxydist_a0p5_seed3042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 12:27:59 到 13:03:52，duration=2153s，exit_code=0；预训练在 epoch 28 早停，finetune 跑满 60 epoch，最终 best epoch=80。
  - 该 run 全局 `finetune_validation`：n=2703，MAE=0.6402，RMSE=0.9280，R2=0.7537；固定 test：n=2730，MAE=1.0278，RMSE=1.3393，R2=0.4637。
  - 当前 proxydist 已完成 4 个 seed（42/1042/2042/3042）的 test 暂定均值：MAE=0.9961 +/- 0.0213，RMSE=1.3114 +/- 0.0206，R2=0.4857 +/- 0.0163。新增 seed3042 明显弱于 proxydist 前 3 个 seed，整体上不支持其作为 overall 主模型。
  - 当前正在运行 `transfer_f100_proxydist_a0p5_seed4042_cebin_lw0025_censored_w0p01`，截至 2026-06-24 13:29 已进入 finetune epoch 30。5-seed summary 目录尚未产出文件，需等待 seed4042 训练、proxydist 新增 audit 与最终 summary 完成。
- 2026-06-24 13:37 最终完成：
  - `transfer_f100_proxydist_a0p5_seed4042_cebin_lw0025_censored_w0p01` 已完成，运行时间 2026-06-24 13:03:52 到 13:32:33，duration=1721s，exit_code=0；最终 best epoch=79。
  - 该 run 固定 test：n=2730，MAE=1.0140，RMSE=1.3266，R2=0.4738。
  - proxydist seed3042/4042 的 AD audit 均完成，duration=93s，exit_code=0。
  - v1.2.12 全部新增训练、AD audit 和 5-seed ensemble summary 已完成；最终日志：`outputs/logs/run_v1_2_12_f100_5seed_confirmation_20260624_101407.log`。
  - 本地已拉回 summary：`outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary`。
- 5-seed 单模型固定 test 均值（mean +/- sd）：
  - anchor `tanimoto_to_finetune alpha=1.0`：MAE=0.9883 +/- 0.0168，RMSE=1.3091 +/- 0.0204，R2=0.4875 +/- 0.0158。
  - `tanimoto_proxy_to_finetune alpha=0.5`：MAE=0.9878 +/- 0.0294，RMSE=1.3035 +/- 0.0323，R2=0.4917 +/- 0.0253。
  - `proxy_distance_to_finetune alpha=0.5`：MAE=0.9997 +/- 0.0201，RMSE=1.3144 +/- 0.0191，R2=0.4834 +/- 0.0151。
- 5-seed ensemble 固定 test：
  - anchor ensemble：n=2594，MAE=0.9277，RMSE=1.2368，R2=0.5388。
  - `tanimoto_proxy alpha=0.5` ensemble：n=2594，MAE=0.9345，RMSE=1.2388，R2=0.5373。
  - `proxy_distance alpha=0.5` ensemble：n=2594，MAE=0.9427，RMSE=1.2434，R2=0.5338。
- AD gate 结论：
  - anchor ensemble 在 all-test、overall-in-domain MAE、species-task-family-seen、species-extrapolation-only 上整体最稳；species-extrapolation-only 为 MAE=0.9854，RMSE=1.2268，R2=0.5312。
  - `tanimoto_proxy alpha=0.5` 仅在 overall-in-domain R2 和 species-task-family-unseen 层略有优势，但 all-test 与 species-extrapolation-only 均不如 anchor。
  - `proxy_distance alpha=0.5` 不再支持作为 species extrapolation 主候选；5-seed ensemble 下它的 species-extrapolation-only MAE=1.0156，弱于 anchor。
- endpoint-family 结论：
  - anchor ensemble 明显最优于 ECx：MAE=0.7918，RMSE=1.0487，R2=0.6295。
  - `tanimoto_proxy alpha=0.5` 对 NOEC 更友好：MAE=0.9462，RMSE=1.2678，R2=0.5013；LOEC 的 R2/RMSE 略优但 MAE 略弱。
  - 因此 proxy 的价值更像 endpoint-family 诊断，不是 overall 主胜出。
- 最终判定：
  - 当前主线锁定为 f100 censored anchor；若允许 ensemble，报告 anchor 5-seed ensemble 作为稳健性增强后的最佳结果。
  - 单模型结论仍需报告 5-seed mean +/- sd，不能把 ensemble 性能等同于单模型性能。
  - 停止 proxy alpha/source-rule 扩展；`tanimoto_proxy alpha=0.5` 只作为 secondary endpoint/in-domain 诊断，`proxy_distance alpha=0.5` 不作为主模型。
  - 不启动 DANN/MMD/CORAL；下一阶段优先做 anchor validation-policy/final-model 检验，而不是新增迁移机制。

## 2026-06-24 v1.2.10 proxy alpha 敏感性完成与 v1.2.11 seed 稳定性启动

- 背景问题：当前 `finetune_validation` 是从 `finetune` 内部按 task head 分层随机抽出的 20%，由固定 `project.seed=42` 控制；同一数据、同一 split、同一 seed 下是固定样本集。它用于微调阶段 validation loss、学习率调度和 early stopping，因此只能作为开发集/模型选择信号，最终泛化仍以独立 `test` 和 AD gate 为准。
- 最小代码补充：
  - `qsar_tl/training/train.py` 新增 `--seed` 覆盖参数；默认仍读取 `project.seed`，所以既有脚本默认行为不变。
  - 新增 `scripts/run_v1_2_11_f100_seed_stability_remote.sh`，用于 f100 anchor 与可配置 challenger 的多 seed 稳定性验证，自动输出 summary 与 AD gate。
  - 本地验证：`E:\TOOLS\anaconda\python.exe -m py_compile qsar_tl\training\train.py` 通过；`E:\TOOLS\anaconda\python.exe -m pytest tests\test_deep_experiment_cache.py -q` 为 `34 passed`；`git diff --check` 通过。远端 `train.py` 编译、`--help` 检查和 `bash -n scripts/run_v1_2_11_f100_seed_stability_remote.sh` 通过。
- v1.2.10 f100 proxy alpha 敏感性已完成：
  - 远端时间：2026-06-23 21:46:23 到 2026-06-24 00:18:11 (+08:00)。
  - 输出根目录：`outputs/experiments/v1_2_10_f100_proxy_alpha_remote`；汇总目录：`outputs/experiments/v1_2_10_f100_proxy_alpha_remote_summary`。
  - 4 个训练 + 4 个 AD audit 均 `exit_code=0`。
  - test 指标：
    - `transfer_f100_proxydist_a0p25_cebin_lw0025_censored_w0p01`：n=2,594，MAE=0.9930，RMSE=1.3133，R2=0.4800。
    - `transfer_f100_tanimoto_proxy_a0p25_cebin_lw0025_censored_w0p01`：n=2,594，MAE=1.0060，RMSE=1.3237，R2=0.4717。
    - `transfer_f100_proxydist_a0p75_cebin_lw0025_censored_w0p01`：n=2,594，MAE=1.0083，RMSE=1.3273，R2=0.4688。
    - `transfer_f100_tanimoto_proxy_a0p75_cebin_lw0025_censored_w0p01`：n=2,594，MAE=0.9934，RMSE=1.3158，R2=0.4780。
  - family 层诊断：a0.75 pure proxy 的 ECx R2=0.5841 但 NOEC R2=0.4052，说明更强 proxy 权重仍偏移 endpoint-family 平衡；a0.75 tanimoto+proxy 的 LOEC 相对较好（MAE=1.0224/R2=0.4999），但 ECx/NOEC 不足，总体 test 仍弱。
  - AD gate 诊断：
    - a0.25 pure proxy：species-extrapolation-only n=223，MAE=1.0620，R2=0.4411；species-task-family-seen n=1,121，MAE=0.9215，R2=0.5543。
    - a0.75 tanimoto+proxy：overall in-domain n=2,371，MAE=0.9828，R2=0.4844；species-extrapolation-only n=223，MAE=1.1058，R2=0.4044；species-task-family-seen n=1,121，MAE=0.9535，R2=0.5360。
  - 结论：v1.2.10 的 alpha=0.25/0.75 均弱于 v1.2.8 f100 censored anchor（MAE=0.9692/R2=0.5072），也弱于 v1.2.9 alpha=0.5 的两个 f100 proxy 候选。因此不继续扩展 proxy alpha 网格，也不启动 DANN/MMD/CORAL。
- v1.2.11 f100 seed 稳定性已完成：
  - 运行时间：2026-06-24 00:20:35 到 2026-06-24 05:41:48 (+08:00)。
  - 日志：`outputs/logs/run_v1_2_11_f100_seed_stability_matrix_20260624_002035.log`。
  - 输出根目录：`outputs/experiments/v1_2_11_f100_seed_stability_remote`；汇总目录：`outputs/experiments/v1_2_11_f100_seed_stability_remote_summary`。
  - 完成范围：`SEEDS="42 1042 2042"`；f100 anchor `tanimoto_to_finetune alpha=1.0`、`tanimoto_proxy_to_finetune alpha=0.5`、`proxy_distance_to_finetune alpha=0.5`，共 9 个训练 + 9 个 AD audit，均 `exit_code=0`。
  - 单模型 test 均值（mean +/- sd）：
    - anchor `tanimoto_to_finetune alpha=1.0`：MAE=0.9935 +/- 0.0252，RMSE=1.3103 +/- 0.0286，R2=0.4822 +/- 0.0225。
    - `tanimoto_proxy_to_finetune alpha=0.5`：MAE=0.9915 +/- 0.0188，RMSE=1.3093 +/- 0.0193，R2=0.4831 +/- 0.0152。
    - `proxy_distance_to_finetune alpha=0.5`：MAE=0.9979 +/- 0.0023，RMSE=1.3141 +/- 0.0129，R2=0.4793 +/- 0.0103。
  - 单模型结论：seed=42 的 anchor 强结果（MAE=0.9654/R2=0.5072）不是跨 seed 完全稳定的固定结论；改 seed 后 anchor 会退到 MAE=1.0142 或 1.0008。`tanimoto_proxy alpha=0.5` 单模型均值只比 anchor 小幅好 0.0020 log units，R2 只高 0.0009，科学意义很小，不能作为明确优胜证据。`proxy_distance alpha=0.5` 方差最小，但均值弱于 anchor 和 tanimoto proxy。
  - AD gate 均值诊断：
    - anchor：all test MAE=0.9935/R2=0.4822；overall in-domain MAE=0.9857/R2=0.4865；species-task-family seen MAE=0.9187/R2=0.5562；species-extrapolation-only MAE=1.0766/R2=0.4317。
    - `tanimoto_proxy alpha=0.5`：all test MAE=0.9915/R2=0.4831；overall in-domain MAE=0.9813/R2=0.4891；species-task-family seen MAE=0.9154/R2=0.5624；species-extrapolation-only MAE=1.0992/R2=0.4138。
    - `proxy_distance alpha=0.5`：all test MAE=0.9979/R2=0.4793；overall in-domain MAE=0.9899/R2=0.4826；species-task-family seen MAE=0.9174/R2=0.5610；species-extrapolation-only MAE=1.0834/R2=0.4404。
  - AD 解释：`tanimoto_proxy alpha=0.5` 主要改善 in-domain 与 species-task-family seen 场景，但 species extrapolation 变差，不能作为提高跨物种生态风险外推能力的证据；`proxy_distance alpha=0.5` 在 species-extrapolation R2 上略高，但 MAE 并未优于 anchor，仍不是通用优胜。
  - 已完成轻量 ensemble 检验：用每组 3 个 seed 的 test 逐行平均预测，不重新训练。输出：`seed_mean_ensemble_focus_summary.csv`、`seed_mean_ensemble_family_summary.csv`、`seed_mean_ensemble_ad_gate_summary.csv`。
  - 可复现脚本：新增 `scripts/summarize_seed_ensembles.py`，显式传入 `--group output_name=run_a,run_b,run_c`，读取 `audits/*/ad_prediction_rows.csv`，先校验同一 split 的测试行 key 完全一致，再输出 focus、family 与 AD gate summary；`--write-prediction-rows` 可同时保存 ensemble 后的 AD prediction rows。
  - 脚本验证：本地 `E:\TOOLS\anaconda\python.exe -m pytest tests\test_summarize_seed_ensembles.py tests\test_summarize_deep_runs.py tests\test_deep_experiment_cache.py -q` 为 `38 passed`（仅保留既有本地 Torch/NumPy warning）；远端 `/opt/anaconda3/bin/python -m pytest tests/test_summarize_seed_ensembles.py -q` 为 `2 passed`；远端真实 v1.2.11 audit 目录复跑脚本成功，并已拉回 summary。
  - seed-mean ensemble test：
    - anchor ensemble：MAE=0.9408，RMSE=1.2473，R2=0.5310。
    - `tanimoto_proxy alpha=0.5` ensemble：MAE=0.9377，RMSE=1.2479，R2=0.5305。
    - `proxy_distance alpha=0.5` ensemble：MAE=0.9438，RMSE=1.2517，R2=0.5276。
  - ensemble 解释：跨 seed 平均预测确实显著优于任何单 seed 均值，说明随机初始化/`finetune_validation` 抽样带来的误差方向可被模型平均抵消；但这是 ensemble 收益，不等于单模型最终表现自动提高。若论文或后续报告允许 ensemble，可作为稳健性增强方案；若坚持单模型部署，则仍应报告跨 seed mean +/- sd。
  - 当前决策：不扩展 proxy alpha，不启动 DANN/MMD/CORAL。主线仍保留 f100 censored anchor；`tanimoto_proxy alpha=0.5` 可作为 in-domain 诊断/secondary candidate，`proxy_distance alpha=0.5` 可作为稳定性和 species extrapolation 诊断，不作为主优胜模型。

## 2026-06-23 v1.2.8 censored loss 递进矩阵

- 目标：按递进式计划进入后续阶段 A/B；优先验证 censored loss 与 AD gate，不启用 DANN/MMD/CORAL。
- 代码范围：
  - `qsar_tl/training/censored_loss.py`：`censored_hinge_loss` 支持 string/tensor direction，适配 batch loss。
  - `qsar_tl/training/deep_train.py` 与 `qsar_tl/training/deep_experiment.py`：新增默认关闭的 `training.censored_loss`，精确样本仍走 Huber；删失样本不参与 Huber，只按同 task head 的预测值走 one-sided hinge。
  - `qsar_tl/training/train.py`：新增 CLI `--censored-loss`、`--censored-loss-weight`、`--censored-loss-margin`。
  - `scripts/summarize_ad_gate.py`：读取 AD prediction rows，输出 all / in-domain / not species extrapolation / species-task-family seen 等 confidence tier 指标。
  - `scripts/run_v1_2_8_censored_loss_matrix_remote.sh`：远端 smoke/matrix/summarize 脚本，输出根目录 `outputs/experiments/v1_2_8_censored_loss_matrix_remote`，汇总目录 `outputs/experiments/v1_2_8_censored_loss_matrix_remote_summary`。
- 本地验证：
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_censored_loss.py tests\test_modeling_shapes.py tests\test_deep_experiment_cache.py tests\test_toxicity_binning.py tests\test_ordinal_binning.py tests\test_summarize_deep_runs.py -q`，`48 passed`。
  - `git diff --check` 通过。
- 远端验证：
  - `bash -n scripts/run_v1_2_8_censored_loss_matrix_remote.sh` 通过。
  - `/opt/anaconda3/bin/python -m py_compile ... scripts/summarize_ad_gate.py` 通过。
  - `/opt/anaconda3/bin/python -m pytest tests/test_censored_loss.py tests/test_modeling_shapes.py tests/test_deep_experiment_cache.py tests/test_toxicity_binning.py tests/test_ordinal_binning.py -q`，`46 passed`。
- 阶段 B AD gate summary 已生成：
  - 输出：`outputs/experiments/v1_2_8_censored_loss_matrix_remote_summary/v1_2_7_ce_best_ad_gate_summary.csv`。
  - f100 CE 最佳：all test n=2,594，MAE=0.9768/R2=0.4978；overall in-domain n=2,371，MAE=0.9579/R2=0.5130；species-extrapolation-only n=223，MAE=1.1783/R2=0.3286；species-task-family-seen n=1,121，MAE=0.9018/R2=0.5726。
  - f20 CE 最佳：all test n=2,557，MAE=1.1478/R2=0.2978；overall in-domain n=2,335，MAE=1.1239/R2=0.3203；species-extrapolation-only n=222，MAE=1.3996/R2=0.0554；species-task-family-seen n=1,093，MAE=1.0154/R2=0.4445。
- v1.2.8 smoke 已完成：
  - 运行：`smoke_transfer_f20_cebin_lw005_censored_w003`，split `M_v2_aquatic_to_soil_ptox_adapt_C_f20`，1 epoch pretrain + 1 epoch finetune。
  - 结束时间：2026-06-23 15:17:29 +08:00。
  - manifest 中 `censored_loss.candidate_rows=56,345`，`usable_rows=937`，`train_rows=937`，`finetune_rows=0`；跳过原因主要是 `missing_or_unsupported_bound=37,654`、`task_not_trained=7,333`、`unmapped_task=6,205`、`unmatched_split=4,216`。
  - history 显示 pretrain 阶段 `censored_samples=937`、`mean_censored_loss=0.4605`、`censored_loss_weight=0.03`；finetune/validation 阶段 `censored_samples=0`，确认删失样本没有污染微调验证和 test。
- 正式 v1.2.8 matrix 已启动：
  - 启动时间：2026-06-23 15:18:05 +08:00。
  - PID：`3001550`。
  - 日志：`outputs/logs/run_v1_2_8_censored_loss_matrix_20260623_151805.log`。
  - 计时表：`outputs/logs/run_v1_2_8_censored_loss_matrix_times.csv`。
  - 6 个 run：f20 CE authority-bin + censored weight `0.01/0.03/0.10`，f100 CE authority-bin + censored weight `0.01/0.03/0.10`。
  - `transfer_f20_cebin_lw005_censored_w0p01` 已完成：duration=1,523s，best_epoch=46，pretrain early-stop epoch=23，finetune early-stop epoch=33；`censored_loss.usable_rows=937`，全部进入 train/pretrain，`finetune_rows=0`。
  - f20/w0.01 focus test：n=2,557，MAE=1.1835，RMSE=1.5289，R2=0.2782；弱于 v1.2.6 f20 CE 基线 MAE=1.1478/R2=0.2978，因此第一个 censored-loss 点未通过继续扩大门槛。
  - `transfer_f100_cebin_lw0025_censored_w0p01` 已完成：duration=2,246s，best_epoch=89，pretrain 跑满 30 epoch，finetune 跑满 60 epoch；`censored_loss.usable_rows=984`，全部进入 train/pretrain，`finetune_rows=0`。
  - f100/w0.01 focus test：n=2,594，MAE=0.9692，RMSE=1.2785，R2=0.5072；优于 v1.2.6 f100 CE 基线 MAE=0.9768/R2=0.4978，初步通过 f100 继续门槛。family test：ECx MAE=0.9144/R2=0.5204，LOEC MAE=0.9826/R2=0.5416，NOEC MAE=0.9827/R2=0.4639。
  - `transfer_f20_cebin_lw005_censored_w0p03` 已完成：duration=1,855s，best_epoch=73，pretrain 跑满 30 epoch，finetune early-stop epoch=53；`censored_loss.usable_rows=937`，全部进入 train/pretrain，`finetune_rows=0`。
  - f20/w0.03 focus test：n=2,557，MAE=1.1412，RMSE=1.5117，R2=0.2944；MAE 优于 v1.2.6 f20 CE 基线 1.1478，R2 略低于基线 0.2978 但接近，说明 f20 分支存在权重敏感的正收益。family test：ECx MAE=0.9750/R2=0.4591，LOEC MAE=1.1576/R2=0.3158，NOEC MAE=1.2062/R2=0.1952。
  - `transfer_f100_cebin_lw0025_censored_w0p03` 已完成：duration=2,229s，best_epoch=89，pretrain/finetune 均跑满；`censored_loss.usable_rows=984`，全部进入 train/pretrain，`finetune_rows=0`。
  - f100/w0.03 focus test：n=2,594，MAE=0.9837，RMSE=1.2972，R2=0.4927；弱于 f100/w0.01 与 v1.2.6 f100 CE 基线。family test 显示 ECx 明显改善（MAE=0.7858/R2=0.6240），但 LOEC/NOEC 退化（MAE=1.0443/1.0203），提示更强 censored loss 会偏移 endpoint-family 平衡。
  - `transfer_f20_cebin_lw005_censored_w0p10` 已完成：duration=1,885s，best_epoch=73，pretrain 跑满 30 epoch，finetune early-stop epoch=53；focus test n=2,557，MAE=1.1612，RMSE=1.5356，R2=0.2719，弱于 f20/w0.03（MAE=1.1412/R2=0.2944），因此 f20 的 v1.2.9 默认锚点仍保留 censored weight `0.03`。
  - `transfer_f100_cebin_lw0025_censored_w0p10` 已完成：duration=2,229s，best_epoch=90，pretrain/finetune 跑满；focus test n=2,594，MAE=1.0007，RMSE=1.3193，R2=0.4752，弱于 f100/w0.01 与 f100/w0.03。
- v1.2.8 完整结论：
  - f20 最佳：`transfer_f20_cebin_lw005_censored_w0p03`，test MAE=1.1412、RMSE=1.5117、R2=0.2944；相对 v1.2.6 f20 CE 基线 MAE=1.1478/R2=0.2978，是“MAE 小幅改善、R2 近似持平略低”的可用但不强的改进。
  - f100 最佳：`transfer_f100_cebin_lw0025_censored_w0p01`，test MAE=0.9692、RMSE=1.2785、R2=0.5072；相对 v1.2.6 f100 CE 基线 MAE=0.9768/R2=0.4978，MAE 与 R2 均改善，是当前最稳的 censored-loss 正向证据。
  - censored loss 存在明显权重敏感性：f20 的 0.10 退化；f100 的 0.03/0.10 虽提升 ECx family，但 LOEC/NOEC 退化导致总体下降。解释上更像“低权重可作为删失约束正则项”，而不是可大权重替代精确毒性监督。
  - 最终锚点：v1.2.9 proxy/source-rule 阶段默认使用 f20 `censored_weight=0.03`、toxicity bin CE loss=0.05；f100 `censored_weight=0.01`、toxicity bin CE loss=0.025。
- v1.2.8 best censored candidates 的化合物+物种 AD gate 已补跑：
  - 输出：`outputs/experiments/v1_2_8_censored_loss_matrix_remote_summary/censored_best_ad_gate_summary.csv`。
  - f100/w0.01：all test MAE=0.9692/R2=0.5072；overall in-domain MAE=0.9644/R2=0.5058；species-task-family-seen MAE=0.9047/R2=0.5739；species-extrapolation-only MAE=1.0207/R2=0.5195。相对旧 CE 基线，外推层显著改善，说明低权重 censored loss 对 f100 的收益不是只来自易样本。
  - f20/w0.03：all test MAE=1.1412/R2=0.2944；overall in-domain MAE=1.1128/R2=0.3163；species-task-family-seen MAE=0.9885/R2=0.4645；species-extrapolation-only MAE=1.4395/R2=0.0588。域内/seen 层略有改善，但物种外推仍是 f20 的主要风险。
- v1.2.9 proxy/source-rule 阶段已启动：
  - 训练侧最小扩展：`source_weighting_method` 新增 `proxy_distance_to_finetune` 与 `tanimoto_proxy_to_finetune`；使用 `molecular_numeric` 中 MolWt/TPSA/MolLogP 三个 proxy 维度计算到 soil finetune 的最近距离，权重继续做均值归一化与 min/max clipping。
  - 汇总侧：`scripts/summarize_deep_runs.py` 的 `audit_summary.csv` 新增 `source_weighting_method`、`source_weighting_alpha`、`source_proxy_distance_mean`，便于比较 proxy/source-rule 矩阵。
  - 远端脚本：`scripts/run_v1_2_9_proxy_source_rule_remote.sh`，默认 4 个正式候选：f20/f100 × `proxy_distance_to_finetune`/`tanimoto_proxy_to_finetune`，默认锚点 `F20_CENSORED_WEIGHT=0.03`、`F100_CENSORED_WEIGHT=0.01`、`PROXY_ALPHA=0.5`；如果 v1.2.8 的 w0.10 胜出，启动前用环境变量覆盖。
  - 本地验证：`E:\TOOLS\anaconda\python.exe -m pytest tests/test_deep_experiment_cache.py tests/test_modeling_shapes.py tests/test_censored_loss.py tests/test_ordinal_binning.py -q` 为 `43 passed`；`tests/test_summarize_deep_runs.py -q` 为 `2 passed`；`py_compile` 和 `git diff --check` 通过。
  - 远端验证：`bash -n scripts/run_v1_2_9_proxy_source_rule_remote.sh` 通过；`/opt/anaconda3/bin/python -m py_compile qsar_tl/training/deep_experiment.py qsar_tl/training/train.py scripts/summarize_deep_runs.py scripts/audit_prediction_application_domain.py` 通过；`/opt/anaconda3/bin/python -m pytest tests/test_deep_experiment_cache.py tests/test_summarize_deep_runs.py -q` 为 `34 passed`。
  - smoke 已完成：`smoke_transfer_f20_proxydist_a0p5_cebin_lw005_censored_w0p03`，1 epoch pretrain + 1 epoch finetune，duration=534s，best_epoch=2；manifest 确认 `source_weighting_method=proxy_distance_to_finetune`、`source_weighting_alpha=0.5`、`source_proxy_distance_mean=0.2099`。
  - 正式 matrix 已启动：2026-06-23 19:26:48 +08:00，PID=`3694`，日志 `outputs/logs/run_v1_2_9_proxy_source_rule_matrix_20260623_192648.log`。
  - 正式任务顺序：`transfer_f20_proxydist_a0p5_cebin_lw005_censored_w0p03`、`transfer_f100_proxydist_a0p5_cebin_lw0025_censored_w0p01`、`transfer_f20_tanimoto_proxy_a0p5_cebin_lw005_censored_w0p03`、`transfer_f100_tanimoto_proxy_a0p5_cebin_lw0025_censored_w0p01`，随后自动跑 4 个 AD audit 并生成 `ad_gate_summary.csv`。
  - 运行中优化：首个正式任务启动后，已将 `min_proxy_distance` 从逐样本循环改为批量矩阵乘法，结果与直接欧氏最近距离一致；本地 `tests/test_deep_experiment_cache.py -q` 为 `33 passed`，远端同步后同一测试为 `33 passed`。当前首个 f20 proxy-distance 进程已加载旧实现但已顺利进入 epoch；后续 f100/proxy 与 tanimoto+proxy 任务会使用优化后的批量实现。
- `transfer_f20_proxydist_a0p5_cebin_lw005_censored_w0p03` 已完成：duration=1,562s，best_epoch=62；test n=2,557，MAE=1.1907，RMSE=1.5628，R2=0.2459，弱于 v1.2.8 f20 censored 锚点 MAE=1.1412/R2=0.2944。初步判断：单独用 proxy-distance 替代 Tanimoto 源域权重会损失结构相似度信息，不适合作为 f20 主线。
- `transfer_f100_proxydist_a0p5_cebin_lw0025_censored_w0p01` 已完成：duration=2,227s，best_epoch=84；test n=2,594，MAE=0.9630，RMSE=1.2829，R2=0.5038。相对 v1.2.8 f100 censored 锚点 MAE=0.9692/R2=0.5072，是“MAE 小幅改善、R2 小幅下降”；family test 中 ECx 明显改善（MAE=0.8439/R2=0.5689），LOEC 基本持平略好（MAE=0.9795/R2=0.5301），NOEC 略退化（MAE=1.0045/R2=0.4460）。说明 pure proxy-distance 对 f100 有可用信号，但仍需看 `tanimoto_proxy_to_finetune` 是否能兼顾结构相似性与 proxy 环境行为相似性。
- `max_tanimoto_similarity` 已补充重复指纹去重优化：对 source/target fingerprint 先 unique，再将最大 Tanimoto 映射回原 source 行；测试确认重复 source/target 时分数与直接逐行计算一致。本地与远端 `tests/test_deep_experiment_cache.py -q` 均为 `34 passed`。当前 f20 tanimoto+proxy 进程启动早于该同步，仍使用旧实现；后续 f100 tanimoto+proxy 会加载优化后的实现。
- `transfer_f20_tanimoto_proxy_a0p5_cebin_lw005_censored_w0p03` 已完成：duration=1,869s，best_epoch=61；test n=2,557，MAE=1.1501，RMSE=1.5334，R2=0.2740。相对 f20 pure proxy MAE=1.1907/R2=0.2459 明显恢复，说明加入 Tanimoto 后结构相似度信号有效；但仍弱于 v1.2.8 f20 censored 锚点 MAE=1.1412/R2=0.2944，也未超过旧 CE authority-bin f20 的 MAE=1.1478/R2=0.2978。family test：ECx MAE=0.9922/R2=0.4269，LOEC MAE=1.1759/R2=0.2723，NOEC MAE=1.2019/R2=0.2051；NOEC 比 v1.2.8 f20/w0.03 略好，但 ECx/LOEC 与总 R2 不足。当前结论：f20 不支持继续扩大 proxy 权重，只可作为组合权重优于 pure proxy 的证据。
- `transfer_f100_tanimoto_proxy_a0p5_cebin_lw0025_censored_w0p01` 已完成：duration=1,968s，best_epoch=84；test n=2,594，MAE=0.9710，RMSE=1.2748，R2=0.5100。相对 v1.2.8 f100 censored 锚点 MAE=0.9692/R2=0.5072，是“R2 小幅提升、MAE 小幅退化”；相对 f100 pure proxy MAE=0.9630/R2=0.5038，是“结构+proxy 组合改善 RMSE/R2，但不改善 MAE”。family test：ECx MAE=0.8777/R2=0.5428，LOEC MAE=0.9618/R2=0.5548，NOEC MAE=1.0249/R2=0.4468；说明组合权重主要改善 LOEC 与总体方差解释，ECx 不如 pure proxy，NOEC 仍有退化。
- v1.2.9 已完整完成：4 个正式训练候选 + 4 个 AD audit 均 exit_code=0；正式汇总目录 `outputs/experiments/v1_2_9_proxy_source_rule_remote_summary`，包含 `focus_summary.csv`、`family_summary.csv`、`common_task_comparison.csv`、`audit_summary.csv`、`ad_gate_summary.csv`。
- v1.2.9 AD gate 结论：
  - f20 两个 proxy 方案均未超过 v1.2.8 f20 censored 锚点，不继续扩大 f20 proxy。`tanimoto_proxy` 比 pure proxy 明显恢复：all test MAE/R2 从 1.1907/0.2459 改为 1.1501/0.2740；species-task-family seen 层 MAE/R2 为 0.9957/0.4619，但 species-extrapolation-only 仍差，MAE=1.4628、R2=-0.0128。
  - f100 pure proxy：all test MAE=0.9630/R2=0.5038，是 v1.2.9 内 MAE 最好；species-extrapolation-only MAE=0.9884/R2=0.4786，MAE 优于 v1.2.8 f100 censored 锚点的外推层，但总体 R2 和 NOEC 稳定性不足。
  - f100 tanimoto+proxy：all test MAE=0.9710/R2=0.5100，是 v1.2.9 内 R2/RMSE 最好；overall in-domain MAE=0.9499/R2=0.5257，family_seen_train MAE=0.9316/R2=0.5290，说明结构+proxy 组合更偏向已覆盖域内收益；但 species-extrapolation-only MAE=1.1953/R2=0.3355，明显弱于 pure proxy，提示该组合会牺牲物种外推层。
  - 阶段判断：proxy/source-rule 对 f100 有真实但不干净的信号；不能直接作为最终主线，也不应进入大矩阵。下一步采用 f100-only alpha 敏感性，确认 pure proxy 与 tanimoto+proxy 的权衡是否可通过更温和/更强 alpha 改善；继续不启用 DANN/MMD/CORAL。
- v1.2.10 f100 proxy alpha 敏感性已启动：
  - 脚本：`scripts/run_v1_2_10_f100_proxy_alpha_remote.sh`。
  - 输出根目录：`outputs/experiments/v1_2_10_f100_proxy_alpha_remote`；汇总目录：`outputs/experiments/v1_2_10_f100_proxy_alpha_remote_summary`。
  - 远端启动时间：2026-06-23 21:46:23 +08:00；PID=`674041`；日志 `outputs/logs/run_v1_2_10_f100_proxy_alpha_matrix_20260623_214623.log`。
  - 4 个候选：f100 × `proxy_distance_to_finetune` / `tanimoto_proxy_to_finetune` × alpha `0.25/0.75`；固定 CE authority-bin loss=0.025、censored_weight=0.01、finetune 60 epoch；随后自动跑 4 个 AD audit 与 `ad_gate_summary.csv`。

## 2026-06-22 v1.2.7 censored / ordinal / AD 第一批

- 目标：按新计划先做阶段 1/2/6 的最小代码改动，并启动第一批 10 个小矩阵任务；暂不实现或启用 DANN/MMD/CORAL。
- 代码范围：
  - 新增 `qsar_tl/training/ordinal_binning.py`：toxicity-bin head 仍输出 K 类 logits，`ordinal` mode 使用 CDF/EMD 风格 ordinal loss，跨多档错误惩罚更大；`aux_classification` CE 路径保持不变。
  - 新增 `qsar_tl/training/censored_loss.py` 和 `scripts/audit_censored_records.py`：当前第一批只做 `censored_audit_only`，统计 `> / < / >= / <=` 在 endpoint、unit_family、medium、task_family/task_head 中的规模；hinge loss 方向函数已有测试，但未接入本批训练。
  - 新增 `scripts/audit_proxy_bins.py`：只做 MolLogP/TPSA/MolWt/logKoc proxy 分层审计，不实现 proxy source weighting。
  - 扩展 `qsar_tl/evaluation/application_domain.py` 与新增 `scripts/audit_prediction_application_domain.py`：输出化合物 Tanimoto/Williams/proxy-distance 与物种 taxon/seen-train/life-stage/task-family 覆盖审计，生成 `chemical_ad_metrics.csv`、`species_ad_metrics.csv`、`ad_stratified_metrics.csv`、`ad_failure_cases.csv`。
  - 新增远端脚本 `scripts/run_v1_2_7_censored_ordinal_ad_first_batch_remote.sh`，输出根目录 `outputs/experiments/v1_2_7_censored_ordinal_ad_first_batch_remote`。
- 本地/远端验证：
  - 本地：`E:\TOOLS\anaconda\python.exe -m pytest tests\test_ordinal_binning.py tests\test_censored_loss.py tests\test_toxicity_binning.py tests\test_modeling_shapes.py tests\test_deep_experiment_cache.py tests\test_application_domain.py tests\test_summarize_deep_runs.py -q`，`48 passed`。
  - 远端：`bash -n scripts/run_v1_2_7_censored_ordinal_ad_first_batch_remote.sh` 通过；`/opt/anaconda3/bin/python -m pytest tests/test_ordinal_binning.py tests/test_censored_loss.py tests/test_toxicity_binning.py tests/test_modeling_shapes.py tests/test_application_domain.py -q`，`17 passed`。
- 第一批 10 个任务：
  1. `transfer_f20_no_bin_rerun`
  2. `transfer_f20_bin_ordinal_lw005`
  3. `transfer_f100_no_bin_rerun`
  4. `transfer_f100_bin_ordinal_lw0025`
  5. `soil_fullC_no_bin_anchor`
  6. `soil_fullC_bin_ordinal_lw005`
  7. `censored_audit_only`
  8. `proxy_audit_logp_koc_tpsa_bins`
  9. `best_transfer_f100_ad_audit`
  10. `best_transfer_f20_ad_audit`
- 2026-06-23 最终状态：
  - 第一批 10 个任务全部完成，最后一次恢复日志为 `outputs/logs/run_v1_2_7_censored_ordinal_ad_first_batch_resume5_20260623_141758.log`；结束时间 `2026-06-23T14:19:51+08:00`。
  - 汇总目录：`outputs/experiments/v1_2_7_censored_ordinal_ad_first_batch_remote_summary`，包含 `focus_summary.csv`、`best_by_split.csv`、`common_task_summary.csv`、`common_task_comparison.csv`、`family_summary.csv`、`effect_level_summary.csv`、`toxicity_bin_summary.csv`、`audit_summary.csv`。
  - 远端运行中发现 `best_transfer_f20_ad_audit` 旧孤儿进程与当前恢复进程重复写同一输出目录；已停掉孤儿进程，并将 AD audit 脚本改为只对 query 预测行计算 taxonomy/species context，训练端仅建 reference set。修补后 f20 AD audit 用 89 秒完成。
  - `censored_audit_only` 完成：全表 1,234,077 行，其中 censored 行 56,699，说明 censored 约束值得进入下一轮受控训练，但本批只审计、未接入训练 loss。
  - `proxy_audit_logp_koc_tpsa_bins` 完成：记录 299,139 行，summary 285 行；本批只做 MolLogP/TPSA/MolWt/logKoc proxy 分层诊断，未启用 proxy source weighting。
  - `best_transfer_f100_ad_audit` 完成：focus prediction rows=5,200，test n=2,594；`species_extrapolation` 223 条，`in_domain` 2,371 条。
  - `best_transfer_f20_ad_audit` 完成：focus prediction rows=3,086，test n=2,557；`species_extrapolation` 222 条，`in_domain` 2,335 条。
- v1.2.7 ordinal / no-bin 主要 test 指标：
  - `transfer_f20_no_bin_rerun`：n=2,557，任务头=25，MAE=1.1958，RMSE=1.5555，R2=0.2529。
  - `transfer_f20_bin_ordinal_lw005`：n=2,557，任务头=25，MAE=1.1717，RMSE=1.5365，R2=0.2711；相对 no-bin 改善 MAE 0.0241、R2 0.0182。
  - `transfer_f100_no_bin_rerun`：n=2,594，任务头=30，MAE=1.0033，RMSE=1.3166，R2=0.4774。
  - `transfer_f100_bin_ordinal_lw0025`：n=2,594，任务头=30，MAE=0.9895，RMSE=1.3043，R2=0.4871；相对 no-bin 改善 MAE 0.0138、R2 0.0097。
  - `soil_fullC_no_bin_anchor`：n=2,266，任务头=17，MAE=1.0409，RMSE=1.3749，R2=0.4241。
  - `soil_fullC_bin_ordinal_lw005`：n=2,266，任务头=17，MAE=1.0339，RMSE=1.3731，R2=0.4256；相对 no-bin 改善很小。
- 与 v1.2.6 CE authority-bin 基线对比：
  - f20：ordinal MAE=1.1717/R2=0.2711，仍弱于 CE authority-bin 最佳 `transfer_f20_source_alpha1_authority_bin_aux_lw005` 的 MAE=1.1478/R2=0.2978。
  - f100：ordinal MAE=0.9895/R2=0.4871，仍弱于 CE authority-bin 最佳 `transfer_f100_source_alpha1_authority_bin_aux_lw0025` 的 MAE=0.9768/R2=0.4978。
  - soil fullC：ordinal MAE=1.0339/R2=0.4256，弱于 CE authority-bin `soil_fullC_authority_bin_aux_lw005_core` 的 MAE=1.0036/R2=0.4677。
  - 当前结论：binning 应保留在训练中，但当前 CDF/EMD 风格 ordinal loss 不应替代普通 CE authority-bin；CE authority-bin 继续作为迁移主线，ordinal 可保留为诊断/备选。
- AD 审计结论：
  - v1.2.6 f100 CE 最佳在 test 全部样本 MAE=0.9768/R2=0.4978；in-domain 子集 n=2,371，MAE=0.9579/R2=0.5130；species-extrapolation 子集 n=223，MAE=1.1783/R2=0.3286。
  - v1.2.6 f20 CE 最佳在 test 全部样本 MAE=1.1478/R2=0.2978；in-domain 子集 n=2,335，MAE=1.1239/R2=0.3203；species-extrapolation 子集 n=222，MAE=1.3996/R2=0.0554。
  - 物种覆盖是明显风险源：f20 中 `species_task_family_seen_train=True` 子集 MAE=1.0154/R2=0.4445，而 False 子集 MAE=1.2467/R2=0.1825；f100 中 True 子集 MAE=0.9018/R2=0.5726，而 False 子集 MAE=1.0339/R2=0.4357。
  - 化合物 AD 当前阈值下几乎不构成主要失败来源，主要警告来自物种外推；下一轮不宜只按化学相似度过滤，应把 species/task-family coverage 作为报告分层或 AD gate。
- 远端启动状态：
  - 启动时间：2026-06-22 22:24:11 +08:00。
  - PID：`2112558`。
  - 日志：`outputs/logs/run_v1_2_7_censored_ordinal_ad_first_batch_20260622_222411.log`。
  - 计时表：`outputs/logs/run_v1_2_7_censored_ordinal_ad_first_batch_times.csv`。
  - 首个 run 已开始：`transfer_f20_no_bin_rerun`，split `M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
- 2026-06-22 约 23:48 +08:00 阶段性结果：
  - 远端 PID `2112558` 仍在运行；10 个任务中前 3 个训练 run 完成，当前正在跑 `transfer_f100_bin_ordinal_lw0025`。
  - `transfer_f20_no_bin_rerun`：test soil ECx/NOEC/LOEC focus n=2,557，任务头=25，R2=0.2529，RMSE=1.5555，MAE=1.1958。
  - `transfer_f20_bin_ordinal_lw005`：test soil ECx/NOEC/LOEC focus n=2,557，任务头=25，R2=0.2711，RMSE=1.5365，MAE=1.1717；相对 no-bin 改善 MAE 约 0.0241、R2 约 0.0182，但仍差于 v1.2.6 CE authority-bin f20 最佳 `MAE=1.1478/R2=0.2978`。
  - f20 分任务族：ordinal 对 LOEC/NOEC 有小幅改善（LOEC MAE 1.2107 -> 1.1829；NOEC MAE 1.2768 -> 1.2507），ECx MAE 也略降（0.9999 -> 0.9874），但 ECx R2 从 0.4340 降到 0.4245。
  - `transfer_f100_no_bin_rerun`：test soil ECx/NOEC/LOEC focus n=2,594，任务头=30，R2=0.4774，RMSE=1.3166，MAE=1.0033；弱于 v1.2.6 CE authority-bin f100 最佳 `MAE=0.9768/R2=0.4978`。
  - 初步判断：截至当前，分箱仍有价值；f20 ordinal 优于 no-bin，但没有超过普通 CE authority-bin。是否保留 ordinal 要等 f100 ordinal 和 soil-only anchor/audit 完成后再定。

## 2026-06-22 v1.2.6 authority-based toxicity binning 矩阵

- 目标：在当前框架内加入默认关闭的权威阈值毒性分箱辅助任务，并围绕迁移/不迁移、不同土壤样本比例和最佳既有超参组合设计新矩阵。
- 分支/代码状态：
  - 当前分支：`codex/authority-toxicity-binning-matrix`。
  - 新增 `configs/toxicity_bins.authority_v1.yaml`，按 EPA aquatic、OECD TG 207 soil screening scaffold、EPA terrestrial oral 分类记录阈值来源。
  - 新增 `qsar_tl/training/toxicity_binning.py`：为每个样本写入 `toxicity_bin_*` 字段；`water_mol_l` 优先用 `standard_value_mg_l`，否则用 RDKit/cache 的 `MolWt` 将 `standard_value_mol_l` 转为 `mg/L` 后分箱。未能分箱样本仍参与主回归，只从辅助 CE loss 中 mask。
  - `EcotoxMultiTaskNetwork` 新增默认关闭的 toxicity-bin auxiliary logits；训练 loss 为原 weighted Huber regression + `loss_weight * CE(toxicity_bin)`，batch 无 eligible bin 时 CE 跳过。
  - 训练输出新增 `toxicity_bin_metrics.csv` 与 `toxicity_bin_boundary_audit.csv`；`manifest.json` 记录 eligible、boundary、conversion/status counts；`predictions.csv` 保留分箱和单位换算字段。
  - 新增远端矩阵脚本：`scripts/run_v1_2_6_authority_binning_matrix_remote.sh`，支持 `smoke/core/hpo/all`；新增汇总脚本：`scripts/summarize_deep_runs.py`。
- 本地/远端验证：
  - 本地：`E:\TOOLS\anaconda\python.exe -m pytest tests\test_toxicity_binning.py tests\test_modeling_shapes.py tests\test_deep_experiment_cache.py -q`，`38 passed`。
  - 远端：`bash -n scripts/run_v1_2_6_authority_binning_matrix_remote.sh` + 同一 pytest，`38 passed`。
- 远端 smoke 已通过：
  - 输出根目录：`outputs/experiments/v1_2_6_authority_binning_matrix_remote`。
  - soil smoke：`v1.2.6_smoke_soil_f100_authority_bin_aux_lw005/deep/full/SoilPtoxQC2_C_low_f100`，1 epoch，必需文件无缺失，eligible bin 样本 11,587，boundary 样本 1,701，eval aquatic=0。
  - transfer smoke：`v1.2.6_smoke_transfer_f20_source_alpha1_authority_bin_aux_lw005/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`，1 epoch pretrain + 1 epoch finetune，必需文件无缺失，eligible bin 样本 158,728，boundary 样本 17,933，eval aquatic=0。
  - `water_mol_l -> mg/L` 转换已实际发生：soil smoke 中 5,137 行，transfer smoke 中 22,240 行。
- 已启动 core 矩阵后台队列：
  - 远端脚本：`scripts/run_v1_2_6_authority_binning_matrix_remote.sh core`。
  - 运行状态记录：`outputs/logs/run_v1_2_6_authority_binning_matrix_times.csv`。
  - core 队列 7 个正式 run 均已完成，exit=0；另有 2 个 smoke run，exit=0。
  - 远端汇总目录：`outputs/experiments/v1_2_6_authority_binning_matrix_remote_summary`。
  - 汇总文件：`focus_summary.csv`、`family_summary.csv`、`effect_level_summary.csv`、`toxicity_bin_summary.csv`、`audit_summary.csv`、`best_by_split.csv`、`common_task_summary.csv`、`common_task_comparison.csv`。
- core audit：
  - 9 个 smoke/core run 的 `required_files_present=True`，`aquatic_eval_rows=0`。
  - eligible bin 覆盖：soil f20 2,623；soil f100/fullC 11,587；transfer f20 158,728；transfer f50 166,751；transfer f100 172,773。
- core ECx/NOEC/LOEC focus test 指标：
  - `transfer_f20_source_alpha1_authority_bin_aux_lw005`：n=2,557，任务头=25，R2=0.2978，RMSE=1.5081，MAE=1.1478。
  - `transfer_f20_source_alpha1_effect025_authority_bin_aux_lw005`：n=2,557，任务头=25，R2=0.2119，MAE=1.2009；说明 authority-bin 与 effect-level beta0.25 叠加不稳，会拖累 NOEC/LOEC。
  - `transfer_f50_source_alpha1_authority_bin_aux_lw005`：n=2,591，任务头=29，R2=0.3971，MAE=1.0713。
  - `transfer_f100_source_alpha1_authority_bin_aux_lw005`：n=2,594，任务头=30，R2=0.4770，MAE=0.9993。
  - `soil_f20_authority_bin_aux_lw005_augN001`：n=1,407，任务头=6，R2=0.0342，MAE=1.2815；明显弱于既有 f20 soil-only/transfer。
  - `soil_f100_authority_bin_aux_lw005_eff050`：n=2,266，任务头=17，R2=0.4500，MAE=1.0200；弱于 v1.2.5 f100 最佳 MAE=0.9897。
  - `soil_fullC_authority_bin_aux_lw005_core`：n=2,266，任务头=17，R2=0.4677，MAE=1.0036；与 v1.2.5 fullC 最佳 MAE=1.0013 基本持平。
- 共同任务头公平对比：
  - f20 common 6-task：soil f20 MAE=1.2815/R2=0.0342；transfer f20 authority-bin MAE=1.1637/R2=0.2467。
  - f20 common 6-task + effect0.25：transfer MAE=1.2252/R2=0.1278，差于不加 effect0.25。
  - f100 common 17-task：soil f100 MAE=1.0200/R2=0.4500；transfer f100 authority-bin MAE=0.9972/R2=0.4705。
  - fullC soil vs transfer f100 common 17-task：soil fullC MAE=1.0036/R2=0.4677；transfer f100 authority-bin MAE=0.9972/R2=0.4705。
- 当前判断：
  - authority-bin auxiliary loss 对 `source_weighting=tanimoto_to_finetune alpha=1.0` 的迁移路线有实质收益：f20 test MAE 从上一轮 `matrix_source_tanimoto_alpha1` 的 1.1634 降到 1.1478，R2 从 0.2785 提到 0.2978。
  - authority-bin 对 soil-only fullC 基本中性，对 soil-only f20/f100 不如上一轮 best；因此它更像是迁移阶段的辅助结构正则，而不是 soil-only 的通用提升。
  - 下一步 HPO 不应继续强化 effect-level beta，而应围绕 `toxicity_binning.loss_weight` 做轻量敏感性。
- HPO 队列：
  - 已完成：`bash scripts/run_v1_2_6_authority_binning_matrix_remote.sh hpo`，完成时间 2026-06-22 21:02:27 +08:00。
  - 日志：`outputs/logs/run_v1_2_6_authority_binning_matrix_hpo_20260622_181645.log`。
  - HPO + core 总计 19 个 run；`audit_summary.csv` 核验 `required_files_present=True`、`aquatic_eval_rows=0`、`exit_code=0`。
  - HPO 内容：loss_weight `0.025/0.10` 对 soil f100/fullC、transfer f20/f100 做敏感性，并补跑 transfer f20 + effect0.25 的 lw0025/lw010 联合对照。
- HPO 最终结论：
  - f20 transfer 最优仍是默认 `toxicity_binning.loss_weight=0.05`：test n=2,557，R2=0.2978，MAE=1.1478；`0.025` 和 `0.10` 分别为 MAE=1.1812/R2=0.2588、MAE=1.1827/R2=0.2534。
  - f100 transfer 最优为 `toxicity_binning.loss_weight=0.025`：test n=2,594，R2=0.4978，MAE=0.9768；优于默认 0.05 的 MAE=0.9993/R2=0.4770，也略优于 v1.2.5 soil-only f100 最佳 MAE=0.9897/R2=0.4851。
  - soil-only 中 authority-bin HPO 未超过 v1.2.5 soil-only 上限：soil f100 最好为 lw005/lw010 约 MAE=1.020-1.022，fullC 最好仍是 lw005 MAE=1.0036/R2=0.4677。
  - effect-level beta0.25 与 authority-bin 叠加仍不建议作为主策略：f20 + effect0.25 最好是 lw010，test MAE=1.1532/R2=0.2766；MAE 接近但仍差于不加 effect-level 的 1.1478/R2=0.2978，且默认 lw005 叠加会明显退化到 MAE=1.2009/R2=0.2119。
  - 共同任务公平对比：f20 common 6-task 最优仍是 transfer f20 authority lw005，MAE=1.1637/R2=0.2467；f100 common 17-task 最优为 transfer f100 authority lw0025，MAE=0.9827/R2=0.4820；fullC soil vs transfer f100 common 17-task 中，transfer f100 lw0025 也优于 soil fullC authority lw005（MAE=0.9827 vs 1.0036）。
  - family 分层中，transfer f100 lw0025 的 ECx/LOEC/NOEC test MAE 分别为 0.8491/1.0098/1.0065，R2 分别为 0.5654/0.5088/0.4557；相对 f20 lw005 的 1.0183/1.1704/1.1891 和 0.3707/0.3226/0.2375 明显改善。
  - toxicity-bin 分层显示当前 test 样本主要按水相 mg/L 阈值进入 `water_*` bin；bin 内 R2 因硬分箱后目标范围变窄常为负，分层诊断应以 MAE 和边界样本数量为主。`water_very_high_toxicity` 仍是高误差层，f100 lw0025 non-boundary MAE=1.5061，f20 lw005 non-boundary MAE=2.2165。
  - 下一轮建议：迁移主线保留 `source_weighting=tanimoto_to_finetune alpha=1.0` + authority-bin aux；f20 用 loss_weight=0.05，f100/fullC 方向优先试 loss_weight=0.025；暂不继续 Phase 3 soft-expert 或 effect-level 叠加，除非先做针对高毒 bin 的专门误差修正。

## 2026-06-22 v1.2.5 soil-only upper-bound 矩阵

- 目标：不使用水相预训练，将 `aggregated_task_records_soil_ptox_qc` 土壤样本带入当前深度框架，放大 soil-only 实验矩阵以估计当前框架下的土壤单域性能上限，并报告运行时间。
- 分支/代码状态：
  - 当前分支：`codex/workspace-cleanup-20260622`。
  - 发现传统模型基线存在目标字段泄漏风险：新版聚合字段 `target_value_weighted_mean`、`target_value_unweighted_median`、`target_value_weighted_std` 未被排除，导致初次传统基线出现近零 MAE/近 1 R2 的不可信结果。
  - 已修补 `qsar_tl/training/baseline.py`：排除所有 `target_value*` 与 `tox_value*` 前缀字段；已补充 `tests/test_baseline_models.py` 定向测试。
  - 测试：本地 `E:\TOOLS\anaconda\python.exe -m pytest tests\test_baseline_models.py -q` 通过，远端同一测试也通过，均为 `7 passed`。
- 运行矩阵：
  - 深度模型 99 个 run：`SoilPtoxQC2_C_low_f20`、`SoilPtoxQC2_C_low_f100`、`SoilPtoxQC2_C_chemical_holdout_8_2` 三个 split。
  - 核心网格：learning rate `0.0003/0.0005/0.0008` × dropout `0.05/0.10/0.20` × weight decay `1e-6/9.856751793848817e-06/3e-5`，`epochs=50`，early stopping patience 15。
  - 附加策略：effect-level weighting beta `0.25/0.50`、数值噪声增强 `0.01/0.03`、80 epoch 长跑对照。
  - 修补后 no-leakage 传统基线 18 个 run：RandomForest、ExtraTrees、XGBoost、LightGBM、HistGradientBoosting、ElasticNet × 3 个 split；MLP 因 sklearn `_best_coefs` 异常未纳入 no-leakage 重跑。
- 运行时间：
  - 主矩阵日志：`outputs/logs/run_v1_2_5_soil_only_upper_bound_20260622.log`。
  - 主矩阵墙钟：2026-06-22 08:35:51 到 11:16:43，约 2:40:52；`run_times` 中 120 个计时任务，117 个成功，成功任务累计 9627 秒。其中深度模型 99 个成功，初次传统基线 18 个成功、3 个 MLP 失败。
  - no-leakage 传统基线日志：`outputs/logs/run_v1_2_5_soil_only_traditional_baselines_noleakage_20260622.nohup.log`。
  - no-leakage 传统基线墙钟：2026-06-22 11:35:57 到 11:43:35，约 7:38；18/18 成功，累计 457 秒。
- 输出与本地回收：
  - 远端深度完整结果：`outputs/experiments/v1_2_5_soil_only_upper_bound_remote`，约 746 MB；模型权重约 157 MB。本地仅回收 summary 与日志，完整 artifact 保留远端，避免无必要同步大文件。
  - 本地深度 summary：`outputs/experiments/v1_2_5_soil_only_upper_bound_remote_summary`。
  - no-leakage 传统基线完整小目录：`outputs/experiments/v1_2_5_soil_only_traditional_baselines_noleakage_remote`。
  - 关键新增汇总：`v1_2_5_soil_only_upper_bound_comparison.csv`、`v1_2_5_best_family_summary.csv`、`v1_2_5_best_effect_level_summary.csv`、`traditional_baseline_focus_comparable.csv`。
- 审计：
  - `deep_audit_summary.csv` 中所有 v1.2.5 soil-only run 的 `required_files_present=True`，`aquatic_eval_rows=0`。
  - 代表性 prediction medium counts：f20 为 `{"test|soil": 1407, "train|soil": 1603}`；f100/fullC 为 `{"test|soil": 2322, "train|soil": 11254}`。
- 深度模型最佳结果，均为 ECx/LOEC/NOEC focus test：
  - f20 最低 MAE：`augN001_f20_lr5e4_do010_wd1e5_e50`，n=1407，任务头=6，R2=0.1145，RMSE=1.6232，MAE=1.2310，Huber=0.8279。
  - f20 最高 R2：`core_f20_lr3e4_do005_wd1e5_e50`，n=1407，任务头=6，R2=0.1421，RMSE=1.5976，MAE=1.2407，Huber=0.8275。
  - f100 最佳：`eff050_f100_lr5e4_do010_wd1e5_e50`，n=2266，任务头=17，R2=0.4851，RMSE=1.3001，MAE=0.9897，Huber=0.6041。
  - full_C 最佳：`core_fullC_lr3e4_do020_wd1e5_e50`，n=2266，任务头=17，R2=0.4617，RMSE=1.3293，MAE=1.0013，Huber=0.6166。
- 与 v1.2.4/current transfer 对照：
  - f20：v1.2.5 最低 MAE=1.2310，未超过 v1.2.4 `soilonly_current_low_f20` 的 MAE=1.2084；也弱于迁移 common-task 最好结果，例如 `source_tanimoto_alpha1_effect_beta0p5` common f20 MAE=1.1762。说明低土壤样本下，水相预训练/迁移仍主要提供小样本补偿。
  - f100：v1.2.5 最佳 MAE=0.9897，优于 v1.2.4 `soilonly_current_low_f100` 的 MAE=1.0486，也明显优于 `matrix_source_tanimoto_alpha1` common f100 MAE=1.1680。
  - full_C：v1.2.5 最佳 MAE=1.0013，略优于 v1.2.4 `soilonly_current_full_C` 的 MAE=1.0095，R2 从 0.4456 提升到 0.4617。
- 分层结果：
  - f100 最佳 run：ECx MAE=0.8050、LOEC MAE=1.0331、NOEC MAE=1.0324；ECx/LOEC/NOEC R2 分别为 0.5994/0.4861/0.4377。
  - full_C 最佳 run：ECx MAE=0.8148、LOEC MAE=1.0377、NOEC MAE=1.0511；ECx/LOEC/NOEC R2 分别为 0.5985/0.4777/0.3904。
  - f20 最佳 run：ECx 只覆盖 `ECx_Growth`，MAE=1.0721；LOEC MAE=1.2127；NOEC MAE=1.2977。NOEC 仍是低样本下最难稳定的族。
  - ECx effect-level：f100/full_C 的 EC50 层 n=375，MAE 约 0.82-0.84，R2 约 0.53；EC10/EC25 样本量很小，单层 R2 波动较大，不能过度解释。
- no-leakage 传统基线可比口径：
  - f20 最佳为 HistGradientBoosting：R2=-0.0247，MAE=1.3250。
  - f100/fullC 最佳为 RandomForest：R2=0.3314，MAE=1.0796。
  - 传统模型不再出现不合理近零误差；在相同任务集合下低于深度模型，说明当前深度框架的上限主要来自多任务深度表征而不是树模型基线。
- 当前结论：
  - 土壤样本足够时，soil-only 在当前框架内的上限约为 MAE 1.0 log unit、R2 0.46-0.49；轻量 HPO 与 effect-level beta0.5 能进一步改善 f100。
  - 低样本 f20 未因更大 soil-only 网格而提升，反而略弱于 v1.2.4 和迁移共同任务结果；因此迁移学习的价值目前集中在低土壤样本/任务覆盖不足场景，而不是足量土壤样本下超过 soil-only。
  - 后续若继续探索上限，优先做 soil-only f20 的数据增强/任务重采样和按任务族的验证集调参；不建议再盲目扩大全局 LR/dropout/weight decay 网格。

## 2026-06-21 v1.2.4 soil-only current-framework 对照

- 目标：按当前深度框架单独训练土壤 pTox，不使用水相预训练、不传 finetune 参数、不启用 source/effect-level weighting，用于判断水相迁移相对土壤单域训练的真实增益。
- 已同步当前代码/配置到远端 `/home/easyai/DL1/ecotox_qsar_transfer`，并完成 3 个远端 run：
  - `soilonly_current_low_f20`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_low_f20`。
  - `soilonly_current_low_f100`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_low_f100`。
  - `soilonly_current_full_C`：`source_table=aggregated_task_records_soil_ptox_qc`，`split=SoilPtoxQC2_C_chemical_holdout_8_2`。
- 运行参数：`epochs=30`，`batch_size=512`，`learning_rate=0.0005`，`weight_decay=9.856751793848817e-06`，`scheduler=cosine`，`dropout=0.1`，`target_standardization=per_task_target`，`metric_min_n=5`，`source_weighting=none`，`effect_level_weighting=false`，`swa=false`。
- 运行脚本和日志：
  - 本地脚本：`outputs/logs/run_v1_2_4_soil_only_current_framework_20260621.sh`
  - 远端/本地日志：`outputs/logs/run_v1_2_4_soil_only_current_framework_20260621.log`
  - 远端完成时间：`[done] 2026-06-21T19:16:25+08:00 v1.2.4 soil-only current-framework matrix`
- 输出与本地回收：
  - 远端完整结果：`outputs/experiments/v1_2_4_soil_only_current_framework_remote`
  - 本地 summary：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_summary`
  - 本地无模型 artifact 包：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_artifacts_no_model.tar.gz`
  - 本地完整解包副本：`outputs/experiments/v1_2_4_soil_only_current_framework_remote_artifacts_extracted/v1_2_4_soil_only_current_framework_remote`
  - 汇总文件：`focus_summary.csv`、`common_task_comparison.csv`、`family_summary.csv`、`effect_level_summary.csv`、`audit_summary.csv`、`summary_manifest.json`
- 产物和介质审计：
  - 3 个 run 均生成 `predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `soilonly_current_low_f20` 预测行：test/soil 1,407；train/soil 1,603；评估集 aquatic=0。
  - `soilonly_current_low_f100` 预测行：test/soil 2,322；train/soil 11,254；评估集 aquatic=0。
  - `soilonly_current_full_C` 预测行：test/soil 2,322；train/soil 11,254；评估集 aquatic=0。
  - 3 个 run 的 `source_weighting.applied=false`，`effect_level_weighting.applied=false`。
- ECx/NOEC/LOEC 合并 test 指标，按 `predictions.csv` 重新计算：
  - `soilonly_current_low_f20`：n=1,407，任务头=6，R2=0.1848，RMSE=1.5573，MAE=1.2084。
  - `soilonly_current_low_f100`：n=2,266，任务头=17，R2=0.4202，RMSE=1.3796，MAE=1.0486。
  - `soilonly_current_full_C`：n=2,266，任务头=17，R2=0.4456，RMSE=1.3490，MAE=1.0095。
  - 对照：`matrix_source_tanimoto_alpha1` 全量 focus test 为 n=2,557，任务头=25，R2=0.2785，RMSE=1.5286，MAE=1.1634；`source_tanimoto_alpha1_effect_beta0p25` 为 R2=0.2778，RMSE=1.5293，MAE=1.1594。
- 共同任务头公平对比：
  - 与 `soilonly_current_low_f20` 的 6 个任务头对齐时，`matrix_source_tanimoto_alpha1`：R2=0.1935，RMSE=1.5483，MAE=1.1966；`source_tanimoto_alpha1_effect_beta0p25`：R2=0.1928，RMSE=1.5490，MAE=1.1913；`soilonly_current_low_f20`：R2=0.1848，RMSE=1.5573，MAE=1.2084。说明 f20 低样本下 soil-only 接近但仍略弱于迁移，且任务覆盖明显少于迁移模型。
  - 与 `soilonly_current_low_f100` 的 17 个任务头对齐时，`matrix_source_tanimoto_alpha1`：R2=0.2746，RMSE=1.5335，MAE=1.1680；`source_tanimoto_alpha1_effect_beta0p25`：R2=0.2720，RMSE=1.5363，MAE=1.1664；`soilonly_current_low_f100`：R2=0.4202，RMSE=1.3796，MAE=1.0486；`soilonly_current_full_C`：R2=0.4456，RMSE=1.3490，MAE=1.0095。
- family 分层 test：
  - `soilonly_current_full_C`：ECx MAE=0.8112，LOEC MAE=1.0547，NOEC MAE=1.0565。
  - `soilonly_current_low_f100`：ECx MAE=0.8607，LOEC MAE=1.0777，NOEC MAE=1.1057。
  - `matrix_source_tanimoto_alpha1`：ECx MAE=0.9814，LOEC MAE=1.2069，NOEC MAE=1.2098。
  - `source_tanimoto_alpha1_effect_beta0p25`：ECx MAE=1.0308，LOEC MAE=1.1681，NOEC MAE=1.2136。
- 当前判断：
  - f20 小土壤样本下，水相迁移仍有价值：它保留更多任务头，并在共同 6 个任务头上略优于 soil-only。
  - 一旦土壤训练样本提高到 f100/full_C，同一 current-framework 的 soil-only 明显优于当前水相预训练迁移策略，说明负迁移仍存在；水相预训练的主收益更像是低样本任务覆盖和小样本补偿，而不是在足量土壤 pTox 上提供更强表征。
  - 下一步若继续提升，应优先做更严格 source filtering / curriculum annealing，而不是继续增大 effect beta、source alpha 或单纯延长 finetune。

## 2026-06-21 v1.2.3 source Tanimoto + effect-level weighting 矩阵

- 已在分支 `codex/effect-level-weighting-source-tanimoto` 上新增默认关闭训练开关：
  - `training.effect_level_weighting` / CLI `--effect-level-weighting --effect-level-weighting-beta <beta>`。
  - 仅对实际训练索引 `train + finetune_train` 中 `ECx/LCx/ICx/LDx` 且存在 `effect_level_x` 的样本计算 `frequency^(-beta)` 权重；在 `0.5-3.0` 约束内归一化到均值 1，并与已有 `source_weighting` 样本权重相乘。
  - validation、finetune_validation、test 不参与频率统计，也不在 loss 评估中使用样本权重。
  - `manifest.json` 新增 `effect_level_weighting`；`history.csv` 新增 source/effect-level/final sample weight 审计列。
- 本地验证：
  - `E:\TOOLS\anaconda\python.exe -m compileall qsar_tl\training\deep_experiment.py qsar_tl\training\train.py tests\test_deep_experiment_cache.py` 通过。
  - `E:\TOOLS\anaconda\python.exe -m pytest tests\test_deep_experiment_cache.py tests\test_remote_execution_config.py -q` 通过，32 passed；本地 Torch 仍有既有 NumPy DLL 初始化 warning。
  - `configs\experiment.remote.easyai.yaml` 配置校验通过。
- 远端 smoke 已通过：
  - 输出目录：`outputs/experiments/v1_2_3_effect_level_weighting_smoke_remote/v1.2.3_smoke_source_tanimoto_effect_beta0p5/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 参数：1 epoch pretrain + 1 epoch finetune，`source_weighting=tanimoto_to_finetune alpha=1.0`，`effect_level_weighting beta=0.5`。
  - 必需文件均生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `effect_level_weighting` 实际加权 139,736 个训练样本，effect weight 范围约 0.6712-3.0，均值 1.0；与 source weighting 相乘后的 final sample weight 均值约 0.9966。
  - 介质审计：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
- 已完成同预算远端正式矩阵并拉回本地汇总：
  - 远端/本地汇总目录：`outputs/experiments/v1_2_3_effect_level_weighting_matrix_remote_summary`
  - 队列脚本/日志：`outputs/logs/run_v1_2_3_effect_level_weighting_matrix_20260621_151249.sh`、`outputs/logs/run_v1_2_3_effect_level_weighting_matrix_20260621_151249.log`
  - 汇总文件：`focus_summary.csv`、`focus_comparison_vs_baselines.csv`、`family_summary.csv`、`effect_level_summary.csv`、`manifest_summary.csv`、`audit_summary.json`
  - 5 个矩阵 run 必需文件均无缺失，`finetune`、`finetune_validation`、`test` 的 aquatic 混入均为 0。
- ECx/NOEC/LOEC 合并指标，按 `predictions.csv` 重新计算：
  - `source_tanimoto_alpha1_effect_beta0p25` test 最好：R2=0.2778，RMSE=1.5293，MAE=1.1594；相对 `matrix_source_tanimoto_alpha1` 的 test MAE=1.1634 小幅改善约 0.0040 log unit，但 R2 略低 0.0007。
  - `matrix_source_tanimoto_alpha1` 仍是 R2/RMSE 最稳参照：test R2=0.2785，RMSE=1.5286，MAE=1.1634。
  - `source_tanimoto_alpha1_effect_beta0p75` test：R2=0.2767，RMSE=1.5306，MAE=1.1677，整体接近 alpha1 原策略。
  - `source_tanimoto_alpha1_effect_beta0p5` test：R2=0.2405，RMSE=1.5683，MAE=1.1938，验证集不优且 test 明显退化。
  - `source_tanimoto_alpha1_effect_beta0p5_swa_finetune15` test：R2=0.2294，RMSE=1.5798，MAE=1.1874；SWA 改善 ECx 但牺牲 NOEC/LOEC，combined test 不建议保留。
  - 可选 `source_tanimoto_alpha1p5_effect_beta0p5` finetune_validation 最好：R2=0.7284，RMSE=0.8641，MAE=0.6215；但 test R2=0.2489，MAE=1.1934，提示更强 source alpha 更贴合微调验证集而非化学 holdout test。
- family/effect-level 层面 test：
  - beta0.25 改善 LOEC：MAE=1.1681，相对 `matrix_source_tanimoto_alpha1` 的 1.2069 改善；NOEC 基本持平略差：MAE=1.2136 vs 1.2098；ECx 明显变差：MAE=1.0308 vs 0.9814。
  - beta0.75 改善 ECx 和 LOEC：ECx MAE=0.9891，LOEC MAE=1.1826；但 NOEC 退化：MAE=1.2403。
  - SWA 组合 ECx 最好：MAE=0.9484；但 LOEC/NOEC 退化到 1.2290/1.2636，是 combined test 变差主因。
  - effect-level：beta0.25 对小样本 EC10/EC25 明显改善（EC10 MAE=1.7943、EC25 MAE=1.4874），但 EC50 退化到 MAE=0.9995；beta0.75/beta0.5 对 EC50 更好（约 0.942/0.940），但 EC10/EC25 更差。
- 当前判断：
  - 新的候选最佳可记为 `source_weighting=tanimoto_to_finetune alpha=1.0 + effect_level_weighting beta=0.25`，但收益很小，属于 MAE 方向的细微改善而非全面胜出。
  - 若主目标优先 chemical-holdout combined MAE，可保留 beta0.25 作为下一轮候选；若优先稳健 R2/RMSE 和 ECx 单族，上一轮 `matrix_source_tanimoto_alpha1` 仍更均衡。
  - 不建议继续增大 effect beta 或 source alpha；这些设置会改善 finetune_validation 或 EC50，但削弱 NOEC/LOEC 和 test 泛化。

## 2026-06-20 v1.2.2 第一批迁移优化矩阵启动

- 已在分支 `codex/transfer-matrix-coral-sourceweight-swa` 上实现第一批矩阵的默认关闭训练开关：
  - `training.source_weighting` / CLI `--source-weighting-method tanimoto_to_finetune --source-weighting-alpha <alpha>`：按土壤 finetune 训练样本的最大 Tanimoto 相似度重加权水相 source 训练样本，权重均值归一到 1。
  - `training.domain_alignment` / CLI `--domain-alignment-method coral --domain-alignment-weight <weight>`：在 shared representation 上加入 CORAL 域对齐损失，默认 pretrain 阶段用 water source batch 对齐 soil finetune reference batch。
  - `training.swa` / CLI `--swa --swa-start-epoch <epoch>`：在指定阶段后半程累积 averaged weights，当前默认用于 finetune 阶段。
  - `history.csv` 新增 `mean_task_loss`、`mean_alignment_loss`、`alignment_steps`、`swa_updates`；`manifest.json` 新增 `source_weighting`、`domain_alignment`、`swa` 记录。
- 本地验证：
  - `compileall qsar_tl/modeling/network.py qsar_tl/training/deep_train.py qsar_tl/training/deep_experiment.py qsar_tl/training/train.py tests/test_deep_experiment_cache.py` 通过。
  - `pytest tests/test_deep_experiment_cache.py tests/test_modeling_shapes.py -q` 通过，28 passed；本地 Torch 仍有既有 NumPy DLL 初始化 warning。
- 已同步代码到远端 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端组合 smoke 已通过：
  - 输出目录：`outputs/experiments/v1_2_2_matrix_smoke_remote/v1.2.2_smoke_coral_sourceweight_swa/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 参数：1 epoch pretrain + 1 epoch finetune，`CORAL=0.01`、`source_weighting=tanimoto_to_finetune alpha=1.0`、`SWA start_epoch=1`。
  - 必需文件均生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`、`history.csv`、`manifest.json`。
  - `source_weighting` 实际应用到 241,876 个 aquatic train 样本，soil finetune reference 为 2,193；权重范围约 0.6716-1.3431，均值 1.0。
  - CORAL pretrain reference 为 2,193 个 soil finetune 样本，pretrain alignment steps=945；SWA finetune updates=1 且 `applied=true`。
  - `predictions.csv` 介质计数：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
  - `split_medium_audit.csv` 仍记录剔除错配行：finetune aquatic 238、test aquatic 300、train soil 1,111。
- 已启动同预算远端正式矩阵后台队列：
  - 队列脚本：`outputs/logs/run_v1_2_2_transfer_matrix_20260620_115110.sh`。
  - 队列总日志：`outputs/logs/v1_2_2_transfer_matrix_20260620_115110.log`。
  - 队列进程：bash PID `1948322`；当前第一项为 `matrix_coral_0p01`。
  - 公共预算：pretrain epochs=30，finetune epochs=60，batch_size=512，learning_rate=0.0005，weight_decay=9.856751793848817e-06，dropout=0.1，target_standardization=`per_task_target`，finetune_lr=0.0003082636455810776，finetune_scheduler=`reduce_on_plateau`，finetune_validation_fraction=0.2。
  - 输出根目录：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote`。
  - 队列顺序：`matrix_coral_0p01`、`matrix_coral_0p05`、`matrix_source_tanimoto_alpha1`、`matrix_swa_finetune15`、`matrix_coral_0p01_source_alpha1`。
  - `GradNorm/PCGrad` 暂未进入本队列，因为它会明显改动多任务梯度路径，需单独分支或后续小步实现。

## 2026-06-21 v1.2.2 第一批迁移优化矩阵结果

- 远端正式矩阵队列已全部完成；汇总目录已拉回本地：
  - 远端：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote_summary`
  - 本地：`outputs/experiments/v1_2_2_transfer_optimization_matrix_remote_summary`
  - 关键文件：`focus_summary.csv`、`family_summary.csv`、`effect_level_summary.csv`、`manifest_summary.csv`、`audit_summary.json`
- split audit：
  - 所有 v1.2.2 run 的 `finetune`、`finetune_validation`、`test` 均为 soil，`aquatic_in_soil_eval=0`。
  - 介质计数一致：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685。
- ECx/NOEC/LOEC 合并指标，按 `predictions.csv` 重新计算：
  - 上一轮 `v1.2.1_best_rerun` test：R2=0.1700，RMSE=1.6396，MAE=1.2311；finetune_validation：R2=0.6735，RMSE=0.9475，MAE=0.6844。
  - `matrix_source_tanimoto_alpha1` test 最好：R2=0.2785，RMSE=1.5286，MAE=1.1634；相对 v1.2.1 best，test MAE 改善约 0.0677 log unit，R2 提升约 0.1085。
  - `matrix_swa_finetune15` test 次优：R2=0.2340，RMSE=1.5750，MAE=1.1967。
  - `matrix_coral_0p01_source_alpha1` test：R2=0.2086，RMSE=1.6010，MAE=1.2196，小幅优于 v1.2.1 best。
  - `matrix_coral_0p01` test：R2=0.1696，RMSE=1.6399，MAE=1.2560；与 v1.2.1 best 基本持平或略差。
  - `matrix_coral_0p05` test：R2=0.1912，RMSE=1.6185，MAE=1.2577；R2/RMSE 略改善但 MAE 变差。
  - finetune_validation 最好是 `matrix_coral_0p01`：R2=0.7170，RMSE=0.8822，MAE=0.6219；但该改善未能转化到 test，提示 CORAL 单独使用可能更贴合微调验证集而非化学 holdout test。
- family 层面 test：
  - ECx：`matrix_source_tanimoto_alpha1` 最好，R2=0.4359，RMSE=1.2974，MAE=0.9814；比 v1.2.1 best 的 MAE=1.0404 改善。
  - LOEC：`matrix_source_tanimoto_alpha1` 最好，R2=0.2695，RMSE=1.5891，MAE=1.2069；比 v1.2.1 best 的 MAE=1.2598 改善。
  - NOEC：`matrix_source_tanimoto_alpha1` 最好，R2=0.2152，RMSE=1.5723，MAE=1.2098；比 v1.2.1 best 的 MAE=1.2960 改善，是本轮最重要收益。
- effect-level 层面 test：
  - EC50：`matrix_swa_finetune15` MAE=0.9221、R2=0.4414；`matrix_source_tanimoto_alpha1` MAE=0.9332、R2=0.4448；二者均优于 v1.2.1 best 的 MAE=0.9795。
  - EC10/EC25 样本量小且仍不稳定；`matrix_source_tanimoto_alpha1` 对 EC10 MAE=1.8702，相比 v1.2.1 best 的 2.0121 略改善，但 R2 仍为负。
  - NOEC/LOEC 的 `effect_level=none` 层由 `matrix_source_tanimoto_alpha1` 同时最佳。
- 当前判断：
  - 第一批里应优先保留 `source_weighting=tanimoto_to_finetune, alpha=1.0` 作为新的候选最佳迁移策略。
  - SWA 对 EC50 有帮助，但对 EC10/EC25 和 NOEC/LOEC 不如 source weighting 稳。
  - CORAL 单独或与 source weighting 组合在 finetune_validation 上较好，但 test 泛化不足，下一步不建议继续单纯增大 CORAL weight。

## 2026-06-17 v1.2.1 效应水平、介质切分和特征 z-score 纠偏修正

- 已修正 `effect_level_x` 的建模与报告逻辑：
  - `ECx/LCx/ICx/LDx` 的效应水平数字继续作为输入，但新增 `effect_level_x_fraction`、`effect_level_x_log1p`、`effect_level_x_present` 派生通道。
  - 深度网络新增 effect-level 专门分支，避免效应水平只被混在普通数值特征中。
  - `predictions.csv` 保留 `effect_level_x`，并新增 `effect_level_metrics.csv` 与 `effect_level_metrics_filtered.csv`，用于按 EC10/EC50/LC50 等水平分层诊断。
- 已修正迁移 split 加载时的介质混入风险：
  - `load_split_frame` 会按 split assignment 的 `group_key` 校验介质域，`soil_*` 只允许 soil 行，`aquatic` 只允许 aquatic 行。
  - 训练输出新增 `split_medium_audit.csv`，记录各 split 的 aquatic/soil 数量以及被剔除的跨介质 join 行。
- 已新增核心指标过滤：
  - 原始 `metrics.csv` 保留所有分组。
  - 新增 `metrics_filtered.csv`，默认只保留 `n >= 5` 且 R2 有定义的分组，避免 `n=1/2` 的 NaN R2 子任务进入主结论和 HPO 目标。
- 已新增特征 z-score 自动纠偏：
  - `training.feature_zscore_correction.enabled=true` 默认开启，阈值默认 6。
  - 仅使用训练/微调训练样本拟合均值、标准差和截断边界，验证集、测试集和扰动评估复用同一纠偏参数。
- 本地验证：
  - `compileall qsar_tl scripts tests` 通过。
  - `pytest tests/test_deep_experiment_cache.py tests/test_modeling_shapes.py tests/test_task_mapping.py tests/test_medium_domain_tables.py tests/test_remote_execution_config.py tests/test_aquatic_soil_adaptation_split.py` 通过，49 passed。
  - 本地根目录 `ecotox_clean.sqlite` 无 `split_assignments`，真实训练 smoke 需在远端派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 同步代码后执行。

## 2026-06-17 v1.2.1 远端 smoke 与 best trial 复跑

- 已同步最新代码到远端 `/home/easyai/DL1/ecotox_qsar_transfer`，使用远端派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
- 远端 smoke 已通过：
  - source table：`aggregated_task_records_aquatic_soil_ptox_qc`。
  - split：`M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 预算：1 epoch pretrain + 1 epoch finetune，`batch_size=256`，`device=cuda:0`。
  - 输出目录：`outputs/experiments/v1_2_1_smoke_remote/v1.2.1_smoke_medium_audit_zscore/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 必需文件均在远端生成：`predictions.csv`、`metrics.csv`、`metrics_filtered.csv`、`effect_level_metrics_filtered.csv`、`split_medium_audit.csv`。
  - `predictions.csv` 介质计数：train/aquatic 241,876；finetune/soil 2,193；finetune_validation/soil 551；test/soil 2,685；finetune/finetune_validation/test 中 aquatic=0。
  - `split_medium_audit.csv` 显示被校验剔除的错配行：finetune 中 aquatic 238、test 中 aquatic 300、train 中 soil 1,111。
- 已按上一轮 HPO best trial 010 同预算复跑当前最佳迁移实验：
  - 参数：pretrain epochs=30，finetune epochs=60，batch_size=512，learning_rate=0.0005，weight_decay=9.856751793848817e-06，dropout=0.1，target_standardization=`per_task_target`，finetune_lr=0.0003082636455810776，finetune_scheduler=`reduce_on_plateau`，finetune_validation_fraction=0.2。
  - 输出目录：`outputs/experiments/v1_2_1_best_retrain_remote/v1.2.1_best_trial_010_effect_level_medium_zscore_rerun/deep/full/M_v2_aquatic_to_soil_ptox_adapt_C_f20`。
  - 远端日志：`outputs/logs/v1_2_1_best_retrain_20260617_212406.log`。
  - 训练结果：pretrain 完成 30 epoch；finetune 在第 30 个 finetune epoch 早停；最佳全局 epoch=50，即 finetune 第 20 epoch，monitor loss=0.093797。
  - 必需文件均生成，且 `predictions.csv` 介质计数同 smoke：finetune/finetune_validation/test 全为 soil，aquatic=0。
- 已生成本地对比汇总目录：`outputs/experiments/v1_2_1_best_retrain_remote_summary`。
  - `comparison_summary_v1_2_1.csv`：上一轮 baseline、上一轮 HPO best trial 010 与本轮复跑的 split 汇总。
  - `comparison_delta_vs_hpo_best_trial_010.csv`：本轮复跑相对上一轮 HPO best 的差值。
  - `comparison_family_v1_2_1.csv` 与 `comparison_family_delta_vs_hpo_best_trial_010.csv`：ECx/NOEC/LOEC 在 test 与 finetune_validation 的家族级对比。
  - `effect_level_focus_v1_2_1.csv`：ECx/NOEC/LOEC 的 effect-level 分层汇总。
- 关键对比结论：
  - 主族 `ECx/NOEC/LOEC` 的 `finetune_validation`：R2 从 0.6186 升至 0.6735，但 MAE 从 0.6324 升至 0.6844，RMSE 从 0.8556 升至 0.9475，说明验证集方差解释略好但绝对误差更差。
  - 主族 `ECx/NOEC/LOEC` 的 `test`：R2 从 0.1849 降至 0.1700，MAE 从 1.1095 升至 1.2311，RMSE 从 1.4365 升至 1.6396；本轮修正后测试误差高于上一轮 HPO best。
  - family 层面：test ECx R2 从 0.2673 升至 0.4044，但 MAE 从 0.9480 升至 1.0404；test LOEC/NOEC 均下降，LOEC MAE 1.1373 -> 1.2598，NOEC MAE 1.1727 -> 1.2960。
  - effect-level 层面：test EC50 为主要 ECx 层级，n=441，R2=0.4174，MAE=0.9795；EC10 与 EC25 样本量较小且误差较高，分别为 n=29/MAE=2.0121、n=15/MAE=1.9282。NOEC/LOEC 无独立 effect level 数值，按 `none` 层汇总。

## 2026-06-16 v2.0.0 新清洗数据远端重训启动

- 已按新清洗派生库 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite` 启动远端重训队列。
- 新版 RDKit/Morgan512 分子特征缓存已重建：
  - 文件：`outputs/features/molecular_features_rdkit_morgan512.jsonl`
  - `source_table=aggregated_task_records_qc`
  - `unique_smiles=7837`
  - `fallback_count=0`
- 远端同步已完成，远端项目目录仍为 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端预检已通过：
  - `PRAGMA quick_check` 返回 `ok`。
  - GPU 可用：RTX 4060 Ti，`cuda:0`。
  - v2 关键表规模：`aggregated_task_records_qc=477656`，`aggregated_task_records_aquatic_ptox_qc=281435`，`aggregated_task_records_soil_ptox_qc=16020`，`aggregated_task_records_soil_mg_kg_qc=17506`，`aggregated_task_records_aquatic_soil_ptox_qc=297455`。
- 已生成新版 split：
  - `AquaticPtoxQC2_` A-F splits。
  - `SoilPtoxQC2_` A-F splits。
  - `SoilMgkgQC2_` A-F splits。
  - 低土壤样本迁移 split：`M_v2_aquatic_to_soil_ptox_adapt_C_f10/f20/f50/f100`。
  - 配对 soil-only split：`SoilPtoxQC2_C_low_f10/f20/f50/f100`。
- 泄漏检查：
  - `AquaticPtoxQC2_C_chemical_holdout_8_2`、`SoilPtoxQC2_C_chemical_holdout_8_2`、`SoilMgkgQC2_C_chemical_holdout_8_2` 的 train/test 化合物分组重叠均为 0。
  - `AquaticPtoxQC2_F_chemical_adapt_7_2_1`、`SoilPtoxQC2_F_chemical_adapt_7_2_1`、`SoilMgkgQC2_F_chemical_adapt_7_2_1` 的 train 与 finetune/test 化合物分组重叠均为 0。
- Phase 0 smoke 已通过：
  - 水相 pTox B split，1 epoch，输出完整。
  - 土壤 pTox C split，1 epoch，输出完整。
  - 水相预训练到土壤 pTox f20，1 epoch pretrain + 1 epoch finetune，输出完整。
- 后台正式队列已进入 Phase 1：
  - 当前日志：`outputs/logs/v2_0_0_retrain_20260616_195035.log`。
  - 已完成：`v2.0.0_reclean_v2_aquatic_ptox_B_random/deep/full/AquaticPtoxQC2_B_random_8_2`，30 epoch，`best_epoch=30`。
  - 当前运行：`v2.0.0_reclean_v2_aquatic_ptox_C_chemical/deep/full/AquaticPtoxQC2_C_chemical_holdout_8_2`，30 epoch。
  - 调度脚本：`scripts/run_v2_0_0_retrain_remote.sh`。
- 队列后续自动执行：
  - Phase 1：水相 pTox、土壤 pTox、土壤 mg/kg 的 B/C full 30 epoch。
  - Phase 2：`f10/f20/f50/f100` 水相预训练 + 土壤 pTox 微调，以及配对 soil-only 对照。
  - Phase 3：按 Phase 2 test 主终点加权 MAE 自动选择最佳 f，并补跑 `finetune40_cosine` 与 `finetune20_constant` 诊断，同时输出 ECx/NOEC/LOEC 解释和 AD 分层指标。
- 待队列完成后：
  - 拉回 `outputs/experiments/v2_0_0_*`、`outputs/ad/v2_0_0` 和日志。
  - 汇总 R2、RMSE、MAE、Huber loss，并与 v1.0.0/v1.1.0 结果对比。
  - 生成正式 v2.0.0 实验总结文档。

## 2026-06-13 v1.1.0 迁移步长诊断与主终点解释

- 已完成远端 v1.1.0 迁移步长诊断，日志：`outputs/logs/v1_1_0_transfer_step_diagnostics_20260613_204650.log`。
- 实验目录：`outputs/experiments/v1_1_0_transfer_step_diagnostics`；三组诊断为：
  - `v1.1.0_迁移步长诊断_预训练30微调40_cosine`
  - `v1.1.0_迁移步长诊断_预训练30微调20_微调恒定学习率`
  - `v1.1.0_迁移步长诊断_预训练60微调20_cosine`
- 已新增训练调度参数：
  - `python -m qsar_tl.training.train --scheduler {none,cosine,reduce_on_plateau}`
  - `python -m qsar_tl.training.train --finetune-scheduler {none,cosine,reduce_on_plateau}`
  - `scripts/train_remote.ps1` 同步支持 `-Scheduler` 和 `-FinetuneScheduler`。
- 已新增解释汇总链路：
  - `scripts/analyze_endpoint_family_explanations.py`：按 ECx/NOEC/LOEC 输出测试误差分布，并选择代表任务运行 SHAP。
  - `scripts/summarize_transfer_step_diagnostics.py`：汇总步长诊断、终点分组误差和 SHAP Top 特征。
  - `scripts/explain_deep_model.py` 已修正 target standardization 后的解释尺度，现在 SHAP 解释的是反标准化后的原始 pTox 预测。
- 关键结果：
  - 基线 `30预训练+20微调 cosine` 的微调 loss：final 0.093646，min 0.091554。
  - `30预训练+40微调 cosine` 的微调 loss：final 0.051203，min 0.047837。
  - `30预训练+20微调 恒定LR` 的微调 loss：final/min 0.052589。
  - `60预训练+20微调 cosine` 的预训练 loss 从第 30 轮约 0.050387 继续降到第 60 轮 0.035730，但微调 final 仍为 0.091088。
- 主终点合并测试表现：
  - 基线 ECx/LOEC/NOEC 样本加权 MAE：1.1709。
  - `30预训练+40微调 cosine`：MAE 1.1327，当前最优。
  - `30预训练+20微调 恒定LR`：MAE 1.1405，接近最优。
  - `60预训练+20微调 cosine`：MAE 1.1591，仅小幅优于基线。
- 当前结论：微调阶段确实存在有效步长不足或 cosine 衰减过快的问题，但测试集误差不随训练 loss 单调改善；下一轮应优先比较微调调度、验证集早停和更稳健的迁移策略，而不是简单继续堆 epoch。
- 报告：`docs/v1.1.0_transfer_step_diagnostics.md`。

## 2026-06-12 土壤毒性预测与迁移学习重设计方案

- 已完成正式方案文档：`docs/soil_toxicity_transfer_redesign_plan.md`。
- 方案已固化以下共识：
  - 土壤毒性预测为主线，水相迁移用于学习分子构效知识并支撑土壤小样本预测。
  - 重新设计 QC 与聚合：endpoint 数字保留为 `effect_level_x`，robust z-score > 4 的明显离群值排除，跨文献采用最新文献优先、0.3 log unit 冲突阈值和年份权重加权平均。
  - 模型结构采用 shared molecular/taxon/context trunk + target-scale/medium adapter + endpoint task head。
  - 物种上下文改为 taxon 层级；`species_number` 只作为审计和消融字段，不作为正式主输入。
  - 化学应用域以 Williams leverage 为主、Tanimoto 为补充；物种应用域使用 taxon distance。
  - 不确定度第一版只输出模型不确定度和外推/应用域不确定度。
  - 数据划分同时覆盖 random、chemical holdout、AD-aware、low-soil-data transfer、target-scale-specific，并保留 A-F 兼容对照。
- 已在 `qsar_tl/data/task_tables.py` 的聚合键和聚合函数处加入显著注释，说明当前聚合不会混合 EC10/EC50 或不同量纲，但存在跨文献/跨 test batch 聚合风险。

## 2026-06-12 重设计方案第一批代码落地

- 已新增正式 QC 聚合链路：
  - `qsar_tl/data/reference_weighting.py`
  - `qsar_tl/data/qc_aggregation.py`
  - CLI：`python -m qsar_tl.cli build-qc-task-tables --config configs/experiment.example.yaml --db outputs/derived/modeling_dataset.sqlite`
- `wide_records -> target_records -> task_records` 已透传 `publication_year`，用于跨文献聚合时按最新文献优先加权。
- `task_records_qc` 输出每条任务记录的 `medium_domain`、`tox_qc_status`、`tox_qc_reason`、`robust_z`、`qc_group_n`、`reference_weight`。
- `aggregated_task_records_qc` 输出正式训练目标：
  - `target_value_median` 兼容旧训练入口，但在 `_qc` 表内实际承载文献/测试单元加权平均。
  - `target_value_unweighted_median` 保留原始中位数。
  - `cross_reference_conflict_flag` 和 `cross_reference_range` 用于审计跨文献冲突。
  - 同一 `reference_number + test_id` 先内部取中位数，再进行跨文献年份权重加权，避免同一文献多条记录放大权重。
- `scripts/build_medium_domain_tables.py` 已支持 `--table-suffix _qc`，可从 `aggregated_task_records_qc` 生成：
  - `aggregated_task_records_aquatic_ptox_qc`
  - `aggregated_task_records_soil_mg_kg_qc`
  - `aggregated_task_records_soil_ptox_qc`
  - 以及其他同尺度 `_qc` 子表。
- 正式深度模型输入已同步：
  - 不再把 `species_number` 放入正式 categorical embedding。
  - 使用 `latin_name + kingdom/phylum/class_name/tax_order/family/genus/species + taxon_group_l1/l2/l3 + organism_lifestage` 作为物种上下文。
  - 新增 `medium_domain + target_name` 的 target-scale/medium adapter，shared trunk 输出先经 adapter 残差校正，再进入 endpoint task head。
  - 新增 `no_medium_adapter` 消融项；`no_context` 和 `descriptors_only` 自动关闭 adapter。
  - `heads_embeddings` 微调模式现在会同时微调 task heads、categorical embeddings 和 adapters。
- 正式传统 baseline 输入已同步：
  - `species_number`、`publication_year`、`reference_numbers`、`publication_years`、`test_ids`、`result_ids` 等审计字段不进入特征。
  - taxon 层级字段保留为可用 categorical context。
- 已新增应用域报告模块：
  - `qsar_tl/evaluation/application_domain.py`
  - CLI：`python -m qsar_tl.cli build-ad-report --config configs/experiment.example.yaml --db outputs/derived/modeling_dataset.sqlite --source-table aggregated_task_records_soil_mg_kg_qc --split-name <split> --out outputs/ad/<split>.csv`
  - 输出 Williams leverage、Tanimoto 最大相似度、taxon distance、化学/物种/总体 in-domain 标记和 `ad_warning`。
- 已同步 `scripts/explain_deep_model.py`，解释训练好的 adapter 模型时会传入 adapter id，并把 `target_scale_medium_adapter` 纳入 permutation/SHAP 的候选上下文字段。
- 已更新 `configs/experiment.example.yaml`、`configs/experiment.remote.easyai.yaml` 和 `docs/config_schema.md`：
  - 正式推荐源表改为 `_qc` 子表。
  - 新增 QC 聚合、应用域和 `model.use_medium_adapters` 配置说明。
- 本轮验证：
  - `pytest tests/test_modeling_shapes.py tests/test_deep_experiment_cache.py tests/test_application_domain.py tests/test_baseline_models.py tests/test_qc_aggregation.py tests/test_medium_domain_tables.py tests/test_task_tables.py tests/test_task_mapping.py tests/test_splits.py`，48 passed。
  - `pytest tests`，61 passed。
  - `py_compile` 已覆盖新增/修改的核心 Python 模块和解释脚本。

## 2026-06-12 `_qc` 低土壤样本迁移远端运行

- 已扩展 `scripts/build_aquatic_soil_adaptation_split.py`：
  - 支持 `--soil-finetune-fraction`。
  - 支持同时生成 transfer split 和共享 test set 的 soil-only split。
  - 当前 paired split：
    - `M_qc_aquatic_to_soil_ptox_adapt_C_f10/f20/f50/f100`
    - `SoilPtoxQC_C_low_f10/f20/f50/f100`
- 已扩展 `python -m qsar_tl.cli build-molecule-feature-cache`：
  - 新增 `--source-table`，可直接为 `_qc` 源表构建分子特征缓存。
  - cache manifest 记录 `source_table`。
- 本地验证：
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m pytest tests -q`，63 passed。
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m compileall qsar_tl scripts tests` 通过。
  - `E:\TOOLS\anaconda\envs\qsar-ph3\python.exe -m qsar_tl.cli validate-config --config configs\experiment.remote.easyai.yaml` 通过。
- 远端 `_qc` 数据链已重建：
  - `task_records_qc` / `aggregated_task_records_qc` 已生成。
  - QC 后 `aggregated_task_records_qc`：120,753。
  - `aggregated_task_records_aquatic_ptox_qc`：89,471。
  - `aggregated_task_records_soil_ptox_qc`：2,847。
  - `aggregated_task_records_soil_mg_kg_qc`：4,999。
  - `aggregated_task_records_aquatic_soil_ptox_qc`：92,318。
- 远端已生成 `_qc` A-F split：
  - `AquaticPtoxQC_`
  - `SoilPtoxQC_`
  - `SoilMgkgQC_`
- paired low-soil split 规模：
  - f10：transfer finetune 239 / test 532；soil-only train 239 / test 532。
  - f20：transfer finetune 520 / test 532；soil-only train 520 / test 532。
  - f50：transfer finetune 1,179 / test 532；soil-only train 1,179 / test 532。
  - f100：transfer finetune 2,315 / test 532；soil-only train 2,315 / test 532。
- 远端验证：
  - `_qc` pTox 表目标量纲单一。
  - paired split 的 test set 完全一致。
  - `SoilPtoxQC_` 和 `SoilMgkgQC_` 的 C/F 化合物泄漏为 0。
  - 本机 RDKit 生成并同步 `outputs/features/molecular_features_rdkit_morgan512.jsonl`，覆盖 `_qc` 水相+土壤 pTox 5,523 个唯一 SMILES，fallback_count=0。
- 远端 smoke：
  - `SoilPtoxQC_B_random_8_2` 使用 `limit=2000`、1 epoch 通过；`limit=500` 被任务阈值保护拒绝，原因是所有 task head 均低于 `min_total=200`。
  - `M_qc_aquatic_to_soil_ptox_adapt_C_f10` 使用 1 epoch pretrain + 1 epoch finetune 通过。
- 已启动远端正式批处理：
  - job script：`outputs/logs/qc_low_soil_batch_job.sh`
  - log：`outputs/logs/qc_low_soil_batch_20260612.log`
  - job PID：`3108179`
  - 当前队列：f10/f20/f50/f100，每个比例先跑 transfer，再跑 soil-only。
  - transfer 输出目录：`outputs/experiments/qc_low_soil_transfer_epoch30_20_lr3e4`
  - soil-only 输出目录：`outputs/experiments/qc_low_soil_soilonly_epoch30`

## 2026-06-12 量纲审计与建模源表约束（当前窗口）

- 当前窗口目标已切换为：数据量纲统一、量纲审计、排除样本过少的量纲、禁止不同不可换算单位混用建模；后续正式实验另开窗口推进。
- 已新增 `scripts/audit_medium_retention_targeted.py`，用于一次性审计水相、沉积物、土壤在 `target_records -> task_records -> aggregated_task_records` 的保留数量、目标尺度、单位族、排除原因和介质子表规模。
- 最新远端审计报告已保存并拉回：`outputs/audits/medium_retention_targeted_v2.txt`。
- 当前建模源表选择规则：
  - 禁止直接使用混合目标尺度宽表建模，例如 `aggregated_task_records_aquatic`、`aggregated_task_records_soil` 只能用于审计或对照说明。
  - 正式建模必须使用同量纲子表，例如 `aggregated_task_records_aquatic_ptox`、`aggregated_task_records_soil_mg_kg`、`aggregated_task_records_soil_ptox`。
  - `qsar_tl.training.baseline.load_split_frame` 已加入保护：同一 source table/split 中若出现多个 `target_name`，传统基线和深度模型都会直接报错，防止不可换算量纲混用。
- 当前按 `minimum_modeling_rows=100` 的量纲状态：
  - 水相：`aquatic_ptox` 89,679、`aquatic_mg_kg` 11,564、`aquatic_percent` 770 为可建模；`aquatic_mg_per_organism` 34、`aquatic_mg_per_experimental_unit` 17 标记为样本不足。
  - 土壤：`soil_mg_kg` 5,171、`soil_ptox` 2,867、`soil_g_ha` 3,146、`soil_percent` 159 为可建模；`soil_l_ha` 48、`soil_seed` 13 标记为样本不足。
  - 沉积物：总聚合 36、`sediment_ptox` 34、`sediment_mg_kg` 1，均低于 100；不建议单独训练，只适合作为迁移外部观察或与水相同 pTox 尺度合并的参考。
- 已更新 `configs/experiment.remote.easyai.yaml` 和 `configs/experiment.example.yaml`：
  - 水相推荐源表：`aggregated_task_records_aquatic_ptox`。
  - 土壤推荐源表：`aggregated_task_records_soil_mg_kg`，并保留 `aggregated_task_records_soil_ptox` 作为 pTox 迁移参照。
  - 沉积物标记 `modeling_status: too_few_rows`，默认不推荐单独建模。
- 已在 A-F split 生成入口加入正式建模阈值保护：
  - `generate-experiment-splits` 在未指定 `--limit` 时读取 `targets.medium_units.minimum_modeling_rows`。
  - 远端已验证 `aggregated_task_records_sediment_ptox` 因 34 条低于 100 被拒绝。
  - 远端已验证 `aggregated_task_records_aquatic_ptox` 带 `--limit 200` 的 smoke split 可正常生成；临时 `GuardSmoke_` split 已清理。
- 已验证：
  - 本地：`pytest tests/test_baseline_models.py tests/test_splits.py tests/test_medium_domain_tables.py tests/test_unit_normalizer.py`，28 passed。
  - 本地/远端：`py_compile` 通过，远端 `configs/experiment.remote.easyai.yaml` 配置校验通过。

## 当前阶段

已完成第一版 `A -> B -> C` 起步交付：

1. 技术蓝图文档。
2. 配置文件规范。
3. 示例实验配置。
4. 第一版 Python 包骨架。
5. CLI 入口。
6. 本机/远程训练执行器接口。
7. PySide6 GUI 训练控制台空壳。

## 2026-06-12 介质筛选与单位 v2 校正进度

- 已删除目标值构建中“剂量组数不少于 3 才允许 min/max 中点”的要求；当前规则为优先 mean，mean 缺失且 min/max 完整时使用中点，并记录 `tox_value_source=min_max_midpoint`。
- 已加入原始单位 v2 编码，覆盖水相质量/摩尔浓度、土壤/沉积物 mg/kg、面积施用量、体积施用量、种子处理、百分比、按个体剂量和按实验单元剂量。
- 远端全量重建后，`included_targets` 从旧流程约 54.6 万提升到 1,040,655；`aggregated_task_records` 为 121,240。
- 介质聚合表最新规模：
  - 水相：`aggregated_task_records_aquatic` 102,652；水相主实验推荐使用 `aggregated_task_records_aquatic_ptox` 89,679。
  - 土壤：`aggregated_task_records_soil` 11,438；其中 `soil_mg_kg` 5,171、`soil_ptox` 2,867、`soil_g_ha` 3,146。
  - 沉积物：`target_records` 289、目标可用 209、任务映射纳入 69、聚合后 36；当前主要限制是数据库中明确沉积物证据记录本身少，不是单位编码导致的大量损失。
- 已补充介质分表：水相和沉积物按 `ptox`、`mg/kg`、百分比、按个体剂量、按实验单元剂量等目标尺度输出，避免后续把不同目标尺度混入同一主实验。
- 已新增审计脚本：
  - `scripts/audit_medium_domain_summary.py`
  - `scripts/audit_medium_sample_flow_fast.py`
- 已修正传统 ML baseline 特征选择，排除原始浓度、标准化浓度和 v2 单位元数据，避免目标/剂量泄漏。
- 后续介质实验入口建议：
  - “水相”主实验：`aggregated_task_records_aquatic_ptox`。
  - “水相 + 沉积物”可比 pTox：`aggregated_task_records_aquatic_sediment_ptox`。
  - “土壤”按问题选择 `aggregated_task_records_soil` 或同尺度子表；迁移实验优先用 `soil_ptox` 与水相 pTox 对齐。
  - “沉积物”不建议单独完整训练，可作为小样本外部验证或与水相合并的 pTox 参照。

## 2026-06-12 校正后远端重跑队列

- 已确认远端 `AquaticPtox_` 与 `AquaticSedimentPtox_` 各 14 个 A-F/fold split 均已生成，source table 分别为：
  - `aggregated_task_records_aquatic_ptox`，每个 split 89,679 条。
  - `aggregated_task_records_aquatic_sediment_ptox`，每个 split 89,713 条。
- 远端 smoke 已通过：
  - baseline smoke：`AquaticPtox_B_random_8_2` × 7 个传统模型 × `limit=500`。
  - deep smoke：`AquaticPtox_B_random_8_2` × `full` × 1 epoch × `limit=500`，`encoder_source=rdkit_cache`。
- 当前正在远端后台运行：
  - `baseline_aquatic_ptox_v2`，PID `442143`，日志 `outputs/logs/baseline_aquatic_ptox_v2_20260612_011447.log`，范围为 `AquaticPtox_` A-F/fold × 7 个传统模型 × `limit=10000`。
  - `deep_aquatic_sediment_ptox_full_epoch30`，PID `470111` 等待脚本已接上并开始运行，日志 `outputs/logs/deep_aquatic_sediment_ptox_full_epoch30_20260612_011951.log`，范围为 `AquaticSedimentPtox_` A-F/fold × `full` × 30 epoch × `limit=10000`。
- 当前已排队等待：
  - `baseline_aquatic_sediment_ptox_v2`，等待 PID `442143` 完成后启动，日志 `outputs/logs/baseline_aquatic_sediment_ptox_v2_20260612_011859.log`。
- 已拉回本机的轻量结果：
  - `outputs/experiments/baseline_aquatic_ptox_v2`
  - `outputs/logs/baseline_aquatic_ptox_v2_20260612_011447.log`
  - `outputs/logs/deep_aquatic_ptox_full_epoch30_20260612_011616.log`
- 已完成并汇总：
  - `deep_aquatic_ptox_full_epoch30`，14 个 A-F/fold split 均完成。
  - 汇总文件：`outputs/experiments/deep_aquatic_ptox_full_epoch30/summary_test_metrics.csv`、`summary_training_convergence.csv`、`figures/`。
  - 按 split family 的测试均值：A R2 0.6768；B R2 0.6470；C R2 0.4336；D R2 0.3608；E R2 0.5956；F R2 0.3070。随机/适应划分高于化合物严格划分，符合外推难度预期。
  - `deep_aquatic_sediment_ptox_full_epoch30`，14 个 A-F/fold split 均完成并汇总。
  - 按 split family 的测试均值：A R2 0.6580；B R2 0.6390；C R2 0.4555；D R2 0.3595；E R2 0.5995；F R2 0.2878。与水相单独结果接近，符合沉积物 pTox 仅新增 34 条样本的审计结论。
- 已发现并修正 XGBoost baseline 配置问题：
  - 远端 `xgboost 3.2.0` 的 `reg:pseudohubererror` 在 pTox 任务上输出约 49 的常数预测，导致 R2 极端负值，不能作为有效基线。
  - 已将 XGBoost 训练目标改为稳定的 `reg:squarederror`；Huber loss 仍作为统一评价指标输出。
  - 已启动 `baseline_aquatic_ptox_v2` 的 XGBoost 全 split 补跑，日志 `outputs/logs/baseline_aquatic_ptox_v2_xgboost_rerun_20260612_014328.log`。
- 当前仍在远端运行：
  - `baseline_soil_ptox_v2`，PID `888067`，日志 `outputs/logs/baseline_soil_ptox_v2_20260612_023008.log`，范围为 `SoilPtox_` A-F/fold × 7 个传统模型 × `limit=10000`。
  - `deep_soil_ptox_full_epoch30_v2`，PID `891131`，日志 `outputs/logs/deep_soil_ptox_full_epoch30_v2_20260612_023038.log`，范围为 `SoilPtox_` A-F/fold × `full` × 30 epoch × `limit=10000`。
  - `baseline_soil_mgkg_v2`，PID `896453`，等待 `baseline_soil_ptox_v2` 完成后自动运行，日志 `outputs/logs/baseline_soil_mgkg_v2_20260612_023125.log`。
  - `deep_soil_mgkg_full_epoch30_v2`，PID `899111`，等待 `deep_soil_ptox_full_epoch30_v2` 完成后自动运行，日志 `outputs/logs/deep_soil_mgkg_full_epoch30_v2_20260612_023151.log`。
- 已完成并重新汇总 baseline：
  - `baseline_aquatic_ptox_v2`：98/98 metrics，XGBoost 已按 `reg:squarederror` 重新补跑并覆盖旧结果。
  - `baseline_aquatic_sediment_ptox_v2`：98/98 metrics，XGBoost 已按 `reg:squarederror` 重新补跑并覆盖旧结果。
  - 已拉回本机：`outputs/experiments/baseline_aquatic_ptox_v2`、`outputs/experiments/baseline_aquatic_sediment_ptox_v2`。
- 已生成土壤相关 A-F split：
  - `Soil_`：source table `aggregated_task_records_soil`，总样本 11,438。
  - `SoilPtox_`：source table `aggregated_task_records_soil_ptox`，总样本 2,867。
  - `SoilMgkg_`：source table `aggregated_task_records_soil_mg_kg`，总样本 5,171。
- 已完成并汇总土壤 deep full：
  - `deep_soil_full_epoch30_v2`，14 个 A-F/fold split 均完成。
  - 汇总文件：`outputs/experiments/deep_soil_full_epoch30_v2/summary_test_metrics.csv`、`summary_training_convergence.csv`、`figures/`。
  - 按 split family 的测试均值：A R2 0.9184；B R2 0.9137；C R2 0.7726；D R2 0.7567；E R2 0.9079；F R2 0.7071。
  - 注意：`Soil_` 是广义土壤表，包含 mg/kg、g/ha、pTox、percent 等不同目标尺度；该结果适合作为广义土壤任务对照，但同尺度结论仍需 `SoilPtox_` 和 `SoilMgkg_`。
- 已完成并汇总土壤 baseline：
  - `baseline_soil_v2`，98/98 metrics 完成。
  - 已拉回本机：`outputs/experiments/baseline_soil_v2`。

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

## 远端训练迁移进度

- 已新增远端配置：`configs/experiment.remote.easyai.yaml`。
- 已补充 `execution.remote.port` 解析和 SSH 命令生成，支持非默认 SSH 端口。
- 已新增 PowerShell 远端流程脚本：
  - `scripts/check_remote_gpu.ps1`
  - `scripts/sync_to_server.ps1`
  - `scripts/train_remote.ps1`
  - `scripts/sync_from_server.ps1`
  - `scripts/remote_common.ps1`
- 已新增流程文档：`docs/remote_training_workflow.md`。
- 推荐长期同步策略：GitHub 管理代码和配置，`rsync` 管理大型 SQLite 数据和输出；当前脚本使用 Windows/OpenSSH `scp` 作为不依赖额外工具的基线方案。

## SSH 与实验计划进度

- 本机已有 `id_ed25519` 密钥对，公钥指纹：`SHA256:JDw1/1LbcUFpufq2B+la1Et3hsTUWzTGLeC60ULw6AU`。
- 已配置 SSH 免密别名：
  - `qsar-gpu` -> `i.easy-ai.cloud:32136`
  - `qsar-gpu-backup` -> `b.easy-ai.cloud:32136`
- 主/备用别名均已通过 `BatchMode=yes` 免密验证。
- 当前远端 GPU 状态异常：`nvidia-smi` 返回退出码 9，提示无法与 NVIDIA driver 通信。
- 当前远端 `/opt/anaconda3/bin/python` 是 Python 3.12.7，但未安装 `torch`。
- 已新增实验计划文档：`docs/experiment_plan_ablation_splits.md`。
- 已确认模型架构文档：`docs/model_architecture_agreed.md`。
- 已将 Morgan 指纹位数调整为 512 bit。
- 已新增 A-F 实验划分生成入口：`python -m qsar_tl.cli generate-experiment-splits`。
- 已新增传统机器学习 baseline matrix 入口：`python -m qsar_tl.cli run-baseline-matrix`。
- 已新增远端实验序列脚本：`scripts/run_remote_sequence.ps1`。
- 已在真实派生库 `outputs/derived/modeling_dataset.sqlite` 生成 A-F split；C/F/D 化合物严格划分泄漏检查为 0。
- 已完成本机小样本 baseline matrix 冒烟：RF、XGBoost、LightGBM、PLS、ExtraTrees、ElasticNet、MLP 均可调度。
- 已同步代码和 `outputs/derived/modeling_dataset.sqlite` 到远端 `/home/easyai/DL1/ecotox_qsar_transfer`。
- 远端 GPU 已恢复可用：RTX 4060 Ti，NVIDIA driver 575.57.08，CUDA 12.9。
- 已安装远端 PyTorch CUDA 版，验证结果：`torch 2.11.0+cu128`，`torch.cuda.is_available() == True`，GPU 数量 1。
- 远端 XGBoost、LightGBM、SHAP 安装因 PyPI 下载超时暂未完成；RF、ExtraTrees、PLS、ElasticNet、MLP 已可用于远端 CPU baseline。
- 已完成远端小样本 baseline matrix 冒烟：`B_random_8_2` + `random_forest/extra_trees/pls/elastic_net/mlp`，`limit=500`。
- 已完成远端训练入口冒烟：`scripts/train_remote.ps1 -SmokeTest`。当前 `qsar_tl.training.train` 仍为 placeholder，下一步需接入真实深度训练数据加载与多任务网络训练。
- 已将 `qsar_tl.training.train` 接入真实深度训练流程：
  - 从 `split_assignments` 和 `aggregated_task_records` 读取数据。
  - 按任务样本阈值过滤任务头。
  - 构建 512 bit 指纹、分子/上下文数值特征、物种/life stage/介质等 categorical embedding。
  - 输出 `history.csv`、`metrics.csv`、`predictions.csv`、`manifest.json`。
- 已完成本机深度 smoke：`B_random_8_2`，`limit=500`，`epochs=1`，使用 RDKit 编码。
- 已完成远端 CUDA 深度 smoke：`B_random_8_2`，`limit=500`，`epochs=1`，使用 `stable_smiles_fallback` 编码。
- 已完成远端 CUDA 深度 pilot：`B_random_8_2`，`limit=10000`，`epochs=5`，训练任务头 `ECx_Growth`、`ECx_Immobilization`、`ECx_Mortality`。
  - 训练 loss 从 `9.740133` 降至 `1.675708`。
  - 测试 R2：`ECx_Growth=0.249`，`ECx_Immobilization=0.452`，`ECx_Mortality=0.411`。
  - 输出目录：`outputs/experiments/deep_pilot_remote/deep/B_random_8_2`。
- 当前远端未安装 RDKit，正式 RDKit 描述符训练前需要安装 RDKit 或先在本机构建分子特征缓存并同步到远端。
- 已新增 RDKit 分子特征缓存流程：
  - CLI：`python -m qsar_tl.cli build-molecule-feature-cache`
  - 缓存文件：`outputs/features/molecular_features_rdkit_morgan512.jsonl`
  - 当前缓存包含 4,831 个唯一 SMILES，RDKit fallback 计数为 0。
  - 远端训练已能读取缓存，`encoder_source=rdkit_cache`。
- 已完成远端 RDKit-cache CUDA pilot：
  - `B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1`
  - 每个 split 使用 `limit=10000`、`epochs=5`、`batch_size=256`、`device=cuda:0`
  - 汇总文件：`outputs/experiments/deep_pilot_rdkit_cache_remote/summary_test_metrics.csv`
  - 结果显示从随机划分到化合物严格划分，测试/验证表现整体下降，符合新化合物外推难度上升预期。
- 已新增深度模型消融开关：
  - `full`
  - `no_fingerprint`
  - `no_duration`
  - `no_species_lifestage`
  - `no_context`
  - `no_descriptors`
  - `descriptors_only`
  - `no_molecular_residual`
- 已新增远端批量深度训练脚本：`scripts/train_remote_batch.ps1`。
- 已新增消融结果汇总脚本：`scripts/summarize_deep_ablation.py`，可输出指标表、相对 full 的 delta 表、训练收敛曲线和符合 `style_journal_clean_v1.yaml` 的 PNG/SVG 图。
- 已完成远端消融 pilot：
  - 目录：`outputs/experiments/deep_ablation_pilot_remote`
  - 范围：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1` × `full/no_fingerprint/no_duration/no_species_lifestage/no_context`
  - 参数：`limit=10000`、`epochs=5`、`batch_size=256`、`device=cuda:0`
  - 结论：5 epoch 只适合作为链路和代码验证，不足以解释模块贡献。
- 已完成远端 30 epoch 消融收敛检查：
  - 目录：`outputs/experiments/deep_ablation_epoch30_remote`
  - 范围同上，缺失组合已补跑完整。
  - 结论：相较 5 epoch 指标明显改善；结合用户后续判断，30 epoch 作为当前主实验训练长度更合理。
- 已完成远端 80 epoch 消融收敛检查：
  - 目录：`outputs/experiments/deep_ablation_epoch80_remote`
  - 汇总：`outputs/experiments/deep_ablation_epoch80_remote/summary_test_metrics.csv`
  - 收敛曲线：`outputs/experiments/deep_ablation_epoch80_remote/summary_training_convergence.csv`
  - 图表：`outputs/experiments/deep_ablation_epoch80_remote/figures`
  - 文档：`docs/deep_ablation_epoch80_summary.md`
  - 结论：80 epoch 比 30 epoch 训练 loss 仍有小幅下降，但部分迁移 split 的消融项后段 loss 有反弹；该结果仅作为收敛诊断，正式主实验仍按用户确认的约 30 epoch 策略执行，必要时再显式启用 validation/finetune early stopping。
- 用户随后确认：30 epoch 左右比较合理，后续实验按约 30 epoch 训练。
- 已将远端配置 `configs/experiment.remote.easyai.yaml` 的默认 `training.epochs` 调整为 `30`。
- 已将远端批量训练脚本 `scripts/train_remote_batch.ps1` 默认 `Epochs` 调整为 `30`。
- 已新增可选 early stopping 与 best epoch 输出：
  - 默认关闭，不影响固定 30 epoch 策略。
  - 训练输出会记录 `best_epoch`、`best_model.pt`、`history.csv` 中的 monitor 字段。
  - 如果后续需要验证集早停，可通过 `--early-stopping` 或 PowerShell `-EarlyStopping` 显式启用。
- 已完成 A-F 全划分 full 模型远端 30 epoch 训练：
  - 目录：`outputs/experiments/deep_full_af_epoch30_remote`
  - 范围：A、B、C、D 5 折、E 5 折、F
  - 模型：`full`
  - 参数：`limit=10000`、`epochs=30`、`batch_size=256`、`device=cuda:0`
  - 汇总：`outputs/experiments/deep_full_af_epoch30_remote/summary_test_metrics.csv`
  - 文档：`docs/deep_full_af_epoch30_summary.md`
  - 初步结果：随机 5 折 E 平均 R2 最高，化合物严格/迁移划分 C/D/F 明显更难，符合新化合物外推难度更高的预期。
- 已补齐 A/D/E 的 30 epoch 关键消融项，并与既有 B/C/F 消融结果合并：
  - 新消融目录：`outputs/experiments/deep_ablation_af_epoch30_remote`
  - 合并目录：`outputs/experiments/deep_ablation_af_combined_epoch30`
  - 合并脚本：`scripts/combine_deep_ablation_results.py`
  - 文档：`docs/deep_ablation_af_epoch30_summary.md`
  - 总组合数：70 个 split-ablation 组合。
  - 初步结果：
    - 512 bit Morgan fingerprint 对 A/B/E 随机或适应划分贡献最稳定。
    - 上下文模块与物种/life stage 在 C/D/F 化合物严格或迁移划分中贡献更明显。
    - 当前 duration 模块整体增益较小，需进一步做 raw/log/RBF 的细粒度消融。
- 已完成远端传统机器学习可用基线 A-F 矩阵：
  - 目录：`outputs/experiments/baseline_af_available_limit10000`
  - 范围：A、B、C、D 5 折、E 5 折、F
  - 模型：`random_forest`、`extra_trees`、`pls`、`elastic_net`、`mlp`
  - 暂缺：远端尚未安装 `xgboost`、`lightgbm`，需后续补跑。
  - 汇总脚本：`scripts/summarize_baseline_matrix.py`
  - 文档：`docs/baseline_af_available_summary.md`
  - 初步结果：Random Forest 是当前可用传统模型中的最强基线，但 A-F 全部划分族上均低于 30 epoch deep full；C/D/F 严格化合物或迁移划分差距最大。
- 已补齐远端 boosting 传统基线：
  - 已安装并验证：`xgboost 3.2.0`、`lightgbm 4.6.0`、`shap 0.48.0`、`numpy 1.26.4`。
  - 目录：`outputs/experiments/baseline_af_boosting_limit10000`
  - 合并目录：`outputs/experiments/baseline_af_all_models_limit10000`
  - 文档：`docs/baseline_af_all_models_summary.md`
  - 结果：7 个传统模型、14 个 split/fold、630 行指标已汇总；最强传统基线在 A-F 全部划分族上仍低于 30 epoch deep full。
  - 修复：LightGBM 不接受特殊字符特征名，已在 `qsar_tl/training/baseline.py` 中加入安全特征名映射。
- 已完成深度模型学习率敏感性测试：
  - 目录：`outputs/experiments/deep_lr_sensitivity`
  - 范围：`B_random_8_2`、`C_chemical_holdout_8_2`、`F_chemical_adapt_7_2_1` × `full`
  - 学习率：`1e-4`、`3e-4`、`5e-4`、`1e-3`
  - 参数：`limit=10000`、`epochs=30`、`batch_size=256`、`device=cuda:0`
  - 汇总脚本：`scripts/summarize_learning_rate_sweep.py`
  - 文档：`docs/deep_learning_rate_sensitivity_summary.md`
  - 结论：`1e-4` 明显偏小；`5e-4` 在 B/C/F 总体均值上最佳，已将远端配置默认 `training.learning_rate` 调整为 `0.0005`；C-only 严格化合物外推可保留 `0.0003` 作为保守对照。
- 已启动介质迁移实验：
  - 已扩展 `medium_transfer_split` 的介质代码识别，支持 FW/SW/AQU 等水相代码和 ART/NAT/UKS/LIT/MIN 等固相代码。
  - `generate-split` 已支持 `--medium-train-domains`、`--medium-test-domains`、`--medium-unknown-part` 参数。
  - 已生成并运行 `M_water_to_solid` full 深度模型：
    - split 规模：train 73,164；test 1,548；valid 6,013
    - 目录：`outputs/experiments/deep_medium_transfer_water_to_solid`
    - 参数：`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`
    - 文档：`docs/deep_medium_transfer_water_to_solid_summary.md`
  - 初步结果：测试集 R2 大幅为负，主要风险是水相 `pTox mol/L` 与固相 `-log10 mg/kg` 目标尺度不等价；该结果应视为迁移链路验证和目标定义风险提示，不能直接作为模型最终迁移能力结论。
- 已修复介质域表与远端 SQLite 同步问题：
  - `aggregated_task_records` 聚合表已保留 `primary_medium`、`habitat_labels`、`organism_habitat`。
  - 已新增介质子表构建脚本：`scripts/build_medium_domain_tables.py`。
  - 当前介质子表规模：水相 76,133；水相+沉积物 76,163；土壤 3,125；沉积物 30；固相 3,155。
  - `scripts/sync_to_server.ps1` 已在上传 SQLite 前生成稳定 backup 快照，避免 WAL 模式下只传主库导致远端 `malformed database schema`。
- 已完成迁移前介质来源对比的 full 模型 30 epoch 训练：
  - 目录：`outputs/experiments/deep_medium_pretransfer_epoch30`
  - 范围：水相、水相+沉积物、土壤、固相 × B 随机 8:2 / C 化合物严格 8:2。
  - 参数：`limit=10000`、`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`。
  - 文档：`docs/deep_medium_pretransfer_epoch30_summary.md`
  - 初步结果：水相 B 平均 R2 约 0.669，水相 C 约 0.486；水相+沉积物与水相接近，因沉积物样本仅 30 条；土壤/固相 R2 较高但测试样本量较低且任务头结构不同，不能直接和水相均值做强结论比较。
- 已补齐深度模型解释分析链路：
  - 深度训练输出已新增 `preprocessing.json`，记录数值特征名、标准化参数、类别映射、指纹维度和消融配置。
  - 已新增解释脚本：`scripts/explain_deep_model.py`。
  - 解释输出包括 permutation importance、暴露时长 PDP、小样本 SHAP permutation explainer，并将 512 个 Morgan 指纹位聚合为一个可读的指纹组。
  - 已完成首轮正式解释：`Aquatic_C_chemical_holdout_8_2` + `ECx_Mortality`。
  - 输出目录：`outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Aquatic_C_ECx_Mortality`
  - 初步解释：Morgan 指纹组贡献最大；`MolWt`、`MolLogP`、`latin_name`、`species_number` 和 `duration_bin_h` 均进入重要特征前列；暴露时长 PDP 呈现先升高后平台/回落的非线性趋势，符合暴露-毒性关系可能存在平台期的模型假设。
  - 已扩展解释到土壤/固相 C 划分 `ECx_Mortality`：
    - `outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Soil_C_ECx_Mortality`
    - `outputs/experiments/deep_medium_pretransfer_epoch30/explainability/Solid_C_ECx_Mortality`
  - 跨介质解释对比显示：水相更明显依赖分子结构和 RDKit 描述符；土壤/固相中 `target_basis`、物种、life stage、介质和短时长 RBF 贡献更突出。土壤/固相暴露时长 PDP 与水相方向相反，提示目标单位、介质实验设计和样本组成差异可能强烈影响迁移解释。
- 已完成同目标尺度的水相到土壤 pTox 正式迁移实验：
  - 已新增 pTox 可比子表：
    - `aggregated_task_records_aquatic_soil_ptox`: 77,859
    - `aggregated_task_records_soil_ptox`: 1,884
    - `aggregated_task_records_solid_ptox`: 1,914
  - 已修复 `medium_transfer_split`：存在审计后的 `medium_domain` 字段时优先使用该字段，避免 `primary_medium/media_type` 冲突导致水相行进入土壤 test。
  - split：`M_aquatic_to_soil_ptox`
    - train：水相 `ptox_mol_l` 75,975
    - test：土壤 `ptox_mol_l` 1,884
  - 训练目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30`
  - 参数：`epochs=30`、`learning_rate=0.0005`、`batch_size=512`、`device=cuda:0`、不使用 `limit`。
  - 测试任务头平均 R2 为 -0.1027；`ECx_Mortality` R2 为 -0.4079。
  - 已新增土壤 pTox 内部参照：`outputs/experiments/deep_soil_ptox_reference_epoch30`
    - `SoilPtox_B_random_8_2` 平均 R2 约 0.4516。
    - `SoilPtox_C_chemical_holdout_8_2` 平均 R2 约 -0.1120。
  - 已完成迁移模型 `ECx_Mortality` 的 SHAP/PDP/Permutation 解释：
    - 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_epoch30/explainability/M_aquatic_to_soil_ptox_ECx_Mortality`
    - 解释显示迁移模型主要依赖 Morgan 指纹、`MolWt`、`HeavyAtomCount`、`MolLogP` 等分子结构特征，土壤上下文校正较弱，提示水相训练偏置明显。
  - 文档：`docs/deep_transfer_aquatic_to_soil_ptox_summary.md`
- 已新增 aquatic pretrain + soil finetune 二阶段训练能力：
  - `qsar_tl.training.train` 新增 `--finetune-epochs`、`--finetune-learning-rate`、`--finetune-batch-size`。
  - `scripts/train_remote.ps1` 和 `scripts/train_remote_batch.ps1` 已支持对应远端参数。
  - 已新增 split 构建脚本：`scripts/build_aquatic_soil_adaptation_split.py`。
  - adaptation split：`M_aquatic_to_soil_ptox_adapt_C`
    - train：水相 pTox 75,975
    - finetune：土壤 pTox 1,589
    - test：土壤 pTox 295
  - 已完成同一 test split 的 zero-shot C 对照：
    - 目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_adaptC_zeroshot_epoch30`
    - 共同任务 `ECx_Growth/ECx_Mortality` 平均 R2：-3.0625
  - 已完成 pretrain 30 epoch + finetune 10 epoch：
    - 目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10`
    - finetune 参数：`epochs=10`、`learning_rate=0.0001`、`batch_size=512`
    - 共同任务平均 R2：-0.4047
    - 相比 zero-shot C 明显改善，但仍弱于 soil-only C reference 平均 R2 -0.1120。
  - 已完成 finetune 模型 `ECx_Mortality` 的 SHAP/PDP/Permutation 解释：
    - 输出目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10/explainability/M_aquatic_to_soil_ptox_adapt_C_ECx_Mortality`
    - 微调后 `species_number`、`latin_name`、`media_type` 进入 SHAP 前列，但 Permutation 仍以 `MolWt`、`HeavyAtomCount` 和 Morgan 指纹组为主，说明微调开始利用土壤上下文但分子结构偏置仍强。
- 已完成 aquatic pretrain + soil finetune 学习率/步数诊断：
  - 诊断目录：
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_10_lr3e4`
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20`
    - `outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_epoch30_20_lr3e4`
  - 汇总目录：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity`
  - 图表：`outputs/experiments/deep_transfer_aquatic_to_soil_ptox_finetune_sensitivity/finetune_sensitivity_common_tasks.png`
  - 文档：`docs/deep_transfer_aquatic_to_soil_ptox_summary.md`
  - 共同任务 `ECx_Growth/ECx_Mortality` 上，`full finetune 3e-4/20` 当前最佳：
    - 平均 R2：0.0833
    - 平均 RMSE：1.5170
    - 平均 MAE：1.2083
    - 平均 Huber loss：0.7966
  - 对照：`soil-only C reference` 平均 R2 为 -0.1120，平均 Huber loss 为 0.8623；`zero-shot C` 平均 R2 为 -3.0625，平均 Huber loss 为 2.1831。
  - 结论：原 `1e-4/10` 微调学习率和步数均偏保守；`3e-4/20` 更适合作为当前二阶段迁移主配置。冻结到 `heads_embeddings` 明显欠拟合，不建议作为主策略。
  - 风险提示：本轮诊断使用同一土壤 holdout 比较超参数，正式结论应再配独立验证划分或嵌套验证，避免测试集调参偏乐观。

## 2026-06-22 工作区与远端保守整理

- 整理原则：本次只做保守归档，不直接删除实验结果或派生数据库；归档目录可回溯。
- GitHub 备份：
  - 分支：`codex/workspace-cleanup-20260622`
  - 整理前源码/文档安全提交：`bc3040c Record QSAR cleanup baseline and project updates`
  - 已新增 `.gitattributes`，固定文本换行策略，减少 Windows LF/CRLF 噪声。
- 本地整理：
  - 清单文档：`docs/cleanup_inventory_20260622.md`
  - 本地归档目录：`outputs/_archive/cleanup_20260622/`
  - 已移动 17 项旧中间文件/目录，包括旧派生库、smoke/debug 目录、legacy v1.0/v1.1/v1.2.0 原始实验目录。
  - 当前主派生库仍保留在 `outputs/derived/modeling_dataset_v2_0_0_rebuild.sqlite`。
  - 整理后 `outputs/derived` 约 6.210 GB，`outputs/experiments` 约 0.319 GB，`outputs/_archive/cleanup_20260622` 约 5.954 GB。
- 远端整理：
  - 远端项目目录：`/home/easyai/DL1/ecotox_qsar_transfer`
  - 整理前确认远端不是 Git 工作树，代码已通过 `scripts/sync_to_server.ps1` 从本地同步，不重传大 SQLite。
  - 远端归档目录：`/home/easyai/DL1/ecotox_qsar_transfer/outputs/_archive/cleanup_20260622/`
  - 已移动 19 项旧中间文件/目录；保留当前 v2 主结果、v1.2.2-v1.2.4 当前结果、features、logs、audits 和应用域结果。
  - 远端 `outputs/derived` 仅保留 `modeling_dataset_v2_0_0_rebuild.sqlite`，约 7.7 GB；`outputs/experiments` 约 3.4 GB；远端归档约 13 GB。
- 核验结果：
  - 本地测试：`E:\TOOLS\anaconda\python.exe -m pytest` 通过，95 passed、1 skipped、1 warning。
  - 配置核验：`E:\TOOLS\anaconda\python.exe -m qsar_tl.cli validate-config --config configs/experiment.remote.easyai.yaml` 通过。
- 远端训练进程检查未发现实际训练进程。

## 2026-07-19 v1.2.40 paired result and v1.2.41 stage-3 optimization

- v1.2.40 paired mass/molar matrix completed 16/16 runs on remote
  `qsar-gpu-new` using seeds `42, 2042, 3407, 8417`.
- Three-stage `X0_molar` four-seed mean native test R2 is `0.6973`.
  Back-converted common mg/kg R2 is `0.6654` and MAE is `0.5633`.
  The native mol/kg R2 is a valid result; the two R2 values differ because the
  per-row `3 + log10(MW)` offset changes target variance, while paired residuals,
  MAE, and RMSE remain unchanged.
- Random 8:2 is the primary evaluation boundary for this heterogeneous
  chemical-species-context model. Scaffold/similarity-cluster holdout is kept
  as a separate structural-family boundary rather than a prerequisite for the
  primary result.
- v1.2.22 and the current three-stage first two stages are not protocol
  equivalent: stage-2 epochs, learning rate, scheduler, exact-source routing,
  and target-head/data-table contracts changed. Therefore v1.2.41 retains the
  current four seeds instead of switching to the old v1.2.22 five-seed set.
- v1.2.41 screening seeds are locked to the top two fixed-validation baselines:
  `3407` and `42`. Final seed set is `42, 2042, 3407, 8417`.
- Matrix cells:
  - S1: stage3 40 epochs, full unfreeze, head LR `5e-4`, trunk LR `1e-4`, cosine.
  - S2: S1 plus stage3-only SWA over epochs 31-40.
  - S3: S2 plus standardized-target `0.7 Huber + 0.3 MSE`.
- Stage3 early stopping is disabled to preserve the fixed 40-epoch comparison.
  Formal validation fails closed unless S0/challenger validation `n`, aggregate
  hash, and result-id hash are identical. The first selection pass writes only
  validation metrics; test metrics are report-only after winner lock.
- Local targeted QA: `70 passed`; remote targeted QA passed with one
  environment-conditional skip. Remote smoke completed in about 6m52s.
- Formal run started at `2026-07-19T07:11:50+08:00`, two concurrent S1 runs.
  Controller log: `outputs/logs/v1_2_41_three_stage_optimization_matrix.log`.
  Expected completion is approximately 2-2.5 hours after launch.
