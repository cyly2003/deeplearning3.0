# CST-AD 应用域方法草案

更新时间：2026-07-02

## 1. 方法定位

CST-AD 是 Chemical-Species-Task Applicability Domain 的缩写，用于当前多物种土壤生态毒性 QSAR 模型的样本级可靠性评估。

传统 QSAR 应用域通常关注待预测化合物是否落在训练集化学结构空间内。但本项目的预测对象不是单纯的“化合物 -> 毒性值”，而是“化合物结构 + 物种分类学信息 + 毒性终点/效应类型 -> pTox”。因此，应用域也应从化学空间扩展到物种空间和终点空间。

第一版 CST-AD 不改变训练流程，只读取当前固定 test 的 ensemble prediction rows。这样可以先回答一个直接问题：模型误差是否集中出现在化合物外推、物种外推、终点覆盖不足或 ensemble 不确定性较高的样本中。

2026-07-02 后，随机 8:2 与随机 5-fold 主线采用一个更窄的版本：Chemical + Species + Task AD，不使用 seed-ensemble 标准差作为分级依据。这是为了避免把“多 seed 集成的经验不确定性”和“应用域外推类型”混在同一个层级解释中。

## 2. Taxonomy embedding 与 AD 的关系

当前模型中 taxonomy/context embedding 的实现是分层 embedding，而不是整条 taxonomy path 合成一个 embedding。

- `kingdom`
- `phylum`
- `class_name`
- `tax_order`
- `family`
- `genus`
- `species`
- `primary_medium`
- `habitat_labels`
- `organism_habitat`
- `media_type`
- `organism_lifestage`
- `target_basis`
- `effect_family`

这些字段分别建立 embedding，再与化合物描述符和 fingerprint 拼接进入共享 MLP trunk。

当前建议保留这种设计。原因是生态毒性数据中很多物种、属、科的样本数并不均衡。如果把整条 taxonomy path 合成一个类别，模型可能更容易记住常见物种组合，但对少见物种、新物种和缺失分类层级的泛化会更弱。分层 embedding 可以让模型在粗层级共享信息，同时保留 genus/species 层面的细粒度差异。

应用域分析不直接使用 learned embedding 距离作为主指标。第一版先使用可解释的 taxonomy prefix similarity。这样更适合写进论文方法部分，也更容易解释给环境毒理和生态风险评估读者。

当前 learned species embedding 可以作为补充诊断，但不能单独作为正式 AD 阈值。v1.2.23 中，已见物种 S0/S1 的最近邻 embedding cosine distance 接近 0，属/科/目层级外推时距离升高；但 embedding distance 与 absolute error 的秩相关约为 0.018，说明它更适合作为几何覆盖说明，而不是误差预测器。

## 3. CST-AD 四个维度

### 3.1 Chemical AD

主指标：

- `ad_max_tanimoto_to_train`

解释：待预测化合物与训练集中最近化合物的 Morgan fingerprint Tanimoto 相似度。值越高，说明化合物结构越接近训练集覆盖范围。

辅助指标：

- `ad_williams_leverage`
- `ad_williams_critical_h`
- `ad_proxy_distance_to_train`

这些指标保留为补充诊断，不作为第一版 CST-AD 分级的唯一依据。

### 3.2 Species AD

主指标：

- `ad_max_taxon_similarity_to_train`

解释：待预测物种与训练集中物种在 taxonomy 层级上的最大共享程度。当前已有审计脚本默认使用 `kingdom/phylum/class_name/tax_order/family`，后续可扩展到 `genus/species`。

辅助标记：

- `ad_species_seen_train`
- `ad_genus_seen_train`
- `ad_family_seen_train`
- `ad_order_seen_train`
- `ad_life_stage_seen_train`
- `ad_species_task_family_seen_train`

其中 `ad_species_task_family_seen_train` 是目前很有解释价值的标记：它表示该物种和该毒性终点家族的组合是否在训练集中出现过。

### 3.3 Task AD

第一版不对 ECx、LOEC、NOEC 或具体 effect 进行训练加权，只作为可靠性解释层。

使用字段：

- `task_head`
- `task_family`
- `effect_family`
- `target_basis`
- `ad_species_task_family_seen_train`

后续如果要增强 Task AD，可在重新运行 AD audit 时增加 `task_head_seen_train`、`task_family_seen_train` 和 `effect_family_seen_train`。

### 3.4 Uncertainty AD

当前 v1.2.12 anchor ensemble prediction rows 已保留：

- `y_pred_member_1`
- `y_pred_member_2`
- `y_pred_member_3`
- `y_pred_member_4`
- `y_pred_member_5`

第一版 CST-AD 用这些列计算：

- `cst_ensemble_sd`

解释：5 个 seed 模型在同一样本上的预测标准差。值越高，说明模型对该样本预测不稳定。该指标不等价于严格校准后的预测区间，但可以作为后续 conformal prediction 或经验 prediction interval 的前置诊断。

注意：v1.2.23 随机主线 CST-AD 明确禁用这一维度。当前随机 8:2/5-fold 报告只使用 Chemical、Species 和 Task 三类应用域信息，`uncertainty_tier` 在 manifest 中记录为 `disabled`。

## 4. AD 分级规则

第一版使用数据驱动阈值：

- `cst_chemical_threshold`：`cst_chemical_score` 的 P10，除非命令行手动指定。
- `cst_species_threshold`：`cst_species_score` 的 P10，除非命令行手动指定。
- `cst_uncertainty_threshold`：`cst_ensemble_sd` 的 P90，除非命令行手动指定。

分级规则：

| Tier | 判定逻辑 | 科研解释 |
| --- | --- | --- |
| AD-A | 化合物域内、物种域内、species-task-family 已见、uncertainty 不高 | 最可信的插值预测 |
| AD-B | 化合物和物种均域内，但 species-task-family 未见 | 结构和分类学相似，但终点组合存在一定外推 |
| AD-C | 只有化合物或物种一侧外推，或 ensemble SD 较高 | 单侧外推或不确定性偏高 |
| AD-D | 化合物和物种同时外推，或外推且 ensemble SD 高 | 低可信预测，应作为风险提示 |

这套规则的重点不是人为制造性能提升，而是把样本级误差解释为可追溯的外推类型。

### 4.1 v1.2.23 随机主线分级规则

当前随机主线脚本为 `scripts/build_random_mainline_cst_ad.py`，输出目录为 `outputs/experiments/v1_2_23_random_mainline_cst_ad`。

- Chemical tier：`C0` 表示最大 Tanimoto >= 0.7；`C1` 表示最大 Tanimoto >= 0.5 或 Williams leverage 域内；`C2` 表示低相似度或缺失。
- Species tier：`S0` 表示 species-task-family 已见；`S1` 表示物种已见；`S2` 表示 genus/family 已见；`S3` 表示 order 已见或 taxonomy similarity >= 0.8；`S4` 表示低覆盖。
- Task tier：`T0` 表示 chemical-species-task exact seen；`T1` 表示 species-task head/family 已见；`T2` 表示只有 task head/family 覆盖；`T3` 表示 task 未覆盖。
- 总 tier：C+S+T 同时强覆盖为 `AD-A`；中等覆盖为 `AD-B`；单一主要风险为 `AD-C`；两个及以上主要风险为 `AD-D`。

本规则不使用 `y_pred_member_*`、`cst_ensemble_sd` 或 seed 间标准差。

## 5. 推荐图表

### 图 1：CST-AD 化合物-物种空间图

数据矩阵：

`outputs/experiments/v1_2_14_cst_ad_initial/cst_ad_prediction_rows.csv`

作图字段：

- x：`cst_chemical_score`
- y：`cst_species_score`
- color：`abs_error`
- facet：`task_family`，即 ECx、LOEC、NOEC 分开绘制
- 辅助线：`cst_chemical_threshold` 与 `cst_species_threshold`

科研解释：ECx、LOEC 和 NOEC 的观测机制、数据噪声和风险含义不同，应作为主图分面展示。如果高误差点在某一 endpoint family 内集中于低化学相似度或低物种相似度区域，说明传统化学 AD 与物种 taxonomy AD 能共同解释该终点类型的外推风险。总体混合图只能作为附图或方法示意，不应作为主要结论图。

### 图 2：AD tier 性能分层图

数据矩阵：

`outputs/experiments/v1_2_14_cst_ad_initial/cst_ad_tier_summary.csv`

作图字段：

- x：`cst_ad_tier`
- y1：`mae`
- y2：`coverage_fraction`

科研解释：如果 AD-A 的 MAE 低于 AD-C/AD-D，说明 CST-AD 对可靠预测有分层能力。

### 图 3：真实值-预测值图

数据矩阵：

`outputs/experiments/v1_2_14_cst_ad_initial/cst_ad_prediction_rows.csv`

作图字段：

- x：`y_true`
- y：`y_pred`
- color：`cst_ad_tier`

科研解释：观察不同 AD tier 的误差是否系统性偏离 1:1 线。

### 图 4：ensemble 不确定性-误差关系图

数据矩阵：

`outputs/experiments/v1_2_14_cst_ad_initial/cst_ad_prediction_rows.csv`

作图字段：

- x：`cst_ensemble_sd`
- y：`abs_error`
- color：`cst_ad_tier`

科研解释：如果 ensemble SD 与 absolute error 呈正相关，后续可以进一步构建 prediction interval 或 conformal prediction。

## 6. 输入与输出

初版轻量输入：

`outputs/experiments/v1_2_12_f100_5seed_confirmation_remote_summary/anchor_tanimoto_a1_5seed_ensemble_ad_prediction_rows.csv`

推荐完整输入同时传入 train reference：

```powershell
E:\TOOLS\anaconda\python.exe scripts\build_cst_ad_matrix.py `
  --prediction-rows outputs\experiments\v1_2_12_f100_5seed_confirmation_remote_summary\anchor_tanimoto_a1_5seed_ensemble_ad_prediction_rows.csv `
  --db outputs\derived\modeling_dataset_v2_0_0_rebuild.sqlite `
  --source-table aggregated_task_records_aquatic_soil_ptox_qc `
  --split-name M_v2_aquatic_to_soil_ptox_adapt_C_f100 `
  --chemical-threshold 0.5 `
  --species-threshold 0.8 `
  --out-dir outputs\experiments\v1_2_14_cst_ad_initial

E:\TOOLS\anaconda\python.exe scripts\plot_cst_ad.py `
  --cst-ad-rows outputs\experiments\v1_2_14_cst_ad_initial\cst_ad_prediction_rows.csv `
  --tier-summary outputs\experiments\v1_2_14_cst_ad_initial\cst_ad_tier_summary.csv `
  --out-dir outputs\figures\cst_ad_20260625
```

输出：

- `cst_ad_prediction_rows.csv`
- `cst_ad_tier_summary.csv`
- `cst_ad_family_summary.csv`
- `cst_ad_failure_cases.csv`
- `cst_ad_manifest.json`
- `cst_ad_chemical_species_space.png/.svg`
- `cst_ad_tier_performance.png/.svg`
- `cst_ad_observed_vs_predicted.png/.svg`
- `cst_ad_uncertainty_error.png/.svg`
- `cst_ad_chemical_species_space_by_task_family.png/.svg`
- `cst_ad_chemical_species_space_ecx.png/.svg`
- `cst_ad_chemical_species_space_loec.png/.svg`
- `cst_ad_chemical_species_space_noec.png/.svg`
- `cst_ad_tier_performance_by_task_family.png/.svg`

如果暂时不想读取 SQLite，或本地 SQLite 不含 `split_assignments`，可省略 `--db/--source-table/--split-name`。此时脚本仍会生成 Chemical AD、Species AD、ensemble uncertainty 和基础 `ad_species_task_family_seen_train` 分级，但不会补充 `cst_train_n_*` 组合计数。

当前本地快速出图采用轻量模式和既有 AD audit 阈值：chemical Tanimoto `0.5`，taxonomy similarity `0.8`。原因是本地派生库副本暂时无法读取 split assignments，而当前 test 集中大量化合物最近邻 Tanimoto 为 `1.0`，若完全采用 P10 自动阈值会得到过严的 chemical threshold。

v1.2.23 随机主线命令：

```powershell
E:\TOOLS\anaconda\python.exe scripts\build_random_mainline_cst_ad.py `
  --out-dir outputs\experiments\v1_2_23_random_mainline_cst_ad

E:\TOOLS\anaconda\python.exe scripts\plot_cst_ad.py `
  --cst-ad-rows outputs\experiments\v1_2_23_random_mainline_cst_ad\random_mainline_cst_ad_prediction_rows.csv `
  --tier-summary outputs\experiments\v1_2_23_random_mainline_cst_ad\cst_ad_tier_summary.csv `
  --out-dir outputs\figures\v1_2_23_random_mainline_cst_ad `
  --prefix random_mainline_cst_ad
```

v1.2.23 直接读取 `outputs/experiments/v1_2_15_random_split_policy_formal_remote_summary` 中已经拉回本地的远端预测结果，不重新训练。参考空间在内存中重建为 matching random split 的 aquatic source train + soil target finetune pool，排除 test fold。

关键结果：

| split policy | tier | n | coverage | MAE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random 8:2 | AD-A | 3164 | 96.49% | 0.5896 | 0.7871 |
| random 8:2 | AD-B | 114 | 3.48% | 0.6400 | 0.7658 |
| random 8:2 | AD-C | 1 | 0.03% | 1.6373 |  |
| random 5-fold | AD-A | 15648 | 96.26% | 0.5904 | 0.7914 |
| random 5-fold | AD-B | 585 | 3.60% | 0.6832 | 0.6795 |
| random 5-fold | AD-C | 23 | 0.14% | 0.5220 | 0.4961 |

解释：随机划分下大多数样本处于训练覆盖范围内，因此 CST-AD 的主要作用是证明当前随机主线确实是同分布插值评估，并识别少量中等覆盖样本。高错误检测 AUROC 约 0.50，说明在随机插值条件下，CST-AD 不是强二分类误差探测器。

## 7. 预期论文表述

本研究提出一种面向多物种生态毒性 QSAR 的联合应用域评价框架 CST-AD。该框架继承传统 QSAR 中基于化学结构相似性的应用域思想，同时扩展到物种分类学相似性、毒性终点覆盖和模型预测不确定性。CST-AD 不仅判断预测是否位于化学结构空间的插值区域，还进一步区分预测误差可能来源于化合物外推、物种外推、终点组合外推或模型随机性。该框架为多物种土壤毒性预测提供了比传统单一化学 AD 更贴近生态毒理场景的可靠性解释。

## 8. 当前边界

- 第一版 CST-AD 不改变训练集、test 集或模型结构。
- 第一版不处理 censored loss，只保留 censored 样本分布分析作为下一阶段。
- 第一版不声称 prediction interval 已经校准；ensemble SD 只是经验不确定性指标。
- v1.2.23 随机主线版本不使用 ensemble SD。
- CAS 号 chemical holdout 不应继续作为当前“科学化学外推”依据；如果要建立真正新化合物外推，应补 scaffold/cluster/Tanimoto-distance split。
- 如果 AD-A 与 AD-C/AD-D 的误差没有明显分层差异，则 CST-AD 仍可作为透明报告工具，但不能作为强方法创新主张。
