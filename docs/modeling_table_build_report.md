# Modeling Table Build Report

更新时间：2026-06-10

## 输入与输出

- Source database: `ecotox_clean.sqlite`
- Output database: `outputs/derived/modeling_dataset.sqlite`
- Output tables:
  - `wide_records`
  - `target_records`
  - `build_manifest`

`outputs/derived/modeling_dataset.sqlite` 是本地派生数据，不提交到 GitHub。

## 完整构建统计

| Item | Count |
|---|---:|
| wide_records | 1,234,077 |
| target_records | 1,234,077 |
| included_targets | 510,934 |
| excluded_targets | 723,143 |

## 目标变量纳入情况

| Status | Target | Reason | Count |
|---|---|---|---:|
| excluded |  | missing_mean_and_invalid_min_max_midpoint | 717,534 |
| included | ptox_mol_l |  | 474,929 |
| included | neg_log10_mg_kg |  | 27,801 |
| included | neg_log10_mg_kg_bw_day |  | 8,204 |
| excluded |  | missing_or_invalid_molecular_weight_for_mg_l_to_mol_l | 5,558 |
| excluded |  | non_positive_toxicity_value | 51 |

## 解释

第一版构建严格遵守当前规则：

- 优先使用标准化 mean 浓度。
- mean 缺失时，只有 `num_doses_mean >= 3` 且 min/max 完整时才使用 min/max 中点。
- 水相 `mg/L` 通过分子量换算为 `mol/L` 后生成 `ptox_mol_l`。
- 水相 `mol/L` 直接生成 `ptox_mol_l`。
- `mg/kg` 生成 `neg_log10_mg_kg`，用于土壤/沉积物任务。
- `mg/kg/d` 生成 `neg_log10_mg_kg_bw_day`，用于经口辅助任务。
- 其他单位、非法浓度、分子量缺失等情况进入排除原因。

当前排除量较大，主要原因是 mean/min/max 规则不满足。这不一定代表数据没有价值，而是说明第一版主模型入口较严格。后续可以在敏感性分析中评估是否纳入 NOEL/LOEL、范围值、删失值或其他单位体系。

