# Scaffold/Cluster Holdout Split Method Note

Date: 2026-07-02

## Purpose

The previous `C_chemical_holdout_8_2` split used CAS number grouping. This is
identifier-disjoint, but it is not a chemically meaningful new-structure
extrapolation test. The new v1.2.24 split uses structure-derived groups so that
same parent structure, same Murcko scaffold, and high-fingerprint-similarity
chemicals do not cross train/test boundaries.

## Literature Basis

- Bemis and Murcko introduced the molecular framework/scaffold concept that is
  widely used to group molecules by ring systems and linkers:
  https://doi.org/10.1021/jm9602928
- RDKit implements Murcko scaffold extraction through
  `rdkit.Chem.Scaffolds.MurckoScaffold`, including scaffold generation and
  optional generic scaffold conversion:
  https://www.rdkit.org/docs/source/rdkit.Chem.Scaffolds.MurckoScaffold.html
- Rogers and Hahn described extended-connectivity fingerprints (ECFP), the
  circular fingerprint family underlying Morgan fingerprints:
  https://doi.org/10.1021/ci100050t
- Butina clustering is a standard similarity-clustering method for molecular
  fingerprints; RDKit documents its implementation as based on Butina JCICS
  39:747-750 (1999):
  https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html
- MoleculeNet popularized scaffold split as a standard molecular machine
  learning benchmark split, while also emphasizing benchmark consistency:
  https://arxiv.org/abs/1703.00564
- Recent work by Guo, Hernandez-Hernandez, and Ballester shows that scaffold
  split alone can still overestimate molecular-model performance because
  different scaffolds may remain highly similar; they compare random, scaffold,
  Butina, and UMAP cluster splits:
  https://arxiv.org/abs/2406.00873

## Implemented Rule

Script:

`scripts/build_scaffold_cluster_splits.py`

Default parameters:

- SMILES normalization: RDKit parse, largest carbon-containing parent fragment,
  uncharge when possible, canonical non-isomeric SMILES.
- Invalid/no-organic-parent policy: `exclude` for the formal v1.2.24 run.
- Exact duplicate grouping: canonical parent SMILES.
- Scaffold grouping: non-empty Bemis-Murcko scaffold SMILES.
- Similarity grouping: Morgan/ECFP radius 2, 2048 bits, Butina clustering with
  Tanimoto similarity threshold `0.65`.
- Final structure group: union of exact canonical parent, shared scaffold, and
  Butina similarity cluster.
- Split assignment: group-disjoint row-count balancing.

The `0.65` Tanimoto threshold is a pragmatic initial threshold rather than a
universal regulatory cutoff. It is stricter than the repository's current AD
screening threshold `0.5`, but not so strict that most analog series fragment
into singleton groups. The decisive quality gate is therefore the generated
audit, especially `tanimoto_leakage_summary.csv`.

## Local v1.2.24 Split Audit

Input database:

`outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite`

Input source table:

`aggregated_task_records_soil_ptox_qc_no_metal_inorganic`

Audit directory:

`outputs/experiments/v1_2_24_scaffold_cluster_holdout_mainline_audit`

Generated split names:

- `NoMetalSoilPtoxQC2_G_scaffold_cluster_8_2`
- `NoMetalSoilPtoxQC2_H_scaffold_cluster_5fold_fold1`
- `NoMetalSoilPtoxQC2_H_scaffold_cluster_5fold_fold2`
- `NoMetalSoilPtoxQC2_H_scaffold_cluster_5fold_fold3`
- `NoMetalSoilPtoxQC2_H_scaffold_cluster_5fold_fold4`
- `NoMetalSoilPtoxQC2_H_scaffold_cluster_5fold_fold5`

Key audit facts:

- Original soil rows: 13,549.
- Included organic-parent rows: 11,992.
- Included chemical units: 1,088.
- Excluded no-organic-parent chemical units: 117.
- Excluded records: 1,557.
- Structure groups: 550.
- Holdout split: train 9,136; test 2,856.
- 5-fold splits: four folds have test 2,284 and one fold has test 2,856,
  reflecting the largest indivisible structure group.
- Structure overlap audit: group, canonical parent, and scaffold overlaps are
  all zero for each split.
- Tanimoto leakage audit: no holdout test chemical has max Tanimoto to train
  above 0.65; two 5-fold test partitions have a very small share above 0.65
  due to Butina centroid behavior, but none exceed 0.80.

## Interpretation Boundary

This split should be described as a scaffold/similarity-cluster chemical-family
extrapolation test, not as an absolute prospective external validation. It is
more chemically defensible than CAS grouping, but still relies on a chosen
fingerprint, scaffold definition, and similarity threshold. Results should be
reported alongside random-split interpolation results and, if needed, future
threshold sensitivity checks.
