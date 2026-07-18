# Molecular Signal Strength Branch Design

Date: 2026-07-07

## Scope

This note records the side-branch implementation for the "molecular signal
strength" exploration. The branch is intentionally additive: the default deep
model still uses the existing RDKit descriptor + Morgan fingerprint pathway
unless an experiment explicitly provides a PaDEL cache and descriptor head
configuration.

Current evidence boundary: fixed chemical holdout / CAS-number `v1.2.18` is
excluded from current conclusions and is retained only as a historical source.

## Implemented Interfaces

### PaDEL Descriptor Cache

New entry point:

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_padel_feature_cache.py `
  --padel-csv outputs/features/padel_descriptors.csv `
  --out outputs/features/molecular_features_padel_morgan512.jsonl `
  --smiles-column smiles `
  --fingerprint-size 512 `
  --source-table aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic
```

Optional PaDEL jar execution is supported through `--padel-jar` and
`--padel-input`. The produced JSONL is compatible with
`experiment.molecular_feature_cache` and includes:

- `feature_source`
- `descriptor_schema_hash`
- `descriptor_names`
- `descriptors`
- `fingerprint_size`
- `fingerprint`

Morgan fingerprints are retained in the PaDEL cache for compatibility with
existing source-weighting and application-domain utilities. Descriptor-only or
graph-only experiments can still mask or bypass fingerprints through ablation
or later graph configuration.

### Descriptor Heads

`EcotoxMultiTaskNetwork` now supports three descriptor modes through config:

- `raw`: default, unchanged mainline behavior.
- `dense_head`: project all molecular descriptors through one small MLP before
  context/fingerprint fusion.
- `prior_clustered_heads`: resolve descriptor groups from YAML and encode each
  group through a small independent MLP before fusion.

Example config fragment:

```yaml
features:
  molecule:
    encoder: padel_descriptor_morgan
    morgan_n_bits: 512
    descriptor_head:
      mode: prior_clustered_heads
      cluster_file: configs/padel_descriptor_clusters.example.yaml
      head_dim: 64
      group_head_dim: 16
```

The existing `configs/descriptor_clusters.example.yaml` is no longer assumed to
be active by default. Group membership is written into `preprocessing.json`
only when clustered mode is selected.

### Descriptor-Name Safety

The training path now reads descriptor names from cache rows and writes them to
`preprocessing.json`. The following logic is descriptor-name aware:

- `no_molecular_size_descriptors` masking.
- toxicity-bin molecular-weight lookup.
- proxy-distance source weighting.
- descriptor-head group index resolution.

This avoids assuming that PaDEL descriptor column 0 is `MolWt`.

### Molecular Graph Reserved Interface

New entry point:

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_molecular_graph_cache.py `
  --db outputs/derived/modeling_dataset_v2_0_0_rebuild_no_metal_inorganic.sqlite `
  --source-table aggregated_task_records_aquatic_soil_ptox_qc_no_metal_inorganic `
  --out outputs/features/molecular_graphs_rdkit_v1.jsonl
```

The graph cache currently records atom features, directed edge indices, and
bond features. It is a reserved interface only; graph tensors are not yet
collated into the deep training loop. This keeps the current model stable while
making the graph-only branch explicit.

Graph-only experiment definition for future training: molecular input channel
uses molecular graph only, without RDKit descriptors or Morgan fingerprint,
while species/context embeddings remain enabled.

## Dependencies

- Existing RDKit remains the primary chemistry dependency.
- Optional PaDEL support is declared as `padelpy>=0.1.16`.
- Java is required when invoking a PaDEL jar directly.
- No PyTorch Geometric or DGL dependency is added yet.

## Verification

Passed:

```powershell
E:\TOOLS\anaconda\python.exe -m py_compile qsar_tl\modeling\network.py qsar_tl\training\deep_experiment.py qsar_tl\features\descriptor_groups.py qsar_tl\features\padel.py qsar_tl\features\molecular_graph.py scripts\build_padel_feature_cache.py scripts\build_molecular_graph_cache.py scripts\build_molecular_signal_strength_summary.py scripts\explain_deep_model.py

E:\TOOLS\anaconda\python.exe -m pytest tests\test_deep_experiment_cache.py tests\test_traditional_ml_descriptor_effect_baselines.py
```

The pytest run passed 51 tests with one pre-existing PyTorch/NumPy warning in
the base Anaconda environment.

The molecular signal summary package was generated with:

```powershell
E:\TOOLS\anaconda\envs\qsar-ph3\python.exe scripts\build_molecular_signal_strength_summary.py
```

## Next Experiment Matrix

Do not use `v1.2.18` to fill current evidence gaps. The current-boundary
minimum matrix is:

| Priority | Ablation | Split policy | Seed |
|---:|---|---|---:|
| 1 | `no_descriptors` | `random_8_2` | 2042 |
| 2 | `no_descriptors` | `scaffold_cluster_8_2` | 2042 |
| 3 | `no_fingerprint` | `random_8_2` | 2042 |
| 4 | `no_fingerprint` | `scaffold_cluster_8_2` | 2042 |
| 5 | `descriptors_only` | `random_8_2` | 2042 |
| 6 | `descriptors_only` | `scaffold_cluster_8_2` | 2042 |

PaDEL experiments should follow after the cache is built:

1. `padel_unclustered`: PaDEL cache + `dense_head`.
2. `padel_prior_clustered`: PaDEL cache + `prior_clustered_heads`.
3. `rdkit_full_unclustered` if full RDKit 2D descriptors are desired as a
   separate comparator.
4. `graph_only_no_descriptors_no_fingerprint` after graph tensors are connected
   to the training loop.
