# v1.2.31 Molecular-Size Descriptor Sensitivity Plan

## Question

Some pTox targets are converted from mass concentration by molecular weight:

```text
mol/L = mg/L / 1000 / MW
pTox = -log10(mol/L) = -log10(mg/L) + 3 + log10(MW)
```

Because the model also uses molecular descriptors, including `MolWt`, molecular
weight and closely related size descriptors can be mathematically coupled to the
target scale for records converted through `mg/L -> mol/L -> pTox`. This is not
direct target-column leakage, but it can inflate the apparent importance of
molecular-size features. v1.2.31 tests whether the current conclusions remain
stable when these features are masked.

## Deep-Model Sensitivity

New ablation: `no_molecular_size_descriptors`.

The ablation keeps Morgan fingerprints, context features, species/lifestage
embeddings, medium adapters, source weighting, toxicity-bin auxiliary loss, and
censored loss unchanged. It masks only the molecular descriptor dimensions most
closely tied to molecular weight or molecular size:

- `MolWt`
- `TPSA`
- `HeavyAtomCount`
- `NumHAcceptors`
- `NumHDonors`
- `RingCount`
- `RotatableBonds`

`MolLogP` is intentionally retained as a hydrophobicity control rather than a
direct molecular-weight conversion proxy.

Remote launcher:

```bash
scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh priority
```

Priority mode runs `full` and `no_molecular_size_descriptors` on:

- no-metal random 8:2 transfer split
- no-metal scaffold/similarity-cluster 8:2 transfer split

Full matrix modes are available if the priority result shows material
sensitivity:

```bash
scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh random
scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh scaffold
scripts/run_v1_2_31_molecular_size_sensitivity_remote.sh all
```

## Traditional-ML Descriptor Baseline Extension

`scripts/run_traditional_ml_descriptor_effect_baselines.py` now accepts:

```bash
--descriptor-sensitivity drop_molecular_size_related
```

This removes the broader RDKit 2D descriptor families that are direct molecular
weight, atom-count, size, surface-area, ring/flexibility, BCUT, Chi, and VSA
proxies before training local descriptor-effect baselines. The default remains
`--descriptor-sensitivity full`, so v1.2.28-v1.2.30 are still reproducible.

## Interpretation Rule

- If performance is close to `full`, molecular-size coupling is unlikely to be
  the main driver of model skill, but `MolWt` importance should still be
  described cautiously.
- If MAE/RMSE worsen strongly or R2 drops sharply, report that part of the pTox
  signal is scale-coupled to molecular size and avoid interpreting molecular
  weight as an independent toxicity mechanism.
- Always stratify interpretation by split policy: random splits reflect
  interpolation; scaffold/similarity-cluster splits are the chemical-family
  extrapolation stress test.
