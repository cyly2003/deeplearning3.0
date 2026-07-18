# Species/context embedding interpretation notes

Date: 2026-07-07

## Recommended interpretation boundary

Species/context/taxonomy embeddings should be described as predictive latent representations of biological and experimental context. They can quantify whether non-chemical context improves prediction and generalization, but they should not be described as direct biological mechanisms.

The embedding coordinates themselves are not stable biological axes. They can rotate or change after retraining, and may encode a mixture of true species sensitivity, taxonomy, endpoint distribution, protocol differences, exposure medium, duration, and database sampling bias.

## Recommended quantitative analyses

1. Group ablation

Compare full, chemical-only, no-species-context, no-taxonomy-context, no-endpoint-context, and context-only diagnostic models. Report delta R², RMSE, MAE, and Huber loss with seed-level or bootstrap intervals.

2. Group permutation importance

Permute feature groups rather than individual embedding dimensions. Suggested groups: molecular descriptors, Morgan fingerprint bits, species/taxonomy context, exposure/effect context, and medium/adapter context. Because taxonomy, endpoint, medium, and duration can be correlated, subgroup or conditional permutation should be preferred for confirmatory claims.

3. Group SHAP

Aggregate SHAP values over context feature groups. Do not interpret single embedding dimensions. Report group mean |SHAP| overall and stratified by endpoint, medium, and taxonomic level.

4. Leave-one-context-out validation

Use leave-one-species, leave-one-genus, leave-one-family, leave-one-endpoint, or leave-one-medium tests to evaluate whether the context representation supports generalization beyond frequent categories.

5. Embedding-space diagnostics

Use UMAP/PCA of learned embeddings, nearest-neighbor purity by taxonomy, embedding distance versus taxonomy distance, embedding distance versus residual similarity, and embedding distance versus species-level sensitivity differences. These are representation diagnostics, not mechanism proofs.

6. Categorical ICE / counterfactual context sensitivity

For representative chemicals, replace species/genus/family/medium/duration context while holding molecular inputs fixed. Use ICE-style panels rather than a single averaged PDP when interactions are expected.

## Suggested figure language

Preferred: "Species/context features contributed substantially to prediction, indicating that toxicity estimates depend on biological and experimental context in addition to chemical structure."

Avoid: "The species embedding reveals the toxicological mechanism."

Preferred: "Embedding-space diagnostics were used to evaluate whether learned species representations were consistent with taxonomy or residual structure."

Avoid: "The embedding dimension corresponds to a specific physiological trait."

## Key references

- Wu et al. used chemical, taxonomic, and experimental information for cross-taxa toxicity prediction and compared the gain from adding biological and experimental context.
- Myklebust et al. used ecotoxicological knowledge graph embeddings to represent chemical, species taxonomy, and effect relationships for chemical-effect prediction.
- Guo and Berkhahn introduced entity embeddings for high-cardinality categorical variables, showing that supervised training can learn useful categorical representations.
- Lundberg and Lee introduced SHAP as a unified feature-attribution framework.
- Goldstein et al. introduced ICE plots, which are better than averaged PDPs when feature effects are heterogeneous.
- Molnar et al. discuss limitations of PDP and permutation importance, especially with dependent features.
- OECD QSAR validation principles motivate clear endpoints, validation, applicability domain, and cautious mechanistic interpretation.
