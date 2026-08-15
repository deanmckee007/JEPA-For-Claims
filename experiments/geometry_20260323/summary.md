# Representation Geometry Summary

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Seed: `42`

## Baseline Source Sweep

Checkpoint: `A6_baseline_hybrid_dollar_seed42`

| source | dim | participation ratio | mean pairwise cosine | top-1 variance share |
| --- | ---: | ---: | ---: | ---: |
| `context_mean_pool` | 128 | 8.22 | 0.356 | 0.291 |
| `patient_representation_pre_sae` | 256 | 6.78 | 0.482 | 0.330 |
| `context_pooled` | 256 | 6.78 | 0.482 | 0.330 |
| `patient_representation` | 256 | 5.06 | 0.593 | 0.401 |
| `context_max_pool` | 128 | 4.98 | 0.544 | 0.405 |
| `next_claim_prediction` | 128 | 3.43 | 0.362 | 0.469 |

Read:

- The post-SAE `patient_representation` is more compressed than the pre-SAE pooled patient state.
- The mean-pooled sequence state is the healthiest geometry among the tested sources.
- The predicted next-claim latent is the narrowest of the main sources.

## Structural Ablations

All ablations use the `sigreg_dense_hybrid_dollar` base recipe for 10 stage-1 epochs.

| run | config diff | final val_rmse ($) | pre-SAE PR | pre-SAE cosine | post-SAE PR | post-SAE cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | none | 3140.16 | 6.78 | 0.482 | 5.06 | 0.593 |
| `Geom_no_sae_seed42_ep10` | `use_sparse_autoencoder=false` | 3142.11 | 5.95 | 0.504 | 5.95 | 0.504 |
| `Geom_no_dense_seed42_ep10` | `use_level2_dense_prediction=false` | 3157.40 | 7.91 | 0.453 | 5.52 | 0.602 |
| `Geom_lvl2_shared_seed42_ep10` | `target_encoder_mode_lvl2=shared` | 3140.40 | 6.78 | 0.482 | 5.04 | 0.594 |

Read:

- Removing SAE did not improve geometry. It lowered the pooled patient-state participation ratio and slightly hurt `val_rmse`.
- Removing dense Level 2 prediction improved pre-SAE geometry, but materially hurt `val_rmse`.
- Switching Level 2 from EMA to shared targets was almost a no-op for both geometry and `val_rmse`.

## Current Takeaway

- The main geometry bottleneck is not Level 2 EMA targets.
- SAE is not the root cause either; if anything, it is helping slightly relative to the no-SAE ablation.
- Dense observed-plus-next supervision appears to trade some geometric spread for better cost-side behavior.
- If the goal is broader-use representations, `context_mean_pool` and `patient_representation_pre_sae` look better than the current post-SAE `patient_representation`.
