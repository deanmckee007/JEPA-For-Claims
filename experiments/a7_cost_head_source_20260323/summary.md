# A7 Cost Head Source Summary

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Seed: `42`

Base recipe: `sigreg_dense_hybrid_dollar_cost_head`

Task weight: `task_loss_weight=1.0`

## Runs

| run | predictor_head_source | final val_rmse ($) | context_mean_pool PR | pre-SAE PR | post-SAE PR |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | no cost head | 3140.16 | 8.22 | 6.78 | 5.06 |
| `H10_cost_head_weight100_seed42` | `context_mean_pool` | 3137.33 | 8.22 | 6.79 | 5.12 |
| `A7_cost_head_pre_sae_weight100_seed42` | `patient_representation_pre_sae` | 3016.90 | 8.04 | 7.05 | 5.45 |
| `A7_cost_head_post_sae_weight100_seed42` | `patient_representation` | 3001.94 | 8.24 | 7.34 | 4.27 |

## Geometry Read

- `context_mean_pool` as the cost-head source barely changes the hierarchy geometry.
- `patient_representation_pre_sae` is the best tradeoff so far:
  - large `val_rmse` gain
  - broader pre-SAE patient state (`6.78 -> 7.05`)
  - broader post-SAE patient state (`5.06 -> 5.45`)
  - lower cosine crowding and lower top-1 variance share at the patient level
- `patient_representation` as the cost-head source gives the best train-side
  `val_rmse`, but it narrows the final patient embedding:
  - pre-SAE gets broader (`6.78 -> 7.34`)
  - post-SAE gets materially narrower (`5.06 -> 4.27`)
  - post-SAE cosine crowding rises (`0.593 -> 0.598`)

## Current Takeaway

If the goal is cost-first prediction with minimal concern for the final
post-SAE geometry, `patient_representation` is the strongest source on this
seed. If the goal is to improve cost while preserving a broader reusable
patient representation, `patient_representation_pre_sae` is the better choice.
