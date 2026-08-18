# H10 Cost Head Summary

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Seed: `42`

Base recipe: `sigreg_dense_hybrid_dollar`

## Setup

These runs turn on the in-training cost head (`use_predictor_head=True`) and vary
`task_loss_weight` while keeping the current Stage 1 SSL recipe fixed. Geometry
is compared against the no-cost-head baseline in
`experiments/hierarchy_geometry_20260323/baseline_seed42.json`.

## Results

| run | task_loss_weight | final val_rmse ($) | mean-pool PR | pre-SAE PR | post-SAE PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0 | 3140.16 | 8.22 | 6.78 | 5.06 |
| `H10_cost_head_weight025_seed42` | 0.25 | 3138.11 | 8.25 | 6.79 | 5.13 |
| `H10_cost_head_weight100_seed42` | 1.0 | 3137.33 | 8.22 | 6.79 | 5.12 |

## Read

- Turning on the cost head improved in-training `val_rmse` slightly at both
  tested weights.
- Geometry barely moved. The pooled patient-state participation ratios changed
  by only a few hundredths relative to the baseline.
- The cost head did not create an obvious additional geometry bottleneck in
  this first pass.

## Current Takeaway

Weak-to-moderate cost-head supervision looks safe so far. The next useful test
is stronger or differently placed cost supervision, not whether the current
head immediately collapses the patient representation.
