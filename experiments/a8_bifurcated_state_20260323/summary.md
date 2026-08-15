# A8 Bifurcated Patient State Summary

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Seed: `42`

Base recipe: `sigreg_dense_hybrid_dollar`

## Setup

This branch adds a dedicated `predictive_state` projection inside the Level 2
prediction block. The dense observed-plus-next decoder reads from that
predictive state, while the exposed patient representation remains on its own
path.

Recipe: `sigreg_dense_hybrid_dollar_bifurcated_state`

## Results

| run | final val_rmse ($) | mean-pool PR | pre-SAE PR | post-SAE PR | next-slot PR |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline `sigreg_dense_hybrid_dollar` | 3140.16 | 8.22 | 6.78 | 5.06 | 3.43 |
| `A8_bifurcated_state_seed42` | 3131.04 | 8.45 | 6.47 | 5.35 | 3.45 |

## Read

- The built-in `val_rmse` improved by about `$9.1` without turning on the cost head.
- The exposed post-SAE patient representation improved modestly:
  - PR `5.06 -> 5.35`
  - top-1 variance share `0.401 -> 0.380`
- The mean-pooled sequence state also broadened:
  - PR `8.22 -> 8.45`
  - top-1 variance share `0.291 -> 0.252`
- The pre-SAE pooled patient state narrowed slightly:
  - PR `6.78 -> 6.47`
- The dedicated `predictive_state` absorbed a narrow predictive geometry:
  - PR `5.43`
  - mean cosine `0.616`
  - low-variance dims `<1e-2`: `0.008`

## Current Takeaway

The bifurcation appears to move some specialization into the dedicated
predictive path while modestly improving the exposed final patient
representation and the built-in cost probe. That is the right direction for the
project’s “broad representation + predictive utility” goal.
