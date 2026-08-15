# A6 Zero-SIGReg Long-Schedule Check

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Recipe family: `sigreg_dense_hybrid_dollar_masked_grounding`

Config override: `sigreg_weight_lvl2=0.0`

Probe source: `patient_representation_pre_sae`

## Purpose

Check whether masked next-claim token grounding can keep the representation from
collapsing when the Level 2 SIGReg weight is set to zero and the stage-1
schedule is pushed beyond the original 10-epoch run.

## Runs

| run | stage-1 epochs | final val_rmse ($) | MAE ($) | WAPE (%) | RMSE ($) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A6_masked_grounding_sigreg000_seed42` | 10 | 3141.39 | 2047.43 | 44.18 | 3126.92 |
| `A6_masked_grounding_sigreg000_seed42_ep20` | 20 | 3133.34 | 2046.71 | 44.17 | 3124.17 |
| `A6_masked_grounding_sigreg000_seed42_ep40` | 40 | 3137.65 | 2046.76 | 44.17 | 3124.18 |

## Geometry Diagnostics

Computed on the full filtered training set using `patient_representation_pre_sae`.

| run | std mean | std min | low-var frac (<1e-3) | low-var frac (<1e-2) | participation ratio | mean pairwise cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ep10` | 0.1430 | 0.0724 | 0.0000 | 0.0000 | 6.8302 | 0.4787 |
| `ep20` | 0.1427 | 0.0727 | 0.0000 | 0.0000 | 6.7971 | 0.4810 |
| `ep40` | 0.1426 | 0.0727 | 0.0000 | 0.0000 | 6.7987 | 0.4810 |

## Read

- The zero-SIGReg A6 path did not show an obvious collapse through 40 epochs on seed 42.
- Dollar-space downstream metrics were stable from 20 to 40 epochs.
- The representation geometry was also stable: no near-zero-variance dimensions and no collapse in effective rank.
- This is still not enough evidence to retire SIGReg. It only says masked grounding can keep this seed/train schedule healthy for at least one long run.
