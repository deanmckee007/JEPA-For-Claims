# Hierarchy Geometry Summary

Date: 2026-03-23

Checkpoint: `A6_baseline_hybrid_dollar_seed42`

Recipe: `sigreg_dense_hybrid_dollar`

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Seed: `42`

## Participation By Level

| level | samples | dim | PR | PR / dim | mean cosine | top-1 variance share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `target_next_slot` | 15,374 | 128 | 13.98 | 0.109 | 0.165 | 0.208 |
| `target_all_slots` | 45,278 | 128 | 13.26 | 0.104 | 0.176 | 0.214 |
| `sequence_output` | 542,217 | 128 | 13.16 | 0.103 | 0.280 | 0.190 |
| `context_claim_representation` | 542,217 | 128 | 10.79 | 0.084 | 0.168 | 0.224 |
| `context_mean_pool` | 15,374 | 128 | 8.22 | 0.064 | 0.356 | 0.291 |
| `context_pooled` | 15,374 | 256 | 6.78 | 0.026 | 0.482 | 0.330 |
| `patient_representation_pre_sae` | 15,374 | 256 | 6.78 | 0.026 | 0.482 | 0.330 |
| `patient_representation` | 15,374 | 256 | 5.06 | 0.020 | 0.593 | 0.401 |
| `prediction_next_slot` | 15,374 | 128 | 3.43 | 0.027 | 0.362 | 0.469 |
| `prediction_all_slots` | 45,278 | 128 | 3.29 | 0.026 | 0.373 | 0.477 |

## Read

- The representation is not narrow everywhere. Claim-level and sequence-level states have much healthier effective rank than the final patient probe state.
- The first big squeeze happens when the sequence is pooled into a patient state:
  - `context_mean_pool`: PR `8.22 / 128 = 6.4%`
  - `context_pooled`: PR `6.78 / 256 = 2.6%`
- The second squeeze happens after SAE / gated fusion:
  - `patient_representation_pre_sae`: PR `6.78`
  - `patient_representation`: PR `5.06`
- The narrowest objects are the prediction heads, especially the next-claim prediction latent.
- The target-side next-claim representations are much broader than the predicted next-claim representations.

## Current Takeaway

The low participation issue is mainly a patient-state bottleneck, not a universal collapse across the hierarchy.
