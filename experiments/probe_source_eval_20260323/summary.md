# Probe Representation Source Summary

Date: 2026-03-23

Checkpoint: [A6_masked_grounding_seed42/encoder.ckpt](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A6_masked_grounding_seed42/encoder.ckpt)

Recipe: `sigreg_dense_hybrid_dollar_masked_grounding`

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Evaluation set: full filtered training set (`15374` patient sequences)

## Overall Ranking By Dollar RMSE

| representation_source | embedding_dim | MAE ($) | WAPE (%) | RMSE ($) | q4 high-cost MAE ($) | q4 high-cost RMSE ($) | silhouette | TTNC proxy retrieval@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `patient_representation_pre_sae` | `256` | `2047.12` | `44.18` | `3123.28` | `2365.96` | `3699.89` | `0.4365` | `0.6506` |
| `context_pooled` | `256` | `2047.12` | `44.18` | `3123.28` | `2365.96` | `3699.89` | `0.4365` | `0.6506` |
| `patient_representation` | `256` | `2047.99` | `44.20` | `3125.28` | `2363.32` | `3695.35` | `0.4366` | `0.6719` |
| `context_mean_pool` | `128` | `2067.14` | `44.61` | `3134.72` | `2321.21` | `3692.52` | `0.4708` | `0.6372` |
| `next_claim_prediction` | `128` | `2068.38` | `44.64` | `3199.65` | `2363.80` | `3722.40` | `0.5016` | `0.6581` |
| `context_max_pool` | `128` | `2107.63` | `45.48` | `3221.43` | `2363.94` | `3716.53` | `0.4384` | `0.6516` |

## Read

- The best overall probe source on the trusted dollar metrics is `patient_representation_pre_sae`.
- Under the current config, `patient_representation_pre_sae` and `context_pooled` are identical because the pre-SAE patient state is the pooled sequence state.
- The current default probe source, `patient_representation`, is slightly worse overall than the pre-SAE state, but it keeps a small edge on `q4_high_cost` RMSE and the TTNC-proxy retrieval metric.
- `next_claim_prediction` is not a good default probe source for cost evaluation. It is materially worse than the patient-level pooled states on MAE, WAPE, and RMSE.
- `context_mean_pool` is mixed. It is worse overall than the pooled patient states, but it slightly improves the high-cost bucket metrics and silhouette.

## Current Conclusion

For cost-first downstream probing, the best default source is:

- `patient_representation_pre_sae`

If high-cost tail behavior becomes the primary guardrail, keep `patient_representation`
and compare it directly against `context_mean_pool`.
