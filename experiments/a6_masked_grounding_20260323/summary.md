# A6 Masked Next-Claim Grounding Summary

Date: 2026-03-23

Dataset: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`

Evaluation set: full filtered training set (`15374` patient sequences)

## Runs

| experiment | recipe | overrides | target_probe_mae_dollars | target_probe_wape_percent | target_probe_rmse_dollars | q4_high_cost_rmse_dollars | cluster_silhouette | ttnc_proxy_retrieval_hit_rate_at_5 | final_val_rmse |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A6_baseline_hybrid_dollar_seed42` | `sigreg_dense_hybrid_dollar` | none | `2059.15` | `44.44` | `3136.79` | `3690.47` | `0.4443` | `0.6435` | `3140.16` |
| `A6_masked_grounding_seed42` | `sigreg_dense_hybrid_dollar_masked_grounding` | none | `2047.99` | `44.20` | `3125.28` | `3695.35` | `0.4366` | `0.6719` | `3140.41` |
| `A6_masked_grounding_sigreg005_seed42` | `sigreg_dense_hybrid_dollar_masked_grounding` | `sigreg_weight_lvl2=0.05` | `2047.57` | `44.19` | `3125.82` | `3696.62` | `0.4435` | `0.6436` | `3141.78` |
| `A6_masked_grounding_sigreg000_seed42` | `sigreg_dense_hybrid_dollar_masked_grounding` | `sigreg_weight_lvl2=0.0` | `2047.43` | `44.18` | `3126.92` | `3695.23` | `0.4505` | `0.6426` | `3141.39` |

## Read

- Masked next-claim token grounding improved the primary overall cost metrics versus the current `sigreg_dense_hybrid_dollar` baseline.
- The best overall dollar RMSE in this matrix came from plain A6 with grounding and the default `sigreg_weight_lvl2=0.1`.
- Lowering or removing SIGReg did not collapse training on this seed, but it also did not produce a clear win over plain A6.
- All A6 variants slightly regressed `q4_high_cost` dollar RMSE versus the baseline, so A6 is not yet a clean promotion if high-cost tail error remains a hard guardrail.
- The training-time `val_rmse` did not improve with A6, even though the downstream full-data cost probes did.

## Current Conclusion

Do not treat masked grounding as a replacement for anti-collapse pressure yet.

The strongest current read is:

- keep SIGReg on
- keep masked next-claim grounding as a promising auxiliary direction
- rerun the baseline vs plain A6 vs zero-SIGReg A6 comparison across more seeds before changing the default
