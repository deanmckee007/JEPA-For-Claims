# SAE / Fusion Multiseed Confirmation (`seeds=7,42,123`)

Recipe family:
- `sigreg_dense_hybrid_dollar_masked_grounding`
- `masked_next_claim_mask_ratio=0.7`
- `observed_claim_loss_weight=0.0`

Compared:
- baseline `obs000_mask070`
- `sae_weight=2.0`
- `sae_k=5`

## Aggregate Results

| Variant | MAE | WAPE | RMSE | Q4 RMSE | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `obs000_mask070` | $2069.22 ± 19.28 | 44.69% ± 0.34 | $3152.26 ± 34.06 | $3450.24 ± 46.90 | current default candidate |
| `sae_weight=2.0` | $2067.38 ± 19.19 | 44.65% ± 0.38 | $3150.27 ± 34.34 | $3449.41 ± 44.92 | tiny overall edge, but within seed noise |
| `sae_k=5` | $2068.81 ± 20.93 | 44.68% ± 0.37 | $3153.09 ± 36.95 | $3448.79 ± 47.20 | essentially a wash; slight tail edge only |

## Read

- The seed-42 SAE gains did not turn into a decisive multiseed win.
- `sae_weight=2.0` has the best mean overall cost metrics, but only by a couple of dollars and well within the run-to-run spread.
- `sae_k=5` has the best mean tail RMSE, but the edge is also tiny and not robust enough to justify a new default.

## Recommendation

- Keep baseline `obs000_mask070` as the default.
- Treat `sae_weight=2.0` as a minor follow-up knob, not a promoted recipe.
- Do not spend more time on this SAE/fusion line unless we decide to optimize specifically for a very small cost-margin target.
