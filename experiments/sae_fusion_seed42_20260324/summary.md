# SAE / Fusion Sweep on `obs000_mask070` (`seed=42`)

Base recipe:
- `sigreg_dense_hybrid_dollar_masked_grounding`
- `masked_next_claim_mask_ratio=0.7`
- `observed_claim_loss_weight=0.0`

Primary metrics:
- overall: MAE / WAPE / RMSE
- tail: `q4_high_cost` RMSE

## Results

| Variant | MAE | WAPE | RMSE | Q4 RMSE | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `obs000_mask070` | $2050.02 | 44.24% | $3119.47 | $3414.66 | current masked-grounding baseline |
| `use_gated_fusion=False` | $2054.28 | 44.33% | $3118.23 | $3444.90 | tiny overall RMSE gain, but clearly worse tail |
| `sae_weight=0.5` | $2052.48 | 44.29% | $3119.31 | $3400.80 | flat overall, best tail result in this batch |
| `sae_weight=2.0` | $2044.42 | 44.12% | $3110.49 | $3438.46 | best overall cost, worse tail |
| `sae_k=5` | $2047.25 | 44.18% | $3116.05 | $3413.63 | good balanced point, better overall and slightly better tail |
| `sae_k=20` | $2048.52 | 44.21% | $3117.22 | $3414.90 | modest overall gain, basically flat tail |

## Read

- `use_gated_fusion=False` is not a promising direction for cost.
- `sae_weight=2.0` is the strongest overall-cost point from this seed.
- `sae_weight=0.5` is the best tail-cost point from this seed.
- `sae_k=5` is the cleanest balanced variant: it improves overall MAE/WAPE/RMSE and also slightly improves the high-cost tail.

## Next

- Best single follow-up: multiseed confirm `sae_weight=2.0` and `sae_k=5` against the current `obs000_mask070` baseline.
- If overall cost is primary, prioritize `sae_weight=2.0`.
- If a balanced overall-plus-tail tradeoff is primary, prioritize `sae_k=5`.
