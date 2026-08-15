# Grounding Retune Follow-Up (`seed=42`)

Base recipe:
- `sigreg_dense_hybrid_dollar_masked_grounding`
- `observed_claim_loss_weight=0.0`
- baseline grounding: `masked_next_claim_token_weight=0.2`, `masked_next_claim_mask_ratio=0.7`

This sweep was intentionally stopped early after the first two weight points because both missed the baseline.

## Results

| Variant | MAE | WAPE | RMSE | Q4 RMSE | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `obs000_mask070` | $2050.02 | 44.24% | $3119.47 | $3414.66 | current default candidate |
| `masked_next_claim_token_weight=0.1` | $2052.09 | 44.28% | $3121.38 | $3428.35 | worse overall and worse tail |
| `masked_next_claim_token_weight=0.4` | $2049.46 | 44.23% | $3122.30 | $3429.06 | tiny MAE gain, but worse RMSE and worse tail |

## Read

- The current grounding weight `0.2` still looks like the right setting.
- Both lower and higher grounding weight regressed the high-cost tail.
- This knob is now low-signal compared with bigger changes to the temporal objective.

## Recommendation

- Stop mask/weight retuning here.
- Next meaningful move should be a new temporal target, not another grounding-weight sweep.
