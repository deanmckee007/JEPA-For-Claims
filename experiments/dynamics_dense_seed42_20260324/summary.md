# Dynamics / Dense Follow-Up (`seed=42`)

Base recipe:
- `sigreg_dense_hybrid_dollar_masked_grounding`
- `masked_next_claim_mask_ratio=0.7`
- `observed_claim_loss_weight=0.0`

## Results

| Variant | MAE | WAPE | RMSE | Q4 RMSE | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline `obs000_mask070` | $2050.02 | 44.24% | $3119.47 | $3414.66 | current default candidate |
| EMA + `sigreg_weight_lvl2=0.0` | $2048.85 | 44.21% | $3117.56 | $3414.01 | basically flat to slightly better overall |
| shared target + `sigreg_weight_lvl2=0.1` | $2049.06 | 44.22% | $3118.71 | $3419.34 | slightly worse than EMA on tail and overall RMSE |
| `use_level2_dense_prediction=False` | $2077.50 | 44.83% | $3164.66 | $3411.87 | clearly worse overall; tiny tail gain not worth it |

## Read

- Removing SIGReg pressure entirely does not break the grounded recipe and is essentially a wash on this seed.
- Switching Level 2 targets from EMA to shared is mildly worse, so the earlier hybrid/EMA lesson still appears to hold.
- Fully removing the dense Level 2 path is a real regression on overall cost, even with observed-claim loss already set to `0.0`.

## Recommendation

- Keep EMA targets at Level 2.
- Do not disable the dense Level 2 prediction path.
- If we want to keep pushing this line, the only live knob from this batch is `sigreg_weight_lvl2`, and even that looks low-signal enough that it should only be revisited with a small multiseed confirm.
