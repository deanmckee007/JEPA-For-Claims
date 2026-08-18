## Zero-SIGReg Multiseed Confirmation

Compared the current grounded baseline against the same recipe with `sigreg_weight_lvl2=0.0`.

- Baseline: `sigreg_dense_hybrid_dollar_masked_grounding`, `observed_claim_loss_weight=0.0`, `masked_next_claim_mask_ratio=0.7`
- Variant: baseline + `sigreg_weight_lvl2=0.0`
- Seeds: `42`, `7`, `123`
- Representation source for evaluation: `patient_representation_pre_sae`

### Mean Metrics

| Variant | MAE ($) | WAPE (%) | RMSE ($) | q4 RMSE ($) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2075.59 +/- 24.57 | 44.83 +/- 0.55 | 3161.70 +/- 32.95 | 3458.95 +/- 58.34 |
| sigreg0 | 2068.73 +/- 19.40 | 44.68 +/- 0.34 | 3152.15 +/- 34.55 | 3449.20 +/- 45.82 |

### Per-Seed Notes

- Seed `42`: `sigreg0` beat baseline on MAE, WAPE, RMSE, and q4 RMSE.
- Seed `7`: `sigreg0` beat baseline on overall MAE, WAPE, and RMSE, but slightly lost on q4 RMSE.
- Seed `123`: `sigreg0` beat baseline on MAE, WAPE, RMSE, and q4 RMSE.

### Read

Within this grounded regime, `sigreg_weight_lvl2=0.0` looks like a real simplification candidate rather than a seed-42 fluke. It improved the mean overall cost metrics and also slightly improved mean q4 high-cost RMSE across the three seeds.
