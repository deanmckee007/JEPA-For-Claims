# R1 multihorizon seed 42

Recipe:
- `sigreg_dense_hybrid_dollar_masked_grounding_multihorizon`
- `future_claim_k=2`
- `future_claim_loss_weight=0.5`
- `future_claim_loss_decay=0.5`
- `observed_claim_loss_weight=0.0`
- `masked_next_claim_mask_ratio=0.7`
- `masked_next_claim_token_weight=0.2`

Stage 1 training:
- `10` epochs
- final `val_rmse`: `$3155.33`
- mean-target baseline RMSE: `$3674.94`
- improvement vs mean baseline: `$519.61`

Cost probe on `patient_representation_pre_sae`:
- MAE: `$2083.24`
- WAPE: `44.96%`
- RMSE: `$3156.80`
- q4 high-cost RMSE: `$3482.62`

Reference comparison against the current `obs000_mask070` seed-42 baseline:
- baseline MAE: `$2050.02`
- baseline WAPE: `44.24%`
- baseline RMSE: `$3119.47`
- baseline q4 high-cost RMSE: `$3414.66`

Seed-42 read:
- multihorizon is worse than the current baseline on overall MAE, WAPE, RMSE, and q4 high-cost RMSE
- this specific future-slot formulation is a reject unless later tuning reverses the gap
