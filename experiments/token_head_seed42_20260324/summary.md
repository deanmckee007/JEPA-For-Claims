# Token head screen, seed 42

Stage-1 opt-in fix:
- added `allow_stage1_token_prediction_head` so [scripts/train.py](C:/Users/tmcke/code/JEPA-For-Claims/scripts/train.py) can preserve `use_token_prediction_head=True` when explicitly requested
- default Stage 1 behavior is unchanged

Base comparison point:
- current leader: masked grounding baseline
- recipe: `sigreg_dense_hybrid_dollar_masked_grounding`
- overrides: `observed_claim_loss_weight=0.0`, `masked_next_claim_mask_ratio=0.7`

Results:

| Variant | MAE ($) | WAPE (%) | RMSE ($) | q4 RMSE ($) |
|---|---:|---:|---:|---:|
| Baseline masked grounding | 2050.02 | 44.24 | 3119.47 | 3414.66 |
| Token head only | 2065.73 | 44.58 | 3142.46 | 3433.15 |
| Token head + masked grounding | 2078.21 | 44.85 | 3152.14 | 3406.02 |

Read:
- `token head only` is a clear reject on cost
- `token head + masked grounding` slightly improved q4 tail RMSE, but lost too much on overall MAE, WAPE, and RMSE
- neither token-head variant beats the current masked-grounding baseline on the trusted cost-first criteria

Artifacts:
- [baseline reference](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R5_ttnc_seed42_cpt_icd/cost_probe.json)
- [token head only](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R6b_token_head_only_seed42/cost_probe.json)
- [token head + masked grounding](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R6b_token_head_plus_masked_seed42/cost_probe.json)
