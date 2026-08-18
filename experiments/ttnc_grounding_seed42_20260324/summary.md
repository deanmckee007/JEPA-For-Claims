# TTNC grounding screen, seed 42

Base setup:
- recipe: `sigreg_dense_hybrid_dollar_masked_grounding`
- `observed_claim_loss_weight=0.0`
- `masked_next_claim_mask_ratio=0.7`
- Stage 1 only, `10` epochs
- probe source: `patient_representation_pre_sae`

Results:

| Variant | MAE ($) | WAPE (%) | RMSE ($) | q4 RMSE ($) |
|---|---:|---:|---:|---:|
| CPT+ICD | 2050.02 | 44.24 | 3119.47 | 3414.66 |
| CPT+ICD+TTNC | 2064.97 | 44.56 | 3138.81 | 3431.10 |
| ICD+TTNC | 2067.97 | 44.63 | 3141.62 | 3420.06 |

Read:
- adding TTNC to masked next-claim grounding hurt overall cost metrics
- `ICD+TTNC` slightly beat `CPT+ICD+TTNC` on q4 tail RMSE, but both lost to the plain `CPT+ICD` baseline
- seed 42 does not support promoting TTNC grounding into the default cost-first recipe

Artifacts:
- [baseline run](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R5_ttnc_seed42_cpt_icd/cost_probe.json)
- [CPT+ICD+TTNC run](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R5_ttnc_seed42_cpt_icd_ttnc/cost_probe.json)
- [ICD+TTNC run](C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/R5_ttnc_seed42_icd_ttnc/cost_probe.json)
