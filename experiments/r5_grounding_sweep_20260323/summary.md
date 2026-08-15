# R5 Grounding Sweep

| Run | Overrides | MAE ($) | WAPE (%) | RMSE ($) | Q4 RMSE ($) | Q4 MAE ($) | Q4 WAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_hybrid_dollar_seed42 | none | 2057.23 | 44.40 | 3134.77 | 3437.98 | 2285.33 | 26.13 |
| a6_mask050_cpt_icd_seed42 | none | 2047.12 | 44.18 | 3123.28 | 3454.17 | 2303.46 | 26.34 |
| a6_mask050_cpt_only_seed42 | masked_next_claim_use_icd=False | 2044.14 | 44.11 | 3120.08 | 3448.87 | 2288.79 | 26.17 |
| a6_mask050_icd_only_seed42 | masked_next_claim_use_cpt=False | 2052.80 | 44.30 | 3131.92 | 3415.02 | 2271.22 | 25.97 |
| a6_mask030_cpt_icd_seed42 | masked_next_claim_mask_ratio=0.3 | 2053.04 | 44.30 | 3131.86 | 3459.17 | 2301.42 | 26.32 |
| a6_mask070_cpt_icd_seed42 | masked_next_claim_mask_ratio=0.7 | 2041.56 | 44.06 | 3116.93 | 3446.93 | 2291.34 | 26.20 |
