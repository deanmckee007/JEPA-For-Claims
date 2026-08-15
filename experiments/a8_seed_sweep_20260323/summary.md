# A8 Seed Sweep

Fixed eval settings:
- full dataset path with fixed probe split random_state=42
- dollar probes on patient_representation_pre_sae and patient_representation only
- geometry on context_mean_pool, context_pooled, patient_representation_pre_sae, patient_representation, and predictive_state when present

## Per Run

| Run | Pre-SAE RMSE | Pre-SAE MAE | Patient RMSE | Patient MAE | Pre-SAE PR | Patient PR | Predictive PR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_seed42 | 3134.77 | 2057.23 | 3136.90 | 2059.20 | 6.78 | 5.06 | 6.78 |
| a8_seed42 | 3130.69 | 2055.89 | 3123.63 | 2048.23 | 6.46 | 5.34 | 5.43 |
| baseline_seed7 | 3175.43 | 2084.33 | 3173.70 | 2085.51 | 6.77 | 5.42 | 6.77 |
| a8_seed7 | 3150.20 | 2084.97 | 3149.79 | 2083.68 | 6.57 | 5.53 | 4.79 |
| baseline_seed123 | 3145.11 | 2062.42 | 3144.25 | 2060.89 | 6.62 | 4.52 | 6.62 |
| a8_seed123 | 3176.62 | 2078.66 | 3176.06 | 2077.29 | 6.32 | 5.20 | 4.99 |

## A8 Minus Baseline By Seed

| Seed | Pre-SAE RMSE Delta | Pre-SAE MAE Delta | Patient RMSE Delta | Patient MAE Delta | Pre-SAE PR Delta | Patient PR Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | -4.08 | -1.34 | -13.27 | -10.97 | -0.32 | 0.29 |
| 7 | -25.24 | 0.64 | -23.91 | -1.83 | -0.20 | 0.11 |
| 123 | 31.52 | 16.24 | 31.81 | 16.40 | -0.30 | 0.68 |
