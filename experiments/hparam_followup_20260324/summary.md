# Hyperparameter Follow-up Sweep

| Run | MAE ($) | WAPE (%) | RMSE ($) | Q4 RMSE ($) | Delta MAE | Delta WAPE | Delta RMSE | Delta Q4 RMSE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 2041.56 | 44.06 | 3116.93 | 3446.93 | 0 | 0 | 0 | 0 | AdamW, StepLR, batch 512, wd=1e-4 |
| weightdecay0 | 2041.71 | 44.06 | 3117.52 | 3448.82 | 0.16 | 0 | 0.59 | 1.89 | AdamW, StepLR, batch 512, wd=0 |
| batch256_cosine | 2056.5 | 44.38 | 3130.57 | 3438.68 | 14.94 | 0.32 | 13.64 | -8.24 | cosine + warmup, batch 256, wd=1e-4 |
| batch256_cosine_wd0 | 2057.23 | 44.4 | 3130.63 | 3430.41 | 15.67 | 0.34 | 13.7 | -16.51 | cosine + warmup, batch 256, wd=0 |
| cosine | 2053.27 | 44.31 | 3133.74 | 3437.37 | 11.71 | 0.25 | 16.82 | -9.56 | cosine + warmup, batch 512, wd=1e-4 |
| batch256 | 2053.41 | 44.31 | 3133.77 | 3418.17 | 11.85 | 0.26 | 16.84 | -28.76 | StepLR, batch 256, wd=1e-4 |
