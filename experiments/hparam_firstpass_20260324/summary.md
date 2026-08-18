# First-Pass Hyperparameter Sweep

| Run | MAE ($) | WAPE (%) | RMSE ($) | Q4 RMSE ($) | ? MAE | ? WAPE | ? RMSE | ? Q4 RMSE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 2041.56 | 44.06 | 3116.93 | 3446.93 | 0 | 0 | 0 | 0 | AdamW, StepLR, batch 512, adapter_lr=1e-4, generator_lr=5e-4 |
| adam | 2043.46 | 44.1 | 3118.85 | 3451 | 1.9 | 0.04 | 1.92 | 4.07 | optimizer_type=adam |
| cosine | 2053.27 | 44.31 | 3133.74 | 3437.37 | 11.71 | 0.25 | 16.82 | -9.56 | cosine + 1 epoch warmup |
| batch256 | 2053.41 | 44.31 | 3133.77 | 3418.17 | 11.85 | 0.26 | 16.84 | -28.76 | train_batch_size=256 |
| lr_low | 2062.42 | 44.51 | 3136.46 | 3458.57 | 20.87 | 0.45 | 19.53 | 11.65 | adapter_lr=5e-5, generator_lr=2.5e-4 |
| lr_high | 2061.9 | 44.5 | 3151.09 | 3387.31 | 20.34 | 0.44 | 34.16 | -59.62 | adapter_lr=2e-4, generator_lr=1e-3 |
