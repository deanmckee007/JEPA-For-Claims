# Multi-encoder SigReg robustness

The selected downstream SigReg setting was repeated across three independently
pretrained, recipe-matched `lejepa_anycode_l1w01` encoders (pretraining seeds
42--44). Each encoder used downstream seeds 42--44, for nine paired runs per
objective. The frozen validation split was used and test data was not accessed.

| Objective | MAE ($) | RMSE ($) | Hidden feature std |
|---|---:|---:|---:|
| SigReg 0.2 | 1,707.70 +/- 43.19 | 2,715.78 +/- 46.35 | 0.938 +/- 0.010 |
| Consistency 300 + SigReg 0.2 | **1,680.57 +/- 28.95** | **2,696.61 +/- 41.30** | 0.443 +/- 0.004 |

Consistency plus SigReg remained non-collapsed for every encoder. Per-encoder
mean MAE was $1,679.08, $1,698.59, and $1,664.04 for pretraining seeds 42, 43,
and 44 respectively. This is also better than the prior aligned-token pooled
result for the same three encoders ($1,717.70 MAE), although aligned supervision
still provides the actual generation capability.

Detailed run records are in the `encoder_seed42`, `encoder_seed43`, and
`encoder_seed44` directories.
