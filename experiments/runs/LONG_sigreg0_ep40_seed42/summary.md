# Long-Schedule Checkpoint Study

Recipe:
- `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
- `checkpoint_every_n_epochs=2`
- `checkpoint_save_top_k=5`
- planned schedule: 40 stage-1 epochs
- run stopped manually during epoch 24 after later checkpoints had been evaluated

Checkpoint comparison on the trusted cost probe (`patient_representation_pre_sae`):

| Checkpoint | Approx stage epoch | MAE ($) | WAPE (%) | RMSE ($) | Q4 RMSE ($) |
| --- | --- | ---: | ---: | ---: | ---: |
| existing 10-epoch leader | 10 | 2048.85 | 44.2145 | 3117.56 | 3414.01 |
| `stage1-epochepoch=07.ckpt` | 8 | 2049.49 | 44.2283 | 3117.46 | 3413.39 |
| `stage1-epochepoch=15.ckpt` | 16 | 2048.58 | 44.2085 | 3118.40 | 3412.44 |
| `stage1-epochepoch=19.ckpt` | 20 | 2048.46 | 44.2059 | 3118.31 | 3412.23 |
| `stage1-epochepoch=23.ckpt` | 24 | 2048.42 | 44.2052 | 3118.30 | 3412.13 |

Read:
- The earlier 10-epoch default was too aggressive for the cost-first objective.
- Later checkpoints improved MAE, WAPE, and high-cost (`q4`) RMSE.
- Overall RMSE in dollars became slightly worse than the 10-epoch point, but only by about $0.74 at epoch 24.
- The best late-checkpoint tradeoff in this run was around epoch 20 to epoch 24.

Observed log-var / precision drift from `tb_logs/stage1/version_131`:

| Global step | `logvar_ssl_lvl2` | `precision_ssl_lvl2` | `logvar_sae` | `precision_sae` | `loss` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 123 | 0.037001 | 0.963675 | -0.044808 | 1.045827 | 2.469524 |
| 278 | 0.045875 | 0.955161 | -0.056948 | 1.058601 | 2.381921 |
| 526 | 0.047994 | 0.953140 | -0.059615 | 1.061428 | 2.367928 |
| 650 | 0.048113 | 0.953026 | -0.059750 | 1.061571 | 2.367873 |

Interpretation:
- The learned weighting kept moving well past the old 10-epoch point.
- The drift flattened by about epoch 20 rather than blowing up.
- This makes later checkpoint evaluation necessary; relying on early monitor minima is too simplistic for this weighted objective.
