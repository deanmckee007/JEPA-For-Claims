# Cost-First Rerank (2026-03-23)

This rerank compares the strongest current checkpoints on the updated
cost-first evaluation contract. TTNC-proxy metrics are excluded from the
decision. The comparison uses:

- overall `target_probe_mae_dollars`
- overall `target_probe_wape_percent`
- overall `target_probe_rmse_dollars`
- `q4_high_cost` MAE / WAPE / RMSE
- `val_rmse_improvement_vs_mean_baseline` from the recorded training runs

## Candidates

| Candidate | Overall MAE ($) | Overall WAPE (%) | Overall RMSE ($) | Q4 MAE ($) | Q4 WAPE (%) | Q4 RMSE ($) | Val RMSE Improvement ($) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sigreg_dense_hybrid_dollar` | 2051.72 | 44.28 | 3133.95 | 2393.84 | 26.29 | 3764.98 | 557.84 |
| `sigreg_dense_control` | 2058.35 | 44.42 | 3138.23 | 2398.46 | 26.34 | 3728.04 | 570.89 |
| `sigreg_dense_hybrid_repr` (A5 checkpoint) | 2060.53 | 44.47 | 3138.89 | 2432.52 | 26.71 | 3769.42 | 559.92 |

## Decision

`sigreg_dense_hybrid_dollar` is the current cost-first default.

Why:

- It is best on all three overall trusted dollar metrics.
- It is also best on `q4_high_cost` MAE and WAPE.
- The plain dense control keeps a narrow edge only on `q4_high_cost` RMSE.
- The 20-epoch A5 / repr variant does not earn its extra compute on the trusted cost metrics.

## Practical Read

If the goal is broad representation research, keep the 20-epoch hybrid recipe
available. If the goal is the best current cost-facing default, use the 10-epoch
hybrid recipe and treat the dense control as the main heavy-tail baseline.
