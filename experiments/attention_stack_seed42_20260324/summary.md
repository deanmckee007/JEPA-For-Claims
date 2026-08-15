# Attention Stack Seed-42 Check

| Label | MAE $ | WAPE % | RMSE $ | Q4 RMSE $ | Δ RMSE | Δ Q4 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| obs000_baseline | 2050.02 | 44.24 | 3119.47 | 3414.66 | +0.00 | +0.00 |
| code_attention | 2054.91 | 44.35 | 3118.61 | 3438.00 | -0.86 | +23.34 |
| code_plus_component | 2091.84 | 45.14 | 3181.94 | 3439.23 | +62.48 | +24.57 |

Batch stopped early after the first two attention variants because both moved away from the cost-first objective.