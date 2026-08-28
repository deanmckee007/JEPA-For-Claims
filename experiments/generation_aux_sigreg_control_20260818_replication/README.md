# SigReg anti-collapse replication

This experiment tests whether the repo-native LE-JEPA convex SigReg objective
is sufficient to prevent collapse in the label-free consistency cost probe.

## Protocol

- Frozen selected epoch-20 encoder; next-claim prediction embedding.
- Width-matched, target-excluding signed raw-history hash as the control.
- Ten percent of training cost labels; full training cohort for label-free
  regularization.
- Frozen validation split for model selection and reporting; test untouched.
- Downstream seeds 42, 43, and 44.
- Consistency weight 300, Gaussian feature noise 0.1.
- SigReg weight 0.2, LE-JEPA convex formulation, 256 slices, 17 points.
- Twenty downstream epochs.

## Results

| Feature source / objective | MAE ($) | RMSE ($) | Hidden feature std | Hidden norm |
|---|---:|---:|---:|---:|
| Pretrained, SigReg only | 1,706.73 +/- 65.95 | 2,724.11 +/- 73.24 | 0.927 +/- 0.012 | 14.61 +/- 0.09 |
| Pretrained, consistency + SigReg | **1,665.67 +/- 44.15** | **2,691.27 +/- 52.76** | 0.451 +/- 0.002 | 7.42 +/- 0.03 |
| Raw history, SigReg only | 2,123.03 +/- 53.03 | 3,185.29 +/- 65.34 | 0.946 +/- 0.005 | 14.80 +/- 0.08 |
| Raw history, consistency + SigReg | 2,124.61 +/- 33.46 | 3,188.69 +/- 62.94 | 0.464 +/- 0.003 | 7.45 +/- 0.05 |

SigReg alone preserves broad hidden geometry. Combined with the previously
collapsing consistency objective, it retains substantial variation and improves
the pretrained-feature cost result. The same objective is $458.94 MAE worse on
raw history, so generic regularization does not explain the pretrained result.

Machine-readable results and the exact protocol are in `summary.json`.
