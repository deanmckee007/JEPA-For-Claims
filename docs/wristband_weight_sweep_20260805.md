# Wristband marginal-weight sweep — 2026-08-05

## Protocol

- Frozen complete-claim cohort: 10,659 train and 2,324 validation patients.
- Seed 42, four representation-training epochs, deterministic evaluation.
- Canonical representation: `patient_representation_pre_sae`.
- Ridge cost probe fit on train and scored on validation.
- No test data was accessed.
- New Wristband weights: 0.005, 0.01, 0.025, and 0.05. The existing 0.1,
  SIGReg, and unregularized runs are unchanged anchors.

## Results

| Treatment | MAE ($) | RMSE ($) | WAPE (%) | Retrieval@5 | Silhouette |
|---|---:|---:|---:|---:|---:|
| None | 1,877.15 | 2,983.71 | 41.027 | 0.6988 | 0.1126 |
| SIGReg | **1,859.77** | **2,959.49** | **40.648** | 0.6984 | **0.1250** |
| Wristband 0.005 | 1,920.52 | 3,025.75 | 41.975 | **0.7104** | 0.1013 |
| Wristband 0.01 | 1,926.63 | 3,029.50 | 42.109 | 0.7070 | 0.1022 |
| Wristband 0.025 | 1,930.77 | 3,032.71 | 42.200 | 0.7057 | 0.0875 |
| Wristband 0.05 | 1,932.16 | 3,033.40 | 42.230 | 0.7057 | 0.0881 |
| Wristband 0.1 | 1,934.66 | 3,035.36 | 42.285 | 0.7083 | 0.0953 |

The weakest Wristband treatment, 0.005, is its best cost point, but it remains
$43.37 worse in MAE and $42.04 worse in RMSE than no marginal regularizer. It
is $60.75 worse in MAE, $66.26 worse in RMSE, and 1.327 WAPE points worse than
SIGReg. There is no useful interior optimum in the tested range.

## Missing-modality tradeoff

| Treatment | Missing CPT RMSE ($) | Full/missing CPT cosine | Missing ICD RMSE ($) | Full/missing ICD cosine |
|---|---:|---:|---:|---:|
| None | **3,315.93** | 0.8893 | 3,105.51 | 0.8991 |
| SIGReg | 3,372.41 | 0.8872 | 3,073.69 | 0.9032 |
| Wristband 0.005 | 3,478.41 | 0.9139 | 3,046.05 | 0.9169 |
| Wristband 0.01 | 3,482.81 | 0.9173 | 3,043.29 | 0.9182 |
| Wristband 0.025 | 3,496.76 | 0.9197 | 3,034.43 | **0.9191** |
| Wristband 0.05 | 3,502.48 | 0.9203 | **3,033.38** | 0.9188 |
| Wristband 0.1 | 3,506.21 | **0.9207** | 3,034.62 | 0.9189 |

Increasing Wristband weight makes the ablated representation geometrically
closer to the full representation. That helps missing-ICD RMSE, but missing-CPT
RMSE worsens monotonically and full-input cost quality degrades. The objective
is enforcing invariance, but the shared scalar weight applies the same pressure
to CPT and ICD even though their cost relevance is asymmetric.

## Decision

Do not promote any tested Wristband weight as the general Level-1 marginal
regularizer. Retain SIGReg for the primary claims representation. If Wristband
is revisited, an asymmetric or ICD-only formulation is more justified than
searching additional values of the same shared scalar weight.
