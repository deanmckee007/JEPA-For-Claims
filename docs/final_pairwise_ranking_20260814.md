# Final Pairwise Cost Ranking — 2026-08-14

## Locked model

- Encoder recipe: `composable_level1_lejepa_any_code`
- Representation schedule: 20 epochs, seed 42
- Frozen data contract: `claims_seed42_v1.json`
- Ranking representation: `patient_representation_pre_sae`
- Ranking-head training split: train
- Variant selection split: validation
- Calibration selection/fitting split: validation
- Final reporting split: sealed test

The test split did not participate in representation selection, ranking-head
training, variant selection, or calibrator fitting.

## Full-schedule encoder validation

The 20-epoch encoder improved the matching four-epoch seed-42 cost result:

| Metric | 4 epochs | 20 epochs |
|---|---:|---:|
| MAE ($) | 1634.74 | 1628.27 |
| RMSE ($) | 2629.32 | 2618.50 |
| WAPE | 35.16% | 35.02% |
| Silhouette | 0.2445 | 0.2291 |

The longer schedule is mildly cost-positive but trades away some unsupervised
cluster separation. Missing-CPT and missing-ICD validation RMSE were $2892.73
and $2893.60 respectively.

## Validation ranking selection

The preregistered selection rule was pairwise accuracy, with Spearman
correlation as the tie-breaker. All calibration values below use leakage-free
five-fold out-of-fold validation predictions.

| Variant | Pair accuracy | Spearman | Top-decile recall | Calibrated RMSE ($) |
|---|---:|---:|---:|---:|
| rolled RankNet | 0.7227 | 0.6113 | 0.4734 | 2465.56 |
| **rolled RankNet + pointwise** | **0.7242** | **0.6132** | 0.4828 | **2449.76** |
| tail RankNet | 0.6936 | 0.5382 | 0.5110 | 2452.17 |
| tail RankNet + pointwise | 0.6971 | 0.5466 | **0.5141** | 2445.87 |
| all-pairs RankNet | 0.7133 | 0.5861 | 0.4514 | 2499.39 |
| LambdaRank | 0.6958 | 0.5445 | 0.5016 | 2492.79 |
| LambdaRank + pointwise | 0.6989 | 0.5497 | 0.5016 | 2477.95 |

`rolled_ranknet_pointwise` wins both locked ranking criteria and validation MAE.
Its validation calibrated MAE/RMSE/WAPE are $1507.39, $2449.76, and 32.42%.
Relative to the encoder's ridge validation probe, the ranking head improves
RMSE by $168.74 and MAE by $120.88.

The tail-weighted variants do improve top-decile recall, but their global
ordering metrics are materially worse. They were therefore not selected.

## Sealed test result

The exact saved `rolled_ranknet_pointwise` head and its validation-fitted
isotonic calibrator were applied without retraining or reselection to 3,274
held-out patients.

| Metric | Validation | Sealed test | Test − validation |
|---|---:|---:|---:|
| Pairwise accuracy | 0.7242 | **0.7152** | -0.0090 |
| Spearman | 0.6132 | **0.5914** | -0.0218 |
| Top-decile recall | 0.4828 | **0.4085** | -0.0742 |
| Calibrated MAE ($) | 1507.39 | **1592.60** | +85.20 |
| Calibrated RMSE ($) | 2449.76 | **2624.16** | +174.40 |
| Calibrated WAPE | 32.42% | **33.90%** | +1.48 pp |

The general ranking signal holds up well: pairwise accuracy loses less than one
percentage point and Spearman remains near 0.59. Tail identification is the
weakest transfer dimension, losing 7.4 percentage points of top-decile recall.
Because the test set is now open, a tail-focused variant must not be selected
using these results; it requires a newly locked validation protocol or future
external data.

## Audit note

The first sealed-test invocation loaded test representations but failed before
score computation because the saved ranking head remained on CPU while the
embeddings were on GPU. No metric was produced. The evaluator was corrected to
move the already locked head to the requested device, then the identical
protocol was rerun. There was no retraining, reselection, or hyperparameter
change.

## Artifacts

- Encoder experiment: `experiments/final_pairwise_20260814/selected_lejepa_anycode_seed42_e20/`
- Validation ranking report: `experiments/final_pairwise_20260814/ranking/validation.json`
- Locked ranking heads: `experiments/final_pairwise_20260814/ranking/heads.pt`
- Selection and hash manifest: `experiments/final_pairwise_20260814/ranking/selection_manifest.json`
- Sealed test report: `experiments/final_pairwise_20260814/ranking/sealed_test.json`
