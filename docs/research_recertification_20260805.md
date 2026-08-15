# JEPA claims recertification and composability pilot (2026-08-05)

## Protocol

- Data contract: `artifacts/data_contracts/claims_seed42_v1.json`
- Contract hash: `349d6b49a743282dbeb84b105338674cada85ee78b3fe2f2229d9ad5b4492644`
- Vocabulary hash: `7554200552fdf0e3609d927cfadb68c5444a656df4c77d929cf2ff71341893f0`
- Frozen processed split sizes: train 10,659; validation 2,324; test 2,391.
- Every representation comparison used seed 42, four training epochs, the
  deterministic evaluation collator, and a Ridge probe fit on train and scored
  on validation. The test split was not read during representation selection.
- `patient_representation_pre_sae` is the canonical downstream representation.

## Repaired anchor matrix

| Anchor | Validation RMSE | MAE | WAPE | Retrieval@5 | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repaired grounded, Level-2 SIGReg 0 | $2,988.98 | $1,903.21 | 41.60% | 0.6820 | 0.2029 |
| Repaired grounded, Level-2 SIGReg 0.1 | $2,989.19 | $1,903.01 | 41.59% | 0.6846 | 0.2098 |
| VICReg | $3,000.51 | $1,907.27 | 41.69% | 0.6859 | 0.1528 |

The Level-2 SIGReg coefficient is effectively neutral at this horizon. The
grounded SIGReg-0 recipe remained the base for the composability pilot.

## Composable Level-1 pilot

Level-2 receives the composed Level-1 claim state. CPT and ICD have separate
projected residual paths plus an explicit shared interaction path based on both
marginals. Regularizers are applied independently to the CPT and ICD marginals;
the dependent concatenated joint is never forced to factorize.

| Marginal regularizer | Validation RMSE | MAE | WAPE | Retrieval@5 | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | $2,983.71 | $1,877.15 | 41.03% | 0.6988 | 0.1126 |
| SIGReg | **$2,959.49** | **$1,859.77** | **40.65%** | 0.6984 | **0.1250** |
| Wristband, weight 0.1 | $3,035.36 | $1,934.66 | 42.28% | **0.7083** | 0.0953 |

SIGReg is selected. Wristband uses the reflected joint repulsion, radial OT,
W2 moment term, and deterministic null calibration from the
[`ml-tidbits` Wristband formulation](https://github.com/mvparakhin/ml-tidbits),
but weight 0.1 is rejected at this training horizon. It improved local retrieval
and ablation geometry while over-regularizing the cost-relevant representation.

### Missing-modality validation

The same full-input train probe was applied without refitting to representations
with one code modality zeroed.

| Regularizer | Missing CPT RMSE | Full/missing CPT cosine | Missing ICD RMSE | Full/missing ICD cosine |
| --- | ---: | ---: | ---: | ---: |
| None | $3,315.93 | 0.8893 | $3,105.51 | 0.8991 |
| SIGReg | $3,372.41 | 0.8872 | $3,073.69 | 0.9032 |
| Wristband 0.1 | $3,506.21 | **0.9207** | **$3,034.62** | **0.9189** |

The comparison separates geometric stability from useful cost signal. Wristband
makes ablated states closer to full states, but that does not improve missing-CPT
cost prediction. Missing CPT remains the principal robustness gap.

## Pairwise cost-ranking head

After selecting composable SIGReg on validation, a 128-hidden-unit RankNet head
was fit on frozen train representations for 30 fixed epochs. An increasing
isotonic calibrator was fit on validation scores. The following metrics are from
the single untouched test evaluation; no test tuning or retraining followed.

- Pairwise accuracy: **72.23%** over 499,957 sampled non-tied pairs.
- Spearman correlation: **0.6120**.
- Top-decile recall: **36.25%** (3.625 times random overlap).
- Monotonic dollar calibration: MAE $1,915.02; RMSE $3,022.71; WAPE 40.95%.
- RankNet loss: 0.5719 first epoch to 0.4751 final epoch.

Artifacts:

- `experiments/composable_level1_20260805/composable_level1_sigreg_seed42_e4/encoder.ckpt`
- `experiments/pairwise_ranking_20260805/sigreg_pairwise_head.pt`
- `experiments/pairwise_ranking_20260805/sigreg_test_metrics.json`

## Interpretation and stop rule

This is a compact, single-seed, four-epoch pilot, not a final variance estimate.
It is sufficient to accept the composable dataflow and retain typed-marginal
SIGReg as the next research baseline. Do not promote Wristband at weight 0.1.
The next bounded experiment, if desired, is missing-CPT modality dropout or a
lower Wristband weight; neither is justified as part of this completed matrix.
