# Claims JEPA improvement follow-up — 2026-08-05

## Protocol

- Frozen patient assignments and train-only vocabulary from
  `artifacts/data_contracts/claims_seed42_v1.json`.
- Representation and ranking-head selection used validation only. After the
  configuration was locked, exactly one held-out test evaluation was run; no
  test-driven retry or model reselection was performed.
- Canonical representation source: `patient_representation_pre_sae`.
- Common-cohort comparisons use the historical complete-claim cohort and
  remove partial claims from the evaluated sequence as well as cohort selection.
- Expanded-cohort comparisons retain TTNC-delimited claims containing CPT or ICD.

## Data-path finding

The historical parser discarded 163,152 ICD-only claims (6.86% of claims).
Retaining claims with either code family increased eligible patients from
15,374 to 21,512. The frozen split and vocabulary artifacts were unchanged.

## Representation matrix

### Exact historical complete-claim validation cohort (n=2,324)

| Treatment | MAE ($) | RMSE ($) | WAPE (%) | Retrieval@5 | Missing-CPT RMSE ($) |
|---|---:|---:|---:|---:|---:|
| Complete-only SIGReg, epoch 10 | 1,865.03 | 2,958.87 | 40.763 | 0.7027 | 3,384.26 |
| Any-code SIGReg, epoch 10 | **1,845.22** | **2,938.23** | **40.330** | **0.7100** | **3,496.69** |
| Any-code + CPT/ICD context dropout 0.07/0.03 | 1,847.98 | 2,938.88 | 40.390 | 0.7087 | 3,521.06 |

Natural partial-claim retention is selected. It improves the three primary
cost metrics and retrieval on the identical cohort. The tested dropout rates
are rejected: they slightly reduce full-data quality and do not repair the
complete-cohort missing-CPT stress test.

The complete-only epoch series plateaued. Epoch 2 had the best MAE/WAPE
($1,855.18 / 40.547%), while epoch 10 had the best RMSE ($2,958.87); the
differences were negligible compared with the data-policy treatment.

### Natural expanded validation cohort (n=3,188)

| Treatment | MAE ($) | RMSE ($) | WAPE (%) | Retrieval@5 | Missing-CPT RMSE ($) |
|---|---:|---:|---:|---:|---:|
| Any-code SIGReg | **1,652.13** | **2,649.99** | **35.534** | 0.7748 | 3,868.52 |
| Any-code + context dropout | 1,659.29 | 2,651.73 | 35.688 | **0.7760** | **3,772.15** |

Dropout improves the expanded-cohort missing-CPT stress result by about $96,
but the common-cohort result reverses and all primary full-data metrics worsen.
It is therefore not selected at these rates.

## Ranking-head validation comparison

Ranking heads were fit on 15,050 expanded-cohort train patients and evaluated
on all 3,188 frozen validation patients. Dollar calibration uses stratified
five-fold out-of-fold isotonic predictions. Test was not loaded.

| Variant | Pair acc. | Spearman | Top-decile recall | Cal. MAE ($) | Cal. RMSE ($) | Cal. WAPE (%) |
|---|---:|---:|---:|---:|---:|---:|
| Rolled RankNet | **0.7199** | **0.6020** | **0.4796** | **1,512.05** | **2,432.19** | **32.522** |
| Rolled RankNet + pointwise | 0.7185 | 0.5986 | **0.4796** | 1,513.95 | 2,434.73 | 32.563 |
| Tail-weighted RankNet | 0.6923 | 0.5356 | **0.4796** | 1,606.47 | 2,457.55 | 34.552 |
| Tail-weighted + pointwise | 0.6945 | 0.5399 | 0.4671 | 1,592.64 | 2,452.00 | 34.255 |
| All-pairs RankNet | 0.7016 | 0.5576 | 0.4263 | 1,592.20 | 2,504.41 | 34.245 |
| LambdaRank | 0.6770 | 0.4979 | 0.4608 | 1,696.60 | 2,545.12 | 36.491 |
| LambdaRank + pointwise | 0.6845 | 0.5142 | 0.4451 | 1,657.61 | 2,532.63 | 35.652 |

Plain rolled RankNet remains selected. Dense variants received 14.18 million
pairs versus 451,500 for the rolled baseline, but validation quality worsened
across every metric. Arbitrary patient groups do not provide the query structure
that makes delta-NDCG Lambda weights well behaved, and the dense objectives
overfit this small head.

## Selected configuration

- Recipe: `composable_level1_sigreg_any_code`
- Claim inclusion: TTNC plus at least one of CPT or ICD
- Context modality dropout: disabled
- Representation: pre-SAE patient state
- Ranking head: rolled RankNet
- Test status: accessed once after selection was locked; no retries

## Locked held-out test (n=3,274)

| Pair acc. | Spearman | Top-decile recall | Cal. MAE ($) | Cal. RMSE ($) | Cal. WAPE (%) |
|---:|---:|---:|---:|---:|---:|
| 0.7101 | 0.5788 | 0.4390 | 1,595.06 | 2,597.91 | 33.951 |

The ranking head was fit on the 15,050-patient train split. Monotonic dollar
calibration was fit on the 3,188-patient validation split, and the table above
is the sole report from the 3,274-patient test split. The artifact records
`test_accessed: true` and the explicit held-out reporting authorization.

## Artifacts

- Selected encoder: `experiments/composable_level1_followup_20260805/composable_sigreg_anycode_seed42_e10/encoder.ckpt`
- Natural-cohort representation report: `experiments/composable_level1_followup_20260805/composable_sigreg_anycode_seed42_e10/representation_eval.json`
- Common-cohort representation report: `experiments/composable_level1_followup_20260805/composable_sigreg_anycode_seed42_e10/representation_eval_common_complete.json`
- Initial ranking matrix: `experiments/pairwise_ranking_followup_20260805/anycode_validation_variants.json`
- Tail-aware ranking matrix: `experiments/pairwise_ranking_followup_20260805/anycode_validation_tail_variants.json`
- Locked held-out ranking result: `experiments/pairwise_ranking_followup_20260805/anycode_rolled_ranknet_final_test.json`
- Locked held-out ranking head: `experiments/pairwise_ranking_followup_20260805/anycode_rolled_ranknet_final_test_head.pt`
