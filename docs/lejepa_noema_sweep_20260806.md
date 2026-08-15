# Corrected SIGReg / EMA Experiment — 2026-08-06

## Conclusion

EMA is not necessary once Level-2 SIGReg is actually active and scaled according to the LeJEPA formulation. Across paired training seeds 42, 43, and 44, a single shared claim composer beat the EMA-composer control on cost probes, representation geometry, and averaged missing-modality robustness.

The earlier sharing sweep reached the opposite conclusion because `sigreg_weight_lvl2` was zero. Its shared-composer rows tested unregularized MSE, not no-EMA SIGReg.

## Corrected objective

The opt-in `lejepa_convex` formulation follows the reference LeJEPA statistic:

- characteristic-function integration over `[0, 3]` with Gaussian windowing and trapezoidal weights;
- statistic scaled by the number of valid samples;
- convex mixing rather than adding an arbitrarily scaled regularizer;
- one shared encoder/composer with gradients through prediction and target branches.

For Level 2:

`loss = (1 - lambda) * predictive_mse + lambda * mean(SIGReg(prediction), SIGReg(target))`

The separate CPT and ICD marginal SIGReg remains active, preserving typed Level-1 composability rather than factorizing their joint concatenation.

Reference: [LeJEPA minimal implementation](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md).

## Protocol

All runs used four epochs, complete-only claims, the frozen seed-42 validation split, identical vocabulary and data-contract hashes, and `patient_representation_pre_sae`. Training seeds varied independently from the split seed. The test split was not accessed.

## Lambda search without EMA

| Lambda | MAE $ | RMSE $ | WAPE % | Retrieval@5 | Silhouette | Missing CPT RMSE $ | Missing ICD RMSE $ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 1,846.27 | 2,931.69 | 40.353 | **0.7044** | 0.1870 | 3,135.46 | **3,040.46** |
| 0.02 | 1,836.86 | 2,912.19 | 40.147 | 0.6867 | 0.2107 | **3,094.96** | 3,077.56 |
| 0.05 | **1,830.20** | **2,903.51** | **40.001** | 0.6867 | **0.2235** | 3,110.16 | 3,129.81 |

Increasing lambda through 0.05 consistently improved the intact-claim cost probe and silhouette. Retrieval favored 0.005, while deletion robustness was mixed. Lambda 0.05 is the best tested primary-metric setting, but it was also the upper boundary of this compact search; it should not be described as a global optimum.

## Matched EMA comparison at lambda 0.05

| Composer | Mean MAE $ | Mean RMSE $ | Mean WAPE % | Mean Retrieval@5 | Mean Silhouette | Missing CPT RMSE $ | Missing ICD RMSE $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared, no EMA | **1,851.33** | **2,932.43** | **40.463** | 0.6896 | **0.2186** | **3,224.10** | **3,130.37** |
| EMA target | 1,868.76 | 2,971.63 | 40.844 | **0.6955** | 0.1968 | 3,241.88 | 3,284.56 |

The no-EMA model improved mean RMSE by $39, MAE by $17, silhouette by 0.022, missing-CPT RMSE by $18, and missing-ICD RMSE by $154. EMA improved Retrieval@5 by 0.0059.

No-EMA won intact-claim RMSE in every paired seed:

| Seed | Shared RMSE $ | EMA RMSE $ | Shared minus EMA $ |
|---:|---:|---:|---:|
| 42 | 2,903.51 | 2,972.57 | -69.06 |
| 43 | 2,957.72 | 2,983.20 | -25.48 |
| 44 | 2,936.06 | 2,959.13 | -23.07 |

## Comparison with the old production candidate

The previous unregularized EMA candidate averaged $2,960.05 RMSE, 0.1175 silhouette, $3,313.31 missing-CPT RMSE, and $3,252.75 missing-ICD RMSE across the same seeds. The corrected shared SIGReg model improves all four to $2,932.43, 0.2186, $3,224.10, and $3,130.37 respectively. Retrieval falls from 0.7025 to 0.6896.

## Implementation decision

The historical `composable_level1_sigreg` recipe remains unchanged for reproducibility. A new explicit candidate recipe, `composable_level1_lejepa`, captures the corrected configuration:

- shared Level-1 encoder;
- shared claim composer, no EMA target in the active path;
- reference-style convex SIGReg;
- Level-2 lambda 0.05;
- existing typed marginal SIGReg and missing-modality training behavior preserved.

EMA support remains available as an ablation/control, but it is no longer the recommended composable Level-2 path.
