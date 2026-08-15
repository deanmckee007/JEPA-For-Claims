# Level 1 / Level 2 Representation Sharing Sweep — 2026-08-06

> **Superseded conclusion:** this sweep used `sigreg_weight_lvl2=0`, so its
> shared-composer rows were unregularized MSE rather than a valid no-EMA
> SIGReg test. See `lejepa_noema_sweep_20260806.md` for the corrected
> experiment, which supports removing EMA when reference-style Level-2
> SIGReg is active.

## Answer

The composable implementation shares representations **across the hierarchy**, but does not share all context/target parameters.

- The online `context_encoder_lvl1` produces the CPT and ICD marginal representations used by both the Level-1 prediction objective and the Level-2 claim input.
- `ComposableClaimEncoder` combines those Level-1 states with TTNC into the claim representation consumed by the Level-2 sequence predictor.
- The Level-2 sequence predictor is one online/shared module.
- Under the current hybrid recipe, the Level-1 target encoder is shared with the online encoder, while the target claim composer is a stop-gradient EMA copy.
- The legacy `Level2Encoder` modules are inactive in the composable representation path. They remain only for checkpoint compatibility.

So the concise answer is: **yes, Level 2 consumes the shared Level-1 representation; no, the online and target claim-composition functions are not currently the same parameters.**

## Protocol

All runs used the frozen validation split, the same vocabulary/data-contract hashes, complete-only claims, seed 42 unless noted, four epochs, and the composable Level-1 SIGReg recipe. The test split was not accessed. Metrics are from the frozen-train probe applied to held-out validation representations.

## Sharing matrix

| Level-1 target | Composer target | MAE $ | RMSE $ | WAPE % | Retrieval@5 | Silhouette | Missing CPT RMSE $ | Missing ICD RMSE $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared | EMA | 1,859.77 | 2,959.49 | 40.648 | 0.6984 | 0.1250 | 3,372.41 | 3,073.69 |
| EMA | EMA | 1,859.73 | 2,959.14 | 40.647 | 0.6992 | 0.1232 | 3,370.23 | 3,075.33 |
| shared | shared | 1,873.91 | 2,986.79 | 40.957 | 0.7126 | 0.1071 | 3,534.48 | 4,060.83 |
| EMA | shared | 1,873.12 | 2,985.67 | 40.940 | 0.7143 | 0.1080 | 3,544.28 | 4,094.46 |

Changing the Level-1 target from shared to EMA was effectively neutral. Sharing the claim composer raised retrieval by roughly 0.014–0.016, but worsened the cost probe and raised missing-ICD RMSE by about $1,000. That is the wrong trade for the claims use case, where partial modality robustness is part of the contract.

## EMA-decay sweep

The seed-42 sweep initially made 0.9 look slightly better:

| EMA decay | MAE $ | RMSE $ | Retrieval@5 | Silhouette |
|---:|---:|---:|---:|---:|
| 0.9 | 1,855.88 | 2,954.77 | 0.7001 | 0.1312 |
| 0.99 | 1,858.55 | 2,957.80 | 0.6992 | 0.1162 |
| 0.999 | 1,859.77 | 2,959.49 | 0.6984 | 0.1250 |
| 0.9999 | 1,859.75 | 2,959.69 | 0.6984 | 0.1230 |

Paired seeds 42, 43, and 44 did not confirm it:

| EMA decay | Mean MAE $ | Mean RMSE $ | Mean WAPE % | Mean Retrieval@5 | Mean Silhouette | Missing CPT RMSE $ | Missing ICD RMSE $ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 1,866.81 | 2,962.58 | 40.802 | 0.7015 | 0.1265 | 3,315.47 | 3,253.94 |
| 0.999 | 1,866.41 | 2,960.05 | 40.793 | 0.7025 | 0.1175 | 3,313.31 | 3,252.75 |

Decay 0.999 was slightly better on every averaged cost, retrieval, and missing-modality metric. Decay 0.9 improved only silhouette. The seed-42 difference was noise, so the current 0.999 setting stays.

## Implementation findings and repairs

The sweep exposed that `Config.seed` was doing two jobs: training randomness and frozen split identity. `data_split_seed` now independently pins the frozen split, so paired training seeds can vary without violating the data contract.

The representation path is now emitted into every evaluation JSON as `representation_sharing`, which makes the actual encoder/composer modes auditable.

The audit also found three leaks from the inactive legacy Level-2 encoder: diffusion embedding initialization, token-head conditioning, and autoregressive conditioning. These now use active Level-1 CPT/ICD embeddings plus the active composer TTNC embedding. Graph-pretrained TTNC transfer also reaches the active composer.

Finally, Stage 1 to Stage 2 loading now has an explicit transition mode. Normal checkpoint loads remain strict. The transition permits only the addition of token/diffusion generator state, rejects missing or unexpected non-generator state, and synchronizes generator embeddings only after the trained Stage-1 representation weights are loaded.

## Decision

Retain the current hybrid:

- shared online/target Level-1 encoder;
- EMA target claim composer;
- EMA decay 0.999;
- Level 2 consumes composed Level-1 claim states;
- inactive legacy Level-2 embeddings must not condition generators.

Do not promote either a fully shared composer or EMA decay 0.9 from this sweep.
