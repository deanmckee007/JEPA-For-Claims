# Core Architecture Ablations — 2026-08-14

## Decision

Use `composable_level1_lejepa_any_code` as the canonical composable recipe. It
uses shared (non-EMA) Level-1 and Level-2 targets, feeds the composed Level-1
claim representation into Level 2, retains naturally partial claims, and gives
the Level-1 cross-modal predictive objective an explicit weight of `0.1`.

Keep the existing 50-claim history, five CPT and five ICD tokens per claim,
moment pooling, untied TTNC embeddings, and dense Level-2 decoder. The new
architectures remain available as isolated config ablations, but none cleared
the joint cost, geometry, and missing-modality gate.

No test data was used. All selection used the frozen validation split and the
vocabulary/split contract in
`artifacts/data_contracts/claims_seed42_v1.json`.

## What changed in the architecture

- The Level-1 predictive loss now has its own
  `level1_predictive_weight`; composition is no longer disabled when this loss
  is zero.
- Level 2 explicitly consumes `ComposableClaimEncoder` outputs. Level 1 and
  Level 2 use shared targets under the LeJEPA/SIGReg recipe; EMA is not required.
- Claim representation evaluation now measures CPT-to-ICD and ICD-to-CPT
  retrieval, paired cosine, count/rarity strata, and missing-modality behavior.
- Optional claim poolers include learned-query attention and typed
  self-attention, with optional rarity bias.
- Optional Level-2 cross-attention queries, masked historical-claim JEPA,
  multi-hypothesis future prediction, and TTNC sharing/ordinal placement are
  implemented and tested.
- Optimizer construction deduplicates tied parameters and includes all new
  auxiliary heads.
- Strict checkpoint validation now includes history and per-claim token caps.

## Multiseed Level-1 result

Four-epoch runs used seeds 42, 43, and 44. Values are mean ± sample standard
deviation.

| Level-1 weight | RMSE ($) | Silhouette | CPT→ICD hit@5 | ICD→CPT hit@5 |
|---:|---:|---:|---:|---:|
| 1.0 | 2634.70 ± 5.36 | 0.2317 ± 0.0115 | 0.2690 ± 0.0022 | 0.04875 ± 0.00037 |
| 0.1 | 2634.72 ± 5.34 | 0.2381 ± 0.0101 | 0.2692 ± 0.0018 | 0.04883 ± 0.00042 |
| 0.0 (seed 42 diagnostic) | 2629.32 | 0.2445 | 0.00562 | 0.00684 |

The important separation is conceptual: patient cost is insensitive to the
Level-1 predictive loss, while cross-modal claim geometry is not. Turning the
loss off leaves composition intact but destroys the learned CPT/ICD mapping.
A weight of `0.1` is therefore sufficient and materially safer than zero.

## Architecture screens

All screens below were two epochs on seed 42 and are compared with the matching
two-epoch anchor (RMSE $2910.29, silhouette 0.2215). Screens were promoted only
when they improved cost without an unacceptable geometry or missing-modality
regression.

| Arm | RMSE ($) | Silhouette | Missing-CPT RMSE ($) | Decision |
|---|---:|---:|---:|---|
| Token cap 10, moments | 2939.37 | 0.2343 | 3437.58 | Reject: cost worse |
| Token cap 10, query pooling | 2892.64 | 0.2004 | 3325.75 | Reject: geometry loss |
| Query + rarity | 2892.65 | 0.2003 | 3325.47 | Reject: rarity had no effect |
| Token self-attention | 2925.56 | 0.2019 | 4280.51 | Reject |
| Shared TTNC table | 2910.46 | 0.2160 | 3313.24 | No clear gain |
| TTNC composer only | 2913.06 | 0.2291 | 4462.13 | Reject: missing-CPT failure |
| TTNC sequence only | 2996.99 | 0.1711 | 3201.35 | Reject |
| Ordinal TTNC | 2889.90 | 0.2105 | 5080.19 | Reject: missing-CPT failure |
| Cross-attention decoder | 2887.56 | 0.1971 | 4000.20 | Reject: robustness/geometry |
| Decoder + masked claim | 2887.12 | 0.1874 | 4099.80 | Reject |
| Decoder + four hypotheses | 2885.76 | 0.1967 | 4073.51 | Reject |
| Decoder + both auxiliaries | 2883.80 | 0.1977 | 4198.15 | Reject |
| History 100 | 2877.79 | 0.1944 | 3200.18 | Promote to multiseed only |

The query-attention rarity toggle was effectively inert. That is useful in
itself: with the current normalized rarity scores and attention formulation,
the bias is too weak or too uniform to change optimization.

## History-length promotion

At four epochs over three seeds, history 100 improved mean RMSE by $13.25 and
mean MAE by $7.76. It simultaneously reduced silhouette by 0.0231 on every
seed and worsened mean missing-CPT RMSE by $519.35. Missing-CPT RMSE also had a
$1478 sample standard deviation, including a $5895 result on seed 43.

The longer history is therefore a cost-specialized arm, not the canonical
representation. The direct claim-retrieval numbers are not compared across
history caps because changing the cap changes the sampled claims.

## Truncation audit

- 35,976 patients; claim-count p50 15, p90 271.5, p99 363, max 364.
- 11,851 patients (32.94%) exceed the canonical 50-claim history.
- CPT codes per claim: p50 1, p90 2, p99 8, max 64. Cap five drops 6.88% of
  CPT tokens.
- ICD codes per claim: p50 1, p90 6, p99 10, max 37. Cap five drops 14.72% of
  ICD tokens.

The cap is a real information bottleneck, especially for ICD and long-history
patients. The simple cap-10/history-100 expansions did not provide a stable
joint benefit, so future work should use selective compression or hierarchical
memory rather than only increasing dense sequence length.

## Remaining high-value ablations

The core architecture is no longer missing an obvious composability, collapse
prevention, target-sharing, claim-query, or multimodality mechanism. The best
next work is narrower:

1. Learn a compressed long-history memory while retaining the recent 50 claims.
2. Make rarity supervision explicit (rare-code reconstruction/retrieval loss)
   rather than relying on a small attention-logit bias.
3. Train the pairwise cost-ranking head on the selected recipe, then evaluate
   held-out pairwise accuracy, Spearman correlation, top-decile recall, and
   monotonic dollar calibration.
4. Repeat the selected recipe at the full intended training schedule before a
   single final test evaluation.

## Artifacts

- Full per-run results: `experiments/core_architecture_ablations_20260814/summary.json`
- Compact table: `experiments/core_architecture_ablations_20260814/summary.csv`
- Machine-readable decision: `experiments/core_architecture_ablations_20260814/decision_summary.json`
- Truncation audit: `experiments/core_architecture_ablations_20260814/truncation_audit.json`
- Run logs, checkpoints, and evaluations are in each named subdirectory.
