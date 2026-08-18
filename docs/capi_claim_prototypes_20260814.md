# CAPI-style composed-claim prototype experiment

## Question

Can an auxiliary CAPI-style objective recover semantic neighborhood quality by
training Level 2 to predict balanced pseudo-categories of the complete composed
future claim, without reintroducing EMA or replacing the continuous LeJEPA
representation?

## Implementation

- The canonical continuous representation remains the shared
  `ComposableClaimEncoder` output before the SAE.
- Complete future-claim targets are detached before entering a separately
  learned clustering head.
- A distinct student head predicts position-wise Sinkhorn assignments from the
  Level-2 future-claim prediction.
- Only target slots with positive Level-2 loss weight participate. The two
  zero-weight observed slots are excluded.
- Prototype construction uses an isolated RNG scope, so prototype count does
  not perturb initialization of later model components.
- No EMA target is used.

The prototype loss is added under the existing Level-2 precision term, avoiding
a new learned loss variance or a change in the number of homoscedastic tasks.

## Protocol

- Frozen data contract: `claims_seed42_v1.json`
- Contract hash: `349d6b49a743282dbeb84b105338674cada85ee78b3fe2f2229d9ad5b4492644`
- Vocabulary hash: `7554200552fdf0e3609d927cfadb68c5444a656df4c77d929cf2ff71341893f0`
- Validation split only; test split was not accessed
- Complete-only claim cohort
- Four epochs, training seed 42, split seed 42
- Continuous anchor: LeJEPA SIGReg lambda 0.05, shared Level-1 encoder and
  shared claim composer
- Full 2,324-patient validation evaluation, including missing CPT and missing
  ICD

## Direct paired result: K=16

The weight-zero control and weighted treatments instantiate the same prototype
module under the same isolated seed. These are the causal comparisons.

| Prototype weight | MAE ($) | RMSE ($) | WAPE (%) | Retrieval@5 | Silhouette | Prototype top-1 | Effective K | Missing CPT RMSE ($) | Missing ICD RMSE ($) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1841.14 | 2914.69 | 40.2404 | 0.68847 | 0.20811 | 0.1015 | 15.31 | 3104.45 | 3058.12 |
| 0.05 | 1840.77 | 2913.96 | 40.2324 | 0.68804 | 0.20686 | 0.2298 | 15.16 | 3105.35 | 3059.32 |
| 0.10 | 1840.57 | 2913.70 | 40.2281 | 0.68675 | 0.20846 | 0.2324 | 15.16 | 3106.48 | 3060.80 |

At weight 0.10 relative to the paired control:

- MAE improved by $0.56 and RMSE by $0.99.
- Retrieval@5 declined by 0.00172.
- Silhouette increased by only 0.00035.
- Missing-CPT and missing-ICD RMSE worsened by $2.02 and $2.68.

The prototype task itself clearly learned: top-1 assignment prediction rose
from 10.2% in the weight-zero control to 23.2%, with 15.2/16 effective
prototypes. That competence did not transfer into a material improvement in the
canonical patient representation.

## Prototype-count pilots

Initial K=16/64/256 pilots all trained stably and used nearly the full codebook.
They were run before prototype initialization was isolated from the global RNG,
so their downstream metric deltas are diagnostic rather than causal. Effective
prototype counts were 15.1, 59.1, and 239.6, respectively. Increasing K reduced
top-1 predictability and did not reveal a downstream advantage; K=256 also
weakened silhouette and modality-ablation stability.

## Decision

Do not promote the CAPI auxiliary into the canonical recipe. Keep it as an
opt-in research branch. It is stable, non-collapsed, and semantically
predictable, but the tested formulation mostly learns the prototype head rather
than improving the exposed patient representation. The effect is far below the
threshold that would justify a multi-seed confirmation.

A worthwhile follow-up would need to change the mechanism, not merely tune the
weight: for example, predict prototypes at multiple future horizons with
independent cross-attention queries, or use prototype consistency specifically
across partial-modality views. Those hypotheses were not tested here.

## Verification

The complete repository suite passes: 140 tests.
