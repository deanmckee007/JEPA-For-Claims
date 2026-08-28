# LeVJEPA Patient-View Port

## Scope

`levjepa_patient_views` is an opt-in Stage-1 representation recipe. It keeps
the selected composable claims encoder and its light CPT/ICD predictive loss,
but replaces the Level-2 next-claim objective with LeVJEPA-style agreement
between one complete context history and independently thinned local views.
Existing recipes and downstream representation sources are unchanged.

## Objective

- One shared encoder/patient sequence model; no EMA target or stop-gradient.
- Two local views by default.
- Each local view drops valid context claims independently at probability 0.3.
- The latest valid context claim is always retained.
- Dropped claims are removed from CPT, ICD, and TTNC inputs and are not
  reconstructed.
- A `patient_dim -> 2048 -> 256` projector with BatchNorm and GELU is used only
  by the pretraining loss.
- The Level-2 loss is projected global/local MSE plus additive SIGReg with
  lambda 0.02, 1,024 random directions, and 17 quadrature knots over `[0, 3]`.
- Distributed runs broadcast the random directions and all-reduce only the
  empirical characteristic-function sums and sample count, so SIGReg is
  computed over the global batch without gathering patient embeddings.
- An evaluation-only Polyak shadow is updated every 32 optimizer batches at
  decay 0.9999. Validation and standalone representation evaluation use the
  shadow after its first update, then restore the online weights before
  checkpoints are written.
- `patient_representation_pre_sae` remains the canonical downstream feature.

This is a claims-safe adaptation rather than a literal single-loss video port:
the validated Level-1 CPT/ICD predictive objective remains active at weight
0.1 because removing it previously destroyed cross-modal claim geometry.

## Run

```powershell
python scripts/train.py `
  --recipe levjepa_patient_views `
  --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet `
  --data-contract artifacts/data_contracts/claims_seed42_v1.json `
  --representation-pretrain-epochs 4 `
  --generator-train-epochs 0 `
  --joint-train-epochs 0
```

For a validation-only anchor comparison:

```powershell
python scripts/run_recipe_sweep.py `
  --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet `
  --recipes composable_level1_lejepa_any_code levjepa_patient_views `
  --representation-pretrain-epochs 4
```

Claim-drop sweeps can use repeated `--set` overrides, for example
`--set levjepa_claim_drop_ratio=0.5`.

The optional attentive probe learns only a query-attention readout over frozen
per-claim sequence states; it does not fine-tune the claims encoder:

```powershell
python scripts/evaluate_representations.py `
  --checkpoint <checkpoint.ckpt> `
  --recipe levjepa_patient_views `
  --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet `
  --data-contract artifacts/data_contracts/claims_seed42_v1.json `
  --attentive-probe
```

## Verification

- Full unit suite: 210 passed.
- One-batch CPU Lightning smoke completed for the new named recipe.
- Smoke diagnostics were finite and reported a retained-claim fraction of
  0.764 on four-claim synthetic histories, consistent with latest-claim
  preservation under 30% independent dropping.

## Real-data seed-42 pilot

The four-epoch pilot used the established 21,512-patient claims dataset,
frozen validation contract `349d6b49...`, and Polyak evaluation weights. The
held-out validation split contains 3,188 patients.

| Model/readout | MAE ($) | WAPE (%) | RMSE ($) | log RMSE | TTNC retrieval@5 | Silhouette |
|---|---:|---:|---:|---:|---:|---:|
| Selected LeJEPA any-code, 20 epochs, Ridge | 1,628.27 | 35.02 | 2,618.50 | 0.8373 | 0.7544 | 0.2291 |
| Selected LeJEPA any-code, 20 epochs, attentive probe | 1,541.23 | 33.15 | 2,531.03 | 0.8177 | n/a | n/a |
| LeVJEPA patient views, 4 epochs, Ridge | 1,659.35 | 35.69 | 2,642.69 | 0.8395 | 0.7732 | 0.1290 |
| LeVJEPA patient views, 4 epochs, attentive probe | 1,559.00 | 33.53 | 2,510.01 | 0.8262 | n/a | n/a |

The pooled LeVJEPA representation does not beat the selected incumbent under
the same Ridge protocol after four epochs: cost metrics are about 1-2% worse,
although TTNC retrieval@5 is 1.88 percentage points higher. The attentive
readout is promising (about 4.3% lower MAE/WAPE than the incumbent Ridge
result), but the matched incumbent attentive probe reaches lower MAE, WAPE,
and log RMSE. LeVJEPA has a slightly lower dollar RMSE under attention. This
means the attentive probe is useful across both encoders, while the four-epoch
LeVJEPA recipe is not yet a promotion candidate on cost metrics.

Artifacts are under
`experiments/levjepa_patient_views_20260828/levjepa_patient_views_seed42_e4/`.

## High-cost tail extension

The patient-view representation was subsequently connected to the established
validation-only high-cost proxy: train a balanced top-1.5% tail head, rank all
3,188 validation histories, and select exactly 48. Mortality remains unavailable,
so this is not yet the historical death-plus-high-cost endpoint. Test was not
accessed.

### Seed-42 bridge and attentive readout

Against the selected 20-epoch LeJEPA incumbent, the four-epoch LeVJEPA online
embedding retrieved the same number of true tail members but ranked the tail
substantially better:

| Frozen input + downstream head | Hits / 48 | Precision | PR-AUC | NDCG@1.5% | Cost capture |
|---|---:|---:|---:|---:|---:|
| Incumbent embedding, logistic | 9 | 0.188 | 0.103 | 0.174 | 3.02% |
| LeVJEPA online embedding, logistic | 9 | 0.188 | **0.151** | **0.248** | **3.07%** |
| Raw + incumbent, boosted tail | 10 | 0.208 | 0.133 | 0.226 | 3.33% |
| Raw + LeVJEPA online, boosted tail | 10 | 0.208 | **0.239** | **0.339** | **3.34%** |

The commit's query-attention readout did not improve tail ranking. Over three
downstream seeds, LeVJEPA online sequence-only attention averaged 0.160
precision and 0.105 PR-AUC; raw+attention averaged 0.208 and 0.112. The pooled
raw+embedding boosted head averaged 0.201 precision and 0.190 PR-AUC. Attention
occasionally moved one extra member across the cutoff but degraded the broader
ranking.

The evaluation weight choice matters. Polyak pooled embeddings reached 0.229
precision under a linear tail probe but only 0.112 PR-AUC. Online pooled
embeddings reached 0.188 precision and 0.151 PR-AUC; raw+online reached 0.201
precision, 0.190 PR-AUC, and 0.293 NDCG over downstream seeds. The online
representation is therefore used for tail experiments, while the documented
global representation evaluation remains Polyak-based.

### Sparse patient views and representation fusion

`levjepa_patient_views_reprelu` changes only the canonical Level-2 link to
RepReLU. Patient-view sampling, projected SIGReg, Level-1 predictive geometry,
and Polyak settings remain matched. Its seed-42 Polyak evaluation is worse than
dense LeVJEPA:

| Representation | Cost MAE | Cost RMSE | TTNC retrieval@5 | Silhouette |
|---|---:|---:|---:|---:|
| Dense LeVJEPA | **1,659.35** | **2,642.69** | **0.7732** | **0.1290** |
| RepReLU LeVJEPA | 1,712.29 | 2,678.11 | 0.7594 | 0.1049 |

At the online seed-42 tail cutoff, raw+dense+sparse LeVJEPA retrieves 12/48,
versus 10 raw+dense and 11 raw+sparse, but PR-AUC falls from 0.239 for
raw+dense to 0.166. Dense+sparse embeddings without raw history also retrieve
12/48 and improve PR-AUC to 0.192 over dense-only 0.149 and sparse-only 0.131.
As in the earlier cap-32 fusion, sparsity adds nonlinear embedding-only and
cutoff-specific signal but does not improve the full tail ordering once raw
history is present.

A cross-recipe seed-42 fusion of cap-32 dense and LeVJEPA embeddings reaches
12/48, 0.175 PR-AUC, 0.317 NDCG, and 3.55% cost capture without raw features.
Adding both to raw history retrieves only 10/48; raw+cap-32 remains best at
12/48. Rank fusion is preferable to feature concatenation for broader 3-5%
budgets, but does not improve the exact seed-42 1.5% cutoff.

### Three-encoder-seed tail replication

Dense LeVJEPA was trained for four epochs at encoder seeds 42-44 on the same
frozen seed-42 split. The table reports mean precision / future-cost capture:

| Ranking | Top 0.5% | Top 1% | Top 1.5% | Top 2% | Top 3% | Top 5% |
|---|---:|---:|---:|---:|---:|---:|
| Raw + LeVJEPA tail | **.229/.014** | .167/.024 | .222/.035 | .260/.044 | .306/.061 | .333/.092 |
| 25% raw-cost + 75% LeVJEPA rank | .188/.014 | **.188/.026** | **.229/.036** | **.276/.045** | **.312/.064** | .360/.098 |
| 50% raw-cost + 50% LeVJEPA rank | .188/.014 | .167/.026 | .201/.036 | .260/.045 | .299/.063 | .379/.098 |
| 75% raw-cost + 25% LeVJEPA rank | .188/.014 | .156/.026 | .201/.036 | .260/.045 | .299/.063 | **.390/.099** |
| Raw cost | .167/.013 | .104/.024 | .201/.035 | .240/.045 | .288/.062 | .354/.097 |

At 1.5%, the 25/75 blend retrieves 10, 12, and 11 true members by encoder
seed. Its aggregate precision is 0.229 +/- 0.021, PR-AUC 0.190, NDCG 0.298,
and cost capture 3.64%. The previous cap-32 50/50 raw-cost blend achieved 0.222
+/- 0.012 precision, 0.184 PR-AUC, 0.297 NDCG, and 3.68% capture. LeVJEPA is a
small Pareto update: about one-third of an additional true member per run and
better ranking quality for 0.04 percentage points less captured cost.

The validation candidate for the historical 1.5% operating point is therefore
the predeclared 25% raw-cost / 75% online-LeVJEPA-tail percentile-rank blend,
with the prior cap-32 blend retained as the slightly higher-capture comparator.
Neither should enter test or operational interpretation until stable member
IDs, dates, mortality labels, and a prospective boundary are available.

### Tail label efficiency

The raw-history, cap-32, and LeVJEPA tail heads were next retrained with 1%,
5%, 10%, 25%, and 100% of the training labels. Subsets were independently
stratified per encoder seed; the fixed validation set and 1.5% operating point
were unchanged. At the smallest budgets the learned representations provide
the clearest advantage over engineered history:

| Boosted input | 1% labels precision / AP | 5% | 10% | 25% | 100% |
|---|---:|---:|---:|---:|---:|
| Raw history | .035 / .030 | .090 / .079 | .118 / .080 | .181 / .140 | .188 / .135 |
| LeVJEPA | .076 / .063 | .111 / .079 | .104 / .073 | .160 / .108 | .194 / .144 |
| Raw + cap-32 | .069 / .044 | .111 / .080 | **.139 / .110** | .181 / .130 | **.222 / .203** |
| Raw + LeVJEPA | **.083 / .054** | **.132 / .092** | .111 / .085 | **.201 / .122** | .208 / .170 |

One percent corresponds to about 152 labeled members and only three positive
examples. In that regime raw + LeVJEPA more than doubles raw-history precision;
the advantage persists at 5%. The ordering is non-monotonic at intermediate
budgets, and raw + cap-32 remains the best full-label model under this fixed,
no-early-stopping boosted-head protocol. The defensible conclusion is therefore
that LeVJEPA has a strong low-label tail advantage, not universal dominance.

### Missing-history robustness

A frozen representation audit removed 0%, 10%, 30%, 50%, or 70% of context
claims at validation time. CPT, ICD, and TTNC were removed jointly; held-out
future claims were never changed; and the latest context claim was always
retained. Each logistic or boosted tail head was fit once on complete training
histories and reused at every corruption level. The nominal 70% drop retained
32.3% of claims after protecting the latest claim.

| Frozen input + head | No drop precision / AP | 30% drop | 70% drop |
|---|---:|---:|---:|
| Cap-32, boosted | .188 / .127 | .146 / .109 | .160 / .101 |
| LeVJEPA, boosted | **.194 / .144** | **.181 / .124** | **.194 / .123** |
| Cap-32, logistic | .097 / .078 | .125 / .070 | .069 / .056 |
| LeVJEPA, logistic | **.153 / .108** | **.160 / .100** | **.153 / .092** |

At 70% nominal dropout, LeVJEPA preserves boosted precision exactly and loses
0.020 AP, versus a 0.028 precision and 0.026 AP loss for cap-32. Its cost
capture is also unchanged within 0.03 percentage points. The matched linear
heads show the same qualitative result. This is direct evidence that the
patient-view objective produces a more missing-history-tolerant patient state.

### Frozen next-claim generation

The existing copy-plus-residual decoder was trained on frozen pooled patient
states. It starts from the last observed claim and learns code and TTNC residuals,
so the encoder must add information beyond a strong persistence baseline. A
seed-42 shuffled-representation control confirmed alignment: at full labels,
LeVJEPA beat shuffle by 0.145 CPT AP, 0.029 ICD AP, 0.041 TTNC accuracy, and
$438 cost MAE. At 10% labels it retained a 0.043 CPT AP and $290 MAE advantage.

The aligned comparison was then repeated across independently pretrained
cap-32 and LeVJEPA encoder seeds 42--44, with decoder seed matched to encoder
seed:

| Labels | Frozen recipe | CPT AP | ICD AP | TTNC accuracy | Cost MAE | Cost RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 10% | Cap-32 | **.334** | .438 | **.591** | **$1,994** | $3,027 |
| 10% | LeVJEPA | .312 | **.454** | .582 | $2,007 | **$3,008** |
| 100% | Cap-32 | .444 | .493 | **.634** | **$1,648** | $2,662 |
| 100% | LeVJEPA | **.468** | **.529** | .604 | $1,655 | **$2,627** |

At full labels LeVJEPA improves CPT and ICD AP on all three encoder seeds; its
paired mean deltas are +0.024 and +0.036. It also improves RMSE on all three,
while MAE is effectively tied. At 10% labels the tradeoff reverses for CPT and
TTNC but LeVJEPA retains an ICD advantage. This argues for treating patient-view
pretraining as a robust, code-decodable representation variant rather than a
replacement for cap-32 on every endpoint. The stable result does not justify
another flat-decoder architecture sweep; candidate retrieval remains the more
meaningful future generative direction.

Additional artifacts:

- `experiments/levjepa_patient_views_20260828/high_cost_tail_seed42_e4_v0`
- `experiments/levjepa_patient_views_20260828/attentive_high_cost_tail_seed42_probe_seeds_v1`
- `experiments/levjepa_patient_views_20260828/levjepa_patient_views_reprelu_seed42_e4`
- `experiments/levjepa_patient_views_20260828/dense_sparse_levjepa_fusion_seed42_v0`
- `experiments/levjepa_patient_views_20260828/cross_recipe_cap32_levjepa_tail_fusion_seed42_v0`
- `experiments/levjepa_patient_views_20260828/levjepa_patient_views_seed{43,44}_e4`
- `experiments/levjepa_patient_views_20260828/online_tail_candidate_showdown_seeds42_44_v1`
- `experiments/levjepa_patient_views_20260828/tail_label_efficiency_cap32_levjepa_seeds42_44_v0`
- `experiments/levjepa_patient_views_20260828/missing_history_robustness_cap32_levjepa_seeds42_44_v0`
- `experiments/levjepa_patient_views_20260828/frozen_generation_probe_{cap32_,}seed42_v0`
- `experiments/levjepa_patient_views_20260828/frozen_generation_probe_{cap32,levjepa}_seed{43,44}_v0`
