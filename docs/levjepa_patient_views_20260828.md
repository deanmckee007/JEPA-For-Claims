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
