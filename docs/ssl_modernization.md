# SSL Objective Modernization

## What Changed

- The Level 1 and Level 2 JEPA regularizer is now pluggable through `ssl_objective_type`.
- `vicreg` preserves the existing predictive-plus-anti-collapse behavior for baseline comparisons.
- `sigreg` adds a LeJEPA-style claims-native objective built from predictive MSE plus random-projection Gaussian regularization.
- Level 2 can now predict a dense target set of claim representations: the last `K` observed claims and the next claim.
- Clean SSL is now an explicit runtime preset that disables token prediction, diffusion, and generative export while leaving SAE optional.

## Why

- The project objective is broader than next-claim prediction. The model should learn patient and claim representations that transfer to retrieval, clustering, similarity, forecasting, and specialty inference.
- VICReg remains useful as a baseline, but the refactor makes JEPA regularization an ablation point rather than a hard-coded assumption.
- Dense Level 2 supervision keeps the patient bottleneck faithful to recent observed structure without turning pretraining into raw-token reconstruction.

## New Configuration Flags

- `ssl_objective_type`: `vicreg` or `sigreg`
- `target_encoder_mode`: `ema` or `shared`
- `clean_ssl_mode`: disable token and diffusion side objectives
- `use_level2_dense_prediction`: enable last-K-observed plus next-claim decoding
- `observed_claim_k`: number of observed claims to decode
- `observed_claim_loss_weight`: weight for observed-claim representation loss
- `next_claim_loss_weight`: weight for next-claim representation loss
- `sigreg_weight_lvl1` / `sigreg_weight_lvl2`: SIGReg regularization strength per level
- `sigreg_num_slices`: number of random projection directions
- `sigreg_num_points`: number of characteristic-function evaluation points
- `intermediate_sequence_supervision_weight`: scaffolded for future deep supervision, disabled by default

## Migration Recipes

### VICReg Baseline

```python
config.ssl_objective_type = "vicreg"
config.target_encoder_mode = "ema"
config.use_level2_dense_prediction = False
```

### SIGReg Core

```python
config.ssl_objective_type = "sigreg"
config.target_encoder_mode = "shared"
config.use_level2_dense_prediction = False
config.clean_ssl_mode = True
```

### SIGReg + Dense Observed/Next Prediction

```python
config.ssl_objective_type = "sigreg"
config.target_encoder_mode = "shared"
config.use_level2_dense_prediction = True
config.observed_claim_k = 2
config.observed_claim_loss_weight = 0.25
config.next_claim_loss_weight = 1.0
config.clean_ssl_mode = True
```

## Smoke Run

Use the synthetic one-batch smoke runner to verify the three migration recipes
through a real Lightning fit without requiring a local claims parquet:

```bash
python scripts/smoke_ssl_recipes.py
```

It writes a temporary parquet, runs `vicreg_baseline`, `sigreg_core`, and
`sigreg_dense`, and prints a JSON summary for each recipe.

## Training And Evaluation

Run a named recipe through the main training entrypoint:

```bash
python scripts/train.py --recipe sigreg_core --data-path /path/to/claims.parquet
```

Evaluate a saved checkpoint with simple downstream representation probes:

```bash
python scripts/evaluate_representations.py \
  --checkpoint /path/to/model.ckpt \
  --data-path /path/to/claims.parquet \
  --recipe sigreg_core
```

The evaluation report writes JSON with retrieval, clustering, specialty-probe,
and regression-probe metrics.

Run the full three-recipe train+eval comparison as one command:

```bash
python scripts/run_recipe_sweep.py --data-path /path/to/claims.parquet
```

This writes one checkpoint and one eval report per recipe plus
`summary.json`, `summary.csv`, and `summary.md` in the sweep output
directory.
