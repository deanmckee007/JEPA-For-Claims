# Experiment Program

## Goal

Run JEPA-for-Claims experiments in a way that is easy to compare, easy to reproduce, and hard to lose track of.

The program is split into two tracks:

- Hyperparameter tuning on the current architecture
- Architecture experiments on isolated branches

## Recording Standard

Every experiment should produce a dedicated artifact directory under `experiments/runs/<experiment_id>/` containing:

- `experiment_spec.json`: the intended recipe and config overrides
- `encoder.ckpt`: the stage-1 checkpoint
- `representation_eval.json`: downstream evaluation output
- `experiment_result.json`: merged metadata plus evaluation metrics

`representation_eval.json` now also carries an optional `slices` section for:

- equal-frequency target-cost buckets
- equal-frequency raw sequence-length buckets
- equal-frequency effective sequence-length buckets
- equal-frequency TTNC-proxy-frequency buckets

`sequence_length_bucket` uses raw pre-truncation claim counts. A separate
`effective_sequence_length_bucket` shows the post-truncation length seen by the
model after the `max_claims_len` cap.

The shared index is [experiments/registry.csv](C:/Users/tmcke/code/JEPA-For-Claims/experiments/registry.csv). That file is the human-maintained ledger for status, branch, and headline results.

## Core Metrics

Primary model-selection metrics:

- `val_rmse_improvement_vs_mean_baseline`
- `val_rmse_improvement_pct_vs_mean_baseline`
- `target_probe_mae_dollars`
- `target_probe_wape_percent`
- `target_probe_rmse_dollars`
- `cluster_silhouette`

Secondary metrics:

- `target_probe_rmse_log1p`
- `ttnc_proxy_retrieval_hit_rate_at_5`
- `ttnc_proxy_label_cluster_ari`
- TTNC-proxy probe metrics when the label distribution supports them

Important: the `ttnc_proxy_*` metrics are derived from the last valid TTNC label
in the patient history. They are diagnostic proxies, not true specialty metrics,
and should not outrank the dollar-space cost metrics during model selection.

## Current Leader

Under the current cost-first evaluation contract, the leader is the
`sigreg_dense_hybrid_dollar` recipe. This is the 10-epoch hybrid target-dynamics
variant from the A5 line of work, evaluated with TTNC-proxy metrics demoted.

For the composable Level-1/Level-2 line specifically, the selected architecture
is now `composable_level1_lejepa_any_code`. It uses a light (`0.1`) Level-1
predictive objective with shared, non-EMA targets. See
[core_architecture_ablations_20260814.md](core_architecture_ablations_20260814.md)
for the multiseed and missing-modality decision record.

The selected recipe has now completed its 20-epoch representation run and
sealed pairwise-ranking evaluation. The locked `rolled_ranknet_pointwise` head
reached 0.7152 test pairwise accuracy and 0.5914 Spearman. See
[final_pairwise_ranking_20260814.md](final_pairwise_ranking_20260814.md).

A subsequent validation-only direct-supervision study found that the best
cost/representation tradeoff is sequential: pretrain the selected SSL encoder,
then fine-tune a pre-SAE cost head and encoder for 20 task-only epochs. Keep the
unfine-tuned checkpoint for general representation use. See
[cost_supervision_ablation_20260814.md](cost_supervision_ablation_20260814.md).

The cost-only rerank in
[summary.md](C:/Users/tmcke/code/JEPA-For-Claims/experiments/cost_rerank_20260323/summary.md)
showed:

- `10` epochs is the best current default on overall dollar MAE / WAPE / RMSE
- the plain dense control keeps a narrow edge only on `q4_high_cost` RMSE
- the `20`-epoch hybrid / A5 variant no longer earns its extra compute on the trusted cost metrics

```bash
python scripts/train.py --recipe sigreg_dense_hybrid_dollar --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet
python scripts/train.py --recipe sigreg_dense_hybrid_repr --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet
```

## Active Follow-Up

`A6` is the next architecture experiment: masked next-claim token grounding.
It keeps SIGReg as the core anti-collapse path and adds a low-weight auxiliary
ordered-token loss on a masked subset of the next claim. This is explicitly a
grounding experiment, not a replacement for the JEPA regularizer.

Latest follow-up: the zero-SIGReg A6 variant stayed stable on the real parquet
at 10, 20, and 40 stage-1 epochs on seed 42. That does not prove SIGReg is
unnecessary, but it does mean masked grounding alone was enough to avoid an
obvious collapse in the first long-schedule check.

The first runnable recipe is:

```bash
python scripts/train.py --recipe sigreg_dense_hybrid_dollar_masked_grounding --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet
```

Key controls:

- `use_masked_next_claim_token_grounding`
- `masked_next_claim_token_weight`
- `masked_next_claim_mask_ratio`
- `masked_next_claim_include_ttnc`
- `masked_next_claim_sort_target_tokens`

## Probe Representation Sweeps

The downstream linear probes and retrieval diagnostics no longer have to use
only `patient_representation`. Evaluation now supports:

- `patient_representation`
- `patient_representation_pre_sae`
- `context_mean_pool`
- `context_max_pool`
- `context_pooled`
- `next_claim_prediction`
- `dense_decoder_latent` when that auxiliary latent exists

The current cost-first default probe source is `patient_representation_pre_sae`.
Under the current architecture, that is the pooled patient state before SAE and
it slightly outperformed the post-SAE patient representation on the trusted
dollar metrics.

This makes it possible to answer a separate question from the training recipe:
which sequence-level state is actually best for the downstream probe contract?

Example:

```bash
python scripts/evaluate_representations.py \
  --checkpoint C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A6_masked_grounding_seed42/encoder.ckpt \
  --recipe sigreg_dense_hybrid_dollar_masked_grounding \
  --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet \
  --max-samples 20000 \
  --representation-source patient_representation_pre_sae \
  --output-json C:/Users/tmcke/code/JEPA-For-Claims/experiments/probe_source_eval/pre_sae.json
```

## Branch Policy

Hyperparameter experiments stay on the current branch and use config overrides only.

Architecture experiments get their own branches:

- `codex/arch-intermediate-seq-supervision`
- `codex/arch-bottleneck-dense-decoder`
- `codex/arch-l2-predictor-depth`
- `codex/arch-level1-ssl`
- `codex/arch-target-encoder-hybrid`

## Experiment Order

### Controls

- `H0_vicreg_baseline`
- `H0_sigreg_core`
- `H0_sigreg_dense`

### Hyperparameter Track

1. `H1` Dense-loss weighting
2. `H2` Dense horizon (`observed_claim_k`)
3. `H3` SIGReg strength
4. `H4` Representation capacity
5. `H5` Patient pooling ablation
6. `H6` Training schedule

### Architecture Track

1. `A1` Intermediate sequence supervision
2. `A2` Bottlenecked dense decoder
3. `A3` Stronger Level 2 predictor
4. `A4` Level 1 supervision refinement
5. `A5` Shared-target vs EMA hybrid
6. `A6` Masked next-claim token grounding

### Additional Investigation Arms

- `H10` Cost-head supervision geometry checks
  - Turn on `use_predictor_head`
  - Sweep `task_loss_weight`
  - Track hierarchy geometry at `context_mean_pool`, `patient_representation_pre_sae`, and `patient_representation`
  - Treat `val_rmse` as the paired task metric for this arm
- `A7` Cost-head source placement
  - Compare `predictor_head_source=context_mean_pool`, `patient_representation_pre_sae`, and `patient_representation`
  - Use the same `task_loss_weight`
  - Prefer `patient_representation_pre_sae` if it keeps the cost gain while avoiding a narrower final patient state
- `A8` Patient-state bifurcation
  - Keep one exposed patient representation path
  - Add a dedicated predictive state for dense observed-plus-next decoding
  - Watch whether narrow predictive geometry moves into the predictive state while the exposed patient embedding stays broader

## How To Run One Experiment

Use the single-experiment runner:

```bash
python scripts/run_experiment.py \
  --name H1_dense_weight_0p10 \
  --recipe sigreg_dense \
  --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet \
  --representation-pretrain-epochs 20 \
  --set observed_claim_loss_weight=0.1
```

The same override list is passed to both training and evaluation so checkpoint loading stays config-compatible.

## Initial Hyperparameter Commands

### H1 Dense-Loss Weighting

```bash
python scripts/run_experiment.py --name H1_dense_weight_0p10 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_loss_weight=0.1
python scripts/run_experiment.py --name H1_dense_weight_0p25 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_loss_weight=0.25
python scripts/run_experiment.py --name H1_dense_weight_0p50 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_loss_weight=0.5
```

### H2 Dense Horizon

```bash
python scripts/run_experiment.py --name H2_observed_k1 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_k=1
python scripts/run_experiment.py --name H2_observed_k2 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_k=2
python scripts/run_experiment.py --name H2_observed_k3 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set observed_claim_k=3
```

### H3 SIGReg Strength

```bash
python scripts/run_experiment.py --name H3_sigreg_0p05 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set sigreg_weight_lvl2=0.05
python scripts/run_experiment.py --name H3_sigreg_0p10 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set sigreg_weight_lvl2=0.1
python scripts/run_experiment.py --name H3_sigreg_0p20 --recipe sigreg_dense --data-path C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet --representation-pretrain-epochs 20 --set sigreg_weight_lvl2=0.2
```

## Merge Rule

Do not merge an architecture branch unless it beats the current best tuned config on at least one downstream representation metric and does not materially regress the dollar-error metrics.
