## A10 Temporal Contrastive vs Fixed-Horizon Baseline

Protocol:
- fixed 24-epoch Stage 1 run
- checkpoints every 4 epochs
- full-data post-hoc cost-probe selection
- representation source: `patient_representation_pre_sae`

Baseline control:
- recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
- series: `experiments/runs/A9_compare_baseline_seed42/checkpoint_series_cost_probe.json`

A10 candidate:
- recipe: `a10_temporal_contrastive`
- series: `experiments/runs/A10_temporal_contrastive_seed42/checkpoint_series_cost_probe.json`

Best baseline overall checkpoint:
- `stage1-epochepoch=19.ckpt`
- MAE `$2045.18`
- WAPE `44.1352%`
- RMSE `$3116.11`

Best baseline q4 checkpoint:
- `stage1-epochepoch=11.ckpt`
- q4 RMSE `$3403.36`

Best A10 overall checkpoint:
- `stage1-epochepoch=19.ckpt`
- MAE `$2079.43`
- WAPE `44.8744%`
- RMSE `$3151.13`

Best A10 q4 checkpoint:
- `stage1-epochepoch=07.ckpt`
- q4 RMSE `$3454.68`

Conclusion:
- A10 temporal contrastive is a clear regression on the trusted cost metrics.
- It is not promotable over the current grounded baseline.
