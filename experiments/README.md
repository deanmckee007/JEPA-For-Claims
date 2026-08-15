# Experiments

This directory holds the experiment ledger and per-run artifacts.

## Files

- `registry.csv`: manually curated experiment index
- `runs/<experiment_id>/experiment_spec.json`: intended recipe and overrides
- `runs/<experiment_id>/encoder.ckpt`: trained checkpoint
- `runs/<experiment_id>/representation_eval.json`: downstream evaluation report
- `runs/<experiment_id>/experiment_result.json`: merged experiment summary

The downstream evaluation report may also include a `slices` block with
bucketed metrics by target cost, raw sequence length, effective sequence
length, and TTNC-proxy label frequency.

`ttnc_proxy_*` metrics in the report are diagnostic proxy-label metrics derived
from TTNC, not ground-truth specialty metrics.

## Workflow

1. Add or update the experiment row in `registry.csv`.
2. Run the experiment through `scripts/run_experiment.py`.
3. Copy the headline metrics from `experiment_result.json` back into `registry.csv`.
4. Mark the experiment `completed`, `rejected`, or `promoted`.
