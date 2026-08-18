## Long-Schedule Check For `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`

Compared the current 10-epoch seed-42 leader against the best intermediate checkpoint from a 20-epoch run.

- Recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
- Data: `C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet`
- Evaluation source: `patient_representation_pre_sae`

### 10-Epoch Leader

- checkpoint: `experiments/runs/DYN_sigreg000_seed42/encoder.ckpt`
- MAE: `$2048.85`
- WAPE: `44.214%`
- RMSE: `$3117.56`
- q4 high-cost RMSE: `$3414.01`

### 20-Epoch Best Intermediate

- checkpoint: `checkpoints/stage1-best-v123.ckpt`
- probe output: `experiments/runs/TUNE_sigreg0_recipe_ep20_seed42/intermediate_cost_probe.json`
- MAE: `$2048.72`
- WAPE: `44.212%`
- RMSE: `$3118.44`
- q4 high-cost RMSE: `$3412.47`

### Read

The longer schedule did not produce a meaningful improvement. It slightly improved MAE, WAPE, and q4 high-cost RMSE, but slightly worsened overall RMSE. This is effectively a tie, so the 10-epoch recipe remains the practical default.
