# Cosine + LogVar Freeze Pilot

Recipe base:
- `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`

Overrides:
- `scheduler_type=cosine`
- `scheduler_t_max=30`
- `freeze_logvars_after_epoch=10`
- `checkpoint_every_n_epochs=4`
- `checkpoint_save_top_k=0`
- planned schedule: 30 stage-1 epochs
- run stopped during epoch 22 after the epoch-20 checkpoint had been evaluated

Key mechanics:
- `logvars_frozen` flipped from `0.0` to `1.0` around epoch 11.
- After that point, `logvar_ssl_lvl2` stayed flat at `0.097218` and
  `logvar_sae` stayed flat at `-0.137915`.

Epoch-20 cost probe (`stage1-epochepoch=19-v1.ckpt`):
- MAE `$2055.85`
- WAPE `44.3654%`
- RMSE `$3134.03`
- q4 RMSE `$3437.74`

Comparison to the step-scheduler long-run baseline at epoch 20:
- step baseline epoch 20:
  - MAE `$2048.46`
  - WAPE `44.2059%`
  - RMSE `$3118.31`
  - q4 RMSE `$3412.23`
- cosine + freeze epoch 20:
  - materially worse on all trusted cost metrics

Read:
- Freezing the log-vars did make the weighted objective stationary.
- That stabilization was not enough to help cost.
- The combination of cosine schedule plus frozen log-vars is a reject at this checkpoint.
