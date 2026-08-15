## A10 Seed 314 Pilot

Protocol:
- fixed 24-epoch Stage 1 run
- checkpoints every 4 epochs
- post-hoc checkpoint-series cost probe
- seed `314`

Completed runs:

| Experiment | Best overall ckpt | MAE | WAPE | RMSE | q4 RMSE at overall best | Best q4 ckpt | Best q4 RMSE |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `A10pilot314_baseline` | `stage1-epochepoch=07.ckpt` | 2053.09 | 44.6819 | 3193.09 | 3521.62 | `stage1-epochepoch=11.ckpt` | 3484.74 |
| `A10pilot314_cpc_w005` | `stage1-epochepoch=03.ckpt` | 2073.72 | 45.1310 | 3209.11 | 3480.12 | `stage1-epochepoch=03.ckpt` | 3480.12 |
| `A10pilot314_ts2vec_w005` | `stage1-epochepoch=03.ckpt` | 2074.45 | 45.1468 | 3209.49 | 3478.94 | `stage1-epochepoch=03.ckpt` | 3478.94 |
| `A10pilot314_cpc_w010` | `stage1-epochepoch=15.ckpt` | 2078.51 | 45.2352 | 3216.52 | 3491.17 | `stage1-epochepoch=03.ckpt` | 3486.46 |

Read:
- The tuned seed-314 baseline is still best on the overall trusted cost metrics.
- Lowering the temporal loss from `0.1` to `0.05` helped both CPC and TS2Vec materially.
- `ts2vec_w005` and `cpc_w005` are essentially tied overall.
- Both low-weight temporal variants slightly improved `q4` tail RMSE versus the baseline, but they lost too much on overall MAE, WAPE, and RMSE.

Interrupted / not completed:
- `A10pilot314_cpc_w020`
- `A10pilot314_cpc_ctx8_future2_w005`
