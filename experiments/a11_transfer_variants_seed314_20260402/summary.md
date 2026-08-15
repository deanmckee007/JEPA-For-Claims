## A11 Transfer Variants Seed 314

Protocol:
- fixed 24-epoch Stage 1 run
- checkpoints every 4 epochs
- post-hoc checkpoint-series cost probe
- shared graph embeddings: `C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A11pilot_seed314/graph_embeddings.pt`

| Experiment | Overrides | Best overall ckpt | MAE | WAPE | RMSE | q4 RMSE at overall best | Best q4 ckpt | Best q4 RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| A11variant314_cpt_icd_only_mix050 | graph_transfer_ttnc=False, graph_embedding_mix=0.5 | stage1-epochepoch=23.ckpt | 2047.25 | 44.5548 | 3174.03 | 3511.21 | stage1-epochepoch=03.ckpt | 3496.10 |
| A11variant314_cpt_icd_only_mix025 | graph_transfer_ttnc=False, graph_embedding_mix=0.25 | stage1-epochepoch=23.ckpt | 2051.46 | 44.6465 | 3189.33 | 3511.00 | stage1-epochepoch=03.ckpt | 3487.84 |
| A11variant314_all_streams_mix025 | graph_embedding_mix=0.25 | stage1-epochepoch=03.ckpt | 2058.45 | 44.7986 | 3204.26 | 3489.05 | stage1-epochepoch=11.ckpt | 3475.21 |
| A11variant314_cpt_icd_only_hard | graph_transfer_ttnc=False, graph_embedding_mix=1.0 | stage1-epochepoch=15.ckpt | 2196.03 | 47.7929 | 3370.06 | 3619.31 | stage1-epochepoch=11.ckpt | 3615.97 |

Baseline compare:
- overall checkpoint `stage1-epochepoch=07.ckpt`
- overall RMSE `$3193.09`
- best q4 checkpoint `stage1-epochepoch=11.ckpt`
- best q4 RMSE `$3484.74`
