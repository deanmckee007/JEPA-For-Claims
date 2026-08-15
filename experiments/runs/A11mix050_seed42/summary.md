## A11 Graph SSL Transfer Pilot

Run: `A11mix050_seed42`
Base recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
Transfer recipe: `a11_graph_ssl_transfer`
Graph embedding path: `C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A11mix050_seed42/graph_embeddings.pt`

Graph pretrain:
- nodes: `3179`
- edges: `633190`
- epochs: `120`
- device: `cuda`

Transferred Stage 1 best overall checkpoint:
- `stage1-epochepoch=19.ckpt`
- MAE `$2071.02`
- WAPE `44.6928%`
- RMSE `$3158.33`
- q4 RMSE `$3515.01`

Transferred Stage 1 best q4 checkpoint:
- `stage1-epochepoch=03.ckpt`
- q4 RMSE `$3489.17`

Baseline compare:
- overall checkpoint `stage1-epochepoch=19.ckpt`
- overall RMSE `$3116.11`
- best q4 checkpoint `stage1-epochepoch=11.ckpt`
- best q4 RMSE `$3403.36`
