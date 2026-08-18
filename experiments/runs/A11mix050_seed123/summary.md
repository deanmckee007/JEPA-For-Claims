## A11 Graph SSL Transfer Pilot

Run: `A11mix050_seed123`
Base recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
Transfer recipe: `a11_graph_ssl_transfer`
Graph embedding path: `C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A11mix050_seed123/graph_embeddings.pt`

Graph pretrain:
- nodes: `3179`
- edges: `633190`
- epochs: `120`
- device: `cuda`

Transferred Stage 1 best overall checkpoint:
- `stage1-epochepoch=23.ckpt`
- MAE `$2079.22`
- WAPE `44.6923%`
- RMSE `$3168.44`
- q4 RMSE `$3478.87`

Transferred Stage 1 best q4 checkpoint:
- `stage1-epochepoch=23.ckpt`
- q4 RMSE `$3478.87`

Baseline compare:
- overall checkpoint `stage1-epochepoch=15.ckpt`
- overall RMSE `$3179.83`
- best q4 checkpoint `stage1-epochepoch=15.ckpt`
- best q4 RMSE `$3503.72`
