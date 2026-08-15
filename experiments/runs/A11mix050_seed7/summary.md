## A11 Graph SSL Transfer Pilot

Run: `A11mix050_seed7`
Base recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
Transfer recipe: `a11_graph_ssl_transfer`
Graph embedding path: `C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A11mix050_seed7/graph_embeddings.pt`

Graph pretrain:
- nodes: `3179`
- edges: `633190`
- epochs: `120`
- device: `cuda`

Transferred Stage 1 best overall checkpoint:
- `stage1-epochepoch=03.ckpt`
- MAE `$2043.18`
- WAPE `44.3798%`
- RMSE `$3108.43`
- q4 RMSE `$3357.39`

Transferred Stage 1 best q4 checkpoint:
- `stage1-epochepoch=11.ckpt`
- q4 RMSE `$3352.18`

Baseline compare:
- overall checkpoint `stage1-epochepoch=23.ckpt`
- overall RMSE `$3129.65`
- best q4 checkpoint `stage1-epochepoch=23.ckpt`
- best q4 RMSE `$3392.34`
