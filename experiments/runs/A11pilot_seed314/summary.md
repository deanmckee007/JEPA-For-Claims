## A11 Graph SSL Transfer Pilot

Run: `A11pilot_seed314`
Base recipe: `sigreg_dense_hybrid_dollar_masked_grounding_sigreg0`
Transfer recipe: `a11_graph_ssl_transfer`
Graph embedding path: `C:/Users/tmcke/code/JEPA-For-Claims/experiments/runs/A11pilot_seed314/graph_embeddings.pt`

Graph pretrain:
- nodes: `3179`
- edges: `633190`
- epochs: `120`
- device: `cuda`

Transferred Stage 1 best overall checkpoint:
- `stage1-epochepoch=07.ckpt`
- MAE `$2179.57`
- WAPE `47.4346%`
- RMSE `$3377.35`
- q4 RMSE `$3476.35`

Transferred Stage 1 best q4 checkpoint:
- `stage1-epochepoch=19.ckpt`
- q4 RMSE `$3460.43`

Baseline compare:
- overall checkpoint `stage1-epochepoch=07.ckpt`
- overall RMSE `$3193.09`
- best q4 checkpoint `stage1-epochepoch=11.ckpt`
- best q4 RMSE `$3484.74`
