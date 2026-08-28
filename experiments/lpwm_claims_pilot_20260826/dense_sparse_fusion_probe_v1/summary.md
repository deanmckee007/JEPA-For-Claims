# High-Cost Tail Probe (Top 1.5%)

| Condition | Runs | Precision | Recall | Lift | PR-AUC | NDCG@budget | Cost capture | Cost lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_boosted_tail | 3 | 0.1736 | 0.1736 | 11.53 | 0.1203 | 0.1861 | 0.0310 | 2.06 |
| dense_embedding_logistic | 3 | 0.0972 | 0.0972 | 6.46 | 0.0778 | 0.1029 | 0.0259 | 1.72 |
| dense_sparse_boosted_tail | 3 | 0.2014 | 0.2014 | 13.38 | 0.1318 | 0.2252 | 0.0325 | 2.16 |
| dense_sparse_embedding_logistic | 3 | 0.1111 | 0.1111 | 7.38 | 0.0798 | 0.1055 | 0.0275 | 1.82 |
| raw_boosted_tail | 3 | 0.1806 | 0.1806 | 11.99 | 0.1409 | 0.2194 | 0.0307 | 2.04 |
| raw_dense_boosted_tail | 3 | 0.2222 | 0.2222 | 14.76 | 0.1791 | 0.2813 | 0.0333 | 2.21 |
| raw_dense_sparse_boosted_tail | 3 | 0.2014 | 0.2014 | 13.38 | 0.1541 | 0.2414 | 0.0339 | 2.25 |
| raw_dense_sparse_neural_concat | 3 | 0.1875 | 0.1875 | 12.45 | 0.1344 | 0.2326 | 0.0304 | 2.02 |
| raw_dense_sparse_neural_gate | 3 | 0.2153 | 0.2153 | 14.30 | 0.1316 | 0.2274 | 0.0328 | 2.18 |
| raw_sparse_boosted_tail | 3 | 0.2083 | 0.2083 | 13.84 | 0.1512 | 0.2447 | 0.0333 | 2.21 |
| sparse_boosted_tail | 3 | 0.1667 | 0.1667 | 11.07 | 0.1172 | 0.1905 | 0.0317 | 2.10 |
| sparse_embedding_logistic | 3 | 0.1389 | 0.1389 | 9.22 | 0.0815 | 0.1246 | 0.0276 | 1.84 |
