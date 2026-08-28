# High-Cost Tail Probe (Top 1.5%)

| Condition | Runs | Precision | Recall | Lift | PR-AUC | NDCG@budget | Cost capture | Cost lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_embedding_ordinal_severity | 3 | 0.0556 | 0.0556 | 3.69 | 0.0548 | 0.0467 | 0.0198 | 1.31 |
| dense_embedding_ordinal_tail_probability | 3 | 0.0417 | 0.0417 | 2.77 | 0.0449 | 0.0333 | 0.0184 | 1.22 |
| dense_embedding_tail | 3 | 0.0972 | 0.0972 | 6.46 | 0.0778 | 0.1029 | 0.0259 | 1.72 |
| dense_hybrid_cost | 3 | 0.1736 | 0.1736 | 11.53 | 0.1537 | 0.2362 | 0.0344 | 2.29 |
| dense_hybrid_ordinal_severity | 3 | 0.1597 | 0.1597 | 10.61 | 0.1024 | 0.1485 | 0.0319 | 2.12 |
| dense_hybrid_ordinal_tail_probability | 3 | 0.1389 | 0.1389 | 9.22 | 0.1008 | 0.1877 | 0.0286 | 1.90 |
| dense_hybrid_tail | 3 | 0.2222 | 0.2222 | 14.76 | 0.1791 | 0.2813 | 0.0333 | 2.21 |
| raw_boosted_cost | 3 | 0.2014 | 0.2014 | 13.38 | 0.1273 | 0.2283 | 0.0354 | 2.35 |
| raw_boosted_ordinal_severity | 3 | 0.1528 | 0.1528 | 10.15 | 0.1154 | 0.1913 | 0.0279 | 1.85 |
| raw_boosted_ordinal_tail_probability | 3 | 0.1528 | 0.1528 | 10.15 | 0.1096 | 0.2045 | 0.0274 | 1.82 |
| raw_boosted_tail | 3 | 0.1806 | 0.1806 | 11.99 | 0.1409 | 0.2194 | 0.0307 | 2.04 |
| sparse_embedding_ordinal_severity | 3 | 0.0833 | 0.0833 | 5.53 | 0.0669 | 0.0884 | 0.0221 | 1.47 |
| sparse_embedding_ordinal_tail_probability | 3 | 0.0625 | 0.0625 | 4.15 | 0.0548 | 0.0676 | 0.0213 | 1.42 |
| sparse_embedding_tail | 3 | 0.1389 | 0.1389 | 9.22 | 0.0815 | 0.1246 | 0.0276 | 1.84 |
| sparse_hybrid_cost | 3 | 0.1736 | 0.1736 | 11.53 | 0.1607 | 0.2605 | 0.0347 | 2.31 |
| sparse_hybrid_ordinal_severity | 3 | 0.2083 | 0.2083 | 13.84 | 0.1366 | 0.2239 | 0.0324 | 2.15 |
| sparse_hybrid_ordinal_tail_probability | 3 | 0.1528 | 0.1528 | 10.15 | 0.1160 | 0.2069 | 0.0281 | 1.87 |
| sparse_hybrid_tail | 3 | 0.2083 | 0.2083 | 13.84 | 0.1512 | 0.2447 | 0.0333 | 2.21 |
