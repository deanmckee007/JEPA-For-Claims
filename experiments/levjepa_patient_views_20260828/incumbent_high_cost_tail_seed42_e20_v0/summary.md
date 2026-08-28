# High-Cost Tail Probe (Top 1.5%)

| Condition | Runs | Precision | Recall | Lift | PR-AUC | NDCG@budget | Cost capture | Cost lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent_embedding_ordinal_severity | 1 | 0.0833 | 0.0833 | 5.53 | 0.0755 | 0.0913 | 0.0212 | 1.41 |
| incumbent_embedding_ordinal_tail_probability | 1 | 0.0625 | 0.0625 | 4.15 | 0.0646 | 0.0779 | 0.0207 | 1.37 |
| incumbent_embedding_tail | 1 | 0.1875 | 0.1875 | 12.45 | 0.1027 | 0.1742 | 0.0302 | 2.01 |
| incumbent_hybrid_cost | 1 | 0.2083 | 0.2083 | 13.84 | 0.1859 | 0.3080 | 0.0357 | 2.37 |
| incumbent_hybrid_ordinal_severity | 1 | 0.1667 | 0.1667 | 11.07 | 0.1237 | 0.1745 | 0.0309 | 2.05 |
| incumbent_hybrid_ordinal_tail_probability | 1 | 0.2083 | 0.2083 | 13.84 | 0.1277 | 0.2583 | 0.0309 | 2.05 |
| incumbent_hybrid_tail | 1 | 0.2083 | 0.2083 | 13.84 | 0.1330 | 0.2260 | 0.0333 | 2.21 |
| raw_boosted_cost | 1 | 0.1875 | 0.1875 | 12.45 | 0.1221 | 0.2221 | 0.0343 | 2.28 |
| raw_boosted_ordinal_severity | 1 | 0.1458 | 0.1458 | 9.69 | 0.1187 | 0.1872 | 0.0263 | 1.74 |
| raw_boosted_ordinal_tail_probability | 1 | 0.1875 | 0.1875 | 12.45 | 0.1260 | 0.2578 | 0.0276 | 1.84 |
| raw_boosted_tail | 1 | 0.1667 | 0.1667 | 11.07 | 0.1322 | 0.1930 | 0.0298 | 1.98 |
