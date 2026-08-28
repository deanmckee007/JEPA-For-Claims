# High-Cost Tail Probe (Top 1.5%)

| Condition | Runs | Precision | Recall | Lift | PR-AUC | NDCG@budget | Cost capture | Cost lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| levjepa_embedding_ordinal_severity | 1 | 0.0417 | 0.0417 | 2.77 | 0.0608 | 0.0483 | 0.0198 | 1.32 |
| levjepa_embedding_ordinal_tail_probability | 1 | 0.0833 | 0.0833 | 5.53 | 0.0648 | 0.0744 | 0.0218 | 1.45 |
| levjepa_embedding_tail | 1 | 0.1875 | 0.1875 | 12.45 | 0.1508 | 0.2478 | 0.0307 | 2.04 |
| levjepa_hybrid_cost | 1 | 0.1667 | 0.1667 | 11.07 | 0.1281 | 0.2095 | 0.0349 | 2.32 |
| levjepa_hybrid_ordinal_severity | 1 | 0.1250 | 0.1250 | 8.30 | 0.1036 | 0.1128 | 0.0296 | 1.97 |
| levjepa_hybrid_ordinal_tail_probability | 1 | 0.1250 | 0.1250 | 8.30 | 0.1091 | 0.1665 | 0.0284 | 1.89 |
| levjepa_hybrid_tail | 1 | 0.2083 | 0.2083 | 13.84 | 0.2391 | 0.3387 | 0.0334 | 2.22 |
| raw_boosted_cost | 1 | 0.1875 | 0.1875 | 12.45 | 0.1221 | 0.2221 | 0.0343 | 2.28 |
| raw_boosted_ordinal_severity | 1 | 0.1458 | 0.1458 | 9.69 | 0.1187 | 0.1872 | 0.0263 | 1.74 |
| raw_boosted_ordinal_tail_probability | 1 | 0.1875 | 0.1875 | 12.45 | 0.1260 | 0.2578 | 0.0276 | 1.84 |
| raw_boosted_tail | 1 | 0.1667 | 0.1667 | 11.07 | 0.1322 | 0.1930 | 0.0298 | 1.98 |
