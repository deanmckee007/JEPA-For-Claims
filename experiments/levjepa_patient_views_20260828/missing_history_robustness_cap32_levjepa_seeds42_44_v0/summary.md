# Missing-History Robustness

Each cell is mean precision / PR-AUC across encoder seeds. The tail head is fit once on complete training histories.

| Condition | drop 0% | drop 10% | drop 30% | drop 50% | drop 70% |
|---|---:|---:|---:|---:|---:|
| cap32_boosted | 0.188 / 0.127 | 0.167 / 0.122 | 0.146 / 0.109 | 0.181 / 0.106 | 0.160 / 0.101 |
| cap32_logistic | 0.097 / 0.078 | 0.125 / 0.078 | 0.125 / 0.070 | 0.104 / 0.059 | 0.069 / 0.056 |
| levjepa_boosted | 0.194 / 0.144 | 0.188 / 0.137 | 0.181 / 0.124 | 0.194 / 0.121 | 0.194 / 0.123 |
| levjepa_logistic | 0.153 / 0.108 | 0.139 / 0.094 | 0.160 / 0.100 | 0.132 / 0.078 | 0.153 / 0.092 |
