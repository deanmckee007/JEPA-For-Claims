# Multi-encoder cost plus generation replication

Three independently pretrained encoders used the selected
`lejepa_anycode_l1w01` recipe, pretraining seeds 42, 43, and 44, and four
pretraining epochs. Each encoder was evaluated with downstream seeds 42--44,
10% cost labels, and auxiliary weights 0, 3, and 10. The data contract and
vocabulary hashes match across all checkpoints. Test data was not accessed.

| Aux weight | MAE ($), mean +/- SD | RMSE ($), mean +/- SD | CPT AP | ICD AP | TTNC accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,909.53 +/- 56.68 | 2,920.67 +/- 98.62 | 0.0877 | 0.0909 | 0.3242 |
| 3 | 1,767.22 +/- 43.60 | 2,753.00 +/- 61.16 | 0.3573 | 0.2027 | 0.6143 |
| 10 | **1,717.70 +/- 44.37** | **2,718.35 +/- 64.66** | **0.3629** | **0.2034** | 0.6139 |

Weight 10 improved MAE in all nine paired encoder/downstream-seed comparisons.
Its paired auxiliary-minus-cost-only MAE delta was -$191.82 +/- $46.92.
Weight 3 also won all nine comparisons, with a delta of -$142.30 +/- $33.42.

Per-encoder mean weight-10 MAE changes were:

- Encoder seed 42: $1,920.23 to $1,734.21 (-$186.01).
- Encoder seed 43: $1,934.22 to $1,727.29 (-$206.93).
- Encoder seed 44: $1,874.13 to $1,691.60 (-$182.53).

Detailed run records and saved heads are in the `encoder_seed42`,
`encoder_seed43`, and `encoder_seed44` subdirectories.
