# Generation auxiliary control ablation

This experiment compares aligned versus patient-shuffled auxiliary labels,
individual CPT/ICD/TTNC targets, and a matched-width raw-history baseline. All
runs use 10% cost labels, auxiliary weight 10, three downstream seeds, the same
optimizer-step count, and frozen validation evaluation. Test data was not
accessed.

The raw control is a target-excluding 128-dimensional signed hash of last-claim
indicators, recency-weighted CPT/ICD/TTNC history, and basic sequence
statistics. It therefore matches the pretrained representation width and uses
the identical downstream head.

| Condition | Cost MAE ($) | Cost RMSE ($) | CPT AP | ICD AP | TTNC accuracy |
|---|---:|---:|---:|---:|---:|
| Pretrained, cost only | 1,945.55 | 2,975.91 | 0.0877 | 0.0909 | 0.3242 |
| Pretrained, shuffled all | 1,760.21 | 2,751.64 | 0.0819 | 0.0923 | 0.1909 |
| Pretrained, aligned all | **1,721.37** | **2,721.61** | 0.3637 | 0.2111 | 0.6101 |
| Pretrained, CPT only | 1,747.19 | 2,745.29 | 0.3592 | 0.0909 | 0.3242 |
| Pretrained, ICD only | 1,774.93 | 2,768.04 | 0.0877 | 0.2137 | 0.3242 |
| Pretrained, TTNC only | 1,723.61 | 2,725.66 | 0.0877 | 0.0909 | **0.6125** |
| Pretrained, CPT + ICD | 1,740.48 | 2,735.95 | **0.3680** | 0.2078 | 0.3242 |
| Raw history, cost only | 2,378.05 | 3,580.82 | 0.0877 | 0.0909 | 0.3242 |
| Raw history, shuffled all | 2,219.77 | 3,309.38 | 0.0645 | 0.0760 | 0.2447 |
| Raw history, aligned all | 2,183.68 | 3,247.75 | 0.3428 | **0.4805** | 0.6171 |

Aligned labels beat shuffled labels in every downstream seed. The paired MAE
advantage was $38.84 +/- $9.91 for pretrained features and $36.09 +/- $11.74
for raw history. However, shuffled labels account for most of the improvement
over cost-only: $185.34 of the pretrained model's $224.18 total gain. The
auxiliary branch therefore provides strong generic stochastic/multitask
regularization plus a smaller, reproducible semantic-transfer component.

TTNC-only supervision almost exactly matches the full aligned auxiliary cost
result on pretrained features ($1,723.61 versus $1,721.37). It does not do so on
raw history ($2,267.80), suggesting that temporal/specialty structure is
particularly well aligned with the pretrained representation. Raw history is
much better at ICD persistence but substantially worse for cost.

The next control should match the auxiliary branch's extra feature batches and
gradient noise without labels, for example feature consistency or randomized
zero-mean gradients. This will quantify how much of the shuffled-label gain is
ordinary regularization rather than a useful auxiliary objective.
