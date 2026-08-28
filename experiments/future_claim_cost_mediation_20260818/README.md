# Future-claim cost mediation

This experiment tests whether cost value passes through explicitly predicted
future claims. A CPT/ICD/TTNC decoder was trained on the full training cohort
without cost labels and then frozen. Cost probes used either the original
embedding, the decoder hidden state, predicted claim probabilities, or actual
next-claim codes. Predicted and actual claim vectors used linear cost heads to
make the strict bottleneck explicit.

The decoder generalized normally: validation CPT micro-AP 0.3604, ICD micro-AP
0.2098, and TTNC accuracy 0.6152.

| Cost labels | Direct embedding MLP | Decoder hidden MLP | Predicted claims linear | Actual claims linear |
|---:|---:|---:|---:|---:|
| 1% | **2,161.89** | 2,190.31 | 4,419.63 | 3,881.78 |
| 5% | **1,950.33** | 2,038.02 | 2,523.58 | 3,080.38 |
| 10% | **1,753.75** | 1,889.83 | 2,005.30 | 2,611.13 |
| 25% | **1,617.43** | 1,673.85 | 1,777.17 | 2,309.48 |

The strict mediation hypothesis is rejected under this decoder and probe: the
predicted distribution never beat the direct embedding, and the token-trained
hidden state also lost at every budget. Because even actual next-claim codes
performed poorly through the high-dimensional sparse linear head, this does not
show that future claims lack cost information. It shows that a flat sparse claim
distribution is an inefficient cost bottleneck in the low-label regime. The
joint auxiliary benefit is better explained by representation shaping than by a
simple `embedding -> predicted codes -> linear cost` pathway.

The complete protocol and per-seed results are in `summary.json`.
