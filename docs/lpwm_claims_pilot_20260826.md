# LpWM-style sparse claims pilot (2026-08-26)

## Question

Does making the canonical claim latent sparse improve the downstream value of
pretrained JEPA embeddings, and does the effect depend on predictor capacity as
reported in *A Case for Sparse Representations in World Models*?

## Implementation and protocol

- Added paper-faithful RepReLU: exact ReLU forward values with a GELU surrogate
  gradient.
- Added RDMReg using random normalized projections and sliced Wasserstein-2
  matching to unit-variance Gaussian or Laplace references.
- Applied the configured link to both canonical claim-encoder outputs and
  Level-2 predictor outputs. The historical Top-K SAE was disabled.
- Used the selected composable/shared-target LeJEPA architecture, `any_code`
  cohort, and Level-1 predictive weight 0.1 for every controlled comparison.
- Used frozen contract `claims_seed42_v1.json`, validation only. The test split
  was not touched.
- Screens used two representation-training epochs. These are pilot results,
  not promotion runs.

The first attempted screen accidentally inherited the older hybrid-target
anchor. It was stopped and preserved under
`screen_confounded_hybrid_anchor_seed42_e2`; none of its metrics are used below.

## Dense / sparse screen (seed 42, full validation n=3,188)

| Representation | Reference | mu | Cost MAE | Cost RMSE | Log RMSE | TTNC retrieval@5 |
|---|---|---:|---:|---:|---:|---:|
| Dense | Gaussian | 0 | 1,633.68 | 2,630.36 | 0.8249 | 0.7729 |
| Dense | Laplace | 0 | 1,633.88 | 2,630.26 | 0.8249 | 0.7742 |
| RepReLU | Gaussian | 0 | 1,626.32 | 2,629.34 | 0.8320 | 0.7582 |
| RepReLU | Laplace | 0 | 1,626.94 | 2,630.48 | 0.8318 | 0.7575 |
| RepReLU | Laplace | -1 | **1,625.61** | 2,630.40 | 0.8314 | 0.7572 |

At full validation size, rectification improved MAE by roughly $7-8 (0.4-0.5%)
but did not improve dollar RMSE, and it reduced TTNC neighborhood quality. The
Gaussian/Laplace controls were effectively tied, so there is no evidence that
heavy tails matter at RDM weight 0.1.

## Mechanism ablations (seed 42, matched validation prefix n=2,000)

The smaller cohort changes absolute dollar metrics, so these values should only
be compared within this section.

| Variant | Cost MAE | Cost RMSE | Log RMSE |
|---|---:|---:|---:|
| Dense + MSE only | 1,832.16 | 2,957.35 | 0.8088 |
| RepReLU + MSE only | 1,796.93 | 2,915.63 | 0.8125 |
| Dense + weak RDM (0.1) | 1,816.02 | 2,937.58 | 0.8063 |
| RepReLU + weak RDM (0.1) | **1,790.16** | **2,912.67** | 0.8119 |
| Hard ReLU + weak RDM (0.1) | 1,796.30 | 2,912.02 | 0.8155 |
| RepReLU + RDM 1.0 | 1,771.92 | 2,873.15 | 0.8124 |
| RepReLU + RDM 1.0, mu=-1 | 1,787.53 | 2,890.17 | 0.8122 |
| RepReLU + RDM 1.0, mu=-1, match target+predictor | 1,780.68 | 2,890.09 | 0.8116 |

RepReLU itself provides the largest clean dense-to-sparse improvement at weak
RDM (about $26 MAE). Weak RDM helps both dense and rectified models. A matched
full-validation re-score of RepReLU + RDM 1.0 produced **$1,623.64 MAE and
$2,605.44 RMSE**, improving the weak sparse checkpoint by $3.30 MAE and $25.04
RMSE. It reduced TTNC retrieval@5 from 0.758 to 0.753, reinforcing the cost vs
neighborhood-quality tradeoff.

## Support audit

For the `mu=-1`, weak-RDM sparse checkpoint on all 3,188 validation patients:

- target active fraction: 58.36%
- predictor active fraction: 87.84%
- target effective active dimensions: 114.5 / 128
- target/predictor support Jaccard: 0.608
- target support turnover vs observed code-set turnover: Spearman rho 0.471
  over 101,880 adjacent claim pairs

Support is semantically meaningful, but not independently superior downstream:

| Frozen next-claim features | Cost MAE | TTNC accuracy | TTNC macro-F1 |
|---|---:|---:|---:|
| Magnitude | 1,660.17 | 0.493 | 0.280 |
| Support only | 1,754.06 | 0.404 | 0.166 |
| Magnitude + support | 1,659.30 | 0.484 | 0.268 |

The sparse target distribution is not being reproduced by the predictor.
Matching target and predictor at weight 1 only reduced predictor activity from
87.84% to 86.64% (target activity 59.68%). It did make support more useful when
concatenated with magnitude: cost MAE improved by $4.91 and RMSE by $10.70 over
magnitude alone, while TTNC performance still fell. Direct support alignment or
an explicit activity budget is therefore a cleaner follow-up than still more
RDM pressure.

## Predictor-capacity interaction

The capacity proxy is the dense Level-2 decoder bottleneck. All rows use the
same n=2,000 validation prefix.

| Capacity | Dense MAE | Sparse MAE | Sparse MAE gain |
|---|---:|---:|---:|
| 32 | 1,837.04 | **1,779.60** | +57.44 |
| 64 | **1,790.76** | 1,820.77 | -30.01 |
| Full | 1,816.02 | **1,790.16** | +25.86 |

The interaction is non-monotonic at seed 42, so the cap-32 result was replicated:

| Seed | Dense MAE | Sparse MAE | MAE gain | Dense RMSE | Sparse RMSE | RMSE gain |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1,837.04 | 1,779.60 | +57.44 | 2,963.98 | 2,906.93 | +57.05 |
| 43 | 1,827.47 | 1,817.91 | +9.55 | 2,945.16 | 2,937.85 | +7.31 |
| 44 | 1,827.19 | 1,831.04 | -3.85 | 2,951.83 | 2,945.03 | +6.80 |
| Mean | 1,830.57 | 1,809.52 | +21.05 | 2,953.66 | 2,929.94 | +23.72 |

Sparse cap-32 improves RMSE in all three seeds and MAE in two of three, but the
mean is dominated by seed 42. This is evidence of a capacity interaction, not
yet evidence for a stable best production recipe.

## Four-epoch cap-32 promotion

Dense Gaussian and RepReLU/Laplace cap-32 were retrained for four epochs over
encoder seeds 42--44 and evaluated on the full frozen validation set (n=3,188).

| Seed | Dense MAE | Sparse MAE | MAE gain | Dense RMSE | Sparse RMSE | RMSE gain |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1,643.75 | **1,597.26** | +46.49 | 2,659.53 | **2,594.70** | +64.83 |
| 43 | 1,644.64 | **1,617.32** | +27.31 | 2,649.94 | **2,619.10** | +30.84 |
| 44 | 1,655.63 | **1,654.70** | +0.92 | 2,675.04 | **2,666.18** | +8.87 |
| Mean | 1,648.00 | **1,623.10** | +24.91 | 2,661.51 | **2,626.66** | +34.85 |

Sparse wins both dollar metrics in all three seeds. Log-RMSE is effectively
tied on average (sparse-minus-dense gain -0.0004), while TTNC retrieval@5 and
ARI fall by 0.0118 and 0.0460. Silhouette rises by 0.0129. The replicated result
therefore supports a real cost-oriented capacity/sparsity interaction, with a
consistent neighborhood-structure tradeoff rather than a universal embedding
improvement.

## Frozen next-claim generation

The promoted checkpoints were tested with the established copy-plus-residual
decoder for 20 epochs. All three encoder seeds used the same decoder seed and
all training labels, isolating encoder-training variance.

| Encoder seed | Dense CPT AP | Sparse CPT AP | Dense ICD AP | Sparse ICD AP | Dense cost MAE | Sparse cost MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.4597 | **0.4674** | 0.5046 | **0.5145** | 1,704.25 | **1,645.81** |
| 43 | 0.4505 | **0.4676** | 0.5038 | **0.5170** | 1,765.80 | **1,648.33** |
| 44 | 0.4514 | **0.4774** | **0.5053** | 0.4963 | 1,747.26 | **1,672.00** |

Sparse improves CPT AP in all three seeds (+0.0169 mean), CPT micro-F1 in all
three (+0.0132 mean), and frozen cost MAE/RMSE in all three (-$83.72/-$90.83
mean). ICD AP improves in two of three (+0.0047 mean); ICD F1 is tied on average
(+0.0003). TTNC accuracy is nearly flat but slightly worse (-0.0035).

At 10% decoder labels for seed 42, sparse slightly improves CPT/ICD AP and TTNC
accuracy but worsens frozen cost MAE/RMSE by $25.27/$67.49. Sparse cap-32 is
therefore a strong full-supervision generative representation, not a demonstrated
low-label cost improvement.

## Future-claim cost mediation

A seed-42 comparison repeated the strict
`embedding -> predicted claim probabilities -> linear cost` test at 10% and
100% cost labels. The result remains negative:

| Cost labels | Representation | Direct embedding MLP MAE | Predicted-claims linear MAE | Decoder-hidden MLP MAE |
|---:|---|---:|---:|---:|
| 10% | Dense | **1,745.94** | 2,136.32 | 1,882.20 |
| 10% | Sparse | **1,813.78** | 2,134.45 | 1,820.77 |
| 100% | Dense | **1,529.00** | 1,766.83 | 1,542.47 |
| 100% | Sparse | 1,535.70 | 1,701.27 | **1,532.87** |

Flat predicted claim probabilities do not mediate the cost value: they trail
the corresponding direct embedding by $166--390 MAE. Even oracle next-claim
codes are worse through the same sparse linear bottleneck. The sparse decoder
hidden state, however, is within $7 at 10% labels and improves by $2.83 at full
labels. This motivates a compact learned claim-set embedding or retrieval
summary, not another flat-distribution mediation run.

### Compact learned mediator control

The proposed nonlinear claim-set mediator was tested on the sparse cap-32 e4
checkpoint. At 10% cost labels, three downstream seeds initially made the
16-unit predicted-claim MLP look useful: it beat the original 256-unit direct
embedding MLP in all three seeds. A width-matched direct control reverses that
interpretation:

| Frozen pathway (10% cost labels) | Hidden width | Mean MAE | Mean RMSE |
|---|---:|---:|---:|
| Direct embedding MLP | 256 | 1,825.26 | 2,972.80 |
| Predicted claims MLP | 16 | 1,810.35 | 2,933.54 |
| Direct embedding linear | linear | 1,730.61 | 2,838.49 |
| Direct embedding MLP | 16 | **1,661.05** | **2,728.08** |

The compact direct probe beats the compact predicted-claim mediator by
$149.30 MAE and $205.46 RMSE. The apparent mediator gain was therefore a
downstream-capacity/regularization effect, not evidence that generated claims
carry the embedding's cost signal. The oracle-claim controls are worse still.
At full cost supervision, the best predicted-claim compact probe (width 32,
$1,572.70 MAE) also trails the direct embedding MLP ($1,535.70). This closes the
claim-set mediation branch for the current decoder.

## Direct support-alignment ablation

A differentiable support loss was added to RDMReg and swept at 0.01, 0.03, and
0.10 on the seed-42 sparse cap-32 two-epoch recipe. All rows use the full frozen
validation set (n=3,188):

| Support weight | Cost MAE | Cost RMSE | Log RMSE | Retrieval@5 | ARI |
|---:|---:|---:|---:|---:|---:|
| 0 (matched baseline) | 1,608.87 | 2,612.88 | 0.8273 | 0.7688 | 0.1477 |
| 0.01 | 1,608.88 | **2,612.31** | 0.8273 | **0.7694** | 0.1523 |
| 0.03 | **1,608.26** | 2,612.57 | 0.8269 | 0.7688 | 0.1522 |
| 0.10 | 1,608.53 | 2,613.00 | **0.8266** | 0.7666 | **0.1528** |

These are noise-scale movements. At weight 0.03, support Jaccard rises only
from 0.5636 to 0.5663 and prediction/target support MSE falls from 0.1747 to
0.1733, while predictor activity actually rises from 67.31% to 67.56% (target
activity is about 56.4%). The smooth magnitude proxy cannot enforce exact-zero
activity. Stronger weight also begins to erode retrieval. Reject this loss as
an activity-control mechanism; a future revisit would need an explicit gate,
top-k budget, or L0-style relaxation.

## Full-capacity RDM weight sweep

Dense Gaussian and RepReLU/Laplace full-capacity controls were swept at RDM
weights 0.03, 0.10, and 0.30 with identical seed-42 two-epoch training and
full-validation scoring:

| RDM weight | Distribution | Cost MAE | Cost RMSE | Log RMSE | Retrieval@5 |
|---:|---|---:|---:|---:|---:|
| 0.03 | Dense Gaussian | 1,643.46 | 2,640.10 | 0.8266 | 0.7716 |
| 0.03 | RepReLU/Laplace | **1,628.28** | **2,630.79** | 0.8320 | 0.7607 |
| 0.10 | Dense Gaussian | 1,633.68 | **2,630.36** | **0.8249** | **0.7729** |
| 0.10 | RepReLU/Laplace | **1,626.94** | 2,630.48 | 0.8318 | 0.7575 |
| 0.30 | Dense Gaussian | **1,616.15** | **2,613.08** | **0.8222** | **0.7738** |
| 0.30 | RepReLU/Laplace | 1,626.32 | 2,626.65 | 0.8319 | 0.7616 |

The interaction is clear enough for a next gate. Dense Gaussian improves
monotonically through 0.30 and is the best full-capacity arm on both dollar
metrics, log RMSE, and retrieval. RepReLU/Laplace is almost flat from 0.03 to
0.30 and no longer wins at the strongest regularization. Sparse geometry does
not generally require or tolerate the same RDM pressure as dense geometry.
This does not displace sparse cap-32 e4 as the replicated cost/generation
candidate, but dense RDM 0.30 is the correct full-capacity control to carry
forward.

## Cost attribution ladder

The large improvement over a constant predictor was decomposed using the same
frozen standardized Ridge probe, the same cost-stratified label subsets, encoder
seeds 42--44, and all 3,188 validation patients. Raw-history features exclude
the held-out next claim. Random encoders match each dense or sparse model's
architecture and configuration; shuffled rows break the train feature/label
relationship as a negative control.

One bookkeeping correction matters: the previously printed $3,152.63 baseline
is a train-cohort RMSE. The correctly held-out validation baseline is $3,080.97.
The sparse cap-32 e4 mean therefore reduces held-out constant-baseline RMSE by
$454.31 (14.7%), not 16.7%.

| Frozen condition | 10% labels MAE / RMSE | 25% labels MAE / RMSE | 100% labels MAE / RMSE |
|---|---:|---:|---:|
| Constant log-mean | 2,102 / 3,075 | 2,109 / 3,082 | 2,108 / 3,081 |
| Four history statistics | **1,972 / 2,998** | 1,965 / 3,003 | 1,951 / 2,982 |
| Raw history hash | 2,002 / 3,040 | 1,871 / 2,867 | 1,798 / 2,779 |
| Dense random encoder | 2,005 / **2,993** | 1,789 / **2,772** | 1,672 / 2,662 |
| Dense SSL | 2,027 / 3,068 | 1,790 / 2,800 | 1,648 / 2,662 |
| Sparse random encoder | 2,040 / 3,013 | 1,827 / 2,802 | 1,702 / 2,690 |
| Sparse SSL | 2,033 / 3,064 | **1,744 / 2,750** | **1,623 / 2,627** |
| Sparse SSL, shuffled | 2,433 / 3,432 | 2,179 / 3,176 | 2,126 / 3,106 |

This changes the attribution, not the usefulness result:

- At full labels, sparse SSL beats the raw-history hash by $174.61 MAE and
  $152.04 RMSE, and its matched random encoder by $79.38/$62.97.
- Dense SSL is effectively tied with its random encoder on dollar RMSE
  ($2,661.51 vs $2,661.76), although it improves MAE by $23.67 and log RMSE.
- Sparse SSL's advantage appears at 25% labels and strengthens at 100%. It is
  not a low-label cost advantage: at 1--5%, high-dimensional frozen probes are
  unstable, and at 10% the simple statistics/random controls are competitive or
  better.
- Shuffling returns performance to approximately the constant baseline,
  confirming that the aligned history representation—not probe leakage—drives
  the full-label result.

The defensible claim is therefore narrower than “pretraining explains the full
14.7%”: claims history and random nonlinear features explain a large share,
while sparse predictive pretraining contributes an additional replicated
2.3% RMSE improvement over its matched random architecture and 5.5% over the
raw-history hash at full supervision.

### End-to-end supervised control

Dense cap-32 encoders were trained from scratch directly on all cost labels for
20 epochs over seeds 42--44. Sparse cap-32 e4 SSL checkpoints were separately
fine-tuned with the same nonlinear cost head and schedule, using the established
smaller encoder learning rate.

| Seed | Supervised scratch MAE / RMSE | Sparse SSL fine-tune MAE / RMSE |
|---:|---:|---:|
| 42 | 1,483.65 / 2,477.59 | **1,435.86 / 2,395.89** |
| 43 | 1,490.51 / 2,473.39 | **1,438.97 / 2,469.04** |
| 44 | **1,456.48 / 2,409.71** | 1,511.99 / 2,507.03 |
| Mean | 1,476.88 / **2,453.56** | **1,462.27** / 2,457.32 |

Fine-tuning improves mean MAE by $14.61 but slightly worsens mean RMSE by $3.76
and has much higher seed variance (MAE standard deviation $43.08 vs $18.00).
It wins both metrics in two seeds and loses sharply in seed 44. Pretraining can
provide a strong initialization, but this is not a stable full-label cost win
over supervised scratch.

### Strong exact and nonlinear raw-history controls

The earlier 128-dimensional history hash was replaced by an exact 5,900-column
sparse representation: recency-weighted CPT/ICD/TTNC counts, separate
last-claim indicators, and four history statistics. It excludes the held-out
next claim. Ridge regularization is selected using an inner training-only split.
A nonlinear control uses histogram gradient boosting over the 256 most frequent
history columns, selected without cost labels.

| Cost labels | Exact history Ridge MAE / RMSE | Boosted history MAE / RMSE |
|---:|---:|---:|
| 1% | 1,907.65 / 2,945.83 | **1,767.97 / 2,799.37** |
| 5% | 1,863.81 / 2,883.15 | **1,663.12 / 2,707.57** |
| 10% | 1,803.75 / 2,804.21 | **1,586.25 / 2,606.86** |
| 25% | 1,742.94 / 2,741.00 | **1,513.71 / 2,532.48** |
| 100% | 1,682.32 / 2,674.73 | **1,449.57 / 2,444.69** |

The boosted baseline is the most stable cost model tested: at full labels its
three-seed standard deviations are only $3.83 MAE and $12.34 RMSE. It beats the
three-seed mean of supervised scratch by $27.31 MAE/$8.87 RMSE and sparse SSL
fine-tuning by $12.70/$12.63. It also decisively beats frozen sparse SSL at every
label fraction. This overturns the earlier cost headline: the apparent large
downstream gain was mostly a weak-baseline artifact. The current evidence for
pretrained embeddings is strongest in reusable/frozen CPT and ICD generation,
not state-of-the-art cost prediction.

## High-cost top-1.5% proxy

The earlier cost comparison optimized population-wide error, but the historical
operational use case selected only the highest-risk 1.5% of members. A new
validation-only probe therefore ranks all 3,188 validation histories, selects
exactly 48, and measures overlap with the true top-cost 48. The training label
is the top 226 of 15,050 histories (training threshold $15,644). The exact same
256 raw-history features and pretrained dense/sparse embeddings are evaluated
alone and together across seeds 42, 43, and 44.

| Ranking model | Precision / recall @1.5% | Lift | PR-AUC | NDCG@1.5% | Cost capture | Cost-capture lift |
|---|---:|---:|---:|---:|---:|---:|
| Dense embedding, tail objective | 0.097 | 6.46x | 0.0778 | 0.1029 | 2.59% | 1.72x |
| Sparse embedding, tail objective | 0.139 | 9.22x | 0.0815 | 0.1246 | 2.76% | 1.84x |
| Raw boosted history, cost objective | 0.201 | 13.38x | 0.1273 | 0.2283 | **3.54%** | **2.35x** |
| Raw boosted history, tail objective | 0.181 | 11.99x | 0.1409 | 0.2194 | 3.07% | 2.04x |
| Dense embedding + raw, cost objective | 0.174 | 11.53x | 0.1537 | 0.2362 | 3.44% | 2.29x |
| Sparse embedding + raw, cost objective | 0.174 | 11.53x | 0.1607 | 0.2605 | 3.47% | 2.31x |
| Dense embedding + raw, tail objective | **0.222** | **14.76x** | **0.1791** | **0.2813** | 3.33% | 2.21x |
| Sparse embedding + raw, tail objective | 0.208 | 13.84x | 0.1512 | 0.2447 | 3.33% | 2.21x |

Precision and recall are equal because both the predicted and actual sets have
48 members. The dense tail hybrid retrieves 12, 10, and 10 true top-cost
members by seed, versus 9, 10, and 10 for boosted raw-cost regression. Thus the
hybrid's 0.222 versus 0.201 mean is directionally consistent but corresponds to
only one additional member on average. The hybrid result is nonetheless not a
random-ranking artifact: even eight overlaps has a one-sided hypergeometric
probability below 3.5e-7 when 48 of 3,188 members are selected at random.

The objective matters. Direct tail training produces the best top-set overlap
and ranking metrics, while raw cost regression captures slightly more dollars.
Sparse embeddings are stronger than dense embeddings in isolation, but the
dense hybrid is best once raw history is present. This supports a narrow claim:
pretraining contributes complementary tail-ranking signal, but it does not yet
establish a robust operational win over boosted history.

This is only a **high-cost proxy**. The parquet has no mortality outcome,
service dates, or stable patient identifier. It therefore cannot reproduce the
combined death-plus-high-cost endpoint, establish a prospective prediction
boundary, or guarantee that multiple histories from one member cannot cross
splits. Those fields are required before test-set promotion or clinical use.

### Ordinal cost-category ablation

A follow-up replaced both continuous cost and the binary tail target with five
balanced training categories: remainder, 5-20%, 1.5-5%, 0.5-1.5%, and top 0.5%.
The corresponding training thresholds were $5,913, $9,270, $15,644, and
$22,025. Rankings used either expected category severity or the summed
probability of the two classes comprising the top 1.5%.

| Features / objective | Precision / recall @1.5% | Lift | PR-AUC | NDCG@1.5% | Cost capture |
|---|---:|---:|---:|---:|---:|
| Raw continuous cost | **0.201** | **13.38x** | 0.1273 | **0.2283** | **3.54%** |
| Raw binary tail | 0.181 | 11.99x | **0.1409** | 0.2194 | 3.07% |
| Raw ordinal severity | 0.153 | 10.15x | 0.1154 | 0.1913 | 2.79% |
| Raw ordinal top-tail probability | 0.153 | 10.15x | 0.1096 | 0.2045 | 2.74% |
| Dense hybrid binary tail | **0.222** | **14.76x** | **0.1791** | **0.2813** | **3.33%** |
| Dense hybrid ordinal severity | 0.160 | 10.61x | 0.1024 | 0.1485 | 3.19% |
| Sparse hybrid binary tail | 0.208 | 13.84x | **0.1512** | **0.2447** | **3.33%** |
| Sparse hybrid ordinal severity | **0.208** | **13.84x** | 0.1366 | 0.2239 | 3.24% |

The ordinal target does not improve the operating point. It reduces raw and
dense-hybrid performance substantially. Sparse-hybrid ordinal severity ties
the sparse binary model on top-48 overlap (10 members per run on average), but
loses PR-AUC, NDCG, and cost capture. Intermediate cost bands therefore add
optimization burden without useful ordering signal for this exact tail. Keep
continuous raw-cost regression and binary dense-hybrid tail classification as
the two candidates; do not promote this multiclass formulation.

### Raw-cost / dense-tail candidate showdown

The two surviving scores were evaluated at intervention budgets from 0.5% to
5%. Three predeclared blends combine percentile ranks rather than incompatible
raw dollar and probability scales. Each cell reports mean precision / future
cost capture over seeds 42-44.

| Ranking | Top 0.5% | Top 1% | Top 1.5% | Top 2% | Top 3% | Top 5% |
|---|---:|---:|---:|---:|---:|---:|
| Dense hybrid tail | **.208/.014** | **.177/.024** | **.222/.033** | .240/.044 | .295/.062 | .358/.094 |
| Blend, 25% raw | .188/**.014** | .167/**.026** | .215/.035 | .255/.045 | **.309/.064** | .388/.099 |
| Blend, 50% raw | .146/.013 | .156/**.026** | **.222/.037** | **.276/.046** | .306/**.064** | **.396/.100** |
| Blend, 75% raw | .167/.013 | .135/.025 | .215/**.037** | .260/**.046** | .302/.063 | .385/.099 |
| Raw boosted cost | .167/.013 | .104/.024 | .201/.035 | .240/.045 | .288/.062 | .354/.097 |

At the original 1.5% budget, the 50/50 blend retrieves 10, 11, and 11 of 48
true tail members. Dense hybrid retrieves 12, 10, and 10; raw cost retrieves 9,
10, and 10. The blend therefore retains the hybrid's mean 0.222 precision while
raising cost capture from 3.33% to 3.68%, above raw cost's 3.54%. Its mean NDCG
also rises to 0.297 from 0.281 hybrid and 0.228 raw. The 25% raw blend has the
best PR-AUC (0.193), while the 75% raw blend has slightly higher cost capture
(3.72%) at the expense of one-third of a true positive per run.

Raw and hybrid share only 24-27 of their 48 selected members per seed. Their
complementarity is real, and blending is useful from 1.5% through 5%. Below
1.5%, dense hybrid alone has the highest precision. Carry the 50/50 blend as the
balanced validation candidate for the historical 1.5% operating point. Because
the blend weights were compared on this validation set, this is exploratory
selection rather than independent confirmation; test remains untouched.

### Frozen dense + sparse representation fusion

Matched dense and sparse cap-32 embeddings were extracted from separately
pretrained, frozen encoders and supplied together to downstream tail heads. This
tests representational complementarity, not a jointly trained dual-projection
encoder.

| Frozen inputs / tail head | Precision / recall @1.5% | Lift | PR-AUC | NDCG@1.5% | Cost capture |
|---|---:|---:|---:|---:|---:|
| Dense, logistic | 0.097 | 6.46x | 0.0778 | 0.1029 | 2.59% |
| Sparse, logistic | **0.139** | **9.22x** | **0.0815** | **0.1246** | **2.76%** |
| Dense + sparse, logistic | 0.111 | 7.38x | 0.0798 | 0.1055 | 2.75% |
| Dense, boosted | 0.174 | 11.53x | 0.1203 | 0.1861 | 3.10% |
| Sparse, boosted | 0.167 | 11.07x | 0.1172 | 0.1905 | 3.17% |
| Dense + sparse, boosted | **0.201** | **13.38x** | **0.1318** | **0.2252** | **3.25%** |
| Raw + dense, boosted | **0.222** | **14.76x** | **0.1791** | **0.2813** | 3.33% |
| Raw + sparse, boosted | 0.208 | 13.84x | 0.1512 | 0.2447 | 3.33% |
| Raw + dense + sparse, boosted | 0.201 | 13.38x | 0.1541 | 0.2414 | **3.39%** |
| Raw + dense + sparse, neural concat | 0.188 | 12.45x | 0.1344 | 0.2326 | 3.04% |
| Raw + dense/sparse learned gate | 0.215 | 14.30x | 0.1316 | 0.2274 | 3.28% |

The embedding-only nonlinear control provides the clean positive result. Joint
dense+sparse input retrieves 10, 10, and 9 true top-48 members, versus 9, 10,
6 for dense and 9, 8, 7 for sparse. It improves every reported ranking metric
over both single embeddings. The linear concat does not improve on sparse,
showing that the complementarity is interaction-dependent rather than simply
more linearly exposed signal.

Once raw history is present, adding sparse to raw+dense hurts precision and
ranking quality, although it slightly increases dollar capture. A matched
learned gate recovers most of the precision lost by neural concatenation (0.215
versus 0.188). Its mean dense weight is approximately 0.50 but its within-run
standard deviation is 0.31-0.35, evidence of active per-dimension routing rather
than a fixed average. It early-stops after only 1-3 epochs and still trails the
boosted raw+dense head on PR-AUC and NDCG.

Thus producing both frozen representations is useful when embeddings must stand
alone and a nonlinear downstream learner is available. It does not improve the
current raw-history-assisted top-1.5% candidate. Keep the 50/50 raw-cost/rank
blend with raw+dense tail score for this task; consider dual representation
output as a reusable embedding interface, not as an automatic concatenation.

### LeVJEPA patient-view update

The later LeVJEPA patient-view objective was replicated over encoder seeds
42-44 and evaluated with the same online-weight tail protocol. A 25% raw-cost /
75% LeVJEPA-tail percentile-rank blend reaches 0.229 +/- 0.021 precision,
0.190 PR-AUC, 0.298 NDCG, and 3.64% cost capture at 1.5%. This is a small Pareto
improvement over the cap-32 50/50 blend (0.222 precision, 0.184 PR-AUC, 0.297
NDCG, 3.68% capture), not a decisive replacement. See
`docs/levjepa_patient_views_20260828.md` for the Polyak/online, attention,
sparse-link, and cross-recipe fusion ablations.

## Twenty-epoch seed-42 gate

One matched pair was restarted from scratch for 20 epochs before considering a
three-seed long-run promotion.

| Epochs | Representation | Cost MAE | Cost RMSE | Log RMSE | TTNC retrieval@5 |
|---:|---|---:|---:|---:|---:|
| 4 | Dense | 1,643.75 | 2,659.53 | 0.8269 | **0.7845** |
| 4 | Sparse | **1,597.26** | **2,594.70** | 0.8250 | 0.7682 |
| 20 | Dense | 1,640.22 | 2,674.28 | 0.8294 | 0.7829 |
| 20 | Sparse | **1,599.23** | **2,602.50** | **0.8220** | 0.7679 |

Sparse remains clearly better than matched dense at 20 epochs (-$40.98 MAE,
-$71.78 RMSE), so the four-epoch result is not merely a transient early-training
artifact. But sparse e20 is $1.97/$7.80 worse than sparse e4 on MAE/RMSE. A
three-seed e20 cost promotion is therefore not justified; retain early stopping
near four epochs for the cost-oriented checkpoint.

The generative result has a different optimum:

| Checkpoint | CPT AP | ICD AP | CPT F1 | ICD F1 | TTNC acc. |
|---|---:|---:|---:|---:|---:|
| Dense e20 | 0.4583 | 0.5002 | 0.5039 | 0.5838 | **0.6048** |
| Sparse e20 | **0.4863** | **0.5051** | **0.5305** | **0.5924** | 0.5920 |

Against dense e20, sparse e20 gains 0.0279 CPT AP, 0.0266 CPT F1, 0.0049
ICD AP, and 0.0086 ICD F1. It loses 0.0129 TTNC accuracy. Compared with sparse
e4, e20 improves CPT AP/F1 by 0.0189/0.0132 but slightly reduces ICD AP and does
not improve cost. The practical output is a Pareto frontier: sparse e4 for the
cost checkpoint, sparse e20 for the strongest CPT-oriented generative proof of
concept, and neither checkpoint as a TTNC specialist.

## Next queue

1. Keep the replicated sparse cap-32 e4 recipe as the cost candidate; do not
   spend on a three-seed e20 cost promotion.
2. Replicate sparse e20 generation only if the product decision prioritizes CPT
   set ranking over cost and TTNC accuracy.
3. Carry dense Gaussian RDM 0.30 as the full-capacity control in future encoder
   comparisons; replicate it only if full capacity becomes a product candidate.
4. Do not continue compact claim-probability mediation with the current decoder.
   If interpretability is needed, evaluate retrieval-based exemplars or typed
   claim-state summaries instead.
5. Only revisit support matching with a genuinely discrete activity mechanism
   (learned gates, top-k budget, or L0-style relaxation), not the rejected smooth
   magnitude proxy.
6. Treat sparse SSL as a reusable generative-representation result, not a
   population-wide cost promotion candidate. Use boosted raw history as the
   current global-error baseline.
7. For the top-1.5% use case, carry forward both Pareto candidates: the 25/75
   raw-cost/LeVJEPA-tail blend for slightly higher precision and ranking quality,
   and the 50/50 raw-cost/cap-32-tail blend for slightly higher cost capture.
   Keep component scores for interpretability and budget changes.
   Require mortality labels, stable member IDs, and a prospective time boundary
   before treating the validation gain as operational evidence.
8. Evaluate the real utility rule explicitly (death risk, high cost, and the
   action capacity/cost), rather than using high cost alone as a permanent
   surrogate.
9. If a product consumes embeddings without raw engineered history, carry
   dense+sparse output with a nonlinear downstream head. Do not assume linear
   concatenation will expose the complementary signal.
10. Keep test untouched until a recipe is selected from replicated validation
   results.

## Artifacts

- `experiments/lpwm_claims_pilot_20260826/screen_selected_anchor_seed42_e2`
- `experiments/lpwm_claims_pilot_20260826/mechanism_ablation_seed42_e2`
- `experiments/lpwm_claims_pilot_20260826/capacity_ablation_seed42_e2`
- `experiments/lpwm_claims_pilot_20260826/cap32_replication_seed43_e2`
- `experiments/lpwm_claims_pilot_20260826/cap32_replication_seed44_e2`
- `experiments/lpwm_claims_pilot_20260826/cap32_promotion_seed42_e4`
- `experiments/lpwm_claims_pilot_20260826/cap32_promotion_seed43_e4`
- `experiments/lpwm_claims_pilot_20260826/cap32_promotion_seed44_e4`
- `experiments/lpwm_claims_pilot_20260826/cap32_generation_seed{42,43,44}_{dense,sparse}_e4`
- `experiments/lpwm_claims_pilot_20260826/cap32_mediation_seed42_{dense,sparse}_e4`
- `experiments/lpwm_claims_pilot_20260826/cap32_longrun_seed42_e20`
- `experiments/lpwm_claims_pilot_20260826/cap32_generation_seed42_{dense,sparse}_e20`
- `experiments/lpwm_claims_pilot_20260826/compact_claim_mediator_seed42_sparse_e4`
- `experiments/lpwm_claims_pilot_20260826/compact_claim_mediator_width_matched_seed42_sparse_e4`
- `experiments/lpwm_claims_pilot_20260826/support_alignment_seed42_e2`
- `experiments/lpwm_claims_pilot_20260826/rdm_weight_sweep_seed42_e2`
- `experiments/lpwm_claims_pilot_20260826/cost_attribution_ladder_v0`
- `experiments/lpwm_claims_pilot_20260826/cost_attribution_supervised_scratch_dense_seed42`
- `experiments/lpwm_claims_pilot_20260826/cost_attribution_supervised_scratch_dense_seed{43,44}`
- `experiments/lpwm_claims_pilot_20260826/cost_attribution_ssl_finetune_sparse_seed{42,43,44}`
- `experiments/lpwm_claims_pilot_20260826/exact_raw_cost_baselines_v0`
- `experiments/lpwm_claims_pilot_20260826/high_cost_tail_probe_v0`
- `experiments/lpwm_claims_pilot_20260826/high_cost_tail_probe_ordinal_v1`
- `experiments/lpwm_claims_pilot_20260826/tail_candidate_showdown_v0`
- `experiments/lpwm_claims_pilot_20260826/dense_sparse_fusion_probe_v1`
