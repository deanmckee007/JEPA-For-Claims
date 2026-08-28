# Frozen next-claim generation probe (2026-08-18)

## Decision

Advance a **copy-plus-residual next-claim decoder** as the generative proof of
concept. Do not revive diffusion yet.

The frozen pretrained representations contain clear CPT/ICD next-claim signal
when decoder supervision is scarce. The correct generative factorization is not
to synthesize the whole claim independently: the immediately preceding claim is
an extremely strong persistence baseline, especially for ICD. The generator
should copy persistent codes and predict additions, removals, cardinality, and
TTNC transitions.

## Protocol

- Encoder checkpoint:
  `experiments/final_pairwise_20260814/selected_lejepa_anycode_seed42_e20/encoder.ckpt`
- Frozen data contract: `artifacts/data_contracts/claims_seed42_v1.json`
- Contract hash: `349d6b49a743282dbeb84b105338674cada85ee78b3fe2f2229d9ad5b4492644`
- Vocabulary hash: `7554200552fdf0e3609d927cfadb68c5444a656df4c77d929cf2ff71341893f0`
- Eligible cohort: 15,048 train and 3,186 validation patients with at least one
  historical claim and one held-out target claim.
- The test split was not accessed.
- Decoder: one hidden layer, train-only marginal-prior initialization, balanced
  CPT/ICD binary loss, TTNC cross-entropy, and cardinality regression.
- Frozen conditions: pretrained representation, patient-shuffled pretrained
  representation, and matched random encoder.
- Representation sources: `next_claim_prediction` and
  `patient_representation_pre_sae`.

## Ten-percent labels: three decoder seeds

Values are mean ± sample SD over seeds 42, 43, and 44. Each seed changes the
cost-stratified labeled subset, decoder initialization, and minibatch order. One
fixed pretrained checkpoint and one fixed random encoder are used, so this does
not measure encoder-training variance.

| Source | Condition | CPT AP | ICD AP | TTNC acc. | Cost MAE ($) | Cost RMSE ($) |
|---|---|---:|---:|---:|---:|---:|
| Next-claim state | Pretrained | **0.2219 ± 0.0136** | **0.1491 ± 0.0125** | 0.5998 ± 0.0036 | **1,839.79 ± 48.83** | **2,837.22 ± 40.51** |
| Next-claim state | Random encoder | 0.1201 ± 0.0033 | 0.0975 ± 0.0038 | **0.6036 ± 0.0022** | 1,919.51 ± 108.80 | 2,919.18 ± 128.23 |
| Patient pre-SAE | Pretrained | **0.2594 ± 0.0088** | **0.1645 ± 0.0022** | **0.6050 ± 0.0018** | 1,975.80 ± 103.56 | 2,956.24 ± 117.82 |
| Patient pre-SAE | Random encoder | 0.1387 ± 0.0021 | 0.1154 ± 0.0005 | 0.6042 ± 0.0024 | 1,978.75 ± 71.40 | **2,942.88 ± 96.05** |

Paired pretrained-minus-random AUPRC deltas were +0.1018 CPT and +0.0516 ICD
for the next-claim state, and +0.1207 CPT and +0.0491 ICD for patient pre-SAE.
TTNC showed no pretraining advantage. The next-claim state improved mean cost
MAE by $79.73 and RMSE by $81.96 versus random features, but with only three
downstream seeds this cost result remains a pilot.

Naively concatenating patient and next-claim states was rejected at 10% labels.
The pretrained concatenation reached CPT/ICD AP of 0.2744/0.1689, but cost MAE
worsened to $2,115.23, consistent with high-dimensional Ridge overfitting.

## Full-label seed-42 screen

| Source / baseline | Condition | CPT AP | ICD AP | TTNC acc. | Cost MAE ($) | Cost RMSE ($) |
|---|---|---:|---:|---:|---:|---:|
| Empirical marginal prior | Unconditional | 0.0877 | 0.0909 | 0.3242 | — | — |
| Last observed claim | Persistence | 0.2613 | **0.5113** | 0.5499 | — | — |
| Next-claim state | Pretrained | **0.3612** | 0.2085 | 0.6086 | **1,680.14** | **2,694.62** |
| Next-claim state | Random encoder | 0.2745 | **0.2165** | **0.6265** | 1,739.27 | 2,753.52 |
| Patient pre-SAE | Pretrained | **0.3838** | **0.2495** | 0.6127 | **1,626.44** | **2,616.90** |
| Patient pre-SAE | Random encoder | 0.3330 | 0.2375 | **0.6328** | 1,671.86 | 2,646.28 |

Pretraining clearly helps CPT ranking. Its full-label ICD and TTNC advantage is
small or absent because frozen random features preserve substantial raw-history
information. Patient pre-SAE remains the better full-label cost representation;
the next-claim state is more label-efficient for cost.

## Copy-plus-residual screen

The last-observed-claim baseline achieved CPT/ICD set F1 of 0.511/0.715. That
made independent whole-claim decoding the wrong comparison. A fixed positive
copy-logit boost was therefore added before the learned residual logits.

| Source | Condition | CPT AP | ICD AP | CPT F1 | ICD F1 | TTNC acc. |
|---|---|---:|---:|---:|---:|---:|
| Last observed claim | Copy only | 0.2613 | 0.5113 | **0.5107** | **0.7146** | 0.5499 |
| Next-claim state | Pretrained residual | **0.4471** | **0.5344** | 0.4723 | 0.5702 | 0.5873 |
| Next-claim state | Random residual | 0.4152 | 0.5201 | 0.4877 | 0.5805 | **0.6042** |
| Patient pre-SAE | Pretrained residual | **0.4496** | 0.5257 | **0.4932** | **0.5923** | 0.5954 |
| Patient pre-SAE | Random residual | 0.4429 | **0.5271** | 0.4853 | 0.5923 | **0.6149** |

The pretrained next-claim state provides the clearest residual ranking gain over
matched random features (+0.0319 CPT AP, +0.0143 ICD AP). Thresholded F1 still
trails copy-only because the current total-cardinality head underselects
persistent codes. The next implementation should predict code additions and
removals explicitly or retain copied codes unless a removal head fires.

## Calibration caveat

The balanced binary objective is useful for rare-code ranking but is not yet a
calibrated probability model. For example, the empirical prior's all-token
binary NLL is much lower because negatives dominate the vocabulary. Before
sampling claims, run a loss/calibration sweep using standard BCE plus a small
balanced/ranking term, validation-only temperature scaling, and copy-boost
calibration. AUPRC and calibrated NLL should be treated as separate objectives.

## Explicit add/remove result

The explicit event formulation was implemented and rejected as the primary
generator. The final version used calibrated event BCE plus a 0.1 balanced
ranking term, retained a code unless its removal probability exceeded 0.5, and
added a code only when its addition probability exceeded 0.5. Earlier
uncalibrated and expected-cardinality runs are marked invalid in their artifact
directories.

| Source | Condition | CPT set AP | ICD set AP | CPT F1 | ICD F1 |
|---|---|---:|---:|---:|---:|
| Next-claim state | Pretrained | 0.1629 | 0.4400 | 0.1813 | **0.4419** |
| Next-claim state | Random encoder | **0.1894** | **0.4571** | **0.1883** | 0.4290 |
| Patient pre-SAE | Pretrained | 0.1840 | **0.4681** | **0.1911** | **0.4511** |
| Patient pre-SAE | Random encoder | **0.2065** | 0.4614 | 0.1890 | 0.4347 |

Direct change-event ranking gave the same answer. On the next-claim state,
pretrained versus random add AP was 0.0996 versus 0.1112 for CPT and 0.0353
versus 0.0343 for ICD. Removal AP was 0.8639 versus 0.8579 for CPT and 0.5844
versus 0.5855 for ICD. The pretrained deltas are inconsistent and small, and
reconstructed sets remain well below the copy-only F1 baseline. No low-label
seed sweep is warranted for this head.

The likely issue is the task itself: additions are exceptionally sparse when
flattened over the full vocabulary (CPT prevalence 0.00026, ICD 0.00061), while
removal is primarily predictable from raw persistence. The earlier independent
set decoder remains the cleaner representation probe; a production generator
would need candidate retrieval or a sparse vocabulary-conditioned decoder.

## Cost plus next-claim auxiliary experiment

A shared one-hidden-layer trunk was trained from the frozen
`next_claim_prediction` representation. Every run used 10% cost labels (1,500
training patients), while auxiliary runs used next-claim CPT, ICD, and TTNC
labels for the full 15,048-patient training cohort. Cost-only and auxiliary
conditions used identical labeled subsets, initialization seeds, minibatch
seeds, epochs, and optimizer-step counts. Results are means over seeds 42--44;
test data was not accessed.

| Aux weight | Pretrained MAE ($) | Pretrained RMSE ($) | Random MAE ($) | Random RMSE ($) |
|---:|---:|---:|---:|---:|
| 0 | 1,945.55 | 2,975.91 | 2,379.22 | 3,534.80 |
| 0.03 | 1,906.23 | 2,907.40 | 2,329.46 | 3,457.95 |
| 0.1 | 1,886.04 | 2,888.62 | 2,280.57 | 3,378.03 |
| 0.3 | 1,847.34 | 2,843.33 | 2,220.98 | 3,280.02 |
| 1 | 1,811.72 | 2,810.89 | 2,149.90 | 3,158.44 |
| 3 | 1,776.88 | 2,775.10 | 2,101.51 | 3,093.35 |
| 10 | **1,721.37** | **2,721.61** | **1,960.83** | **2,922.63** |

At weight 10, auxiliary training improved the pretrained-feature model over its
paired cost-only baseline by $224.18 MAE and $254.30 RMSE on average. All three
seeds improved. The same auxiliary task improved random features even more, so
the gain is partly generic multitask/data regularization rather than a unique
property of JEPA pretraining. Crucially, pretrained features still finished
$239.46 better in MAE and $201.02 better in RMSE than the matched random encoder.

The high-weight pretrained head also decoded the held-out claim with mean CPT
AP 0.3637, ICD AP 0.2111, and TTNC accuracy 0.6101. The random encoder reached
0.3091, 0.2280, and 0.6288 respectively: pretraining retains a clear CPT and
cost advantage, but not an ICD or TTNC advantage. Cost improvement had not
turned over by weight 10, so 10 is the best tested setting, not a claimed
optimum.

This is the strongest proof of concept in the current repository: pretrained
embeddings support a useful next-claim auxiliary task, and sharing that task
with a low-label cost head materially improves dollar prediction. The next
step should add a raw sparse-history control and test whether candidate-based
generation improves ICD without sacrificing the cost benefit.

## Independent encoder-seed replication

The auxiliary result was reproduced on independently pretrained seeds 42, 43,
and 44 of the selected `lejepa_anycode_l1w01` recipe. These checkpoints use the
same architecture, data contract, vocabulary, and four-epoch pretraining
budget. Each encoder was evaluated with three downstream seeds, producing nine
paired comparisons per weight.

| Aux weight | MAE ($) | RMSE ($) | CPT AP | ICD AP | TTNC acc. |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,909.53 +/- 56.68 | 2,920.67 +/- 98.62 | 0.0877 | 0.0909 | 0.3242 |
| 3 | 1,767.22 +/- 43.60 | 2,753.00 +/- 61.16 | 0.3573 | 0.2027 | **0.6143** |
| 10 | **1,717.70 +/- 44.37** | **2,718.35 +/- 64.66** | **0.3629** | **0.2034** | 0.6139 |

Weight 10 improved MAE in all 9/9 encoder/downstream-seed pairs. The paired
MAE change was -$191.82 +/- $46.92. Weight 3 also won 9/9, with a paired change
of -$142.30 +/- $33.42. Per-encoder weight-10 improvements were $186.01,
$206.93, and $182.53 for encoder seeds 42, 43, and 44. This clears the original
concern that the result might be peculiar to one pretrained checkpoint.

## Parameter-efficient adapter pilot

A zero-initialized, 32-dimensional residual bottleneck was attached directly to
the frozen 128-dimensional next-claim state. It contains 8,608 trainable
adapter parameters and feeds linear task heads, making representation movement
explicit.

| Adapter | Aux weight | MAE ($) | RMSE ($) | Relative movement | Cosine to frozen |
|---|---:|---:|---:|---:|---:|
| Unconstrained | 0 | 2,256.40 | 3,508.46 | 0.227 | 0.964 |
| Unconstrained | 10 | 1,755.12 | 2,758.15 | 1.691 | 0.592 |
| Drift penalty 10 | 0 | 2,188.85 | 3,349.27 | 0.043 | 0.998 |
| Drift penalty 10 | 10 | **1,774.25** | **2,764.53** | 0.268 | 0.960 |

The unconstrained adapter recovers strong cost and generation performance but
rewrites the representation too aggressively. Drift regularization preserves
the pretrained geometry while retaining a large auxiliary gain, yet its MAE
still trails the simpler frozen shared-MLP result of $1,721.37. Therefore the
adapter is a valid geometry-preserving fallback, not the promoted default.

## Semantic, modality, and raw-history controls

The promoted weight-10 head was then tested against patient-shuffled auxiliary
labels, modality-specific targets, and a target-excluding 128-dimensional raw
history hash. Every comparison used the same cost subsets and downstream seeds.

| Condition | MAE ($) | RMSE ($) | CPT AP | ICD AP | TTNC acc. |
|---|---:|---:|---:|---:|---:|
| Pretrained cost only | 1,945.55 | 2,975.91 | 0.0877 | 0.0909 | 0.3242 |
| Pretrained shuffled all | 1,760.21 | 2,751.64 | 0.0819 | 0.0923 | 0.1909 |
| Pretrained aligned all | **1,721.37** | **2,721.61** | 0.3637 | 0.2111 | 0.6101 |
| Pretrained CPT only | 1,747.19 | 2,745.29 | 0.3592 | 0.0909 | 0.3242 |
| Pretrained ICD only | 1,774.93 | 2,768.04 | 0.0877 | 0.2137 | 0.3242 |
| Pretrained TTNC only | 1,723.61 | 2,725.66 | 0.0877 | 0.0909 | **0.6125** |
| Pretrained CPT + ICD | 1,740.48 | 2,735.95 | **0.3680** | 0.2078 | 0.3242 |
| Raw history cost only | 2,378.05 | 3,580.82 | 0.0877 | 0.0909 | 0.3242 |
| Raw history shuffled all | 2,219.77 | 3,309.38 | 0.0645 | 0.0760 | 0.2447 |
| Raw history aligned all | 2,183.68 | 3,247.75 | 0.3428 | **0.4805** | 0.6171 |

Aligned labels beat shuffled labels for all three seeds. Their paired MAE
advantage was $38.84 +/- $9.91 with pretrained features and $36.09 +/- $11.74
with raw history. Nevertheless, shuffled labels provide $185.34 of the
pretrained model's $224.18 gain over cost-only. Thus the cost improvement is
mostly generic multitask/stochastic regularization, with a smaller but stable
semantic component. Aligned supervision remains essential for actual decoding:
shuffled targets collapse held-out CPT/ICD AP and TTNC accuracy.

TTNC-only supervision nearly matches the full auxiliary cost result on
pretrained features, whereas it is much weaker on raw history. This makes TTNC
transition prediction the most efficient auxiliary cost regularizer currently
tested. Raw history strongly predicts persistent ICD/CPT codes but remains
$462.31 worse in MAE than aligned pretrained features, supporting the claim
that pretraining organizes history more usefully for cost rather than merely
retaining code identity.

The raw control is deliberately width-matched through signed hashing. It is a
fair downstream-capacity comparison, but not an upper bound on sparse-history
models; an exact sparse linear or retrieval model remains worth testing.

## Label-free consistency control

The shuffled-label result motivated a label-free control using the same
full-cohort feature batches. Clean and Gaussian-perturbed inputs were passed
through independent dropout views of the shared trunk, and their hidden states
were matched without claim labels.

| Objective | Pretrained MAE ($) | Raw-history MAE ($) |
|---|---:|---:|
| Cost only | 1,945.55 | 2,378.05 |
| Consistency 1 | 1,978.54 | 2,381.53 |
| Consistency 10 | 1,951.87 | 2,190.25 |
| Consistency 100 | 1,799.69 | 2,006.01 |
| Consistency 300 | 1,757.65 | 1,931.51 |
| Consistency 1000 | 1,811.65 | **1,839.58** |
| Shuffled claim labels | 1,760.21 | 2,219.77 |
| Aligned claim labels | **1,721.37** | 2,183.68 |

Consistency at weight 300 nearly matches shuffled-label MAE for pretrained
features, but geometry diagnostics reject it as an equivalent explanation. On
seed 42, shared-hidden feature standard deviation was 0.446 for cost-only,
1.129 for shuffled labels, and 1.203 for aligned labels. Across three seeds it
collapsed to 0.0225 at consistency weight 300 and 0.0119 at weight 1000; mean
hidden norm fell to 0.525 and 0.221. Supervised auxiliary tasks instead expand
and structure the shared state.

Thus label-free smoothness can uncover a compact cost solution and explains
part of the generic regularization benefit, but it does not reproduce a healthy
joint cost/generation representation. The full aligned objective remains the
promoted proof of concept when generation quality itself matters.

## SigReg anti-collapse control

The follow-up used the repo's own LE-JEPA convex SigReg formulation on the
shared hidden state (256 slices, 17 quadrature points), rather than introducing
a separate VICReg objective. A seed-42 weight sweep found that SigReg weight
0.2 was sufficient to counter the collapse induced by consistency weight 300.
The selected setting was then repeated for downstream seeds 42, 43, and 44.

| Feature source / objective | MAE ($) | RMSE ($) | Hidden feature std | Hidden norm |
|---|---:|---:|---:|---:|
| Pretrained, SigReg 0.2 | 1,706.73 +/- 65.95 | 2,724.11 +/- 73.24 | 0.927 +/- 0.012 | 14.61 +/- 0.09 |
| Pretrained, consistency 300 + SigReg 0.2 | **1,665.67 +/- 44.15** | **2,691.27 +/- 52.76** | 0.451 +/- 0.002 | 7.42 +/- 0.03 |
| Raw history, SigReg 0.2 | 2,123.03 +/- 53.03 | 3,185.29 +/- 65.34 | 0.946 +/- 0.005 | 14.80 +/- 0.08 |
| Raw history, consistency 300 + SigReg 0.2 | 2,124.61 +/- 33.46 | 3,188.69 +/- 62.94 | 0.464 +/- 0.003 | 7.45 +/- 0.05 |

This answers the anti-collapse question directly: nothing beyond SigReg was
required. Consistency plus SigReg retained a non-degenerate shared state and
outperformed aligned claim-label supervision on the same pretrained features
(MAE 1,665.67 versus 1,721.37). The identical label-free objective remained
$458.94 worse with width-matched raw history, strengthening the evidence that
the pretrained embedding carries useful cost structure rather than the head
merely benefiting from generic regularization. The result is a cost-oriented
proof of concept; because this objective has no token targets, it does not
replace aligned auxiliary supervision when code generation is the endpoint.

## Robustness and label efficiency follow-ups

The consistency-plus-SigReg result was repeated across three independently
pretrained recipe-matched encoders and three downstream seeds per encoder. It
achieved $1,680.57 +/- $28.95 MAE over all nine runs, with hidden feature
standard deviation 0.443 +/- 0.004. Per-encoder means ranged only from $1,664.04
to $1,698.59, so the effect is not specific to the selected epoch-20 encoder.

A six-point label-efficiency curve on the selected encoder showed where the
auxiliary objectives help:

| Cost-label fraction | Cost only | TTNC only | All tokens | Consistency + SigReg |
|---:|---:|---:|---:|---:|
| 1% | 2,770.34 | 2,450.64 | 2,348.65 | **2,232.11** |
| 2% | 2,534.73 | 2,125.17 | 2,118.77 | **1,974.94** |
| 5% | 2,364.17 | 2,014.21 | 1,984.72 | **1,848.36** |
| 10% | 1,945.55 | 1,723.61 | 1,721.37 | **1,665.67** |
| 25% | 1,683.11 | 1,636.10 | **1,624.98** | 1,625.45 |
| 100% | **1,575.44** | 1,620.19 | 1,654.12 | 1,624.16 |

The benefit is concentrated in the low-label regime. Consistency plus SigReg
beat cost-only in every paired seed through 25%, but all auxiliary objectives
hurt once every training cost label was available.

## Future-claim cost mediation

A separately trained frozen decoder tested the strict pathway
`embedding -> predicted CPT/ICD/TTNC probabilities -> linear cost`. The decoder
had normal validation quality (CPT AP 0.360, ICD AP 0.210, TTNC accuracy 0.615),
but predicted claims were worse than a direct embedding probe at every tested
label fraction. At 10%, direct-embedding MLP MAE was $1,753.75 versus $2,005.30
for predicted claims. The decoder hidden-state MLP was also worse ($1,889.83).

Even actual next-claim codes were weak through the same high-dimensional sparse
linear bottleneck ($2,611.13 at 10%), so this is not evidence that future claims
contain no cost information. It rejects the simple flat-distribution mediation
mechanism and instead supports the interpretation that generation helps by
shaping a compact shared representation.

## Next experiments

1. Compare the width-matched raw hash with an exact sparse linear/history model.
2. Repeat the label-efficiency curve across encoder seeds at the most informative
   1%, 5%, and 25% budgets.
3. Replace the flat claim-probability mediator with a compact learned claim-set
   embedding or retrieval summary before rejecting richer mediation paths.
4. Use retrieval/candidate generation before revisiting explicit sparse changes.
5. If parameter-efficient adaptation is operationally required, retain the
   drift penalty and compare bottlenecks 8, 16, and 32 against the frozen MLP.
6. Consider discrete masked diffusion only after a sparse decoder has a
   calibrated, reproducible advantage.

## Artifacts

- Correct seed-42 matrix: `experiments/frozen_generation_probe_20260818_context_seed42/`
- Low-label seeds 43/44: `experiments/frozen_generation_probe_20260818_context_lowlabel_seeds43_44/`
- Persistence baselines: `experiments/frozen_generation_probe_20260818_context_baselines/`
- Copy-plus-residual screen: `experiments/frozen_generation_probe_20260818_copy_residual_seed42/`
- Concatenated low-label screen: `experiments/frozen_generation_probe_20260818_concat_lowlabel/`
- Final explicit event screen: `experiments/frozen_generation_probe_20260818_add_remove_events_seed42/`
- Auxiliary weights 0--0.1: `experiments/generation_aux_cost_probe_20260818/`
- Auxiliary weights 0.3--1: `experiments/generation_aux_cost_probe_20260818_high_weights/`
- Auxiliary weights 3--10: `experiments/generation_aux_cost_probe_20260818_turnover/`
- Three-encoder replication: `experiments/generation_aux_cost_multiencoder_20260818/`
- Unconstrained adapter: `experiments/generation_aux_cost_adapter_20260818_selected_e20/`
- Drift-constrained adapter: `experiments/generation_aux_cost_adapter_20260818_l2w10/`
- Semantic/modality/raw controls: `experiments/generation_aux_control_ablation_20260818/`
- Label-free consistency controls: `experiments/generation_aux_regularization_control_20260818/`
- High-weight consistency controls: `experiments/generation_aux_regularization_control_20260818_high_weights/`
- Reference hidden geometry: `experiments/generation_aux_regularization_control_20260818_geometry_reference/`
- SigReg pilot: `experiments/generation_aux_sigreg_control_20260818_pilot/`
- SigReg weight extension: `experiments/generation_aux_sigreg_control_20260818_weight_extension/`
- Three-seed SigReg replication: `experiments/generation_aux_sigreg_control_20260818_replication/`
- Multi-encoder SigReg replication: `experiments/generation_aux_sigreg_multiencoder_20260818/`
- Cost-label efficiency curve: `experiments/generation_aux_label_efficiency_20260818/`
- Future-claim cost mediation: `experiments/future_claim_cost_mediation_20260818/`
- Runner: `scripts/run_frozen_generation_probe.py`
- Cost-auxiliary runner: `scripts/run_generation_aux_cost_probe.py`
- Control-ablation runner: `scripts/run_generation_aux_control_ablation.py`
- Regularization-control runner: `scripts/run_generation_aux_regularization_control.py`
- SigReg-control runner: `scripts/run_generation_aux_sigreg_control.py`
- Label-efficiency runner: `scripts/run_generation_aux_label_efficiency.py`
- Future-claim mediation runner: `scripts/run_future_claim_cost_mediation.py`
