# Retrieval, decoder and order mechanism pilots

These four pilots reuse encoder 42 / decoder seed 201, the frozen data contract,
and the same 12,077 fitting / 2,973 calibration / 3,188 validation rows as the
retrieval replication. They are single-encoder-seed experiments. No held-out
test data was accessed and no configuration was promoted.

## Comparisons

1. **Neighbor votes without a learned decoder.** Weight training next-claim
   memberships by cosine similarity. Mix these probabilities with the last
   observed claim, selecting a copy weight from `[0, .25, .5, .75, 1]` on mean
   inner CPT/ICD calibrated micro F1. The selected weight is .25 for dense,
   raw and hybrid retrieval. A fixed logit shift of +4 supplies a useful range
   for the existing threshold grid without changing ranking.
2. **Diverse retrieval.** Compare 16 embedding neighbors with eight embedding
   plus eight raw-history neighbors. Raw features are exact recency-weighted
   CPT/ICD/TTNC counts and last-claim indicators, using TF-IDF fitted only on
   the inner fitting subset. Hybrid candidates alternate the two ranked code
   lists after preserving the last claim. Each condition has 128 slots.
3. **Retention/addition heads.** Use separate token-conditioned heads with a
   shared trunk. Normalize positive and negative loss separately within each
   task, then average the retention and addition losses. Retrieved non-target
   codes are the addition negatives. Hidden width, epochs and seed match the
   original decoder; parameter counts do not, because the new head has extra
   token embeddings.
4. **Older-history order.** Shuffle complete CPT/ICD/TTNC claim triples before
   the protected latest context claim. Preserve padding and every held-out
   future claim. Use three fixed permutation seeds. Keep retrieval candidates
   and inner-calibrated thresholds fixed to the clean histories.

Raw-history retrieval sees uncapped within-claim token lists, while the encoder
uses the existing token caps. This is the repository's strong exact-history
baseline, not an assertion that the two representations contain identical inputs.

## Main validation results

AP uses original scores; set F1 uses total-F1 calibration selected on the inner
split. Scores from the replayed dense decoder match the previous saved-head
pilot to better than 1e-7 for ICD AP.

| Condition | CPT AP | ICD AP | Calibrated CPT F1 | Calibrated ICD F1 |
|---|---:|---:|---:|---:|
| Learned dense candidate decoder | .4755 | .6219 | .5605 | .7129 |
| Dense neighbor votes | .5030 | .6087 | .5569 | .7138 |
| Raw-history neighbor votes | .5088 | .6507 | .5763 | .7148 |
| Hybrid neighbor votes | .5225 | .6619 | .5758 | .7128 |
| Learned hybrid candidate decoder | .4785 | .6271 | .5606 | .7121 |
| Separate retention/addition heads | .3429 | .3396 | .5666 | .7188 |

The strongest new baseline is simple voting: raw and hybrid votes beat the
learned candidate decoder on CPT set F1, and hybrid votes also lead ICD ranking.
This changes the next comparison to make; it does not establish population-level
superiority from one encoder and one frozen validation split.

Separate heads improve total set F1 slightly but sharply reduce raw-score AP.
Scores from the two heads may be poorly aligned across retention/addition groups;
this experiment does not distinguish that explanation from worse ranking within
groups. The set-F1 benefit requires task-specific calibration and is not a
general improvement in ranking.

## Novelty and candidate diversity

New-code metrics exclude PAD/UNK and exclude tokens already in the last claim.

| Retrieval | New CPT candidate recall | New ICD candidate recall |
|---|---:|---:|
| Dense | .3992 | .3898 |
| Raw | .3794 | .4052 |
| Hybrid | .4147 | .4280 |

| Condition, additions-aware calibration | CPT new precision / recall | ICD new precision / recall |
|---|---:|---:|
| Dense learned decoder | .3003 / .1968 | .1824 / .0856 |
| Hybrid learned decoder | .3156 / .1883 | .1841 / .0694 |
| Split heads | .3498 / .1749 | .1357 / .1076 |
| Dense votes | .3584 / .1481 | .2094 / .0448 |
| Raw votes | .4360 / .1368 | .2016 / .1335 |
| Hybrid votes | .2872 / .2327 | .1777 / .1106 |

More diverse candidates modestly improve new-code coverage but do not improve
the learned decoder's new-code recall. Raw votes offer a stronger ICD-additions
precision/recall pair than the learned alternatives in this pilot. Split heads
increase ICD addition recall at a precision cost.

The common 128-slot cap does not guarantee equal non-padding candidate counts.
An additional control in `count_matched/` trims each hybrid list to the dense
list's actual size, backfilling from dense if the hybrid list is shorter. It
therefore isolates source composition more tightly than the cap alone.

The cap was loose in practice: dense lists averaged 7.01 CPT / 24.22 ICD codes;
the initial hybrid lists averaged only 5.91 / 18.79. Thus the initial hybrid
coverage gain was not caused by proposing more codes. After matching the dense
count exactly, hybrid new-code recall is .4295 CPT / .4649 ICD, with AP .4796 /
.6228 and calibrated F1 .5650 / .7160. This modestly improves both set F1 metrics
over the dense learned baseline, but still trails raw votes on CPT F1.
The backfill can draw extra code proposals from the dense list; this control
matches candidate counts, not the number of unique reference patients consulted.

## Chronological-order diagnostic

Mean absolute inner CPT/ICD AP change over the three permutations was .00148,
below the predeclared .01 gate for an order-invariant pretraining control.

| Validation input | CPT AP | ICD AP | Calibrated CPT F1 | Calibrated ICD F1 |
|---|---:|---:|---:|---:|
| Original order | .4755 | .6219 | .5605 | .7129 |
| Shuffle 401 | .4790 | .6209 | .5603 | .7118 |
| Shuffle 402 | .4762 | .6228 | .5602 | .7116 |
| Shuffle 403 | .4751 | .6218 | .5579 | .7113 |

This checkpoint/readout is largely insensitive to older context order when the
latest event is preserved. That does not prove chronology is useless for other
endpoints or encoders. The matched pretraining run was not triggered. The opt-in
`rnn_type=deepsets` control is implemented and permutation-tested: it applies
shared per-claim MLP layers without position embeddings and uses symmetric
pooling. It is not a trained or promoted checkpoint from this pilot.

## Artifacts and reproduction

`experiments/mechanism_probes_20260905/summary.json` contains both calibration
objectives, all conditions, inner vote-weight selection, candidate coverage,
and permutation results. Local trained heads and loss histories are alongside it.

Run `scripts/run_mechanism_probes.py` with the seed-42 LeVJEPA checkpoint,
data path, frozen contract, `--replication-dir experiments/retrieval_replication_20260905`,
and a fresh output directory. Run `scripts/run_count_matched_retrieval.py` with
the same inputs for the stricter diversity control. Neither script accesses test.

Verification: 244 tests passed. Tests cover weighted unique-token voting,
candidate budgets and backfill, train self-neighbor exclusion, disjoint head
gradients, protected future/latest claims, and permutation invariance. Five
existing Lightning warnings remain in diffusion tests. `git diff --check` passes.

## Decision

Prioritize replication of the raw and hybrid voting baselines before increasing
decoder complexity. Retain the stricter diverse-retrieval control as a modest
positive result. Split heads have a ranking/set-quality tradeoff, and the older
order diagnostic did not justify another pretraining run under its declared gate.
