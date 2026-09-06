# New-code retrieval and calibration pilot

The nine-run replication established a CPT set-quality gain but left ICD sets
tied with copy-only. This next pilot isolates candidate coverage and the
calibration objective using encoder 42 / decoder 201. It is a single-pair
experiment, not a replicated promotion.

## Protocol

- Reuse the saved candidate and flat heads from the replication; the original
  candidate's replayed ICD AP matches to better than 1e-7.
- Keep the same 12,077 fitting / 2,973 calibration / 3,188 validation rows.
  Only the fitting subset supplies retrieval targets and decoder labels.
- Compare total micro-F1 calibration with a fixed objective equal to
  `0.5 * total micro F1 + 0.5 * known-new-code micro F1`. Both use the same
  predeclared threshold and cardinality grids, selected on the inner split.
- Compare 16 neighbors / 128 candidates with 64 neighbors / 256 candidates.
  Train the expanded decoder only if mean CPT/ICD new-code recall improves by
  at least .05 on the inner calibration split. The observed inner gain was
  .2012, so the gate passed before outer coverage was scored.
- Train the expanded token-conditioned decoder for 20 epochs at seed 201,
  matching the original head's hidden width and optimizer settings. Retain
  copy-only, flat, and both candidate-filter-only controls.
- Freeze these choices before validation scoring. No test split was accessed.

## Retrieval coverage

These recalls exclude PAD/UNK and count targets absent from the last observed
claim, rather than copied tokens.

| Budget | Validation new CPT recall | Validation new ICD recall |
|---|---:|---:|
| 16 neighbors / 128 candidates | .3992 | .3898 |
| 64 neighbors / 256 candidates | .5818 | .6392 |

The expansion supplies substantially more new codes. That alone does not
establish that the decoder can distinguish the right additions.

## Decoding tradeoff

| Decoder / calibration | CPT total F1 | CPT new precision / recall | ICD total F1 | ICD new precision / recall |
|---|---:|---:|---:|---:|
| Copy only | .5105 | .0000 / .0000 | .7145 | .0000 / .0000 |
| Candidate 16 / total F1 | .5605 | .4171 / .1241 | .7129 | .6364 / .0061 |
| Candidate 16 / additions-aware | .5478 | .3003 / .1968 | .6883 | .1824 / .0856 |
| Candidate 64 / total F1 | .5538 | .4000 / .1269 | .7122 | .6522 / .0066 |
| Candidate 64 / additions-aware | .5405 | .2941 / .2094 | .6897 | .1849 / .0764 |
| Flat / additions-aware | .5299 | .2465 / .1509 | .6841 | .1637 / .0737 |
| Flat + filter 16 / additions-aware | .5274 | .2641 / .1650 | .6879 | .1747 / .0698 |
| Flat + filter 64 / additions-aware | .5309 | .2497 / .1495 | .6845 | .1652 / .0737 |

The additions-aware objective exposes the tradeoff that total F1 hid: the
original candidate decoder recovers more new ICD codes (0.61% → 8.56%), but only
18.24% of its emitted new diagnoses are correct, and total ICD F1 falls by
2.46 percentage points. The expanded decoder does not resolve that tradeoff.

Original-score CPT AP improves with the expanded bank (.4755 → .4979), while
ICD AP worsens (.6219 → .6065). Candidate coverage, ranking, and set quality have
different optima in this pilot.

## Decision

Keep the replicated 16/128 decoder as the current CPT set-quality candidate.
Neither additions-aware calibration nor the expanded bank is a demonstrated
general generation upgrade. Retrieval now exposes more valid new-code targets,
but the scoring/training of additions cannot use that coverage reliably.

A focused next experiment would train a candidate-conditioned additions scorer
with separate supervision for new codes and copied-code retention, keeping the
same inner-split controls. It should report addition precision/recall and total
set quality together. That experiment has not been run here.

The subsequent [mechanism pilots](mechanism_probes_20260905.md) implemented
separate retention/addition heads. They slightly improved set F1 but sharply
reduced raw ranking quality; simpler raw/hybrid neighbor-voting controls were
the stronger new result.

## Artifacts and verification

Artifacts: `experiments/new_code_probe_20260905/protocol.json`,
`inner_retrieval_gate.json`, `summary.json`, and the local `candidate64.pt`.
The summary contains all calibration choices and all six controls/conditions
under both objectives. The original saved heads remain unchanged.

Run `scripts/run_new_code_probe.py` with the seed-42 checkpoint, data path,
frozen data contract, `--replication-dir experiments/retrieval_replication_20260905`,
and a fresh `--output-dir`. Defaults reproduce this seed-42 / decoder-201 pilot.

236 tests passed, including a synthetic test that exposes the total-F1 versus
new-code-recall tradeoff and the validation-label isolation test. Five existing
Lightning warnings remain in diffusion tests. `git diff --check` passes.
