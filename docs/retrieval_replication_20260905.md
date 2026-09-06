# Retrieval replication and calibration — September 5

Completed 9 of 9 planned encoder/decoder pairs.

Three existing online LeVJEPA encoders (42–44) are crossed with decoder seeds 201–203.
A fixed content-ID-grouped 80/20 split inside training reserves 20% for calibration.
Only the 80% fitting subset supplies decoder labels and retrieved next claims.
The frozen encoders were already pretrained on the original training cohort.
Calibration selects copy/add logit thresholds and cardinality scaling by inner micro F1.
The search grid was fixed before scoring validation; test was never accessed.

Fit / calibration / validation rows: 12077 / 2973 / 3188.
The original pilot fit all training rows; these scores use less decoder supervision.

## Decision

The ranking gain replicates, and calibrated CPT generation improves over all
three controls in every seed pair. Keep the candidate decoder as the leading
CPT experiment and an ICD ranking component. It is not a demonstrated upgrade
over copy-only ICD claim sets: mean calibrated ICD F1 is .7140 versus .7145,
and only five of nine paired runs improve on copy-only.

The next bottleneck has two parts. Retrieval misses roughly 60% of genuinely
new codes, and total-F1 calibration suppresses most ICD additions even when
they are retrieved. Before enlarging the decoder, test an additions-specific
selection objective on an inner split, alongside better candidate recall.
These follow-up experiments have not been run here.

Subsequent [new-code pilot](new_code_probe_20260905.md): a larger bank raises
new-code coverage, but does not improve decoded sets. Additions-aware calibration
raises new-ICD recall at a substantial precision and total-F1 cost. Separate
additions scoring remains an untested next mechanism.

## Ranking and decoded sets

AP uses the original model scores. Calibrated F1 uses decoded sets and is reported separately.

| Model | CPT AP | ICD AP | CPT F1 before → after calibration | ICD F1 before → after calibration |
|---|---:|---:|---:|---:|
| Copy only | 0.2610 | 0.5111 | 0.5105 → 0.5105 | 0.7145 → 0.7145 |
| Flat residual | 0.4675 | 0.5262 | 0.5057 → 0.5339 | 0.5848 → 0.7083 |
| Flat + candidate filter | 0.4622 | 0.5295 | 0.5073 → 0.5382 | 0.5875 → 0.7083 |
| Candidate decoder | 0.4780 | 0.6221 | 0.5181 → 0.5637 | 0.6367 → 0.7140 |

## Paired comparisons

Each cell reports candidate mean delta and the number of positive deltas across matched runs.

| Comparator | CPT AP delta / wins | ICD AP delta / wins | Calibrated CPT F1 delta / wins | Calibrated ICD F1 delta / wins |
|---|---:|---:|---:|---:|
| Copy only | +0.2169 / 9/9 | +0.1110 / 9/9 | +0.0532 / 9/9 | -0.0005 / 5/9 |
| Flat residual | +0.0105 / 9/9 | +0.0959 / 9/9 | +0.0297 / 9/9 | +0.0057 / 9/9 |
| Flat + candidate filter | +0.0158 / 9/9 | +0.0926 / 9/9 | +0.0255 / 9/9 | +0.0056 / 9/9 |

## New-code coverage

Known tokens only (IDs > 1): PAD and UNK are excluded. Persistent targets already occur
in the last claim; new targets do not. Coverage is independent of the decoder seed.

| Encoder seed | CPT overall / persistent / new recall | ICD overall / persistent / new recall |
|---|---:|---:|
| 42 | 0.6980 / 1.0000 / 0.3992 | 0.8206 / 1.0000 / 0.3898 |
| 43 | 0.7029 / 1.0000 / 0.4090 | 0.8210 / 1.0000 / 0.3911 |
| 44 | 0.7061 / 1.0000 / 0.4154 | 0.8248 / 1.0000 / 0.4043 |

Calibration maximizes total micro F1; it may prefer persistence over emitting new codes.

| Model | Calibrated CPT new-code precision / recall | Calibrated ICD new-code precision / recall |
|---|---:|---:|
| Copy only | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| Flat residual | 0.2625 / 0.1109 | 0.4042 / 0.0055 |
| Flat + candidate filter | 0.2813 / 0.1241 | 0.4071 / 0.0053 |
| Candidate decoder | 0.4112 / 0.1320 | 0.3667 / 0.0024 |

## Interpretation limits and artifacts

These are validation results on shared patients, not nine independent population samples.
Crossed-run SD is descriptive. Copy-only is deterministic and is repeated for pairing,
not counted as independent replication. Candidate and flat heads share budgets, data and
hidden width but are not parameter-count matched. Grouping uses frozen content IDs because
stable member IDs are not present. No test-set or production promotion is implied.

The run directory contains protocol.json, inner_split.json, summary.json and one directory
per seed pair with the calibrated settings, metrics, loss histories, and local trained heads.
Reproduce with scripts/run_retrieval_replication.py using the three LeVJEPA checkpoints,
--decoder-seeds 201 202 203, the frozen data contract, and a new output directory.
Render this report with scripts/summarize_retrieval_replication.py SUMMARY_JSON --output REPORT_MD.

Verification: 235 tests passed, including an explicit check that changing outer
validation labels cannot alter calibration settings. Five existing Lightning
warnings remain in diffusion tests. All nine real-data runs completed; no
held-out test data was accessed.
