# Neighbor summaries for cost prediction — 2026-09-05

## Protocol

Frozen seed-42 online LeVJEPA embeddings; the existing content-group inner split gives 12,077 fitting, 2,973 calibration and 3,188 validation rows with at least two claims. This is the generation-eligible cohort, so absolute errors are not directly comparable with older cost probes on the broader representation cohort. No test split was accessed.

The four main conditions share 256 exact-history columns selected on fitting histories. JEPA adds 256 embedding dimensions. Neighbors add 144 features: for each of raw TF-IDF and JEPA cosine retrieval, eight cost statistics and 64 signed CPT/ICD vote bins from 16 neighbors. Costs include weighted log mean/dispersion, unweighted quantiles, weighted dollar mean, maximum and effective neighbor count. The vote bins contain only next-code labels from fitting reference rows; the query’s future codes are never features.

Five-fold cross-fitting excludes every query’s entire content-ID group from its training reference bank. Calibration and validation use only labeled fitting references. All reference labels, including code votes, are restricted to the same sampled fitting budget. Random content-group sampling does not consult unsampled costs. Calibration labels are separately restricted to the stated fraction and counted in the artifacts. Training banks contain about 80% of available fitting labels, versus 100% for evaluation; this finite-bank distribution shift is a limitation.

Each condition selects 100 or 200 boosting iterations by labeled inner log-MSE. Trees use learning rate 0.05, 31 leaves, L2=1 and no internal early stopping, avoiding cross-fit label contamination of an internal stopping set. Models are not refit on calibration labels. The 1% and 5% budgets use seeds 101–103; full labels are deterministic and run once. Two additional full-label controls separate neighbor costs from code votes.

Dollar scores retain the existing log1p target capped at 10 (about $22,025). Top-5% capture is the fraction of capped validation cost among the 5% ranked highest. Content IDs are sequence hashes, not verified member identities; this is not a prospective member-disjoint study. The encoder saw the full training cohort during representation pretraining. Raw features retain uncapped within-claim codes while embeddings use token caps.

## Results

Means ± population SD across label seeds; full-label rows have one run.

| Labels | Features | MAE ($) | RMSE ($) | Log RMSE | Top-5% cost capture |
|---:|---|---:|---:|---:|---:|
| 1% | raw | 1907.02 ± 43.97 | 2952.91 ± 65.82 | 0.9174 ± 0.0145 | 7.89 ± 0.39% |
| 1% | raw_jepa | 1967.67 ± 109.79 | 2995.20 ± 116.49 | 0.9332 ± 0.0223 | 7.29 ± 0.17% |
| 1% | raw_neighbors | 1967.62 ± 134.38 | 3021.86 ± 114.29 | 0.9359 ± 0.0255 | 6.86 ± 0.77% |
| 1% | raw_jepa_neighbors | 1942.91 ± 125.99 | 2981.04 ± 119.01 | 0.9261 ± 0.0251 | 7.50 ± 0.17% |
| 5% | raw | 1707.20 ± 47.06 | 2729.49 ± 76.97 | 0.8538 ± 0.0070 | 8.65 ± 0.25% |
| 5% | raw_jepa | 1732.39 ± 58.95 | 2732.67 ± 85.62 | 0.8657 ± 0.0083 | 8.48 ± 0.35% |
| 5% | raw_neighbors | 1717.08 ± 19.23 | 2741.41 ± 32.90 | 0.8616 ± 0.0043 | 8.50 ± 0.08% |
| 5% | raw_jepa_neighbors | 1717.24 ± 32.59 | 2722.95 ± 59.28 | 0.8571 ± 0.0056 | 8.41 ± 0.34% |
| 100% | raw | 1460.29 | 2475.47 | 0.7911 | 9.74% |
| 100% | raw_jepa | 1480.05 | 2473.97 | 0.7977 | 10.03% |
| 100% | raw_neighbors | 1474.28 | 2469.76 | 0.7945 | 9.87% |
| 100% | raw_jepa_neighbors | 1472.60 | 2461.13 | 0.7948 | 9.89% |
| 100% | raw_cost_only | 1458.53 | 2456.19 | 0.7895 | 9.65% |
| 100% | raw_votes_only | 1489.80 | 2500.58 | 0.7949 | 9.62% |

## Paired changes

Negative dollar deltas indicate improvement over the matching baseline.

| Labels | Comparison | MAE delta ($) | RMSE delta ($) | MAE wins |
|---:|---|---:|---:|---:|
| 1% | raw_neighbors − raw | 60.60 | 68.95 | 1/3 |
| 1% | raw_jepa_neighbors − raw_jepa | -24.77 | -14.16 | 2/3 |
| 5% | raw_neighbors − raw | 9.88 | 11.91 | 1/3 |
| 5% | raw_jepa_neighbors − raw_jepa | -15.15 | -9.72 | 2/3 |
| 100% | raw_neighbors − raw | 13.99 | -5.71 | 0/1 |
| 100% | raw_jepa_neighbors − raw_jepa | -7.45 | -12.84 | 1/1 |

## Interpretation

The combined neighbor summaries do not consistently beat raw history: mean MAE worsens by $60.60 at 1% labels, $9.88 at 5%, and $13.99 at full labels. They improve the weaker raw-plus-JEPA baseline modestly, which is insufficient evidence to replace raw history.

At full labels, cost-only neighbor statistics provide the best observed MAE ($1,458.53) and RMSE ($2,456.19), improvements of only $1.76 and $19.28 over raw history. Code votes alone worsen MAE by $29.51 and RMSE by $25.11. The code-generation ranking improvements therefore have not transferred into a clear cost-prediction benefit through these summaries. Cost-only top-5% capture also falls slightly, from 9.74% to 9.65%.

This is a pilot with one encoder seed and repeatedly explored validation data. Keep raw history as the baseline. A useful next cost-focused experiment would compare dollar-oriented training losses against the current log-MSE objective under the same split and label budget, retaining cost-only retrieval as a small ablation. The endpoint cap should remain explicit; uncapped cost prediction would require a separately defined target contract.

## Validation and artifacts

245 repository tests passed (five existing Lightning logging warnings). A direct perturbation test verifies that changing every cost in one held-out fold leaves that fold’s own features unchanged. `git diff --check` passed.

- Runner: `scripts/run_neighbor_cost_probe.py`
- Feature implementation: `jepa_utils/neighbor_cost.py`
- Results, exact label/fold assignments and predictions: `experiments/neighbor_cost_20260905/`
- Prior generation results: [mechanism probes](mechanism_probes_20260905.md)
