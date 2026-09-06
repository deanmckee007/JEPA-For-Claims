# Cost training objectives — 2026-09-05

## Protocol

Compare log squared error, dollar squared error and dollar absolute error using the same histogram gradient-boosted trees. Every objective is independently evaluated with three inner selection criteria: log MSE, dollar MAE and dollar MSE. Each criterion selects between 100 and 200 iterations, so changing the training target is not confounded with a unique selection rule. No validation labels select iterations.

The [neighbor-cost pilot](neighbor_cost_20260905.md) defines the unchanged data protocol: 12,077 fitting, 2,973 calibration and 3,188 validation rows; 1%, 5% and full labels; three random group-sampling seeds at low budgets and one deterministic full-label run. Exact labeled positions, content IDs and five-fold assignments are checked against that pilot. Raw history is compared with raw plus 16 cost-only neighbor statistics from raw and frozen seed-42 online JEPA retrieval. Costs are cross-fitted and reference banks respect the label budget. Code votes and direct JEPA features are omitted from these models.

Targets retain the existing log1p cap at 10, approximately $22,025. Dollar predictions are clipped to this endpoint range and converted to log1p for shared scoring. Dollar training uses unscaled dollars; fixed L2 and tree constraints are retained. This is an objective pilot, not a separate hyperparameter search for each loss. The representation was pretrained on the broader training cohort; content IDs do not establish verified member separation. No held-out test split was accessed.

## Selected on inner dollar MAE

Low-budget cells are means across three label seeds.

| Labels | Features | Training loss | MAE ($) | RMSE ($) | Top-5% capture | Total-cost bias |
|---:|---|---|---:|---:|---:|---:|
| 1% | raw | log_squared | 1907.02 | 2952.91 | 7.89% | -12.55% |
| 1% | raw | dollar_squared | 1890.25 | 2839.81 | 7.39% | +0.26% |
| 1% | raw | dollar_absolute | 1695.18 | 2727.26 | 8.26% | -6.82% |
| 1% | raw_cost_only | log_squared | 1971.12 | 3026.66 | 7.11% | -15.19% |
| 1% | raw_cost_only | dollar_squared | 1899.37 | 2871.59 | 7.29% | +1.04% |
| 1% | raw_cost_only | dollar_absolute | 1701.34 | 2760.98 | 7.90% | -8.85% |
| 5% | raw | log_squared | 1707.20 | 2729.49 | 8.65% | -12.63% |
| 5% | raw | dollar_squared | 1688.32 | 2639.12 | 8.51% | +1.34% |
| 5% | raw | dollar_absolute | 1520.92 | 2526.69 | 9.09% | -5.34% |
| 5% | raw_cost_only | log_squared | 1697.20 | 2724.05 | 8.41% | -12.04% |
| 5% | raw_cost_only | dollar_squared | 1686.79 | 2619.87 | 8.56% | +2.22% |
| 5% | raw_cost_only | dollar_absolute | 1511.75 | 2524.08 | 8.93% | -5.22% |
| 100% | raw | log_squared | 1471.08 | 2469.92 | 9.62% | -12.89% |
| 100% | raw | dollar_squared | 1468.08 | 2390.96 | 9.75% | -0.70% |
| 100% | raw | dollar_absolute | 1381.45 | 2402.99 | 9.69% | -6.10% |
| 100% | raw_cost_only | log_squared | 1458.53 | 2456.19 | 9.65% | -13.22% |
| 100% | raw_cost_only | dollar_squared | 1452.21 | 2347.25 | 9.99% | -0.16% |
| 100% | raw_cost_only | dollar_absolute | 1374.70 | 2382.36 | 9.65% | -5.61% |

## Selected on inner dollar MSE

Low-budget cells are means across three label seeds.

| Labels | Features | Training loss | MAE ($) | RMSE ($) | Top-5% capture | Total-cost bias |
|---:|---|---|---:|---:|---:|---:|
| 1% | raw | log_squared | 1907.02 | 2952.91 | 7.89% | -12.55% |
| 1% | raw | dollar_squared | 1935.61 | 2887.46 | 7.32% | +0.28% |
| 1% | raw | dollar_absolute | 1695.18 | 2727.26 | 8.26% | -6.82% |
| 1% | raw_cost_only | log_squared | 1971.12 | 3026.66 | 7.11% | -15.19% |
| 1% | raw_cost_only | dollar_squared | 1899.37 | 2871.59 | 7.29% | +1.04% |
| 1% | raw_cost_only | dollar_absolute | 1693.31 | 2736.54 | 8.19% | -7.79% |
| 5% | raw | log_squared | 1707.20 | 2729.49 | 8.65% | -12.63% |
| 5% | raw | dollar_squared | 1688.32 | 2639.12 | 8.51% | +1.34% |
| 5% | raw | dollar_absolute | 1520.26 | 2522.19 | 9.08% | -5.18% |
| 5% | raw_cost_only | log_squared | 1739.01 | 2753.01 | 8.35% | -11.24% |
| 5% | raw_cost_only | dollar_squared | 1686.79 | 2619.87 | 8.56% | +2.22% |
| 5% | raw_cost_only | dollar_absolute | 1510.79 | 2516.96 | 9.03% | -5.15% |
| 100% | raw | log_squared | 1471.08 | 2469.92 | 9.62% | -12.89% |
| 100% | raw | dollar_squared | 1468.08 | 2390.96 | 9.75% | -0.70% |
| 100% | raw | dollar_absolute | 1381.45 | 2402.99 | 9.69% | -6.10% |
| 100% | raw_cost_only | log_squared | 1474.87 | 2465.53 | 9.49% | -12.57% |
| 100% | raw_cost_only | dollar_squared | 1452.21 | 2347.25 | 9.99% | -0.16% |
| 100% | raw_cost_only | dollar_absolute | 1374.70 | 2382.36 | 9.65% | -5.61% |

## Selected on inner log MSE

Low-budget cells are means across three label seeds.

| Labels | Features | Training loss | MAE ($) | RMSE ($) | Top-5% capture | Total-cost bias |
|---:|---|---|---:|---:|---:|---:|
| 1% | raw | log_squared | 1907.02 | 2952.91 | 7.89% | -12.55% |
| 1% | raw | dollar_squared | 1935.61 | 2887.46 | 7.32% | +0.28% |
| 1% | raw | dollar_absolute | 1693.95 | 2721.09 | 8.30% | -6.69% |
| 1% | raw_cost_only | log_squared | 1971.12 | 3026.66 | 7.11% | -15.19% |
| 1% | raw_cost_only | dollar_squared | 1899.37 | 2871.59 | 7.29% | +1.04% |
| 1% | raw_cost_only | dollar_absolute | 1701.34 | 2760.98 | 7.90% | -8.85% |
| 5% | raw | log_squared | 1707.20 | 2729.49 | 8.65% | -12.63% |
| 5% | raw | dollar_squared | 1688.32 | 2639.12 | 8.51% | +1.34% |
| 5% | raw | dollar_absolute | 1519.82 | 2529.34 | 9.01% | -5.49% |
| 5% | raw_cost_only | log_squared | 1697.20 | 2724.05 | 8.41% | -12.04% |
| 5% | raw_cost_only | dollar_squared | 1686.79 | 2619.87 | 8.56% | +2.22% |
| 5% | raw_cost_only | dollar_absolute | 1510.79 | 2516.96 | 9.03% | -5.15% |
| 100% | raw | log_squared | 1460.29 | 2475.47 | 9.74% | -13.77% |
| 100% | raw | dollar_squared | 1468.08 | 2390.96 | 9.75% | -0.70% |
| 100% | raw | dollar_absolute | 1381.45 | 2402.99 | 9.69% | -6.10% |
| 100% | raw_cost_only | log_squared | 1458.53 | 2456.19 | 9.65% | -13.22% |
| 100% | raw_cost_only | dollar_squared | 1457.77 | 2357.59 | 9.79% | -0.40% |
| 100% | raw_cost_only | dollar_absolute | 1374.70 | 2382.36 | 9.65% | -5.61% |

## Paired objective changes

All pairs use the same inner dollar-MAE selection rule. Negative deltas favor the dollar loss.

| Labels | Features | Loss versus log squared | MAE delta ($) | RMSE delta ($) | MAE wins |
|---:|---|---|---:|---:|---:|
| 1% | raw | dollar_squared | -16.78 | -113.11 | 2/3 |
| 1% | raw | dollar_absolute | -211.84 | -225.65 | 3/3 |
| 1% | raw_cost_only | dollar_squared | -71.75 | -155.07 | 2/3 |
| 1% | raw_cost_only | dollar_absolute | -269.77 | -265.68 | 3/3 |
| 5% | raw | dollar_squared | -18.89 | -90.37 | 2/3 |
| 5% | raw | dollar_absolute | -186.28 | -202.80 | 3/3 |
| 5% | raw_cost_only | dollar_squared | -10.41 | -104.18 | 1/3 |
| 5% | raw_cost_only | dollar_absolute | -185.44 | -199.98 | 3/3 |
| 100% | raw | dollar_squared | -3.00 | -78.96 | 1/1 |
| 100% | raw | dollar_absolute | -89.63 | -66.93 | 1/1 |
| 100% | raw_cost_only | dollar_squared | -6.32 | -108.94 | 1/1 |
| 100% | raw_cost_only | dollar_absolute | -83.83 | -73.83 | 1/1 |

## Interpretation

Dollar absolute-error training provides the clearest individual-cost improvement. With the same inner dollar-MAE selection criterion, raw-history MAE falls from $1,907.02 to $1,695.18 at 1% labels, $1,707.20 to $1,520.92 at 5%, and $1,471.08 to $1,381.45 at full labels. Every low-budget seed improves. RMSE also improves against the matching log-loss model at each budget.

At full labels, adding cost-only neighbors to dollar absolute-error training gives MAE $1,374.70 and RMSE $2,382.36. This is a modest further improvement over raw dollar absolute-error training, not evidence that the representation itself is necessary: the neighbor features combine raw and JEPA retrieval and need a raw-only ablation.

Dollar squared-error training addresses a different objective: with full-label neighbor features, RMSE is $2,347.25 and aggregate cost bias is -0.16%, versus $2,382.36 and -5.61% for dollar absolute error under the same inner dollar-MAE selection. Conditional median predictions can help individual MAE while underpredicting total spending; the loss should follow the intended use.

The raw full-label log baseline selected on dollar MAE is $1,471.08, whereas the previous pilot selected on log MSE and obtained $1,460.29. Both controls are shown above; the objective improvement persists without choosing whichever baseline looks weaker. Keep dollar absolute-error training as the leading MAE candidate and dollar squared error as the spending/RMSE candidate. These remain validation experiments with one frozen encoder seed, capped costs and repeated validation exploration; no test-set promotion or production default was changed.

## Artifacts and checks

Results and prediction arrays: `experiments/cost_objectives_20260905/`. Runner: `scripts/run_neighbor_cost_probe.py --objective-sweep`. The objective-selection and cross-fit leakage tests passed (2 tests). All seven original raw baseline prediction arrays reproduced within absolute tolerance 1e-12. `git diff --check` passed.
