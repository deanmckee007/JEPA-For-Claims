# Direct Cost Supervision Ablation — 2026-08-14

## Decision

For cost prediction, use the selected composable SSL encoder as initialization,
then fine-tune the encoder and a pre-SAE MLP cost head for 20 task-only epochs.
Keep the original unfine-tuned SSL checkpoint as the canonical general-purpose
representation.

This is a two-artifact conclusion:

- **General representation:** frozen `composable_level1_lejepa_any_code`
- **Cost-specialized model:** that checkpoint followed by task-only pre-SAE
  fine-tuning at encoder LR `2e-5` and head LR `1e-3`

The test split was not accessed. All comparisons use the frozen validation
split and fixed epoch schedules without validation early stopping.

## Why a standalone head was used

The legacy internal cost head normalizes its target independently in each
minibatch. That makes its output scale batch-dependent and unsuitable for a
clean representation-value comparison. This ablation instead uses a fixed
mean and standard deviation computed once from the labeled training patients.
Every condition uses the same three-layer MLP and normalized log1p-cost MSE.

## Full-label seed-42 matrix

| Condition | Direct MAE ($) | Direct RMSE ($) | WAPE | Pair acc. | Spearman | Top-decile recall | Fresh linear-probe RMSE ($) | Silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen SSL + MLP | 1526.81 | 2566.10 | 32.84% | 0.7097 | 0.5757 | 0.4232 | 2618.50 | **0.2291** |
| SSL → five-epoch fine-tune | 1525.14 | 2575.32 | 32.80% | 0.7171 | 0.5936 | 0.4577 | **2577.40** | 0.2282 |
| **SSL → 20-epoch fine-tune** | **1464.76** | **2431.76** | **31.50%** | 0.7251 | 0.6124 | 0.4734 | 2607.75 | 0.2179 |
| Joint SSL + cost from scratch | 1530.62 | 2483.85 | 32.92% | 0.7108 | 0.5764 | **0.4922** | 2770.27 | 0.1765 |
| Supervised-only from scratch | 1548.55 | 2550.82 | 33.31% | **0.7274** | **0.6161** | 0.4828 | 2724.83 | 0.1392 |

The distinction between task performance and representation value is real:

- Joint training learns a strong nonlinear cost solution but makes cost much
  less linearly recoverable from the resulting embedding.
- Supervised scratch learns excellent global ordering but produces the
  narrowest, least clustered representation.
- Long task-only fine-tuning from SSL gives the best cost metrics while
  retaining substantially more of the original representation geometry.
- Five fine-tuning epochs were insufficient; its task loss had not converged.

## Matched three-seed head/fine-tuning comparison

Seeds 42, 43, and 44 vary head initialization, minibatch order, and dropout on
one fixed seed-42 SSL checkpoint and patient split. Values are mean ± sample SD.

| Condition | MAE ($) | RMSE ($) | WAPE | Pair acc. | Spearman | Top-decile recall |
|---|---:|---:|---:|---:|---:|---:|
| Frozen SSL + MLP | 1525.62 ± 9.61 | 2511.13 ± 55.48 | 32.81 ± 0.21% | 0.7111 ± 0.0015 | 0.5794 ± 0.0035 | 0.4639 ± 0.0362 |
| **SSL → fine-tune** | **1467.98 ± 22.11** | **2452.52 ± 41.70** | **31.57 ± 0.48%** | **0.7263 ± 0.0011** | **0.6153 ± 0.0026** | **0.4786 ± 0.0048** |

Fine-tuning improves the mean by $57.64 MAE, $58.61 RMSE, 1.24 WAPE points,
1.52 pairwise-accuracy points, and 0.0358 Spearman. RMSE and MAE improve on
every paired seed. This measures downstream optimization variance, not
independent SSL-pretraining variance.

## Label efficiency

The frozen SSL MLP curve was:

| Labeled patients | Fraction | MAE ($) | RMSE ($) | Pair acc. | Top-decile recall |
|---:|---:|---:|---:|---:|---:|
| 150 | 1% | 1976.41 | 2940.72 | 0.6497 | 0.1755 |
| 750 | 5% | 1818.17 | 2777.16 | 0.6732 | 0.3480 |
| 1,500 | 10% | 1706.88 | 2769.83 | 0.6817 | 0.3386 |
| 3,760 | 25% | 1536.55 | 2556.00 | 0.7108 | 0.4639 |
| 15,050 | 100% | 1526.81 | 2566.10 | 0.7097 | 0.4232 |

At 10% labels, equal-budget end-to-end results were:

| Condition | MAE ($) | RMSE ($) | Pair acc. | Spearman | Top-decile recall | Silhouette |
|---|---:|---:|---:|---:|---:|---:|
| SSL → fine-tune | 1638.39 | 2624.07 | 0.6938 | 0.5379 | 0.4107 | **0.2302** |
| Joint SSL + cost | 1624.07 | **2588.64** | 0.6925 | 0.5330 | 0.4389 | 0.1834 |
| Supervised scratch | **1598.70** | 2664.41 | **0.6982** | **0.5465** | **0.4451** | 0.1346 |

This does not support universal low-label SSL dominance. Joint SSL helps RMSE,
scratch leads MAE and ordering, and SSL initialization mainly preserves a more
general representation. More unlabeled examples than labeled examples were not
provided in this control; that is a separate semi-supervised experiment.

## Missing-modality result

Seed-42 direct-head validation metrics:

| Model | Full RMSE ($) | Missing-CPT RMSE ($) | Missing-ICD RMSE ($) |
|---|---:|---:|---:|
| Frozen SSL + MLP | 2566.10 | 2823.32 | 2677.12 |
| SSL → fine-tune | **2431.76** | **2781.02** | **2523.29** |

The fine-tuned nonlinear head improves RMSE under both missing modalities.
However, a newly fitted full-modality linear probe transferred much worse after
fine-tuning: missing-CPT RMSE rose from $2892.75 to $3656.95 and missing-ICD
RMSE from $2893.65 to $3381.45. Cost knowledge has become more dependent on the
trained nonlinear head.

Missing-CPT ordering also remains fragile: pairwise accuracy falls from 0.7251
full to 0.5887 when CPT is absent. Missing-modality augmentation remains a
worthwhile follow-up for the cost-specialized model.

## Interpretation

The original question has a nuanced answer:

1. The frozen SSL representation has real downstream value, but it is not an
   optimal cost representation.
2. Direct supervision is necessary to organize the representation for dollars.
3. SSL initialization is still valuable: with equal supervised budgets it
   produces a much better cost/geometry tradeoff than supervised scratch.
4. Joint training is not the best default. It over-specializes the nonlinear
   head and weakens fresh linear-probe transfer.
5. The strongest current system is sequential, not purely self-supervised or
   purely supervised: **SSL pretrain → cost fine-tune**.

## Artifacts

- Main four-condition run: `experiments/cost_supervision_ablation_20260814/`
- Fine-tune seed 42: `experiments/cost_supervision_ablation_20260814_finetune20/`
- Fine-tune seeds 43/44: `experiments/cost_supervision_ablation_20260814_finetune20_seed43/`, `...seed44/`
- Frozen-head seeds 43/44: `experiments/cost_supervision_ablation_20260814_frozen_seed43/`, `...seed44/`
- Ten-percent-label matrix: `experiments/cost_supervision_ablation_20260814_label10_seed42/`
