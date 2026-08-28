# Label-free consistency regularization control

This experiment replaces next-claim labels with a label-free consistency loss
over the same full-cohort feature batches. Two independently dropped-out views
of the shared trunk receive clean and Gaussian-perturbed inputs; their hidden
states are matched. Cost labels remain restricted to the same 10% subsets.

| Condition | Pretrained MAE ($) | Raw-history MAE ($) |
|---|---:|---:|
| Cost only | 1,945.55 | 2,378.05 |
| Consistency 1 | 1,978.54 | 2,381.53 |
| Consistency 10 | 1,951.87 | 2,190.25 |
| Consistency 100 | 1,799.69 | 2,006.01 |
| Consistency 300 | 1,757.65 | 1,931.51 |
| Consistency 1000 | 1,811.65 | **1,839.58** |
| Shuffled claim labels | 1,760.21 | 2,219.77 |
| Aligned claim labels | **1,721.37** | 2,183.68 |

Consistency weight 300 nearly matches shuffled-label cost performance on the
pretrained representation, but it uses a qualitatively different and
undesirable solution. A seed-42 reference geometry audit found shared-hidden
feature standard deviations of 0.446 for cost-only, 1.129 for shuffled labels,
and 1.203 for aligned labels. Pooled consistency values were only 0.0225 at
weight 300 and 0.0119 at weight 1000. Mean hidden norms similarly fell to 0.525
and 0.221, versus 8.10 for cost-only and 23--25 for supervised auxiliary tasks.

The raw-history improvement at weight 1000 is likewise accompanied by hidden
collapse. Therefore label-free smoothness can regularize a cost predictor, but
it does not reproduce the non-collapsed shared representation learned from
claim supervision. It is rejected as a replacement for the semantic auxiliary
task.

Artifacts:

- Weights 1--100: this directory's `summary.json`.
- Weights 300--1000: `../generation_aux_regularization_control_20260818_high_weights/summary.json`.
- Reference geometry: `../generation_aux_regularization_control_20260818_geometry_reference/summary.json`.
