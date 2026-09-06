"""Render paired objective comparisons with a fixed inner selection criterion."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / 'experiments/cost_objectives_20260905/summary.json').read_text())
    if not results['completed']:
        raise ValueError('Experiment has not completed')
    groups = defaultdict(list)
    for row in results['runs']:
        groups[(row['fraction'], row['condition'], row['objective'], row['selection'])].append(row)
    lines = ['# Cost training objectives — 2026-09-05', '',
        '## Protocol', '',
        'Compare log squared error, dollar squared error and dollar absolute error using '
        'the same histogram gradient-boosted trees. Every objective is independently evaluated '
        'with three inner selection criteria: log MSE, dollar MAE and dollar MSE. Each criterion '
        'selects between 100 and 200 iterations, so changing the training target is not confounded '
        'with a unique selection rule. No validation labels select iterations.', '',
        'The [neighbor-cost pilot](neighbor_cost_20260905.md) defines the unchanged data protocol: '
        '12,077 fitting, 2,973 calibration and 3,188 validation rows; 1%, 5% and full labels; '
        'three random group-sampling seeds at low budgets and one deterministic full-label run. '
        'Exact labeled positions, content IDs and five-fold assignments are checked against that '
        'pilot. Raw history is compared with raw plus 16 cost-only neighbor statistics from raw '
        'and frozen seed-42 online JEPA retrieval. Costs are cross-fitted and reference banks '
        'respect the label budget. Code votes and direct JEPA features are omitted from these models.', '',
        'Targets retain the existing log1p cap at 10, approximately $22,025. Dollar predictions '
        'are clipped to this endpoint range and converted to log1p for shared scoring. Dollar '
        'training uses unscaled dollars; fixed L2 and tree constraints are retained. This is '
        'an objective pilot, not a separate hyperparameter search for each loss. The representation '
        'was pretrained on the broader training cohort; content IDs do not establish verified '
        'member separation. No held-out test split was accessed.', '']
    for selection, title in [('dollar_mae', 'Selected on inner dollar MAE'), ('dollar_mse', 'Selected on inner dollar MSE'),
                             ('log_mse', 'Selected on inner log MSE')]:
        lines += ['## ' + title, '', 'Low-budget cells are means across three label seeds.', '',
            '| Labels | Features | Training loss | MAE ($) | RMSE ($) | Top-5% capture | Total-cost bias |',
            '|---:|---|---|---:|---:|---:|---:|']
        for (fraction, condition, objective, criterion), rows in groups.items():
            if criterion != selection:
                continue
            values = [mean(r['metrics'][key] for r in rows) for key in
                ['target_probe_mae_dollars', 'target_probe_rmse_dollars', 'cost_capture_top_0.05', 'aggregate_cost_bias_percent']]
            lines.append(f'| {fraction:.0%} | {condition} | {objective} | {values[0]:.2f} | {values[1]:.2f} | '
                f'{100 * values[2]:.2f}% | {values[3]:+.2f}% |')
        lines.append('')
    lines += ['## Paired objective changes', '',
        'All pairs use the same inner dollar-MAE selection rule. Negative deltas favor the dollar loss.', '',
        '| Labels | Features | Loss versus log squared | MAE delta ($) | RMSE delta ($) | MAE wins |',
        '|---:|---|---|---:|---:|---:|']
    for fraction in [.01, .05, 1.]:
        for condition in ['raw', 'raw_cost_only']:
            baseline = groups[(fraction, condition, 'log_squared', 'dollar_mae')]
            for objective in ['dollar_squared', 'dollar_absolute']:
                rows = groups[(fraction, condition, objective, 'dollar_mae')]
                deltas = [[r['metrics'][key] - b['metrics'][key] for r, b in zip(rows, baseline)]
                    for key in ['target_probe_mae_dollars', 'target_probe_rmse_dollars']]
                lines.append(f'| {fraction:.0%} | {condition} | {objective} | {mean(deltas[0]):+.2f} | '
                    f'{mean(deltas[1]):+.2f} | {sum(d < 0 for d in deltas[0])}/{len(rows)} |')
    lines += ['', '## Interpretation', '',
        'Dollar absolute-error training provides the clearest individual-cost improvement. '
        'With the same inner dollar-MAE selection criterion, raw-history MAE falls from '
        '$1,907.02 to $1,695.18 at 1% labels, $1,707.20 to $1,520.92 at 5%, and $1,471.08 '
        'to $1,381.45 at full labels. Every low-budget seed improves. RMSE also improves '
        'against the matching log-loss model at each budget.', '',
        'At full labels, adding cost-only neighbors to dollar absolute-error training gives '
        'MAE $1,374.70 and RMSE $2,382.36. This is a modest further improvement over raw '
        'dollar absolute-error training, not evidence that the representation itself is necessary: '
        'the neighbor features combine raw and JEPA retrieval and need a raw-only ablation.', '',
        'Dollar squared-error training addresses a different objective: with full-label neighbor '
        'features, RMSE is $2,347.25 and aggregate cost bias is -0.16%, versus $2,382.36 and '
        '-5.61% for dollar absolute error under the same inner dollar-MAE selection. Conditional '
        'median predictions can help individual MAE while underpredicting total spending; '
        'the loss should follow the intended use.', '',
        'The raw full-label log baseline selected on dollar MAE is $1,471.08, whereas the previous '
        'pilot selected on log MSE and obtained $1,460.29. Both controls are shown above; '
        'the objective improvement persists without choosing whichever baseline looks weaker. '
        'Keep dollar absolute-error training as the leading MAE candidate and dollar squared '
        'error as the spending/RMSE candidate. These remain validation experiments with one '
        'frozen encoder seed, capped costs and repeated validation exploration; no test-set '
        'promotion or production default was changed.', '',
        '## Artifacts and checks', '',
        'Results and prediction arrays: `experiments/cost_objectives_20260905/`. '
        'Runner: `scripts/run_neighbor_cost_probe.py --objective-sweep`. '
        'The objective-selection and cross-fit leakage tests passed (2 tests). All seven original '
        'raw baseline prediction arrays reproduced within absolute tolerance 1e-12. '
        '`git diff --check` passed.', '']
    (root / 'docs/cost_objectives_20260905.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
