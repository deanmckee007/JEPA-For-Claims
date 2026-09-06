"""Render the completed neighbor-cost pilot without rerunning models."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def main():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / 'experiments/neighbor_cost_20260905/summary.json').read_text())
    if not result['completed']:
        raise ValueError('Experiment is still running')
    groups = defaultdict(list)
    for run in result['runs']:
        groups[(run['fraction'], run['condition'])].append(run)
    lines = ['# Neighbor summaries for cost prediction — 2026-09-05', '',
        '## Protocol', '',
        'Frozen seed-42 online LeVJEPA embeddings; the existing content-group inner split gives '
        '12,077 fitting, 2,973 calibration and 3,188 validation rows with at least two claims. '
        'This is the generation-eligible cohort, so absolute errors are not directly comparable '
        'with older cost probes on the broader representation cohort. No test split was accessed.', '',
        'The four main conditions share 256 exact-history columns selected on fitting histories. '
        'JEPA adds 256 embedding dimensions. Neighbors add 144 features: for each of raw TF-IDF '
        'and JEPA cosine retrieval, eight cost statistics and 64 signed CPT/ICD vote bins from '
        '16 neighbors. Costs include weighted log mean/dispersion, unweighted quantiles, weighted '
        'dollar mean, maximum and effective neighbor count. The vote bins contain only next-code '
        'labels from fitting reference rows; the query’s future codes are never features.', '',
        'Five-fold cross-fitting excludes every query’s entire content-ID group from its training '
        'reference bank. Calibration and validation use only labeled fitting references. All '
        'reference labels, including code votes, are restricted to the same sampled fitting '
        'budget. Random content-group sampling does not consult unsampled costs. Calibration '
        'labels are separately restricted to the stated fraction and counted in the artifacts. '
        'Training banks contain about 80% of available fitting labels, versus 100% for evaluation; '
        'this finite-bank distribution shift is a limitation.', '',
        'Each condition selects 100 or 200 boosting iterations by labeled inner log-MSE. '
        'Trees use learning rate 0.05, 31 leaves, L2=1 and no internal early stopping, avoiding '
        'cross-fit label contamination of an internal stopping set. Models are not refit on '
        'calibration labels. The 1% and 5% budgets use seeds 101–103; full labels are deterministic '
        'and run once. Two additional full-label controls separate neighbor costs from code votes.', '',
        'Dollar scores retain the existing log1p target capped at 10 (about $22,025). '
        'Top-5% capture is the fraction of capped validation cost among the 5% ranked highest. '
        'Content IDs are sequence hashes, not verified member identities; this is not a prospective '
        'member-disjoint study. The encoder saw the full training cohort during representation '
        'pretraining. Raw features retain uncapped within-claim codes while embeddings use token caps.', '',
        '## Results', '',
        'Means ± population SD across label seeds; full-label rows have one run.', '',
        '| Labels | Features | MAE ($) | RMSE ($) | Log RMSE | Top-5% cost capture |',
        '|---:|---|---:|---:|---:|---:|']
    for (fraction, condition), runs in groups.items():
        cells = []
        for key, scale, decimals in [('target_probe_mae_dollars', 1, 2), ('target_probe_rmse_dollars', 1, 2),
                ('target_probe_rmse_log1p', 1, 4), ('cost_capture_top_0.05', 100, 2)]:
            values = [r['metrics'][key] * scale for r in runs]
            cell = f'{mean(values):.{decimals}f}'
            if len(values) > 1:
                cell += f' ± {pstdev(values):.{decimals}f}'
            if scale == 100:
                cell += '%'
            cells.append(cell)
        lines.append(f'| {fraction:.0%} | {condition} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired changes', '',
        'Negative dollar deltas indicate improvement over the matching baseline.', '',
        '| Labels | Comparison | MAE delta ($) | RMSE delta ($) | MAE wins |',
        '|---:|---|---:|---:|---:|']
    for fraction in [.01, .05, 1.]:
        for added, baseline in [('raw_neighbors', 'raw'), ('raw_jepa_neighbors', 'raw_jepa')]:
            a, b = groups[(fraction, added)], groups[(fraction, baseline)]
            deltas = [[x['metrics'][key] - y['metrics'][key] for x, y in zip(a, b)]
                for key in ['target_probe_mae_dollars', 'target_probe_rmse_dollars']]
            lines.append(f'| {fraction:.0%} | {added} − {baseline} | {mean(deltas[0]):.2f} | '
                f'{mean(deltas[1]):.2f} | {sum(d < 0 for d in deltas[0])}/{len(a)} |')
    lines += ['', '## Interpretation', '',
        'The combined neighbor summaries do not consistently beat raw history: mean MAE worsens '
        'by $60.60 at 1% labels, $9.88 at 5%, and $13.99 at full labels. They improve the weaker '
        'raw-plus-JEPA baseline modestly, which is insufficient evidence to replace raw history.', '',
        'At full labels, cost-only neighbor statistics provide the best observed MAE ($1,458.53) '
        'and RMSE ($2,456.19), improvements of only $1.76 and $19.28 over raw history. Code votes '
        'alone worsen MAE by $29.51 and RMSE by $25.11. The code-generation ranking improvements '
        'therefore have not transferred into a clear cost-prediction benefit through these summaries. '
        'Cost-only top-5% capture also falls slightly, from 9.74% to 9.65%.', '',
        'This is a pilot with one encoder seed and repeatedly explored validation data. Keep raw '
        'history as the baseline. A useful next cost-focused experiment would compare dollar-oriented '
        'training losses against the current log-MSE objective under the same split and label budget, '
        'retaining cost-only retrieval as a small ablation. The endpoint cap should remain explicit; '
        'uncapped cost prediction would require a separately defined target contract.', '',
        '## Validation and artifacts', '',
        '245 repository tests passed (five existing Lightning logging warnings). A direct '
        'perturbation test verifies that changing every cost in one held-out fold leaves '
        'that fold’s own features unchanged. `git diff --check` passed.', '',
        '- Runner: `scripts/run_neighbor_cost_probe.py`',
        '- Feature implementation: `jepa_utils/neighbor_cost.py`',
        '- Results, exact label/fold assignments and predictions: `experiments/neighbor_cost_20260905/`',
        '- Prior generation results: [mechanism probes](mechanism_probes_20260905.md)', '']
    (root / 'docs/neighbor_cost_20260905.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
