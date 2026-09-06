"""Disentangle hybrid retrieval diversity from actual candidate-set size."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mechanism_probes import load_data
from scripts.run_exact_raw_cost_baselines import build_exact_history_matrix
from scripts.run_frozen_generation_probe import token_ids_to_multi_hot, write_json
from scripts.run_retrieval_generation_probe import SOURCE, build_candidates, fit_candidate
from scripts.run_retrieval_replication import evaluate_prediction
from jepa_utils.claim_set_calibration import candidate_coverage
from jepa_utils.retrieval_mechanisms import (
    weighted_neighbors, truncate_neighbors, neighbor_code_votes, hybrid_candidates, match_candidate_counts,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['checkpoint', 'data-path', 'data-contract', 'replication-dir', 'output-dir']:
        parser.add_argument(f'--{flag}', required=True)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'], default='auto')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**vars(args), 'test_accessed': False, 'seed': 201, 'epochs': 20,
        'candidate_matching': 'Hybrid proposals trimmed to dense non-padding count per row; dense backfill if hybrid is shorter',
        'reference_budget': '16 dense vs 8 dense + 8 raw; at most 128 slots; same inner split'}
    write_json(output / 'protocol.json', protocol)
    config, dataset, model, train, evaluation, calibration, train_indices, val_indices, fit_positions, cal_positions, device = load_data(args)
    del model
    cal_size = len(calibration['ttnc'])
    raw_all, _, _ = build_exact_history_matrix(train_indices, dataset, future_claim_k=config.future_claim_k)
    raw_val, _, _ = build_exact_history_matrix(val_indices, dataset, future_claim_k=config.future_claim_k)
    transform = TfidfTransformer()
    raw_train = transform.fit_transform(raw_all[fit_positions])
    raw_eval = transform.transform(sparse.vstack([raw_all[cal_positions], raw_val]))
    candidates, coverage, counts = [], {}, {}
    for split, data, raw, training in [('fit', train, raw_train, True), ('evaluation', evaluation, raw_eval, False)]:
        dense = weighted_neighbors(train['features'][SOURCE], data['features'][SOURCE], training=training)
        raw_neighbors = weighted_neighbors(raw_train, raw, training=training)
        pair = []
        for key in ['cpt', 'icd']:
            baseline = build_candidates(data[f'previous_{key}_ids'], train[f'{key}_ids'], [ids for ids, _ in dense], 128)
            hybrid = hybrid_candidates(data[f'previous_{key}_ids'],
                neighbor_code_votes(train[f'{key}_ids'], truncate_neighbors(dense, 8)),
                neighbor_code_votes(train[f'{key}_ids'], truncate_neighbors(raw_neighbors, 8)), 128)
            matched = match_candidate_counts(hybrid, baseline)
            assert np.array_equal(np.count_nonzero(matched, 1), np.count_nonzero(baseline, 1))
            pair.append(matched)
            if split == 'evaluation':
                counts[key] = {name: float(np.count_nonzero(value[cal_size:], 1).mean())
                    for name, value in [('dense', baseline), ('hybrid', hybrid), ('matched', matched)]}
                coverage[key] = candidate_coverage(matched[cal_size:], data[f'{key}_ids'][cal_size:],
                    data[f'previous_{key}_ids'][cal_size:])
        candidates.append(pair)
    support = [token_ids_to_multi_hot(train[f'{key}_ids'], getattr(config, f'{key}_vocab_size')).sum(0) for key in ['cpt', 'icd']]
    run_args = SimpleNamespace(seed=201, hidden_dim=256, epochs=20, batch_size=256,
        candidate_artifact_path=str(output / 'candidate_matched.pt'))
    prediction, history = fit_candidate(train, evaluation, *candidates, config, run_args, device)
    results = {objective: evaluate_prediction(prediction, evaluation, cal_size, config, support, objective)
        for objective in ['micro_f1', 'balanced_new_f1']}
    write_json(output / 'summary.json', {'protocol': protocol, 'candidate_counts': counts,
        'candidate_coverage': coverage, 'conditions': {'candidate_count_matched': results},
        'loss_history': history, 'completed': True})


if __name__ == '__main__':
    main()
