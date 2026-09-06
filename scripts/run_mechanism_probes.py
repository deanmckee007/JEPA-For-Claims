"""Votes, diverse retrieval, split heads and chronological-order diagnostics."""
import argparse
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.claim_set_calibration import candidate_coverage, shortlist_logits, fit_set_calibration
from jepa_utils.retrieval_mechanisms import (
    weighted_neighbors, truncate_neighbors, neighbor_code_votes, mix_votes,
    hybrid_candidates, PermutedHistoryLoader,
)
from scripts.run_exact_raw_cost_baselines import build_exact_history_matrix
from scripts.run_frozen_generation_probe import (
    collect_frozen_dataset, token_ids_to_multi_hot, resolve_device, score_generation, write_json,
)
from scripts.run_retrieval_generation_probe import (
    SOURCE, build_candidates, CandidateDecoder, predict_candidate, fit_candidate,
)
from scripts.run_retrieval_replication import (
    grouped_inner_split, eligible_sample_ids, subset_data, concatenate_data, evaluate_prediction, filter_prediction,
)


def vote_prediction(votes, timing_votes, data, sizes, maxima, copy_weight):
    result = {}
    counts = []
    for key, size, modality_votes in zip(['cpt', 'icd'], sizes, votes):
        probabilities = np.zeros((len(data['ttnc']), size), dtype=np.float32)
        for row, tokens in enumerate(modality_votes):
            for token, value in tokens.items():
                probabilities[row, token] = value
        previous = token_ids_to_multi_hot(data[f'previous_{key}_ids'], size)
        probabilities = (1 - copy_weight) * probabilities + copy_weight * previous
        probabilities[:, 0] = 0
        counts.append(probabilities.sum(1))
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        # A fixed shift gives vote probabilities a useful range under the same
        # inner calibration grid; it does not alter code ranking.
        result[f'{key}_logits'] = np.log(probabilities / (1 - probabilities)) + 4.0
    timing = np.zeros((len(data['ttnc']), sizes[2]), dtype=np.float32)
    for row, tokens in enumerate(timing_votes):
        for token, value in tokens.items():
            timing[row, token] = (1 - copy_weight) * value
        timing[row, int(data['previous_ttnc'][row])] += copy_weight
    result['ttnc_logits'] = np.log(np.clip(timing, 1e-6, 1))
    result['cardinality_fraction'] = np.stack(counts, 1) / np.asarray(maxima)
    return result


def inner_calibration_score(prediction, data, config):
    scores = []
    for column, key in enumerate(['cpt', 'icd']):
        maximum = getattr(config, f'max_{key}_tokens')
        shortlist = shortlist_logits(prediction[f'{key}_logits'], data[f'previous_{key}_ids'], maximum)
        choice = fit_set_calibration(shortlist, prediction['cardinality_fraction'][:, column] * maximum,
            data[f'{key}_ids'], previous_ids=data[f'previous_{key}_ids'])
        scores.append(choice['inner_micro_f1'])
    return float(np.mean(scores))


def raw_generation_score(prediction, data, config, support):
    targets = [token_ids_to_multi_hot(data[f'{key}_ids'], getattr(config, f'{key}_vocab_size')) for key in ['cpt', 'icd']]
    return score_generation(prediction, *targets, data['ttnc'], train_cpt_support=support[0],
        train_icd_support=support[1], max_cpt_tokens=config.max_cpt_tokens,
        max_icd_tokens=config.max_icd_tokens, min_class_support=10)


def load_data(args):
    source = Path(args.replication_dir)
    original = json.loads((source / 'protocol.json').read_text())
    if not any(item['seed'] == 42 and Path(item['path']).resolve() == Path(args.checkpoint).resolve()
        for item in original['checkpoints']):
        raise ValueError('This seed-42 pilot requires the matching replication checkpoint')
    if any(original[key] != value for key, value in [('epochs', 20), ('hidden_dim', 256), ('neighbors', 16), ('max_candidates', 128)]):
        raise ValueError('This pilot requires the original 20-epoch 16/128 replication budget')
    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    config.data_path, config.data_contract_path = args.data_path, args.data_contract
    config.use_generative_save = config.use_plotting = config.pretrain_diffusion = False
    config = apply_runtime_config_overrides(config)
    train_subset, _, val_subset, _, config, dataset = prepare_data(config, requested_eval_split='val')
    loaders = [DataLoader(subset, batch_size=256, shuffle=False, collate_fn=dataset.collate_eval_fn)
        for subset in [train_subset, val_subset]]
    ids = eligible_sample_ids(train_subset, loaders[0])
    val_ids = eligible_sample_ids(val_subset, loaders[1])
    manifest = json.loads((source / 'inner_split.json').read_text())
    for values, key in [(ids, 'ordered_sample_ids_sha256'), (val_ids, 'validation_ids_sha256')]:
        if hashlib.sha256('\n'.join(values).encode()).hexdigest() != manifest[key]:
            raise ValueError('Probe row identities do not match the replication')
    fit_positions, cal_positions = grouped_inner_split(ids, original['calibration_fraction'], original['calibration_seed'])
    if fit_positions.tolist() != manifest['fit_positions'] or cal_positions.tolist() != manifest['calibration_positions']:
        raise ValueError('Probe inner split does not match the replication')
    device = resolve_device(args.accelerator)
    model = load_claims_model_checkpoint(HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device)
    train_all, val = [collect_frozen_dataset(model, loader, [SOURCE], device) for loader in loaders]
    train = subset_data(train_all, fit_positions)
    calibration = subset_data(train_all, cal_positions)
    evaluation = concatenate_data(calibration, val)
    train_lookup = {dataset.sample_ids[index]: index for index in train_subset.indices}
    val_lookup = {dataset.sample_ids[index]: index for index in val_subset.indices}
    aligned_train = np.asarray([train_lookup[sample] for sample in ids])
    aligned_val = np.asarray([val_lookup[sample] for sample in val_ids])
    return config, dataset, model, train, evaluation, calibration, aligned_train, aligned_val, fit_positions, cal_positions, device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['checkpoint', 'data-path', 'data-contract', 'replication-dir', 'output-dir']:
        parser.add_argument(f'--{flag}', required=True)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'], default='auto')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**vars(args), 'test_accessed': False, 'encoder_seed': 42, 'decoder_seed': 201,
        'neighbors': {'dense': 16, 'hybrid': '8 dense + 8 raw'}, 'candidate_slots': 128,
        'raw_neighbors': 'TF-IDF of exact recency counts and last-claim indicators; IDF fit on inner fitting split',
        'vote_copy_weights': [0.0, 0.25, 0.5, 0.75, 1.0], 'vote_logit_offset': 4.0,
        'vote_selection': 'mean inner CPT/ICD calibrated micro F1', 'decoder_epochs': 20,
        'split_loss': 'average of independently balanced retention and addition BCE; all retrieved non-target additions are negatives',
        'permutation_seeds': [401, 402, 403], 'temporal_candidates_and_calibration': 'fixed to clean histories',
        'order_invariant_training_gate': 'mean absolute inner CPT/ICD AP change over three permutations >= 0.01',
        'calibration_objectives': ['micro_f1', 'balanced_new_f1']}
    if (output / 'protocol.json').exists() and json.loads((output / 'protocol.json').read_text()) != protocol:
        raise ValueError('Choose a new output directory for a different protocol')
    write_json(output / 'protocol.json', protocol)
    config, dataset, model, train, evaluation, calibration, train_indices, val_indices, fit_positions, cal_positions, device = load_data(args)
    cal_size = len(calibration['ttnc'])
    sizes = [config.cpt_vocab_size, config.icd_vocab_size, config.ttnc_vocab_size]
    maxima = [config.max_cpt_tokens, config.max_icd_tokens]
    support = [token_ids_to_multi_hot(train[f'{key}_ids'], size).sum(0) for key, size in zip(['cpt', 'icd'], sizes)]
    raw_all, _, _ = build_exact_history_matrix(train_indices, dataset, future_claim_k=config.future_claim_k)
    raw_val, _, _ = build_exact_history_matrix(val_indices, dataset, future_claim_k=config.future_claim_k)
    transformer = TfidfTransformer()
    raw_train = transformer.fit_transform(raw_all[fit_positions])
    raw_eval = transformer.transform(sparse.vstack([raw_all[cal_positions], raw_val]))
    dense_neighbors = [weighted_neighbors(train['features'][SOURCE], data['features'][SOURCE], training=is_train)
        for data, is_train in [(train, True), (evaluation, False)]]
    raw_neighbors = [weighted_neighbors(raw_train, data, training=is_train) for data, is_train in [(raw_train, True), (raw_eval, False)]]
    candidates, vote_banks, coverage = {}, {}, {}
    candidates['dense'] = [[build_candidates(data[f'previous_{key}_ids'], train[f'{key}_ids'],
        [ids for ids, _ in neighbors], 128) for key in ['cpt', 'icd']]
        for data, neighbors in zip([train, evaluation], dense_neighbors)]
    candidates['raw'] = [[build_candidates(data[f'previous_{key}_ids'], train[f'{key}_ids'],
        [ids for ids, _ in neighbors], 128) for key in ['cpt', 'icd']]
        for data, neighbors in zip([train, evaluation], raw_neighbors)]
    candidates['hybrid'] = []
    for data, dense, raw in zip([train, evaluation], dense_neighbors, raw_neighbors):
        candidate_pair = []
        for key in ['cpt', 'icd']:
            left = neighbor_code_votes(train[f'{key}_ids'], truncate_neighbors(dense, 8))
            right = neighbor_code_votes(train[f'{key}_ids'], truncate_neighbors(raw, 8))
            candidate_pair.append(hybrid_candidates(data[f'previous_{key}_ids'], left, right, 128))
        candidates['hybrid'].append(candidate_pair)
    for name in ['dense', 'raw', 'hybrid']:
        banks = []
        for key in ['cpt', 'icd', 'ttnc']:
            targets = train[f'{key}_ids'] if key != 'ttnc' else train['ttnc'][:, None]
            if name == 'hybrid':
                bank = mix_votes(neighbor_code_votes(targets, truncate_neighbors(dense_neighbors[1], 8)),
                    neighbor_code_votes(targets, truncate_neighbors(raw_neighbors[1], 8)))
            else:
                bank = neighbor_code_votes(targets, dense_neighbors[1] if name == 'dense' else raw_neighbors[1])
            banks.append(bank)
        vote_banks[name] = banks
    # All conditions are specified in advance; coverage never selects a model.
    for name, candidate in candidates.items():
        coverage[name] = {split: {key: candidate_coverage(ids[positions], evaluation[f'{key}_ids'][positions],
            evaluation[f'previous_{key}_ids'][positions]) for key, ids in zip(['cpt', 'icd'], candidate[1])}
            for split, positions in [('inner', slice(None, cal_size)), ('validation', slice(cal_size, None))]}
    results, mixtures, temporal = {}, {}, {}
    def persist(completed=False):
        write_json(output / 'summary.json', {'protocol': protocol, 'conditions': results, 'vote_mixtures': mixtures,
            'candidate_coverage': coverage, 'temporal': temporal, 'completed': completed})
    def score(name, prediction):
        results[name] = {objective: evaluate_prediction(prediction, evaluation, cal_size, config, support, objective)
            for objective in protocol['calibration_objectives']}
        persist()
        print(f'Scored {name}', flush=True)
    saved = Path(args.replication_dir) / 'encoder42_decoder201' / 'candidate.pt'
    state = torch.load(saved, map_location=device, weights_only=False)
    head = CandidateDecoder(state['input_dim'], state['hidden_dim'], state['vocab_sizes']).to(device)
    head.load_state_dict(state['state_dict'])
    baseline = predict_candidate(head, evaluation, candidates['dense'][1], state['feature_mean'], state['feature_std'], sizes, 256, device)
    score('candidate_dense', baseline)
    for name, banks in vote_banks.items():
        choices = []
        for alpha in protocol['vote_copy_weights']:
            inner_banks = [[votes for votes in bank[:cal_size]] for bank in banks]
            prediction = vote_prediction(inner_banks[:2], inner_banks[2], calibration, sizes, maxima, alpha)
            if name in candidates:
                prediction = filter_prediction(prediction, [ids[:cal_size] for ids in candidates[name][1]])
            choices.append({'copy_weight': alpha, 'inner_score': inner_calibration_score(prediction, calibration, config)})
        best = max(choices, key=lambda item: item['inner_score'])
        mixtures[name] = {'selected': best, 'choices': choices}
        prediction = vote_prediction(banks[:2], banks[2], evaluation, sizes, maxima, best['copy_weight'])
        if name in candidates:
            prediction = filter_prediction(prediction, candidates[name][1])
        score(f'votes_{name}', prediction)
        del prediction
    for name, separate, candidate_source in [('candidate_hybrid', False, 'hybrid'), ('split_heads_dense', True, 'dense')]:
        run_args = SimpleNamespace(seed=201, hidden_dim=256, epochs=20, batch_size=256,
            separate_retention=separate, candidate_artifact_path=str(output / f'{name}.pt'))
        prediction, history = fit_candidate(train, evaluation, *candidates[candidate_source], config, run_args, device)
        score(name, prediction)
        write_json(output / f'{name}_loss.json', history)
        del prediction
    inner_baseline = raw_generation_score({key: value[:cal_size] for key, value in baseline.items()}, calibration, config, support)
    cal_loader = DataLoader(Subset(dataset, train_indices[cal_positions].tolist()), batch_size=256,
        shuffle=False, collate_fn=dataset.collate_eval_fn)
    val_loader = DataLoader(Subset(dataset, val_indices.tolist()), batch_size=256,
        shuffle=False, collate_fn=dataset.collate_eval_fn)
    changes = []
    for seed in protocol['permutation_seeds']:
        loader = itertools.chain(PermutedHistoryLoader(cal_loader, config.future_claim_k, seed),
            PermutedHistoryLoader(val_loader, config.future_claim_k, seed + 10000))
        shuffled = collect_frozen_dataset(model, loader, [SOURCE], device)
        for key in ['cpt_ids', 'icd_ids', 'ttnc', 'previous_cpt_ids', 'previous_icd_ids', 'previous_ttnc']:
            if not np.array_equal(shuffled[key], evaluation[key]):
                raise ValueError('Permutation changed protected latest or future claims')
        prediction = predict_candidate(head, shuffled, candidates['dense'][1], state['feature_mean'], state['feature_std'], sizes, 256, device)
        inner = raw_generation_score({key: value[:cal_size] for key, value in prediction.items()}, calibration, config, support)
        delta = {key: inner[key]['micro_average_precision'] - inner_baseline[key]['micro_average_precision'] for key in ['cpt', 'icd']}
        changes.extend(abs(value) for value in delta.values())
        locked = {key: np.concatenate([baseline[key][:cal_size], value[cal_size:]]) for key, value in prediction.items()}
        score(f'order_shuffle_{seed}', locked)
        for objective in protocol['calibration_objectives']:
            if results[f'order_shuffle_{seed}'][objective]['calibration'] != results['candidate_dense'][objective]['calibration']:
                raise ValueError('Permutation must not change the calibrated thresholds')
        temporal[str(seed)] = {'inner_AP_delta': delta}
        del prediction, shuffled, locked
    temporal['mean_absolute_inner_AP_change'] = float(np.mean(changes))
    temporal['train_order_invariant_control'] = bool(np.mean(changes) >= 0.01)
    persist(completed=True)
    print(f"Temporal gate: {temporal['train_order_invariant_control']}", flush=True)


if __name__ == '__main__':
    main()
