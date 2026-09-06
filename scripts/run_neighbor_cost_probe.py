"""Compare raw, JEPA and grouped out-of-fold retrieval features for cost."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from threadpoolctl import threadpool_limits
from jepa_utils.neighbor_cost import crossfit_summaries
from jepa_utils.representation_eval import score_regression_predictions
from scripts.run_exact_raw_cost_baselines import build_exact_history_matrix, top_frequency_columns
from scripts.run_mechanism_probes import load_data, SOURCE


def label_positions(groups, fraction, seed):
    """Random group sampling without consulting unpurchased cost labels."""
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    chosen = unique[:max(5, int(round(len(unique) * fraction))) ]
    return np.flatnonzero(np.isin(groups, chosen))


def fit_objective_candidates(x, y, q, objective):
    """Return log-scale predictions so scoring preserves the existing endpoint."""
    target = y if objective == 'log_squared' else np.expm1(y)
    loss = 'absolute_error' if objective == 'dollar_absolute' else 'squared_error'
    predictions = []
    for iterations in [100, 200]:
        model = HistGradientBoostingRegressor(loss=loss, max_iter=iterations, learning_rate=.05,
            max_leaf_nodes=31, min_samples_leaf=max(5, min(20, len(y) // 10)),
            l2_regularization=1., early_stopping=False, random_state=201)
        with threadpool_limits(limits=4):
            model.fit(x, target)
            prediction = model.predict(q)
        if objective != 'log_squared':
            prediction = np.log1p(np.clip(prediction, 0, np.expm1(10)))
        predictions.append((iterations, prediction))
    return predictions


def select_objective_candidate(candidates, targets, positions, metric):
    choices = []
    for iterations, prediction in candidates:
        if metric == 'log_mse':
            value = np.mean((prediction[positions] - targets[positions]) ** 2)
        else:
            error = np.expm1(np.clip(prediction[positions], 0, 10)) - np.expm1(targets[positions])
            value = np.mean(np.abs(error) if metric == 'dollar_mae' else error ** 2)
        choices.append((float(value), iterations, prediction))
    return min(choices, key=lambda item: item[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['checkpoint', 'data-path', 'data-contract', 'replication-dir', 'output-dir']:
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--objective-sweep', action='store_true')
    parser.add_argument('--prior-dir', default='experiments/neighbor_cost_20260905')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config, dataset, model, train, evaluation, calibration, indices, val_indices, fit, cal, device = load_data(args)
    raw_all, y_all, _ = build_exact_history_matrix(indices, dataset, future_claim_k=config.future_claim_k)
    raw_val, y_val, _ = build_exact_history_matrix(val_indices, dataset, future_claim_k=config.future_claim_k)
    raw = raw_all[fit]
    raw_eval = sparse.vstack([raw_all[cal], raw_val]).tocsr()
    y, eval_y = y_all[fit], np.r_[y_all[cal], y_val]
    groups = np.asarray([dataset.sample_ids[i] for i in indices])
    train_groups, cal_groups = groups[fit], groups[cal]
    val_groups = np.asarray([dataset.sample_ids[i] for i in val_indices])
    assert not set(train_groups) & (set(cal_groups) | set(val_groups))
    columns = top_frequency_columns(raw, 256)
    base, base_eval = raw[:, columns].toarray(), raw_eval[:, columns].toarray()
    tfidf = TfidfTransformer()
    spaces = [tfidf.fit_transform(raw), train['features'][SOURCE]]
    eval_spaces = [tfidf.transform(raw_eval), evaluation['features'][SOURCE]]
    protocol = {**vars(args), 'encoder_seed': 42, 'label_seeds': [101, 102, 103],
        'label_fractions': [.01, .05, 1.0], 'folds': 5, 'neighbors_per_space': 16,
        'fit_rows': len(y), 'calibration_rows': len(cal), 'validation_rows': len(y_val),
        'target': 'existing log1p cost capped at 10; dollar scores retain this endpoint',
        'selection': '100 or 200 boosting iterations by labeled inner log-MSE; no internal early stopping',
        'labels': 'random content groups; fitting and calibration both restricted to stated fraction',
        'references': 'only labeled fitting rows, including code votes; OOF bank ~80%, evaluation bank 100%',
        'raw': '256 fitting-frequency columns + mandatory stats; TF-IDF fit on unlabeled fitting histories',
        'test_accessed': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    if args.objective_sweep:
        protocol.update(objectives=['log_squared', 'dollar_squared', 'dollar_absolute'],
            selection_metrics=['log_mse', 'dollar_mae', 'dollar_mse'],
            selection='Each objective independently selects 100/200 iterations under each of the same three inner metrics',
            conditions=['raw', 'raw_cost_only'])
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    runs = []
    for fraction in protocol['label_fractions']:
        for seed in protocol['label_seeds']:
            if fraction == 1 and seed != 101:
                continue  # identical full-label experiment with deterministic trees/folds
            pos = label_positions(train_groups, fraction, seed)
            select = label_positions(cal_groups, fraction, seed + 1000)
            codes = [train[key + '_ids'][pos] for key in ['cpt', 'icd']]
            neighbor, neighbor_eval, folds = crossfit_summaries([x[pos] for x in spaces], eval_spaces,
                y[pos], codes, train_groups[pos])
            manifest = {'fit_positions': fit[pos].tolist(), 'calibration_positions': cal[select].tolist(),
                'folds': folds.tolist(), 'fit_groups': train_groups[pos].tolist()}
            if args.objective_sweep:
                prior = json.loads((Path(args.prior_dir) / f'labels_{fraction}_{seed}.json').read_text())
                if manifest != prior:
                    raise ValueError('Objective sweep must reuse identical labels and folds')
            (out / f'labels_{fraction}_{seed}.json').write_text(json.dumps(manifest))
            conditions = {
                'raw': (base[pos], base_eval),
                'raw_jepa': (np.c_[base[pos], spaces[1][pos]], np.c_[base_eval, eval_spaces[1]]),
                'raw_neighbors': (np.c_[base[pos], neighbor], np.c_[base_eval, neighbor_eval]),
                'raw_jepa_neighbors': (np.c_[base[pos], spaces[1][pos], neighbor], np.c_[base_eval, eval_spaces[1], neighbor_eval]),
            }
            if fraction == 1:
                for name, cols in [('cost_only', np.r_[0:8, 72:80]), ('votes_only', np.r_[8:72, 80:144])]:
                    conditions['raw_' + name] = (np.c_[base[pos], neighbor[:, cols]], np.c_[base_eval, neighbor_eval[:, cols]])
            if args.objective_sweep:
                cols = np.r_[0:8, 72:80]
                conditions = {'raw': (base[pos], base_eval),
                    'raw_cost_only': (np.c_[base[pos], neighbor[:, cols]], np.c_[base_eval, neighbor_eval[:, cols]])
                }
                for name, (x, q) in conditions.items():
                    for objective in protocol['objectives']:
                        candidates = fit_objective_candidates(x, y[pos], q, objective)
                        for selection in protocol['selection_metrics']:
                            loss, iterations, prediction = select_objective_candidate(candidates, eval_y, select, selection)
                            prediction = prediction[len(cal):]
                            metrics = score_regression_predictions(prediction, y_val)
                            dollars = np.expm1(y_val)
                            metrics['aggregate_cost_bias_percent'] = float(100 * (np.expm1(np.clip(prediction, 0, 10)).sum() / dollars.sum() - 1))
                            for rate in [.01, .05, .1]:
                                top = np.argsort(-prediction, kind='stable')[:max(1, int(np.ceil(len(prediction) * rate)))]
                                metrics[f'cost_capture_top_{rate}'] = float(dollars[top].sum() / dollars.sum())
                            runs.append({'condition': name, 'objective': objective, 'selection': selection,
                                'fraction': fraction, 'seed': seed, 'fit_labels': len(pos), 'calibration_labels': len(select),
                                'iterations': iterations, 'inner_loss': loss, 'metrics': metrics})
                            np.savez_compressed(out / f'predictions_{fraction}_{seed}_{name}_{objective}_{selection}.npz', prediction=prediction, target=y_val)
                        (out / 'summary.json').write_text(json.dumps({'protocol': protocol, 'runs': runs, 'completed': False}, indent=2))
                        print(f'{fraction} {seed} {name} {objective} completed', flush=True)
                continue
            for name, (x, q) in conditions.items():
                choices = []
                for iterations in [100, 200]:
                    regressor = HistGradientBoostingRegressor(max_iter=iterations, learning_rate=.05,
                        max_leaf_nodes=31, min_samples_leaf=max(5, min(20, len(pos) // 10)),
                        l2_regularization=1., early_stopping=False, random_state=201)
                    with threadpool_limits(limits=4):
                        regressor.fit(x, y[pos])
                        prediction = regressor.predict(q)
                    loss = float(np.mean((prediction[select] - eval_y[select]) ** 2))
                    choices.append((loss, iterations, prediction[len(cal):]))
                loss, iterations, prediction = min(choices, key=lambda item: item[0])
                metrics = score_regression_predictions(prediction, y_val)
                dollars = np.expm1(y_val)
                for rate in [.01, .05, .1]:
                    top = np.argsort(-prediction, kind='stable')[:max(1, int(np.ceil(len(prediction) * rate)))]
                    metrics[f'cost_capture_top_{rate}'] = float(dollars[top].sum() / dollars.sum())
                runs.append({'condition': name, 'fraction': fraction, 'seed': seed, 'fit_labels': len(pos),
                    'calibration_labels': len(select), 'iterations': iterations, 'inner_log_mse': loss, 'metrics': metrics})
                np.savez_compressed(out / f'predictions_{fraction}_{seed}_{name}.npz', prediction=prediction, target=y_val)
                (out / 'summary.json').write_text(json.dumps({'protocol': protocol, 'runs': runs, 'completed': False}, indent=2))
                print(f'{fraction} {seed} {name}: {metrics}', flush=True)
    (out / 'summary.json').write_text(json.dumps({'protocol': protocol, 'runs': runs, 'completed': True}, indent=2))


if __name__ == '__main__':
    main()
