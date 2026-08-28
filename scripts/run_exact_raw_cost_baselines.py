"""Strong leakage-safe raw-history baselines for downstream cost prediction."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_utils.checkpointing import read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import score_regression_predictions
from scripts.run_cost_attribution_ladder import aggregate_runs, render_markdown
from scripts.run_cost_supervision_ablation import stratified_fraction_positions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--label-fractions", nargs="+", type=float,
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--ridge-alphas", nargs="+", type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--boost-top-columns", type=int, default=256)
    parser.add_argument("--boost-iterations", type=int, default=200)
    parser.add_argument("--recency-decay", type=float, default=0.8)
    args = parser.parse_args(argv)
    if any(not 0.0 < value <= 1.0 for value in args.label_fractions):
        parser.error("--label-fractions must be in (0, 1]")
    if any(value <= 0 for value in args.ridge_alphas):
        parser.error("--ridge-alphas must be positive")
    if args.boost_top_columns <= 0 or args.boost_iterations <= 0:
        parser.error("boost dimensions and iterations must be positive")
    if not 0.0 < args.recency_decay <= 1.0:
        parser.error("--recency-decay must be in (0, 1]")
    return args


def configure(args):
    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    if config is None:
        raise ValueError("Checkpoint does not contain a saved Config")
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.use_generative_save = False
    config.use_plotting = False
    config.pretrain_diffusion = False
    return apply_runtime_config_overrides(config)


def feature_layout(dataset):
    cpt = len(dataset.cpt_vocab)
    icd = len(dataset.icd_vocab)
    ttnc = len(dataset.ttnc_vocab)
    recency_cpt = 0
    recency_icd = recency_cpt + cpt
    recency_ttnc = recency_icd + icd
    last_cpt = recency_ttnc + ttnc
    last_icd = last_cpt + cpt
    last_ttnc = last_icd + icd
    statistics = last_ttnc + ttnc
    return {
        "recency_cpt": recency_cpt,
        "recency_icd": recency_icd,
        "recency_ttnc": recency_ttnc,
        "last_cpt": last_cpt,
        "last_icd": last_icd,
        "last_ttnc": last_ttnc,
        "statistics": statistics,
        "dimension": statistics + 4,
    }


def _evaluation_claims(dataset, dataset_index):
    claims = dataset.processed_data[dataset_index]
    policy = getattr(dataset, "evaluation_claim_inclusion_policy", None)
    if policy is not None:
        claims = [
            claim for claim in claims
            if dataset.claim_matches_policy(claim, policy)
        ]
    return claims[-dataset.max_claims_len :]


def build_exact_history_matrix(
    indices,
    dataset,
    *,
    future_claim_k=0,
    recency_decay=0.8,
):
    """Sparse recency counts and last-claim indicators, excluding the target claim."""
    layout = feature_layout(dataset)
    row_indices = []
    column_indices = []
    values = []
    targets = []
    future_slots = int(future_claim_k) + 1

    for output_row, dataset_index in enumerate(indices):
        claims = _evaluation_claims(dataset, int(dataset_index))
        available_future = min(future_slots, max(len(claims) - 1, 1))
        target_position = max(len(claims) - available_future, 0)
        history = claims[:target_position]
        accumulated = defaultdict(float)
        cpt_count = 0
        icd_count = 0
        for history_position, claim in enumerate(history):
            age = len(history) - 1 - history_position
            weight = float(recency_decay**age)
            is_last = history_position == len(history) - 1
            for token in claim.get("cpt", []):
                token_id = dataset.cpt_vocab.get(token, dataset.cpt_vocab.get("<UNK>", 1))
                if token_id == 0:
                    continue
                accumulated[layout["recency_cpt"] + token_id] += weight
                cpt_count += 1
                if is_last:
                    accumulated[layout["last_cpt"] + token_id] = 1.0
            for token in claim.get("icd", []):
                token_id = dataset.icd_vocab.get(token, dataset.icd_vocab.get("<UNK>", 1))
                if token_id == 0:
                    continue
                accumulated[layout["recency_icd"] + token_id] += weight
                icd_count += 1
                if is_last:
                    accumulated[layout["last_icd"] + token_id] = 1.0
            ttnc_id = dataset.ttnc_vocab.get(
                claim.get("ttnc"), dataset.ttnc_vocab.get("<UNK>", 1)
            )
            if ttnc_id != 0:
                accumulated[layout["recency_ttnc"] + ttnc_id] += weight
                if is_last:
                    accumulated[layout["last_ttnc"] + ttnc_id] = 1.0

        stats = layout["statistics"]
        if history:
            last_ttnc = dataset.ttnc_vocab.get(
                history[-1].get("ttnc"), dataset.ttnc_vocab.get("<UNK>", 1)
            )
            accumulated[stats] = len(history) / max(dataset.max_claims_len, 1)
            accumulated[stats + 1] = cpt_count / len(history)
            accumulated[stats + 2] = icd_count / len(history)
            accumulated[stats + 3] = last_ttnc / max(len(dataset.ttnc_vocab) - 1, 1)

        for column, value in accumulated.items():
            if value != 0.0:
                row_indices.append(output_row)
                column_indices.append(column)
                values.append(value)
        targets.append(dataset.targets[int(dataset_index)])

    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
        shape=(len(indices), layout["dimension"]),
        dtype=np.float32,
    )
    return matrix, np.asarray(targets, dtype=np.float32), layout


def inner_train_val_positions(targets, seed, validation_fraction=0.2):
    targets = np.asarray(targets)
    sorted_positions = np.argsort(targets, kind="mergesort")
    strata = np.array_split(sorted_positions, min(10, len(sorted_positions)))
    rng = np.random.default_rng(seed + 7717)
    train = []
    val = []
    for stratum in strata:
        shuffled = np.array(stratum, copy=True)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * validation_fraction)))
        val.extend(shuffled[:count].tolist())
        train.extend(shuffled[count:].tolist())
    if not train:
        train, val = val[:-1], val[-1:]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def fit_tuned_sparse_ridge(train_x, train_y, val_x, val_y, positions, alphas, seed):
    labeled_y = train_y[positions]
    mean = float(labeled_y.mean())
    std = float(max(labeled_y.std(), 1e-6))
    normalized_y = (labeled_y - mean) / std
    inner_train, inner_val = inner_train_val_positions(labeled_y, seed)
    best_alpha = None
    best_loss = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x[positions[inner_train]], normalized_y[inner_train])
        prediction = model.predict(train_x[positions[inner_val]])
        loss = float(np.mean((prediction - normalized_y[inner_val]) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    model = Ridge(alpha=best_alpha, solver="lsqr")
    model.fit(train_x[positions], normalized_y)
    prediction = model.predict(val_x) * std + mean
    return score_regression_predictions(prediction, val_y), best_alpha


def top_frequency_columns(matrix, count, mandatory_tail=4):
    count = min(int(count), matrix.shape[1])
    mandatory = np.arange(max(matrix.shape[1] - mandatory_tail, 0), matrix.shape[1])
    remaining_count = max(count - len(mandatory), 0)
    nonzero = np.asarray(matrix.getnnz(axis=0)).reshape(-1)
    nonzero[mandatory] = -1
    if remaining_count:
        selected = np.argpartition(nonzero, -remaining_count)[-remaining_count:]
        selected = selected[np.argsort(nonzero[selected])[::-1]]
    else:
        selected = np.asarray([], dtype=np.int64)
    return np.unique(np.concatenate([selected, mandatory])).astype(np.int64)


def fit_boosted_history(train_x, train_y, val_x, val_y, positions, *, seed, iterations):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=iterations,
        max_leaf_nodes=31,
        min_samples_leaf=max(5, min(20, len(positions) // 10)),
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(train_x[positions], train_y[positions])
    return score_regression_predictions(model.predict(val_x), val_y), int(model.n_iter_)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = configure(args)
    train_subset, _, val_subset, _, config, dataset = prepare_data(
        config, requested_eval_split="val"
    )
    train_x, train_y, layout = build_exact_history_matrix(
        train_subset.indices,
        dataset,
        future_claim_k=getattr(config, "future_claim_k", 0),
        recency_decay=args.recency_decay,
    )
    val_x, val_y, _ = build_exact_history_matrix(
        val_subset.indices,
        dataset,
        future_claim_k=getattr(config, "future_claim_k", 0),
        recency_decay=args.recency_decay,
    )
    top_columns = top_frequency_columns(train_x, args.boost_top_columns)
    boosted_train = train_x[:, top_columns].toarray().astype(np.float32)
    boosted_val = val_x[:, top_columns].toarray().astype(np.float32)

    runs = []
    for seed in args.seeds:
        for fraction in sorted(set(args.label_fractions)):
            positions = stratified_fraction_positions(train_y, fraction, seed)
            ridge_metrics, alpha = fit_tuned_sparse_ridge(
                train_x, train_y, val_x, val_y, positions, args.ridge_alphas, seed
            )
            runs.append(
                {
                    "condition": "exact_history_tuned_ridge",
                    "label_fraction": float(fraction),
                    "seed": int(seed),
                    "num_labels": int(len(positions)),
                    "selected_alpha": alpha,
                    "metrics": ridge_metrics,
                }
            )
            boost_metrics, iterations = fit_boosted_history(
                boosted_train,
                train_y,
                boosted_val,
                val_y,
                positions,
                seed=seed,
                iterations=args.boost_iterations,
            )
            runs.append(
                {
                    "condition": "top_frequency_history_boosted",
                    "label_fraction": float(fraction),
                    "seed": int(seed),
                    "num_labels": int(len(positions)),
                    "iterations": iterations,
                    "metrics": boost_metrics,
                }
            )

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "train_samples": int(train_x.shape[0]),
            "validation_samples": int(val_x.shape[0]),
            "exact_feature_dimension": int(train_x.shape[1]),
            "exact_train_nnz": int(train_x.nnz),
            "boost_feature_dimension": int(len(top_columns)),
            "future_target_excluded": True,
            "feature_layout": layout,
            "ridge_alphas": args.ridge_alphas,
            "label_fractions": sorted(set(args.label_fractions)),
            "seeds": args.seeds,
        },
        "aggregate": aggregate,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(aggregate), encoding="utf-8"
    )
    print(render_markdown(aggregate))


if __name__ == "__main__":
    main()
