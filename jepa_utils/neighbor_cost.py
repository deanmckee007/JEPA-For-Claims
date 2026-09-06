"""Cross-fitted neighbor summaries for cost probes."""
import numpy as np
from sklearn.model_selection import GroupKFold
from jepa_utils.retrieval_mechanisms import weighted_neighbors


def summarize_neighbors(reference_x, query_x, costs, codes, count=16):
    """Eight cost statistics and 64 signed code-vote bins per retrieval space."""
    rows = []
    for ids, weights in weighted_neighbors(reference_x, query_x, count=count):
        y = np.asarray(costs)[ids]
        dollars = np.expm1(np.clip(y, 0, 10))
        mean = weights @ y
        row = [mean, np.sqrt(weights @ ((y - mean) ** 2)),
               *np.quantile(y, [.1, .5, .9]), np.log1p(weights @ dollars),
               float(y.max()), float(1 / np.sum(weights ** 2))]
        votes = np.zeros(64)
        for modality, tokens in enumerate(codes):
            for index, weight in zip(ids, weights):
                for token in set(tokens[index].tolist()):
                    if token > 1:
                        # Stable signed hashing, independent of labels and vocabulary frequency.
                        hashed = int(token) * 2654435761
                        votes[modality * 32 + hashed % 32] += weight * (1 if (hashed // 32) % 2 else -1)
        rows.append(np.r_[row, votes])
    return np.asarray(rows, dtype=np.float32)


def crossfit_summaries(spaces, evaluation_spaces, costs, codes, groups, folds=5):
    """Every training row's reference bank excludes its entire content-ID group."""
    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    training = np.empty((len(costs), 72 * len(spaces)), dtype=np.float32)
    assignments = np.empty(len(costs), dtype=int)
    for fold, (reference, query) in enumerate(splitter.split(np.arange(len(costs)), groups=groups)):
        assert not set(groups[reference]) & set(groups[query])
        training[query] = np.concatenate([
            summarize_neighbors(x[reference], x[query], costs[reference], [c[reference] for c in codes])
            for x in spaces], axis=1)
        assignments[query] = fold
    evaluation = np.concatenate([
        summarize_neighbors(x, q, costs, codes) for x, q in zip(spaces, evaluation_spaces)], axis=1)
    return training, evaluation, assignments
