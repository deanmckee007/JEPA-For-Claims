"""Training-bank voting, fixed-budget hybrid candidates, and context permutations."""
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def weighted_neighbors(reference, query, count=16, training=False):
    search = NearestNeighbors(metric="cosine", algorithm="brute").fit(reference)
    output = []
    k = min(reference.shape[0], count + (32 if training else 0))
    for start in range(0, query.shape[0], 256):
        distance, indices = search.kneighbors(query[start:start + 256], n_neighbors=k)
        for offset, (dist, ids) in enumerate(zip(distance, indices)):
            if training:
                keep = (ids != start + offset) & (dist > 1e-6)
                ids, dist = ids[keep], dist[keep]
            ids, dist = ids[:count], dist[:count]
            weights = np.maximum(1.0 - dist, 1e-6)
            weights = weights / max(weights.sum(), 1e-12)
            output.append((ids, weights))
    return output


def truncate_neighbors(neighbors, count):
    result = []
    for ids, weights in neighbors:
        ids, weights = ids[:count], weights[:count]
        result.append((ids, weights / max(weights.sum(), 1e-12)))
    return result


def neighbor_code_votes(target_ids, neighbors):
    result = []
    for ids, weights in neighbors:
        votes = defaultdict(float)
        for reference, weight in zip(ids, weights):
            for token in set(target_ids[reference].tolist()):
                if token > 0:
                    votes[int(token)] += float(weight)
        result.append(dict(votes))
    return result


def mix_votes(left, right):
    result = []
    for first, second in zip(left, right):
        votes = {token: 0.5 * value for token, value in first.items()}
        for token, value in second.items():
            votes[token] = votes.get(token, 0.0) + 0.5 * value
        result.append(votes)
    return result


def hybrid_candidates(previous_ids, left_votes, right_votes, budget=128):
    if budget < previous_ids.shape[1]:
        raise ValueError("Budget must accommodate the previous claim")
    result = np.zeros((len(previous_ids), budget), dtype=np.int64)
    for row, (previous, left, right) in enumerate(zip(previous_ids, left_votes, right_votes)):
        selected = sorted(set(int(token) for token in previous if token > 0))
        seen = set(selected)
        ranked = [sorted(votes, key=lambda token: (-votes[token], token)) for votes in (left, right)]
        for rank in range(max(map(len, ranked), default=0)):
            for source in ranked:
                if rank < len(source) and source[rank] not in seen and len(selected) < budget:
                    selected.append(source[rank])
                    seen.add(source[rank])
        result[row, :len(selected)] = selected
    return result


def match_candidate_counts(proposed, reference):
    """Match each row's actual non-padding budget, with reference backfill."""
    if proposed.shape != reference.shape:
        raise ValueError('Candidate slot shapes must match')
    result = np.zeros_like(reference)
    for row, (candidate, baseline) in enumerate(zip(proposed, reference)):
        count = int(np.count_nonzero(baseline))
        ordered = list(dict.fromkeys(int(token) for token in np.r_[candidate, baseline] if token > 0))
        result[row, :count] = ordered[:count]
    return result


def shuffle_older_context(cpt, icd, ttnc, *, future_claim_k=0, generator):
    """Shuffle claim triples before the protected latest context/future suffix."""
    output = [value.clone() for value in (cpt, icd, ttnc)]
    for row in range(len(ttnc)):
        valid = ttnc[row].ne(0).nonzero().flatten()
        future = min(future_claim_k + 1, max(len(valid) - 1, 1))
        older = valid[:max(len(valid) - future - 1, 0)]
        if len(older) > 1:
            permutation = older[torch.randperm(len(older), generator=generator)]
            for original, shuffled in zip((cpt, icd, ttnc), output):
                shuffled[row, older] = original[row, permutation]
    return tuple(output)


class PermutedHistoryLoader:
    def __init__(self, loader, future_claim_k, seed):
        self.loader, self.future_claim_k, self.seed = loader, future_claim_k, seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for cpt, icd, ttnc, target in self.loader:
            yield (*shuffle_older_context(cpt, icd, ttnc, future_claim_k=self.future_claim_k,
                generator=generator), target)
