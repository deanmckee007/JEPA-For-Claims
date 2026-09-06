"""Choose bounded claim sets with copy/add thresholds on an inner split only."""
import numpy as np


DEFAULT_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, -2.0)
DEFAULT_COUNT_SCALES = (None, 1.0, 0.5, 1.5, 2.0)


def shortlist_logits(logits, previous_ids, max_tokens):
    """Keep top max_tokens from each group; enough for every bounded decode.

    A group-specific threshold preserves order within each group. Thus no token
    outside its group's top max_tokens can enter the final max_tokens-sized set.
    PAD and candidates masked to -30 by retrieval are ineligible.
    """
    logits = np.asarray(logits)
    width = min(max_tokens, logits.shape[1] - 1)
    if width < 1:
        raise ValueError("Need a positive token budget and non-padding vocabulary")
    rows = np.arange(len(logits))[:, None]
    copied = np.zeros(logits.shape, dtype=bool)
    copied[rows, previous_ids] = True
    copied[:, 0] = False
    available = logits > -29.0
    available[:, 0] = False
    ids, values, groups = [], [], []
    for group in (True, False):
        scores = np.where(available & (copied == group), logits, -np.inf)
        top = np.argpartition(-scores, width - 1, axis=1)[:, :width]
        ids.append(top)
        values.append(np.take_along_axis(scores, top, axis=1))
        groups.append(np.full(top.shape, group, dtype=bool))
    return {"ids": np.concatenate(ids, 1), "logits": np.concatenate(values, 1),
        "copied": np.concatenate(groups, 1), "max_tokens": int(max_tokens)}


def ordered_margins(shortlist, copy_threshold, add_threshold, return_order=False):
    margins = shortlist["logits"] - np.where(shortlist["copied"], copy_threshold, add_threshold)
    # Stable token-ID tie break, independent of argpartition's arbitrary order.
    order = np.lexsort((shortlist["ids"], -margins), axis=1)
    result = (np.take_along_axis(shortlist["ids"], order, 1),
        np.take_along_axis(margins, order, 1))
    return (*result, order) if return_order else result


def selection_mask(margins, predicted_counts, max_tokens, count_scale):
    if count_scale is None:
        counts = np.full(len(margins), max_tokens, dtype=np.int64)
    else:
        counts = np.clip(np.rint(np.asarray(predicted_counts) * count_scale), 0, max_tokens).astype(np.int64)
    return (margins > 0) & (np.arange(margins.shape[1])[None, :] < counts[:, None])


def fit_set_calibration(shortlist, predicted_counts, target_ids,
        thresholds=DEFAULT_THRESHOLDS, count_scales=DEFAULT_COUNT_SCALES,
        objective="micro_f1", previous_ids=None):
    """Maximize micro F1 on supplied inner-calibration labels, never validation."""
    truth = [set(int(t) for t in row if t > 0) for row in target_ids]
    total = sum(map(len, truth))
    if objective not in {"micro_f1", "balanced_new_f1"}:
        raise ValueError(f"Unknown calibration objective: {objective}")
    if objective == "balanced_new_f1" and previous_ids is None:
        raise ValueError("New-code calibration requires previous-claim IDs")
    new_total = sum(len({t for t in actual - set(previous) if t > 1})
        for actual, previous in zip(truth, previous_ids)) if previous_ids is not None else 0
    base_hits = np.asarray([[int(t) in row_truth for t in row]
        for row, row_truth in zip(shortlist["ids"], truth)])
    best = None
    for copy_threshold in thresholds:
        for add_threshold in thresholds:
            ids, margins, order = ordered_margins(shortlist, copy_threshold, add_threshold, return_order=True)
            hits = np.take_along_axis(base_hits, order, 1)
            new = (~np.take_along_axis(shortlist["copied"], order, 1)) & (ids > 1)
            for count_scale in count_scales:
                selected = selection_mask(margins, predicted_counts, shortlist["max_tokens"], count_scale)
                tp = int((hits & selected).sum())
                predicted = int(selected.sum())
                f1 = 2.0 * tp / max(total + predicted, 1)
                new_f1 = 2.0 * int((hits & selected & new).sum()) / max(new_total + int((selected & new).sum()), 1)
                score = f1 if objective == "micro_f1" else 0.5 * (f1 + new_f1)
                if best is None or score > best["inner_objective_score"]:
                    best = {"copy_logit_threshold": float(copy_threshold),
                        "add_logit_threshold": float(add_threshold), "count_scale": count_scale,
                        "inner_micro_f1": float(f1), "objective": objective,
                        "inner_objective_score": float(score)}
                    if previous_ids is not None:
                        best["inner_new_code_f1"] = float(new_f1)
    return best


def decode_calibrated_sets(shortlist, predicted_counts, calibration):
    ids, margins = ordered_margins(shortlist, calibration["copy_logit_threshold"], calibration["add_logit_threshold"])
    selected = selection_mask(margins, predicted_counts, shortlist["max_tokens"], calibration["count_scale"])
    return [row[mask].tolist() for row, mask in zip(ids, selected)]


def score_decoded_sets(predicted_ids, target_ids, previous_ids):
    tp = predicted_count = truth_count = add_tp = add_pred = add_true = 0
    exact, jaccard, count_errors = [], [], []
    for predicted, target, previous in zip(predicted_ids, target_ids, previous_ids):
        prediction = set(int(t) for t in predicted if t > 0)
        truth = set(int(t) for t in target if t > 0)
        old = set(int(t) for t in previous if t > 0)
        tp += len(prediction & truth)
        predicted_count += len(prediction)
        truth_count += len(truth)
        # Actual new code identities exclude both reserved IDs, including UNK.
        new_prediction = {t for t in prediction - old if t > 1}
        new_truth = {t for t in truth - old if t > 1}
        add_tp += len(new_prediction & new_truth)
        add_pred += len(new_prediction)
        add_true += len(new_truth)
        exact.append(prediction == truth)
        jaccard.append(len(prediction & truth) / max(len(prediction | truth), 1))
        count_errors.append(abs(len(prediction) - len(truth)))
    return {"micro_f1": 2 * tp / max(predicted_count + truth_count, 1),
        "micro_precision": tp / max(predicted_count, 1), "micro_recall": tp / max(truth_count, 1),
        "exact_set_match": float(np.mean(exact)), "mean_set_jaccard": float(np.mean(jaccard)),
        "cardinality_mae": float(np.mean(count_errors)),
        "new_code_precision": add_tp / max(add_pred, 1), "new_code_recall": add_tp / max(add_true, 1),
        "new_code_f1": 2 * add_tp / max(add_pred + add_true, 1)}


def candidate_coverage(candidates, target_ids, previous_ids):
    counts = {key: [0, 0] for key in ("all_known", "persistent", "new")}
    for candidate, target, previous in zip(candidates, target_ids, previous_ids):
        truth = {int(t) for t in target if t > 1}
        old = {int(t) for t in previous if t > 1}
        proposed = set(candidate)
        for key, actual in [("all_known", truth), ("persistent", truth & old), ("new", truth - old)]:
            counts[key][0] += len(actual & proposed)
            counts[key][1] += len(actual)
    return {key: {"covered": hit, "total": total, "recall": hit / total if total else None}
        for key, (hit, total) in counts.items()}
