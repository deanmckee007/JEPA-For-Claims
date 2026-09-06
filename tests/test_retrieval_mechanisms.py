import numpy as np
import torch

from jepa_utils.retrieval_mechanisms import (
    neighbor_code_votes, hybrid_candidates, weighted_neighbors, shuffle_older_context,
    match_candidate_counts,
)
from scripts.run_retrieval_generation_probe import RetentionAdditionDecoder, split_membership_loss
from scripts.run_mechanism_probes import vote_prediction
from jepa_models.prediction_blocks import Level2PredictionBlock


def test_votes_weight_distinct_reference_tokens_and_ignore_padding():
    reference = np.asarray([[2, 2, 0], [3, 4, 0]])
    votes = neighbor_code_votes(reference, [(np.asarray([0, 1]), np.asarray([0.25, 0.75]))])
    assert votes == [{2: 0.25, 3: 0.75, 4: 0.75}]


def test_hybrid_budget_preserves_copy_and_diverse_sources():
    result = hybrid_candidates(np.asarray([[9, 0]]), [{2: 1., 3: .5}], [{4: 1., 5: .5}], 4)
    assert result.tolist() == [[9, 2, 4, 3]]


def test_neighbor_training_excludes_self_and_duplicates():
    features = np.asarray([[1., 0.], [1., 0.], [0., 1.]])
    result = weighted_neighbors(features, features, 2, training=True)
    assert result[0][0].tolist() == [2]
    assert np.isclose(result[0][1].sum(), 1.)


def test_count_matching_trims_and_backfills_without_duplicates():
    proposed = np.asarray([[9, 2, 4, 5, 0], [9, 2, 0, 0, 0]])
    baseline = np.asarray([[9, 3, 0, 0, 0], [9, 2, 3, 4, 0]])
    matched = match_candidate_counts(proposed, baseline)
    assert matched.tolist() == [[9, 2, 0, 0, 0], [9, 2, 3, 4, 0]]


def test_permutation_preserves_latest_future_padding_and_joint_claims():
    timing = torch.tensor([[0, 2, 3, 4, 5, 6, 7]])
    cpt, icd = timing.unsqueeze(-1) * 10, timing.unsqueeze(-1) * 20
    out = shuffle_older_context(cpt, icd, timing, future_claim_k=1, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out[2][:, -3:], timing[:, -3:])
    assert torch.equal(out[2][:, :1], timing[:, :1])
    assert sorted(out[2][0].tolist()) == sorted(timing[0].tolist())
    assert torch.equal(out[0], out[2].unsqueeze(-1) * 10)
    assert torch.equal(out[1], out[2].unsqueeze(-1) * 20)


def test_retention_and_addition_receive_disjoint_token_gradients():
    model = RetentionAdditionDecoder(4, 8, [10, 10, 5])
    ids = [torch.tensor([[2, 3, 0]])] * 2
    copied = [torch.tensor([[1., 0., 0.]])] * 2
    logits, _, _ = model(torch.ones(1, 4), ids, copied)
    loss = split_membership_loss(logits[0], torch.tensor([[1., 0., 0.]]), ids[0], copied[0])
    loss.backward()
    assert model.retention_tokens[0].weight.grad[2].abs().sum() > 0
    assert model.retention_tokens[0].weight.grad[3].abs().sum() == 0
    assert model.tokens[0].weight.grad[2].abs().sum() == 0
    assert model.tokens[0].weight.grad[3].abs().sum() > 0
    assert torch.isfinite(loss)


def test_deepsets_is_permutation_invariant_with_padding():
    model = Level2PredictionBlock(8, 8, 10, 10, 10, 6, num_layers=1,
        num_heads=2, dropout=0, rnn_type='deepsets').eval()
    context = torch.randn(2, 6, 8)
    timing = torch.tensor([[0, 2, 3, 4, 5, 6], [0, 0, 2, 3, 4, 5]])
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        original = model(context, timing)
        reordered = model(context[:, permutation], timing[:, permutation])
    for left, right in zip(original, reordered):
        assert torch.allclose(left, right, atol=1e-6)


def test_vote_copy_weight_one_uses_only_previous_claim():
    data = {'ttnc': np.asarray([2]), 'previous_cpt_ids': np.asarray([[2]]),
        'previous_icd_ids': np.asarray([[3]]), 'previous_ttnc': np.asarray([2])}
    prediction = vote_prediction([[{4: 1.}], [{4: 1.}]], [{4: 1.}], data, [5, 5, 5], [2, 2], 1.)
    assert prediction['cpt_logits'][0].argmax() == 2
    assert prediction['icd_logits'][0].argmax() == 3
    assert prediction['ttnc_logits'][0].argmax() == 2
