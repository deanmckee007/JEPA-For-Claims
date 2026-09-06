import numpy as np
import torch

from scripts.run_retrieval_generation_probe import (
    build_candidates, retrieve_neighbors, candidate_recall, CandidateDecoder,
)


def test_retrieval_excludes_self_and_duplicate_context():
    features = np.asarray([[1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
    neighbors = retrieve_neighbors(features, features, 2, training=True)
    assert neighbors[0].tolist() == [2]
    assert neighbors[1].tolist() == [2]
    assert 2 not in neighbors[2]


def test_candidates_preserve_copy_and_only_use_references():
    previous = np.asarray([[7, 0]])
    references = np.asarray([[2, 3], [4, 5]])
    candidates = build_candidates(previous, references, [np.asarray([0])], 3)
    assert candidates.tolist() == [[7, 2, 3]]
    assert candidate_recall(candidates, np.asarray([[7, 4]])) == 0.5


def test_candidate_head_learns_with_sparse_candidate_slots():
    model = CandidateDecoder(4, 8, [10, 12, 5])
    candidates = [torch.tensor([[2, 3, 0]]), torch.tensor([[4, 0, 0]])]
    output, timing, counts = model(torch.randn(1, 4), candidates, [torch.zeros(1, 3)] * 2)
    sum(value.sum() for value in [*output, timing, counts]).backward()
    assert model.tokens[0].weight.grad[2].abs().sum() > 0
    assert model.tokens[0].weight.grad[8].abs().sum() == 0
