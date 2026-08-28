import torch

from scripts.run_missing_history_robustness import drop_context_claims


def test_zero_drop_is_identity():
    cpt = torch.tensor([[[0], [2], [3], [4]]])
    icd = cpt + (cpt != 0)
    ttnc = torch.tensor([[0, 1, 2, 3]])
    output = drop_context_claims(
        cpt, icd, ttnc, drop_ratio=0.0, future_claim_k=0,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(output[0], cpt)
    assert torch.equal(output[1], icd)
    assert torch.equal(output[2], ttnc)
    assert output[3] == 1.0


def test_drop_preserves_latest_context_and_future_suffix():
    cpt = torch.tensor([[[0], [2], [3], [4], [5]]])
    icd = torch.tensor([[[0], [12], [13], [14], [15]]])
    ttnc = torch.tensor([[0, 1, 2, 3, 4]])
    out_cpt, out_icd, out_ttnc, retention = drop_context_claims(
        cpt, icd, ttnc, drop_ratio=0.999, future_claim_k=1,
        generator=torch.Generator().manual_seed(2),
    )
    # With two held-out future slots, index 2 is the latest context claim.
    assert out_ttnc.tolist() == [[0, 0, 2, 3, 4]]
    assert out_cpt.tolist() == [[[0], [0], [3], [4], [5]]]
    assert out_icd.tolist() == [[[0], [0], [13], [14], [15]]]
    assert retention == 0.5
