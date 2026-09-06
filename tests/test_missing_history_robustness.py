import torch
import pytest

from scripts.run_missing_history_robustness import drop_context_claims


@pytest.mark.parametrize("mode", ["contiguous", "latest", "cpt", "icd"])
def test_structured_corruption_preserves_future(mode):
    cpt = torch.arange(2, 8).reshape(1, 6, 1)
    icd = cpt + 10
    ttnc = torch.arange(2, 8).reshape(1, 6)
    out = drop_context_claims(cpt, icd, ttnc, drop_ratio=0.99,
        future_claim_k=1, generator=torch.Generator().manual_seed(2), corruption=mode)
    for original, changed in zip((cpt, icd, ttnc), out[:3]):
        assert torch.equal(original[:, -2:], changed[:, -2:])
    if mode in {"cpt", "icd"}:
        assert torch.equal(out[2], ttnc)
        assert torch.equal(out[1 if mode == "cpt" else 0], icd if mode == "cpt" else cpt)
    else:
        assert out[2][:, :-2].ne(0).sum() == 1
    if mode == "latest":
        assert out[2][0, 3] == 0


def test_latest_keeps_sole_context_to_preserve_boundary():
    tokens = torch.tensor([[[2], [3], [4]]])
    timing = tokens.squeeze(-1)
    out = drop_context_claims(tokens, tokens, timing, drop_ratio=0.9,
        future_claim_k=1, generator=torch.Generator().manual_seed(2), corruption="latest")
    assert torch.equal(out[2], timing)


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
