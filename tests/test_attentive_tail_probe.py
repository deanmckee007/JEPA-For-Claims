import torch

from scripts.run_attentive_tail_probe import FrozenAttentiveTailProbe


def test_attentive_tail_probe_handles_empty_masks_and_raw_features():
    model = FrozenAttentiveTailProbe(8, raw_dim=3, num_heads=4)
    states = torch.randn(4, 5, 8)
    masks = torch.tensor(
        [[False] * 5, [True, True, False, False, False], [True] * 5, [True] * 5]
    )
    raw = torch.randn(4, 3)
    logits = model(states, masks, raw)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_attentive_tail_probe_selects_compatible_head_count():
    model = FrozenAttentiveTailProbe(10, num_heads=4)
    assert model.num_heads == 2
