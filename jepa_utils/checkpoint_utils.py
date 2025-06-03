import torch
from .freezing import freeze_non_diffusion, freeze_jepa


def checkpoint_has_prefixed_keys(path: str) -> bool:
    """Return True if the checkpoint contains keys starting with 'diffusion_model.'."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    return any(k.startswith("diffusion_model.") for k in state_dict.keys())
