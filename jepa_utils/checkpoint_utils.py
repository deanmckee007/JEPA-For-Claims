import torch


def checkpoint_has_prefixed_keys(path: str) -> bool:
    """Return True if the checkpoint contains keys starting with 'diffusion_model.'."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    return any(k.startswith("diffusion_model.") for k in state_dict.keys())


def freeze_non_diffusion(model):
    """Freeze all parameters except those belonging to the diffusion model."""
    for name, param in model.named_parameters():
        if not name.startswith("diffusion_model"):
            param.requires_grad = False

def freeze_jepa(model):
    """Freeze JEPA parameters (everything except diffusion)."""
    for name, param in model.named_parameters():
        if name.startswith("diffusion_model"):
            continue
        param.requires_grad = False
