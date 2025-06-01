import warnings
from models.diffusion import ClaimD3PM

warnings.warn(
    "Importing ClaimD3PM from jepa_models.diffusion_models is deprecated; use models.diffusion.claim_d3pm instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ClaimD3PM"]
