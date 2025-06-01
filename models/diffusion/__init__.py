import warnings
from diffusion_models import ClaimD3PM as _ClaimD3PM

warnings.warn(
    "Importing ClaimD3PM from models.diffusion is deprecated; use diffusion_models.ClaimD3PM instead.",
    DeprecationWarning,
    stacklevel=2,
)

ClaimD3PM = _ClaimD3PM

__all__ = ["ClaimD3PM"]
