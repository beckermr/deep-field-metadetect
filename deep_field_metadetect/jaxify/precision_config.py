"""Precision configuration for mixed precision support."""

import jax
import jax.numpy as jnp

# Module-level dtype variables
# Default: MIXED PRECISION (float32 for images, float64 for moments)
IMAGE_DTYPE = jnp.float32
WEIGHT_DTYPE = jnp.float32
NOISE_DTYPE = jnp.float32
PSF_DTYPE = jnp.float32
BMASK_DTYPE = jnp.int32
MFRAC_DTYPE = jnp.float32
MOMENT_DTYPE = jnp.float64
COORD_DTYPE = jnp.float64  # Not yet used, kept for RA and DEC later
_CURRENT_MODE = "mixed"


def use_mixed_precision(enabled: bool = True):
    """Enable or disable mixed precision.

    Parameters
    ----------
    enabled : bool
        If True, use float32 for images and float64 for moments (DEFAULT).
        If False, use float64 everywhere.
    """
    global IMAGE_DTYPE, WEIGHT_DTYPE, NOISE_DTYPE, PSF_DTYPE
    global MFRAC_DTYPE, MOMENT_DTYPE, COORD_DTYPE, _CURRENT_MODE

    if enabled:
        IMAGE_DTYPE = jnp.float32
        WEIGHT_DTYPE = jnp.float32
        NOISE_DTYPE = jnp.float32
        PSF_DTYPE = jnp.float32
        MFRAC_DTYPE = jnp.float32
        MOMENT_DTYPE = jnp.float64
        COORD_DTYPE = jnp.float64
        _CURRENT_MODE = "mixed"
    else:
        # Full precision: float64 everywhere
        IMAGE_DTYPE = jnp.float64
        WEIGHT_DTYPE = jnp.float64
        NOISE_DTYPE = jnp.float64
        PSF_DTYPE = jnp.float64
        MFRAC_DTYPE = jnp.float64
        MOMENT_DTYPE = jnp.float64
        COORD_DTYPE = jnp.float64
        _CURRENT_MODE = "full"

    # Clear JAX compilation cache to force recompilation with new dtypes
    jax.clear_caches()


def get_precision_summary() -> dict[str, str]:
    """Return summary of current precision settings.

    Returns
    -------
    summary : dict
        Dictionary with current precision mode and dtype settings
    """
    return {
        "mode": _CURRENT_MODE,
        "image_dtype": str(IMAGE_DTYPE),
        "weight_dtype": str(WEIGHT_DTYPE),
        "noise_dtype": str(NOISE_DTYPE),
        "psf_dtype": str(PSF_DTYPE),
        "bmask_dtype": str(BMASK_DTYPE),
        "mfrac_dtype": str(MFRAC_DTYPE),
        "moment_dtype": str(MOMENT_DTYPE),
        "coord_dtype": str(COORD_DTYPE),
    }
