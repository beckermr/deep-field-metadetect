from functools import partial

import jax
import jax_galsim


@partial(jax.jit, static_argnames=["fft_size"])
def jax_compute_mfrac_interp_image(mfrac, wcs, fwhm=1.2, fft_size=256):
    """Compute the Gaussian-convolved masked fraction map for an image using jax_galsim.

    This is the JAX version of the original galsim-based function.

    Parameters
    ----------
    mfrac : array-like
        The masked fraction image.
    wcs : jax_galsim.BaseWCS
        The WCS of the image.
    fwhm : float, optional
        The FWHM of the Gaussian to convolve the mask, by default 1.2 arcseconds.

    Returns
    -------
    interp_image : jax_galsim.InterpolatedImage
        The interpolated masked fraction map.
    """
    _gsimage_orig = jax_galsim.ImageD(mfrac, wcs=wcs)
    _gsimage_interp = jax_galsim.InterpolatedImage(
        _gsimage_orig,
        normalization="sb",
    )
    _gsimage = jax_galsim.Convolve(_gsimage_interp, jax_galsim.Gaussian(fwhm=fwhm))
    _gsimage = _gsimage.withGSParams(
        minimum_fft_size=fft_size,
        maximum_fft_size=fft_size,
    )
    _gsimage = _gsimage.drawImage(
        image=_gsimage_orig.copy(),
        method="sb",
    )
    _gsimage.wcs = None
    _gsimage.scale = 1.0
    _gsimage_final = jax_galsim.InterpolatedImage(
        _gsimage,
        normalization="sb",
        offset=(_gsimage_interp.image.xmin, _gsimage_interp.image.ymin),
        use_true_center=False,
    )

    return _gsimage_final
