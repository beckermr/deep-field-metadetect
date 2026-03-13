import galsim


def compute_mfrac_interp_image(mfrac, wcs, fwhm=1.2):
    """Compute the Gaussian-convolved masked fraction map for an image.

    Parameters
    ----------
    mfrac : list of array-like
        The masked fraction image.
    wcs : galsim.BaseWCS
        The WCS of the image.
    fwhm : float, optional
        The FWHM of the Gaussian to convolve the mask, by default 1.2 arcseconds.

    Returns
    -------
    interp_image : galsim.Image
        The interpolated masked fraction map.
    """
    _gsimage_orig = galsim.ImageD(mfrac, wcs=wcs)
    _gsimage_interp = galsim.InterpolatedImage(
        _gsimage_orig,
        normalization="sb",
    )
    _gsimage = galsim.Convolve(_gsimage_interp, galsim.Gaussian(fwhm=fwhm))
    _gsimage = _gsimage_interp.drawImage(
        image=_gsimage_orig.copy(),
        method="sb",
    )
    _gsimage.wcs = None
    _gsimage.scale = 1.0
    _gsimage_final = galsim.InterpolatedImage(
        _gsimage,
        normalization="sb",
        offset=(_gsimage_interp.image.xmin, _gsimage_interp.image.ymin),
        use_true_center=False,
    )

    return _gsimage_final
