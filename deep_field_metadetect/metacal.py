import galsim
import ngmix
import numpy as np

DEFAULT_SHEARS = ("noshear", "1p", "1m", "2p", "2m")
DEFAULT_STEP = 0.01


def get_shear_tuple(shear, step):
    if shear == "noshear":
        return (0, 0)
    elif shear == "1p":
        return (step, 0)
    elif shear == "1m":
        return (-step, 0)
    elif shear == "2p":
        return (0, step)
    elif shear == "2m":
        return (0, -step)
    else:
        raise RuntimeError("Shear value '%s' not regonized!" % shear)


def get_gauss_reconv_psf_galsim(psf, step=DEFAULT_STEP, flux=1):
    """Gets the target reconvolution PSF for an input PSF object.

    This is taken from galsim/tests/test_metacal.py and assumes the psf is
    centered.

    Parameters
    ----------
    psf : galsim object
        The PSF.
    flux : float
        The output flux of the PSF. Defaults to 1.

    Returns
    -------
    reconv_psf : galsim object
        The reconvolution PSF.
    """
    dk = 2 * np.pi / (53 * 0.2) / 4.0

    small_kval = 1.0e-2  # Find the k where the given psf hits this kvalue
    smaller_kval = 3.0e-3  # Target PSF will have this kvalue at the same k

    kim = psf.drawKImage(scale=dk)
    karr_r = kim.real.array
    # Find the smallest r where the kval < small_kval
    nk = karr_r.shape[0]
    kx, ky = np.meshgrid(np.arange(-nk / 2, nk / 2), np.arange(-nk / 2, nk / 2))
    ksq = (kx**2 + ky**2) * dk**2
    ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

    # We take our target PSF to be the (round) Gaussian that is even smaller at
    # this ksq
    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2.0 * np.log(smaller_kval) / ksq_max

    dilation = 1.0 + 2.0 * step
    return galsim.Gaussian(sigma=np.sqrt(sigma_sq) * dilation).withFlux(flux)


def get_gauss_reconv_psf(obs, step=DEFAULT_STEP):
    """Get the Gaussian reconv PSF for an ngmix obs."""
    psf = get_galsim_object_from_ngmix_obs_nopix(obs.psf, kind="image")
    return get_gauss_reconv_psf_galsim(psf, step=step)


def get_max_gauss_reconv_psf_galsim(psf_w, psf_d, step=DEFAULT_STEP):
    """Get the larger of two Gaussian reconvolution PSFs for two galsim objects."""
    mc_psf_w = get_gauss_reconv_psf_galsim(psf_w, step=step)
    mc_psf_d = get_gauss_reconv_psf_galsim(psf_d, step=step)
    if mc_psf_w.fwhm > mc_psf_d.fwhm:
        return mc_psf_w
    else:
        return mc_psf_d


def get_max_gauss_reconv_psf(obs_w, obs_d, step=DEFAULT_STEP):
    """Get the larger of two reconv PSFs for two ngmix.Observations."""
    psf_w = get_galsim_object_from_ngmix_obs_nopix(obs_w.psf, kind="image")
    psf_d = get_galsim_object_from_ngmix_obs_nopix(obs_d.psf, kind="image")
    return get_max_gauss_reconv_psf_galsim(psf_w, psf_d, step=step)


def _render_psf_and_build_obs(
    image, obs, reconv_psf, weight_fac=1, max_min_fft_size=None
):
    if max_min_fft_size is not None:
        reconv_psf = reconv_psf.withGSParams(
            minimum_fft_size=max_min_fft_size,
            maximum_fft_size=max_min_fft_size,
        )

    pim = reconv_psf.drawImage(
        nx=obs.psf.image.shape[1],
        ny=obs.psf.image.shape[0],
        wcs=obs.psf.jacobian.get_galsim_wcs(),
        center=galsim.PositionD(
            x=obs.psf.jacobian.get_col0() + 1,
            y=obs.psf.jacobian.get_row0() + 1,
        ),
    ).array
    psf_obs = obs.psf.copy()
    psf_obs.image = pim
    obs = obs.copy()
    obs.image = image
    obs.psf = psf_obs
    obs.weight = obs.weight * weight_fac
    return obs


def _metacal_op_g1g2_impl(*, wcs, image, noise, psf_inv, dims, reconv_psf, g1, g2):
    """Run metacal on an ngmix observation.

    Note that the noise image should already be rotated by 90 degrees here.
    """

    ims = galsim.Convolve(
        [
            galsim.Convolve([image, psf_inv]).shear(g1=g1, g2=g2),
            reconv_psf,
        ]
    )

    ns = galsim.Convolve(
        [
            galsim.Convolve([noise, psf_inv]).shear(g1=g1, g2=g2),
            reconv_psf,
        ]
    )

    ims = ims.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array
    ns = np.rot90(
        ns.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array,
        k=-1,
    )
    return ims + ns


def metacal_op_g1g2(obs, reconv_psf, g1, g2, max_min_fft_size=None):
    """Run metacal on an ngmix observation."""
    mcal_image = _metacal_op_g1g2_impl(
        wcs=obs.jacobian.get_galsim_wcs(),
        image=get_galsim_object_from_ngmix_obs(obs, kind="image"),
        # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
        # rotates back after deconv and shearing
        noise=get_galsim_object_from_ngmix_obs(obs, kind="noise", rot90=1),
        psf_inv=galsim.Deconvolve(
            get_galsim_object_from_ngmix_obs(obs.psf, kind="image")
        ),
        dims=obs.image.shape,
        reconv_psf=reconv_psf,
        g1=g1,
        g2=g2,
    )
    return _render_psf_and_build_obs(
        mcal_image, obs, reconv_psf, weight_fac=0.5, max_min_fft_size=max_min_fft_size
    )


def metacal_op_shears(
    obs, reconv_psf=None, shears=None, step=DEFAULT_STEP, max_min_fft_size=None
):
    """Run metacal on an ngmix observation."""
    if shears is None:
        shears = DEFAULT_SHEARS

    if reconv_psf is None:
        reconv_psf = get_gauss_reconv_psf(obs, step=step)

    wcs = obs.jacobian.get_galsim_wcs()
    image = get_galsim_object_from_ngmix_obs(obs, kind="image")
    # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
    # rotates back after deconv and shearing
    noise = get_galsim_object_from_ngmix_obs(obs, kind="noise", rot90=1)
    psf = get_galsim_object_from_ngmix_obs(obs.psf, kind="image")
    psf_inv = galsim.Deconvolve(psf)

    mcal_res = {}
    for shear in shears:
        g1, g2 = get_shear_tuple(shear, step)
        mcal_image = _metacal_op_g1g2_impl(
            wcs=wcs,
            image=image,
            noise=noise,
            psf_inv=psf_inv,
            dims=obs.image.shape,
            reconv_psf=reconv_psf,
            g1=g1,
            g2=g2,
        )
        mcal_res[shear] = _render_psf_and_build_obs(
            mcal_image,
            obs,
            reconv_psf,
            weight_fac=0.5,
            max_min_fft_size=max_min_fft_size,
        )
    return mcal_res


def match_psf(
    obs,
    reconv_psf,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    max_min_fft_size=None,
):
    """Match the PSF on an ngmix observation to a new PSF."""
    wcs = obs.jacobian.get_galsim_wcs()
    image = get_galsim_object_from_ngmix_obs(
        obs,
        kind="image",
        _force_stepk=force_stepk_field,
        _force_maxk=force_maxk_field,
    )

    psf = get_galsim_object_from_ngmix_obs(
        obs.psf,
        kind="image",
        _force_stepk=force_stepk_psf,
        _force_maxk=force_maxk_psf,
    )

    if max_min_fft_size is None:
        ims = galsim.Convolve(
            [image, galsim.Deconvolve(psf), reconv_psf],
        )

    else:
        ims = galsim.Convolve(
            [image, galsim.Deconvolve(psf), reconv_psf],
            gsparams=galsim.GSParams(
                minimum_fft_size=max_min_fft_size, maximum_fft_size=max_min_fft_size
            ),
        )
        ims = ims.withGSParams(
            minimum_fft_size=max_min_fft_size,
            maximum_fft_size=max_min_fft_size,
        )

    ims = ims.drawImage(nx=obs.image.shape[1], ny=obs.image.shape[0], wcs=wcs).array
    if return_k_info:
        return _render_psf_and_build_obs(
            ims, obs, reconv_psf, weight_fac=1, max_min_fft_size=max_min_fft_size
        ), (
            image._stepk,
            image._maxk,
            psf._stepk,
            psf._maxk,
        )

    return (
        _render_psf_and_build_obs(
            ims, obs, reconv_psf, weight_fac=1, max_min_fft_size=max_min_fft_size
        ),
        None,
    )


def _extract_attr(obs, attr, dtype):
    if getattr(obs, "has_" + attr)():
        return getattr(obs, attr)
    else:
        return np.zeros_like(obs.image, dtype=dtype)


def add_ngmix_obs(obs1, obs2, ignore_psf=False, skip_mfrac_for_second=False):
    """Add two ngmix observations"""

    if repr(obs1.jacobian) != repr(obs2.jacobian):
        raise RuntimeError(
            "Jacobians must be equal to add ngmix observations! %s != %s"
            % (repr(obs1.jacobian), repr(obs2.jacobian)),
        )

    if obs1.image.shape != obs2.image.shape:
        raise RuntimeError(
            "Image shapes must be equal to add ngmix observations! %s != %s"
            % (
                obs1.image.shape,
                obs2.image.shape,
            ),
        )

    if obs1.has_psf() != obs2.has_psf() and not ignore_psf:
        raise RuntimeError(
            "Observations must both either have or not have a "
            "PSF to add them. %s != %s"
            % (
                obs1.has_psf(),
                obs2.has_psf(),
            ),
        )

    if obs1.has_psf() and obs2.has_psf() and not ignore_psf:
        # We ignore the PSF in this call since PSFs do not have PSFs
        new_psf = add_ngmix_obs(obs1.psf, obs2.psf, ignore_psf=True)
    else:
        new_psf = None

    msk = (obs1.weight > 0) & (obs2.weight > 0)
    new_wgt = np.zeros_like(obs1.weight)
    new_wgt[msk] = 1 / (1 / obs1.weight[msk] + 1 / obs2.weight[msk])
    obs = ngmix.Observation(
        image=obs1.image + obs2.image,
        weight=new_wgt,
        psf=new_psf,
        jacobian=obs1.jacobian,  # makes a copy
    )

    if obs1.has_bmask() or obs2.has_bmask():
        obs.bmask = _extract_attr(obs1, "bmask", np.int32) | _extract_attr(
            obs2, "bmask", np.int32
        )

    if obs1.has_ormask() or obs2.has_ormask():
        obs.ormask = _extract_attr(obs1, "ormask", np.int32) | _extract_attr(
            obs2, "ormask", np.int32
        )

    if obs1.has_noise() or obs2.has_noise():
        obs.noise = _extract_attr(obs1, "noise", np.float32) + _extract_attr(
            obs2, "noise", np.float32
        )

    if skip_mfrac_for_second:
        if obs1.has_mfrac():
            obs.mfrac = _extract_attr(obs1, "mfrac", np.float32)
    else:
        if obs1.has_mfrac() or obs2.has_mfrac():
            obs.mfrac = (
                _extract_attr(obs1, "mfrac", np.float32)
                + _extract_attr(obs2, "mfrac", np.float32)
            ) / 2

    obs.update_meta_data(obs1.meta)
    obs.update_meta_data(obs2.meta)

    return obs


def get_galsim_object_from_ngmix_obs(
    obs, kind="image", rot90=0, _force_stepk=0.0, _force_maxk=0.0
):
    """Make an interpolated image from an ngmix obs."""
    return galsim.InterpolatedImage(
        galsim.ImageD(
            np.rot90(getattr(obs, kind).copy(), k=rot90),
            wcs=obs.jacobian.get_galsim_wcs(),
        ),
        x_interpolant="lanczos15",
        _force_stepk=_force_stepk,
        _force_maxk=_force_maxk,
    )


def get_galsim_object_from_ngmix_obs_nopix(obs, kind="image"):
    """Make an interpolated image from an ngmix obs w/o a pixel."""
    wcs = obs.jacobian.get_galsim_wcs()
    return galsim.Convolve(
        [
            get_galsim_object_from_ngmix_obs(obs, kind=kind),
            galsim.Deconvolve(wcs.toWorld(galsim.Pixel(scale=1))),
        ]
    )


def metacal_wide_and_deep_psf_matched(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    shears=None,
    step=DEFAULT_STEP,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_noshear_deep=False,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    max_min_fft_size=None,
):
    """Do metacalibration for a combination of wide+deep datasets.

    Parameters
    ----------
    obs_wide : ngmix.Observation
        The wide-field observation.
    obs_deep : ngmix.Observation
        The deep-field observation.
    obs_deep_noise : ngmix.Observation
        The deep-field noise observation.
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    shears : list, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    nodet_flags : int, optional
        The bmask flags marking area in the image to skip, by default 0.
    return_k_info : bool, optional
        return _force stepk and maxk values in the following order
        _force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf.
        Used mainly for testing.
    force_stepk_field : float, optional
        Force stepk for drawing field images.
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_maxk_field: float, optional
        Force maxk for drawing field images.
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    force_stepk_psf: float, optional
        Force stepk for drawing PSF images.
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    force_maxk_psf: float, optional
        Force stepk for drawing PSF images
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    max_min_fft_size: int, optional
        To fix max and min values of FFT size.
        Defaults to None which lets Galsim determine the values.
        Used mainly to test against JaxGalsim.

    Returns
    -------
    mcal_res : dict
        Output from metacal_op_shears.
    kinfo: tuple, optional
        returns _force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf
        if return_k_into is True, else returns None.
        Used mainly for testing.
    """

    # first get the biggest reconv PSF of the two
    reconv_psf = get_max_gauss_reconv_psf(obs_wide, obs_deep)
    mcal_obs_wide, kinfo = match_psf(
        obs_wide,
        reconv_psf,
        return_k_info=return_k_info,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        max_min_fft_size=max_min_fft_size,
    )
    if return_k_info:
        force_stepk_field, force_maxk_field, force_stepk_psf, force_maxk_psf = kinfo

    if not skip_obs_wide_corrections:
        mcal_obs_wide = add_ngmix_obs(
            mcal_obs_wide,
            metacal_op_g1g2(
                obs_deep_noise, reconv_psf, 0, 0, max_min_fft_size=max_min_fft_size
            ),
            skip_mfrac_for_second=True,
        )

    # get PSF matched noise
    obs_wide_noise = obs_wide.copy()
    obs_wide_noise.image = obs_wide.noise
    wide_noise_corr, _ = match_psf(
        obs_wide_noise,
        reconv_psf,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        return_k_info=False,
        max_min_fft_size=max_min_fft_size,
    )

    # now run mcal on deep
    mcal_res = metacal_op_shears(
        obs_deep,
        reconv_psf=reconv_psf,
        shears=shears,
        step=step,
        max_min_fft_size=max_min_fft_size,
    )

    # now add in noise corr to make it match the wide noise
    if not skip_obs_deep_corrections:
        for k in mcal_res:
            mcal_res[k] = add_ngmix_obs(
                mcal_res[k],
                wide_noise_corr,
                skip_mfrac_for_second=True,
            )

    # we report the wide obs as noshear for later measurements
    noshear_res = mcal_res.pop("noshear")
    mcal_res["noshear"] = mcal_obs_wide
    if return_noshear_deep:
        mcal_res["noshear_deep"] = noshear_res

    for k in mcal_res:
        mcal_res[k].psf.galsim_obj = reconv_psf

    return mcal_res, kinfo
