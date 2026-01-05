from functools import partial

import galsim as galsim
import jax
import jax.numpy as jnp
import jax_galsim
import numpy as np

from deep_field_metadetect.jaxify.jax_utils import compute_stepk
from deep_field_metadetect.jaxify.observation import (
    DFMdetObservation,
    dfmd_obs_to_ngmix_obs,
)
from deep_field_metadetect.metacal import DEFAULT_SHEARS, DEFAULT_STEP

DEFAULT_FFT_SIZE = 256


def get_shear_tuple(shear, step):
    """Convert shear string identifier to (g1, g2) tuple.

    Parameters
    ----------
    shear : str
        Shear identifier. Valid values are:
        - "noshear": No shear applied
        - "1p": Positive shear in g1 direction
        - "1m": Negative shear in g1 direction
        - "2p": Positive shear in g2 direction
        - "2m": Negative shear in g2 direction
    step : float
        Magnitude of the shear step to apply.
        Defaults to DEFAULT_STEP.

    Returns
    -------
    tuple of float
        Two-element tuple (g1, g2) representing the shear components.
    """
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
        raise RuntimeError("Shear value '%s' not recognized!" % shear)


@partial(jax.jit, static_argnames=["dk", "nxy_psf", "kim_size"])
def jax_get_gauss_reconv_psf_galsim(
    psf, dk, nxy_psf=53, step=DEFAULT_STEP, flux=1.0, kim_size=None
):
    """Gets the target reconvolution PSF for an input PSF object.

    This is taken from galsim/tests/test_metacal.py and assumes the psf is
    centered.
    Note: Order of parameters differs from the corresponding non-jax versions

    Parameters
    ----------
    psf : galsim.GSObject
        The input point spread function (PSF) object.
    dk : float
        The Fourier-space pixel scale.
    nxy_psf : int, optional
        The size of the PSF image in pixels (default is 53).
    step : float, optional
        Factor by which to expand the PSF to suppress noise from high-k
        fourier modes introduced due to shearing of pre-PSF images.
        Defaults to deep_field_metadetect.metacal.DEFAULT_STEP.
    flux : float, optional
        The total flux of the output PSF (default is 1).
    kim_size : int
        k image size.
        Defaults to None, which sets size as 4*nxy_psf

    Returns
    -------
    reconv_psf : JaxGalsim object
        The reconvolution PSF.
    """
    small_kval = 1.0e-2  # Find the k where the given psf hits this kvalue
    smaller_kval = 3.0e-3  # Target PSF will have this kvalue at the same k

    """
    The dk and kim_size are set for jitting purposes.
    This will lead to a difference in reconv PSF size between GS and JGS
    if similar settings are not used."""
    if kim_size is None:
        kim = psf.drawKImage(nx=4 * nxy_psf, ny=4 * nxy_psf, scale=dk)
    else:
        kim = psf.drawKImage(nx=kim_size, ny=kim_size, scale=dk)

    karr_r = kim.real.array
    # Find the smallest r where the kval < small_kval
    nk = karr_r.shape[0]
    kx, ky = jnp.meshgrid(jnp.arange(-nk / 2, nk / 2), jnp.arange(-nk / 2, nk / 2))
    ksq = (kx**2 + ky**2) * dk**2
    ksq_max = jnp.min(jnp.where(karr_r < small_kval * psf.flux, ksq, jnp.inf))

    # We take our target PSF to be the (round) Gaussian that is even smaller at
    # this ksq
    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2.0 * jnp.log(smaller_kval) / ksq_max

    dilation = 1.0 + 2.0 * step
    return jax_galsim.Gaussian(sigma=jnp.sqrt(sigma_sq) * dilation).withFlux(flux)


@partial(jax.jit, static_argnames=["dk", "nxy_psf"])
def jax_get_gauss_reconv_psf(dfmd_obs, nxy_psf, dk, step=DEFAULT_STEP):
    """Get the Gaussian reconv PSF for a DFMdetObs.

    Parameters
    ----------
    dfmd_obs : DFMdetObservation
        The observation containing the PSF to process.
    nxy_psf : int
        Size of the PSF image in pixels.
    dk : float
        Fourier-space pixel scale.
    step : float, optional
        Factor by which to expand the PSF to suppress noise from high-k
        fourier modes introduced due to shearing of pre-PSF images.
        Defaults to DEFAULT_STEP.

    Returns
    -------
    jax_galsim.Gaussian
        The Gaussian reconvolution PSF object.
    """
    psf = get_jax_galsim_object_from_dfmd_obs_nopix(dfmd_obs.psf, kind="image")
    return jax_get_gauss_reconv_psf_galsim(psf, nxy_psf=nxy_psf, dk=dk, step=step)


@partial(jax.jit, static_argnames=["nxy_psf", "scale"])
def jax_get_max_gauss_reconv_psf_galsim(
    psf_w, psf_d, nxy_psf, scale=0.2, step=DEFAULT_STEP
):
    """Get the larger of two Gaussian reconvolution PSFs for two galsim objects."""
    dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)
    mc_psf_w = jax_get_gauss_reconv_psf_galsim(psf_w, dk, nxy_psf, step=step)
    mc_psf_d = jax_get_gauss_reconv_psf_galsim(psf_d, dk, nxy_psf, step=step)

    return jax.lax.cond(
        mc_psf_w.fwhm > mc_psf_d.fwhm, lambda: mc_psf_w, lambda: mc_psf_d
    )


@partial(jax.jit, static_argnames=["scale", "nxy_psf"])
def jax_get_max_gauss_reconv_psf(obs_w, obs_d, nxy_psf, scale=0.2, step=DEFAULT_STEP):
    """Get the larger of two reconv PSFs for two DFMdetObservations."""
    psf_w = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w.psf, kind="image")
    psf_d = get_jax_galsim_object_from_dfmd_obs_nopix(obs_d.psf, kind="image")
    return jax_get_max_gauss_reconv_psf_galsim(
        psf_w, psf_d, nxy_psf, scale=scale, step=step
    )


@partial(jax.jit, static_argnames=["nxy_psf", "fft_size"])
def _jax_render_psf_and_build_obs(
    image, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1, fft_size=DEFAULT_FFT_SIZE
):
    reconv_psf = reconv_psf.withGSParams(
        minimum_fft_size=fft_size,
        maximum_fft_size=fft_size,
    )

    pim = reconv_psf.drawImage(
        nx=nxy_psf,
        ny=nxy_psf,
        wcs=dfmd_obs.psf.wcs.local(),
        offset=jax_galsim.PositionD(
            x=dfmd_obs.psf.wcs.origin.x - (nxy_psf + 1) / 2,
            y=dfmd_obs.psf.wcs.origin.y - (nxy_psf + 1) / 2,
        ),
    ).array

    obs_psf = dfmd_obs.psf.replace(image=pim)
    return dfmd_obs.replace(
        image=jnp.array(image), psf=obs_psf, weight=dfmd_obs.weight * weight_fac
    )


@partial(jax.jit, static_argnames=["dims", "fft_size"])
def _jax_metacal_op_g1g2_impl(
    *, wcs, image, noise, psf_inv, dims, reconv_psf, g1, g2, fft_size=DEFAULT_FFT_SIZE
):
    """Run metacal on an dfmd observation.

    Note that the noise image should already be rotated by 90 degrees here.
    """

    ims = jax_galsim.Convolve(
        [
            jax_galsim.Convolve([image, psf_inv]).shear(g1=g1, g2=g2),
            reconv_psf,
        ]
    )

    ns = jax_galsim.Convolve(
        [
            jax_galsim.Convolve([noise, psf_inv]).shear(g1=g1, g2=g2),
            reconv_psf,
        ]
    )

    ims = ims.withGSParams(
        minimum_fft_size=fft_size,
        maximum_fft_size=fft_size,
    )
    ims = ims.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array

    ns = ns.withGSParams(
        minimum_fft_size=fft_size,
        maximum_fft_size=fft_size,
    )
    ns = jnp.rot90(
        ns.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array,
        k=-1,
    )
    return ims + ns


def jax_metacal_op_g1g2(
    dfmd_obs, reconv_psf, g1, g2, nxy_psf, fft_size=DEFAULT_FFT_SIZE
):
    """Run metacal on an dfmd observation with specified shear.

    Parameters
    ----------
    dfmd_obs : DFMdetObservation
        The observation to process.
    reconv_psf : jax_galsim.GSObject
        The reconvolution PSF object.
    g1 : float
        g1 shear components to apply.
    g2 : float
        g2 shear components to apply.
    nxy_psf : int
        Size of the PSF image in pixels.
    fft_size : int, optional
        FFT size for convolution operations (default is DEFAULT_FFT_SIZE).

    Returns
    -------
    DFMdetObservation
        New observation with metacal applied.
    """
    mcal_image = _jax_metacal_op_g1g2_impl(
        wcs=dfmd_obs.wcs.local(),
        image=get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="image"),
        # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
        # rotates back after deconv and shearing
        noise=get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="noise", rot90=1),
        psf_inv=jax_galsim.Deconvolve(
            get_jax_galsim_object_from_dfmd_obs(dfmd_obs.psf, kind="image")
        ),
        dims=dfmd_obs.image.shape,
        reconv_psf=reconv_psf,
        g1=g1,
        g2=g2,
        fft_size=fft_size,
    )

    return _jax_render_psf_and_build_obs(
        mcal_image,
        dfmd_obs,
        reconv_psf,
        nxy_psf=nxy_psf,
        weight_fac=0.5,
        fft_size=fft_size,
    )


@partial(jax.jit, static_argnames=["nxy_psf", "scale", "shears", "fft_size"])
def jax_metacal_op_shears(
    dfmd_obs,
    nxy_psf=53,
    reconv_psf=jax_galsim.Gaussian(sigma=0.0).withFlux(1.0),
    shears=DEFAULT_SHEARS,
    step=DEFAULT_STEP,
    scale=0.2,
    fft_size=DEFAULT_FFT_SIZE,
):
    """Run metacal on an dfmd observation with multiple shear values.

    Parameters
    ----------
    dfmd_obs : DFMdetObservation
        The observation to process.
    nxy_psf : int, optional
        Size of the PSF image in pixels (default is 53).
    reconv_psf : jax_galsim.GSObject, optional
        The reconvolution PSF.
        Default: a proper reconvolution PSF will be computed automatically.
        using jax_get_gauss_reconv_psf function.
    shears : tuple of str, optional
        Shear identifiers to process (default is DEFAULT_SHEARS).
    step : float, optional
        Shear step magnitude (default is DEFAULT_STEP).
    scale : float, optional
        Pixel scale in arcseconds (default is 0.2).
    fft_size : int, optional
        FFT size for convolution operations (default is DEFAULT_FFT_SIZE).

    Returns
    -------
    dict
        Dictionary mapping shear identifiers to processed DFMdetObservation objects.
    """
    dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)

    def compute_reconv():
        return jax_get_gauss_reconv_psf(dfmd_obs, dk=dk, nxy_psf=nxy_psf, step=step)

    def use_provided_reconv():
        return reconv_psf

    reconv_psf = jax.lax.cond(
        reconv_psf.sigma == 0, compute_reconv, use_provided_reconv
    )
    wcs = dfmd_obs.wcs.local()
    image = get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="image")
    # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
    # rotates back after deconv and shearing
    noise = get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="noise", rot90=1)
    psf = get_jax_galsim_object_from_dfmd_obs(dfmd_obs.psf, kind="image")
    psf_inv = jax_galsim.Deconvolve(psf)

    shear_tuples = jnp.array([get_shear_tuple(shear, step) for shear in shears])
    g1_vals = shear_tuples[:, 0]
    g2_vals = shear_tuples[:, 1]

    # Vectorized metacal operation across all shears
    def single_shear_op(g1, g2):
        mcal_image = _jax_metacal_op_g1g2_impl(
            wcs=wcs,
            image=image,
            noise=noise,
            psf_inv=psf_inv,
            dims=dfmd_obs.image.shape,
            reconv_psf=reconv_psf,
            g1=g1,
            g2=g2,
            fft_size=fft_size,
        )
        return _jax_render_psf_and_build_obs(
            mcal_image,
            dfmd_obs,
            reconv_psf,
            nxy_psf=nxy_psf,
            weight_fac=0.5,
            fft_size=fft_size,
        )

    # Use vmap to parallelize across shears
    vectorized_shear_op = jax.vmap(single_shear_op)
    mcal_obs_list = vectorized_shear_op(g1_vals, g2_vals)

    # Convert back to dictionary format
    mcal_res = {}
    for i, shear in enumerate(shears):
        mcal_res[shear] = jax.tree.map(lambda x: x[i], mcal_obs_list)

    return mcal_res


@partial(
    jax.jit,
    static_argnames=[
        "nxy",
        "nxy_psf",
        "return_k_info",
        "force_stepk_field",
        "force_maxk_field",
        "force_stepk_psf",
        "force_maxk_psf",
        "fft_size",
    ],
)
def jax_match_psf(
    dfmd_obs,
    reconv_psf,
    nxy,
    nxy_psf,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=DEFAULT_FFT_SIZE,
):
    """Match the PSF on an dfmd observation to a new PSF."""
    wcs = dfmd_obs.wcs.local()
    image = get_jax_galsim_object_from_dfmd_obs(
        dfmd_obs,
        kind="image",
        force_stepk=force_stepk_field,
        force_maxk=force_maxk_field,
    )
    psf = get_jax_galsim_object_from_dfmd_obs(
        dfmd_obs.psf,
        kind="image",
        force_stepk=force_stepk_psf,
        force_maxk=force_maxk_psf,
    )

    ims = jax_galsim.Convolve([image, jax_galsim.Deconvolve(psf), reconv_psf])

    ims = ims.withGSParams(
        minimum_fft_size=fft_size,
        maximum_fft_size=fft_size,
    )
    ims = ims.drawImage(nx=nxy, ny=nxy, wcs=wcs).array

    if return_k_info:
        return _jax_render_psf_and_build_obs(
            ims, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1
        ), (image.stepk, image.maxk, psf.stepk, psf.maxk)
    else:
        return _jax_render_psf_and_build_obs(
            ims, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1
        ), (np.nan, np.nan, np.nan, np.nan)


def _extract_attr(obs, attr, dtype=np.float32):
    if getattr(obs, "has_" + attr)():
        return getattr(obs, attr)
    else:
        return np.zeros_like(obs.image, dtype=dtype)


def jax_add_dfmd_psf(psf1, psf2):
    """Add two DFMdetPSF objects"""
    from deep_field_metadetect.jaxify.observation import DFMdetPSF

    added_image = psf1.image + psf2.image

    new_wgt = jnp.where(
        (psf1.weight > 0) & (psf2.weight > 0),
        1 / (1 / psf1.weight + 1 / psf2.weight),
        0,
    )

    return DFMdetPSF(
        image=added_image,
        weight=new_wgt,
        wcs=psf1.wcs,  # Assume same WCS
        meta={**psf1.meta, **psf2.meta},
        store_pixels=psf1.store_pixels,
        ignore_zero_weight=psf1.ignore_zero_weight,
    )


def jax_add_dfmd_obs(
    dfmd_obs1, dfmd_obs2, ignore_psf=False, skip_mfrac_for_second=False
) -> DFMdetObservation:
    """Add two DFMD observations.

    Parameters
    ----------
    dfmd_obs1 : DFMdetObservation
        The first observation to add.
    dfmd_obs2 : DFMdetObservation
        The second observation to add.
    ignore_psf : bool, optional
        If True, the output PSF will be set to zero instead of combining
        the input PSFs. Default is False.
    skip_mfrac_for_second : bool, optional
        If True, only use the mfrac from the first observation instead of
        averaging both. Default is False.

    Returns
    -------
    DFMdetObservation
        A new observation containing the combined data from both inputs.
        The image is the sum of input images, weights are combined using
        inverse variance weighting, and masks are combined using bitwise OR,
        and noise is summed.
    """

    if repr(dfmd_obs1.wcs) != repr(dfmd_obs2.wcs):
        # This if statement will not perform any action at runtime
        raise RuntimeError(
            "AffineTransforms must be equal to add dfmd observations! %s != %s"
            % (repr(dfmd_obs1.wcs), repr(dfmd_obs2.wcs)),
        )

    if dfmd_obs1.image.shape != dfmd_obs2.image.shape:
        raise RuntimeError(
            "Image shapes must be equal to add dfmd observations! %s != %s"
            % (
                dfmd_obs1.image.shape,
                dfmd_obs2.image.shape,
            ),
        )

    # Handle PSF addition using dedicated function
    def add_psfs():
        return jax_add_dfmd_psf(dfmd_obs1.psf, dfmd_obs2.psf)

    def no_psf():
        from deep_field_metadetect.jaxify.observation import DFMdetPSF

        return DFMdetPSF(
            image=jnp.zeros_like(dfmd_obs1.psf.image, dtype=jnp.float32),
            wcs=dfmd_obs1.psf.wcs,
            meta=dfmd_obs1.psf.meta,
            store_pixels=dfmd_obs1.psf.store_pixels,
            ignore_zero_weight=dfmd_obs1.psf.ignore_zero_weight,
        )

    # Add PSFs if both exist and we're not ignoring PSF
    has_psf1 = dfmd_obs1.has_psf()
    has_psf2 = dfmd_obs2.has_psf()
    should_add_psf = (not ignore_psf) & has_psf1 & has_psf2

    new_psf = jax.lax.cond(should_add_psf, add_psfs, no_psf)

    new_wgt = jnp.where(
        (dfmd_obs1.weight > 0) & (dfmd_obs2.weight > 0),
        1 / (1 / dfmd_obs1.weight + 1 / dfmd_obs2.weight),
        0,
    )

    new_meta_data = {}

    # Handle bmask, ormask, noise, and mfrac
    # Unlike the non-jax version we do not need to test conditions here
    # because now the default values are zeros instead of None
    new_bmask = dfmd_obs1.bmask | dfmd_obs2.bmask
    new_ormask = dfmd_obs1.ormask | dfmd_obs2.ormask
    new_noise = dfmd_obs1.noise + dfmd_obs2.noise

    def mfrac_skip_second():
        return dfmd_obs1.mfrac

    def mfrac_use_both():
        return (dfmd_obs1.mfrac + dfmd_obs2.mfrac) / 2

    new_mfrac = jax.lax.cond(skip_mfrac_for_second, mfrac_skip_second, mfrac_use_both)

    new_meta_data.update(dfmd_obs1.meta)
    new_meta_data.update(dfmd_obs2.meta)

    obs = DFMdetObservation(
        image=dfmd_obs1.image + dfmd_obs2.image,
        weight=new_wgt,
        bmask=new_bmask,
        ormask=new_ormask,
        noise=new_noise,
        wcs=dfmd_obs1.wcs,
        psf=new_psf,
        meta=new_meta_data,
        mfrac=new_mfrac,
        store_pixels=getattr(dfmd_obs1, "store_pixels", True),
        ignore_zero_weight=getattr(dfmd_obs1, "ignore_zero_weight", True),
    )

    return obs


def get_jax_galsim_object_from_dfmd_obs(
    dfmd_obs,
    kind="image",
    rot90=0,
    force_stepk=0.0,
    force_maxk=0.0,
):
    """Make an interpolated image from an dfmd obs."""
    return jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(
            jnp.rot90(getattr(dfmd_obs, kind).copy(), k=rot90),
            wcs=dfmd_obs.wcs.local(),
        ),
        x_interpolant="lanczos15",
        wcs=dfmd_obs.wcs.local(),
        _force_stepk=force_stepk,
        _force_maxk=force_maxk,
    )


def get_jax_galsim_object_from_dfmd_obs_nopix(dfmd_obs, kind="image"):
    """Make an interpolated image from an DFMdet obs w/o a pixel."""
    wcs = dfmd_obs.wcs.local()
    return jax_galsim.Convolve(
        [
            get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind=kind),
            jax_galsim.Deconvolve(wcs.toWorld(jax_galsim.Pixel(scale=1))),
        ]
    )


@partial(
    jax.jit,
    static_argnames=[
        "nxy",
        "nxy_psf",
        "shears",
        "skip_obs_wide_corrections",
        "skip_obs_deep_corrections",
        "return_noshear_deep",
        "scale",
        "return_k_info",
        "force_stepk_field",
        "force_maxk_field",
        "force_stepk_psf",
        "force_maxk_psf",
        "fft_size",
    ],
)
def _jax_helper_metacal_wide_and_deep_psf_matched(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    reconv_psf,
    nxy,
    nxy_psf,
    shears=None,
    step=DEFAULT_STEP,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_noshear_deep=False,
    scale=0.2,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=DEFAULT_FFT_SIZE,
):
    """Do metacalibration for a combination of wide+deep datasets.

    Parameters
    ----------
    obs_wide : DFMdetObservation
        The wide-field observation.
    obs_deep : DFMdetObservation
        The deep-field observation.
    obs_deep_noise : DFMdetObservation
        The deep-field noise observation.
    reconv_psf : JaxGalsim object
        The reconvolution PSF.
    shears : tuple of strings, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    return_noshear_deep : bool, optional
        adds deep field no shear results to the output. Default - False.
        This is a static variable so changing it would trigger recompilation.
    scale : float, optional
        pixel scale. default to 0.2.
        Note this parameter is not present in non-jax version.
        This is later used for compute_stepk to compute the pixel scale in
        fourier space and this is a static variable so changing it would
        trigger recompilation.
    return_k_info : bool, optional
        return _force_stepk and _force_maxk values in the following order
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
    fft_size: int, optional
        To fix max and min values of FFT size.
        Defaults to None which lets Galsim determine the values.
        Used mainly to test against JaxGalsim.

    Returns
    -------
    mcal_res : dict
        Output from metacal_op_shears for shear cases listed by the shears input,
        optionally no shear deep field case if return_noshear_deep is True
        and kinfo for debugging if return_k_info is set to True.
        kinfo is returned in the following order:
        _force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf.
    """
    # make the wide obs

    mcal_obs_wide, kinfo = jax_match_psf(
        obs_wide,
        reconv_psf,
        nxy,
        nxy_psf,
        return_k_info=return_k_info,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        fft_size=fft_size,
    )
    if not skip_obs_wide_corrections:
        mcal_obs_wide = jax_add_dfmd_obs(
            mcal_obs_wide,
            jax_metacal_op_g1g2(obs_deep_noise, reconv_psf, 0, 0, nxy_psf=nxy_psf),
            skip_mfrac_for_second=True,
        )

    # get PSF matched noise
    obs_wide_noise = obs_wide.replace(image=obs_wide.noise)
    wide_noise_corr, _ = jax_match_psf(
        obs_wide_noise,
        reconv_psf,
        nxy,
        nxy_psf,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        fft_size=fft_size,
    )

    # now run mcal on deep
    mcal_res = jax_metacal_op_shears(
        obs_deep,
        reconv_psf=reconv_psf,
        shears=shears,
        step=step,
        nxy_psf=nxy_psf,
        scale=scale,
        fft_size=fft_size,
    )

    # now add in noise corr to make it match the wide noise
    if not skip_obs_deep_corrections:
        for k in shears:
            mcal_res[k] = jax_add_dfmd_obs(
                mcal_res[k],
                wide_noise_corr,
                skip_mfrac_for_second=True,
            )

    # we report the wide obs as noshear for later measurements
    noshear_res = mcal_res.pop("noshear")
    mcal_res["noshear"] = mcal_obs_wide
    if return_noshear_deep:
        mcal_res["noshear_deep"] = noshear_res

    if return_k_info:
        mcal_res["kinfo"] = kinfo

    return mcal_res


def jax_metacal_wide_and_deep_psf_matched(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    nxy,
    nxy_psf,
    shears=DEFAULT_SHEARS,
    step=DEFAULT_STEP,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_noshear_deep=False,
    scale=0.2,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=DEFAULT_FFT_SIZE,
):
    """Do metacalibration for a combination of wide+deep datasets.

    Parameters
    ----------
    obs_wide : DFMdetObservation
        The wide-field observation.
    obs_deep : DFMdetObservation
        The deep-field observation.
    obs_deep_noise : DFMdetObservation
        The deep-field noise observation.
    shears : tuple of strings, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    return_noshear_deep : bool, optional
        adds deep field no shear results to the output. Default - False.
        This is a static variable so changing it would trigger recompilation.
    scale : float, optional
        pixel scale. default to 0.2.
        Note this parameter is not present in non-jax version.
        This is later used for compute_stepk to compute the pixel scale in
        fourier space and this is a static variable so changing it would
        trigger recompilation.
    return_k_info : bool, optional
        return _force_stepk and _force_maxk values in the following order
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
    fft_size: int, optional
        To fix max and min values of FFT size.
        Defaults to None which lets Galsim determine the values.
        Used mainly to test against JaxGalsim.

    Returns
    -------
    mcal_res : dict
        Output from metacal_op_shears for shear cases listed by the shears input,
        optionally no shear deep field case if return_noshear_deep is True
        and kinfo for debugging if return_k_info is set to True.
        kinfo is returned in the following order:
        _force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf.
    """
    # first get the biggest reconv PSF of the two
    reconv_psf = jax_get_max_gauss_reconv_psf(obs_wide, obs_deep, nxy_psf, scale)

    mcal_res = _jax_helper_metacal_wide_and_deep_psf_matched(
        obs_wide=obs_wide,
        obs_deep=obs_deep,
        obs_deep_noise=obs_deep_noise,
        reconv_psf=reconv_psf,
        nxy=nxy,
        nxy_psf=nxy_psf,
        shears=shears,
        step=step,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        return_noshear_deep=return_noshear_deep,
        scale=scale,
        return_k_info=return_k_info,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        fft_size=fft_size,
    )

    for k in shears:
        mcal_res[k] = dfmd_obs_to_ngmix_obs(mcal_res[k])
        mcal_res[k].psf.galsim_obj = reconv_psf

    return mcal_res
