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


@partial(jax.jit, static_argnames=["dk", "nxy_psf", "kim_size"])
def jax_get_gauss_reconv_psf_galsim(
    psf, dk, nxy_psf=53, step=DEFAULT_STEP, flux=1, kim_size=None
):
    """Gets the target reconvolution PSF for an input PSF object.

    This is taken from galsim/tests/test_metacal.py and assumes the psf is
    centered.

    Parameters
    ----------
    psf : galsim.GSObject
        The input point spread function (PSF) object.
    dk : float
        The Fourier-space pixel scale.
    nxy_psf : int, optional
        The size of the PSF image in pixels (default is 53).
        Used to set k_image size, but is overridden if kim_size is passed.
    step : float, optional
        The step size for coordinate grids (default is `DEFAULT_STEP`).
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

    if kim_size is None:
        kim = psf.drawKImage(nx=4 * nxy_psf, ny=4 * nxy_psf, scale=dk)
    else:
        kim = psf.drawKImage(nx=kim_size, ny=kim_size, scale=dk)

    # This will lead to a differnce in reconv psf size between GS and JGS

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
    """Get the Gaussian reconv PSF for a DFMdetObs."""
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


@partial(jax.jit, static_argnames=["nxy_psf", "max_min_fft_size"])
def _jax_render_psf_and_build_obs(
    image, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1, max_min_fft_size=1024
):
    reconv_psf = reconv_psf.withGSParams(
        minimum_fft_size=max_min_fft_size,
        maximum_fft_size=max_min_fft_size,
    )

    pim = reconv_psf.drawImage(
        nx=nxy_psf,
        ny=nxy_psf,
        wcs=dfmd_obs.psf.wcs._local_wcs,
        offset=jax_galsim.PositionD(
            x=dfmd_obs.psf.wcs.origin.x - nxy_psf / 2,
            y=dfmd_obs.psf.wcs.origin.y - nxy_psf / 2,
        ),
    ).array

    obs_psf = dfmd_obs.psf._replace(image=pim)
    return dfmd_obs._replace(
        image=jnp.array(image), psf=obs_psf, weight=dfmd_obs.weight * weight_fac
    )


@partial(jax.jit, static_argnames=["dims", "max_min_fft_size"])
def _jax_metacal_op_g1g2_impl(
    *, wcs, image, noise, psf_inv, dims, reconv_psf, g1, g2, max_min_fft_size=1024
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
        minimum_fft_size=max_min_fft_size,
        maximum_fft_size=max_min_fft_size,
    )
    ims = ims.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array

    ns = ns.withGSParams(
        minimum_fft_size=max_min_fft_size,
        maximum_fft_size=max_min_fft_size,
    )
    ns = jnp.rot90(
        ns.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array,
        k=-1,
    )
    return ims + ns


def jax_metacal_op_g1g2(dfmd_obs, reconv_psf, g1, g2, nxy_psf, max_min_fft_size=1024):
    """Run metacal on an dfmd obs."""
    mcal_image = _jax_metacal_op_g1g2_impl(
        wcs=dfmd_obs.wcs._local_wcs,
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
        max_min_fft_size=max_min_fft_size,
    )

    return _jax_render_psf_and_build_obs(
        mcal_image,
        dfmd_obs,
        reconv_psf,
        nxy_psf=nxy_psf,
        weight_fac=0.5,
        max_min_fft_size=max_min_fft_size,
    )


@partial(jax.jit, static_argnames=["nxy_psf", "scale", "shears", "max_min_fft_size"])
def jax_metacal_op_shears(
    dfmd_obs,
    nxy_psf=53,
    reconv_psf=None,
    shears=None,
    step=DEFAULT_STEP,
    scale=0.2,
    max_min_fft_size=1024,
):
    """Run metacal on an dfmd observation."""
    if shears is None:
        shears = DEFAULT_SHEARS

    dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)
    if reconv_psf is None:
        reconv_psf = jax_get_gauss_reconv_psf(
            dfmd_obs,
            dk=dk,
            nxy_psf=nxy_psf,
            step=step,
        )

    wcs = dfmd_obs.wcs._local_wcs
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
            max_min_fft_size=max_min_fft_size,
        )
        return _jax_render_psf_and_build_obs(
            mcal_image,
            dfmd_obs,
            reconv_psf,
            nxy_psf=nxy_psf,
            weight_fac=0.5,
            max_min_fft_size=max_min_fft_size,
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
        "max_min_fft_size",
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
    max_min_fft_size=1024,
):
    """Match the PSF on an dfmd observation to a new PSF."""
    wcs = dfmd_obs.wcs._local_wcs
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

    ims = jax_galsim.Convolve(
        [image, jax_galsim.Deconvolve(psf), reconv_psf],
        gsparams=jax_galsim.GSParams(
            minimum_fft_size=max_min_fft_size,
            maximum_fft_size=max_min_fft_size,
        ),
    )

    ims = ims.withGSParams(
        minimum_fft_size=max_min_fft_size,
        maximum_fft_size=max_min_fft_size,
    )
    ims = ims.drawImage(nx=nxy, ny=nxy, wcs=wcs).array

    def return_obs_and_kinfo(_):
        return _jax_render_psf_and_build_obs(
            ims, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1
        ), (image.stepk, image.maxk, psf.stepk, psf.maxk)

    def return_obs_only(_):
        return _jax_render_psf_and_build_obs(
            ims, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1
        ), (0.0, 0.0, 0.0, 0.0)

    return jax.lax.cond(
        return_k_info, return_obs_and_kinfo, return_obs_only, operand=None
    )


def _extract_attr(obs, attr, dtype=jnp.float64):
    if getattr(obs, "has_" + attr)():
        return getattr(obs, attr)
    else:
        return np.zeros_like(obs.image, dtype=dtype)


@partial(jax.jit, static_argnames=["ignore_psf", "skip_mfrac_for_second"])
def jax_add_dfmd_obs(
    dfmd_obs1, dfmd_obs2, ignore_psf=False, skip_mfrac_for_second=False
) -> DFMdetObservation:
    """Add two dfmd observations"""

    if repr(dfmd_obs1.wcs) != repr(dfmd_obs2.wcs):
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

    if dfmd_obs1.has_psf() != dfmd_obs2.has_psf() and not ignore_psf:
        raise RuntimeError(
            "Observations must both either have or not have a "
            "PSF to add them. %s != %s"
            % (
                dfmd_obs1.has_psf(),
                dfmd_obs2.has_psf(),
            ),
        )

    if dfmd_obs1.has_psf() and dfmd_obs2.has_psf() and not ignore_psf:
        # We ignore the PSF in this call since PSFs do not have PSFs
        new_psf = jax_add_dfmd_obs(dfmd_obs1.psf, dfmd_obs2.psf, ignore_psf=True)
    else:
        new_psf = None

    new_wgt = jnp.where(
        (dfmd_obs1.weight > 0) & (dfmd_obs2.weight > 0),
        1 / (1 / dfmd_obs1.weight + 1 / dfmd_obs2.weight),
        0,
    )

    new_bmask = None
    new_ormask = None
    new_noise = None
    new_mfrac = None
    new_meta_data = {}

    if dfmd_obs1.has_bmask() or dfmd_obs2.has_bmask():
        new_bmask = _extract_attr(dfmd_obs1, "bmask", jnp.int32) | _extract_attr(
            dfmd_obs2, "bmask", jnp.int32
        )

    if dfmd_obs1.has_ormask() or dfmd_obs2.has_ormask():
        new_ormask = _extract_attr(dfmd_obs1, "ormask", jnp.int32) | _extract_attr(
            dfmd_obs2, "ormask", jnp.int32
        )

    if dfmd_obs1.has_noise() or dfmd_obs2.has_noise():
        new_noise = _extract_attr(dfmd_obs1, "noise") + _extract_attr(
            dfmd_obs2, "noise"
        )

    if skip_mfrac_for_second:
        if dfmd_obs1.has_mfrac():
            new_mfrac = _extract_attr(dfmd_obs1, "mfrac")
    else:
        if dfmd_obs1.has_mfrac() or dfmd_obs2.has_mfrac():
            new_mfrac = (
                _extract_attr(dfmd_obs1, "mfrac") + _extract_attr(dfmd_obs2, "mfrac")
            ) / 2

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
            wcs=dfmd_obs.wcs._local_wcs,
        ),
        x_interpolant="lanczos15",
        wcs=dfmd_obs.wcs._local_wcs,
        _force_stepk=force_stepk,
        _force_maxk=force_maxk,
    )


def get_jax_galsim_object_from_dfmd_obs_nopix(dfmd_obs, kind="image"):
    """Make an interpolated image from an DFMdet obs w/o a pixel."""
    wcs = dfmd_obs.wcs._local_wcs
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
        "max_min_fft_size",
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
    max_min_fft_size=1024,
):
    """Do metacalibration for a combination of wide+deep datasets."""

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
        max_min_fft_size=max_min_fft_size,
    )
    if not skip_obs_wide_corrections:
        mcal_obs_wide = jax_add_dfmd_obs(
            mcal_obs_wide,
            jax_metacal_op_g1g2(obs_deep_noise, reconv_psf, 0, 0, nxy_psf=nxy_psf),
            skip_mfrac_for_second=True,
        )

    # get PSF matched noise
    obs_wide_noise = obs_wide._replace(image=obs_wide.noise)
    wide_noise_corr, _ = jax_match_psf(
        obs_wide_noise,
        reconv_psf,
        nxy,
        nxy_psf,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        max_min_fft_size=max_min_fft_size,
    )

    # now run mcal on deep
    mcal_res = jax_metacal_op_shears(
        obs_deep,
        reconv_psf=reconv_psf,
        shears=shears,
        step=step,
        nxy_psf=nxy_psf,
        scale=scale,
        max_min_fft_size=max_min_fft_size,
    )

    # now add in noise corr to make it match the wide noise
    # TODO: is it after to vextorize?
    if not skip_obs_deep_corrections:
        for k in mcal_res:
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

    return mcal_res, kinfo


def jax_metacal_wide_and_deep_psf_matched(
    obs_wide,
    obs_deep,
    obs_deep_noise,
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
    max_min_fft_size=1024,
):
    """Do metacalibration for a combination of wide+deep datasets."""

    # first get the biggest reconv PSF of the two
    reconv_psf = jax_get_max_gauss_reconv_psf(obs_wide, obs_deep, nxy_psf, scale)

    mcal_res, kinfo = _jax_helper_metacal_wide_and_deep_psf_matched(
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
        max_min_fft_size=max_min_fft_size,
    )

    for k in mcal_res:
        mcal_res[k] = dfmd_obs_to_ngmix_obs(mcal_res[k])
        mcal_res[k].psf.galsim_obj = reconv_psf

    return mcal_res, kinfo
