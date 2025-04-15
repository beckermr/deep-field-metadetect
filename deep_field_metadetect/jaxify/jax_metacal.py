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


@partial(jax.jit, static_argnames=["dk", "nxy_psf"])
def jax_get_gauss_reconv_psf_galsim(psf, dk, nxy_psf=53, step=DEFAULT_STEP, flux=1):
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
    step : float, optional
        The step size for coordinate grids (default is `DEFAULT_STEP`).
    flux : float, optional
        The total flux of the output PSF (default is 1).

    Returns
    -------
    reconv_psf : JaxGalsim object
        The reconvolution PSF.
    """
    small_kval = 1.0e-2  # Find the k where the given psf hits this kvalue
    smaller_kval = 3.0e-3  # Target PSF will have this kvalue at the same k

    kim = psf.drawKImage(nx=nxy_psf * 4, ny=nxy_psf * 4, scale=dk)
    # kim = psf.drawKImage(scale=dk)
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


@partial(jax.jit, static_argnames=["nxy_psf"])
def _jax_render_psf_and_build_obs(image, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1):
    reconv_psf = reconv_psf.withGSParams(
        minimum_fft_size=nxy_psf * 4,
        maximum_fft_size=nxy_psf * 4,
    )

    pim = reconv_psf.drawImage(
        nx=nxy_psf,
        ny=nxy_psf,
        wcs=dfmd_obs.psf.jacobian,
        offset=jax_galsim.PositionD(
            x=dfmd_obs.psf.jac_col0 + 1 - nxy_psf / 2,
            y=dfmd_obs.psf.jac_row0 + 1 - nxy_psf / 2,
        ),
    ).array

    obs_psf = dfmd_obs.psf._replace(image=pim)
    return dfmd_obs._replace(
        image=jnp.array(image), psf=obs_psf, weight=dfmd_obs.weight * weight_fac
    )


@partial(jax.jit, static_argnames="dims")
def _jax_metacal_op_g1g2_impl(*, wcs, image, noise, psf_inv, dims, reconv_psf, g1, g2):
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
        minimum_fft_size=dims[0] * 4,
        maximum_fft_size=dims[0] * 4,
    )
    ims = ims.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array

    ns = ns.withGSParams(
        minimum_fft_size=dims[0] * 4,
        maximum_fft_size=dims[0] * 4,
    )
    ns = jnp.rot90(
        ns.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array,
        k=-1,
    )
    return ims + ns


def jax_metacal_op_g1g2(dfmd_obs, reconv_psf, g1, g2, nxy_psf):
    """Run metacal on an dfmd obs."""
    mcal_image = _jax_metacal_op_g1g2_impl(
        wcs=dfmd_obs.jacobian,
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
    )

    return _jax_render_psf_and_build_obs(
        mcal_image, dfmd_obs, reconv_psf, nxy_psf=nxy_psf, weight_fac=0.5
    )


@partial(jax.jit, static_argnames=["nxy_psf", "scale", "shears"])
def jax_metacal_op_shears(
    dfmd_obs,
    nxy_psf=53,
    reconv_psf=None,
    shears=None,
    step=DEFAULT_STEP,
    scale=0.2,
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

    wcs = dfmd_obs.jacobian
    image = get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="image")
    # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
    # rotates back after deconv and shearing
    noise = get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="noise", rot90=1)
    psf = get_jax_galsim_object_from_dfmd_obs(dfmd_obs.psf, kind="image")
    psf_inv = jax_galsim.Deconvolve(psf)

    mcal_res = {}
    for shear in shears:
        g1, g2 = get_shear_tuple(shear, step)

        mcal_image = _jax_metacal_op_g1g2_impl(
            wcs=wcs,
            image=image,
            noise=noise,
            psf_inv=psf_inv,
            dims=dfmd_obs.image.shape,
            reconv_psf=reconv_psf,
            g1=g1,
            g2=g2,
        )

        mcal_res[shear] = _jax_render_psf_and_build_obs(
            mcal_image,
            dfmd_obs,
            reconv_psf,
            nxy_psf=nxy_psf,
            weight_fac=0.5,
        )
    return mcal_res


@partial(jax.jit, static_argnames=["nxy", "nxy_psf"])
def jax_match_psf(dfmd_obs, reconv_psf, nxy, nxy_psf):
    """Match the PSF on an dfmd observation to a new PSF."""
    wcs = dfmd_obs.jacobian
    image = get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="image")
    psf = get_jax_galsim_object_from_dfmd_obs(dfmd_obs.psf, kind="image")

    ims = jax_galsim.Convolve([image, jax_galsim.Deconvolve(psf), reconv_psf])

    ims = ims.withGSParams(
        minimum_fft_size=nxy * 4,
        maximum_fft_size=nxy * 4,
    )
    ims = ims.drawImage(nx=nxy, ny=nxy, wcs=wcs).array

    return _jax_render_psf_and_build_obs(
        ims, dfmd_obs, reconv_psf, nxy_psf, weight_fac=1
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

    if repr(dfmd_obs1.jacobian) != repr(dfmd_obs2.jacobian):
        raise RuntimeError(
            "Jacobians must be equal to add dfmd observations! %s != %s"
            % (repr(dfmd_obs1.jacobian), repr(dfmd_obs2.jacobian)),
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
        jacobian=jax_galsim.wcs.JacobianWCS(
            dudx=dfmd_obs1.jacobian.dudx,
            dudy=dfmd_obs1.jacobian.dudy,
            dvdx=dfmd_obs1.jacobian.dvdx,
            dvdy=dfmd_obs1.jacobian.dvdy,
        ),
        psf=new_psf,
        meta=new_meta_data,  # Directly copy metadata
        mfrac=new_mfrac,
        store_pixels=getattr(dfmd_obs1, "store_pixels", True),
        ignore_zero_weight=getattr(dfmd_obs1, "ignore_zero_weight", True),
        jac_row0=dfmd_obs1.jac_row0,
        jac_col0=dfmd_obs1.jac_col0,
        jac_det=dfmd_obs1.jac_det,
        jac_scale=dfmd_obs1.jac_scale,
    )

    return obs


def get_jax_galsim_object_from_dfmd_obs(dfmd_obs, kind="image", rot90=0):
    """Make an interpolated image from an dfmd obs."""
    return jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(
            jnp.rot90(getattr(dfmd_obs, kind).copy(), k=rot90),
            wcs=dfmd_obs.jacobian,
        ),
        x_interpolant="lanczos15",
    )


def get_jax_galsim_object_from_dfmd_obs_nopix(dfmd_obs, kind="image"):
    """Make an interpolated image from an DFMdet obs w/o a pixel."""
    wcs = dfmd_obs.jacobian
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
):
    """Do metacalibration for a combination of wide+deep datasets."""

    # make the wide obs
    if skip_obs_wide_corrections:
        mcal_obs_wide = jax_match_psf(obs_wide, reconv_psf, nxy, nxy_psf)
    else:
        mcal_obs_wide = jax_add_dfmd_obs(
            jax_match_psf(obs_wide, reconv_psf, nxy, nxy_psf),
            jax_metacal_op_g1g2(obs_deep_noise, reconv_psf, 0, 0, nxy_psf=nxy_psf),
            skip_mfrac_for_second=True,
        )

    # get PSF matched noise
    obs_wide_noise = obs_wide._replace(image=obs_wide.noise)
    wide_noise_corr = jax_match_psf(obs_wide_noise, reconv_psf, nxy, nxy_psf)

    # now run mcal on deep
    mcal_res = jax_metacal_op_shears(
        obs_deep,
        reconv_psf=reconv_psf,
        shears=shears,
        step=step,
        nxy_psf=nxy_psf,
        scale=scale,
    )

    # now add in noise corr to make it match the wide noise
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

    return mcal_res


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
):
    """Do metacalibration for a combination of wide+deep datasets."""

    # first get the biggest reconv PSF of the two
    reconv_psf = jax_get_max_gauss_reconv_psf(obs_wide, obs_deep, nxy, scale)

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
    )

    for k in mcal_res:
        mcal_res[k] = dfmd_obs_to_ngmix_obs(mcal_res[k])
        mcal_res[k].psf.galsim_obj = reconv_psf

    return mcal_res
