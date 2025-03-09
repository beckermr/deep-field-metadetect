from functools import partial

import galsim as galsim
import jax
import jax.numpy as jnp
import jax_galsim
import numpy as np

from deep_field_metadetect.jaxify.observation import NT_to_ngmix_obs, NTObservation

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


# TODO: what should be the value to nxy?
@partial(jax.jit, static_argnames=["dk", "nxy_psf"])
def jax_get_gauss_reconv_psf_galsim(psf, dk, nxy_psf=53, step=DEFAULT_STEP, flux=1):
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
    sigma : float
        The width of the reconv PSF befor dilation.
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
def jax_get_gauss_reconv_psf(obs, nxy_psf, dk, step=DEFAULT_STEP):
    """Get the Gaussian reconv PSF for an ngmix obs."""
    psf = get_jax_galsim_object_from_NT_obs_nopix(obs.psf, kind="image")
    return jax_get_gauss_reconv_psf_galsim(psf, nxy_psf=nxy_psf, dk=dk, step=step)


@partial(jax.jit, static_argnames=["dk_w", "dk_d", "nxy_psf"])
def jax_get_max_gauss_reconv_psf_galsim(
    psf_w, psf_d, dk_w, dk_d, nxy_psf, step=DEFAULT_STEP
):
    """Get the larger of two Gaussian reconvolution PSFs for two galsim objects."""
    mc_psf_w = jax_get_gauss_reconv_psf_galsim(psf_w, dk_w, nxy_psf, step=step)
    mc_psf_d = jax_get_gauss_reconv_psf_galsim(psf_d, dk_d, nxy_psf, step=step)

    # fwhm_w = jnp.asarray(mc_psf_w.fwhm)
    # fwhm_d = jnp.asarray(mc_psf_d.fwhm)

    return jax.lax.cond(
        mc_psf_w.fwhm > mc_psf_d.fwhm, lambda: mc_psf_w, lambda: mc_psf_d
    )


def jax_get_max_gauss_reconv_psf(obs_w, obs_d, dk_w, dk_d, nxy, step=DEFAULT_STEP):
    """Get the larger of two reconv PSFs for two ngmix.Observations."""
    psf_w = get_jax_galsim_object_from_NT_obs_nopix(obs_w.psf, kind="image")
    psf_d = get_jax_galsim_object_from_NT_obs_nopix(obs_d.psf, kind="image")
    return jax_get_max_gauss_reconv_psf_galsim(psf_w, psf_d, dk_w, dk_d, nxy, step=step)


@partial(jax.jit, static_argnames=["nxy_psf"])
def _jax_render_psf_and_build_obs(image, obs, reconv_psf, nxy_psf, weight_fac=1):
    reconv_psf = reconv_psf.withGSParams(
        minimum_fft_size=nxy_psf * 4,
        maximum_fft_size=nxy_psf * 4,
    )

    pim = reconv_psf.drawImage(
        nx=53,
        ny=53,
        wcs=obs.psf.jacobian,
        offset=jax_galsim.PositionD(
            x=obs.psf.jac_col0 + 1 - nxy_psf / 2,  # TODO: what is the size is odd?
            y=obs.psf.jac_row0 + 1 - nxy_psf / 2,
        ),
    ).array

    obs_psf = obs.psf._replace(image=pim)
    return obs._replace(
        image=jnp.array(image), psf=obs_psf, weight=obs.weight * weight_fac
    )


@partial(jax.jit, static_argnames="dims")
def _jax_metacal_op_g1g2_impl(*, wcs, image, noise, psf_inv, dims, reconv_psf, g1, g2):
    """Run metacal on an ngmix observation.

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


def jax_metacal_op_g1g2(obs, reconv_psf, g1, g2, nxy_psf):
    """Run metacal on an ngmix observation."""
    mcal_image = _jax_metacal_op_g1g2_impl(
        wcs=obs.jacobian,
        image=get_jax_galsim_object_from_NT_obs(obs, kind="image"),
        # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
        # rotates back after deconv and shearing
        noise=get_jax_galsim_object_from_NT_obs(obs, kind="noise", rot90=1),
        psf_inv=jax_galsim.Deconvolve(
            get_jax_galsim_object_from_NT_obs(obs.psf, kind="image")
        ),
        dims=obs.image.shape,
        reconv_psf=reconv_psf,
        g1=g1,
        g2=g2,
    )

    return _jax_render_psf_and_build_obs(
        mcal_image, obs, reconv_psf, nxy_psf=nxy_psf, weight_fac=0.5
    )


@partial(jax.jit, static_argnames=["nxy_psf", "dk", "shears"])
def jax_metacal_op_shears(
    obs, dk, nxy_psf=53, reconv_psf=None, shears=None, step=DEFAULT_STEP
):
    """Run metacal on an ngmix observation."""
    if shears is None:
        shears = DEFAULT_SHEARS

    if reconv_psf is None:
        reconv_psf = jax_get_gauss_reconv_psf(obs, dk=dk, nxy_psf=nxy_psf, step=step)

    wcs = obs.jacobian
    image = get_jax_galsim_object_from_NT_obs(obs, kind="image")
    # we rotate by 90 degrees on the way in and then _metacal_op_g1g2_impl
    # rotates back after deconv and shearing
    noise = get_jax_galsim_object_from_NT_obs(obs, kind="noise", rot90=1)
    psf = get_jax_galsim_object_from_NT_obs(obs.psf, kind="image")
    psf_inv = jax_galsim.Deconvolve(psf)

    mcal_res = {}
    for shear in shears:
        g1, g2 = get_shear_tuple(shear, step)

        mcal_image = _jax_metacal_op_g1g2_impl(
            wcs=wcs,
            image=image,
            noise=noise,
            psf_inv=psf_inv,
            dims=obs.image.shape,
            reconv_psf=reconv_psf,
            g1=g1,
            g2=g2,
        )

        mcal_res[shear] = _jax_render_psf_and_build_obs(
            mcal_image,
            obs,
            reconv_psf,
            nxy_psf=nxy_psf,
            weight_fac=0.5,
        )
    return mcal_res


@partial(jax.jit, static_argnames=["nxy_psf"])
def jax_match_psf(obs, reconv_psf, nxy_psf):
    """Match the PSF on an ngmix observation to a new PSF."""
    wcs = obs.jacobian
    image = get_jax_galsim_object_from_NT_obs(obs, kind="image")
    psf = get_jax_galsim_object_from_NT_obs(obs.psf, kind="image")

    ims = jax_galsim.Convolve([image, jax_galsim.Deconvolve(psf), reconv_psf])

    ims = ims.withGSParams(
        minimum_fft_size=nxy_psf * 4,
        maximum_fft_size=nxy_psf * 4,
    )
    ims = ims.drawImage(nx=nxy_psf, ny=nxy_psf, wcs=wcs).array

    return _jax_render_psf_and_build_obs(ims, obs, reconv_psf, nxy_psf, weight_fac=1)


def _extract_attr(obs, attr, dtype):
    if getattr(obs, "has_" + attr)():
        return getattr(obs, attr)
    else:
        return np.zeros_like(obs.image, dtype=dtype)


@partial(jax.jit, static_argnames=["ignore_psf", "skip_mfrac_for_second"])
def jax_add_ngmix_obs(
    obs1, obs2, ignore_psf=False, skip_mfrac_for_second=False
) -> NTObservation:
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
        # if nxy_psf is None:
        #     raise ValueError("Provide the psf size nxy_psf")
        new_psf = jax_add_ngmix_obs(obs1.psf, obs2.psf, ignore_psf=True)
    else:
        new_psf = None

    new_wgt = jnp.where(
        (obs1.weight > 0) & (obs2.weight > 0),
        1 / (1 / obs1.weight + 1 / obs2.weight),
        0,
    )

    new_bmask = None
    new_ormask = None
    new_noise = None
    new_mfrac = None
    new_meta_data = {}

    if obs1.has_bmask() or obs2.has_bmask():
        new_bmask = _extract_attr(obs1, "bmask", np.int32) | _extract_attr(
            obs2, "bmask", jnp.int32
        )

    if obs1.has_ormask() or obs2.has_ormask():
        new_ormask = _extract_attr(obs1, "ormask", np.int32) | _extract_attr(
            obs2, "ormask", jnp.int32
        )

    if obs1.has_noise() or obs2.has_noise():
        new_noise = _extract_attr(obs1, "noise", np.float32) + _extract_attr(
            obs2, "noise", jnp.float32
        )

    if skip_mfrac_for_second:
        if obs1.has_mfrac():
            new_mfrac = _extract_attr(obs1, "mfrac", np.float32)
    else:
        if obs1.has_mfrac() or obs2.has_mfrac():
            new_mfrac = (
                _extract_attr(obs1, "mfrac", np.float32)
                + _extract_attr(obs2, "mfrac", np.float32)
            ) / 2  # TODO: update statement

    new_meta_data.update(obs1.meta)
    new_meta_data.update(obs2.meta)

    obs = NTObservation(
        image=obs1.image + obs2.image,
        weight=new_wgt,
        bmask=new_bmask,
        ormask=new_ormask,
        noise=new_noise,
        jacobian=jax_galsim.wcs.JacobianWCS(
            dudx=obs1.jacobian.dudx,
            dudy=obs1.jacobian.dudy,
            dvdx=obs1.jacobian.dvdx,
            dvdy=obs1.jacobian.dvdy,
        ),
        psf=new_psf,
        meta=new_meta_data,  # Directly copy metadata
        mfrac=new_mfrac,
        store_pixels=getattr(obs1, "store_pixels", True),
        ignore_zero_weight=getattr(obs1, "ignore_zero_weight", True),
        jac_row0=obs1.jac_row0,
        jac_col0=obs1.jac_col0,
        jac_det=obs1.jac_det,
        jac_scale=obs1.jac_scale,
    )

    return obs


def get_jax_galsim_object_from_NT_obs(obs, kind="image", rot90=0):
    """Make an interpolated image from an ngmix obs."""
    return jax_galsim.InterpolatedImage(
        jax_galsim.ImageD(
            jnp.rot90(getattr(obs, kind).copy(), k=rot90),
            wcs=obs.jacobian,
        ),
        x_interpolant="lanczos15",
    )


def get_jax_galsim_object_from_NT_obs_nopix(obs, kind="image"):
    """Make an interpolated image from an ngmix obs w/o a pixel."""
    wcs = obs.jacobian
    return jax_galsim.Convolve(
        [
            get_jax_galsim_object_from_NT_obs(obs, kind=kind),
            jax_galsim.Deconvolve(wcs.toWorld(jax_galsim.Pixel(scale=1))),
        ]
    )


@partial(
    jax.jit,
    static_argnames=[
        "nxy",
        "nxy_psf",
        "reconv_psf_dk",
        "shears",
        "skip_obs_wide_corrections",
        "skip_obs_deep_corrections",
        "return_noshear_deep",
    ],
)
def jax_helper_metacal_wide_and_deep_psf_matched(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    reconv_psf,
    nxy,
    nxy_psf,
    reconv_psf_dk,
    shears=None,
    step=DEFAULT_STEP,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_noshear_deep=False,
):
    """Do metacalibration for a combination of wide+deep datasets."""

    # make the wide obs
    if skip_obs_wide_corrections:
        mcal_obs_wide = jax_match_psf(obs_wide, reconv_psf, nxy)
    else:
        mcal_obs_wide = jax_add_ngmix_obs(
            jax_match_psf(obs_wide, reconv_psf, nxy),
            jax_metacal_op_g1g2(obs_deep_noise, reconv_psf, 0, 0, nxy_psf=nxy_psf),
            skip_mfrac_for_second=True,
        )

    # get PSF matched noise
    # obs_wide_noise = obs_wide.copy()
    obs_wide_noise = obs_wide._replace(image=obs_wide.noise)
    wide_noise_corr = jax_match_psf(obs_wide_noise, reconv_psf, nxy)

    # now run mcal on deep
    # jax_gal_reconv_psf = get_jax_galsim_object_from_ngmix_obs_nopix(reconv_psf)
    mcal_res = jax_metacal_op_shears(
        obs_deep,
        dk=reconv_psf_dk,
        reconv_psf=reconv_psf,
        shears=shears,
        step=step,
        nxy_psf=nxy_psf,
    )

    # now add in noise corr to make it match the wide noise
    if not skip_obs_deep_corrections:
        for k in mcal_res:
            mcal_res[k] = jax_add_ngmix_obs(
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
    dk_w,
    dk_d,
    nxy,
    nxy_psf,
    shears=None,
    step=DEFAULT_STEP,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_noshear_deep=False,
):
    """Do metacalibration for a combination of wide+deep datasets."""

    # first get the biggest reconv PSF of the two
    reconv_psf = jax_get_max_gauss_reconv_psf(obs_wide, obs_deep, dk_w, dk_d, nxy)

    mcal_res = jax_helper_metacal_wide_and_deep_psf_matched(
        obs_wide=obs_wide,
        obs_deep=obs_deep,
        obs_deep_noise=obs_deep_noise,
        reconv_psf=reconv_psf,
        nxy=nxy,
        nxy_psf=nxy_psf,
        reconv_psf_dk=2 * jnp.pi / (nxy_psf * 0.2) / 4,
        shears=shears,
        step=step,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        return_noshear_deep=return_noshear_deep,
    )

    for k in mcal_res:
        mcal_res[k] = NT_to_ngmix_obs(mcal_res[k])
        mcal_res[k].psf.galsim_obj = reconv_psf

    return mcal_res
