import jax
import jax.numpy as jnp
import ngmix
import numpy as np
from ngmix.moments import MOMENTS_NAME_MAP

from deep_field_metadetect.gaussmom.gaussmom_core import (
    GaussMomData,
    GaussMomObs,
    _eval_gauss2d,
)

jax.config.update("jax_enable_x64", True)


@jax.jit
def fwhm_to_sigma(fwhm: float):
    """
    convert fwhm to sigma for a gaussian
    """
    return fwhm / 2.3548200450309493  # sig = fwhm / sqrt(8 * ln 2)


@jax.jit
def fwhm_to_T(fwhm):
    """
    convert fwhm to T for a gaussian
    """
    sigma = fwhm_to_sigma(fwhm)
    return 2 * sigma**2


@jax.jit
def create_circular_mask(umod, vmod, maxrad):
    distance_from_center = np.sqrt((umod) ** 2 + (vmod) ** 2)

    mask = jnp.where(distance_from_center <= maxrad, 1, 0)

    return mask.astype(int)


def get_weighted_moments(obs, T, maxrad=None):
    """
    Get weighted moments using this mixture as the weight, including
    e1,e2,T,s2n etc.  If you just want the raw moments use
    get_weighted_sums()

    If you want the expected fluxes, you should set the flux to the inverse
    of the normalization which is 2*pi*sqrt(det)

    Parameters
    ----------
    obs: Observation
        The Observation to compare with. See ngmix.observation.Observation
        The Observation must have a weight map set
    maxrad: float, optional
        If sent, limit moments to within the specified maximum radius
    with_higher_order: bool, optional
        If set to True, return higher order moments in the sums/sums_cov
        arrays.  See ngmix.moments.MOMENTS_NAME_MAP for a map between
        name and index.

    Returns
    -------
    result array with basic sums as well as summary statistics
    such as e1,e2,T,s2n etc.
    """

    sums, sums_cov, wsum, _ = get_weighted_sums(
        obs,
        T=T,
        maxrad=maxrad,
    )
    return get_weighted_moments_stats(obs, sums, sums_cov, wsum)


@jax.jit
def get_weighted_sums(gaussmom_obs, T, maxrad):
    """
    Compute weighted moment sums and their covariance for a 2D Gaussian model.

    This function evaluates the weighted image moments within a circular region
    defined by `maxrad`, using a Gaussian weight function with size parameter `T`.
    It returns the weighted sums, the covariance of the sums, total weight,
    and the normalization factor for the weight function.

    Parameters
    ----------
    gaussmom_obs : object
        An object containing observation data with the following attributes:
            - u, v: pixel coordinates
            - image: observed image data
            - pixelwise_wgt: inverse variance for each pixel
            - area: pixel area
    T : float
        Size (trace) of the weighting Gaussian used for computing moments.
    maxrad : float
        Maximum radius (in u, v coordinates) for including pixels in the calculation.

    Returns
    -------
    sums : jax.numpy.ndarray
        Array of shape (6,) containing the weighted image moment sums:
            [v, u, u^2 - v^2, 2uv, u^2 + v^2, 1]
    sums_cov : jax.numpy.ndarray
        Covariance matrix of shape (6, 6) for the weighted sums.
    wsum : float
        Total sum of the weights applied to the image.
    wt_norm : float
        Normalization factor for the Gaussian weight function.
    """
    vcen = 0
    ucen = 0

    mompars = [0, 0, T / 2, 0, T / 2]

    vmod = gaussmom_obs.v - vcen
    umod = gaussmom_obs.u - ucen

    var = 1.0 / (gaussmom_obs.pixelwise_wgt)
    rad2 = umod * umod + vmod * vmod

    circle_mask = jnp.where(jnp.sqrt(rad2) <= maxrad, 1, 0)

    wt_noimage, wt_norm = _eval_gauss2d(
        mompars, gaussmom_obs.u, gaussmom_obs.v, gaussmom_obs.area
    )

    wt_noimage = wt_noimage / (wt_norm * gaussmom_obs.area)
    wt_noimage = circle_mask * wt_noimage

    wdata = wt_noimage * gaussmom_obs.image

    # print(jnp.sum(umod * umod + vmod * vmod))

    F = jnp.stack(
        [
            gaussmom_obs.v,
            gaussmom_obs.u,
            umod * umod - vmod * vmod,
            2 * vmod * umod,
            rad2,
            jnp.ones_like(gaussmom_obs.v),
        ]
    )

    wsum = jnp.sum(wt_noimage)
    # res["npix"] = jnp.sum(circle_mask)

    sums = jnp.sum(F * wdata, axis=[1, 2])

    sums_cov = jnp.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            sums_cov = sums_cov.at[i, j].set(jnp.sum(wt_noimage**2 * var * F[i] * F[j]))

    return sums, sums_cov, wsum, wt_norm


# @jax.jit
def get_weighted_moments_stats(obs, sums, sums_cov, sums_norm=None):
    """Make a fitting results dict from a set of unnormalized moments.

    Parameters
    ----------
    sums : jnp.ndarray
        The array of unnormalized moments in the order [Mv, Mu, M1, M2, MT, MF].
    sums_cov : jnp.ndarray
        The array of unnormalized moment covariances.
    sums_norm : float, optional
        The sum of the moment weight function itself. This is added to the output data.
        The default of None puts in NaN.

    Returns
    -------
    res : dict
        A dictionary of results.
    """
    if sums.shape[0] not in (6, 17):
        raise ValueError("You must pass exactly 6 or 17 unnormalized moments.")
    if sums_cov.shape not in [(6, 6), (17, 17)]:
        raise ValueError("You must pass a 6x6 or 17x17 covariance matrix.")

    valid_shapes = (sums.shape[0] in (6, 17)) & (sums_cov.shape in [(6, 6), (17, 17)])
    sums = jnp.where(valid_shapes, sums, jnp.nan)
    sums_cov = jnp.where(valid_shapes, sums_cov, jnp.nan)

    mv_ind = MOMENTS_NAME_MAP["Mv"]
    mu_ind = MOMENTS_NAME_MAP["Mu"]
    mf_ind = MOMENTS_NAME_MAP["MF"]
    mt_ind = MOMENTS_NAME_MAP["MT"]
    m1_ind = MOMENTS_NAME_MAP["M1"]
    m2_ind = MOMENTS_NAME_MAP["M2"]

    res_flags = 0
    res_flagstr = ""
    res_flux = sums[mf_ind]
    res_sums = sums
    res_sums_cov = sums_cov
    res_sums_norm = sums_norm if sums_norm is not None else jnp.nan
    res_flux_flags = 0
    res_flux_flagstr = ""
    res_T_flags = 0
    res_T_flagstr = ""
    res_flux_err = jnp.nan
    res_T = jnp.nan
    res_T_err = jnp.nan
    res_s2n = jnp.nan
    res_e1 = jnp.nan
    res_e2 = jnp.nan
    res_e = jnp.array([jnp.nan, jnp.nan])
    res_e_err = jnp.array([jnp.nan, jnp.nan])
    res_e_cov = jnp.diag(jnp.array([jnp.nan, jnp.nan]))
    res_sums_err = jnp.full(6, jnp.nan)

    # Flux only
    if sums_cov[mf_ind, mf_ind] > 0:
        res_flux_err = jnp.sqrt(sums_cov[mf_ind, mf_ind])
        res_s2n = res_flux / res_flux_err
    else:
        res_flux_flags |= ngmix.flags.NONPOS_VAR

    # Flux + T only
    if sums_cov[mf_ind, mf_ind] > 0 and sums_cov[mt_ind, mt_ind] > 0:
        if sums[mf_ind] > 0:
            res_T = sums[mt_ind] / sums[mf_ind]
            res_T_err = get_ratio_error(
                sums[mt_ind],
                sums[mf_ind],
                sums_cov[mt_ind, mt_ind],
                sums_cov[mf_ind, mf_ind],
                sums_cov[mt_ind, mf_ind],
            )
        else:
            # flux <= 0.0
            res_T_flags |= ngmix.flags.NONPOS_FLUX
    else:
        res_T_flags |= ngmix.flags.NONPOS_VAR

    # now handle full flags
    if np.all(np.diagonal(sums_cov) > 0):
        res_sums_err = np.sqrt(np.diagonal(sums_cov))
    else:
        res_flags |= ngmix.flags.NONPOS_VAR

    if res_flags == 0:
        if res_flux > 0:
            if res_T > 0:
                res_e1 = sums[m1_ind] / sums[mt_ind]
                res_e2 = sums[m2_ind] / sums[mt_ind]
                res_e = jnp.array([res_e1, res_e2])

                e_err = jnp.array(
                    [
                        get_ratio_error(
                            sums[m1_ind],
                            sums[mt_ind],
                            sums_cov[m1_ind, m1_ind],
                            sums_cov[mt_ind, mt_ind],
                            sums_cov[m1_ind, mt_ind],
                        ),
                        get_ratio_error(
                            sums[m2_ind],
                            sums[mt_ind],
                            sums_cov[m2_ind, m2_ind],
                            sums_cov[mt_ind, mt_ind],
                            sums_cov[m2_ind, mt_ind],
                        ),
                    ]
                )

                if jnp.all(jnp.isfinite(e_err)):
                    res_e_err = e_err
                    res_e_cov = jnp.diag(e_err**2)
                else:
                    # bad e_err
                    res_flags |= ngmix.flags.NONPOS_SHAPE_VAR
            else:
                # T <= 0.0
                res_flags |= ngmix.flags.NONPOS_SIZE
        else:
            # flux <= 0.0
            res_flags |= ngmix.flags.NONPOS_FLUX

    pars = jnp.array(
        [
            sums[mv_ind],
            sums[mu_ind],
            res_e1,
            res_e2,
            res_T,
            res_flux,
        ]
    )

    res_flagstr = ngmix.flags.get_flags_str(res_flags)
    res_flux_flagstr = ngmix.flags.get_flags_str(res_flux_flags)
    res_T_flagstr = ngmix.flags.get_flags_str(res_T_flags)

    res = GaussMomData(
        obs=obs,
        flags=res_flags,
        flagstr=res_flagstr,
        wsum=sums_norm,
        flux=res_flux,
        sums=res_sums,
        sums_cov=res_sums_cov,
        sums_norm=res_sums_norm,
        flux_flags=res_flux_flags,
        flux_flagstr=res_flux_flagstr,
        T_flags=res_T_flags,
        T_flagstr=res_T_flagstr,
        flux_err=res_flux_err,
        T=res_T,
        T_err=res_T_err,
        s2n=res_s2n,
        e1=res_e1,
        e2=res_e2,
        e=res_e,
        e_err=res_e_err,
        e_cov=res_e_cov,
        sums_err=res_sums_err,
        pars=pars,
    )

    # _add_moments_by_name(res) # shouldn't modify sums or sums_cov

    return res


def _add_moments_by_name(res):  # TODO
    sums = res["sums"]
    sums_cov = res["sums_cov"]

    mf_ind = MOMENTS_NAME_MAP["MF"]
    fsum = sums[mf_ind]
    fsum_err = np.sqrt(sums_cov[mf_ind, mf_ind])

    # add in named sums normalized by flux sum (weight * image).sum()
    # we don't store flags or errors for these
    mkeys = list(MOMENTS_NAME_MAP.keys())
    for name in mkeys:
        print(name)
        ind = MOMENTS_NAME_MAP[name]
        if ind > sums.size - 1:
            continue

        err_name = f"{name}_err"

        if name in ["MF", "M00"]:
            res[name] = fsum
            res[err_name] = fsum_err
        else:
            if fsum > 0:
                res[name] = sums[ind] / fsum
                res[err_name] = get_ratio_error(
                    sums[ind],
                    sums[mf_ind],
                    sums_cov[ind, ind],
                    sums_cov[mf_ind, mf_ind],
                    sums_cov[ind, mf_ind],
                )
            else:
                res[name] = np.nan
                res[err_name] = np.nan


@jax.jit
def get_ratio_error(a, b, var_a, var_b, cov_ab):
    """
    Compute the error on the ratio a / b using JAX.
    """
    var = get_ratio_var(a, b, var_a, var_b, cov_ab)

    var = jnp.clip(var, 0.0, jnp.inf)
    error = jnp.sqrt(var)
    return error


@jax.jit
def get_ratio_var(a, b, var_a, var_b, cov_ab):
    """
    Compute the variance of (a/b).
    """

    # Ensure no division by zero
    b = jnp.where(
        b == 0, 1e-100, b
    )  # TODO: This does not raise a value error like ngmix

    rsq = (a / b) ** 2

    var = rsq * (var_a / a**2 + var_b / b**2 - 2 * cov_ab / (a * b))
    return var


def eval_gaussian_moments(gaussmom_obs: GaussMomObs, fwhm: float, maxrad: float):
    T = fwhm_to_T(fwhm=fwhm)
    res = get_weighted_moments(obs=gaussmom_obs, T=T, maxrad=maxrad)
    # e1e2T = mom2e(res[2], res[3], res[4])

    area = gaussmom_obs.area
    fac = 1 / area

    res._replace(
        flux=res.flux * fac,
        flux_err=res.flux_err * fac,
        pars=res.pars.at[5].set(res.pars[5] * fac),
        sums=res.sums * fac,
        sums_cov=res.sums_cov * fac**2,
        sums_norm=res.sums_norm * fac,
        wsum=res.wsum * fac,
        sums_err=res.sums_err * fac,
    )

    return res
