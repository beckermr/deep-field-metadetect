from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ngmix
import numpy as np
from jax import tree_util
from ngmix.moments import MOMENTS_NAME_MAP

from deep_field_metadetect.gaussmom.gaussmom_core import (
    GaussMomData,
    GaussMomObs,
)
from deep_field_metadetect.gaussmom.stats import get_ratio_var
from deep_field_metadetect.gaussmom.utils import (
    _eval_gauss2d,
    _fwhm_to_T,
)


@jax.jit
def _set_fluxerr_s2n_flux_flags(res_flux, sums_cov, res_flux_flags, mf_ind):
    """Compute the flux error and signal-to-noise ratio (S/N),
    and update measurement flags

    Parameters
    ----------
    res_flux : float
        The measured flux value
    sums_cov : ndarray
        Covariance matrix for the moment sums
    res_flux_flags : int
        Initial bitmask flags associated with the flux measurement
    mf_ind : int
        Index in `sums` corresponding to mf

    Returns
    -------
    res_flux_err : jnp.float
        Estimated uncertainty on the flux, or NaN if the variance
        is non-positive
    res_s2n : jnp.float
        Signal-to-noise ratio (flux divided by its error),
        or NaN if the variance is non-positive
    res_flux_flags : int
        Updated flags indicating the validity of the flux error computation
        If the flux variance is non-positive,
        `ngmix.flags.NONPOS_VAR` is set in `res_flux_flags`.
    """

    def pos_var_fn(_):
        res_flux_err = jnp.sqrt(sums_cov[mf_ind, mf_ind])
        res_s2n = res_flux / res_flux_err
        return res_flux_err, res_s2n, res_flux_flags

    def nonpos_var_fn(_):
        res_flux_flags_new = res_flux_flags | ngmix.flags.NONPOS_VAR
        return jnp.nan, jnp.nan, res_flux_flags_new

    return jax.lax.cond(
        sums_cov[mf_ind, mf_ind] > 0,
        pos_var_fn,
        nonpos_var_fn,
        operand=None,
    )


@jax.jit
def _set_T_Terr_Tflags(sums, sums_cov, mt_ind, mf_ind, res_T_flags):
    """
    Compute the size parameter `T`, its associated error `Terr`,
    and update measurement flags.

    Parameters
    ----------
    sums : jax.numpy.ndrray
        Computer moment sums
    sums_cov : jax.numpy.ndrray
        Covariance matrix associated with the `sums` values
    mt_ind : int
        Index in `sums` corresponding to mt
    mf_ind : int
        Index in `sums` corresponding to mf
    res_T_flags : int
        Initial bitmask flags for the `T` measurement

    Returns
    -------
    res_T : float
        The computed T value or NaN if computation was invalid
    res_T_err : float
        The estimated uncertainty on T, or NaN if computation was invalid
    res_T_flags : int
        Updated measurement flags.
        Flags such as `ngmix.flags.NONPOS_FLUX`
        and `ngmix.flags.NONPOS_VAR` are set when
        flux or variance values are non-positive, respectively.
    """

    def pos_var_fn(_):
        def pos_flux_fn(_):
            res_T = sums[mt_ind] / sums[mf_ind]
            res_T_err = get_ratio_error(
                sums[mt_ind],
                sums[mf_ind],
                sums_cov[mt_ind, mt_ind],
                sums_cov[mf_ind, mf_ind],
                sums_cov[mt_ind, mf_ind],
            )
            return res_T, res_T_err, res_T_flags

        def nonpos_flux_fn(_):
            new_res_T_flags = res_T_flags | ngmix.flags.NONPOS_FLUX
            return jnp.nan, jnp.nan, new_res_T_flags

        return jax.lax.cond(
            sums[mf_ind] > 0,
            pos_flux_fn,
            nonpos_flux_fn,
            operand=None,
        )

    def nonpos_var_fn(_):
        new_res_T_flags = res_T_flags | ngmix.flags.NONPOS_VAR
        return jnp.nan, jnp.nan, new_res_T_flags

    return jax.lax.cond(
        (sums_cov[mf_ind, mf_ind] > 0) & (sums_cov[mt_ind, mt_ind] > 0),
        pos_var_fn,
        nonpos_var_fn,
        operand=None,
    )


@jax.jit
def _diag_all_true(sums_cov, res_flags):
    """Validates that all diagonal elements of the covariance matrix are positive

    Parameters
    ----------
    sums_cov : jax.numpy.ndarray
        Covariance matrix corresponding to 'sums'
    res_flags : int
        Bitwise flag indicating the current computation status.

    Returns
    -------
    res_sums_err : jax.numpy.ndarray
        1D array of square root of diagonal of `sums_cov`,
        or NaNs if any diagonal element is non-positive.
    res_flags : int
        Updated flags.
        If any variance is non-positive,
        `NONPOS_VAR` is added to the flag bitmask.
    """

    def diag_all_true(_):
        return jnp.sqrt(jnp.diagonal(sums_cov)), res_flags

    def diag_not_all_true(_):
        return jnp.full(6, jnp.nan), res_flags | ngmix.flags.NONPOS_VAR

    return jax.lax.cond(
        jnp.all(jnp.diagonal(sums_cov) > 0),
        diag_all_true,
        diag_not_all_true,
        operand=None,
    )


@jax.jit
def _compute_shape_params(
    sums, sums_cov, m1_ind, m2_ind, mt_ind, res_T, res_flux, res_flags
):
    """
    Computes shape parameters (ellipticities) and their uncertainties
    using JAX-compatible control flow.

    Parameters
    ----------
    sums : jax.numpy.ndarray
        Array of image moment sums
    sums_cov : jax.numpy.ndarray
        Covariance matrix corresponding to `sums`
    m1_ind : int
        Index into `sums` for m1
    m2_ind : int
        Index into `sums` for m2
    mt_ind : int
        Index into `sums` for the trace
    res_T : float
        Estimated size (trace of second moment matrix)
    res_flux : float
        Estimated flux
    res_flags : int
        Bitwise flag indicating prior computation status.
        If non-zero, computation is skipped

    Returns
    -------
    res_e : jax.numpy.ndarray
        Estimated ellipticity components [e1, e2] or NaNs if computation is invalid
    res_e_err : jax.numpy.ndarray
        Estimated errors for ellipticity components, or NaNs if invalid
    res_e_cov : jax.numpy.ndarray
        2x2 diagonal covariance matrix for ellipticity components,
        or diag NaNs if invalid
    res_flags : int
        Updated flag bitmask.
        Set if inputs are invalid (refer to the notes below)
    """

    def valid_flags_fn(_):
        def flux_positive_fn(_):
            # If flux > 0 then check if T > 0
            def T_positive_fn(_):
                # Compute the two shape estimates
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

                def e_err_finite_fn(_):
                    res_e_cov = jnp.diag(e_err**2)
                    return res_e, e_err, res_e_cov, res_flags

                def e_err_nonfinite_fn(_):
                    new_flags = res_flags | ngmix.flags.NONPOS_SHAPE_VAR
                    return (
                        res_e,
                        e_err,
                        jnp.diag(jnp.array([jnp.nan, jnp.nan])),
                        new_flags,
                    )

                return jax.lax.cond(
                    jnp.all(jnp.isfinite(e_err)),
                    e_err_finite_fn,
                    e_err_nonfinite_fn,
                    operand=(res_e, e_err),
                )

            def T_nonpositive_fn(_):
                # When T <= 0, set output values to deafault and update flags.
                return (
                    jnp.array([jnp.nan, jnp.nan]),
                    jnp.array([jnp.nan, jnp.nan]),
                    jnp.diag(jnp.array([jnp.nan, jnp.nan])),
                    res_flags | ngmix.flags.NONPOS_SIZE,
                )

            return jax.lax.cond(
                res_T > 0, T_positive_fn, T_nonpositive_fn, operand=None
            )

        def flux_nonpositive_fn(_):
            # When res_flux <= 0, set outputs to deafault and update flags
            return (
                jnp.array([jnp.nan, jnp.nan]),
                jnp.array([jnp.nan, jnp.nan]),
                jnp.diag(jnp.array([jnp.nan, jnp.nan])),
                res_flags | ngmix.flags.NONPOS_FLUX,
            )

        return jax.lax.cond(
            res_flux > 0, flux_positive_fn, flux_nonpositive_fn, operand=None
        )

    # If flags are already nonzero, change nothing
    def invalid_flags_fn(_):
        return (
            jnp.array([jnp.nan, jnp.nan]),
            jnp.array([jnp.nan, jnp.nan]),
            jnp.diag(jnp.array([jnp.nan, jnp.nan])),
            res_flags,
        )

    return jax.lax.cond(
        res_flags == 0,
        valid_flags_fn,
        invalid_flags_fn,
        operand=None,
    )


@jax.jit
def get_weighted_moments_stats(
    gaussmom_obs: GaussMomObs, sums, sums_cov, npix, sums_norm=None
):
    """Make a fitting results dict from a set of unnormalized moments.

    Note this function does not check shapes of sums and sums_cov
    as in ngmix

    Parameters
    ----------
    gaussmom_obs : GaussmomObs object
        see deepfield_meta_detect.gaussmom.GaussMomObs
    sums : jnp.ndarray
        The array of unnormalized moments in the order [Mv, Mu, M1, M2, MT, MF].
    sums_cov : jnp.ndarray
        The array of unnormalized moment covariances.
    npix: int
        number of pixels within maxrad
    sums_norm : float, optional
        The sum of the moment weight function itself. This is added to the output data.
        The default of None puts in NaN.

    Returns
    -------
    res: GaussmomData Object
        Results
    """
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

    res_flux = sums[mf_ind]
    res_sums = sums
    res_sums_cov = sums_cov
    res_sums_norm = sums_norm if sums_norm is not None else jnp.nan
    res_flux_flags = 0

    res_T_flags = 0

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
    res_flux_err, res_s2n, res_flux_flags = _set_fluxerr_s2n_flux_flags(
        res_flux, sums_cov, res_flux_flags, mf_ind
    )

    # compute T
    res_T, res_T_err, res_T_flags = _set_T_Terr_Tflags(
        sums, sums_cov, mt_ind, mf_ind, res_T_flags
    )

    # Check if all diag elements of cov is True
    res_sums_err, res_flags = _diag_all_true(sums_cov, res_flags)

    # If flags are still 0, then check flux, then T
    # then compute the shape params if everything is ok
    res_e, res_e_err, res_e_cov, res_flags = _compute_shape_params(
        sums, sums_cov, m1_ind, m2_ind, mt_ind, res_T, res_flux, res_flags
    )

    res_e1 = res_e[0]
    res_e2 = res_e[1]

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

    res = GaussMomData(
        obs=gaussmom_obs,
        npix=npix,
        flags=res_flags,
        wsum=sums_norm,
        flux=res_flux,
        sums=res_sums,
        sums_cov=res_sums_cov,
        sums_norm=res_sums_norm,
        flux_flags=res_flux_flags,
        T_flags=res_T_flags,
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

    res = _add_moments_by_name(res)

    return res


@jax.jit
def _add_moments_by_name(res):
    updates = {}
    mf_ind = MOMENTS_NAME_MAP["MF"]

    fsum = res.sums[mf_ind]
    fsum_err = jnp.sqrt(res.sums_cov[mf_ind, mf_ind])

    mkeys = list(MOMENTS_NAME_MAP.keys())
    for name in mkeys:
        ind = MOMENTS_NAME_MAP[name]
        if ind > 5:
            continue

        err_name = f"{name}_err"

        if name in ["MF", "M00"]:
            updates[name] = fsum
            updates[err_name] = fsum_err

        else:

            def pos_fn(_):
                val = res.sums[ind] / fsum
                err = get_ratio_error(
                    res.sums[ind],
                    res.sums[mf_ind],
                    res.sums_cov[ind, ind],
                    res.sums_cov[mf_ind, mf_ind],
                    res.sums_cov[ind, mf_ind],
                )
                return val, err

            def nan_fn(_):
                return jnp.nan, jnp.nan

            val, err = jax.lax.cond(
                res.flux > 0,
                pos_fn,
                nan_fn,
                operand=None,
            )
            updates[name] = val
            updates[err_name] = err

    res = res._replace(**updates)
    return res


@jax.jit
def get_ratio_error(a, b, var_a, var_b, cov_ab):
    """
    Compute the error on the ratio a / b using JAX.
    """
    var = get_ratio_var(a, b, var_a, var_b, cov_ab)

    var = jnp.clip(var, 0.0, jnp.inf)
    error = jnp.sqrt(var)
    return error


@dataclass
class GaussMom:
    fwhm: float
    with_higher_order: bool = False

    def _set_mompars(self, gaussmom_obs: GaussMomObs):
        T = _fwhm_to_T(self.fwhm)
        mompars = [0, 0, T / 2, 0, T / 2]

        wt_noimage, wt_norm = _eval_gauss2d(
            mompars, gaussmom_obs.u, gaussmom_obs.v, gaussmom_obs.area
        )

        self.weight = wt_noimage / wt_norm

    def go(self, gaussmom_obs, maxrad=None, with_higher_order: bool = False):
        if maxrad is None:
            T = _fwhm_to_T(fwhm=self.fwhm)
            sigma = np.sqrt(T / 2)
            maxrad = 100 * sigma
        res = self._measure_moments(
            gaussmom_obs=gaussmom_obs,
            maxrad=maxrad,
            with_higher_order=with_higher_order,
        )
        return res

    def _measure_moments(
        self, gaussmom_obs: GaussMomObs, maxrad: float, with_higher_order: bool = False
    ):
        res = self.get_weighted_moments(
            gaussmom_obs=gaussmom_obs,
            maxrad=maxrad,
            with_higher_order=with_higher_order,
        )

        area = gaussmom_obs.area
        fac = 1 / area

        res = res._replace(
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

    def get_weighted_moments(
        self, gaussmom_obs: GaussMomObs, maxrad: float, with_higher_order: bool = False
    ):
        """
        Get weighted moments using this mixture as the weight, including
        e1,e2,T,s2n etc.  If you just want the raw moments use
        get_weighted_sums()

        If you want the expected fluxes, you should set the flux to the inverse
        of the normalization which is 2*pi*sqrt(det)

        Parameters
        ----------
        gaussmom_obs : GaussmomObs object
            see deepfield_meta_detect.gaussmom.GaussMomObs
        maxrad: float, optional
            If sent, limit moments to within the specified maximum radius
        with_higher_order: bool, optional
            If set to True, return higher order moments in the sums/sums_cov
            arrays.  [Yet to be implemented]

        Returns
        -------
        res: GaussmomData Object
            results
        """

        if with_higher_order:
            raise ValueError("Not yet implimented")

        sums, sums_cov, wsum, npix = self.get_weighted_sums(
            gaussmom_obs,
            maxrad=maxrad,
        )
        return get_weighted_moments_stats(
            gaussmom_obs=gaussmom_obs,
            sums=sums,
            sums_cov=sums_cov,
            npix=npix,
            sums_norm=wsum,
        )

    @jax.jit
    def get_weighted_sums(self, gaussmom_obs, maxrad):
        """
        Compute weighted moment sums and their covariance for a 2D Gaussian model.

        This function evaluates the weighted image moments within a circular region
        defined by `maxrad`, using a Gaussian weight function with size parameter `T`.
        It returns the weighted sums, the covariance of the sums, total weight,
        and the normalization factor for the weight function.

        Parameters
        ----------
        gaussmom_obs : GaussmomObs object
            see deepfield_meta_detect.gaussmom.GaussMomObs
        maxrad : float
            Maximum radius (in u, v coordinates) for including pixels
            in the calculation.

        Returns
        -------
        sums : jax.numpy.ndarray
            Array of shape (6,) containing the weighted image moment sums:
                [v, u, u^2 - v^2, 2uv, u^2 + v^2, 1]
        sums_cov : jax.numpy.ndarray
            Covariance matrix of shape (6, 6) for the weighted sums.
        wsum : float
            Total sum of the weights applied to the image.
        npix: int
            Number of pixels within maxrad
        """
        vcen = 0
        ucen = 0

        vmod = gaussmom_obs.v - vcen
        umod = gaussmom_obs.u - ucen

        var = 1.0 / (gaussmom_obs.pixelwise_wgt)
        rad2 = umod * umod + vmod * vmod

        circle_mask = jnp.where(jnp.sqrt(rad2) <= maxrad, 1, 0)

        self._set_mompars(gaussmom_obs=gaussmom_obs)
        wt_noimage = circle_mask * self.weight

        wdata = wt_noimage * gaussmom_obs.image

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
        npix = jnp.sum(circle_mask)

        sums = jnp.sum(F * wdata, axis=[1, 2])

        sums_cov = jnp.zeros((6, 6))

        for i in range(6):
            for j in range(6):
                sums_cov = sums_cov.at[i, j].set(
                    jnp.sum(wt_noimage**2 * var * F[i] * F[j])
                )

        return sums, sums_cov, wsum, npix

    def tree_flatten(self):
        children = (self.fwhm, self.with_higher_order)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fwhm, with_higher_order = children
        obj = cls(fwhm=fwhm, with_higher_order=with_higher_order)

        return obj


tree_util.register_pytree_node_class(GaussMom)
