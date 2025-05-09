import jax
import jax.numpy as jnp


@jax.jit
def _fwhm_to_sigma(fwhm: float):
    """
    convert fwhm to sigma for a gaussian
    """
    return fwhm / 2.3548200450309493  # sig = fwhm / sqrt(8 * ln 2)


@jax.jit
def _fwhm_to_T(fwhm):
    """
    convert fwhm to T for a gaussian
    """
    sigma = _fwhm_to_sigma(fwhm)
    return 2 * sigma**2


@jax.jit
def _eval_gauss2d(pars, u, v, area):
    cen_v, cen_u, irr, irc, icc = pars[0:5]

    det = irr * icc - irc * irc
    idet = 1.0 / det
    drr = irr * idet
    drc = irc * idet
    dcc = icc * idet
    norm = 1.0 / (2 * jnp.pi * jnp.sqrt(det))

    # v->row, u->col in gauss
    vdiff = v - cen_v
    udiff = u - cen_u
    chi2 = dcc * vdiff * vdiff + drr * udiff * udiff - 2.0 * drc * vdiff * udiff

    return norm * jnp.exp(-0.5 * chi2) * area, norm
