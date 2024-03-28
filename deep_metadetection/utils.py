import galsim
import ngmix
import numpy as np


def _make_single_sim(*, rng, psf, obj, nse, dither, scale, dim):
    cen = (dim - 1) / 2

    im = obj.drawImage(nx=dim, ny=dim, offset=dither, scale=scale).array
    im += rng.normal(size=im.shape, scale=nse)

    psf_im = psf.drawImage(nx=dim, ny=dim, scale=scale).array

    jac = ngmix.DiagonalJacobian(scale=scale, row=cen + dither[1], col=cen + dither[0])
    psf_jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im) / nse**2,
        jacobian=jac,
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
        noise=rng.normal(size=im.shape, scale=nse),
    )
    return obs


def make_simple_sim(
    *,
    seed,
    g1,
    g2,
    s2n,
    deep_noise_fac,
    deep_psf_fac,
    scale=0.2,
    dim=53,
    obj_flux_factor=1,
):
    """Make a simple simulation for testing deep-field metadetection.

    Parameters
    ----------
    seed : int
        The random seed.
    g1 : float
        The shear component 1.
    g2 : float
        The shear component 2.
    s2n : float
        The signal-to-noise ratio of the object.
    deep_noise_fac : float
        The factor by which to change the noise standard deviation in the deep-field.
    deep_psf_fac : float
        The factor by which to change the Moffat FWHM in the deep-field.
    scale : float, optional
        The pixel scale.
    dim : int, optional
        The image dimension.
    obj_flux_factor : float, optional
        The factor by which to change the object flux.

    Returns
    -------
    obs_wide : ngmix.Observation
        The wide-field observation.
    obs_deep : ngmix.Observation
        The deep-field observation.
    obs_deep_noise : ngmix.Observation
        The deep-field observation with noise but no object.
    """
    rng = np.random.RandomState(seed=seed)

    gal = galsim.Exponential(half_light_radius=0.7).shear(g1=g1, g2=g2)
    psf = galsim.Moffat(beta=2.5, fwhm=0.8)
    deep_psf = galsim.Moffat(beta=2.5, fwhm=0.8 * deep_psf_fac)
    obj = galsim.Convolve([gal, psf])
    deep_obj = galsim.Convolve([gal, deep_psf])

    # estimate noise level
    dither = np.zeros(2)
    im = obj.drawImage(nx=dim, ny=dim, offset=dither, scale=scale).array
    nse = np.sqrt(np.sum(im**2)) / s2n

    # apply the flux factor now that we have the noise level
    obj *= obj_flux_factor
    deep_obj *= obj_flux_factor

    dither = rng.uniform(size=2, low=-0.5, high=0.5)
    obs_wide = _make_single_sim(
        rng=rng,
        psf=psf,
        obj=obj,
        nse=nse,
        dither=dither,
        scale=scale,
        dim=dim,
    )

    obs_deep = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_obj,
        nse=nse * deep_noise_fac,
        dither=dither,
        scale=scale,
        dim=dim,
    )

    obs_deep_noise = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_obj * 0,
        nse=nse * deep_noise_fac,
        dither=dither,
        scale=scale,
        dim=dim,
    )

    return obs_wide, obs_deep, obs_deep_noise
