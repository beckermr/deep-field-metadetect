from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax_galsim
import ngmix
import numpy as np

from deep_field_metadetect.gaussmom.gaussmom import GaussMom
from deep_field_metadetect.gaussmom.gaussmom_core import dfmd_obs_to_gaussmom_obs
from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.observation import DFMdetObservation, DFMdetPSF


def compute_dk(pixel_scale, image_size):
    """Compute psf fourier scale based on pixel scale and PSF image dimension.
    The size is obtained from galsim.GSObject.getGoodImageSize.
    The factor 1/4 from deep_field_metadetect.metacal.get_gauss_reconv_psf_galsim.

    Parameters:
    -----------
    pixel_scale : float
        The scale of a single pixel in the image.
    image_size : int
        The dimension of the image (typically a square size).

    Returns:
    --------
    float
        The computed stepk value, which represents the Fourier-space sampling frequency.
    """
    return 2 * np.pi / (image_size * pixel_scale) / 4


def compute_kim_size(image_size):
    """Compute the size of image in the fourier space used the determine
    the reconv-psf. See jax_metacal::jax_get_gauss_reconv_psf_galsim

    Parameters:
    -----------
    image_size : int
        The dimension of the image (typically a square size).

    Returns:
    --------
    int
        The computed kim size (a power of 2).
    """
    min_size = 4 * image_size
    target_size = 2 ** (int(np.log2(min_size)) + 1)
    return int(target_size)


def compute_max_objects(field_size, pixel_scale, number_density):
    """Compute the maximum number of objects in a field given number density.

    Parameters:
    -----------
    field_size : int
        The dimension of the square field in pixels.
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    number_density : float
        Expected number density of objects per square arcminute.

    Returns:
    --------
    int
        The estimated maximum number of objects in the field.

    """

    field_size_arcmin = (field_size * pixel_scale) / 60.0
    total_area_arcmin2 = field_size_arcmin**2

    max_objects = int(number_density * total_area_arcmin2)

    return max_objects


@jax.jit
def jax_fit_gauss_mom_obs(obs, fwhm=1.2):
    """Fit an DFMdetObservation/DFMdetPSF with Gaussian moments.

    Parameters
    ----------
    obs : DFMDObservation
        The observation to fit.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.

    Returns
    -------
    res: dict-like
        The fit results.
    """
    guassmom_obs = dfmd_obs_to_gaussmom_obs(obs)
    fitter = GaussMom(fwhm)
    return fitter.go(guassmom_obs)


@jax.jit
def jax_fit_gauss_mom_obs_and_psf(obs, fwhm=1.2, psf_res=None):
    """Fit an DFMDObservation with Gaussian moments (JIT-compatible version).

    Parameters
    ----------
    obs : DFMDObservation
        The observation to fit.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.
    psf_res : dict, optional
        The PSF result dict. Default None: the PSF is fit.

    Returns
    -------
    vals : dict
        The fit results as a PyTree (dictionary) with keys:
        - wmom_flags: fit flags
        - wmom_g1: first component of ellipticity
        - wmom_g2: second component of ellipticity
        - wmom_T_ratio: T / psf_T ratio
        - wmom_psf_T: PSF T value
        - wmom_s2n: signal-to-noise ratio
    """
    fitter = GaussMom(fwhm)
    if psf_res is None:
        psf_res = fitter.go(dfmd_obs_to_gaussmom_obs(obs.psf))

    def psf_failed():
        """Return early if PSF fit failed."""
        return {
            "wmom_flags": jnp.array(ngmix.flags.NO_ATTEMPT, dtype=jnp.int32),
            "wmom_g1": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_g2": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_T_ratio": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_psf_T": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_s2n": jnp.array(jnp.nan, dtype=jnp.float64),
        }

    def psf_success():
        """Continue with observation fit if PSF fit succeeded."""
        # Fit the observation
        res = fitter.go(dfmd_obs_to_gaussmom_obs(obs))

        def obs_failed():
            """Return partial results if observation fit failed."""
            return {
                "wmom_flags": jnp.array(res.flags, dtype=jnp.int32),
                "wmom_g1": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_g2": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_T_ratio": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_psf_T": jnp.array(psf_res.T, dtype=jnp.float64),
                "wmom_s2n": jnp.array(jnp.nan, dtype=jnp.float64),
            }

        def obs_success():
            """Return full results if observation fit succeeded."""
            return {
                "wmom_flags": jnp.array(res.flags, dtype=jnp.int32),
                "wmom_g1": jnp.array(res.e[0], dtype=jnp.float64),
                "wmom_g2": jnp.array(res.e[1], dtype=jnp.float64),
                "wmom_T_ratio": jnp.array(res.T / psf_res.T, dtype=jnp.float64),
                "wmom_psf_T": jnp.array(psf_res.T, dtype=jnp.float64),
                "wmom_s2n": jnp.array(res.s2n, dtype=jnp.float64),
            }

        return jax.lax.cond(res.flags != 0, obs_failed, obs_success)

    return jax.lax.cond(psf_res.flags != 0, psf_failed, psf_success)


@jax.jit
def jax_fit_single_detection(
    mbobs,
    psf_res,
    obj_id,
    obj_x,
    obj_y,
    shear_idx,
    bmask_flag,
    mfrac_val,
    det_flag,
    fwhm=1.2,
):
    """Process a single detection and return results as a PyTree.

    This function is designed to be JIT-compatible and can be used with
    jax.vmap for parallel processing of multiple detections.

    Parameters
    ----------
    mbobs : observation
        The observation for a single detected object.
    psf_res : dict
        The PSF fit result.
    obj_id : int
        Object identifier.
    obj_x : float
        Object x position.
    obj_y : float
        Object y position.
    shear_idx : int
        Index of the shear/metadetection step.
    bmask_flag : int
        Bit mask flag value at detection position.
    mfrac_val : float
        Masked fraction value at detection position.
    det_flag : int
        Detection flag (1 = actual detection, 0 = fill value).
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.

    Returns
    -------
    result : dict (PyTree)
        Detection result with scalar JAX array values:
        - id: object identifier
        - x: x position
        - y: y position
        - mdet_step_idx: shear step index
        - bmask_flags: bit mask flags
        - mfrac: masked fraction
        - det_flag: detection flag (1 = actual, 0 = fill)
        - wmom_flags: weighted moments fit flags
        - wmom_g1: first ellipticity component
        - wmom_g2: second ellipticity component
        - wmom_T_ratio: T / psf_T ratio
        - wmom_psf_T: PSF T value
        - wmom_s2n: signal-to-noise ratio
    """
    # Fit the observation with Gaussian moments
    fres = jax_fit_gauss_mom_obs_and_psf(mbobs, fwhm=fwhm, psf_res=psf_res)

    # Return PyTree with scalar values
    return {
        "id": jnp.array(obj_id, dtype=jnp.int64),
        "x": jnp.array(obj_x, dtype=jnp.float_),
        "y": jnp.array(obj_y, dtype=jnp.float_),
        "mdet_step_idx": jnp.array(shear_idx, dtype=jnp.int32),
        "bmask_flags": jnp.array(bmask_flag, dtype=jnp.int32),
        "mfrac": jnp.array(mfrac_val, dtype=jnp.float_),
        "det_flag": jnp.array(det_flag, dtype=jnp.int32),
        "wmom_flags": fres["wmom_flags"],
        "wmom_g1": fres["wmom_g1"],
        "wmom_g2": fres["wmom_g2"],
        "wmom_T_ratio": fres["wmom_T_ratio"],
        "wmom_psf_T": fres["wmom_psf_T"],
        "wmom_s2n": fres["wmom_s2n"],
    }


@partial(
    jax.jit,
    static_argnames=("scale", "dim", "dim_psf", "psf_fft_size", "image_fft_size"),
)
def _make_jax_galsim_single_sim_jitted(
    key: jax.Array,
    psf: jax_galsim.GSObject,
    obj: jax_galsim.GSObject,
    nse: float,
    scale: float,
    dim: int,
    dim_psf: int,
    psf_fft_size: int,
    image_fft_size: int,
) -> DFMdetObservation:
    """JIT-compatible version of single sim generation.

    Similar to non-JAX _make_single_sim but returns DFMdetObservation.

    Parameters
    ----------
    key : jax.Array
        JAX random key for noise generation
    psf : jax_galsim.GSObject
        PSF object (dynamic, contains traced values)
    obj : jax_galsim.GSObject
        Galaxy object (dynamic, contains traced values)
    nse : float
        Noise standard deviation (dynamic, computed from traced values)
    scale : float
        Pixel scale (static)
    dim : int
        Image dimension (static)
    dim_psf : int
        PSF dimension (static)
    psf_fft_size : int
        FFT size for drawing PSF (static)
    image_fft_size : int
        FFT size for drawing object images (static)

    Returns
    -------
    obs : DFMdetObservation
        Observation with image, weight, noise, PSF, WCS, bmask, and mfrac
    """
    # Fix FFT size for JIT
    obj_fixed = obj.withGSParams(
        minimum_fft_size=image_fft_size, maximum_fft_size=image_fft_size
    )
    psf_fixed = psf.withGSParams(
        minimum_fft_size=psf_fft_size, maximum_fft_size=psf_fft_size
    )

    # Draw object image
    obj_image = obj_fixed.drawImage(nx=dim, ny=dim, scale=scale, method="auto").array
    noise = jax.random.normal(key, shape=(dim, dim)) * nse

    image = obj_image + noise
    weight = jnp.ones((dim, dim), dtype=jnp.float_) / nse**2
    psf_image = psf_fixed.drawImage(nx=dim_psf, ny=dim_psf, scale=scale).array
    wcs = jax_galsim.wcs.AffineTransform(
        dudx=scale,
        dudy=0.0,
        dvdx=0.0,
        dvdy=scale,
        origin=jax_galsim.PositionD(
            y=(dim + 1) / 2,
            x=(dim + 1) / 2,
        ),
    )
    psf_wcs = jax_galsim.wcs.AffineTransform(
        dudx=scale,
        dudy=0.0,
        dvdx=0.0,
        dvdy=scale,
        origin=jax_galsim.PositionD(
            y=(dim_psf + 1) / 2,
            x=(dim_psf + 1) / 2,
        ),
    )

    # Create observation
    obs = DFMdetObservation(
        image=image,
        weight=weight,
        noise=noise,
        bmask=jnp.zeros((dim, dim), dtype=jnp.int32),
        mfrac=jnp.zeros((dim, dim), dtype=jnp.float64),
        wcs=wcs,
        psf=DFMdetPSF(
            image=psf_image,
            weight=jnp.ones_like(psf_image),
            wcs=psf_wcs,
        ),
    )

    return obs


@partial(
    jax.jit,
    static_argnames=(
        "g1",
        "g2",
        "deep_psf_fac",
        "max_n_objs",
        "scale",
        "dim",
        "dim_psf",
        "buff",
        "obj_flux_factor",
        "psf_fft_size",
        "image_fft_size",
    ),
)
def make_jax_galsim_simple_sim_jitted(
    key: jax.Array,
    g1: float = 0.0,
    g2: float = 0.0,
    s2n: float = 20.0,
    deep_noise_fac: float = 1.0 / jnp.sqrt(10),
    deep_psf_fac: float = 1.0,
    max_n_objs: int = 10,
    scale: float = 0.2,
    dim: int = 53,
    dim_psf: int = 53,
    buff: int = 26,
    obj_flux_factor: float = 1.0,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    key_positions: jax.Array = None,
) -> Tuple[DFMdetObservation, DFMdetObservation, DFMdetObservation]:
    """JIT-compatible simple simulation using JAX-Galsim.

    Note: key differences compared to non-jax version:
    This function no longet uses hexgrid, so objects are randomly scattered.
    And number of objects is fixed in a field.

    Parameters
    ----------
    key : jax.Array
        JAX random key for noise (dynamic argument)
    g1 : float
        Shear component 1 (static)
    g2 : float
        Shear component 2 (static)
    s2n : float
        Signal-to-noise ratio (static)
    deep_noise_fac : float
        Deep field noise factor (static)
    deep_psf_fac : float
        Deep field PSF size factor (static)
    max_n_objs : int
        Fixed number of objects (static)
    scale : float
        Pixel scale in arcsec/pixel (static)
    dim : int
        Image dimension (static)
    dim_psf : int
        PSF dimension (static)
    buff : int
        Buffer size in pixels from edge for placing galaxies (static, default: 26)
    obj_flux_factor : float
        Flux scaling factor (static)
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (static, default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (static, default: 256)
    key_positions : jax.Array, optional
        JAX random key for galaxy positions (dynamic argument).
        If None, positions are derived from the main key.
        Use the same key_positions across bands to ensure galaxies
        are at the same locations with different noise.

    Returns
    -------
    obs_wide : DFMdetObservation
        Wide field observation
    obs_deep : DFMdetObservation
        Deep field observation
    obs_deep_noise : DFMdetObservation
        Deep noise observation
    """
    # Split keys for positions and noise
    if key_positions is None:
        _key_positions, key_wide, key_deep, key_deep_noise = jax.random.split(key, 4)
    else:
        _key_positions = key_positions
        key_wide, key_deep, key_deep_noise = jax.random.split(key, 3)

    # Generate fixed number of galaxy positions
    xyrange = dim - buff * 2.0
    shifts = (
        jax.random.uniform(
            _key_positions, shape=(max_n_objs, 2), minval=-0.5, maxval=0.5
        )
        * xyrange
        * scale
    )

    # Generate galaxies
    gal = jax_galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)

    # this loop is unrolled at compile time
    galaxy_list = []
    for i in range(max_n_objs):
        shifted_gal = gal.shift(shifts[i, 0], shifts[i, 1])
        galaxy_list.append(shifted_gal)

    gals = jax_galsim.Add(galaxy_list)

    # PSFs
    psf = jax_galsim.Moffat(beta=2.5, fwhm=0.8)
    deep_psf = jax_galsim.Moffat(beta=2.5, fwhm=0.8 * deep_psf_fac)

    objs = jax_galsim.Convolve([gals, psf])
    deep_objs = jax_galsim.Convolve([gals, deep_psf])

    # estimate noise level
    gal_psf_conv = jax_galsim.Convolve([gal, psf]).withGSParams(
        minimum_fft_size=image_fft_size, maximum_fft_size=image_fft_size
    )
    im = gal_psf_conv.drawImage(nx=dim, ny=dim, scale=scale).array
    nse = jnp.sqrt(jnp.sum(im**2)) / s2n

    # Apply flux factor
    objs = objs.withFlux(objs.flux * obj_flux_factor)
    deep_objs = deep_objs.withFlux(deep_objs.flux * obj_flux_factor)

    # Generate wide field observation
    obs_wide = _make_jax_galsim_single_sim_jitted(
        key_wide,
        psf=psf,
        obj=objs,
        nse=nse,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
    )

    # Generate deep field observation
    deep_nse = nse * deep_noise_fac
    obs_deep = _make_jax_galsim_single_sim_jitted(
        key_deep,
        psf=deep_psf,
        obj=deep_objs,
        nse=deep_nse,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
    )

    # Generate deep noise observation (no object)
    deep_objs_zero = deep_objs.withFlux(0.0)
    obs_deep_noise = _make_jax_galsim_single_sim_jitted(
        key_deep_noise,
        psf=deep_psf,
        obj=deep_objs_zero,
        nse=deep_nse,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
    )

    return obs_wide, obs_deep, obs_deep_noise
