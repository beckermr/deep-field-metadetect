"""Test multiband simulations with JAX-Galsim and metadetect."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.jax_galsim_simulation import (
    generate_jax_galsim_multiband_sim_observations_jitted,
)
from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_multi_band_deep_field_metadetect,
)


# TODO: Make a jitted core function for this.
# The return structure needs to be changed for the JAX part
def _run_single_sim_jax(
    key,
    bands: Tuple[str, ...] = ("g", "r", "i"),
    s2n: float = 20.0,
    g1: float = 0.0,
    g2: float = 0.0,
    deep_noise_fac: float = 1.0 / jnp.sqrt(10),
    deep_psf_fac: float = 1.0,
    max_n_objs: int = 10,
    scale: float = 0.2,
    dim: int = jax_dfmd_defaults.DEAFAULT_COADD_SIZE,
    dim_psf: int = jax_dfmd_defaults.DEFAULT_NXY_PSF,
    reconv_psf_dk: float = jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size: float = jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    obj_flux_factor: float = 1.0,
    band_flux_factors: Optional[Dict[str, float]] = None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    detbands: Optional[Tuple[str, ...]] = None,
    skip_obs_wide_corrections: bool = False,
    skip_obs_deep_corrections: bool = False,
) -> Dict:
    """Run a single multi-band simulation with JAX-Galsim and metadetect.

    Parameters
    ----------
    key : jax.Array
        JAX random key for reproducible simulations
    s2n : float
        Signal-to-noise ratio for simulation
    bands : tuple of str
        Band names (e.g., ("g", "r", "i"))
    g1, g2 : float
        Applied shear components
    deep_noise_fac : float
        Noise factor for deep field (1/sqrt(n_exposures))
    deep_psf_fac : float
        PSF size factor for deep field
    max_n_objs : int
        Fixed number of objects in simulation
    scale : float
        Pixel scale in arcsec/pixel
    dim : int
        Image dimension in pixels
    dim_psf : int
        PSF image dimension in pixels
    reconv_psf_dk: float
        The Fourier-space pixel scale used for reconv psf computation.
        Default: jax_dfmd_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size: int
        k image size used for reconv psf computation
        Default: jax_dfmd_defaults.DEFAULT_KIM_SIZE
    obj_flux_factor : float
        Base flux scaling factor
    band_flux_factors : dict, optional
        Per-band flux factors (e.g., {"g": 0.7, "r": 1.0, "i": 0.8})
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (default: 256)
    detbands : tuple of str, optional
        Bands to use for detection (if None, uses all bands)
    skip_obs_wide_corrections : bool
        Skip wide-field observation corrections
    skip_obs_deep_corrections : bool
        Skip deep-field observation corrections


    Returns
    -------
    result : dict
        Dictionary containing metadetect results with key "dfmdet_res"
    """
    obs_wide, obs_deep, obs_deep_noise = (
        generate_jax_galsim_multiband_sim_observations_jitted(
            key,
            bands=bands,
            g1=g1,
            g2=g2,
            s2n=s2n,
            deep_noise_fac=deep_noise_fac,
            deep_psf_fac=deep_psf_fac,
            max_n_objs=max_n_objs,
            scale=scale,
            dim=dim,
            dim_psf=dim_psf,
            obj_flux_factor=obj_flux_factor,
            band_flux_factors=band_flux_factors,
            psf_fft_size=psf_fft_size,
            image_fft_size=image_fft_size,
        )
    )

    result = jax_multi_band_deep_field_metadetect(
        obs_wide,
        obs_deep,
        obs_deep_noise,
        nxy=dim,
        nxy_psf=dim_psf,
        detbands=detbands,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )

    return result


def run_jax_sim_pair(
    key: jax.Array,
    bands: Tuple[str, ...] = ("g", "r", "i"),
    s2n: float = 20.0,
    shear_magnitude: float = 0.02,
    deep_noise_fac: float = 1.0 / jnp.sqrt(10),
    deep_psf_fac: float = 1.0,
    max_n_objs: int = 10,
    scale: float = 0.2,
    dim: int = jax_dfmd_defaults.DEAFAULT_COADD_SIZE,
    dim_psf: int = jax_dfmd_defaults.DEFAULT_NXY_PSF,
    reconv_psf_dk: float = jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size: float = jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    obj_flux_factor: float = 1.0,
    band_flux_factors: Optional[Dict[str, float]] = None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    detbands: Optional[Tuple[str, ...]] = None,
    skip_obs_wide_corrections: bool = False,
    skip_obs_deep_corrections: bool = False,
) -> Tuple[Dict, Dict]:
    """Run a pair of simulations with +/- shear for calibration.

    Parameters
    ----------
    key : jax.Array
        JAX random key
    bands : tuple of str
        Band names
    s2n : float
        Signal-to-noise ratio
    shear_magnitude : float
        Magnitude of applied shear (g1 = +/- this value)
    deep_noise_fac : float
        Deep field noise factor
    deep_psf_fac : float
        Deep field PSF size factor
    max_n_objs : int
        Number of objects
    scale : float
        Pixel scale in arcsec/pixel
    dim : int
        Image dimension in pixels
    dim_psf : int
        PSF image dimension in pixels
    reconv_psf_dk : float
        The Fourier-space pixel scale used for reconv psf computation.
        Default: jax_dfmd_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size : int
        k image size used for reconv psf computation
        Default: jax_dfmd_defaults.DEFAULT_KIM_SIZE
    obj_flux_factor : float
        Base flux scaling factor
    band_flux_factors : dict, optional
        Per-band flux factors (e.g., {"g": 0.7, "r": 1.0, "i": 0.8})
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (default: 256)
    detbands : tuple of str, optional
        Bands to use for detection (if None, uses all bands)
    skip_obs_wide_corrections : bool
        Skip wide-field observation corrections
    skip_obs_deep_corrections : bool
        Skip deep-field observation corrections

    Returns
    -------
    res_p : dict
        Results from positive shear simulation
    res_m : dict
        Results from negative shear simulation
    """
    # Positive shear
    res_p = _run_single_sim_jax(
        key,
        bands=bands,
        s2n=s2n,
        g1=shear_magnitude,
        g2=0.0,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        max_n_objs=max_n_objs,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
        obj_flux_factor=obj_flux_factor,
        band_flux_factors=band_flux_factors,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
        detbands=detbands,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    )

    # Negative shear
    res_m = _run_single_sim_jax(
        key,
        bands=bands,
        s2n=s2n,
        g1=-shear_magnitude,
        g2=0.0,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        max_n_objs=max_n_objs,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
        obj_flux_factor=obj_flux_factor,
        band_flux_factors=band_flux_factors,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
        detbands=detbands,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    )

    return res_p, res_m
