from functools import partial

import jax
import jax.numpy as jnp
import jax_galsim

from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.observation import DFMdetObservation, DFMdetPSF


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
) -> tuple[DFMdetObservation, DFMdetObservation, DFMdetObservation]:
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


def generate_jax_galsim_multiband_sim_observations_jitted(
    key: jax.Array,
    bands: tuple[str, ...] = ("g", "r", "i"),
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
    band_flux_factors=None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
):
    """JIT-compatible multi-band simulation using JAX-Galsim.

    Parameters
    ----------
    key : jax.Array
        JAX random key
    bands : tuple of str
        Band names
    g1, g2 : float
        Shear components
    s2n : float
        Signal-to-noise ratio
    deep_noise_fac : float
        Deep field noise factor
    deep_psf_fac : float
        Deep field PSF size factor
    max_n_objs : int
        Fixed number of objects
    scale : float
        Pixel scale in arcsec/pixel
    dim : int
        Image dimension
    dim_psf : int
        PSF dimension
    buff : int
        Buffer size in pixels from edge for placing galaxies (default: 26)
    obj_flux_factor : float
        Base flux factor
    band_flux_factors : dict, optional
        Per-band flux factors
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (default: 256)

    Returns
    -------
    obs_wide_dict : dict
        Wide field observations by band
    obs_deep_dict : dict
        Deep field observations by band
    obs_deep_noise_dict : dict
        Deep noise observations by band

    """
    # Set default band flux factors
    if band_flux_factors is None:
        band_flux_factors = {"g": 0.7, "r": 1.0, "i": 0.8, "z": 0.6}

    obs_wide_dict = {}
    obs_deep_dict = {}
    obs_deep_noise_dict = {}

    # Split the key: position component (same for all bands) and noise base
    key_positions, key_noise_base = jax.random.split(key, 2)

    for band_idx, band in enumerate(bands):
        # Get band-specific flux factor
        if band in band_flux_factors:
            band_obj_flux_factor = obj_flux_factor * band_flux_factors[band]
        else:
            band_obj_flux_factor = obj_flux_factor

        # Scale S/N with flux
        band_s2n = s2n * jnp.sqrt(band_flux_factors.get(band, 1.0))

        # Create band-specific noise key
        band_noise_key = jax.random.fold_in(key_noise_base, band_idx)

        obs_wide, obs_deep, obs_deep_noise = make_jax_galsim_simple_sim_jitted(
            band_noise_key,  # Noise key (different per band)
            g1=g1,
            g2=g2,
            s2n=band_s2n,
            deep_noise_fac=deep_noise_fac,
            deep_psf_fac=deep_psf_fac,
            max_n_objs=max_n_objs,
            scale=scale,
            dim=dim,
            dim_psf=dim_psf,
            buff=buff,
            obj_flux_factor=band_obj_flux_factor,
            psf_fft_size=psf_fft_size,
            image_fft_size=image_fft_size,
            key_positions=key_positions,  # Position key (same for all bands)
        )

        obs_wide_dict[band] = obs_wide
        obs_deep_dict[band] = obs_deep
        obs_deep_noise_dict[band] = obs_deep_noise

    return obs_wide_dict, obs_deep_dict, obs_deep_noise_dict
