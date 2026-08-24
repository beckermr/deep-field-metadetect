from functools import partial

import jax
import jax.numpy as jnp
import jax_galsim

from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.observation import (
    DFMdetMultiBandObsList,
    DFMdetObservation,
    DFMdetObsList,
    DFMdetPSF,
)


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
        "max_n_objs",
        "scale",
        "dim",
        "dim_psf",
        "buff",
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
        Shear component 1
    g2 : float
        Shear component 2
    s2n : float
        Signal-to-noise ratio
    deep_noise_fac : float
        Deep field noise factor
    deep_psf_fac : float
        Deep field PSF size factor
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
        Flux scaling factor
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


@partial(
    jax.jit,
    static_argnames=(
        "bands",
        "max_n_objs",
        "scale",
        "dim",
        "dim_psf",
        "buff",
        "psf_fft_size",
        "image_fft_size",
    ),
)
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
    band_flux_factors: tuple[float, ...] = None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
):
    """JIT-compatible multi-band simulation using JAX-Galsim.

    Parameters
    ----------
    key : jax.Array
        JAX random key
    bands : tuple of str
        Band names (static)
    g1, g2 : float
        Shear components
    s2n : float
        Signal-to-noise ratio
    deep_noise_fac : float
        Deep field noise factor
    deep_psf_fac : float
        Deep field PSF size factor
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
        Base flux factor
    band_flux_factors : tuple of float, optional
        Per-band flux factors as a tuple in the same order as bands.
        Must have the same length as bands. If None, all bands use factor 1.0.
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (static, default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (static, default: 256)

    Returns
    -------
    mb_obs_wide : DFMdetMultiBandObsList
        Wide field observations for all bands
    mb_obs_deep : DFMdetMultiBandObsList
        Deep field observations for all bands
    mb_obs_deep_noise : DFMdetMultiBandObsList
        Deep noise observations for all bands

    """
    # If band_flux_factors not provided, use 1.0 for all bands
    # Since bands is static, this will be resolved at compile time
    if band_flux_factors is None:
        band_flux_factors = tuple(1.0 for _ in bands)

    obs_wide_list = []
    obs_deep_list = []
    obs_deep_noise_list = []

    # Split the key: position component (same for all bands) and noise base
    key_positions, key_noise_base = jax.random.split(key, 2)

    # bands is static: loop unroll
    for band_idx in range(len(bands)):
        # Get band-specific flux factor from tuple
        band_flux_fac = band_flux_factors[band_idx]
        band_obj_flux_factor = obj_flux_factor * band_flux_fac

        # Scale S/N with flux
        band_s2n = s2n * jnp.sqrt(band_flux_fac)

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

        obs_wide_list.append(DFMdetObsList([obs_wide]))
        obs_deep_list.append(DFMdetObsList([obs_deep]))
        obs_deep_noise_list.append(DFMdetObsList([obs_deep_noise]))

    # Convert to MultiBandObsList
    mb_obs_wide = DFMdetMultiBandObsList(obs_wide_list)
    mb_obs_deep = DFMdetMultiBandObsList(obs_deep_list)
    mb_obs_deep_noise = DFMdetMultiBandObsList(obs_deep_noise_list)

    return mb_obs_wide, mb_obs_deep, mb_obs_deep_noise


@partial(
    jax.jit,
    static_argnames=(
        "bands",
        "n_galaxies",
        "scale",
        "dim",
        "dim_psf",
        "buff",
        "psf_fft_size",
        "image_fft_size",
    ),
)
def make_jax_cosmos_field_jitted(
    key: jax.Array,
    mag_bulge: jax.Array,
    mag_disk: jax.Array,
    radius_bulge: jax.Array,
    radius_disk: jax.Array,
    bulge_axratio: jax.Array,
    disk_axratio: jax.Array,
    angle_bd: jax.Array,
    noise_std: tuple[float, ...],
    psf_fwhm: tuple[float, ...],
    zeropoint: tuple[float, ...],
    bands: tuple[str, ...] = ("g", "r", "i", "z", "y"),
    deep_noise_fac: float = 1.0 / jnp.sqrt(10),
    deep_psf_fac: float = 1.0,
    n_galaxies: int = 10,
    scale: float = 0.2,
    dim: int = jax_dfmd_defaults.DEAFAULT_COADD_SIZE,
    dim_psf: int = jax_dfmd_defaults.DEFAULT_NXY_PSF,
    buff: int = 26,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
):
    """JIT-compatible COSMOS sims with bulge+disk galaxies from catalog.

    Parameters
    ----------
    key : jax.Array
        JAX random key for sampling galaxies and noise generation
    mag_bulge : jax.Array
        Array of shape (n_catalog, n_bands) with AB magnitudes for bulge component
    mag_disk : jax.Array
        Array of shape (n_catalog, n_bands) with AB magnitudes for disk component
    radius_bulge : jax.Array
        Array of shape (n_catalog,) with bulge half-light radii in arcsec
    radius_disk : jax.Array
        Array of shape (n_catalog,) with disk half-light radii in arcsec
    bulge_axratio : jax.Array
        Array of shape (n_catalog,) with bulge axis ratios (b/a)
    disk_axratio : jax.Array
        Array of shape (n_catalog,) with disk axis ratios (b/a)
    angle_bd : jax.Array
        Array of shape (n_catalog,) with position angles in degrees
    noise_std : tuple of float
        Noise standard deviation (in counts) for each band, in the same order as bands.
        Example: (5.0, 4.5, 4.0, 3.5, 3.0) for different band depths
    psf_fwhm : tuple of float
        PSF FWHM in arcsec for each band, in the same order as bands.
        Example: (0.8, 0.7, 0.6, 0.7, 0.7) for HSC (g, r, i, z, y)
    zeropoint : tuple of float
        Magnitude zeropoint for flux conversion for each band.
        flux = 10^((zeropoint - mag) / 2.5)
        Example: (27.0, 27.5, 27.8, 27.6, 27.4) for different band depths
    bands : tuple of str
        Band names (static, default: ("g", "r", "i", "z", "y"))
    deep_noise_fac : float
        Deep field noise factor (default: 1/√10)
    deep_psf_fac : float
        Deep field PSF size factor (default: 1.0)
    n_galaxies : int
        Number of galaxies to sample from catalog (static, default: 10)
    scale : float
        Pixel scale in arcsec/pixel (static, default: 0.2)
    dim : int
        Image dimension in pixels (static, default: 53)
    dim_psf : int
        PSF dimension in pixels (static, default: 53)
    buff : int
        Buffer size in pixels from edge for placing galaxies (static, default: 26)
    psf_fft_size : int
        FFT size for drawing PSFs (static)
    image_fft_size : int
        FFT size for drawing object images (static)

    Returns
    -------
    mb_obs_wide : DFMdetMultiBandObsList
        Wide field observations for all bands
    mb_obs_deep : DFMdetMultiBandObsList
        Deep field observations for all bands
    mb_obs_deep_noise : DFMdetMultiBandObsList
        Deep noise observations for all bands

    Notes
    -----
    - All input arrays should be pre-filtered to contain only valid galaxies
    - Galaxies are randomly sampled and positioned uniformly in the field
    - Disk modeled as Exponential profile (Sersic n=1)
    - Bulge: Spergel(nu=-0.6) profile (approximates de Vaucouleurs n=4)
    - Axis ratios are clipped to [0, 1] to handle catalog edge cases
    """
    # Split keys
    key_sample, key_positions, key_noise_base = jax.random.split(key, 3)

    # Sample galaxy indices from catalog
    n_catalog = mag_bulge.shape[-2]  # No static indexing (for vmap)
    galaxy_indices = jax.random.choice(
        key_sample, n_catalog, shape=(n_galaxies,), replace=False
    )

    # Sample gal properties
    sampled_mag_bulge = mag_bulge[galaxy_indices]  # (n_galaxies, n_bands)
    sampled_mag_disk = mag_disk[galaxy_indices]
    sampled_r_bulge = radius_bulge[galaxy_indices]  # (n_galaxies,)
    sampled_r_disk = radius_disk[galaxy_indices]
    sampled_q_bulge = bulge_axratio[galaxy_indices]
    sampled_q_disk = disk_axratio[galaxy_indices]
    sampled_angle = angle_bd[galaxy_indices]

    # Generate gal positions
    xyrange = dim - buff * 2.0
    shifts = (
        jax.random.uniform(
            key_positions, shape=(n_galaxies, 2), minval=-0.5, maxval=0.5
        )
        * xyrange
        * scale
    )

    # Prepare lists for multi-band observations
    obs_wide_list = []
    obs_deep_list = []
    obs_deep_noise_list = []

    for band_idx in range(len(bands)):
        # Get per-band parameters from tuples
        band_psf_fwhm = psf_fwhm[band_idx]
        band_zeropoint = zeropoint[band_idx]

        psf = jax_galsim.Moffat(beta=2.5, fwhm=band_psf_fwhm)
        deep_psf = jax_galsim.Moffat(beta=2.5, fwhm=band_psf_fwhm * deep_psf_fac)

        # Get magnitudes for this band
        band_mags_bulge = sampled_mag_bulge[:, band_idx]  # (n_galaxies,)
        band_mags_disk = sampled_mag_disk[:, band_idx]  # (n_galaxies,)

        # Convert magnitudes to flux using band-specific zeropoint
        band_fluxes_bulge = jnp.power(10.0, (band_zeropoint - band_mags_bulge) / 2.5)
        band_fluxes_disk = jnp.power(10.0, (band_zeropoint - band_mags_disk) / 2.5)

        # Create galaxies for this band
        galaxy_list = []
        for i in range(n_galaxies):
            # Create disk component (Exponential profile, Sersic n=1)
            disk = jax_galsim.Exponential(
                half_light_radius=sampled_r_disk[i],
            )
            # Apply ellipticity to disk (axis ratio -> shear)
            # TODO: note sure why there are q>1 in cosmos. For now clip values.
            q_disk = jnp.clip(sampled_q_disk[i], 0.0, 1.0)
            g_disk = (1.0 - q_disk) / (1.0 + q_disk)
            disk = disk.shear(
                g=g_disk, beta=0.0 * jax_galsim.degrees
            )  # Shear along x-axis
            disk = disk.withFlux(band_fluxes_disk[i])

            # Create bulge (Spergel with nu=-0.6 approximates de Vaucouleurs/n=4)
            bulge = jax_galsim.Spergel(
                nu=-0.6,
                half_light_radius=sampled_r_bulge[i],
                gsparams=jax_galsim.GSParams(
                    minimum_fft_size=image_fft_size, maximum_fft_size=image_fft_size
                ),
            )
            # Apply ellipticity to bulge
            # Clip q to valid range [0, 1] to handle catalog edge cases
            q_bulge = jnp.clip(sampled_q_bulge[i], 0.0, 1.0)
            g_bulge = (1.0 - q_bulge) / (1.0 + q_bulge)
            bulge = bulge.shear(g=g_bulge, beta=0.0 * jax_galsim.degrees)
            bulge = bulge.withFlux(band_fluxes_bulge[i])

            gal = jax_galsim.Add([bulge, disk])
            gal = gal.rotate(sampled_angle[i] * jax_galsim.degrees)
            gal = gal.shift(shifts[i, 0], shifts[i, 1])

            galaxy_list.append(gal)

        # Combine all galaxies
        gals = jax_galsim.Add(galaxy_list)

        # PSF + noise
        objs = jax_galsim.Convolve([gals, psf])
        deep_objs = jax_galsim.Convolve([gals, deep_psf])

        nse = noise_std[band_idx]
        band_noise_key = jax.random.fold_in(key_noise_base, band_idx)
        key_wide, key_deep, key_deep_noise = jax.random.split(band_noise_key, 3)

        # Generate observations
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

        obs_wide_list.append(DFMdetObsList([obs_wide]))
        obs_deep_list.append(DFMdetObsList([obs_deep]))
        obs_deep_noise_list.append(DFMdetObsList([obs_deep_noise]))

    # Convert to MultiBandObsList
    mb_obs_wide = DFMdetMultiBandObsList(obs_wide_list)
    mb_obs_deep = DFMdetMultiBandObsList(obs_deep_list)
    mb_obs_deep_noise = DFMdetMultiBandObsList(obs_deep_noise_list)

    return mb_obs_wide, mb_obs_deep, mb_obs_deep_noise
