"""Test multiband simulations with JAX-Galsim and metadetect."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.jax_metacal import DEFAULT_SHEARS
from deep_field_metadetect.jaxify.jax_metadetect import (
    convert_multiband_dfmdet_result_to_strings,
    jax_multi_band_deep_field_metadetect_jitted,
)
from deep_field_metadetect.jaxify.jax_sims import (
    generate_jax_galsim_multiband_sim_observations_jitted,
    load_cosmos_catalog,
    make_jax_cosmos_field_jitted,
)
from deep_field_metadetect.utils import (
    assert_m_c_ok,
    estimate_m_and_c,
    measure_mcal_shear_quants,
    print_m_c,
)


@partial(
    jax.jit,
    static_argnames=(
        "bands",
        "n_bands",
        "detband_indices",
        "max_n_objs",
        "scale",
        "dim",
        "dim_psf",
        "reconv_psf_dk",
        "reconv_psf_kim_size",
        "psf_fft_size",
        "image_fft_size",
        "skip_obs_wide_corrections",
        "skip_obs_deep_corrections",
    ),
)
def _run_single_sim_jax(
    key,
    bands: tuple[str, ...] = ("g", "r", "i"),
    n_bands: int = 3,
    detband_indices: tuple[int, ...] | None = None,
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
    band_flux_factors: tuple[float, ...] | None = None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    skip_obs_wide_corrections: bool = False,
    skip_obs_deep_corrections: bool = False,
) -> dict:
    """Run a single multi-band simulation with JAX-Galsim and metadetect.

    Parameters
    ----------
    key : jax.Array
        JAX random key for reproducible simulations
    s2n : float
        Signal-to-noise ratio for simulation
    bands : tuple of str
        Band names (e.g., ("g", "r", "i"))
    n_bands : int
        Number of bands (should match len(bands))
    detband_indices : tuple of int, optional
        Band indices to use for detection (if None, uses all bands)
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
    band_flux_factors : tuple of float, optional
        Per-band flux factors as a tuple in the same order as bands
        (e.g., (0.7, 1.0, 0.8) for bands ("g", "r", "i"))
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (default: 256)
    skip_obs_wide_corrections : bool
        Skip wide-field observation corrections
    skip_obs_deep_corrections : bool
        Skip deep-field observation corrections


    Returns
    -------
    result : dict
        Dictionary containing metadetect results with key "dfmdet_res"
    """
    mb_obs_wide, mb_obs_deep, mb_obs_deep_noise = (
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

    result = jax_multi_band_deep_field_metadetect_jitted(
        mb_obs_wide,
        mb_obs_deep,
        mb_obs_deep_noise,
        nxy=dim,
        nxy_psf=dim_psf,
        n_bands=n_bands,
        detband_indices=detband_indices,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )

    return result


@partial(
    jax.jit,
    static_argnames=(
        "bands",
        "n_bands",
        "detband_indices",
        "max_n_objs",
        "scale",
        "dim",
        "dim_psf",
        "reconv_psf_dk",
        "reconv_psf_kim_size",
        "psf_fft_size",
        "image_fft_size",
        "skip_obs_wide_corrections",
        "skip_obs_deep_corrections",
    ),
)
def run_jax_sim_pair(
    key: jax.Array,
    bands: tuple[str, ...] = ("g", "r", "i"),
    n_bands: int = 3,
    detband_indices: tuple[int, ...] | None = None,
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
    band_flux_factors: tuple[float, ...] | None = None,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    skip_obs_wide_corrections: bool = False,
    skip_obs_deep_corrections: bool = False,
) -> tuple[dict, dict]:
    """Run a pair of simulations with +/- shear for calibration.

    Parameters
    ----------
    key : jax.Array
        JAX random key
    bands : tuple of str
        Band names (e.g., ("g", "r", "i"))
    n_bands : int
        Number of bands (should match len(bands))
    detband_indices : tuple of int, optional
        Band indices to use for detection (if None, uses all bands)
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
    band_flux_factors : tuple of float, optional
        Per-band flux factors as a tuple in the same order as bands
        (e.g., (0.7, 1.0, 0.8) for bands ("g", "r", "i"))
    psf_fft_size : int
        FFT size for JAX-Galsim drawing PSFs (default: 64)
    image_fft_size : int
        FFT size for JAX-Galsim drawing object images (default: 256)
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
        n_bands=n_bands,
        detband_indices=detband_indices,
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
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    )

    # Negative shear
    res_m = _run_single_sim_jax(
        key,
        bands=bands,
        n_bands=n_bands,
        detband_indices=detband_indices,
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
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    )

    return res_p, res_m


def test_jax_multiband_sim_mdet():
    """Test JAX multiband simulations with 100 sims to verify m and c calibration."""
    nsims = 10
    bands = ("g", "r", "i")
    s2n = 1e4
    shear_magnitude = 0.02
    deep_noise_fac = 1.0 / np.sqrt(30)
    deep_psf_fac = 1.0

    n_bands = len(bands)
    detband_indices = None  # Use all bands for detection

    rng = np.random.RandomState(seed=42)
    seeds = rng.randint(size=nsims, low=1, high=2**29)

    res_p = []
    res_m = []

    for seed in seeds:
        key = jax.random.PRNGKey(seed)

        # Run simulation pair (returns raw results with integer indices)
        result_p, result_m = run_jax_sim_pair(
            key,
            bands=bands,
            n_bands=n_bands,
            detband_indices=detband_indices,
            s2n=s2n,
            shear_magnitude=shear_magnitude,
            deep_noise_fac=deep_noise_fac,
            deep_psf_fac=deep_psf_fac,
        )

        # Convert integer indices to string labels
        if result_p is not None and result_m is not None:
            result_p = convert_multiband_dfmdet_result_to_strings(
                result_p, bands, DEFAULT_SHEARS
            )
            result_m = convert_multiband_dfmdet_result_to_strings(
                result_m, bands, DEFAULT_SHEARS
            )

            # Extract and measure shear quantities
            res_p.append(measure_mcal_shear_quants(result_p["dfmdet_res"]))
            res_m.append(measure_mcal_shear_quants(result_m["dfmdet_res"]))

    # Estimate m and c
    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        shear_magnitude,
        jackknife=len(res_p),
    )

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


def test_cosmos_field_generation():
    """Test COSMOS field generation with bulge+disk catalog sampling.

    Uses pre-filtered test catalog with 30 valid B+D galaxies.
    """
    # Load catalog using shared utility function
    catalog = load_cosmos_catalog(use_test_catalog=True)

    print(f"Loaded {catalog['n_galaxies']} pre-filtered B+D galaxies")
    assert catalog["n_galaxies"] == 30, (
        f"Expected 30 objects in test catalog, got {catalog['n_galaxies']}"
    )

    # Test basic function call
    key = jax.random.PRNGKey(42)
    bands = ("g", "r", "i", "z", "y")
    n_galaxies = 10
    noise_std = (0.005, 0.004, 0.003, 0.0035, 0.004)
    psf_fwhm = (0.8, 0.7, 0.6, 0.7, 0.7)
    zeropoint = (27.0, 27.5, 27.8, 27.6, 27.4)

    print(f"\nGenerating field with {n_galaxies} B+D galaxies...")
    mb_obs_wide, mb_obs_deep, mb_obs_deep_noise = make_jax_cosmos_field_jitted(
        key,
        mag_bulge=catalog["mag_bulge"],
        mag_disk=catalog["mag_disk"],
        radius_bulge=catalog["radius_bulge"],
        radius_disk=catalog["radius_disk"],
        bulge_axratio=catalog["bulge_axratio"],
        disk_axratio=catalog["disk_axratio"],
        angle_bd=catalog["angle_bd"],
        noise_std=noise_std,
        psf_fwhm=psf_fwhm,
        zeropoint=zeropoint,
        bands=bands,
        n_galaxies=n_galaxies,
        deep_noise_fac=1.0 / np.sqrt(10),
    )

    # Verify output structure
    assert len(mb_obs_wide) == 5, "Expected 5 bands in wide observation"
    assert len(mb_obs_deep) == 5, "Expected 5 bands in deep observation"
    assert len(mb_obs_deep_noise) == 5, "Expected 5 bands in deep noise observation"

    # Check each band
    for i, band in enumerate(bands):
        print(f"\nBand {band}:")
        # Note: we are not simulating individual visits
        obs_wide = mb_obs_wide[i][0]  # First (and only) observation in ObsList
        obs_deep = mb_obs_deep[i][0]
        obs_deep_noise = mb_obs_deep_noise[i][0]

        # Check image shapes
        from deep_field_metadetect.jaxify import jax_dfmd_defaults

        expected_dim = jax_dfmd_defaults.DEAFAULT_COADD_SIZE
        expected_psf_dim = jax_dfmd_defaults.DEFAULT_NXY_PSF

        assert obs_wide.image.shape == (expected_dim, expected_dim), (
            f"Unexpected image shape for band {band}"
        )
        assert obs_deep.image.shape == (expected_dim, expected_dim), (
            f"Unexpected image shape for band {band}"
        )
        assert obs_deep_noise.image.shape == (expected_dim, expected_dim), (
            f"Unexpected image shape for band {band}"
        )

        # Check that images contain signal (not all zeros)
        assert jnp.sum(jnp.abs(obs_wide.image)) > 0, (
            f"Wide image is all zeros for band {band}"
        )
        assert jnp.sum(jnp.abs(obs_deep.image)) > 0, (
            f"Deep image is all zeros for band {band}"
        )

        # Check PSF exists
        assert obs_wide.psf.image.shape == (expected_psf_dim, expected_psf_dim), (
            f"Unexpected PSF shape for band {band}"
        )
        assert jnp.sum(obs_wide.psf.image) > 0, f"PSF is empty for band {band}"

        print(f"  Image sum (wide): {jnp.sum(obs_wide.image):.2e}")
        print(f"  Image sum (deep): {jnp.sum(obs_deep.image):.2e}")
