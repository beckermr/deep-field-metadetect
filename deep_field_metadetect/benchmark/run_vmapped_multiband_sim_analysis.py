"""Analysis script for vmapped multiband sims with COSMOS galaxies."""

import argparse
import time
from functools import partial
from pathlib import Path

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
    load_cosmos_catalog,
    make_jax_cosmos_field_jitted,
)

jax.config.update("jax_enable_x64", True)


@partial(
    jax.jit,
    static_argnames=(
        "bands",
        "n_bands",
        "n_galaxies",
        "detband_indices",
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
def run_cosmos_sim_pair(
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
    n_bands: int = 5,
    n_galaxies: int = 10,
    detband_indices: tuple[int, ...] | None = None,
    shear_magnitude: float = 0.02,
    deep_noise_fac: float = 1.0 / jnp.sqrt(10),
    deep_psf_fac: float = 1.0,
    scale: float = 0.2,
    dim: int = jax_dfmd_defaults.DEAFAULT_COADD_SIZE,
    dim_psf: int = jax_dfmd_defaults.DEFAULT_NXY_PSF,
    reconv_psf_dk: float = jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size: float = jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    psf_fft_size: int = jax_dfmd_defaults.DEFAULT_PSF_FFT_SIZE,
    image_fft_size: int = jax_dfmd_defaults.DEFAULT_IMAGE_FFT_SIZE,
    skip_obs_wide_corrections: bool = False,
    skip_obs_deep_corrections: bool = False,
) -> tuple[dict, dict]:
    """Run a pair of COSMOS sims with +/- shear for calibration.

    Parameters
    ----------
    key : jax.Array
        JAX random key
    mag_bulge, mag_disk : jax.Array
        Galaxy magnitude arrays from COSMOS catalog
    radius_bulge, radius_disk : jax.Array
        Galaxy radius arrays from COSMOS catalog
    bulge_axratio, disk_axratio : jax.Array
        Axis ratio arrays from COSMOS catalog
    angle_bd : jax.Array
        Position angle array from COSMOS catalog
    noise_std : tuple of float
        Noise standard deviations for each band
    psf_fwhm : tuple of float
        PSF FWHM for each band
    zeropoint : tuple of float
        Magnitude zeropoints for each band
    bands : tuple of str
        Band names
    n_bands : int
        Number of bands
    n_galaxies : int
        Number of galaxies per field
    detband_indices : tuple of int, optional
        Band indices for detection
    shear_magnitude : float
        Applied shear magnitude
    deep_noise_fac : float
        Deep noise factor
    deep_psf_fac : float
        Deep PSF factor
    scale : float
        Pixel scale
    dim : int
        Image dimension
    dim_psf : int
        PSF dimension
    reconv_psf_dk : float
        Reconv PSF dk
    reconv_psf_kim_size : float
        Reconv PSF kim size
    psf_fft_size : int
        PSF FFT size
    image_fft_size : int
        Image FFT size
    skip_obs_wide_corrections : bool
        Skip wide corrections
    skip_obs_deep_corrections : bool
        Skip deep corrections

    Returns
    -------
    res_p : dict
        Positive shear results
    res_m : dict
        Negative shear results
    """
    # Positive shear
    mb_obs_wide_p, mb_obs_deep_p, mb_obs_deep_noise_p = make_jax_cosmos_field_jitted(
        key,
        mag_bulge=mag_bulge,
        mag_disk=mag_disk,
        radius_bulge=radius_bulge,
        radius_disk=radius_disk,
        bulge_axratio=bulge_axratio,
        disk_axratio=disk_axratio,
        angle_bd=angle_bd,
        noise_std=noise_std,
        psf_fwhm=psf_fwhm,
        zeropoint=zeropoint,
        bands=bands,
        g1=shear_magnitude,
        g2=0.0,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        n_galaxies=n_galaxies,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
    )

    res_p = jax_multi_band_deep_field_metadetect_jitted(
        mb_obs_wide_p,
        mb_obs_deep_p,
        mb_obs_deep_noise_p,
        nxy=dim,
        nxy_psf=dim_psf,
        n_bands=n_bands,
        detband_indices=detband_indices,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )

    # Negative shear
    mb_obs_wide_m, mb_obs_deep_m, mb_obs_deep_noise_m = make_jax_cosmos_field_jitted(
        key,
        mag_bulge=mag_bulge,
        mag_disk=mag_disk,
        radius_bulge=radius_bulge,
        radius_disk=radius_disk,
        bulge_axratio=bulge_axratio,
        disk_axratio=disk_axratio,
        angle_bd=angle_bd,
        noise_std=noise_std,
        psf_fwhm=psf_fwhm,
        zeropoint=zeropoint,
        bands=bands,
        g1=-shear_magnitude,
        g2=0.0,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        n_galaxies=n_galaxies,
        scale=scale,
        dim=dim,
        dim_psf=dim_psf,
        psf_fft_size=psf_fft_size,
        image_fft_size=image_fft_size,
    )

    res_m = jax_multi_band_deep_field_metadetect_jitted(
        mb_obs_wide_m,
        mb_obs_deep_m,
        mb_obs_deep_noise_m,
        nxy=dim,
        nxy_psf=dim_psf,
        n_bands=n_bands,
        detband_indices=detband_indices,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )

    return res_p, res_m


def save_raw_catalogs_to_disk(
    output_dir: Path,
    res_p_list: list,
    res_m_list: list,
    config: dict,
):
    """Save raw simulation catalogs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    import json

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save full catalogs for each field
    catalogs_dir = output_dir / "catalogs"
    catalogs_dir.mkdir(exist_ok=True)

    print(f"\nSaving {len(res_p_list)} field catalogs to {catalogs_dir}...")

    for i, (res_p, res_m) in enumerate(zip(res_p_list, res_m_list)):
        # Save positive shear catalog
        catalog_p = res_p["dfmdet_res"]
        np.savez(
            catalogs_dir / f"field_{i:04d}_positive.npz",
            **{k: np.array(v) for k, v in catalog_p.items()},
        )

        # Save negative shear catalog
        catalog_m = res_m["dfmdet_res"]
        np.savez(
            catalogs_dir / f"field_{i:04d}_negative.npz",
            **{k: np.array(v) for k, v in catalog_m.items()},
        )

    print(f"Saved {len(res_p_list)} positive and {len(res_m_list)} negative catalogs")
    print(f"\nRaw catalogs saved to {output_dir}")
    print("  - Configuration: config.json")
    print("  - Full catalogs: catalogs/field_*.npz")
    print("\nTo compute shear calibration, run:")
    print(
        f"  python deep_field_metadetect/analysis/"
        f"compute_shear_calibration.py --input-dir {output_dir}"
    )


def run_vmapped_analysis(
    n_fields: int = 32,
    n_objects: int = 10,
    output_dir: str = "mdet_analysis_output",
    bands: tuple[str, ...] = ("g", "r", "i"),  # currently HSC-like
    noise_std: tuple[float, ...] = (5.31, 7.54, 10.17),
    psf_fwhm: tuple[float, ...] = (0.79, 0.75, 0.61),
    zeropoint: tuple[float, ...] = (28.90, 28.86, 28.61),
    shear_magnitude: float = 0.02,
    deep_noise_fac: float = 1.0 / np.sqrt(30),
    deep_psf_fac: float = 1.0,
    seed: int = 42,
    batch_size: int = 20,
    catalog_path: str = None,
    use_test_catalog: bool = False,
):
    """Run vmapped multiband simulation analysis with COSMOS galaxies.

    Parameters
    ----------
    n_fields : int
        Number of fields to simulate (default: 32)
    n_objects : int
        Number of objects per field (default: 10)
    output_dir : str
        Output directory for results
    bands : tuple of str
        Band names (e.g., ("g", "r", "i", "z", "y"))
    noise_std : tuple of float
        Noise standard deviations for each band
    psf_fwhm : tuple of float
        PSF FWHM for each band in arcsec
    zeropoint : tuple of float
        Magnitude zeropoints for each band
    shear_magnitude : float
        Applied shear magnitude
    deep_noise_fac : float
        Deep field noise factor
    deep_psf_fac : float
        Deep field PSF size factor
    seed : int
        Random seed
    batch_size : int
        Number of fields to process per batch (default: 20)
    catalog_path : str, optional
        Path to COSMOS catalog
    use_test_catalog : bool
        Use small test catalog (default: False)
    """
    print("=" * 80)
    print("Vmapped Multiband Simulation Analysis with COSMOS Galaxies")
    print("=" * 80)
    print(f"Number of fields: {n_fields}")
    print(f"Objects per field: {n_objects}")
    print(f"Batch size: {batch_size}")
    print(f"Bands: {bands}")
    print(f"Noise std: {noise_std}")
    print(f"PSF FWHM: {psf_fwhm}")
    print(f"Zeropoint: {zeropoint}")
    print(f"Shear magnitude: {shear_magnitude}")
    print(f"Deep noise factor: {deep_noise_fac}")
    print(f"Deep PSF factor: {deep_psf_fac}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load COSMOS catalog
    catalog = load_cosmos_catalog(catalog_path, use_test_catalog)

    # Configuration for saving
    config = {
        "n_fields": n_fields,
        "n_objects": n_objects,
        "bands": bands,
        "noise_std": noise_std,
        "psf_fwhm": psf_fwhm,
        "zeropoint": zeropoint,
        "shear_magnitude": shear_magnitude,
        "deep_noise_fac": deep_noise_fac,
        "deep_psf_fac": deep_psf_fac,
        "seed": seed,
        "batch_size": batch_size,
        "use_test_catalog": use_test_catalog,
        "catalog_n_galaxies": catalog["n_galaxies"],
    }

    n_bands = len(bands)
    detband_indices = None  # Use all bands for detection

    # Generate random keys for each field
    keys = jax.random.split(jax.random.PRNGKey(seed), n_fields)

    print("\nCompiling vmapped simulation function...")

    # Wrap the function with static arguments
    run_sim_pair_partial_fn = partial(
        run_cosmos_sim_pair,
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
        n_bands=n_bands,
        n_galaxies=n_objects,
        detband_indices=detband_indices,
        shear_magnitude=shear_magnitude,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
    )

    # Create vmapped version
    run_sim_pair_vmapped = jax.jit(
        jax.vmap(run_sim_pair_partial_fn, in_axes=(0,), out_axes=(0, 0))
    )

    # Lists to accumulate results
    res_p_list = []
    res_m_list = []

    # Process fields in batches
    n_batches = (n_fields + batch_size - 1) // batch_size
    total_time = 0.0
    fields_processed = 0

    print(f"\nProcessing {n_fields} fields in {n_batches} batches...")
    print("=" * 80)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_fields)
        current_batch_size = end_idx - start_idx

        batch_keys = keys[start_idx:end_idx]

        # Time the batch
        batch_start = time.time()

        results_p_batched, results_m_batched = run_sim_pair_vmapped(batch_keys)

        jax.block_until_ready(results_p_batched)
        jax.block_until_ready(results_m_batched)

        batch_end = time.time()
        batch_time = batch_end - batch_start

        # Exclude first 2 batches from cumulative time computation (JIT warmup)
        if batch_idx >= 2:
            total_time += batch_time
            fields_processed += current_batch_size

        avg_time_per_field = batch_time / current_batch_size
        cumulative_avg_time = (
            total_time / fields_processed if fields_processed > 0 else 0
        )
        fields_per_sec = 1.0 / cumulative_avg_time if cumulative_avg_time > 0 else 0

        # Print progress every batch
        warmup_note = (
            " (warmup - excluded from cumulative stats)" if batch_idx < 2 else ""
        )
        print(f"Batch {batch_idx + 1}/{n_batches}:{warmup_note}")
        print(f"  Fields: {start_idx} - {end_idx - 1} ({current_batch_size} fields)")
        print(f"  Batch time: {batch_time:.2f}s")
        print(f"  Avg time/field (this batch): {avg_time_per_field:.3f}s")
        if fields_processed > 0:
            print(
                f"  Cumulative avg time/field: {cumulative_avg_time:.3f}s "
                f"({fields_per_sec:.2f} fields/s)"
            )
        print(f"  Total fields processed: {end_idx}/{n_fields}")
        print("-" * 80)

        # Process batch results
        for i in range(current_batch_size):
            # Extract individual field results from batched outputs
            result_p = jax.tree_util.tree_map(lambda x: x[i], results_p_batched)
            result_m = jax.tree_util.tree_map(lambda x: x[i], results_m_batched)

            # Convert integer indices to string labels
            result_p = convert_multiband_dfmdet_result_to_strings(
                result_p, bands, DEFAULT_SHEARS
            )
            result_m = convert_multiband_dfmdet_result_to_strings(
                result_m, bands, DEFAULT_SHEARS
            )

            res_p_list.append(result_p)
            res_m_list.append(result_m)

    print("=" * 80)
    if fields_processed > 0:
        avg_per_field = total_time / fields_processed
        print(
            f"All {n_fields} fields completed. "
            f"Cumulative time (excluding first 2 warmup batches): "
            f"{total_time:.2f}s (avg: {avg_per_field:.3f}s/field)"
        )
    else:
        print(f"All {n_fields} fields completed.")
    print("=" * 80)

    # Save raw catalogs to disk
    output_path = Path(output_dir)
    save_raw_catalogs_to_disk(
        output_path,
        res_p_list,
        res_m_list,
        config,
    )

    return {
        "res_p_list": res_p_list,
        "res_m_list": res_m_list,
    }


def main():
    """Parse command-line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Run vmapped multiband simulation analysis with COSMOS galaxies"
    )
    parser.add_argument(
        "--n-fields",
        type=int,
        default=32,
        help="Number of fields to simulate (default: 32)",
    )
    parser.add_argument(
        "--n-objects",
        type=int,
        default=10,
        help="Number of objects per field (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mdet_analysis_output",
        help="Output directory for results (default: mdet_analysis_output)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=["g", "r", "i"],
        help="Bands to simulate (default: g r i from HSC galcheat)",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.02,
        help="Applied shear magnitude (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of fields per batch for vmapping (default: 20)",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help=(
            "Path to COSMOS catalog FITS file (optional, uses default if not specified)"
        ),
    )
    parser.add_argument(
        "--use-test-catalog",
        action="store_true",
        help="Use small test catalog with 30 objects (default: False)",
    )

    args = parser.parse_args()

    # Run analysis
    run_vmapped_analysis(
        n_fields=args.n_fields,
        n_objects=args.n_objects,
        output_dir=args.output_dir,
        bands=tuple(args.bands),
        shear_magnitude=args.shear,
        seed=args.seed,
        batch_size=args.batch_size,
        catalog_path=args.catalog_path,
        use_test_catalog=args.use_test_catalog,
    )


if __name__ == "__main__":
    main()
