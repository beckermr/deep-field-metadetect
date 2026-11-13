"""Performance comparison: float32 vs float64 for metadetect pipeline.

Measures execution time and numerical accuracy for shear → moment measurement.
"""

import time

import jax
import numpy as np

from deep_field_metadetect.jaxify import precision_config
from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_single_band_deep_field_metadetect,
)
from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size
from deep_field_metadetect.utils import make_simple_sim


def run_mdet_with_timing(use_float32=False, n_runs=10):
    """Run metadetect with timing, using standard test configuration."""

    if use_float32:
        jax.config.update("jax_enable_x64", True)
        precision_config.use_mixed_precision(enabled=True)
        mode_name = "float32 (mixed precision)"
    else:
        jax.config.update("jax_enable_x64", True)
        precision_config.use_mixed_precision(enabled=False)
        mode_name = "float64 (full precision)"

    print(f"\n{'=' * 70}")
    print(f"Testing with {mode_name}")
    print(f"{'=' * 70}")

    nxy = 201
    nxy_psf = 53
    scale = 0.2

    print("Creating simulation...")
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=42,
        g1=0.02,
        g2=0.00,
        s2n=1000,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=True,
    )

    print(f"Image dtype: {obs_w.image.dtype}")
    print(f"Image shape: {obs_w.image.shape}")

    dk = compute_dk(image_size=nxy_psf, pixel_scale=scale)
    kim_size = compute_kim_size(image_size=nxy_psf)

    # Timing runs (first 2 iterations are warmup, skipped in statistics)
    print(f"\nRunning {n_runs + 2} iterations (first 2 are warmup)...")
    times = []

    for i in range(n_runs + 2):
        start = time.perf_counter()
        result = jax_single_band_deep_field_metadetect(
            obs_w,
            obs_d,
            obs_dn,
            nxy=nxy,
            nxy_psf=nxy_psf,
            reconv_psf_dk=dk,
            reconv_psf_kim_size=kim_size,
            use_sep=True,
        )
        # Block until computation finishes
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            result,
        )
        elapsed = time.perf_counter() - start

        if i == 0:
            print(f"  Warmup 1: {elapsed:.4f}s (not included in statistics)")
        elif i == 1:
            print(f"  Warmup 2: {elapsed:.4f}s (not included in statistics)")
        else:
            times.append(elapsed)
            print(f"  Run {i - 1}/{n_runs}: {elapsed:.4f}s")

    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print("\nResults:")
    print(f"  Mean:   {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"  Median: {np.median(times):.4f}s")
    print(f"  Min:    {min_time:.4f}s")
    print(f"  Max:    {max_time:.4f}s")

    return {
        "result": result,
        "times": times,
        "avg_time": avg_time,
        "std_time": std_time,
        "mode": mode_name,
    }


def compare_numerical_accuracy(res_f64, res_f32):
    """Compare numerical accuracy between float32 and float64."""
    print(f"\n{'=' * 70}")
    print("Testing numerical consistency")
    print(f"{'=' * 70}")

    cat_f64 = res_f64["result"]["dfmdet_res"]
    cat_f32 = res_f32["result"]["dfmdet_res"]

    # Filter for noshear and good measurements
    msk64 = (np.array(cat_f64["mdet_step"]) == "noshear") & (
        np.array(cat_f64["wmom_flags"]) == 0
    )
    msk32 = (np.array(cat_f32["mdet_step"]) == "noshear") & (
        np.array(cat_f32["wmom_flags"]) == 0
    )

    print("\nNumber of valid objects:")
    print(f"  float64: {np.sum(msk64)}")
    print(f"  float32: {np.sum(msk32)}")

    # Compare moment measurements
    for field in ["wmom_g1", "wmom_g2", "wmom_s2n", "wmom_T_ratio"]:
        if field not in cat_f64 or field not in cat_f32:
            continue

        vals64 = np.array(cat_f64[field])[msk64]
        vals32 = np.array(cat_f32[field])[msk32]

        mean64 = np.mean(vals64)
        mean32 = np.mean(vals32)
        diff = mean32 - mean64
        rel_diff = (diff / mean64 * 100) if mean64 != 0 else 0

        # Compute max absolute and relative differences
        if len(vals64) > 0 and len(vals32) > 0:
            max_abs_diff = np.max(np.abs(vals64 - vals32))
            max_rel_diff = np.max(np.abs((vals64 - vals32) / vals64)) * 100
        else:
            max_abs_diff = 0
            max_rel_diff = 0

        print(f"\n  {field}:")
        print(f"    float64 mean: {mean64:.10e}")
        print(f"    float32 mean: {mean32:.10e}")
        print(f"    Mean abs diff: {diff:.10e}")
        print(f"    Mean rel diff: {rel_diff:.6f}%")
        print(f"    Max abs diff: {max_abs_diff:.10e}")
        print(f"    Max rel diff: {max_rel_diff:.6f}%")


def compute_shear_response_from_catalog(cat, step=0.01):
    """Compute metadetect shear response R11 and R22 from catalog."""

    # Get measurements for each shear
    def get_mean_g(shear_name):
        mdet_step = np.array(cat["mdet_step"])
        wmom_flags = np.array(cat["wmom_flags"])
        msk = (mdet_step == shear_name) & (wmom_flags == 0)
        if np.sum(msk) == 0:
            return 0, 0
        wmom_g1 = np.array(cat["wmom_g1"])
        wmom_g2 = np.array(cat["wmom_g2"])
        g1_mean = np.mean(wmom_g1[msk])
        g2_mean = np.mean(wmom_g2[msk])
        return g1_mean, g2_mean

    g1_noshear, g2_noshear = get_mean_g("noshear")
    g1_1p, _ = get_mean_g("1p")
    g1_1m, _ = get_mean_g("1m")
    _, g2_2p = get_mean_g("2p")
    _, g2_2m = get_mean_g("2m")

    # Compute shear response
    R11 = (g1_1p - g1_1m) / (2 * step)
    R22 = (g2_2p - g2_2m) / (2 * step)

    # Shear estimate
    g1_est = g1_noshear / R11 if R11 != 0 else 0
    g2_est = g2_noshear / R22 if R22 != 0 else 0

    return {
        "R11": R11,
        "R22": R22,
        "g1_est": g1_est,
        "g2_est": g2_est,
        "g1_noshear": g1_noshear,
        "g2_noshear": g2_noshear,
    }


def compare_shear_calibration(res_f64, res_f32):
    """Compare shear response and calibration."""

    cat_f64 = res_f64["result"]["dfmdet_res"]
    cat_f32 = res_f32["result"]["dfmdet_res"]

    calib_f64 = compute_shear_response_from_catalog(cat_f64)
    calib_f32 = compute_shear_response_from_catalog(cat_f32)

    print("\nShear Response R11:")
    print(f"  float64: {calib_f64['R11']:.10e}")
    print(f"  float32: {calib_f32['R11']:.10e}")
    print(f"  Abs diff: {calib_f32['R11'] - calib_f64['R11']:.10e}")
    if calib_f64["R11"] != 0:
        rel_diff = (calib_f32["R11"] - calib_f64["R11"]) / calib_f64["R11"] * 100
        print(f"  Rel diff: {rel_diff:.6f}%")

    print("\nShear Response R22:")
    print(f"  float64: {calib_f64['R22']:.10e}")
    print(f"  float32: {calib_f32['R22']:.10e}")
    print(f"  Abs diff: {calib_f32['R22'] - calib_f64['R22']:.10e}")
    if calib_f64["R22"] != 0:
        rel_diff = (calib_f32["R22"] - calib_f64["R22"]) / calib_f64["R22"] * 100
        print(f"  Rel diff: {rel_diff:.6f}%")

    print("\nEstimated shear g1 (input was 0.02):")
    print(f"  float64: {calib_f64['g1_est']:.10e}")
    print(f"  float32: {calib_f32['g1_est']:.10e}")
    print(f"  Abs diff: {calib_f32['g1_est'] - calib_f64['g1_est']:.10e}")
    if calib_f64["g1_est"] != 0:
        rel_diff = calib_f32["g1_est"] - calib_f64["g1_est"]
        rel_diff = rel_diff / calib_f64["g1_est"] * 100
        print(f"  Rel diff: {rel_diff:.6f}%")
    print("  True value: 0.02")

    print("\nEstimated shear g2 (input was 0.00):")
    print(f"  float64: {calib_f64['g2_est']:.10e}")
    print(f"  float32: {calib_f32['g2_est']:.10e}")
    print(f"  Abs diff: {calib_f32['g2_est'] - calib_f64['g2_est']:.10e}")
    print("  True value: 0.00")


def main():
    """Run performance comparison."""

    # Run with float64 first
    res_f64 = run_mdet_with_timing(use_float32=False, n_runs=10)

    # Run with float32
    res_f32 = run_mdet_with_timing(use_float32=True, n_runs=10)

    print("Time: ")

    print(f"\nFloat64: {res_f64['avg_time']:.4f}s ± {res_f64['std_time']:.4f}s")
    print(f"Float32: {res_f32['avg_time']:.4f}s ± {res_f32['std_time']:.4f}s")

    speedup = res_f64["avg_time"] / res_f32["avg_time"]
    print(f"\nSpeedup: {speedup:.2f}x")

    # Numerical comparisons
    compare_numerical_accuracy(res_f64, res_f32)
    compare_shear_calibration(res_f64, res_f32)


if __name__ == "__main__":
    main()
