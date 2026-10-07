"""Accuracy comparison: float32 vs float64 for metadetect pipeline.

Tests numerical consistency and shear calibration accuracy between
mixed precision (float32) and full precision (float64) modes.
"""

import jax
import numpy as np
import pytest

from deep_field_metadetect.jaxify import precision_config
from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_single_band_deep_field_metadetect,
)
from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size
from deep_field_metadetect.utils import make_simple_sim


def run_mdet_with_precision(use_float32=False, enable_x64=None):
    """Run metadetect with specified precision configuration.

    Parameters
    ----------
    use_float32 : bool, optional
        If True, use mixed precision (float32). Default is False (float64).
    enable_x64 : bool, optional
        If specified, sets jax_enable_x64 to this value. If None, sets to
        not use_float32 (True for float64 mode, False for float32 mode).
    """
    if use_float32:
        precision_config.use_mixed_precision(enabled=True)
    else:
        precision_config.use_mixed_precision(enabled=False)

    if enable_x64 is None:
        enable_x64 = not use_float32

    jax.config.update("jax_enable_x64", enable_x64)

    nxy = 201
    nxy_psf = 53
    scale = 0.2

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

    dk = compute_dk(image_size=nxy_psf, pixel_scale=scale)
    kim_size = compute_kim_size(image_size=nxy_psf)

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

    return result


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


@pytest.mark.parametrize("enable_x64", [True, False])
def test_dtype_stability(enable_x64):
    """Test numerical stability between float32 and float64.

    Compares full float64 (enable_x64=True) against mixed precision float32
    with enable_x64 parameterized (True/False).
    When using float32 and with enable_x64 moments are computed in float64.

    Tests:
    1. Object count consistency
    2. Moment measurements (wmom_g1, wmom_g2, wmom_s2n, wmom_T_ratio)
    3. Shear response (R11, R22)
    4. Shear estimation (g1_est, g2_est)
    """
    original_mode = precision_config._CURRENT_MODE
    original_x64 = jax.config.jax_enable_x64

    # Run metadetect with both precisions
    res_f64 = run_mdet_with_precision(use_float32=False, enable_x64=True)
    res_f32 = run_mdet_with_precision(use_float32=True, enable_x64=enable_x64)

    cat_f64 = res_f64["dfmdet_res"]
    cat_f32 = res_f32["dfmdet_res"]

    # Filter for noshear and good measurements
    msk64 = (np.array(cat_f64["mdet_step"]) == "noshear") & (
        np.array(cat_f64["wmom_flags"]) == 0
    )
    msk32 = (np.array(cat_f32["mdet_step"]) == "noshear") & (
        np.array(cat_f32["wmom_flags"]) == 0
    )

    # Set tolerances based on x64 mode
    if enable_x64:
        rtol = 1e-6  # Stricter when both use x64 (for moment calculation)
        rtol_shear_est = 1e-6  # For shear estimation
    else:
        # this fails if rtol is 1e-3
        rtol = 1e-2  # More relaxed when float32 mode doesn't use x64 at all
        rtol_shear_est = 1e-2

    # Test 1: Object count
    n_valid_f64 = np.sum(msk64)
    n_valid_f32 = np.sum(msk32)
    assert n_valid_f64 == n_valid_f32, (
        f"Diff number of valid objects: float64={n_valid_f64}, float32={n_valid_f32}"
    )

    # Test 2: Moment measurements accuracy
    for field in ["wmom_g1", "wmom_g2", "wmom_s2n", "wmom_T_ratio"]:
        if field not in cat_f64 or field not in cat_f32:
            continue

        vals64 = np.array(cat_f64[field])[msk64]
        vals32 = np.array(cat_f32[field])[msk32]

        assert np.allclose(vals64, vals32, rtol=rtol), (
            f"Field {field} differs beyond tolerance (enable_x64={enable_x64}):\n"
            f"  Max abs diff: {np.max(np.abs(vals64 - vals32)):.10e}\n"
            f"  Max rel diff: {np.max(np.abs((vals64 - vals32) / vals64)) * 100:.6f}%\n"
            f"  Mean f64: {np.mean(vals64):.10e}\n"
            f"  Mean f32: {np.mean(vals32):.10e}\n"
            f"  Expected rtol: {rtol}"
        )

    # Compute shear response and estimates
    calib_f64 = compute_shear_response_from_catalog(cat_f64)
    calib_f32 = compute_shear_response_from_catalog(cat_f32)

    # Test 3: Shear response (R11, R22)
    r11_diff = calib_f32["R11"] - calib_f64["R11"]
    assert np.isclose(calib_f32["R11"], calib_f64["R11"], rtol=rtol), (
        f"R11 differs beyond tolerance (enable_x64={enable_x64}):\n"
        f"  float64: {calib_f64['R11']:.10e}\n"
        f"  float32: {calib_f32['R11']:.10e}\n"
        f"  Abs diff: {r11_diff:.10e}\n"
        f"  Rel diff: {r11_diff / calib_f64['R11'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol}"
    )

    r22_diff = calib_f32["R22"] - calib_f64["R22"]
    assert np.isclose(calib_f32["R22"], calib_f64["R22"], rtol=rtol), (
        f"R22 differs beyond tolerance (enable_x64={enable_x64}):\n"
        f"  float64: {calib_f64['R22']:.10e}\n"
        f"  float32: {calib_f32['R22']:.10e}\n"
        f"  Abs diff: {r22_diff:.10e}\n"
        f"  Rel diff: {r22_diff / calib_f64['R22'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol}"
    )

    # Test 4: Shear estimation
    g1_true = 0.02
    g2_true = 0.00

    g1_est_diff = calib_f32["g1_est"] - calib_f64["g1_est"]
    assert np.isclose(calib_f32["g1_est"], calib_f64["g1_est"], rtol=rtol_shear_est), (
        f"g1 estimate differs beyond tolerance (enable_x64={enable_x64}):\n"
        f"  float64: {calib_f64['g1_est']:.10e}\n"
        f"  float32: {calib_f32['g1_est']:.10e}\n"
        f"  True value: {g1_true}\n"
        f"  Abs diff: {g1_est_diff:.10e}\n"
        f"  Rel diff: {g1_est_diff / calib_f64['g1_est'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol_shear_est}"
    )

    g2_est_diff = calib_f32["g2_est"] - calib_f64["g2_est"]
    assert np.isclose(calib_f32["g2_est"], calib_f64["g2_est"], rtol=rtol_shear_est), (
        f"g2 estimate differs beyond tolerance (enable_x64={enable_x64}):\n"
        f"  float64: {calib_f64['g2_est']:.10e}\n"
        f"  float32: {calib_f32['g2_est']:.10e}\n"
        f"  True value: {g2_true}\n"
        f"  Abs diff: {g2_est_diff:.10e}\n"
        f"  Rel diff: {g2_est_diff / calib_f64['g2_est'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol_shear_est}"
    )

    # Restore original precision mode and x64 setting
    precision_config.use_mixed_precision(enabled=(original_mode == "mixed"))
    jax.config.update("jax_enable_x64", original_x64)


def main():
    """Run all accuracy tests and print results."""
    print("=" * 80)
    print("Float32 vs Float64 Accuracy Tests")
    print("=" * 80)
    print("\nComparing full float64 (enable_x64=True) against mixed precision float32")
    print("with enable_x64 parameterized (True/False)\n")

    # Test both x64 modes for float32
    for enable_x64 in [True, False]:
        print(f"\n{'=' * 80}")
        print(f"Testing with float32 enable_x64={enable_x64}")
        print(f"{'=' * 80}\n")

        try:
            test_dtype_stability(enable_x64)
            print("  All differences are within tollerence \n")
        except AssertionError:
            print(" Differences are too large \n")

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
