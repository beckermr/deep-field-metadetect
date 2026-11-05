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
from deep_field_metadetect.jaxify.observation import ngmix_obs_to_dfmd_obs
from deep_field_metadetect.metadetect import single_band_deep_field_metadetect
from deep_field_metadetect.utils import make_simple_sim


def run_mdet_with_precision(use_float32=False):
    """Run metadetect with specified precision configuration.

    Parameters
    ----------
    use_float32 : bool, optional
        If True, use mixed precision (float32 images, float64 moments).
        If False, use full float64 everywhere. Default is False.
    """
    if use_float32:
        precision_config.use_mixed_precision(enabled=True)
    else:
        precision_config.use_mixed_precision(enabled=False)

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


def test_dtype_stability():
    """Test numerical stability between float32 and float64.

    Compares full float64 against mixed precision (float32 images, float64 moments).
    Requires jax_enable_x64=True for float64 support.

    Tests:
    1. Object count consistency
    2. Moment measurements (wmom_g1, wmom_g2, wmom_s2n, wmom_T_ratio)
    3. Shear response (R11, R22)
    4. Shear estimation (g1_est, g2_est)
    """
    original_mode = precision_config._CURRENT_MODE

    # Run metadetect with both precisions
    res_f64 = run_mdet_with_precision(use_float32=False)
    res_f32 = run_mdet_with_precision(use_float32=True)

    cat_f64 = res_f64["dfmdet_res"]
    cat_f32 = res_f32["dfmdet_res"]

    # Filter for noshear and good measurements
    msk64 = (np.array(cat_f64["mdet_step"]) == "noshear") & (
        np.array(cat_f64["wmom_flags"]) == 0
    )
    msk32 = (np.array(cat_f32["mdet_step"]) == "noshear") & (
        np.array(cat_f32["wmom_flags"]) == 0
    )

    # Set tolerances - strict since both use x64 for moment calculations
    rtol = 1e-6  # Strict tolerance for mixed precision with x64 enabled
    rtol_shear_est = 1e-6  # For shear estimation

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
            f"Field {field} differs beyond tolerance :\n"
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
        f"R11 differs beyond tolerance :\n"
        f"  float64: {calib_f64['R11']:.10e}\n"
        f"  float32: {calib_f32['R11']:.10e}\n"
        f"  Abs diff: {r11_diff:.10e}\n"
        f"  Rel diff: {r11_diff / calib_f64['R11'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol}"
    )

    r22_diff = calib_f32["R22"] - calib_f64["R22"]
    assert np.isclose(calib_f32["R22"], calib_f64["R22"], rtol=rtol), (
        f"R22 differs beyond tolerance :\n"
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
        f"g1 estimate differs beyond tolerance :\n"
        f"  float64: {calib_f64['g1_est']:.10e}\n"
        f"  float32: {calib_f32['g1_est']:.10e}\n"
        f"  True value: {g1_true}\n"
        f"  Abs diff: {g1_est_diff:.10e}\n"
        f"  Rel diff: {g1_est_diff / calib_f64['g1_est'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol_shear_est}"
    )

    g2_est_diff = calib_f32["g2_est"] - calib_f64["g2_est"]
    assert np.isclose(calib_f32["g2_est"], calib_f64["g2_est"], rtol=rtol_shear_est), (
        f"g2 estimate differs beyond tolerance :\n"
        f"  float64: {calib_f64['g2_est']:.10e}\n"
        f"  float32: {calib_f32['g2_est']:.10e}\n"
        f"  True value: {g2_true}\n"
        f"  Abs diff: {g2_est_diff:.10e}\n"
        f"  Rel diff: {g2_est_diff / calib_f64['g2_est'] * 100:.6f}%\n"
        f"  Expected rtol: {rtol_shear_est}"
    )

    # Restore original precision mode
    precision_config.use_mixed_precision(enabled=(original_mode == "mixed"))


@pytest.mark.float32_only
def test_jax_float32_vs_non_jax_float64():
    """Test pure float32 JAX vs float64 ngmix baseline.

    Compares JAX metadetect in pure float32 mode (jax_enable_x64=False)
    against the non-JAX ngmix implementation (always float64).
    """
    assert not jax.config.jax_enable_x64, (
        "This test requires jax_enable_x64=False. "
        "Run with: JAX_ENABLE_X64=false pytest --float32-only"
    )

    original_mode = precision_config._CURRENT_MODE
    precision_config.use_mixed_precision(enabled=True)

    nxy = 201
    nxy_psf = 53
    scale = 0.2
    seed = 42

    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=seed,
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
        return_dfmd_obs=False,  # Get ngmix format
    )

    # NON-JAX version (float64 baseline)
    res_ngmix = single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
    )

    # Convert to JAX format and run JAX version (pure float32)
    obs_w_jax = ngmix_obs_to_dfmd_obs(obs_w)
    obs_d_jax = ngmix_obs_to_dfmd_obs(obs_d)
    obs_dn_jax = ngmix_obs_to_dfmd_obs(obs_dn)

    dk = compute_dk(image_size=nxy_psf, pixel_scale=scale)
    kim_size = compute_kim_size(image_size=nxy_psf)

    res_jax = jax_single_band_deep_field_metadetect(
        obs_w_jax,
        obs_d_jax,
        obs_dn_jax,
        nxy=nxy,
        nxy_psf=nxy_psf,
        reconv_psf_dk=dk,
        reconv_psf_kim_size=kim_size,
        use_sep=True,  # Match non-JAX which always uses SEP
    )

    # Compare results
    cat_ngmix = res_ngmix["dfmdet_res"]
    cat_jax = res_jax["dfmdet_res"]

    # Filter for valid measurements
    msk_ngmix = (np.array(cat_ngmix["mdet_step"]) == "noshear") & (
        np.array(cat_ngmix["wmom_flags"]) == 0
    )
    msk_jax = (
        (np.array(cat_jax["mdet_step"]) == "noshear")
        & (np.array(cat_jax["wmom_flags"]) == 0)
        & (np.array(cat_jax["det_flag"]) == 1)  # Only actual detections, not padding
    )

    # Should detect same number of objects
    n_ngmix = np.sum(msk_ngmix)
    n_jax = np.sum(msk_jax)
    assert n_ngmix == n_jax, f"Different object counts: ngmix={n_ngmix}, jax={n_jax}"

    rtol = 0.02  # 2% tolerance for pure float32 mode
    atol = 1e-4

    x_ngmix = np.array(cat_ngmix["x"])[msk_ngmix]
    y_ngmix = np.array(cat_ngmix["y"])[msk_ngmix]
    x_jax = np.array(cat_jax["x"])[msk_jax]
    y_jax = np.array(cat_jax["y"])[msk_jax]

    # Match objects within 0.1 pixel tolerance
    matched_indices_ngmix = []
    matched_indices_jax = []

    for i, (x_n, y_n) in enumerate(zip(x_ngmix, y_ngmix)):
        distances = np.sqrt((x_jax - x_n) ** 2 + (y_jax - y_n) ** 2)
        if np.min(distances) < 0.1:  # Match within 0.1 pixels
            j = np.argmin(distances)
            matched_indices_ngmix.append(i)
            matched_indices_jax.append(j)

    n_matched = len(matched_indices_ngmix)
    assert n_matched == n_ngmix, (
        f"Could not match all objects: {n_matched} matched out of {n_ngmix}"
    )

    for field in ["wmom_g1", "wmom_g2", "wmom_s2n", "wmom_T_ratio"]:
        vals_ngmix_all = np.array(cat_ngmix[field])[msk_ngmix]
        vals_jax_all = np.array(cat_jax[field])[msk_jax]

        vals_ngmix = vals_ngmix_all[matched_indices_ngmix]
        vals_jax = vals_jax_all[matched_indices_jax]

        # Compare individual values
        max_abs_diff = np.max(np.abs(vals_ngmix - vals_jax))
        max_rel_diff = (
            np.max(np.abs((vals_ngmix - vals_jax) / vals_ngmix))
            if np.all(np.abs(vals_ngmix) > 0)
            else 0
        )

        assert np.allclose(vals_jax, vals_ngmix, rtol=rtol, atol=atol), (
            f"{field} values differ beyond tolerance in pure float32 mode:\n"
            f"  Max abs diff: {max_abs_diff:.10e}\n"
            f"  Max rel diff: {max_rel_diff * 100:.6f}%\n"
            f"  Mean ngmix (float64): {np.mean(vals_ngmix):.10e}\n"
            f"  Mean jax (float32): {np.mean(vals_jax):.10e}\n"
            f"  Expected rtol: {rtol} ({rtol * 100:.0f}%)"
        )

    # Restore original precision mode
    precision_config.use_mixed_precision(enabled=(original_mode == "mixed"))
