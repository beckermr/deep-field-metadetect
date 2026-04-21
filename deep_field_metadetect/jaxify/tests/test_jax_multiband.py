import numpy as np

from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_multi_band_deep_field_metadetect,
    jax_single_band_deep_field_metadetect,
)
from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size
from deep_field_metadetect.utils import (
    make_simple_sim,
)


def _make_multiband_sim(
    seed,
    bands,
    g1=0,
    g2=0,
    s2n=20,
    deep_noise_fac=1.0 / np.sqrt(10),
    deep_psf_fac=1.0,
    n_objs=1,
    scale=0.2,
    dim=53,
    dim_psf=53,
    buff=26,
    band_flux_factors=None,
):
    """Make multiband simulation for JAX testing.

    Uses the same seed for all bands so objects are at the same positions.
    Different bands have different fluxes controlled by band_flux_factors.
    """
    if band_flux_factors is None:
        # Default: r-band is brightest, g is ~0.7x, i is ~0.8x
        band_flux_factors = {"g": 0.7, "r": 1.0, "i": 0.8, "z": 0.6}

    obs_wide = {}
    obs_deep = {}
    obs_deep_noise = {}

    for band in bands:
        flux_factor = band_flux_factors.get(band, 1.0)

        # Use same seed for all bands so objects are at same positions
        obs_w, obs_d, obs_dn = make_simple_sim(
            seed=seed,
            g1=g1,
            g2=g2,
            s2n=s2n * np.sqrt(flux_factor),  # Scale S/N with flux
            deep_noise_fac=deep_noise_fac,
            deep_psf_fac=deep_psf_fac,
            n_objs=n_objs,
            scale=scale,
            dim=dim,
            dim_psf=dim_psf,
            buff=buff,
            obj_flux_factor=flux_factor,
            return_dfmd_obs=True,
        )

        obs_wide[band] = obs_w
        obs_deep[band] = obs_d
        obs_deep_noise[band] = obs_dn

    return obs_wide, obs_deep, obs_deep_noise


def test_metadetect_single_r_band_multiband_vs_single_band():
    """Test that multiband metadetect with only r-band detection gives results
    consistent with single-band r-band metadetect.

    This test creates a multiband simulation (g, r, i), then:
    1. Runs jax_multi_band_deep_field_metadetect with detbands=['r']
    2. Runs jax_single_band_deep_field_metadetect on just the r-band
    3. Compares the r-band results from both approaches
    """
    bands = ["g", "r", "i"]
    nxy = 201
    nxy_psf = 53
    scale = 0.2

    # Create mb simulation
    obs_w, obs_d, obs_dn = _make_multiband_sim(
        seed=42,
        bands=bands,
        g1=0.02,
        g2=0.00,
        s2n=1000,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        n_objs=5,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
    )

    dk = compute_dk(image_size=nxy_psf, pixel_scale=scale)
    kim_size = compute_kim_size(image_size=nxy_psf)

    multi_res = jax_multi_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        detbands=("r"),  # Only detect in r-band
        reconv_psf_dk=dk,
        reconv_psf_kim_size=kim_size,
    )["dfmdet_res"]

    # single-band
    single_res = jax_single_band_deep_field_metadetect(
        obs_w["r"],
        obs_d["r"],
        obs_dn["r"],
        nxy=nxy,
        nxy_psf=nxy_psf,
        reconv_psf_dk=dk,
        reconv_psf_kim_size=kim_size,
        use_sep=False,
    )["dfmdet_res"]

    r_band_mask = multi_res["band"] == "r"
    multi_r_res = {k: v[r_band_mask] for k, v in multi_res.items()}

    # Compare results for each metadetect step
    for step in ["noshear", "1p", "1m", "2p", "2m"]:
        multi_step = multi_r_res["mdet_step"] == step
        single_step = single_res["mdet_step"] == step

        # same number of detections
        n_multi = np.sum(multi_step)
        n_single = np.sum(single_step)
        assert n_multi == n_single, (
            f"Different number of detections for step {step}: "
            f"multiband={n_multi}, single-band={n_single}"
        )

        # compare key measurements
        if n_multi > 0:
            # positions
            np.testing.assert_allclose(
                multi_r_res["x"][multi_step],
                single_res["x"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"x positions differ for step {step}",
            )
            np.testing.assert_allclose(
                multi_r_res["y"][multi_step],
                single_res["y"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"y positions differ for step {step}",
            )

            # measurements
            np.testing.assert_allclose(
                multi_r_res["wmom_g1"][multi_step],
                single_res["wmom_g1"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"wmom_g1 differs for step {step}",
            )
            np.testing.assert_allclose(
                multi_r_res["wmom_g2"][multi_step],
                single_res["wmom_g2"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"wmom_g2 differs for step {step}",
            )

            np.testing.assert_allclose(
                multi_r_res["wmom_T_ratio"][multi_step],
                single_res["wmom_T_ratio"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"wmom_T_ratio differs for step {step}",
            )

            np.testing.assert_allclose(
                multi_r_res["wmom_psf_T"][multi_step],
                single_res["wmom_psf_T"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"wmom_psf_T differs for step {step}",
            )

            np.testing.assert_allclose(
                multi_r_res["wmom_s2n"][multi_step],
                single_res["wmom_s2n"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"wmom_s2n differs for step {step}",
            )

            np.testing.assert_array_equal(
                multi_r_res["bmask_flags"][multi_step],
                single_res["bmask_flags"][single_step],
                err_msg=f"bmask_flags differ for step {step}",
            )

            np.testing.assert_allclose(
                multi_r_res["mfrac"][multi_step],
                single_res["mfrac"][single_step],
                rtol=1e-15,
                atol=1e-15,
                err_msg=f"mfrac differs for step {step}",
            )

            np.testing.assert_array_equal(
                multi_r_res["wmom_flags"][multi_step],
                single_res["wmom_flags"][single_step],
                err_msg=f"wmom_flags differ for step {step}",
            )


def test_metadetect_gri_bands_detect_on_ri_only():
    """Test multiband metadetect with gri bands, detecting/measuring only on ri."""
    bands = ["g", "r", "i"]
    nxy = 201
    nxy_psf = 53
    scale = 0.2

    # Create gri simulation with 3 objects
    obs_w, obs_d, obs_dn = _make_multiband_sim(
        seed=123,
        bands=bands,
        g1=0.02,
        g2=-0.01,
        s2n=500,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        n_objs=3,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
    )

    dk = compute_dk(image_size=nxy_psf, pixel_scale=scale)
    kim_size = compute_kim_size(image_size=nxy_psf)

    result = jax_multi_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        detbands=("r", "i"),  # Only use r and i for detection
        reconv_psf_dk=dk,
        reconv_psf_kim_size=kim_size,
    )

    dfmdet_res = result["dfmdet_res"]

    # Check that we only have results for r and i bands (not g)
    unique_bands = np.unique(dfmdet_res["band"])
    assert set(unique_bands) == {"r", "i"}, (
        f"Expected measurements only in r, i bands, got {unique_bands}"
    )

    for step in ["noshear", "1p", "1m", "2p", "2m"]:
        step_mask = dfmdet_res["mdet_step"] == step
        valid_mask = step_mask & (dfmdet_res["det_flag"] == 1)

        # Verify measurements exist for r and i bands only and are finite
        for band in ["r", "i"]:
            band_step_valid_mask = valid_mask & (dfmdet_res["band"] == band)
            n_band_detections = np.sum(band_step_valid_mask)
            assert n_band_detections > 0, (
                f"No detections in {band} band for step {step}"
            )

            g1 = dfmdet_res["wmom_g1"][band_step_valid_mask]
            g2 = dfmdet_res["wmom_g2"][band_step_valid_mask]
            s2n = dfmdet_res["wmom_s2n"][band_step_valid_mask]
            T_ratio = dfmdet_res["wmom_T_ratio"][band_step_valid_mask]

            assert np.all(np.isfinite(g1)), f"Non-finite g1 values in {band} band"
            assert np.all(np.isfinite(g2)), f"Non-finite g2 values in {band} band"
            assert np.all(s2n > 0), f"Non-positive S/N in {band} band"
            assert np.all(T_ratio > 0), f"Non-positive T_ratio in {band} band"

    # Verify that r and i bands have same detection positions
    # (since they're detected from the same ri coadd)
    noshear_valid_mask = (dfmdet_res["mdet_step"] == "noshear") & (
        dfmdet_res["det_flag"] == 1
    )

    r_mask = noshear_valid_mask & (dfmdet_res["band"] == "r")
    i_mask = noshear_valid_mask & (dfmdet_res["band"] == "i")

    r_x = dfmdet_res["x"][r_mask]
    r_y = dfmdet_res["y"][r_mask]
    r_ids = dfmdet_res["id"][r_mask]

    i_x = dfmdet_res["x"][i_mask]
    i_y = dfmdet_res["y"][i_mask]
    i_ids = dfmdet_res["id"][i_mask]

    # Same object IDs and positions across r and i bands
    np.testing.assert_array_equal(
        r_ids, i_ids, err_msg="Object IDs differ between r and i bands"
    )
    np.testing.assert_array_equal(
        r_x, i_x, err_msg="X positions differ between r and i bands"
    )
    np.testing.assert_array_equal(
        r_y, i_y, err_msg="Y positions differ between r and i bands"
    )
