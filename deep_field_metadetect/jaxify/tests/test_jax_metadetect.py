import multiprocessing

import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_single_band_deep_field_metadetect,
)
from deep_field_metadetect.jaxify.observation import ngmix_obs_to_dfmd_obs
from deep_field_metadetect.metadetect import single_band_deep_field_metadetect
from deep_field_metadetect.utils import (
    MAX_ABS_C,
    MAX_ABS_M,
    assert_m_c_ok,
    estimate_m_and_c,
    make_simple_sim,
    measure_mcal_shear_quants,
    print_m_c,
)


def _run_single_sim(
    seed,
    s2n,
    g1,
    g2,
    deep_noise_fac,
    deep_psf_fac,
    skip_wide,
    skip_deep,
):
    nxy = 201
    nxy_psf = 53
    scale = 0.2
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=seed,
        g1=g1,
        g2=g2,
        s2n=s2n,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=True,
    )

    res, _ = jax_single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
        scale=scale,
    )
    return measure_mcal_shear_quants(res)


def _run_sim_pair(seed, s2n, deep_noise_fac, deep_psf_fac, skip_wide, skip_deep):
    res_p = _run_single_sim(
        seed,
        s2n,
        0.02,
        0.0,
        deep_noise_fac,
        deep_psf_fac,
        skip_wide,
        skip_deep,
    )

    res_m = _run_single_sim(
        seed,
        s2n,
        -0.02,
        0.0,
        deep_noise_fac,
        deep_psf_fac,
        skip_wide,
        skip_deep,
    )

    return res_p, res_m


def _run_single_sim_jax_and_ngmix(
    seed,
    s2n,
    g1,
    g2,
    deep_noise_fac,
    deep_psf_fac,
    skip_wide,
    skip_deep,
):
    nxy = 201
    nxy_psf = 53
    scale = 0.2

    # Creating ngmix and dfmdet observations
    obs_w_ngmix, obs_d_ngmix, obs_dn_ngmix = make_simple_sim(
        seed=seed,
        g1=g1,
        g2=g2,
        s2n=s2n,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=False,
    )

    obs_w = ngmix_obs_to_dfmd_obs(obs_w_ngmix)
    obs_d = ngmix_obs_to_dfmd_obs(obs_d_ngmix)
    obs_dn = ngmix_obs_to_dfmd_obs(obs_dn_ngmix)

    (
        res_ngmix,
        (force_stepk_field, force_maxk_field, force_stepk_psf, force_maxk_psf),
    ) = single_band_deep_field_metadetect(
        obs_w_ngmix,
        obs_d_ngmix,
        obs_dn_ngmix,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
        return_k_info=True,
        max_min_fft_size=1024,
    )

    res, kinfo = jax_single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
        scale=scale,
        return_k_info=True,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        max_min_fft_size=1024,
    )

    assert kinfo[0] == force_stepk_field
    assert kinfo[1] == force_maxk_field
    assert kinfo[2] == force_stepk_psf
    assert kinfo[3] == force_maxk_psf

    return measure_mcal_shear_quants(res), measure_mcal_shear_quants(res_ngmix)


def _run_sim_pair_jax_and_ngmix(
    seed, s2n, deep_noise_fac, deep_psf_fac, skip_wide, skip_deep
):
    res_p, res_p_ngmix = _run_single_sim_jax_and_ngmix(
        seed,
        s2n,
        0.02,
        0.0,
        deep_noise_fac,
        deep_psf_fac,
        skip_wide,
        skip_deep,
    )

    res_m, res_m_ngmix = _run_single_sim_jax_and_ngmix(
        seed,
        s2n,
        -0.02,
        0.0,
        deep_noise_fac,
        deep_psf_fac,
        skip_wide,
        skip_deep,
    )

    return (res_p, res_m), (res_p_ngmix, res_m_ngmix)


def test_metadetect_single_band_deep_field_metadetect_smoke():
    res_p, res_m = _run_sim_pair(1234, 1e4, 1.0 / np.sqrt(10), 1, False, False)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


@pytest.mark.parametrize("deep_psf_ratio", [0.8, 1, 1.1])
def test_metadetect_single_band_deep_field_metadetect_jax_vs_ngmix(deep_psf_ratio):
    nsims = 5
    noise_fac = 1 / np.sqrt(30)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    res_p_ngmix = []
    res_m_ngmix = []
    for seed in seeds:
        res, res_ngmix = _run_sim_pair_jax_and_ngmix(
            seed, 1e4, noise_fac, deep_psf_ratio, False, False
        )
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])
            res_p_ngmix.append(res_ngmix[0])
            res_m_ngmix.append(res_ngmix[1])

            assert np.allclose(
                res[0].tolist(),
                res_ngmix[0].tolist(),
                atol=1e-5,
                rtol=0.025,
                equal_nan=True,
            )
            assert np.allclose(
                res[1].tolist(),
                res_ngmix[1].tolist(),
                atol=1e-5,
                rtol=0.025,
                equal_nan=True,
            )

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng = estimate_m_and_c(
        np.concatenate(res_p_ngmix),
        np.concatenate(res_m_ngmix),
        0.02,
        jackknife=len(res_p_ngmix),
    )

    print("JAX results:")
    print_m_c(m, merr, c1, c1err, c2, c2err)
    print("ngmix results:")
    print_m_c(m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)

    assert np.allclose(m, m_ng, atol=1e-4)
    assert np.allclose(merr, merr_ng, atol=1e-4)
    assert np.allclose(c1err, c1err_ng, atol=1e-6)
    assert np.allclose(c1, c1_ng, atol=1e-6)
    assert np.allclose(c2err, c2err_ng, atol=1e-6)
    assert np.allclose(c2, c2_ng, atol=1e-6)


def test_metadetect_single_band_deep_field_metadetect_bmask():
    nxy = 201
    nxy_psf = 53
    scale = 0.2

    rng = np.random.RandomState(seed=1234)
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=1234,
        g1=0.02,
        g2=0.00,
        s2n=1000,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=True,
    )
    obs_w = obs_w._replace(
        bmask=rng.choice([0, 1, 3], p=[0.5, 0.25, 0.25], size=obs_w.image.shape)
    )

    res, _ = jax_single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        skip_obs_wide_corrections=False,
        skip_obs_deep_corrections=False,
        scale=scale,
    )

    xc = (res["x"] + 0.5).astype(int)
    yc = (res["y"] + 0.5).astype(int)
    msk = res["mdet_step"] == "noshear"
    assert np.array_equal(obs_w.bmask[yc[msk], xc[msk]], res["bmask_flags"][msk])
    assert np.any(res["bmask_flags"][msk] != 0)

    for step in ["1p", "1m", "2p", "2m"]:
        msk = res["mdet_step"] == step
        assert not np.array_equal(
            obs_d.bmask[yc[msk], xc[msk]] | obs_dn.bmask[yc[msk], xc[msk]],
            res["bmask_flags"][msk],
        )


def test_metadetect_single_band_deep_field_metadetect_mfrac_wide():
    nxy = 201
    nxy_psf = 53
    scale = 0.2
    rng = np.random.RandomState(seed=1234)
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=1234,
        g1=0.02,
        g2=0.00,
        s2n=1000,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=True,
    )
    obs_w = obs_w._replace(mfrac=rng.uniform(0.5, 0.7, size=obs_w.image.shape))

    res, _ = jax_single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        skip_obs_wide_corrections=False,
        skip_obs_deep_corrections=False,
        scale=scale,
    )

    msk = (res["wmom_flags"] == 0) & (res["mdet_step"] == "noshear")
    assert np.all(res["mfrac"][msk] >= 0.5)
    assert np.all(res["mfrac"][msk] <= 0.7)

    msk = (res["wmom_flags"] == 0) & (res["mdet_step"] != "noshear")
    assert np.all(res["mfrac"][msk] == 0)


def test_metadetect_single_band_deep_field_metadetect_mfrac_deep():
    nxy = 201
    nxy_psf = 53
    scale = 0.2
    rng = np.random.RandomState(seed=1234)
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=1234,
        g1=0.02,
        g2=0.00,
        s2n=1000,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        buff=25,
        n_objs=10,
        return_dfmd_obs=True,
    )
    obs_d = obs_d._replace(mfrac=rng.uniform(0.5, 0.7, size=obs_w.image.shape))

    res, _ = jax_single_band_deep_field_metadetect(
        obs_w,
        obs_d,
        obs_dn,
        nxy=nxy,
        nxy_psf=nxy_psf,
        skip_obs_wide_corrections=False,
        skip_obs_deep_corrections=False,
        scale=scale,
    )

    msk = (res["wmom_flags"] == 0) & (res["mdet_step"] != "noshear")
    assert np.all(res["mfrac"][msk] >= 0.5)
    assert np.all(res["mfrac"][msk] <= 0.7)

    msk = (res["wmom_flags"] == 0) & (res["mdet_step"] == "noshear")
    assert np.all(res["mfrac"][msk] == 0)


@pytest.mark.parametrize("deep_psf_ratio", [0.8, 1, 1.1])
def test_metadetect_single_band_deep_field_metadetect(deep_psf_ratio):
    nsims = 100
    noise_fac = 1 / np.sqrt(30)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_sim_pair(seed, 1e4, noise_fac, deep_psf_ratio, False, False)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    assert np.isfinite(m)
    assert np.isfinite(merr)
    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


@pytest.mark.slow
@pytest.mark.parametrize(
    "skip_wide,skip_deep", [(True, True), (True, False), (False, True), (False, False)]
)
def test_metadetect_single_band_deep_field_metadetect_slow(
    skip_wide, skip_deep
):  # pragma: no cover
    if not skip_wide and not skip_deep:
        nsims = 1_000_000
        s2n = 20
    else:
        nsims = 100_000
        s2n = 10
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size + 1
    noise_fac = 1 / np.sqrt(10)
    nsims = nchunks * chunk_size

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc : loc + chunk_size]
        # jobs = [
        #     joblib.delayed(_run_sim_pair)(
        #         seed, s2n, noise_fac, 0.8, skip_wide, skip_deep
        #     )
        #     for seed in _seeds
        # ]
        # outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
        for seed in _seeds:
            res = _run_sim_pair(seed, s2n, noise_fac, 0.8, skip_wide, skip_deep)
            if res is not None:
                res_p.append(res[0])
                res_m.append(res[1])

        if len(res_p) < 500:
            njack = len(res_p)
        else:
            njack = 100

        m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
            np.concatenate(res_p),
            np.concatenate(res_m),
            0.02,
            jackknife=njack,
        )

        print("# of sims:", len(res_p), flush=True)
        print_m_c(m, merr, c1, c1err, c2, c2err)

        if not skip_wide and not skip_deep:
            assert np.abs(m) < max(MAX_ABS_M, 3 * merr), (m, merr)
        elif 3 * merr < 5e-3:
            assert np.abs(m) >= max(MAX_ABS_M, 3 * merr), (m, merr)
            # if we are more than 10 sigma biased, then the test
            # has passed for sure
            if np.abs(m) / max(MAX_ABS_M / 3, merr) >= 10:
                break
        assert np.abs(c1) < max(4.0 * c1err, MAX_ABS_C), (c1, c1err)
        assert np.abs(c2) < max(4.0 * c2err, MAX_ABS_C), (c2, c2err)

        loc += chunk_size

    print_m_c(m, merr, c1, c1err, c2, c2err)
    if not skip_wide and not skip_deep:
        assert np.abs(m) < max(MAX_ABS_M, 3 * merr), (m, merr)
    else:
        assert np.abs(m) >= max(MAX_ABS_M, 3 * merr), (m, merr)
    assert np.abs(c1) < max(4.0 * c1err, MAX_ABS_C), (c1, c1err)
    assert np.abs(c2) < max(4.0 * c2err, MAX_ABS_C), (c2, c2err)
