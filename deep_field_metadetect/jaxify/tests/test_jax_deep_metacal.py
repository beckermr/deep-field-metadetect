import multiprocessing

import jax.numpy as jnp
import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metacal import (
    jax_metacal_op_shears,
    jax_metacal_wide_and_deep_psf_matched,
)
from deep_field_metadetect.jaxify.observation import dfmd_obs_to_ngmix_obs
from deep_field_metadetect.utils import (
    MAX_ABS_C,
    MAX_ABS_M,
    assert_m_c_ok,
    estimate_m_and_c,
    fit_gauss_mom_mcal_res,
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
    nxy = 53
    nxy_psf = 53
    scale = 0.2

    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=seed,
        g1=g1,
        g2=g2,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_wide_and_deep_psf_matched(
        obs_w,
        obs_d,
        obs_dn,
        dk_w=2 * jnp.pi / (nxy_psf * scale) / 4,
        dk_d=2 * jnp.pi / (nxy_psf * scale) / 4,
        nxy=53,
        nxy_psf=53,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
    )
    res = fit_gauss_mom_mcal_res(mcal_res)
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


def test_deep_metacal_smoke():
    res_p, res_m = _run_sim_pair(1234, 1e8, 1.0 / np.sqrt(10), 1, False, False)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


@pytest.mark.parametrize("deep_psf_ratio", [0.8, 1, 1.2])
def test_deep_metacal(deep_psf_ratio):
    nsims = 50
    noise_fac = 1 / np.sqrt(10)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)

    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_sim_pair(seed, 1e8, noise_fac, deep_psf_ratio, False, False)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


def test_deep_metacal_widelows2n():
    nsims = 500
    noise_fac = 1 / np.sqrt(1000)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)

    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_sim_pair(seed, 20, noise_fac, 1, False, False)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


@pytest.mark.slow
@pytest.mark.parametrize(
    "skip_wide,skip_deep", [(True, True), (True, False), (False, True), (False, False)]
)
def test_deep_metacal_slow(skip_wide, skip_deep):  # pragma: no cover
    if not skip_wide and not skip_deep:
        nsims = 100_000
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


def _run_single_sim_maybe_mcal(
    seed,
    s2n,
    g1,
    g2,
    deep_noise_fac,
    deep_psf_fac,
    use_mcal,
    zero_flux,
):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=seed,
        g1=g1,
        g2=g2,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
        obj_flux_factor=0.0 if zero_flux else 1.0,
        return_dfmd_obs=True,
    )
    if use_mcal:
        mcal_res = jax_metacal_op_shears(
            obs_w,
            dk=jnp.pi / (nxy_psf * scale) / 4,
        )
        for key, value in mcal_res.items():
            mcal_res[key] = dfmd_obs_to_ngmix_obs(value)
    else:
        mcal_res = jax_metacal_wide_and_deep_psf_matched(
            obs_w,
            obs_d,
            obs_dn,
            dk_w=2 * jnp.pi / (nxy_psf * scale) / 4,
            dk_d=2 * jnp.pi / (nxy_psf * scale) / 4,
            nxy=nxy,
            nxy_psf=nxy_psf,
        )
    return fit_gauss_mom_mcal_res(mcal_res), mcal_res


def test_deep_metacal_noise_object_s2n():
    nsims = 100
    noise_fac = 1 / np.sqrt(10)
    s2n = 10

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)

    dmcal_res = []
    mcal_res = []
    for seed in seeds:
        dmcal_res.append(
            _run_single_sim_maybe_mcal(
                seed,
                s2n,
                0.02,
                0.0,
                noise_fac,
                1.0,
                False,
                False,
            )
        )
        mcal_res.append(
            _run_single_sim_maybe_mcal(
                seed,
                s2n,
                0.02,
                0.0,
                noise_fac,
                1.0,
                True,
                False,
            )
        )

    dmcal_res = np.concatenate([d[0] for d in dmcal_res if d is not None], axis=0)
    mcal_res = np.concatenate([d[0] for d in mcal_res if d is not None], axis=0)
    dmcal_res = dmcal_res[dmcal_res["mdet_step"] == "noshear"]
    mcal_res = mcal_res[mcal_res["mdet_step"] == "noshear"]

    ratio = (np.median(dmcal_res["wmom_s2n"]) / np.median(mcal_res["wmom_s2n"])) ** 2
    print("s2n ratio squared:", ratio)
    assert np.allclose(ratio, 2, atol=0, rtol=0.2), ratio

    dmcal_res = []
    mcal_res = []
    for seed in seeds:
        dmcal_res.append(
            _run_single_sim_maybe_mcal(
                seed,
                s2n,
                0.02,
                0.0,
                noise_fac,
                1.0,
                False,
                True,
            )
        )
        mcal_res.append(
            _run_single_sim_maybe_mcal(
                seed,
                s2n,
                0.02,
                0.0,
                noise_fac,
                1.0,
                True,
                True,
            )
        )

    dmcal_res = np.array(
        [np.std(d[1]["noshear"].image) for d in dmcal_res if d is not None]
    )
    mcal_res = np.array(
        [np.std(d[1]["noshear"].image) for d in mcal_res if d is not None]
    )

    ratio = (np.median(dmcal_res) / np.median(mcal_res)) ** 2
    print("noise ratio squared:", ratio)
    assert np.allclose(ratio, 0.5, atol=0, rtol=0.2), ratio
