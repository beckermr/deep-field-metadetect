import multiprocessing

import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metacal import jax_metacal_op_shears
from deep_field_metadetect.jaxify.observation import ngmix_obs_to_dfmd_obs
from deep_field_metadetect.metacal import metacal_op_shears
from deep_field_metadetect.utils import (
    assert_m_c_ok,
    estimate_m_and_c,
    fit_gauss_mom_mcal_res,
    make_simple_sim,
    measure_mcal_shear_quants,
    print_m_c,
)


def _run_single_sim_pair(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return res_p, res_m


def _run_single_sim_pair_jax_and_ngmix(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_plus)

    res_p_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_p_ngmix = measure_mcal_shear_quants(res_p_ngmix)

    obs_plus = ngmix_obs_to_dfmd_obs(obs_plus)

    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_minus)
    res_m_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_m_ngmix = measure_mcal_shear_quants(res_m_ngmix)

    obs_minus = ngmix_obs_to_dfmd_obs(obs_minus)
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        scale=scale,
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return (res_p, res_m), (res_p_ngmix, res_m_ngmix)


def test_metacal_smoke():
    res_p, res_m = _run_single_sim_pair(1234, 1e8)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


def test_metacal_jax_vs_ngmix():
    nsims = 5

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    res_p_ngmix = []
    res_m_ngmix = []
    for seed in seeds:
        res, res_ngmix = _run_single_sim_pair_jax_and_ngmix(seed, 1e8)
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

    assert np.allclose(m, m_ng, atol=1e-4)
    assert np.allclose(merr, merr_ng, atol=1e-7)
    assert np.allclose(c1err, c1err_ng, atol=1e-7)
    assert np.allclose(c1, c1_ng, atol=1e-4)
    assert np.allclose(c2err, c2err_ng, atol=1e-7)
    assert np.allclose(c2, c2_ng, atol=1e-4)

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


def test_metacal():
    nsims = 50

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_single_sim_pair(seed, 1e8)
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
def test_metacal_slow():  # pragma: no cover
    nsims = 100_000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size + 1
    nsims = nchunks * chunk_size

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc : loc + chunk_size]
        for seed in _seeds:
            res = _run_single_sim_pair(seed, 20)
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

        loc += chunk_size

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)
