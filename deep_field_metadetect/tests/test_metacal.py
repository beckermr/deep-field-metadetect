import multiprocessing

import jax.numpy as jnp
import joblib
import numpy as np
import pytest

from deep_field_metadetect.metacal import jax_metacal_op_shears
from deep_field_metadetect.utils import (
    assert_m_c_ok,
    estimate_m_and_c,
    fit_gauss_mom_mcal_res,
    make_simple_sim,
    measure_mcal_shear_quants,
    print_m_c,
)


def _run_single_sim_pair(seed, s2n):
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
    )
    # jax_gal_psf = get_jax_galsim_object_from_NT_obs_nopix(obs_plus.psf)
    mcal_res = jax_metacal_op_shears(obs_plus, dk=2 * jnp.pi / (53 * 0.2) / 4)
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
    )
    # jax_gal_psf = get_jax_galsim_object_from_NT_obs_nopix(obs_minus.psf)
    mcal_res = jax_metacal_op_shears(obs_minus, dk=2 * jnp.pi / (53 * 0.2) / 4)
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return res_p, res_m


def test_metacal_smoke():
    res_p, res_m = _run_single_sim_pair(1234, 1e8)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


def test_metacal():
    nsims = 5

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [joblib.delayed(_run_single_sim_pair)(seed, 1e8) for seed in seeds]
    outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
    res_p = []
    res_m = []
    for res in outputs:
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
        jobs = [joblib.delayed(_run_single_sim_pair)(seed, 20) for seed in _seeds]
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
        for res in outputs:
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
