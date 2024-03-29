import multiprocessing

import joblib
import numpy as np
import pytest

from deep_metadetection.metacal import metacal_op_shears
from deep_metadetection.utils import (
    estimate_m_and_c,
    fit_gauss_mom,
    make_simple_sim,
    measure_mcal_shear_quants,
)


def _run_single_sim_pair(seed, s2n):  # pragma: no cover
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
    )
    mcal_res = metacal_op_shears(obs_plus)
    res_p = fit_gauss_mom(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
    )
    mcal_res = metacal_op_shears(obs_minus)
    res_m = fit_gauss_mom(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return res_p, res_m


def test_metacal():
    nsims = 50

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

    print(f" m: {m / 1e-3: f} +/- {3 * merr / 1e-3: f} [1e-3, 3-sigma]", flush=True)
    print(f"c1: {c1 / 1e-5: f} +/- {3 * c1err / 1e-5: f} [1e-5, 3-sigma]", flush=True)
    print(f"c2: {c2 / 1e-5: f} +/- {3 * c2err / 1e-5: f} [1e-5, 3-sigma]", flush=True)

    assert np.abs(m) < max(5e-4, 3 * merr), (m, merr)
    assert np.abs(c1) < 4.0 * c1err, (c1, c1err)
    assert np.abs(c2) < 4.0 * c2err, (c2, c2err)


@pytest.mark.slow
def test_metacal_slow():  # pragma: no cover
    nsims = 1_000_000
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
        print(f" m: {m / 1e-3: f} +/- {3 * merr / 1e-3: f} [1e-3, 3-sigma]", flush=True)
        print(
            f"c1: {c1 / 1e-5: f} +/- {3 * c1err / 1e-5: f} [1e-5, 3-sigma]", flush=True
        )
        print(
            f"c2: {c2 / 1e-5: f} +/- {3 * c2err / 1e-5: f} [1e-5, 3-sigma]", flush=True
        )

        loc += chunk_size

    assert np.abs(m) < max(5e-4, 3 * merr), (m, merr)
    assert np.abs(c1) < 4.0 * c1err, (c1, c1err)
    assert np.abs(c2) < 4.0 * c2err, (c2, c2err)
