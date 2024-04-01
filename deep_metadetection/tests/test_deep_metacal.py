import multiprocessing

import joblib
import numpy as np
import pytest

from deep_metadetection.metacal import metacal_wide_and_deep_psf_matched
from deep_metadetection.utils import (
    estimate_m_and_c,
    fit_gauss_mom,
    make_simple_sim,
    measure_mcal_shear_quants,
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
    obs_w, obs_d, obs_dn = make_simple_sim(
        seed=seed,
        g1=g1,
        g2=g2,
        s2n=s2n,
        deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac,
    )
    mcal_res = metacal_wide_and_deep_psf_matched(
        obs_w,
        obs_d,
        obs_dn,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
    )
    res = fit_gauss_mom(mcal_res)
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


@pytest.mark.parametrize("deep_psf_ratio", [0.8, 1, 1.2])
def test_deep_metacal(deep_psf_ratio):
    nsims = 50
    noise_fac = 1 / np.sqrt(10)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [
        joblib.delayed(_run_sim_pair)(
            seed, 1e8, noise_fac, deep_psf_ratio, False, False
        )
        for seed in seeds
    ]
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


def test_deep_metacal_widelows2n():
    nsims = 500
    noise_fac = 1 / np.sqrt(1000)

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [
        joblib.delayed(_run_sim_pair)(seed, 20, noise_fac, 1, False, False)
        for seed in seeds
    ]
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
@pytest.mark.parametrize(
    "skip_wide,skip_deep", [(True, True), (True, False), (False, True), (False, False)]
)
def test_deep_metacal_slow(skip_wide, skip_deep):  # pragma: no cover
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
        jobs = [
            joblib.delayed(_run_sim_pair)(
                seed, s2n, noise_fac, 0.8, skip_wide, skip_deep
            )
            for seed in _seeds
        ]
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

        if not skip_wide and not skip_deep:
            assert np.abs(m) < max(5e-4, 3 * merr), (m, merr)
        assert np.abs(c1) < 4.0 * c1err, (c1, c1err)
        assert np.abs(c2) < 4.0 * c2err, (c2, c2err)

        loc += chunk_size

    if not skip_wide and not skip_deep:
        assert np.abs(m) < max(5e-4, 3 * merr), (m, merr)
    else:
        assert np.abs(m) >= max(5e-4, 3 * merr), (m, merr)
    assert np.abs(c1) < 4.0 * c1err, (c1, c1err)
    assert np.abs(c2) < 4.0 * c2err, (c2, c2err)
