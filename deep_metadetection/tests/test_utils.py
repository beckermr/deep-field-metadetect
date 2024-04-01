import numpy as np
import pytest

from deep_metadetection.utils import (
    estimate_m_and_c,
    get_measure_mcal_shear_quants_dtype,
)


@pytest.mark.parametrize('swap12', [False])
@pytest.mark.parametrize('step', [0.005, 0.01])
@pytest.mark.parametrize('g_true', [0.05, 0.01, 0.02])
@pytest.mark.parametrize('jackknife', [100])
def test_estimate_m_and_c(g_true, step, swap12, jackknife):
    rng = np.random.RandomState(seed=10)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step * 10, np.mean(10 * (_gt + _step) + e2)
        else:
            return np.mean(10 * (_gt + _step) + e1), np.mean(e2) + cadd + _step * 10

    sn = 0.01
    n_gals = 10000
    n_sim = 1000
    pres = np.zeros(
        n_sim,
        dtype=get_measure_mcal_shear_quants_dtype("wmom"),
    )
    pres["wmom_num_g1p"] = n_gals
    pres["wmom_num_g1m"] = n_gals
    pres["wmom_num_g1"] = n_gals
    pres["wmom_num_g2p"] = n_gals
    pres["wmom_num_g2m"] = n_gals
    pres["wmom_num_g2"] = n_gals
    mres = np.zeros(
        n_sim,
        dtype=get_measure_mcal_shear_quants_dtype("wmom"),
    )
    mres["wmom_num_g1p"] = n_gals
    mres["wmom_num_g1m"] = n_gals
    mres["wmom_num_g1"] = n_gals
    mres["wmom_num_g2p"] = n_gals
    mres["wmom_num_g2m"] = n_gals
    mres["wmom_num_g2"] = n_gals

    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(g_true, step, e1, e2)
        g1m, g2m = _shear_meas(g_true, -step, e1, e2)
        if i != 0 and i != 750:
            pres[i]["wmom_tot_g1p"] = g1p
            pres[i]["wmom_tot_g1m"] = g1m
            pres[i]["wmom_tot_g1"] = g1
            pres[i]["wmom_tot_g2p"] = g2p
            pres[i]["wmom_tot_g2m"] = g2m
            pres[i]["wmom_tot_g2"] = g2

        g1, g2 = _shear_meas(-g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(-g_true, step, e1, e2)
        g1m, g2m = _shear_meas(-g_true, -step, e1, e2)
        if i != 250 and i != 750:
            mres[i]["wmom_tot_g1p"] = g1p
            mres[i]["wmom_tot_g1m"] = g1m
            mres[i]["wmom_tot_g1"] = g1
            mres[i]["wmom_tot_g2p"] = g2p
            mres[i]["wmom_tot_g2m"] = g2m
            mres[i]["wmom_tot_g2"] = g2

    m, _, c1, _, c2, _ = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step, jackknife=jackknife,
        silent=True
    )

    assert np.allclose(m, 0.01)
    assert np.allclose(c1, 0.00, rtol=0, atol=1e-6)
    assert np.allclose(c2, 0.05, rtol=0, atol=1e-6)


@pytest.mark.parametrize('seed', [1, 3, 454, 3454, 23443, 42])
@pytest.mark.parametrize('jackknife', [100])
def test_estimate_m_and_c_err(jackknife, seed):
    g_true = 0.02
    step = 0.01
    swap12 = False

    rng = np.random.RandomState(seed=seed)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step * 10, np.mean(10 * (_gt + _step) + e2)
        else:
            return np.mean(10 * (_gt + _step) + e1), np.mean(e2) + cadd + _step * 10

    sn = 0.5
    n_gals = 10
    n_sim = 1000
    pres = np.zeros(
        n_sim,
        dtype=get_measure_mcal_shear_quants_dtype("wmom"),
    )
    pres["wmom_num_g1p"] = n_gals
    pres["wmom_num_g1m"] = n_gals
    pres["wmom_num_g1"] = n_gals
    pres["wmom_num_g2p"] = n_gals
    pres["wmom_num_g2m"] = n_gals
    pres["wmom_num_g2"] = n_gals
    mres = np.zeros(
        n_sim,
        dtype=get_measure_mcal_shear_quants_dtype("wmom"),
    )
    mres["wmom_num_g1p"] = n_gals
    mres["wmom_num_g1m"] = n_gals
    mres["wmom_num_g1"] = n_gals
    mres["wmom_num_g2p"] = n_gals
    mres["wmom_num_g2m"] = n_gals
    mres["wmom_num_g2"] = n_gals
    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        e1p = rng.normal(size=n_gals) * sn
        e2p = rng.normal(size=n_gals) * sn

        e1m = rng.normal(size=n_gals) * sn
        e2m = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1 + e1p, e2 + e2p)
        g1p, g2p = _shear_meas(g_true, step, e1 + e1p, e2 + e2p)
        g1m, g2m = _shear_meas(g_true, -step, e1 + e1p, e2 + e2p)
        pres[i]["wmom_tot_g1p"] = g1p
        pres[i]["wmom_tot_g1m"] = g1m
        pres[i]["wmom_tot_g1"] = g1
        pres[i]["wmom_tot_g2p"] = g2p
        pres[i]["wmom_tot_g2m"] = g2m
        pres[i]["wmom_tot_g2"] = g2

        g1, g2 = _shear_meas(-g_true, 0, e1 + e1m, e2 + e2m)
        g1p, g2p = _shear_meas(-g_true, step, e1 + e1m, e2 + e2m)
        g1m, g2m = _shear_meas(-g_true, -step, e1 + e2m, e2 + e2m)
        mres[i]["wmom_tot_g1p"] = g1p
        mres[i]["wmom_tot_g1m"] = g1m
        mres[i]["wmom_tot_g1"] = g1
        mres[i]["wmom_tot_g2p"] = g2p
        mres[i]["wmom_tot_g2m"] = g2m
        mres[i]["wmom_tot_g2"] = g2

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step,
        jackknife=jackknife, silent=True
    )

    assert np.abs(m - 0.01) <= 3 * merr, (m, merr)
    assert np.abs(c1 - 0.00) <= 3 * c1err, (c1, c1err)
    assert np.abs(c2 - 0.05) <= 3 * c2err, (c2, c2err)
