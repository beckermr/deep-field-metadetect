import sys
import time
from contextlib import contextmanager

import galsim
import ngmix
import numpy as np
from ngmix.gaussmom import GaussMom

from deep_metadetection.metacal import DEFAULT_SHEARS

GLOBAL_START_TIME = time.time()


def get_measure_mcal_shear_quants_dtype(kind):
    return [
        (kind + "_tot_g1p", "f8"),
        (kind + "_tot_g1m", "f8"),
        (kind + "_tot_g1", "f8"),
        (kind + "_tot_g2p", "f8"),
        (kind + "_tot_g2m", "f8"),
        (kind + "_tot_g2", "f8"),
        (kind + "_num_g1p", "f8"),
        (kind + "_num_g1m", "f8"),
        (kind + "_num_g1", "f8"),
        (kind + "_num_g2p", "f8"),
        (kind + "_num_g2m", "f8"),
        (kind + "_num_g2", "f8"),
    ]


def measure_mcal_shear_quants(data, kind="wmom", s2n_cut=10, t_ratio_cut=1.2):
    """Measure metacal shear results.

    Parameters
    ----------
    data : array
        An array of the metacal data.
    kind : str, optional
        The kind of data to measure. Default is "wmom".
    s2n_cut : float, optional
        The cut in S/N. Default is 10.
    t_ratio_cut : float, optional
        The cut in T_ratio. Default is 1.2.

    Results
    -------
    res : np.ndarray
        Array with columns for sum of shears and number summed.
    """

    def _msk_it(*, d, mdet_step):
        msk = (
            (d["mdet_step"] == mdet_step)
            & (d[kind + "_flags"] == 0)
            & (d[kind + "_s2n"] > s2n_cut)
            & (d[kind + "_T_ratio"] > t_ratio_cut)
            & np.isfinite(d[kind + "_g1"])
            & np.isfinite(d[kind + "_g2"])
        )
        return msk

    msks = {}
    for mdet_step in DEFAULT_SHEARS:
        msks[mdet_step] = _msk_it(
            d=data,
            mdet_step=mdet_step,
        )

    res = np.zeros(1, dtype=get_measure_mcal_shear_quants_dtype(kind))
    res[kind + "_tot_g1p"] = np.nansum(data[kind + "_g1"][msks["1p"]])
    res[kind + "_tot_g1m"] = np.nansum(data[kind + "_g1"][msks["1m"]])
    res[kind + "_tot_g1"] = np.nansum(data[kind + "_g1"][msks["noshear"]])
    res[kind + "_tot_g2p"] = np.nansum(data[kind + "_g2"][msks["2p"]])
    res[kind + "_tot_g2m"] = np.nansum(data[kind + "_g2"][msks["2m"]])
    res[kind + "_tot_g2"] = np.nansum(data[kind + "_g2"][msks["noshear"]])
    res[kind + "_num_g1p"] = np.sum(msks["1p"])
    res[kind + "_num_g1m"] = np.sum(msks["1m"])
    res[kind + "_num_g1"] = np.sum(msks["noshear"])
    res[kind + "_num_g2p"] = np.sum(msks["2p"])
    res[kind + "_num_g2m"] = np.sum(msks["2m"])
    res[kind + "_num_g2"] = res[kind + "_num_g1"]

    return res


def _fill_nan(vals, i):
    vals["wmom_g1"][i] = np.nan
    vals["wmom_g2"][i] = np.nan
    vals["wmom_T_ratio"][i] = np.nan
    vals["wmom_psf_T"][i] = np.nan
    vals["wmom_s2n"][i] = np.nan


def fit_gauss_mom(mcal_res, fwhm=1.2):
    """Fit a m{cal/det} result dict using Gaussian moments.

    Parameters
    ----------
    mcal_res : dict
        The metacal result.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.

    Returns
    -------
    data : array
        The fit results
    """
    dt = [
        ("wmom_flags", "i4"),
        ("wmom_g1", "f8"),
        ("wmom_g2", "f8"),
        ("wmom_T_ratio", "f8"),
        ("wmom_psf_T", "f8"),
        ("wmom_s2n", "f8"),
        ("mdet_step", "U7"),
    ]
    vals = np.zeros(len(mcal_res), dtype=dt)

    fitter = GaussMom(fwhm)
    psf_res = fitter.go(mcal_res["noshear"].psf)

    for i, (shear, obs) in enumerate(mcal_res.items()):
        vals["mdet_step"][i] = shear

        if psf_res["flags"] != 0:
            vals["wmom_flags"][i] = ngmix.flags.NO_ATTEMPT
            _fill_nan(vals, i)
            continue

        res = fitter.go(obs)
        vals["wmom_flags"][i] = res["flags"]

        if res["flags"] != 0:
            _fill_nan(vals, i)
            continue

        vals["wmom_g1"][i] = res["e"][0]
        vals["wmom_g2"][i] = res["e"][1]
        vals["wmom_T_ratio"][i] = res["T"] / psf_res["T"]
        vals["wmom_s2n"][i] = res["s2n"]
        vals["wmom_psf_T"][i] = psf_res["T"]

    return vals


@contextmanager
def timer(name, silent=False):
    t0 = time.time()
    if not silent:
        print(
            "[% 8ds] %s" % (t0 - GLOBAL_START_TIME, name),
            flush=True,
            file=sys.stderr,
        )
    yield
    t1 = time.time()
    if not silent:
        print(
            "[% 8ds] %s done (%f seconds)" % (t1 - GLOBAL_START_TIME, name, t1 - t0),
            flush=True,
            file=sys.stderr,
        )


def _compute_g_R(d, kind, step):
    g1 = np.sum(d[kind + "_tot_g1"]) / np.sum(d[kind + "_num_g1"])
    g2 = np.sum(d[kind + "_tot_g2"]) / np.sum(d[kind + "_num_g2"])
    g1p = np.sum(d[kind + "_tot_g1p"]) / np.sum(d[kind + "_num_g1p"])
    g1m = np.sum(d[kind + "_tot_g1m"]) / np.sum(d[kind + "_num_g1m"])
    g2p = np.sum(d[kind + "_tot_g2p"]) / np.sum(d[kind + "_num_g2p"])
    g2m = np.sum(d[kind + "_tot_g2m"]) / np.sum(d[kind + "_num_g2m"])
    return g1, (g1p - g1m) / 2 / step, g2, (g2p - g2m) / 2 / step


def _compute_m_c12(pd, md, kind, step, g_true):
    g1p, R11p, g2p, R22p = _compute_g_R(pd, kind, step)
    g1m, R11m, g2m, R22m = _compute_g_R(md, kind, step)

    R11 = (R11p + R11m) / 2
    R22 = (R22p + R22m) / 2
    mg = (g1p - g1m) / 2
    m = mg / R11 / g_true - 1
    c1 = (g1p + g1m) / 2 / R11
    c2 = (g2p + g2m) / 2 / R22
    return m, c1, c2


def _comp_jk_est(theta_hat, theta_js):
    n = len(theta_js)
    mn_theta_js = np.mean(theta_js)
    theta_mn = n * theta_hat - (n - 1) * mn_theta_js
    theta_std = np.sqrt((n - 1) / n * np.sum((theta_js - mn_theta_js) ** 2))
    return theta_mn, theta_std


def _run_jackknife(pd, md, kind, step, g_true, jackknife):
    n_per = pd.shape[0] // jackknife

    pdj = np.zeros(jackknife, dtype=pd.dtype)
    mdj = np.zeros(jackknife, dtype=md.dtype)

    loc = 0
    for i in range(jackknife):
        for col in pd.dtype.names:
            pdj[col][i] = np.sum(pd[col][loc : loc + n_per])
            mdj[col][i] = np.sum(md[col][loc : loc + n_per])
        loc += n_per

    mhat, c1hat, c2hat = _compute_m_c12(pdj, mdj, kind, step, g_true)
    mvals = np.zeros(jackknife)
    c1vals = np.zeros(jackknife)
    c2vals = np.zeros(jackknife)
    for i in range(jackknife):
        _pdj = np.delete(pdj, i, axis=0)
        _mdj = np.delete(mdj, i, axis=0)
        mvals[i], c1vals[i], c2vals[i] = _compute_m_c12(_pdj, _mdj, kind, step, g_true)

    m_mn, m_std = _comp_jk_est(mhat, mvals)
    c1_mn, c1_std = _comp_jk_est(c1hat, c1vals)
    c2_mn, c2_std = _comp_jk_est(c2hat, c2vals)

    return m_mn, m_std, c1_mn, c1_std, c2_mn, c2_std


def _swap12_arr(d, kind):
    tails = ["g1p", "g1m", "g1"]
    stats = ["tot", "num"]
    old_data = {}
    for stat in stats:
        for tail in tails:
            col = kind + "_" + stat + "_" + tail
            old_data[kind + "_" + stat + "_" + tail] = d[col].copy()
    for stat in stats:
        for tail in tails:
            col_old = kind + "_" + stat + "_" + tail
            col_new = kind + "_" + stat + "_" + tail.replace("1", "2")
            d[col_old][:] = d[col_new]
            d[col_new][:] = old_data[col_old]
    return d


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    jackknife=None,
    silent=False,
    kind="wmom",
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : np.ndarray
        An array with the results from the function `measure_mcal_shear_quants`
        for the plus shear sims.
    mresults : np.ndarray
        An array with the results from the function `measure_mcal_shear_quants`
        for the minus shear sims.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.
    silent : bool, optional
        If True, do not print to stderr/stdout.
    kind : str, optional
        The kind of data to measure. Default is "wmom".

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c1 : float
        Estimate of the additive bias in direction with the shear applied.
    c1err : float
        Estimate of the 1-sigma standard error in `c1`.
    c2 : float
        Estimate of the additive bias in direction without shear.
    c2err : float
        Estimate of the 1-sigma standard error in `c2`.
    """

    with timer("prepping data for m,c measurement", silent=silent):
        if swap12:
            presults = _swap12_arr(presults, kind)
            mresults = _swap12_arr(mresults, kind)

    with timer("running jackknife", silent=silent):
        return _run_jackknife(presults, mresults, kind, step, g_true, jackknife)


def _make_single_sim(*, dither=None, rng, psf, obj, nse, scale, dim):
    cen = (dim - 1) / 2

    im = obj.drawImage(nx=dim, ny=dim, scale=scale).array
    im += rng.normal(size=im.shape, scale=nse)

    psf_im = psf.drawImage(nx=dim, ny=dim, scale=scale).array

    if dither is not None:
        jac = ngmix.DiagonalJacobian(
            scale=scale, row=cen + dither[1], col=cen + dither[0]
        )
    else:
        jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
    psf_jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im) / nse**2,
        jacobian=jac,
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
        noise=rng.normal(size=im.shape, scale=nse),
        bmask=np.zeros_like(im, dtype=np.int32),
        mfrac=np.zeros_like(im),
    )
    return obs


def make_simple_sim(
    *,
    seed,
    g1=0,
    g2=0,
    s2n=20,
    deep_noise_fac=1.0 / np.sqrt(10),
    deep_psf_fac=1.0,
    n_objs=1,
    scale=0.2,
    dim=53,
    buff=0,
    obj_flux_factor=1,
):
    """Make a simple simulation for testing deep-field metadetection.

    Parameters
    ----------
    seed : int
        The random seed.
    g1 : float
        The shear component 1.
    g2 : float
        The shear component 2.
    s2n : float
        The signal-to-noise ratio of the object.
    deep_noise_fac : float
        The factor by which to change the noise standard deviation in the deep-field.
    deep_psf_fac : float
        The factor by which to change the Moffat FWHM in the deep-field.
    scale : float, optional
        The pixel scale.
    dim : int, optional
        The image dimension.
    obj_flux_factor : float, optional
        The factor by which to change the object flux.

    Returns
    -------
    obs_wide : ngmix.Observation
        The wide-field observation.
    obs_deep : ngmix.Observation
        The deep-field observation.
    obs_deep_noise : ngmix.Observation
        The deep-field observation with noise but no object.
    """
    rng = np.random.RandomState(seed=seed)

    if n_objs > 1:
        n_objs = rng.poisson(lam=n_objs)
        xyrange = dim - buff * 2.0
        shifts = rng.uniform(size=(n_objs, 2), low=-0.5, high=0.5) * xyrange * scale
    else:
        shifts = rng.uniform(size=(n_objs, 2), low=-0.5, high=0.5) * scale

    gal = galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)
    gals = None
    for shift in shifts:
        if gals is None:
            gals = gal.shift(*shift)
        else:
            gals += gal.shift(*shift)

    psf = galsim.Moffat(beta=2.5, fwhm=0.8)
    deep_psf = galsim.Moffat(beta=2.5, fwhm=0.8 * deep_psf_fac)
    objs = galsim.Convolve([gals, psf])
    deep_objs = galsim.Convolve([gals, deep_psf])

    # estimate noise level
    im = galsim.Convolve([gal, psf]).drawImage(nx=dim, ny=dim, scale=scale).array
    nse = np.sqrt(np.sum(im**2)) / s2n

    # apply the flux factor now that we have the noise level
    objs *= obj_flux_factor
    deep_objs *= obj_flux_factor

    obs_wide = _make_single_sim(
        rng=rng,
        psf=psf,
        obj=objs,
        nse=nse,
        dither=shifts[0] / scale if n_objs == 1 else None,
        scale=scale,
        dim=dim,
    )

    obs_deep = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_objs,
        nse=nse * deep_noise_fac,
        dither=shifts[0] / scale if n_objs == 1 else None,
        scale=scale,
        dim=dim,
    )

    obs_deep_noise = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_objs * 0,
        nse=nse * deep_noise_fac,
        dither=shifts[0] / scale if n_objs == 1 else None,
        scale=scale,
        dim=dim,
    )

    return obs_wide, obs_deep, obs_deep_noise
