import ngmix
import numpy as np

from deep_field_metadetect.detect import (
    generate_mbobs_for_detections,
    run_detection_sep,
)
from deep_field_metadetect.metacal import (
    DEFAULT_SHEARS,
    DEFAULT_STEP,
    metacal_wide_and_deep_psf_matched,
    jax_metacal_wide_and_deep_psf_matched,
)
from deep_field_metadetect.mfrac import compute_mfrac_interp_image
from deep_field_metadetect.utils import fit_gauss_mom_obs, fit_gauss_mom_obs_and_psf

def single_band_deep_field_metadetect(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    nodet_flags=0,
):
    """Run deep-field metadetection for a simple scenario of a single band
    with a single image per band using only post-PSF Gaussian weighted moments.

    Parameters
    ----------
    obs_wide : ngmix.Observation
        The wide-field observation.
    obs_deep : ngmix.Observation
        The deep-field observation.
    obs_deep_noise : ngmix.Observation
        The deep-field noise observation.
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    shears : list, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    nodet_flags : int, optional
        The bmask flags marking area in the image to skip, by default 0.

    Returns
    -------
    dfmdet_res : dict
        The deep-field metadetection results, a dictionary with keys from `shears`
        and values containing the detection+measurement results for the corresponding
        shear.
    """
    if shears is None:
        shears = DEFAULT_SHEARS

    mcal_res = metacal_wide_and_deep_psf_matched(
        obs_wide,
        obs_deep,
        obs_deep_noise,
        step=step,
        shears=shears,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    )
    psf_res = fit_gauss_mom_obs(mcal_res["noshear"].psf)
    dfmdet_res = []
    for shear, obs in mcal_res.items():
        detres = run_detection_sep(obs, nodet_flags=nodet_flags)

        ixc = (detres["catalog"]["x"] + 0.5).astype(int)
        iyc = (detres["catalog"]["y"] + 0.5).astype(int)
        bmask_flags = obs.bmask[iyc, ixc]

        mfrac_vals = np.zeros_like(bmask_flags, dtype="f4")
        if np.any(obs.mfrac > 0):
            _interp_mfrac = compute_mfrac_interp_image(
                obs.mfrac,
                obs.jacobian.get_galsim_wcs(),
            )
            for i, (x, y) in enumerate(
                zip(detres["catalog"]["x"], detres["catalog"]["y"])
            ):
                mfrac_vals[i] = _interp_mfrac.xValue(x, y)

        for ind, (obj, mbobs) in enumerate(
            generate_mbobs_for_detections(
                ngmix.observation.get_mb_obs(obs),
                xs=detres["catalog"]["x"],
                ys=detres["catalog"]["y"],
            )
        ):
            fres = fit_gauss_mom_obs_and_psf(mbobs[0][0], psf_res=psf_res)
            dfmdet_res.append(
                (ind + 1, obj["x"], obj["y"], shear, bmask_flags[ind], mfrac_vals[ind])
                + tuple(fres[0])
            )

    total_dtype = [
        ("id", "i8"),
        ("x", "f8"),
        ("y", "f8"),
        ("mdet_step", "U7"),
        ("bmask_flags", "i4"),
        ("mfrac", "f4"),
    ] + fres.dtype.descr

    return np.array(dfmdet_res, dtype=total_dtype)


def jax_single_band_deep_field_metadetect(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    dk_w,
    dk_d,
    nxy,
    nxy_psf,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    nodet_flags=0,
):
    """Run deep-field metadetection for a simple scenario of a single band
    with a single image per band using only post-PSF Gaussian weighted moments.

    Parameters
    ----------
    obs_wide : ngmix.Observation
        The wide-field observation.
    obs_deep : ngmix.Observation
        The deep-field observation.
    obs_deep_noise : ngmix.Observation
        The deep-field noise observation.
    step : float, optional
        The step size for the metacalibration, by default DEFAULT_STEP.
    shears : list, optional
        The shears to use for the metacalibration, by default DEFAULT_SHEARS
        if set to None.
    skip_obs_wide_corrections : bool, optional
        Skip the observation corrections for the wide-field observations,
        by default False.
    skip_obs_deep_corrections : bool, optional
        Skip the observation corrections for the deep-field observations,
        by default False.
    nodet_flags : int, optional
        The bmask flags marking area in the image to skip, by default 0.

    Returns
    -------
    dfmdet_res : dict
        The deep-field metadetection results, a dictionary with keys from `shears`
        and values containing the detection+measurement results for the corresponding
        shear.
    """
    if shears is None:
        shears = DEFAULT_SHEARS

    mcal_res = jax_metacal_wide_and_deep_psf_matched(
        obs_wide=obs_wide,
        obs_deep=obs_deep,
        obs_deep_noise=obs_deep_noise,
        dk_w=dk_w,
        dk_d=dk_d,
        nxy=nxy,
        nxy_psf=nxy_psf,
        step=step,
        shears=shears,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
    ) # This returns ngmix Obs for now

    psf_res = fit_gauss_mom_obs(mcal_res["noshear"].psf)
    dfmdet_res = []
    for shear, obs in mcal_res.items():
        detres = run_detection_sep(obs, nodet_flags=nodet_flags)

        ixc = (detres["catalog"]["x"] + 0.5).astype(int)
        iyc = (detres["catalog"]["y"] + 0.5).astype(int)
        bmask_flags = obs.bmask[iyc, ixc]

        mfrac_vals = np.zeros_like(bmask_flags, dtype="f4")
        if np.any(obs.mfrac > 0):
            _interp_mfrac = compute_mfrac_interp_image(
                obs.mfrac,
                obs.jacobian.get_galsim_wcs(),
            )
            for i, (x, y) in enumerate(
                zip(detres["catalog"]["x"], detres["catalog"]["y"])
            ):
                mfrac_vals[i] = _interp_mfrac.xValue(x, y)

        for ind, (obj, mbobs) in enumerate(
            generate_mbobs_for_detections(
                ngmix.observation.get_mb_obs(obs),
                xs=detres["catalog"]["x"],
                ys=detres["catalog"]["y"],
            )
        ):
            fres = fit_gauss_mom_obs_and_psf(mbobs[0][0], psf_res=psf_res)
            dfmdet_res.append(
                (ind + 1, obj["x"], obj["y"], shear, bmask_flags[ind], mfrac_vals[ind])
                + tuple(fres[0])
            )

    total_dtype = [
        ("id", "i8"),
        ("x", "f8"),
        ("y", "f8"),
        ("mdet_step", "U7"),
        ("bmask_flags", "i4"),
        ("mfrac", "f4"),
    ] + fres.dtype.descr

    return np.array(dfmdet_res, dtype=total_dtype)


