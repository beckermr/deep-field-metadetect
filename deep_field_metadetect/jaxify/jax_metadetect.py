import jax
import jax.numpy as jnp
import numpy as np

from deep_field_metadetect.detect import (
    run_detection_sep,
)
from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.jax_detection import (
    detect_galaxies,
    jax_generate_mbobs_for_detections,
)
from deep_field_metadetect.jaxify.jax_metacal import (
    DEFAULT_SHEARS,
    DEFAULT_STEP,
    jax_metacal_wide_and_deep_psf_matched,
)
from deep_field_metadetect.jaxify.jax_mfrac import jax_compute_mfrac_interp_image
from deep_field_metadetect.jaxify.jax_utils import (
    jax_fit_gauss_mom_obs,
    jax_fit_single_detection,
    stack_detection_results,
)
from deep_field_metadetect.jaxify.observation import (
    dfmd_obs_to_ngmix_obs,
    jax_get_mb_obs,
)


def jax_single_band_deep_field_metadetect(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    nxy,
    nxy_psf,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    nodet_flags=0,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=jax_dfmd_defaults.DEFAULT_FFT_SIZE,
    reconv_psf_dk=jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size=jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    max_objects=jax_dfmd_defaults.MAX_OBJECTS,
    use_sep=False,
    return_debug_info=False,
    peak_finder_noise=1e-5,
):
    """Run deep-field metadetection for a simple scenario of a single band
    with a single image per band using only post-PSF Gaussian weighted moments.

    Parameters
    ----------
    obs_wide : DFMdetObservation
        The wide-field observation.
    obs_deep : DFMdetObservation
        The deep-field observation.
    obs_deep_noise : DFMdetObservation
        The deep-field noise observation.
    nxy: int
        Image size
    nxy_psf: int
        PSF size
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
    return_k_info : bool, optional
        return _force stepk and maxk values in the following order
        _force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf.
        Used mainly for testing.
    force_stepk_field : float, optional
        Force stepk for drawing field images.
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_maxk_field: float, optional
        Force maxk for drawing field images.
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    force_stepk_psf: float, optional
        Force stepk for drawing PSF images.
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    force_maxk_psf: float, optional
        Force stepk for drawing PSF images
        Defaults to 0.0, which lets Galsim choose the value.
        Used mainly for testing.
    fft_size: int, optional
        To fix max and min values of FFT size.
        Defaults to None which lets Galsim determine the values.
        Used mainly to test against JaxGalsim.
    reconv_psf_dk: float
        The Fourier-space pixel scale used for reconv psf computation.
        Default: jax_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size: int
        k image size used for reconv psf computation
        Default: jax_defaults.DEFAULT_KIM_SIZE
    max_objects: int
        Max number of objects in a field
    use_sep: bool
        use sep for detection. Otherwise jax peak finder is used.
    return_debug_info: bool
        return detections and mcal_res for debugging
    peak_finder_noise: float
        used only when use_sep is False. Sets the noise level for detection.
        Default: 1e-5.

    Returns
    -------
    dfmdet_res : dict
        The deep-field metadetection results as a dictionary containing
        detection and measurement results for all shears. Keys include:
        - id, x, y, mdet_step, bmask_flags, mfrac
        - wmom_flags, wmom_g1, wmom_g2, wmom_T_ratio, wmom_psf_T, wmom_s2n
        Each value is a 1D array with shape (n_total_detections,).
        Note: mdet_step is a NumPy string array ('U7'), all others are JAX arrays.
    mcal_res : dict
        The metacalibration results.
    detections : list
        List of detection catalogs for each shear.

    Note: If return_k_info is set to True for debugging,
    the function returns a tuple containing ((dfmdet_res, kinfo), mcal_res, detections).
    kinfo: (_force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf)
    """
    if shears is None:
        shears = DEFAULT_SHEARS

    mcal_res = jax_metacal_wide_and_deep_psf_matched(
        obs_wide=obs_wide,
        obs_deep=obs_deep,
        obs_deep_noise=obs_deep_noise,
        nxy=nxy,
        nxy_psf=nxy_psf,
        step=step,
        shears=shears,
        skip_obs_wide_corrections=skip_obs_wide_corrections,
        skip_obs_deep_corrections=skip_obs_deep_corrections,
        return_k_info=return_k_info,
        force_stepk_field=force_stepk_field,
        force_maxk_field=force_maxk_field,
        force_stepk_psf=force_stepk_psf,
        force_maxk_psf=force_maxk_psf,
        fft_size=fft_size,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )  # This returns ngmix Obs for now

    if return_k_info:
        mcal_res, kinfo = mcal_res

    psf_res = jax_fit_gauss_mom_obs(mcal_res["noshear"].psf)  # jitted
    all_detection_results = []
    detections = []
    for shear_idx, shear in enumerate(shears):
        if use_sep:
            obs = dfmd_obs_to_ngmix_obs(mcal_res[shear])
            detres = run_detection_sep(obs, nodet_flags=nodet_flags, detect_thresh=3)
            print("num detections : " + str(len(detres["catalog"]["x"])))

            ixc = (detres["catalog"]["x"] + 0.5).astype(int)
            iyc = (detres["catalog"]["y"] + 0.5).astype(int)
            bmask_flags = obs.bmask[iyc, ixc]
            detections.append(detres["catalog"])

            x_coords = detres["catalog"]["x"]
            y_coords = detres["catalog"]["y"]
        else:
            _, detres, _ = detect_galaxies(
                mcal_res[shear].image, noise=peak_finder_noise, max_objects=max_objects
            )  # TODO: Noise threshold
            valid_peak_mask = (detres[:, 0] >= 0) & (detres[:, 1] >= 0)
            detres = detres[valid_peak_mask]

            print("Num detections " + str(len(detres)))

            # Extract coordinates
            x_coords = detres[:, 0]  # TODO: check if invalid peaks handled well
            y_coords = detres[:, 0]

            ixc = (x_coords + 0.5).astype(int)
            iyc = (y_coords + 0.5).astype(int)
            bmask_flags = mcal_res[shear].bmask[iyc, ixc]
            detections.append(detres)

        def get_mfrac_values():
            """Compute interpolated mfrac values."""
            _interp_mfrac = jax_compute_mfrac_interp_image(
                mcal_res[shear].mfrac,
                mcal_res[shear].wcs.local(),
                fft_size=fft_size,
            )

            mfrac_vals = jax.vmap(lambda x, y: _interp_mfrac.xValue(x, y))(
                x_coords, y_coords
            )
            return mfrac_vals

        def get_zero_mfrac():
            """Return zeros when no mfrac data."""
            return jnp.zeros_like(bmask_flags, dtype=jnp.float64)

        mfrac_vals = jax.lax.cond(
            jnp.any(mcal_res[shear].mfrac > 0), get_mfrac_values, get_zero_mfrac
        )

        for ind, (obj, subobs) in enumerate(
            jax_generate_mbobs_for_detections(
                jax_get_mb_obs(mcal_res[shear]),
                xs=x_coords,
                ys=y_coords,
            )
        ):
            # Process single detection and get PyTree result
            single_result = jax_fit_single_detection(
                mbobs=subobs[0][0],
                psf_res=psf_res,
                obj_id=ind + 1,
                obj_x=obj["x"],
                obj_y=obj["y"],
                shear_idx=shear_idx,
                bmask_flag=bmask_flags[ind],
                mfrac_val=mfrac_vals[ind],
            )
            all_detection_results.append(single_result)

    # Stack all detection results into arrays
    dfmdet_res = stack_detection_results(all_detection_results)

    # Convert mdet_step_idx integers to string labels to match original format
    # TODO: check if the string can be returned directly
    mdet_step_strings = np.array(
        [DEFAULT_SHEARS[int(idx)] for idx in dfmdet_res["mdet_step_idx"]], dtype="U7"
    )
    dfmdet_res["mdet_step"] = mdet_step_strings
    del dfmdet_res["mdet_step_idx"]

    if return_debug_info:
        if return_k_info:
            return (dfmdet_res, kinfo), mcal_res, detections

        return dfmdet_res, mcal_res, detections

    if return_k_info:
        return (dfmdet_res, kinfo)

    return dfmdet_res
