import logging

import jax
import jax.numpy as jnp
import numpy as np

from deep_field_metadetect.detect import (
    run_detection_sep,
)
from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.jax_detection import (
    detect_galaxies,
    jax_batch_generate_mbobs_for_detections,
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
)
from deep_field_metadetect.jaxify.observation import (
    dfmd_obs_to_ngmix_obs,
    jax_get_mb_obs,
)

logger = logging.getLogger(__name__)


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
    moment_stamp_size=48,
    use_sep=False,
    return_debug_info=False,
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
    moment_stamp_size: int, optional
        The size of the stamp (box_size) used for moment measurements.
        This determines the padding width (moment_stamp_size // 2) applied
        to observations before extracting sub-observations.
        Default: 48.
    use_sep: bool
        use sep for detection. Otherwise jax peak finder is used.
    return_debug_info: bool
        return detections and mcal_res for debugging

    Returns
    -------
    result : dict
        A dictionary containing the requested results with the following keys:

        - "dfmdet_res" : dict (always present)
            The deep-field metadetection results as a dictionary containing
            detection and measurement results for all shears. Keys include:
            - id, x, y, mdet_step, bmask_flags, mfrac
            - wmom_flags, wmom_g1, wmom_g2, wmom_T_ratio, wmom_psf_T, wmom_s2n
        - "kinfo" : tuple (only if return_k_info=True)
            Tuple containing (_force_stepk_field, _force_maxk_field,
            _force_stepk_psf, _force_maxk_psf).
        - "mcal_res" : dict (only if return_debug_info=True)
            The metacalibration results.
        - "detections" : list (only if return_debug_info=True)
            List of detection catalogs for each shear.
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
    )

    if return_k_info:
        mcal_res, kinfo = mcal_res

    psf_res = jax_fit_gauss_mom_obs(mcal_res["noshear"].psf)  # jitted
    all_detection_results = []
    detections = []
    for shear_idx, shear in enumerate(shears):
        if use_sep:
            obs = dfmd_obs_to_ngmix_obs(mcal_res[shear])
            detres = run_detection_sep(obs, nodet_flags=nodet_flags, detect_thresh=3)
            num_detections = len(detres["catalog"]["x"])
            logger.debug("num detections: %d", num_detections)

            # Pad SEP detections to max_objects
            if num_detections > max_objects:
                # Truncate if we have more detections than max_objects
                x_coords = jnp.array(detres["catalog"]["x"][:max_objects])
                y_coords = jnp.array(detres["catalog"]["y"][:max_objects])
                det_flag = jnp.ones(max_objects, dtype=jnp.int32)
            else:
                # Pad to max_objects with fill values
                x_coords = jnp.concatenate(
                    [
                        jnp.array(detres["catalog"]["x"]),
                        jnp.full(max_objects - num_detections, -1.0),
                    ]
                )
                y_coords = jnp.concatenate(
                    [
                        jnp.array(detres["catalog"]["y"]),
                        jnp.full(max_objects - num_detections, -1.0),
                    ]
                )
                det_flag = jnp.concatenate(
                    [
                        jnp.ones(num_detections, dtype=jnp.int32),
                        jnp.zeros(max_objects - num_detections, dtype=jnp.int32),
                    ]
                )

            # Compute bmask_flags for detections (capped at max_objects)
            num_to_process = min(num_detections, max_objects)
            ixc = (x_coords[:num_to_process] + 0.5).astype(int)
            iyc = (y_coords[:num_to_process] + 0.5).astype(int)
            bmask_flags_actual = obs.bmask[iyc, ixc]

            if num_detections >= max_objects:
                bmask_flags = bmask_flags_actual
            else:
                bmask_flags = jnp.concatenate(
                    [
                        bmask_flags_actual,
                        jnp.zeros(
                            max_objects - num_detections, dtype=bmask_flags_actual.dtype
                        ),
                    ]
                )

            detections.append(detres["catalog"])
        else:
            # Use the standard deviation of the noise image as the noise threshold
            noise_level = jnp.std(mcal_res[shear].noise)
            _, detres, _, det_flag = detect_galaxies(
                mcal_res[shear].image, noise=noise_level, max_objects=max_objects
            )
            num_detections = jnp.sum(det_flag == 1).item()
            logger.debug("Num detections: %d", num_detections)

            x_coords = detres[:, 1]
            y_coords = detres[:, 0]

            # Compute bmask_flags only for actual detections
            valid_mask = det_flag == 1
            ixc = jnp.where(valid_mask, (x_coords + 0.5).astype(int), 0)
            iyc = jnp.where(valid_mask, (y_coords + 0.5).astype(int), 0)
            bmask_flags = jnp.where(valid_mask, mcal_res[shear].bmask[iyc, ixc], 0)
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
            # For fill values (-1, -1), the interpolation will not make
            mfrac_vals = jnp.where(det_flag == 1, mfrac_vals, 0.0)
            return mfrac_vals

        def get_zero_mfrac():
            """Return zeros when no mfrac data."""
            return jnp.zeros(max_objects, dtype=jnp.float64)

        mfrac_vals = jax.lax.cond(
            jnp.any(mcal_res[shear].mfrac > 0), get_mfrac_values, get_zero_mfrac
        )

        # Pad observations once before extracting sub-observations
        # pad_width is half of moment_stamp_size
        pad_width = moment_stamp_size // 2 + 1
        padded_mbobs = jax_get_mb_obs(mcal_res[shear], pad_width=pad_width)

        padded_xs = x_coords + pad_width
        padded_ys = y_coords + pad_width

        # Use batched function to extract all sub-observations at once
        _, padded_xs_out, padded_ys_out, all_subobs = (
            jax_batch_generate_mbobs_for_detections(
                padded_mbobs,
                xs=padded_xs,
                ys=padded_ys,
                box_size=moment_stamp_size,
            )
        )

        # Process all detections in batch
        obj_ids = jnp.arange(1, max_objects + 1, dtype=jnp.int64)

        def process_detection(
            subobs, obj_id, obj_x, obj_y, bmask_flag, mfrac_val, det_flag_val
        ):
            return jax_fit_single_detection(
                mbobs=subobs[0][0],
                psf_res=psf_res,
                obj_id=obj_id,
                obj_x=obj_x,
                obj_y=obj_y,
                shear_idx=shear_idx,
                bmask_flag=bmask_flag,
                mfrac_val=mfrac_val,
                det_flag=det_flag_val,
            )

        # Vectorize
        batch_results = jax.vmap(process_detection)(
            all_subobs,
            obj_ids,
            padded_xs_out - pad_width,
            padded_ys_out - pad_width,
            bmask_flags,
            mfrac_vals,
            det_flag,
        )

        all_detection_results.append(batch_results)

    # Concatenate all batched results at once
    dfmdet_res = jax.tree_util.tree_map(
        lambda *arrays: jnp.concatenate(arrays), *all_detection_results
    )

    # Convert mdet_step_idx integers to string labels to match original format
    # TODO: check if the string can be returned directly
    mdet_step_strings = np.array(
        [DEFAULT_SHEARS[int(idx)] for idx in dfmdet_res["mdet_step_idx"]], dtype="U7"
    )
    dfmdet_res["mdet_step"] = mdet_step_strings
    del dfmdet_res["mdet_step_idx"]

    result = {"dfmdet_res": dfmdet_res}

    if return_k_info:
        result["kinfo"] = kinfo

    if return_debug_info:
        result["mcal_res"] = mcal_res
        result["detections"] = detections

    return result
