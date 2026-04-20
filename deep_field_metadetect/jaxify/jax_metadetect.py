import logging
from functools import partial

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
    jax_make_mb_coadd_from_list,
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
    moment_stamp_size=jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE,
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
        Default: jax_dfmd_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size: int
        k image size used for reconv psf computation
        Default: jax_dfmd_defaults.DEFAULT_KIM_SIZE
    max_objects: int
        Max number of objects in a field
        Default: jax_dfmd_defaults.MAX_OBJECTS
    moment_stamp_size: int, optional
        The size of the stamp (box_size) used for moment measurements.
        This determines the padding width (moment_stamp_size // 2) applied
        to observations before extracting sub-observations.
        Default: jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE.
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


@partial(
    jax.jit,
    static_argnames=[
        "nxy",
        "nxy_psf",
        "n_bands",
        "detband_indices",
        "shears",
        "skip_obs_wide_corrections",
        "skip_obs_deep_corrections",
        "return_k_info",
        "force_stepk_field",
        "force_maxk_field",
        "force_stepk_psf",
        "force_maxk_psf",
        "fft_size",
        "reconv_psf_dk",
        "reconv_psf_kim_size",
        "max_objects",
        "moment_stamp_size",
        "return_debug_info",
    ],
)
def _jax_multi_band_deep_field_metadetect_core(
    mb_obs_wide,
    mb_obs_deep,
    mb_obs_deep_noise,
    nxy,
    nxy_psf,
    n_bands,
    detband_indices=None,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=jax_dfmd_defaults.DEFAULT_FFT_SIZE,
    reconv_psf_dk=jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size=jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    max_objects=jax_dfmd_defaults.MAX_OBJECTS,
    moment_stamp_size=jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE,
    return_debug_info=False,
):
    """JIT-compiled core function for multi-band observations.

    This is the jitable core implementation. Use `jax_multi_band_deep_field_metadetect`
    for a dict-based API wrapper.

    Parameters
    ----------
    mb_obs_wide : DFMdetMultiBandObsList
        Multi-band wide-field observations (one DFMdetObservation per band).
    mb_obs_deep : DFMdetMultiBandObsList
        Multi-band deep-field observations.
    mb_obs_deep_noise : DFMdetMultiBandObsList
        Multi-band deep-field noise observations.
    nxy: int
        Image size
    nxy_psf: int
        PSF size
    n_bands : int
        Number of bands. Must match the length of mb_obs_wide, mb_obs_deep,
        and mb_obs_deep_noise.
    detband_indices : tuple of int, optional
        Tuple of band indices (e.g., (0, 1, 2)) to use for detection coadd.
        If None, all bands are used. Default is None.
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
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_stepk_psf: float, optional
        Force stepk for drawing PSF images.
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_maxk_psf: float, optional
        Force stepk for drawing PSF images
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    fft_size: int, optional
        To fix max and min values of FFT size.
        Default: jax_dfmd_defaults.DEFAULT_FFT_SIZE
    reconv_psf_dk: float
        The Fourier-space pixel scale used for reconv psf computation.
        Default: jax_dfmd_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size: int
        k image size used for reconv psf computation
        Default: jax_dfmd_defaults.DEFAULT_KIM_SIZE
    max_objects: int
        Max number of objects in a field
        Default: jax_dfmd_defaults.MAX_OBJECTS
    moment_stamp_size: int, optional
        The size of the stamp (box_size) used for moment measurements.
        This determines the padding width (moment_stamp_size // 2) applied
        to observations before extracting sub-observations.
        Default: jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE.
    return_debug_info: bool
        return detections and mcal_res for debugging

    Returns
    -------
    result : dict
        A dictionary containing the requested results with the following keys:

        - "dfmdet_res" : dict (always present)
            The deep-field metadetection results as a dictionary containing
            detection and measurement results for all shears and all bands.
            Keys include:
            - id, x, y, mdet_step_idx (int), band_idx (int), bmask_flags, mfrac
            - wmom_flags, wmom_g1, wmom_g2, wmom_T_ratio, wmom_psf_T, wmom_s2n
            Note: mdet_step_idx and band_idx are integers. Wrapper function
            to converts them to strings.
        - "kinfo" : tuple (only if return_k_info=True)
            Tuple containing (_force_stepk_field, _force_maxk_field,
            _force_stepk_psf, _force_maxk_psf) for each band.
        - "mcal_res" : tuple (only if return_debug_info=True)
            The metacalibration results for each band, as a tuple.
        - "detections" : list (only if return_debug_info=True)
            List of detection catalogs for each shear.
    """
    if shears is None:
        shears = DEFAULT_SHEARS

    if detband_indices is None:
        detband_indices = tuple(range(n_bands))

    # shear images for each band
    mcal_res_list = []
    kinfo_list = []
    # TODO: vmap this?
    # TODO: obtain the sheared images only for bands on which tests are to be run
    for band_idx in range(n_bands):
        mcal_res = jax_metacal_wide_and_deep_psf_matched(
            obs_wide=mb_obs_wide[band_idx][0],
            obs_deep=mb_obs_deep[band_idx][0],
            obs_deep_noise=mb_obs_deep_noise[band_idx][0],
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
            mcal_res, kinfo_band = mcal_res
            kinfo_list.append(kinfo_band)

        mcal_res_list.append(mcal_res)

    # Compute PSF results (for each band)
    psf_res_list = []
    for band_idx in range(n_bands):
        psf_res_list.append(
            jax_fit_gauss_mom_obs(mcal_res_list[band_idx]["noshear"].psf)
        )

    # For each shear, create observation list and run detection and measurements
    all_detection_results = []
    detections = []
    for shear_idx, shear in enumerate(shears):
        # Create coadd using detband_indices and run detection
        obs_list = [mcal_res_list[band_idx][shear] for band_idx in range(n_bands)]

        detobs_list = [obs_list[i] for i in detband_indices]
        detobs = jax_make_mb_coadd_from_list(detobs_list)

        noise_level = jnp.std(detobs.noise)
        _, detres, _, det_flag = detect_galaxies(
            detobs.image, noise=noise_level, max_objects=max_objects
        )

        x_coords = detres[:, 1]
        y_coords = detres[:, 0]

        # Compute bmask_flags only for actual detections
        valid_mask = det_flag == 1
        ixc = jnp.where(valid_mask, (x_coords + 0.5).astype(int), 0)
        iyc = jnp.where(valid_mask, (y_coords + 0.5).astype(int), 0)
        bmask_flags = jnp.where(valid_mask, detobs.bmask[iyc, ixc], 0)
        detections.append(detres)

        # Process measurements for each band for every detection
        for band_idx in range(n_bands):

            def get_mfrac_values():
                """Compute interpolated mfrac values."""
                _interp_mfrac = jax_compute_mfrac_interp_image(
                    mcal_res_list[band_idx][shear].mfrac,
                    mcal_res_list[band_idx][shear].wcs.local(),
                    fft_size=fft_size,
                )

                mfrac_vals = jax.vmap(lambda x, y: _interp_mfrac.xValue(x, y))(
                    x_coords, y_coords
                )
                # For fill values (-1, -1), the interpolation will not make sense
                mfrac_vals = jnp.where(det_flag == 1, mfrac_vals, 0.0)
                return mfrac_vals

            def get_zero_mfrac():
                """Return zeros when no mfrac data."""
                return jnp.zeros(max_objects, dtype=jnp.float64)

            mfrac_vals = jax.lax.cond(
                jnp.any(mcal_res_list[band_idx][shear].mfrac > 0),
                get_mfrac_values,
                get_zero_mfrac,
            )

            # Pad observations once before extracting sub-observations
            pad_width = moment_stamp_size // 2 + 1
            padded_mbobs_single = jax_get_mb_obs(
                mcal_res_list[band_idx][shear], pad_width=pad_width
            )

            padded_xs = x_coords + pad_width
            padded_ys = y_coords + pad_width

            # extract subobs
            _, padded_xs_out, padded_ys_out, all_subobs = (
                jax_batch_generate_mbobs_for_detections(
                    padded_mbobs_single,
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
                    psf_res=psf_res_list[band_idx],
                    obj_id=obj_id,
                    obj_x=obj_x,
                    obj_y=obj_y,
                    shear_idx=shear_idx,
                    bmask_flag=bmask_flag,
                    mfrac_val=mfrac_val,
                    det_flag=det_flag_val,
                )

            # Vectorize over detections
            batch_results = jax.vmap(process_detection)(
                all_subobs,
                obj_ids,
                padded_xs_out - pad_width,
                padded_ys_out - pad_width,
                bmask_flags,
                mfrac_vals,
                det_flag,
            )

            batch_results["band_idx"] = jnp.full(max_objects, band_idx, dtype=jnp.int32)
            all_detection_results.append(batch_results)

    # Concatenate all batched results at once
    dfmdet_res = jax.tree_util.tree_map(
        lambda *arrays: jnp.concatenate(arrays), *all_detection_results
    )

    result = {"dfmdet_res": dfmdet_res}

    if return_k_info:
        result["kinfo"] = tuple(kinfo_list)

    if return_debug_info:
        result["mcal_res"] = tuple(mcal_res_list)
        result["detections"] = detections

    return result


def jax_multi_band_deep_field_metadetect(
    obs_wide,
    obs_deep,
    obs_deep_noise,
    nxy,
    nxy_psf,
    detbands=None,
    step=DEFAULT_STEP,
    shears=None,
    skip_obs_wide_corrections=False,
    skip_obs_deep_corrections=False,
    return_k_info=False,
    force_stepk_field=0.0,
    force_maxk_field=0.0,
    force_stepk_psf=0.0,
    force_maxk_psf=0.0,
    fft_size=jax_dfmd_defaults.DEFAULT_FFT_SIZE,
    reconv_psf_dk=jax_dfmd_defaults.DEFAULT_RECONV_DK,
    reconv_psf_kim_size=jax_dfmd_defaults.DEFAULT_KIM_SIZE,
    max_objects=jax_dfmd_defaults.MAX_OBJECTS,
    moment_stamp_size=jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE,
    return_debug_info=False,
):
    """Run deep-field metadetection for multi-band observations using JAX.

    This is a wrapper function that provides a dict-based API for convenience.
    It converts dict inputs to MultiBandObsList and calls the jitted core function.

    Parameters
    ----------
    obs_wide : dict
        Dictionary of wide-field DFMdetObservation objects keyed by band name.
    obs_deep : dict
        Dictionary of deep-field DFMdetObservation objects keyed by band name.
    obs_deep_noise : dict
        Dictionary of deep-field noise DFMdetObservation objects keyed by band name.
    nxy: int
        Image size
    nxy_psf: int
        PSF size
    detbands : tuple of str, optional
        tuple of band names to use for detection. If None, all bands
        in obs_wide are used for detection. Default is None.
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
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_stepk_psf: float, optional
        Force stepk for drawing PSF images.
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    force_maxk_psf: float, optional
        Force stepk for drawing PSF images
        Defaults to 0.0, which lets JaxGalsim choose the value.
        Used mainly for testing.
    fft_size: int, optional
        To fix max and min values of FFT size.
        Default: jax_dfmd_defaults.DEFAULT_FFT_SIZE
    reconv_psf_dk: float
        The Fourier-space pixel scale used for reconv psf computation.
        Default: jax_dfmd_defaults.DEFAULT_RECONV_DK
    reconv_psf_kim_size: int
        k image size used for reconv psf computation
        Default: jax_dfmd_defaults.DEFAULT_KIM_SIZE
    max_objects: int
        Max number of objects in a field
        Default: jax_dfmd_defaults.MAX_OBJECTS
    moment_stamp_size: int, optional
        The size of the stamp (box_size) used for moment measurements.
        This determines the padding width (moment_stamp_size // 2) applied
        to observations before extracting sub-observations.
        Default: jax_dfmd_defaults.DEFAULT_MOMENT_STAMP_SIZE.
    return_debug_info: bool
        return detections and mcal_res for debugging

    Returns
    -------
    result : dict
        A dictionary containing the requested results with the following keys:

        - "dfmdet_res" : dict (always present)
            The deep-field metadetection results as a dictionary containing
            detection and measurement results for all shears and all bands.
            Keys include:
            - id, x, y, mdet_step, band, bmask_flags, mfrac
            - wmom_flags, wmom_g1, wmom_g2, wmom_T_ratio, wmom_psf_T, wmom_s2n
        - "kinfo" : dict (only if return_k_info=True)
            Dictionary keyed by band name containing tuples with
            (_force_stepk_field, _force_maxk_field, _force_stepk_psf, _force_maxk_psf).
        - "mcal_res" : dict (only if return_debug_info=True)
            The metacalibration results for each band, keyed by band name.
        - "detections" : list (only if return_debug_info=True)
            List of detection catalogs for each shear.
    """
    from deep_field_metadetect.jaxify.observation import (
        DFMdetMultiBandObsList,
        DFMdetObsList,
    )

    # Convert inputs to MultiBandObsList
    bands = tuple(obs_wide.keys())
    mb_obs_wide = DFMdetMultiBandObsList(
        [DFMdetObsList([obs_wide[band]]) for band in bands]
    )
    mb_obs_deep = DFMdetMultiBandObsList(
        [DFMdetObsList([obs_deep[band]]) for band in bands]
    )
    mb_obs_deep_noise = DFMdetMultiBandObsList(
        [DFMdetObsList([obs_deep_noise[band]]) for band in bands]
    )

    n_bands = len(bands)

    if detbands is None:
        detband_indices = None
    else:
        band_to_idx = {band: idx for idx, band in enumerate(bands)}
        detband_indices = tuple(band_to_idx[band] for band in detbands)

    if shears is None:
        shears = DEFAULT_SHEARS

    # Call jitted core function
    result = _jax_multi_band_deep_field_metadetect_core(
        mb_obs_wide=mb_obs_wide,
        mb_obs_deep=mb_obs_deep,
        mb_obs_deep_noise=mb_obs_deep_noise,
        nxy=nxy,
        nxy_psf=nxy_psf,
        n_bands=n_bands,
        detband_indices=detband_indices,
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
        max_objects=max_objects,
        moment_stamp_size=moment_stamp_size,
        return_debug_info=return_debug_info,
    )

    dfmdet_res = result["dfmdet_res"]

    # Log detection counts (happens outside jit)
    for shear_idx, shear in enumerate(shears):
        mask = dfmdet_res["mdet_step_idx"] == shear_idx
        num_detections = jnp.sum(mask).item()
        logger.debug("shear=%s, num detections: %d", shear, num_detections)

    # Convert indices to strings
    mdet_step_strings = np.array(
        [shears[int(idx)] for idx in dfmdet_res["mdet_step_idx"]], dtype="U7"
    )
    dfmdet_res["mdet_step"] = mdet_step_strings
    del dfmdet_res["mdet_step_idx"]

    band_strings = np.array(
        [bands[int(idx)] for idx in dfmdet_res["band_idx"]], dtype="U10"
    )
    dfmdet_res["band"] = band_strings
    del dfmdet_res["band_idx"]

    result["dfmdet_res"] = dfmdet_res

    # Convert kinfo tuple to dict
    if return_k_info:
        kinfo_tuple = result["kinfo"]
        result["kinfo"] = {bands[i]: kinfo_tuple[i] for i in range(len(bands))}

    # Convert mcal_res tuple to dict
    if return_debug_info:
        mcal_res_tuple = result["mcal_res"]
        result["mcal_res"] = {bands[i]: mcal_res_tuple[i] for i in range(len(bands))}

    return result
