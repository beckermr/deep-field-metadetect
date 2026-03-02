import jax
import jax.numpy as jnp
import ngmix
import numpy as np

from deep_field_metadetect.gaussmom.gaussmom import GaussMom
from deep_field_metadetect.gaussmom.gaussmom_core import dfmd_obs_to_gaussmom_obs


def compute_dk(pixel_scale, image_size):
    """Compute psf fourier scale based on pixel scale and PSF image dimension.
    The size is obtained from galsim.GSObject.getGoodImageSize.
    The factor 1/4 from deep_field_metadetect.metacal.get_gauss_reconv_psf_galsim.

    Parameters:
    -----------
    pixel_scale : float
        The scale of a single pixel in the image.
    image_size : int
        The dimension of the image (typically a square size).

    Returns:
    --------
    float
        The computed stepk value, which represents the Fourier-space sampling frequency.
    """
    return 2 * np.pi / (image_size * pixel_scale) / 4


def compute_kim_size(image_size):
    """Compute the size of image in the fourier space used the determine
    the reconv-psf. See jax_metacal::jax_get_gauss_reconv_psf_galsim

    Parameters:
    -----------
    image_size : int
        The dimension of the image (typically a square size).

    Returns:
    --------
    int
        The computed kim size (a power of 2).
    """
    min_size = 4 * image_size
    target_size = 2 ** (int(np.log2(min_size)) + 1)
    return int(target_size)


@jax.jit
def jax_fit_gauss_mom_obs(obs, fwhm=1.2):
    """Fit an DFMdetObservation/DFMdetPSF with Gaussian moments.

    Parameters
    ----------
    obs : DFMDObservation
        The observation to fit.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.

    Returns
    -------
    res: dict-like
        The fit results.
    """
    guassmom_obs = dfmd_obs_to_gaussmom_obs(obs)
    fitter = GaussMom(fwhm)
    return fitter.go(guassmom_obs)


@jax.jit
def jax_fit_gauss_mom_obs_and_psf(obs, fwhm=1.2, psf_res=None):
    """Fit an DFMDObservation with Gaussian moments (JIT-compatible version).

    Parameters
    ----------
    obs : DFMDObservation
        The observation to fit.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.
    psf_res : dict, optional
        The PSF result dict. Default None: the PSF is fit.

    Returns
    -------
    vals : dict
        The fit results as a PyTree (dictionary) with keys:
        - wmom_flags: fit flags
        - wmom_g1: first component of ellipticity
        - wmom_g2: second component of ellipticity
        - wmom_T_ratio: T / psf_T ratio
        - wmom_psf_T: PSF T value
        - wmom_s2n: signal-to-noise ratio
    """
    fitter = GaussMom(fwhm)
    if psf_res is None:
        psf_res = fitter.go(dfmd_obs_to_gaussmom_obs(obs.psf))

    def psf_failed():
        """Return early if PSF fit failed."""
        return {
            "wmom_flags": jnp.array(ngmix.flags.NO_ATTEMPT, dtype=jnp.int32),
            "wmom_g1": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_g2": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_T_ratio": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_psf_T": jnp.array(jnp.nan, dtype=jnp.float64),
            "wmom_s2n": jnp.array(jnp.nan, dtype=jnp.float64),
        }

    def psf_success():
        """Continue with observation fit if PSF fit succeeded."""
        # Fit the observation
        res = fitter.go(dfmd_obs_to_gaussmom_obs(obs))

        def obs_failed():
            """Return partial results if observation fit failed."""
            return {
                "wmom_flags": jnp.array(res.flags, dtype=jnp.int32),
                "wmom_g1": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_g2": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_T_ratio": jnp.array(jnp.nan, dtype=jnp.float64),
                "wmom_psf_T": jnp.array(psf_res.T, dtype=jnp.float64),
                "wmom_s2n": jnp.array(jnp.nan, dtype=jnp.float64),
            }

        def obs_success():
            """Return full results if observation fit succeeded."""
            return {
                "wmom_flags": jnp.array(res.flags, dtype=jnp.int32),
                "wmom_g1": jnp.array(res.e[0], dtype=jnp.float64),
                "wmom_g2": jnp.array(res.e[1], dtype=jnp.float64),
                "wmom_T_ratio": jnp.array(res.T / psf_res.T, dtype=jnp.float64),
                "wmom_psf_T": jnp.array(psf_res.T, dtype=jnp.float64),
                "wmom_s2n": jnp.array(res.s2n, dtype=jnp.float64),
            }

        return jax.lax.cond(res.flags != 0, obs_failed, obs_success)

    return jax.lax.cond(psf_res.flags != 0, psf_failed, psf_success)


@jax.jit
def jax_fit_single_detection(
    mbobs, psf_res, obj_id, obj_x, obj_y, shear_idx, bmask_flag, mfrac_val, fwhm=1.2
):
    """Process a single detection and return results as a PyTree.

    This function is designed to be JIT-compatible and can be used with
    jax.vmap for parallel processing of multiple detections.

    Parameters
    ----------
    mbobs : observation
        The observation for a single detected object.
    psf_res : dict
        The PSF fit result.
    obj_id : int
        Object identifier.
    obj_x : float
        Object x position.
    obj_y : float
        Object y position.
    shear_idx : int
        Index of the shear/metadetection step.
    bmask_flag : int
        Bit mask flag value at detection position.
    mfrac_val : float
        Masked fraction value at detection position.
    fwhm : float, optional
        The FWHM of the Gaussian to use in the fit. Default is 1.2.

    Returns
    -------
    result : dict (PyTree)
        Detection result with scalar JAX array values:
        - id: object identifier
        - x: x position
        - y: y position
        - mdet_step_idx: shear step index
        - bmask_flags: bit mask flags
        - mfrac: masked fraction
        - wmom_flags: weighted moments fit flags
        - wmom_g1: first ellipticity component
        - wmom_g2: second ellipticity component
        - wmom_T_ratio: T / psf_T ratio
        - wmom_psf_T: PSF T value
        - wmom_s2n: signal-to-noise ratio
    """
    # Fit the observation with Gaussian moments
    fres = jax_fit_gauss_mom_obs_and_psf(mbobs, fwhm=fwhm, psf_res=psf_res)

    # Return PyTree with scalar values
    return {
        "id": jnp.array(obj_id, dtype=jnp.int64),
        "x": jnp.array(obj_x, dtype=jnp.float_),
        "y": jnp.array(obj_y, dtype=jnp.float_),
        "mdet_step_idx": jnp.array(shear_idx, dtype=jnp.int32),
        "bmask_flags": jnp.array(bmask_flag, dtype=jnp.int32),
        "mfrac": jnp.array(mfrac_val, dtype=jnp.float_),
        "wmom_flags": fres["wmom_flags"],
        "wmom_g1": fres["wmom_g1"],
        "wmom_g2": fres["wmom_g2"],
        "wmom_T_ratio": fres["wmom_T_ratio"],
        "wmom_psf_T": fres["wmom_psf_T"],
        "wmom_s2n": fres["wmom_s2n"],
    }


def stack_detection_results(results_list):
    """Stack a list of single detection PyTree results into arrays.

    Parameters
    ----------
    results_list : list of dict
        List of detection results from jax_fit_single_detection.

    Returns
    -------
    stacked_results : dict
        Dictionary with same keys as input, but values are 1D arrays
        with shape (n_detections,) instead of scalars.
    """
    if not results_list:
        # Return empty arrays if no detections
        return {
            "id": jnp.array([], dtype=jnp.int64),
            "x": jnp.array([], dtype=jnp.float_),
            "y": jnp.array([], dtype=jnp.float_),
            "mdet_step_idx": jnp.array([], dtype=jnp.int32),
            "bmask_flags": jnp.array([], dtype=jnp.int32),
            "mfrac": jnp.array([], dtype=jnp.float_),
            "wmom_flags": jnp.array([], dtype=jnp.int32),
            "wmom_g1": jnp.array([], dtype=jnp.float64),
            "wmom_g2": jnp.array([], dtype=jnp.float64),
            "wmom_T_ratio": jnp.array([], dtype=jnp.float64),
            "wmom_psf_T": jnp.array([], dtype=jnp.float64),
            "wmom_s2n": jnp.array([], dtype=jnp.float64),
        }

    # Stack all results using jax.tree_map
    return jax.tree_util.tree_map(lambda *arrays: jnp.stack(arrays), *results_list)
