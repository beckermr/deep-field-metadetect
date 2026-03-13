from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import jax_galsim

from deep_field_metadetect.jaxify.observation import (
    BMASK_EDGE,
    DFMdetMultiBandObsList,
    DFMdetObservation,
    DFMdetObsList,
)


@partial(jax.jit, static_argnames=["window_size"])
def local_maxima_filter(
    image: jnp.ndarray,
    noise: Union[jnp.ndarray, float],
    window_size: int = 5,
) -> jnp.ndarray:
    """
    Find local maximas in an image within window_size

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    window_size : int
        Size of the neighborhood for local maximum detection
    noise : jnp.ndarray | float
        Pixelwise noise sigma
        Minimum pixel value of central pixel must > 3-sigma

    Returns:
    --------
    jnp.ndarray
        Binary mask indicating local maxima positions
    """
    noise_array = jnp.broadcast_to(noise, image.shape) if jnp.isscalar(noise) else noise

    # Use max pooling: check if pixel val == max pooled value for peak detection
    max_pooled = jax.lax.reduce_window(
        image, -jnp.inf, jax.lax.max, (window_size, window_size), (1, 1), "SAME"
    )

    threshold_mask = image > jnp.abs(3 * noise_array)
    local_max_mask = (image == max_pooled) & threshold_mask

    return local_max_mask


@partial(jax.jit, static_argnames=["window_size", "max_objects"])
def peak_finder(
    image: jnp.ndarray,
    noise: Union[jnp.ndarray, float],
    window_size: int = 5,
    max_objects: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find peaks in an image above a threshold

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    noise : jnp.ndarray | float
        Pixelwise noise sigma
        Minimum pixel value of central pixel must be > 3-sigma
    window_size : int
        Size of the neighborhood for local maximum detection
    max_objects : int
        Maximum number of objects to detect (to make functions jitable)

    Returns:
    --------
    positions : jnp.ndarray
        Array of peak coordinates (y, x) of shape (max_objects, 2)
        Invalid entries filled with (-1, -1)
    det_flag : jnp.ndarray
        Integer array of shape (max_objects,) where:
        0 = actual detection, 1 = fill value
    """
    local_max_mask = local_maxima_filter(
        image=image,
        noise=noise,
        window_size=window_size,
    )

    positions = jnp.argwhere(local_max_mask, size=max_objects, fill_value=(-1, -1))

    # Create detection flag: 0 for actual detections, 1 for fill values
    # Fill values have (-1, -1) coordinates
    det_flag = jnp.all(positions == -1, axis=1).astype(jnp.int32)

    return positions, det_flag


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid(
    image: jnp.ndarray, peak: Tuple[int, int], window_size: int = 5
) -> Tuple[float, float, bool]:
    """
    Refine peak position of single object using intensity-weighted centroid.

    Skips refinement if too close to the border or if proposed shift > 1 pixel.
    The proposed shift can be > 1 if the noise theshold is not properly set
    and detections are in noisy regions.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    peak: jnp.ndarray
        Initial peak position
    window_size : int
        Size of window around peak for centroid calculation
        if window crosses image boundary, optimization is skipped.

    Returns:
    --------
    jnp.ndarray
        Refined peak coordinates (refined_y, refined_x) : float
        Note: original coordinates are returned if refinement is skipped
    refinement_flag : bool
        True (1) if refinement was skipped (near border or large shift),
        False (0) if refinement was applied
    """
    half_window = window_size // 2
    height, width = image.shape

    # If near border, return original coordinates
    near_border = (
        (peak[0] < half_window)
        | (peak[0] >= height - half_window)
        | (peak[1] < half_window)
        | (peak[1] >= width - half_window)
    )

    def border_case():
        return jnp.array([peak[0], peak[1]], dtype=jnp.float_), True

    def normal_case():
        window = jax.lax.dynamic_slice(
            image,
            (peak[0] - half_window, peak[1] - half_window),
            (window_size, window_size),
        )

        y_start = -half_window
        x_start = -half_window
        y_coords = jnp.arange(y_start, y_start + window_size)
        x_coords = jnp.arange(x_start, x_start + window_size)
        y_grid, x_grid = jnp.meshgrid(y_coords, x_coords, indexing="ij")

        total_intensity = jnp.sum(window)

        y_shift = jnp.sum((y_grid) * window) / total_intensity
        x_shift = jnp.sum((x_grid) * window) / total_intensity

        # Check if proposed shift is > 1 pixel
        shift_magnitude = jnp.sqrt(y_shift**2 + x_shift**2)
        shift_too_large = shift_magnitude > 1.0

        def apply_shift():
            refined_y = y_shift + peak[0]
            refined_x = x_shift + peak[1]
            # Clip coordinates to be within valid image bounds
            refined_y = jnp.clip(refined_y, 0.0, height - 1.0)
            refined_x = jnp.clip(refined_x, 0.0, width - 1.0)
            return jnp.array([refined_y, refined_x], dtype=jnp.float_), False

        def skip_shift():
            return jnp.array([peak[0], peak[1]], dtype=jnp.float_), True

        return jax.lax.cond(shift_too_large, skip_shift, apply_shift)

    result, flag = jax.lax.cond(near_border, border_case, normal_case)
    return result, flag


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid_in_cell(
    image: jnp.ndarray,
    peak_positions: jnp.ndarray,
    window_size: int = 5,
):
    """
    vmapped version of refine_centroid

    Returns:
    --------
    refined_positions : jnp.ndarray
        Array of refined coordinates
    refinement_flags : jnp.ndarray
        Array of flags (1 if refinement skipped, 0 if applied)
    """
    return jax.vmap(refine_centroid, in_axes=(None, 0, None))(
        image, peak_positions, window_size
    )


@partial(jax.jit, static_argnames=["window_size", "refine_centroids", "max_objects"])
def detect_galaxies(
    image: jnp.ndarray,
    noise: Union[jnp.ndarray, float],
    window_size: int = 5,
    refine_centroids: bool = True,
    max_objects: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Complete galaxy center detection pipeline with JIT compilation support.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    noise : jnp.ndarray | float
        Pixelwise noise sigma
        Minimum pixel value of central pixel must be > 3-sigma
    window_size : int
        Minimum distance between detected peaks
    refine_centroids : bool
        Whether to refine peak positions using centroid calculation
    max_objects : int
        Maximum number of objects to detect (for fixed array sizes)

    Returns:
    --------
    peak_positions : jnp.ndarray
        Array of detected galaxy centers (y, x) of shape (max_objects, 2).
        Returns only the integral pixel location.
        Invalid entries filled with -1
    refined_positions : jnp.ndarray
        Array of detected galaxy centers (y, x) after centroid refinement.
        Returns the refined floating point values of the center.
    refinement_flags : jnp.ndarray
        Array indicating which objects had refinement skipped (shape max_objects,)
        1 = refinement skipped (near border or large shift), 0 = refinement applied
    det_flag : jnp.ndarray
        Integer array of shape (max_objects,) where:
        0 = actual detection, 1 = fill value
    """
    peak_positions, det_flag = peak_finder(
        image=image,
        noise=noise,
        window_size=window_size,
        max_objects=max_objects,
    )

    if not refine_centroids:
        refinement_flags = jnp.zeros(max_objects, dtype=bool)
        return peak_positions, peak_positions.astype(float), refinement_flags, det_flag

    refined_positions, refinement_flags = refine_centroid_in_cell(
        image, peak_positions, window_size=5
    )  # Using only a single iteration for now.
    # Multiple iter not tested, but can lead to instability for blended objects

    return peak_positions, refined_positions, refinement_flags, det_flag


@partial(jax.jit, static_argnames=["max_iterations"])
def watershed_segmentation(
    inverted_image: jnp.ndarray,
    noise: Union[jnp.ndarray, float],
    markers: jnp.ndarray,
    mask: jnp.ndarray = None,
    max_iterations: int = 30,
) -> jnp.ndarray:
    """
    JAX implementation of watershed segmentation algorithm.

    Parameters:
    -----------
    inverted_image : jnp.ndarray
        2D input image with inverted intensity
    noise : jnp.ndarray | float
        Pixelwise noise sigma
        flooding continues to unmarked neightboring pixel within a limit of sigma
    markers : jnp.ndarray
        2D array of initial markers (labeled regions) where positive values
        indicate different watershed basins and 0 indicates unmarked pixels
    mask : jnp.ndarray, optional
        Binary mask indicating valid pixels for segmentation.
        Pixels with non-zero masked values are masked.
    max_iterations : int
        Maximum number of iterations for the flooding process

    Returns:
    --------
    labels : jnp.ndarray
        2D segmentation map with same shape as input image
    """
    noise_array = (
        jnp.broadcast_to(noise, inverted_image.shape) if jnp.isscalar(noise) else noise
    )  # pythonic if else works: JIT-compilation will be triggered if shape changes
    if mask is None:
        mask = jnp.zeros_like(inverted_image, dtype=bool)

    labels = markers.copy()
    height, width = inverted_image.shape

    def watershed_step(labels_prev):
        """Single iteration of watershed flooding"""
        labels_new = labels_prev.copy()

        def update_pixel(i, j):
            # Skip if masked out
            current_label = labels_prev[i, j]
            is_valid = ~mask[i, j]

            def check_neighbors():
                # Check 4-connected neighbors
                neighbor_coords = jnp.array(
                    [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]
                )

                in_bounds = (
                    (neighbor_coords[:, 0] >= 0)
                    & (neighbor_coords[:, 0] < height)
                    & (neighbor_coords[:, 1] >= 0)
                    & (neighbor_coords[:, 1] < width)
                )

                neighbor_labels = labels_prev[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]
                neighbor_values = inverted_image[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]

                # Mask for valid (labeled and in-bounds) neighbors
                valid_mask = in_bounds & (neighbor_labels > 0)

                has_valid = jnp.any(valid_mask)

                def process_valid_neighbors():
                    # Use large value for invalid neighbors in argmin
                    masked_values = jnp.where(valid_mask, neighbor_values, jnp.inf)
                    min_idx = jnp.argmin(masked_values)

                    # Check if current pixel should be flooded
                    current_value = inverted_image[i, j]
                    current_noise = noise_array[i, j]
                    min_neighbor_value = neighbor_values[min_idx]

                    # Flood if current value is >= minimum neighbor value
                    def should_flood():
                        """Decides when to flood a pixel based on if it is marked"""

                        def unmarked_pixel():
                            return (current_value + current_noise) >= min_neighbor_value

                        def marked_pixel():
                            return (current_value) >= min_neighbor_value

                        is_marked = current_label != 0

                        return jax.lax.cond(
                            is_marked,
                            marked_pixel,
                            unmarked_pixel,
                        )

                    to_flood = should_flood()

                    # leave current value if update is not required
                    return jax.lax.cond(
                        to_flood,
                        lambda: neighbor_labels[min_idx],
                        lambda: current_label,
                    )

                # If no valid neighbors, leave current value else process
                return jax.lax.cond(
                    has_valid, process_valid_neighbors, lambda: current_label
                )

            # If pixel is not maked, check for neightbors
            new_label = jax.lax.cond(is_valid, check_neighbors, lambda: current_label)

            return new_label

        # Vectorized update over all pixels
        i_coords, j_coords = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width), indexing="ij"
        )

        labels_new = jax.vmap(jax.vmap(update_pixel, in_axes=(0, 0)), in_axes=(0, 0))(
            i_coords, j_coords
        )

        return labels_new

    # Iterative flooding using scan
    def scan_fn(labels_current, _):
        labels_next = watershed_step(labels_current)
        return labels_next, None

    final_labels, _ = jax.lax.scan(scan_fn, labels, jnp.arange(max_iterations))

    return final_labels


@partial(jax.jit, static_argnames=["max_iterations"])
def watershed_from_peaks(
    image: jnp.ndarray,
    noise: Union[jnp.ndarray, float],
    peaks: jnp.ndarray,
    mask: jnp.ndarray = None,
    max_iterations: int = 30,
) -> jnp.ndarray:
    """
    Perform watershed segmentation using detected peaks as markers.

    Parameters:
    -----------
    image : jnp.ndarray
        2D input image
    noise : jnp.ndarray | float
        Pixelwise noise sigma
        flooding continues to the neightboring pixel within a limit of sigma
    peaks : jnp.ndarray
        Array of peak positions (y, x) of shape (n_peaks, 2)
    mask : jnp.ndarray
        Array of masked pixels.
        Pixels with non-zero masked values are masked.
    max_iterations : int
        Maximum iterations for watershed algorithm

    Returns:
    --------
    watershed_labels : jnp.ndarray
        2D segmentation map from watershed algorithm
    """
    height, width = image.shape

    inverted_image = -image  # Invert so peaks become valleys

    markers = jnp.zeros((height, width), dtype=jnp.int32)

    # Place markers at peak positions
    def place_marker(markers, i, peak_pos):
        y, x = peak_pos.astype(jnp.int32)
        is_valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)

        marker_value = jax.lax.cond(is_valid, lambda: i + 1, lambda: 0)  # Label from 1

        return jax.lax.cond(
            is_valid, lambda: markers.at[y, x].set(marker_value), lambda: markers
        )

    # Sequential marker placement
    def scan_fn(markers_current, i_peak):
        i, peak = i_peak
        return place_marker(markers_current, i, peak), None

    markers, _ = jax.lax.scan(scan_fn, markers, (jnp.arange(peaks.shape[0]), peaks))

    # Apply watershed algorithm
    watershed_labels = watershed_segmentation(
        inverted_image,
        noise=noise,
        markers=markers,
        max_iterations=max_iterations,
        mask=mask,
    )

    return watershed_labels


@partial(jax.jit, static_argnames=["box_size"])
def _get_subobs_jax(
    obs: DFMdetObservation,
    x: float,
    y: float,
    box_size: int,
) -> DFMdetObservation:
    """Create a sub-observation around a given position using dynamic slicing.

    This function assumes the observation has already been padded appropriately
    so that boundary cases never occur. Uses jax.lax.dynamic_slice for efficient,
    JIT-compatible extraction.

    Parameters
    ----------
    obs : DFMdetObservation
        The source observation (must be pre-padded).
    x, y : float
        The object position coordinates.
        Note: should already include padding offset if obs is padded.
    box_size : int
        Size of the target sub-box.

    Returns
    -------
    DFMdetObservation
        Sub-observation around the given position.
    """
    half_box = box_size // 2

    # Calculate top-left corner of the box
    # Coordinates should already be offset for padding by the caller
    ix = jnp.int32(x)
    iy = jnp.int32(y)
    start_x = ix - half_box + 1
    start_y = iy - half_box + 1

    # Extract sub-images using dynamic_slice (no boundary checks needed!)
    sub_image = jax.lax.dynamic_slice(
        obs.image, (start_y, start_x), (box_size, box_size)
    )
    sub_weight = jax.lax.dynamic_slice(
        obs.weight, (start_y, start_x), (box_size, box_size)
    )
    sub_bmask = jax.lax.dynamic_slice(
        obs.bmask, (start_y, start_x), (box_size, box_size)
    )
    sub_ormask = jax.lax.dynamic_slice(
        obs.ormask, (start_y, start_x), (box_size, box_size)
    )
    sub_noise = jax.lax.dynamic_slice(
        obs.noise, (start_y, start_x), (box_size, box_size)
    )
    sub_mfrac = jax.lax.dynamic_slice(
        obs.mfrac, (start_y, start_x), (box_size, box_size)
    )

    # Update mfrac for edge pixels
    msk = (sub_bmask & BMASK_EDGE) != 0
    sub_mfrac = jnp.where(msk, 1.0, sub_mfrac)

    # Create new WCS with adjusted origin
    # The origin should be at the object position within the sub-image
    new_wcs = jax_galsim.wcs.AffineTransform(
        dudx=obs.wcs.dudx,
        dudy=obs.wcs.dudy,
        dvdx=obs.wcs.dvdx,
        dvdy=obs.wcs.dvdy,
        origin=jax_galsim.PositionD(
            x=(x - start_x) + 1,
            y=(y - start_y) + 1,
        ),
    )

    return DFMdetObservation(
        image=sub_image,
        weight=sub_weight,
        bmask=sub_bmask,
        ormask=sub_ormask,
        noise=sub_noise,
        mfrac=sub_mfrac,
        wcs=new_wcs,
        psf=obs.psf,
        meta=obs.meta,
        store_pixels=obs.store_pixels,
        ignore_zero_weight=obs.ignore_zero_weight,
    )


def jax_generate_subobs_for_detections(
    obs,
    xs,
    ys,
    box_size=48,
    ids=None,
):
    """Generate sub-observations around given positions for a single observation.

    This is the non-JIT compatible generator version.

    Parameters
    ----------
    obs : DFMdetObservation
        The observation to generate sub-observations from.
        Should be pre-padded if extracting near boundaries.
    xs : array-like
        The x positions of the objects.
        Note: should include padding offset if obs is padded.
    ys : array-like
        The y positions of the objects.
        Note: should include padding offset if obs is padded.
    box_size : int, optional
        The size of the sub-boxes around the objects. Default is 48.
    ids : array-like, optional
        The IDs of the objects. If None, the IDs are the indices of the positions.

    Returns
    -------
    generator
        A generator that yields a tuple of the object information and the
        sub-observation.
    """
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Create sub-observation for this detection
        sub_obs = _get_subobs_jax(obs, x, y, box_size)

        yield (
            {
                "id": ids[i] if ids is not None else i,
                "x": x,
                "y": y,
            },
            sub_obs,
        )


def jax_generate_mbobs_for_detections(
    mbobs,
    xs,
    ys,
    box_size=48,
    ids=None,
):
    """Generate sub-mbobs around given positions for JAX multi-band observations.

    This routine is a generator and so should be used like thus:

    This is the non-JIT (because of yield) compatible generator version and is the
    JAX equivalent of ``generate_mbobs_for_detections`` in ``detect.py``.

    Parameters
    ----------
    mbobs : DFMdetMultiBandObsList
        The multi-band observations to generate sub-mbobs from.
        Must be already converted using jax_get_mb_obs() with appropriate padding.
    xs : array-like
        The x positions of the objects.
        Note: should include padding offset if mbobs is padded.
    ys : array-like
        The y positions of the objects.
        Note: should include padding offset if mbobs is padded.
    box_size : int, optional
        The size of the sub-boxes around the objects. Default is 48.
    ids : array-like, optional
        The IDs of the objects. If None, the IDs are the indices of the positions.

    Returns
    -------
    generator
        A generator that yields a tuple of the object information and the
        sub-mbobs (DFMdetMultiBandObsList).
    """
    for i, (x, y) in enumerate(zip(xs, ys)):
        sub_obs_lists = []
        for obslist in mbobs:
            sub_obs_list = DFMdetObsList(
                [_get_subobs_jax(obs, x, y, box_size) for obs in obslist]
            )
            sub_obs_lists.append(sub_obs_list)

        yield (
            {
                "id": ids[i] if ids is not None else i,
                "x": x,
                "y": y,
            },
            DFMdetMultiBandObsList(sub_obs_lists),
        )
