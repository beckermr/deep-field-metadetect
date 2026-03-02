from functools import partial
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import jax_galsim

from deep_field_metadetect.jaxify.observation import (
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
    """
    local_max_mask = local_maxima_filter(
        image=image,
        noise=noise,
        window_size=window_size,
    )

    positions = jnp.argwhere(local_max_mask, size=max_objects, fill_value=(-1, -1))

    return positions


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid(
    image: jnp.ndarray, peak: Tuple[int, int], window_size: int = 5
) -> Tuple[float, float, bool]:
    """
    Refine peak position of single object using intensity-weighted centroid.
    Skips refinement for objects too close to the border.
    Returns whether object was near border for warning purposes.

    Parameters:
    -----------
    image : jnp.ndarray
        2D Input galaxy field
    peak: jnp.ndarray
        Initial peak position
    window_size : int
        Size of window around peak for centroid calculation
        if window crosses image boudary, optimization is skipped.

    Returns:
    --------
    jnp.ndarray
        Refined peak coordinates (refined_y, refined_x) : float
        Note: original coordinatesare returned if near border
    near_border : bool
        True if object was near border and refinement was skipped
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
        return jnp.array([peak[0], peak[1]], dtype=jnp.float_)

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

        refined_y = y_shift + peak[0]
        refined_x = x_shift + peak[1]

        return jnp.array([refined_y, refined_x], dtype=jnp.float_)

    result = jax.lax.cond(near_border, border_case, normal_case)

    return jnp.array([result[0], result[1]]), near_border


@partial(jax.jit, static_argnames=["window_size"])
def refine_centroid_in_cell(
    image: jnp.ndarray,
    peak_positions: jnp.ndarray,
    window_size: int = 5,
):
    """
    vmapped version of refine_centroid
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    border_flags : jnp.ndarray
        Array indicating which objects were near border (shape max_objects,)
    """
    peak_positions = peak_finder(
        image=image,
        noise=noise,
        window_size=window_size,
        max_objects=max_objects,
    )

    if not refine_centroids:
        border_flags = jnp.zeros(max_objects, dtype=bool)
        return peak_positions, peak_positions.astype(float), border_flags

    refined_positions, border_flags = refine_centroid_in_cell(
        image, peak_positions, window_size=5
    )  # Using only a single iteration for now.
    # Multiple iter not tested, but can lead to unstability for blended objects

    return peak_positions, refined_positions, border_flags


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


# Constants for JAX detection functions
BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    "image": 0.0,
    "weight": 0.0,
    "seg": 0,
    "bmask": BMASK_EDGE,
    "noise": 0.0,
    "mfrac": 0.0,
}


def _get_subboxes(
    start: int, end: int, box_size: int, max_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Calculate subboxes for extracting sub-images with proper boundary handling.

    Parameters
    ----------
    start : int
        Start coordinate of the box.
    end : int
        End coordinate of the box.
    box_size : int
        Size of the target box.
    max_size : int
        Maximum size of the source image.

    Returns
    -------
    tuple
        Tuple of (orig_box, sub_box) where each is (start, end) coordinates.
    """
    assert end - start == box_size
    orig_box = [start, end]
    sub_box = [0, box_size]

    if start < 0:
        sub_box[0] = -start
        orig_box[0] = 0
    if end > max_size:
        sub_box[1] = box_size - (end - max_size)
        orig_box[1] = max_size

    return (orig_box[0], orig_box[1]), (sub_box[0], sub_box[1])


def _get_subobs_jax(
    obs,
    x: float,
    y: float,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    box_size: int,
):
    """Create a sub-observation around a given position using JAX arrays.

    Parameters
    ----------
    obs : DFMdetObservation
        The source observation.
    x, y : float
        The object position coordinates.
    start_x, start_y : int
        Start coordinates for the sub-box.
    end_x, end_y : int
        End coordinates for the sub-box.
    box_size : int
        Size of the target sub-box.

    Returns
    -------
    DFMdetObservation
        Sub-observation around the given position.
    """

    max_y, max_x = obs.image.shape
    orig_x_box, sub_x_box = _get_subboxes(start_x, end_x, box_size, max_x)
    orig_y_box, sub_y_box = _get_subboxes(start_y, end_y, box_size, max_y)

    # Create new WCS with adjusted origin
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

    kwargs = {"wcs": new_wcs}

    if obs.has_psf():
        kwargs["psf"] = obs.psf

    for key in ["image", "bmask", "noise", "mfrac", "weight"]:
        subim = None

        if key == "image":
            subim = jnp.full(
                (box_size, box_size), DEFAULT_IMAGE_VALUES[key], dtype=obs.image.dtype
            )
            subim = subim.at[
                sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
            ].set(
                obs.image[orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]]
            )
        elif key == "bmask":
            if obs.has_bmask():
                subim = jnp.full(
                    (box_size, box_size),
                    DEFAULT_IMAGE_VALUES[key],
                    dtype=obs.bmask.dtype,
                )
                subim = subim.at[
                    sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
                ].set(
                    obs.bmask[
                        orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]
                    ]
                )
            else:
                subim = jnp.full(
                    (box_size, box_size), DEFAULT_IMAGE_VALUES[key], dtype=jnp.int32
                )
                subim = subim.at[
                    sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
                ].set(0)
        elif key == "noise" and obs.has_noise():
            subim = jnp.full(
                (box_size, box_size), DEFAULT_IMAGE_VALUES[key], dtype=obs.noise.dtype
            )
            subim = subim.at[
                sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
            ].set(
                obs.noise[orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]]
            )
        elif key == "weight":
            subim = jnp.full(
                (box_size, box_size), DEFAULT_IMAGE_VALUES[key], dtype=obs.weight.dtype
            )
            subim = subim.at[
                sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
            ].set(
                obs.weight[orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]]
            )
        elif key == "mfrac" and obs.has_mfrac():
            subim = jnp.full(
                (box_size, box_size), DEFAULT_IMAGE_VALUES[key], dtype=obs.mfrac.dtype
            )
            subim = subim.at[
                sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]
            ].set(
                obs.mfrac[orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]]
            )

        if subim is not None:
            kwargs[key] = subim

    if "mfrac" in kwargs and "bmask" in kwargs:
        msk = kwargs["bmask"] & BMASK_EDGE != 0
        kwargs["mfrac"] = jnp.where(msk, 1.0, kwargs["mfrac"])

    return DFMdetObservation(**kwargs)


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
    xs : array-like
        The x positions of the objects.
    ys : array-like
        The y positions of the objects.
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
    half_box_size = box_size // 2

    for i, (x, y) in enumerate(zip(xs, ys)):
        ix = int(x)
        iy = int(y)
        start_x = ix - half_box_size + 1
        start_y = iy - half_box_size + 1
        end_x = ix + half_box_size + 1  # plus one for slices
        end_y = iy + half_box_size + 1

        # Create sub-observation for this detection
        sub_obs = _get_subobs_jax(obs, x, y, start_x, start_y, end_x, end_y, box_size)

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
        Must be already converted using jax_get_mb_obs().
    xs : array-like
        The x positions of the objects.
    ys : array-like
        The y positions of the objects.
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
    half_box_size = box_size // 2

    for i, (x, y) in enumerate(zip(xs, ys)):
        ix = int(x)
        iy = int(y)
        start_x = ix - half_box_size + 1
        start_y = iy - half_box_size + 1
        end_x = ix + half_box_size + 1  # plus one for slices
        end_y = iy + half_box_size + 1

        sub_obs_lists = []
        for obslist in mbobs:
            sub_obs_list = DFMdetObsList(
                [
                    _get_subobs_jax(obs, x, y, start_x, start_y, end_x, end_y, box_size)
                    for obs in obslist
                ]
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
