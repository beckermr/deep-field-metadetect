import jax.numpy as jnp
import numpy as np

from deep_field_metadetect.jaxify.jax_detection import (
    detect_galaxies,
    local_maxima_filter,
    peak_finder,
    refine_centroid,
    watershed_from_peaks,
    watershed_segmentation,
)


def create_gaussian_blob(shape, center, sigma=1.0, amplitude=1.0):
    """
    Create a 2D Gaussian blob for testing.

    Parameters:
    -----------
    shape : tuple
        Shape of the output array (height, width)
    center : tuple
        Center position (y, x) of the Gaussian
    sigma : float
        Standard deviation of the Gaussian
    amplitude : float
        Peak amplitude of the Gaussian

    Returns:
    --------
    jnp.ndarray
        2D array containing the Gaussian blob
    """
    y, x = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing="ij")
    cy, cx = center

    gaussian = amplitude * jnp.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))
    return gaussian


def create_multiple_gaussian_blobs(shape, centers, sigmas=None, amplitudes=None):
    """
    Create multiple Gaussian blobs in a single image.

    Parameters:
    -----------
    shape : tuple
        Shape of the output array (height, width)
    centers : list of tuples
        List of center positions [(y1, x1), (y2, x2), ...]
    sigmas : list of floats or None
        Standard deviations for each blob. If None, uses 1.0 for all
    amplitudes : list of floats or None
        Amplitudes for each blob. If None, uses 1.0 for all

    Returns:
    --------
    jnp.ndarray
        2D array containing all Gaussian blobs
    """
    if sigmas is None:
        sigmas = [1.0] * len(centers)
    if amplitudes is None:
        amplitudes = [1.0] * len(centers)

    image = jnp.zeros(shape)
    for center, sigma, amplitude in zip(centers, sigmas, amplitudes):
        blob = create_gaussian_blob(shape, center, sigma, amplitude)
        image = image + blob

    return image


# -------------------
# Test peak detection
# -------------------


def single_gaussian():
    """Test detection of multiple well-separated Gaussian peaks."""
    centers = [(5, 5)]
    amplitudes = [1.0]
    max_objects = 10
    image = create_multiple_gaussian_blobs(
        (10, 10), centers, sigmas=[1.0], amplitudes=amplitudes
    )

    noise = 0.0
    peak = peak_finder(image, noise=noise, max_objects=max_objects)

    assert len(peak) == max_objects
    assert (peak[0][0] == centers[0][0]) & (peak[0][1] == centers[0][1])


def test_multiple_separated_gaussians():
    """Test detection of multiple well-separated Gaussian peaks."""
    centers = [(2, 2), (2, 7), (7, 2), (7, 7)]
    amplitudes = [1.0, 1.5, 2.0, 0.8]
    image = create_multiple_gaussian_blobs(
        (10, 10), centers, sigmas=[1.0] * 4, amplitudes=amplitudes
    )

    noise = 0.0
    result = local_maxima_filter(image, noise=noise, window_size=3)

    # All centers should be detected as peaks
    for center in centers:
        assert result[center[0], center[1]]


def test_threshold_filtering_gaussians():
    """Test that Gaussian peaks below threshold are filtered out."""
    centers = [(3, 3), (3, 9)]
    amplitudes = [0.5, 2.0]  # First below threshold, second above
    image = create_multiple_gaussian_blobs((12, 12), centers, amplitudes=amplitudes)

    noise = 0.5
    result = local_maxima_filter(image, noise=noise, window_size=3)

    # Only the high amplitude peak should be detected
    assert not result[3, 3]  # Below threshold
    assert result[3, 9]  # Above threshold


def test_overlapping_gaussians():
    """Test behavior with overlapping Gaussian blobs."""
    # Two Gaussians close together
    centers = [(4, 4), (4, 6)]
    image = create_multiple_gaussian_blobs(
        (9, 9), centers, sigmas=[1.5, 1.5], amplitudes=[1.0, 1.0]
    )

    noise = 0.0
    result = local_maxima_filter(image, noise=noise, window_size=3)

    # Depending on overlap, may detect one or both peaks
    # At minimum, should detect at least one peak in the region
    peak_region = result[3:6, 3:7]
    assert jnp.any(peak_region)


def test_edge_case_detection():
    """Test detection of edge cases and boundary conditions."""
    # Single pixel "galaxy"
    image = jnp.zeros((7, 7))
    image = image.at[5, 5].set(5.0)

    noise = 0.0
    peaks, refined, border_flags, det_flag = detect_galaxies(
        image=image,
        noise=noise,
        window_size=3,
        refine_centroids=True,
        max_objects=5,
    )

    print(peaks)
    valid_peaks = peaks[peaks[:, 0] > 0]

    assert len(valid_peaks) == 1
    assert jnp.array_equal(valid_peaks[0], jnp.array([5, 5]))


# ------------------------
# Test Centriod Refinement
# ------------------------


def test_gaussian_centroid_refinement():
    """Test centroid refinement on slightly off-center Gaussian."""
    # Create Gaussian slightly off-grid
    true_center = (4.3, 4.7)
    image = create_gaussian_blob((9, 9), true_center, sigma=1.5, amplitude=2.0)

    # Start refinement from nearest grid point
    initial_peak = (4, 5)
    refined_peak, near_border = refine_centroid(image, initial_peak, window_size=5)

    # Refined position should be closer to true center
    initial_distance = np.sqrt(
        (initial_peak[0] - true_center[0]) ** 2
        + (initial_peak[1] - true_center[1]) ** 2
    )
    refined_distance = np.sqrt(
        (refined_peak[0] - true_center[0]) ** 2
        + (refined_peak[1] - true_center[1]) ** 2
    )

    assert refined_distance < initial_distance
    assert not near_border


def test_near_border():
    """Test near border case."""
    # Create two overlapping Gaussians to make asymmetric peak
    centers = [(3, 3)]
    amplitudes = [1.0]
    image = create_multiple_gaussian_blobs(
        (5, 5), centers, sigmas=[1.0], amplitudes=amplitudes
    )

    refined_pos, near_border = refine_centroid(image, (4, 4), window_size=5)

    assert (refined_pos[0] == 4) & (refined_pos[1] == 4)  # refined is same as input
    assert near_border


# -----------------------------
# Test galaxy dection in fields
# -----------------------------


def test_complete_gaussian_detection():
    """Test complete detection pipeline on Gaussian galaxies."""
    centers = [(5, 5), (5, 15), (15, 5), (15, 15)]
    amplitudes = [2.0, 1.5, 1.8, 1.2]
    sigmas = [1.5, 1.2, 1.3, 1]

    image = create_multiple_gaussian_blobs(
        (21, 21), centers, sigmas=sigmas, amplitudes=amplitudes
    )

    noise = 0.0
    peaks, refined, _, _ = detect_galaxies(
        image, noise=noise, window_size=5, refine_centroids=True, max_objects=10
    )

    valid_peaks = peaks[peaks[:, 0] > 0]
    valid_refined = refined[peaks[:, 0] > 0]

    # Should detect all 4 galaxies
    assert len(valid_peaks) == 4

    assert np.all(valid_peaks == jnp.array(centers))

    # Refinement should improve positions for off-grid centers
    for i in range(len(valid_refined)):
        # Refined positions should be reasonable
        assert np.abs(np.asarray(centers)[i, 0] - valid_refined[i, 0]) < 0.5
        assert np.abs(np.asarray(centers)[i, 1] - valid_refined[i, 1]) < 0.5


def test_detection_with_noise():
    """Test detection robustness with added noise."""
    np.random.seed(42)
    # Create clean Gaussian
    peak_location = (6, 6)
    image_clean = create_gaussian_blob((15, 15), (6, 6), sigma=1.0, amplitude=2.0)

    # Add noise
    noise = jnp.array(np.random.normal(0, 0.2, image_clean.shape))
    image_noisy = image_clean + noise

    noise = 0.2
    _, refined, _, _ = detect_galaxies(
        image_noisy, noise=noise, window_size=5, refine_centroids=True, max_objects=5
    )

    valid_peaks = refined[refined[:, 0] > 0]

    # Should still detect the main peak despite noise
    assert len(valid_peaks) >= 1

    # Main peak should be near expected position
    main_peak = valid_peaks[0]
    distance_to_true = np.sqrt(
        (main_peak[0] - peak_location[0]) ** 2 + (main_peak[1] - peak_location[1]) ** 2
    )
    assert distance_to_true < 0.5


def test_faint_galaxy_detection():
    """Test detection of faint galaxies."""
    centers = [(8, 6), (8, 12)]
    amplitudes = [2.0, 0.8]  # One bright, one faint

    image = create_multiple_gaussian_blobs(
        (17, 17), centers, sigmas=[1.5, 1.5], amplitudes=amplitudes
    )

    # Test with threshold that should catch both
    noise = 0.2
    peaks_low, _, _, _ = detect_galaxies(image, noise=noise, max_objects=5)
    valid_low = peaks_low[peaks_low[:, 0] > 0]

    # Test with threshold that should only catch bright one
    noise = 0.3
    peaks_high, _, _, _ = detect_galaxies(image, noise=noise, max_objects=5)
    valid_high = peaks_high[peaks_high[:, 0] > 0]

    assert len(valid_low) == 2
    assert len(valid_high) == 1


# -------------------------
# Test watershed algorithm
# -------------------------


def test_watershed_edge_cases():
    """Test watershed algorithm edge cases."""
    uniform_image = jnp.ones((3, 3)) * 2.0
    uniform_markers = jnp.zeros((3, 3), dtype=int)
    uniform_markers = uniform_markers.at[1, 1].set(1)

    noise = 0.0
    result = watershed_segmentation(
        uniform_image, noise, uniform_markers, max_iterations=5
    )
    # All pixels should eventually be labeled due to uniform flooding
    assert jnp.all(result == 1)


def test_watershed_with_mask():
    """Test watershed segmentation with a mask."""
    image = jnp.ones((5, 5)) * 2.0
    image = image.at[1:4, 1:4].set(1.0)  # Lower values in center

    # Create mask that excludes border pixels
    mask = jnp.ones((5, 5), dtype=bool)
    mask = mask.at[1:4, 1:4].set(False)

    markers = jnp.zeros((5, 5), dtype=int)
    markers = markers.at[2, 2].set(1)

    noise = 0.0
    result = watershed_segmentation(image, noise, markers, mask=mask, max_iterations=5)

    # Only pixels within mask should be labeled
    assert jnp.all(result[mask] == 0)
    assert jnp.all(result[~mask] != 0)


def test_watershed_from_peaks_with_invalid():
    """Test watershed_from_peaks with invalid peak positions."""
    image = jnp.ones((5, 5))
    image = image.at[2, 2].set(3.0)

    # Include some invalid peaks (marked with -999)
    peaks = jnp.array(
        [
            [2, 2],  # Valid peak
            [1, 1],  # Valid peak
            [-999, -999],  # Invalid peak
            [-999, -999],  # Invalid peak
        ]
    )

    noise = 0.0
    result = watershed_from_peaks(image, noise, peaks, max_iterations=10)

    unique_labels = jnp.unique(result)
    unique_labels = unique_labels[unique_labels > 0]

    # Should have exactly 2 regions (for the 2 valid peaks)
    assert len(unique_labels) == 2


def test_noise_rejitting():
    """Test that noise array is used directly when provided."""
    image = jnp.ones((8, 12))
    image = image.at[4, 6].set(3.0)  # Different noise at one pixel

    # Create noise array with same shape as image
    noise_array = jnp.zeros_like(image) + 0.3

    result_array = local_maxima_filter(image, noise=noise_array, window_size=3)
    result_float = local_maxima_filter(image, noise=0.3, window_size=3)

    assert jnp.all(result_array == result_float)


def test_det_flag():
    """Test that det_flag correctly identifies actual detections vs fill values."""
    # Create image with 3 Gaussian blobs
    centers = [(5, 5), (5, 15), (15, 10)]
    amplitudes = [2.0, 1.5, 1.8]
    image = create_multiple_gaussian_blobs(
        (21, 21), centers, sigmas=[1.5] * 3, amplitudes=amplitudes
    )

    noise = 0.0
    max_objects = 10  # More than actual detections

    # Test peak_finder
    positions, det_flag = peak_finder(
        image, noise=noise, window_size=5, max_objects=max_objects
    )

    # First 3 should be actual detections (det_flag=0)
    assert jnp.sum(det_flag == 0) == 3, "Should have 3 actual detections"
    assert jnp.sum(det_flag == 1) == 7, "Should have 7 fill values"

    # Verify that actual detections have valid positions
    actual_detections = positions[det_flag == 0]
    for pos in actual_detections:
        assert pos[0] >= 0 and pos[1] >= 0, (
            "Actual detections should have valid positions"
        )

    # Verify that fill values have (-1, -1) positions
    fill_positions = positions[det_flag == 1]
    for pos in fill_positions:
        assert jnp.array_equal(pos, jnp.array([-1, -1])), (
            "Fill values should be (-1, -1)"
        )

    # Test detect_galaxies
    peaks, refined, border_flags, det_flag_full = detect_galaxies(
        image,
        noise=noise,
        window_size=5,
        refine_centroids=True,
        max_objects=max_objects,
    )

    # Should have same det_flag pattern
    assert jnp.sum(det_flag_full == 0) == 3, (
        "detect_galaxies should have 3 actual detections"
    )
    assert jnp.sum(det_flag_full == 1) == 7, "detect_galaxies should have 7 fill values"

    # Verify consistency between det_flag and positions
    assert jnp.all(det_flag == det_flag_full), "det_flag should be consistent"


def test_det_flag_no_detections():
    """Test det_flag when no objects are detected."""
    # Create empty image (all below threshold)
    image = jnp.ones((10, 10)) * 0.1
    noise = 1.0  # High threshold
    max_objects = 5

    positions, det_flag = peak_finder(
        image, noise=noise, window_size=5, max_objects=max_objects
    )

    # All should be fill values (det_flag=1)
    assert jnp.all(det_flag == 1), (
        "All entries should be fill values when no detections"
    )
    assert jnp.all(positions == -1), (
        "All positions should be (-1, -1) when no detections"
    )


def test_det_flag_max_detections():
    """Test det_flag when number of detections equals max_objects."""
    # Create image with exactly max_objects detections
    max_objects = 4
    centers = [(2, 2), (2, 7), (7, 2), (7, 7)]
    amplitudes = [2.0] * 4
    image = create_multiple_gaussian_blobs(
        (10, 10), centers, sigmas=[1.0] * 4, amplitudes=amplitudes
    )

    noise = 0.0
    positions, det_flag = peak_finder(
        image, noise=noise, window_size=5, max_objects=max_objects
    )

    # All should be actual detections (det_flag=0)
    assert jnp.all(det_flag == 0), "All entries should be actual detections"
    assert jnp.all(positions[:, 0] >= 0), "All positions should be valid"
