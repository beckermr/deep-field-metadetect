import numpy as np


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
