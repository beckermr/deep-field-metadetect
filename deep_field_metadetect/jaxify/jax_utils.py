import jax.numpy as jnp


# @partial(jax.jit, static_argnames=["pixel_scale", "image_size"])
def compute_stepk(pixel_scale, image_size):
    """Compute psf fourier scale based on pixel scale and psf image dimension
    The size if obtained from from galsim.GSObject.getGoodImageSize
    The factor 1/4 from deep_field_metadetect.metacal.get_gauss_reconv_psf_galsim

    Parameters:
    -----------
    pixel_scale : float
        The scale of a single pixel in the image.
    image_size : int
        The dimension of the PSF image (typically a square size).

    Returns:
    --------
    float
        The computed stepk value, which represents the Fourier-space sampling frequency.
    """
    return 2 * jnp.pi / (image_size * pixel_scale) / 4
