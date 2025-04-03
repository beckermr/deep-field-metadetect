import jax.numpy as jnp


# @partial(jax.jit, static_argnames=["pixel_scale", "image_size"])
def compute_stepk(pixel_scale, image_size):
    """Compute psf fourier scale based on pixel scale and image dimension
    The size if obtained from from galsim.GSObject.getGoodImageSize
    The factor 1/4 from deep_field_metadetect.metacal.get_gauss_reconv_psf_galsim
    """
    return 2 * jnp.pi / (image_size * pixel_scale) / 4
