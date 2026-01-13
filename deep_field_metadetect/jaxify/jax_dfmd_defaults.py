from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size

DEFAULT_NXY_PSF = 53
DEFAULT_PIXEL_SCALE = 0.2
DEFAULT_FFT_SIZE = 256

DEFAULT_RECONV_DK = compute_dk(pixel_scale=0.2, image_size=53)
DEFAULT_KIM_SIZE = compute_kim_size(image_size=DEFAULT_NXY_PSF)
