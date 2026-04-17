from deep_field_metadetect.jaxify.jax_utils import (
    compute_dk,
    compute_kim_size,
    compute_max_objects,
)

DEFAULT_NXY_PSF = 53
DEFAULT_PIXEL_SCALE = 0.2
DEFAULT_FFT_SIZE = 256
DEAFAULT_COADD_SIZE = 201
DEFAULT_NUMBER_DENSITY = 50.0  # objects per square arcminute

DEFAULT_RECONV_DK = compute_dk(
    pixel_scale=DEFAULT_PIXEL_SCALE, image_size=DEFAULT_NXY_PSF
)
DEFAULT_KIM_SIZE = compute_kim_size(image_size=DEFAULT_NXY_PSF)

MAX_OBJECTS = compute_max_objects(
    field_size=DEAFAULT_COADD_SIZE,
    pixel_scale=DEFAULT_PIXEL_SCALE,
    number_density=DEFAULT_NUMBER_DENSITY,
)
