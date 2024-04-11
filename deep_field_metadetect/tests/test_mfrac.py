import numpy as np
import galsim
import pytest
from numpy.testing import assert_allclose

from deep_field_metadetect.mfrac import compute_mfrac_interp_image


@pytest.mark.parametrize("loc", [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (50, 50),
    (50, 51),
    (51, 50),
    (51, 51),
    (99, 99),
    (99, 98),
    (98, 99),
    (98, 98),
    (100, 100),
    (100, 99),
    (99, 100),
])
@pytest.mark.parametrize("dim", [100, 101])
def test_compute_mfrac_interp_image_pixel_loc(dim, loc):
    if any([lv >= dim for lv in loc]):
        return

    mfrac = np.zeros((dim, dim))
    mfrac[loc[1], loc[0]] = 1.0  # index is y, x for row, col
    wcs = galsim.PixelScale(0.2)
    interp_image = compute_mfrac_interp_image(mfrac, wcs, fwhm=1e-6)
    assert_allclose(interp_image.xValue(loc[0], loc[1]), 1.0)
    for xo in [-1, 1]:
        for yo in [-1, 1]:
            assert_allclose(interp_image.xValue(loc[0] + xo, loc[1] + yo), 0.0)


@pytest.mark.parametrize("val", [1e-6, 0.5, 1.0])
@pytest.mark.parametrize("fwhm", [0.5, 1.0, 2.0])
def test_compute_mfrac_interp_image_const(val, fwhm):
    mfrac = np.zeros((100, 100)) + val
    wcs = galsim.PixelScale(0.2)
    interp_image = compute_mfrac_interp_image(mfrac, wcs, fwhm=fwhm)

    rng = np.random.RandomState(seed=1234)
    for _ in range(1000):
        x = rng.uniform(10, 90)
        y = rng.uniform(10, 90)
        _val = interp_image.xValue(x, y)
        assert_allclose(val, _val, atol=1e-6, rtol=0), (x, y, _val, val)
