import galsim
import numpy as np
import pytest

from deep_field_metadetect.pixel_cov import meas_pixel_cov


def test_meas_pixel_cov_gauss():
    rng = np.random.RandomState(seed=10)

    covs = []
    for _ in range(1000):
        img = rng.normal(size=(101, 101), scale=2.5)
        msk = rng.uniform(size=(101, 101)) > 0.2

        covs.append(meas_pixel_cov(img, msk))

    mn_cov = np.mean(covs, axis=0)
    true_cov = np.array(
        [
            [0, 0, 0],
            [0, 2.5**2, 0],
            [0, 0, 0],
        ]
    )
    print("\nmeas cov:\n", mn_cov)
    print("true cov:\n", true_cov)
    assert np.allclose(
        mn_cov,
        true_cov,
        atol=1e-2,
        rtol=0,
    )


@pytest.mark.parametrize(
    "g1,g2",
    [
        (0, 0),
        (0, 0.2),
        (0.1, 0),
        (-0.1, 0.5),
        (0, -0.1),
        (0.1, 0.3),
    ],
)
@pytest.mark.parametrize("mfrac", [0.0, 0.1])
def test_meas_pixel_cov_sheared(g1, g2, mfrac):
    rng = np.random.RandomState(seed=10)
    gs_rng = galsim.BaseDeviate(seed=10)

    var = 2.5**2
    covs = []
    for _ in range(1000):
        img = galsim.ImageD(101, 101)
        nse = galsim.UncorrelatedNoise(var, rng=gs_rng).shear(g1=g1, g2=g2)
        nse.applyTo(img)
        msk = rng.uniform(size=img.array.shape) >= mfrac
        covs.append(meas_pixel_cov(img.array, msk))
    mn_cov = np.mean(covs, axis=0)
    true_cov = galsim.ImageD(3, 3)
    true_cov = nse.drawImage(true_cov).array
    print("\nmeas cov:\n", mn_cov)
    print("true cov:\n", true_cov)
    assert np.allclose(
        mn_cov,
        true_cov,
        atol=1e-2,
        rtol=0,
    )


@pytest.mark.parametrize("mfrac", [0.0, 0.1])
def test_meas_pixel_cov_convolved(mfrac):
    rng = np.random.RandomState(seed=10)
    gs_rng = galsim.BaseDeviate(seed=10)

    var = 2.5**2
    covs = []
    for _ in range(1000):
        img = galsim.ImageD(101, 101)
        nse = galsim.UncorrelatedNoise(var, rng=gs_rng).convolvedWith(
            galsim.Gaussian(fwhm=4)
        )
        nse.applyTo(img)
        msk = rng.uniform(size=img.array.shape) >= mfrac
        covs.append(meas_pixel_cov(img.array, msk))
    mn_cov = np.mean(covs, axis=0)
    true_cov = galsim.ImageD(3, 3)
    true_cov = nse.drawImage(true_cov).array
    print("\nmeas cov:\n", mn_cov)
    print("true cov:\n", true_cov)
    assert np.allclose(
        mn_cov,
        true_cov,
        atol=1e-2,
        rtol=0,
    )
