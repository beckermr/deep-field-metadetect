import numba
import numpy as np


@numba.njit()
def _meas_cov(im, msk, cov, nrm):  # pragma: no cover
    for ic in range(im.shape[0]):
        io_min = max(ic - 1, 0)
        io_max = min(ic + 1, im.shape[0] - 1) + 1

        for jc in range(im.shape[1]):
            jo_min = max(jc - 1, 0)
            jo_max = min(jc + 1, im.shape[1] - 1) + 1

            if not msk[ic, jc]:
                continue

            for io in range(io_min, io_max):
                i = io - ic + 1

                for jo in range(jo_min, jo_max):
                    j = jo - jc + 1

                    if not msk[io, jo]:
                        continue

                    cov[i, j] += im[ic, jc] * im[io, jo]
                    nrm[i, j] += 1

    for i in range(3):
        for j in range(3):
            cov[i, j] /= nrm[i, j]


def meas_pixel_cov(im, msk):
    """Measure the one-offset covariance of pixels in an image, ignoring bad ones.

    Parameters
    ----------
    im : np.ndarray
        The image.
    msk : np.ndarray
        The mask where True is a good pixel and False is a bad pixel.

    Returns
    -------
    cov : np.ndarray
        The 3x3 matrix of covariances between the pixels.
    """
    cov = np.zeros((3, 3))
    nrm = np.zeros((3, 3))
    mn = np.mean(im[msk])

    _meas_cov(im.astype(np.float64) - mn, msk, cov, nrm)
    return cov
