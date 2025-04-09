import galsim
import ngmix
import numpy as np
import pytest
from ngmix import Jacobian, Observation
from ngmix.gaussmom import GaussMom as NgmixGaussMom
from ngmix.moments import fwhm_to_T
from ngmix.shape import e1e2_to_g1g2

from deep_field_metadetect.gaussmom.gaussmom import GaussMom
from deep_field_metadetect.gaussmom.gaussmom_core import obs_to_gaussmom_obs


@pytest.mark.parametrize("weight_fac", [1, 1e5])
@pytest.mark.parametrize("wcs_g1", [-0.5, 0, 0.2])
@pytest.mark.parametrize("wcs_g2", [-0.2, 0, 0.5])
@pytest.mark.parametrize("g1_true", [-0.1, 0, 0.2])
@pytest.mark.parametrize("g2_true", [-0.2, 0, 0.1])
def test_gaussmom_smoke(g1_true, g2_true, wcs_g1, wcs_g2, weight_fac):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    image_size = 107
    cen = (image_size - 1) / 2
    gs_wcs = galsim.ShearWCS(0.125, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obj = galsim.Gaussian(fwhm=fwhm).shear(g1=g1_true, g2=g2_true).withFlux(400)
    im = obj.drawImage(
        nx=image_size, ny=image_size, wcs=gs_wcs, method="no_pixel"
    ).array
    noise = np.sqrt(np.sum(im**2)) / 1e18
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []

    # get true flux
    ngmix_jac = ngmix.Jacobian(
        y=cen,
        x=cen,
        dudx=gs_wcs.dudx,
        dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx,
        dvdy=gs_wcs.dvdy,
    )
    ngmix_obs = ngmix.Observation(
        image=im,
        jacobian=ngmix_jac,
    )

    gaussmom_obs = obs_to_gaussmom_obs(obs=ngmix_obs)

    fitter = GaussMom(fwhm=fwhm * weight_fac)
    res = fitter.go(gaussmom_obs=gaussmom_obs)

    flux_true = res.flux
    # run ngmix
    fitter = NgmixGaussMom(fwhm=fwhm * weight_fac)
    # get true flux
    res = fitter.go(obs=ngmix_obs)

    for _ in range(50):
        shift = rng.uniform(low=-scale / 2, high=scale / 2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = (
            obj.shift(dx=shift[0], dy=shift[1])
            .drawImage(
                nx=image_size,
                ny=image_size,
                wcs=gs_wcs,
                method="no_pixel",
                dtype=np.float64,
            )
            .array
        )

        _im = im + (rng.normal(size=im.shape) * noise)
        jac = Jacobian(
            y=cen + xy.y,
            x=cen + xy.x,
            dudx=gs_wcs.dudx,
            dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx,
            dvdy=gs_wcs.dvdy,
        )

        _im = im + (rng.normal(size=im.shape) * noise)
        ngmix_obs = Observation(image=_im, weight=wgt, jacobian=jac)

        gaussmom_obs = obs_to_gaussmom_obs(obs=ngmix_obs)
        # use a huge weight so that we get the raw moments back out
        fitter = GaussMom(fwhm=fwhm * weight_fac)
        res = fitter.go(gaussmom_obs=gaussmom_obs)

        if res.flags == 0:
            if weight_fac > 1:
                # for unweighted we need to convert e to g
                _g1, _g2 = e1e2_to_g1g2(res.e[0], res.e[1])
            else:
                # we are weighting by the round gaussian before shearing.
                # Turns out this gives e that equals the shear g
                _g1, _g2 = res.e[0], res.e[1]

            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res.pars[4])
            farr.append(res.pars[5])

        # compute ngmix moments
        fitter = NgmixGaussMom(fwhm=fwhm * weight_fac)
        # get true flux
        ngmix_res = fitter.go(obs=ngmix_obs)

        # Compare ngmix VS jax versions
        np.testing.assert_allclose(res.sums, ngmix_res["sums"], atol=1e-9)
        np.testing.assert_allclose(res.sums_cov, ngmix_res["sums_cov"], atol=1e-9)
        np.testing.assert_allclose(res.e[0], ngmix_res["e"][0], atol=1e-9)
        np.testing.assert_allclose(res.e[1], ngmix_res["e"][1], atol=1e-9)
        np.testing.assert_allclose(res.s2n, ngmix_res["s2n"])
        np.testing.assert_allclose(res.flux, ngmix_res["flux"])
        np.testing.assert_allclose(res.pars[4], ngmix_res["pars"][4])

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)

    gtol = 1e-9
    assert np.abs(g1 - g1_true) < gtol, (g1, np.std(g1arr) / np.sqrt(len(g1arr)))
    assert np.abs(g2 - g2_true) < gtol, (g2, np.std(g2arr) / np.sqrt(len(g2arr)))

    # T test should only pass when the weight function is constant so
    # weight_fac needs to be rally big
    if g1_true == 0 and g2_true == 0 and weight_fac > 1:
        T = np.mean(Tarr)
        assert np.abs(T - fwhm_to_T(fwhm)) < 1e-6

    if weight_fac > 1:
        assert np.allclose(flux_true, np.sum(im))
    assert np.abs(np.mean(farr) - flux_true) < 1e-4, (np.mean(farr), np.std(farr))


# @pytest.mark.parametrize('do_higher', [False, True])
# def test_gaussmom_higher_smoke(do_higher):
#     fwhm = 0.9
#     scale = 0.263

#     obj = galsim.Gaussian(fwhm=fwhm)
#     im = obj.drawImage(
#         scale=scale,
#     ).array

#     cen = (np.array(im.shape) - 1) / 2
#     jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
#     obs = Observation(image=im, jacobian=jacobian)

#     fitter = GaussMom(fwhm=1.2, with_higher_order=do_higher)

#     res = fitter.go(obs)
#     if do_higher:
#         assert res['sums'].shape == (17, )
#         assert res['sums_cov'].shape == (17, 17)
#     else:
#         assert res['sums'].shape == (6, )
#         assert res['sums_cov'].shape == (6, 6)


# def test_gaussmom_higher_order():
#     rng = np.random.RandomState(seed=35)
#     fwhm = 0.9
#     sigma = fwhm_to_sigma(fwhm)
#     scale = 0.125
#     image_size = 107

#     ntrial = 100

#     rho4s = np.zeros(ntrial)
#     for i in range(ntrial):
#         obj = galsim.Gaussian(fwhm=fwhm)

#         row_offset, col_offset = rng.uniform(low=-0.5, high=0.5, size=2)

#         im = obj.drawImage(
#             nx=image_size,
#             ny=image_size,
#             offset=(col_offset, row_offset),
#             scale=scale,
#             method='no_pixel',
#         ).array

#         imcen = (np.array(im.shape) - 1) / 2

#         cen = imcen + (row_offset, col_offset)

#         jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

#         obs = Observation(image=im, jacobian=jacobian)

#         fitter = GaussMom(fwhm=fwhm, with_higher_order=True)

#         res = fitter.go(obs)

#         f_ind = MOMENTS_NAME_MAP["MF"]
#         M22_ind = MOMENTS_NAME_MAP["M22"]
#         rho4s[i] = res['sums'][M22_ind] / res['sums'][f_ind] / sigma**4

#     rho4_mean = rho4s.mean()
#     rho4_std = rho4s.std()
#     print(f'rho4: {rho4_mean:.3g} std: {rho4_std:.3g}')
#     assert np.abs(rho4_mean - 2) < 1e-5


# def test_gaussmom_flags():
#     """
#     test we get flags for very noisy data
#     """
#     rng = np.random.RandomState(seed=100)

#     ntrial = 10
#     noise = 100000
#     scale = 0.263
#     dims = [32]*2
#     weight = np.zeros(dims) + 1.0/noise**2

#     cen = (np.array(dims)-1)/2
#     jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

#     flags = np.zeros(ntrial)
#     for i in range(ntrial):

#         im = rng.normal(scale=noise, size=dims)

#         obs = Observation(
#             image=im,
#             weight=weight,
#             jacobian=jacobian,
#         )

#         fitter = GaussMom(fwhm=1.2)

#         res = fitter.go(obs)
#         flags[i] = res['flags']

#     assert np.any(flags != 0)
