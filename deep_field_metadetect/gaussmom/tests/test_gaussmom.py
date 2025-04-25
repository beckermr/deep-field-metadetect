import galsim
import jax.numpy as jnp
import ngmix
import numpy as np
import pytest
from ngmix import Jacobian, Observation
from ngmix.gaussmom import GaussMom as NgmixGaussMom
from ngmix.moments import fwhm_to_T
from ngmix.shape import e1e2_to_g1g2

from deep_field_metadetect.gaussmom.gaussmom import (
    GaussMom,
    _compute_shape_params,
    _diag_all_true,
    _set_fluxerr_s2n_flux_flags,
    _set_T_Terr_Tflags,
)
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
        np.testing.assert_allclose(res.sums, ngmix_res["sums"], atol=1e-10)
        np.testing.assert_allclose(res.sums_cov, ngmix_res["sums_cov"], atol=1e-10)
        np.testing.assert_allclose(res.e[0], ngmix_res["e"][0], atol=1e-10)
        np.testing.assert_allclose(res.e[1], ngmix_res["e"][1], atol=1e-10)
        np.testing.assert_allclose(res.s2n, ngmix_res["s2n"])
        np.testing.assert_allclose(res.flux, ngmix_res["flux"])
        np.testing.assert_allclose(res.pars[4], ngmix_res["pars"][4])

        np.testing.assert_allclose(res.Mv, ngmix_res["Mv"], atol=1e-10)
        np.testing.assert_allclose(res.Mu, ngmix_res["Mu"], atol=1e-10)
        np.testing.assert_allclose(res.M1, ngmix_res["M1"], atol=1e-10)
        np.testing.assert_allclose(res.M2, ngmix_res["M2"], atol=1e-10)
        np.testing.assert_allclose(res.MT, ngmix_res["MT"], atol=1e-10)
        np.testing.assert_allclose(res.MF, ngmix_res["MF"], atol=1e-10)

        np.testing.assert_allclose(res.Mv_err, ngmix_res["Mv_err"], atol=1e-10)
        np.testing.assert_allclose(res.Mu_err, ngmix_res["Mu_err"], atol=1e-10)
        np.testing.assert_allclose(res.M1_err, ngmix_res["M1_err"], atol=1e-10)
        np.testing.assert_allclose(res.M2_err, ngmix_res["M2_err"], atol=1e-10)
        np.testing.assert_allclose(res.MT_err, ngmix_res["MT_err"], atol=1e-10)
        np.testing.assert_allclose(res.MF_err, ngmix_res["MF_err"], atol=1e-10)

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


def test_gaussmom_flags():
    """
    test we get flags for very noisy data
    """
    rng = np.random.RandomState(seed=100)

    ntrial = 10
    noise = 100000
    scale = 0.263
    dims = [32] * 2
    weight = np.zeros(dims) + 1.0 / noise**2

    cen = (np.array(dims) - 1) / 2
    jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

    flags = np.zeros(ntrial)
    for i in range(ntrial):
        im = rng.normal(scale=noise, size=dims)

        ngmix_obs = Observation(
            image=im,
            weight=weight,
            jacobian=jacobian,
        )

        gaussmom_obs = obs_to_gaussmom_obs(obs=ngmix_obs)

        fitter = GaussMom(fwhm=1.2)

        res = fitter.go(gaussmom_obs)
        flags[i] = res.flags

    assert np.any(flags != 0)


# Test _set_fluxerr_s2n_flux_flags
def test_set_fluxerr_s2n_flux_flags_positive_var():
    res_flux = 10.0
    cov = jnp.eye(3) * 4.0
    res_flux_flags = 0
    mf_ind = 1

    res_flux_err, res_s2n, flags = _set_fluxerr_s2n_flux_flags(
        res_flux, cov, res_flux_flags, mf_ind
    )

    assert jnp.isclose(res_flux_err, 2.0)
    assert jnp.isclose(res_s2n, 5.0)
    assert flags == 0


def test_set_fluxerr_s2n_flux_flags_nonpos_var():
    res_flux = 10.0
    cov = jnp.array([[4.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    res_flux_flags = 0
    mf_ind = 1

    res_flux_err, res_s2n, flags = _set_fluxerr_s2n_flux_flags(
        res_flux, cov, res_flux_flags, mf_ind
    )

    assert jnp.isnan(res_flux_err)
    assert jnp.isnan(res_s2n)
    assert flags == ngmix.flags.NONPOS_VAR


# Test _set_T_Terr_Tflags
def test_set_T_Terr_Tflags_valid():
    sums = jnp.array([20.0, 10.0])  # mt, mf
    sums_cov = jnp.array([[4.0, 1.0], [1.0, 9.0]])
    mt_ind = 0
    mf_ind = 1
    res_T_flags = 0

    res_T, res_T_err, flags = _set_T_Terr_Tflags(
        sums, sums_cov, mt_ind, mf_ind, res_T_flags
    )

    assert jnp.isclose(res_T, 2.0)
    assert jnp.isfinite(res_T_err)
    assert flags == 0


def test_set_T_Terr_Tflags_nonpos_variance():
    sums = jnp.array([20.0, 10.0])
    sums_cov = jnp.array([[4.0, 1.0], [1.0, -1.0]])  # mf variance non-positive
    mt_ind = 0
    mf_ind = 1
    res_T_flags = 0

    res_T, res_T_err, flags = _set_T_Terr_Tflags(
        sums, sums_cov, mt_ind, mf_ind, res_T_flags
    )

    assert jnp.isnan(res_T)
    assert jnp.isnan(res_T_err)
    assert flags == ngmix.flags.NONPOS_VAR


def test_set_T_Terr_Tflags_nonpos_flux():
    sums = jnp.array([20.0, -5.0])  # Negative flux
    sums_cov = jnp.array([[4.0, 1.0], [1.0, 9.0]])
    mt_ind = 0
    mf_ind = 1
    res_T_flags = 0

    res_T, res_T_err, flags = _set_T_Terr_Tflags(
        sums, sums_cov, mt_ind, mf_ind, res_T_flags
    )

    assert jnp.isnan(res_T)
    assert jnp.isnan(res_T_err)
    assert flags == ngmix.flags.NONPOS_FLUX


# Test  _diag_all_true


def test_diag_all_true_valid_cov():
    cov = jnp.diag(jnp.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]))
    res_flags = 0

    diag, flags = _diag_all_true(cov, res_flags)

    expected = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.float32)

    assert jnp.allclose(diag, expected)
    assert flags == 0


def test_diag_all_true_invalid_cov():
    cov = jnp.diag(
        jnp.array([1.0, 4.0, 0.0, 16.0, -5.0, 36.0])
    )  # contains zero and negative
    res_flags = 0

    diag, flags = _diag_all_true(cov, res_flags)

    assert jnp.all(jnp.isnan(diag))
    assert flags == ngmix.flags.NONPOS_VAR


def test_diag_all_true_invalid_cov_with_existing_flags():
    cov = jnp.diag(jnp.array([1.0, -2.0, 9.0, 16.0, 25.0, 36.0]))
    res_flags = 8  # arbitrary existing flags

    diag, flags = _diag_all_true(cov, res_flags)

    assert jnp.all(jnp.isnan(diag))
    assert flags == (8 | ngmix.flags.NONPOS_VAR)


# Test _compute_shape_params()


def test_compute_shape_valid():
    sums = jnp.array([0.2, 0.3, 10.0])  # m1, m2, T
    sums_cov = jnp.eye(3) * 0.01
    m1, m2, mt = 0, 1, 2
    res_T = 10.0
    res_flux = 5.0
    res_flags = 0

    res_e, e_err, e_cov, flags = _compute_shape_params(
        sums, sums_cov, m1, m2, mt, res_T, res_flux, res_flags
    )

    assert jnp.all(jnp.isfinite(res_e))
    assert jnp.all(jnp.isfinite(e_err))
    assert jnp.all(jnp.isfinite(e_cov))
    assert flags == 0


def test_compute_shape_nonpositive_flux():
    sums = jnp.array([0.2, 0.3, 10.0])
    sums_cov = jnp.eye(3) * 0.01
    m1, m2, mt = 0, 1, 2
    res_T = 10.0
    res_flux = 0.0
    res_flags = 0

    _, _, _, flags = _compute_shape_params(
        sums, sums_cov, m1, m2, mt, res_T, res_flux, res_flags
    )
    assert flags & ngmix.flags.NONPOS_FLUX


def test_compute_shape_nonpositive_T():
    sums = jnp.array([0.2, 0.3, 0.0])
    sums_cov = jnp.eye(3) * 0.01
    m1, m2, mt = 0, 1, 2
    res_T = 0.0
    res_flux = 5.0
    res_flags = 0

    _, _, _, flags = _compute_shape_params(
        sums, sums_cov, m1, m2, mt, res_T, res_flux, res_flags
    )
    assert flags & ngmix.flags.NONPOS_SIZE


def test_compute_shape_nonfinite_error():
    # Force NaN in e_err
    sums = jnp.array([0.2, 0.3, 10.0])
    sums_cov = jnp.array([[jnp.nan, 0.0, 0.0], [0.0, jnp.nan, 0.0], [0.0, 0.0, 0.01]])
    m1, m2, mt = 1, 1, 2
    res_T = 10.0
    res_flux = 5.0
    res_flags = 0

    _, _, _, flags = _compute_shape_params(
        sums, sums_cov, m1, m2, mt, res_T, res_flux, res_flags
    )
    assert flags & ngmix.flags.NONPOS_SHAPE_VAR


def test_compute_shape_early_exit_on_flags():
    sums = jnp.array([0.2, 0.3, 10.0])
    sums_cov = jnp.eye(3) * 0.01
    m1, m2, mt = 1, 1, 2
    res_T = 10.0
    res_flux = 5.0
    res_flags = 4  # Already set some flag

    res_e, e_err, e_cov, flags = _compute_shape_params(
        sums, sums_cov, m1, m2, mt, res_T, res_flux, res_flags
    )

    assert jnp.all(jnp.isnan(res_e))
    assert jnp.all(jnp.isnan(e_err))
    assert jnp.all(jnp.isnan(jnp.diag(e_cov)))
    assert flags == res_flags
