import multiprocessing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metacal import (
    jax_add_dfmd_obs,
    jax_metacal_op_shears,
)
from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size
from deep_field_metadetect.jaxify.observation import (
    DFMdetObservation,
    DFMdetPSF,
    dfmd_obs_to_ngmix_obs,
    ngmix_obs_to_dfmd_obs,
)
from deep_field_metadetect.metacal import add_ngmix_obs, metacal_op_shears
from deep_field_metadetect.utils import (
    assert_m_c_ok,
    estimate_m_and_c,
    fit_gauss_mom_mcal_res,
    make_simple_sim,
    measure_mcal_shear_quants,
    print_m_c,
)


def _run_single_sim_pair(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        reconv_psf_dk=compute_dk(pixel_scale=scale, image_size=nxy_psf),
        reconv_psf_kim_size=compute_kim_size(image_size=nxy_psf),
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=True,
    )
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        reconv_psf_dk=compute_dk(pixel_scale=scale, image_size=nxy_psf),
        reconv_psf_kim_size=compute_kim_size(image_size=nxy_psf),
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return res_p, res_m


def _run_single_sim_pair_jax_and_ngmix(seed, s2n):
    nxy = 53
    nxy_psf = 53
    scale = 0.2
    obs_plus, *_ = make_simple_sim(
        seed=seed,
        g1=0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_plus)

    res_p_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_p_ngmix = measure_mcal_shear_quants(res_p_ngmix)

    obs_plus = ngmix_obs_to_dfmd_obs(obs_plus)

    mcal_res = jax_metacal_op_shears(
        obs_plus,
        nxy_psf=nxy_psf,
        reconv_psf_dk=compute_dk(pixel_scale=scale, image_size=nxy_psf),
        reconv_psf_kim_size=compute_kim_size(image_size=nxy_psf),
    )
    res_p = fit_gauss_mom_mcal_res(mcal_res)
    res_p = measure_mcal_shear_quants(res_p)

    obs_minus, *_ = make_simple_sim(
        seed=seed,
        g1=-0.02,
        g2=0.0,
        s2n=s2n,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    mcal_res_ngmix = metacal_op_shears(obs_minus)
    res_m_ngmix = fit_gauss_mom_mcal_res(mcal_res_ngmix)
    res_m_ngmix = measure_mcal_shear_quants(res_m_ngmix)

    obs_minus = ngmix_obs_to_dfmd_obs(obs_minus)
    mcal_res = jax_metacal_op_shears(
        obs_minus,
        nxy_psf=nxy_psf,
        reconv_psf_dk=compute_dk(pixel_scale=scale, image_size=nxy_psf),
        reconv_psf_kim_size=compute_kim_size(image_size=nxy_psf),
    )
    res_m = fit_gauss_mom_mcal_res(mcal_res)
    res_m = measure_mcal_shear_quants(res_m)

    return (res_p, res_m), (res_p_ngmix, res_m_ngmix)


def test_metacal_smoke():
    res_p, res_m = _run_single_sim_pair(1234, 1e8)
    for col in res_p.dtype.names:
        assert np.isfinite(res_p[col]).all()
        assert np.isfinite(res_m[col]).all()


def test_metacal_jax_vs_ngmix():
    nsims = 5

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    res_p_ngmix = []
    res_m_ngmix = []
    for seed in seeds:
        res, res_ngmix = _run_single_sim_pair_jax_and_ngmix(seed, 1e8)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

            res_p_ngmix.append(res_ngmix[0])
            res_m_ngmix.append(res_ngmix[1])

            assert np.allclose(
                res[0].tolist(),
                res_ngmix[0].tolist(),
                atol=1e-3,
                rtol=0.01,
                equal_nan=True,
            )
            assert np.allclose(
                res[1].tolist(),
                res_ngmix[1].tolist(),
                atol=1e-3,
                rtol=0.01,
                equal_nan=True,
            )

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng = estimate_m_and_c(
        np.concatenate(res_p_ngmix),
        np.concatenate(res_m_ngmix),
        0.02,
        jackknife=len(res_p_ngmix),
    )

    print("JAX results:")
    print_m_c(m, merr, c1, c1err, c2, c2err)
    print("ngmix results:")
    print_m_c(m_ng, merr_ng, c1_ng, c1err_ng, c2_ng, c2err_ng)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)

    assert np.allclose(m, m_ng, atol=1e-4)
    assert np.allclose(merr, merr_ng, atol=1e-4)
    assert np.allclose(c1err, c1err_ng, atol=1e-6)
    assert np.allclose(c1, c1_ng, atol=1e-6)
    assert np.allclose(c2err, c2err_ng, atol=1e-6)
    assert np.allclose(c2, c2_ng, atol=1e-6)


def test_metacal():
    nsims = 50

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    for seed in seeds:
        res = _run_single_sim_pair(seed, 1e8)
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
        np.concatenate(res_p),
        np.concatenate(res_m),
        0.02,
        jackknife=len(res_p),
    )

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


@pytest.mark.slow
def test_metacal_slow():  # pragma: no cover
    nsims = 100_000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size + 1
    nsims = nchunks * chunk_size

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc : loc + chunk_size]
        for seed in _seeds:
            res = _run_single_sim_pair(seed, 20)
            if res is not None:
                res_p.append(res[0])
                res_m.append(res[1])

        if len(res_p) < 500:
            njack = len(res_p)
        else:
            njack = 100

        m, merr, c1, c1err, c2, c2err = estimate_m_and_c(
            np.concatenate(res_p),
            np.concatenate(res_m),
            0.02,
            jackknife=njack,
        )

        print("# of sims:", len(res_p), flush=True)
        print_m_c(m, merr, c1, c1err, c2, c2err)

        loc += chunk_size

    print_m_c(m, merr, c1, c1err, c2, c2err)
    assert_m_c_ok(m, merr, c1, c1err, c2, c2err)


def test_jax_vs_ngmix_render_psf_and_build_obs():
    """Test _jax_render_psf_and_build_obs vs render_psf_and_build_obs"""
    import galsim
    import jax_galsim

    from deep_field_metadetect.jaxify.jax_metacal import (
        _jax_render_psf_and_build_obs,
        jax_get_gauss_reconv_psf_galsim,
    )
    from deep_field_metadetect.jaxify.jax_utils import compute_dk
    from deep_field_metadetect.jaxify.observation import ngmix_obs_to_dfmd_obs
    from deep_field_metadetect.metacal import (
        _render_psf_and_build_obs,
        get_gauss_reconv_psf_galsim,
    )
    from deep_field_metadetect.utils import make_simple_sim

    # Create test observations
    nxy = 53
    nxy_psf = 21
    scale = 0.2

    ngmix_obs, _, _ = make_simple_sim(
        seed=12345,
        g1=0.0,
        g2=0.0,
        s2n=1e8,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        return_dfmd_obs=False,
    )

    # Convert to dfmd observation
    dfmd_obs = ngmix_obs_to_dfmd_obs(ngmix_obs)

    # Create test reconv PSFs
    test_image = jnp.ones((nxy, nxy))

    # JAX version
    jax_psf = jax_galsim.Gaussian(sigma=1.0).withFlux(1.0)
    dk = compute_dk(pixel_scale=scale, image_size=nxy_psf)
    kim_size = compute_kim_size(image_size=nxy_psf)
    jax_reconv_psf = jax_get_gauss_reconv_psf_galsim(jax_psf, dk=dk, kim_size=kim_size)
    jax_result = _jax_render_psf_and_build_obs(
        test_image, dfmd_obs, jax_reconv_psf, nxy_psf=nxy_psf, weight_fac=1
    )

    # ngmix version
    ngmix_psf = galsim.Gaussian(sigma=1.0).withFlux(1.0)
    ngmix_reconv_psf = get_gauss_reconv_psf_galsim(ngmix_psf, dk=dk, kim_size=kim_size)

    ngmix_result = _render_psf_and_build_obs(
        test_image, ngmix_obs, ngmix_reconv_psf, weight_fac=1
    )

    assert jnp.isclose(ngmix_reconv_psf.sigma, jax_reconv_psf.sigma), (
        "reconv psf sigmas are different"
    )
    # Check if shapes match
    assert jax_result.psf.image.shape == ngmix_result.psf.image.shape, (
        f"PSF shapes don't match: JAX {jax_result.psf.image.shape} "
        f"vs ngmix {ngmix_result.psf.image.shape}"
    )

    # Compare PSF images with some tolerance
    diff = jnp.abs(jax_result.psf.image - ngmix_result.psf.image)
    max_diff = jnp.max(diff)
    rel_diff = max_diff / jnp.max(jax_result.psf.image)

    print(f"Max absolute difference: {max_diff}")
    print(f"Max relative difference: {rel_diff}")

    # Test that PSF images are reasonably close
    assert jnp.allclose(
        jax_result.psf.image, ngmix_result.psf.image, atol=1e-10, rtol=1e-6
    ), f"PSF images differ significantly. Max diff: {max_diff}, Rel diff: {rel_diff}"


def _create_test_dfmd_obs(
    has_bmask=False,
    has_ormask=False,
    has_noise=False,
    has_mfrac=False,
    has_psf=True,
    has_wcs=True,
    seed=42,
):
    """Create a test DFMdetObservation with specified attributes.
    If has_* is set to False, the defaults are used."""
    import jax_galsim

    key = jax.random.PRNGKey(seed)

    image = jnp.ones((10, 10))
    weight = jnp.ones((10, 10))

    wcs = jax_galsim.wcs.AffineTransform(
        dudx=0.2, dudy=0.01, dvdx=0.02, dvdy=0.3, origin=jax_galsim.PositionD(5.5, 4.5)
    )

    obs = DFMdetObservation(
        image=image,
        weight=weight,
        bmask=jax.random.randint(key, (10, 10), 0, 2, dtype=jnp.int32)
        if has_bmask
        else None,
        ormask=jax.random.randint(
            jax.random.split(key)[0], (10, 10), 0, 2, dtype=jnp.int32
        )
        if has_ormask
        else None,
        noise=jax.random.uniform(
            jax.random.split(key)[1], (10, 10), minval=0.05, maxval=0.15
        )
        if has_noise
        else None,
        mfrac=jax.random.uniform(
            jax.random.split(jax.random.split(key)[0])[1],
            (10, 10),
            minval=0.01,
            maxval=0.5,
        )
        if has_mfrac
        else None,
        wcs=wcs if has_wcs else None,
        psf=DFMdetPSF(
            image=jax.random.uniform(
                jax.random.split(jax.random.split(jax.random.split(key)[0])[0])[1],
                (5, 5),
                minval=0.8,
                maxval=1.2,
            ).astype(jnp.float32),
            wcs=wcs,
        )
        if has_psf
        else None,
    )
    return obs


def test_jax_add_dfmd_obs_vs_add_dfmd_obs():
    """Test that jax_add_dfmd_obs and add_dfmd_obs
    They should return the same values for image, psf, weight, wcs."""

    obs1 = _create_test_dfmd_obs(
        has_bmask=True,
        has_ormask=True,
        has_noise=True,
        has_mfrac=True,
        has_psf=True,
        has_wcs=False,
        seed=11,
    )
    obs2 = _create_test_dfmd_obs(
        has_bmask=True,
        has_ormask=True,
        has_noise=True,
        has_mfrac=True,
        has_psf=True,
        has_wcs=False,
        seed=13,
    )
    ngmix_obs1 = dfmd_obs_to_ngmix_obs(obs1)
    ngmix_obs2 = dfmd_obs_to_ngmix_obs(obs2)

    jax_result = jax_add_dfmd_obs(
        obs1, obs2, ignore_psf=False, skip_mfrac_for_second=False
    )
    non_jax_result = add_ngmix_obs(
        ngmix_obs1, ngmix_obs2, ignore_psf=False, skip_mfrac_for_second=False
    )

    assert jnp.allclose(jax_result.image, non_jax_result.image, atol=1e-10), (
        "Images do not match"
    )
    assert jnp.allclose(jax_result.weight, non_jax_result.weight, atol=1e-10), (
        "Weights do not match"
    )

    assert jax_result.wcs.dudx == non_jax_result.jacobian.dudcol, (
        "wcs dudx does not match"
    )
    assert jax_result.wcs.dudy == non_jax_result.jacobian.dudrow, (
        "wcs dudy does not match"
    )
    assert jax_result.wcs.dvdx == non_jax_result.jacobian.dvdcol, (
        "wcs dvdx does not match"
    )
    assert jax_result.wcs.dvdy == non_jax_result.jacobian.dvdrow, (
        "wcs dvdy does not match"
    )
    assert jax_result.wcs.origin.x == non_jax_result.jacobian.col0 + 1, (
        "wcs does not match"
    )
    assert jax_result.wcs.origin.y == non_jax_result.jacobian.row0 + 1, (
        "wcs does not match"
    )

    # Compare PSF if both have PSF
    if jax_result.has_psf() and non_jax_result.has_psf():
        assert jnp.allclose(
            jax_result.psf.image, non_jax_result.psf.image, atol=1e-10
        ), "PSF images do not match"
        assert jnp.allclose(
            jax_result.psf.weight, non_jax_result.psf.weight, atol=1e-10
        ), "PSF weights do not match"
        assert jax_result.psf.wcs.dudx == non_jax_result.psf.jacobian.dudcol, (
            "PSF wcs does not match"
        )
        assert jax_result.psf.wcs.dudy == non_jax_result.psf.jacobian.dudrow, (
            "PSF wcs does not match"
        )
        assert jax_result.psf.wcs.dvdx == non_jax_result.psf.jacobian.dvdcol, (
            "PSF wcs does not match"
        )
        assert jax_result.psf.wcs.dvdy == non_jax_result.psf.jacobian.dvdrow, (
            "PSF wcs does not match"
        )

        assert jax_result.psf.wcs.origin.x == non_jax_result.psf.jacobian.col0 + 1, (
            "PSF wcs does not match"
        )
        assert jax_result.psf.wcs.origin.y == non_jax_result.psf.jacobian.row0 + 1, (
            "PSF wcs does not match"
        )

    if jax_result.has_bmask():
        assert jnp.allclose(jax_result.bmask, non_jax_result.bmask), (
            "bmasks do not match"
        )

    if jax_result.has_ormask():
        assert jnp.allclose(jax_result.ormask, non_jax_result.ormask), (
            "ormasks do not match"
        )

    if jax_result.has_noise():
        assert jnp.allclose(jax_result.noise, non_jax_result.noise, atol=1e-10), (
            "noise do not match"
        )

    if jax_result.has_mfrac():
        assert jnp.allclose(jax_result.mfrac, non_jax_result.mfrac, atol=1e-10), (
            "mfrac do not match"
        )


def test_jax_add_dfmd_obs_vs_add_dfmd_obs_ignore_psf():
    """Test that both functions work correctly when ignoring PSF."""
    obs1 = _create_test_dfmd_obs(
        has_bmask=True,
        has_ormask=True,
        has_noise=True,
        has_mfrac=True,
        has_psf=True,
        seed=17,
    )
    obs2 = _create_test_dfmd_obs(
        has_bmask=True,
        has_ormask=True,
        has_noise=True,
        has_mfrac=True,
        has_psf=True,
        seed=19,
    )
    ngmix_obs1 = dfmd_obs_to_ngmix_obs(obs1)
    ngmix_obs2 = dfmd_obs_to_ngmix_obs(obs2)

    jax_result = jax_add_dfmd_obs(
        obs1, obs2, ignore_psf=True, skip_mfrac_for_second=False
    )
    non_jax_result = add_ngmix_obs(
        ngmix_obs1, ngmix_obs2, ignore_psf=True, skip_mfrac_for_second=False
    )

    assert jnp.allclose(jax_result.image, non_jax_result.image, atol=1e-10), (
        "Images do not match with ignore_psf=True"
    )
    assert jnp.allclose(jax_result.weight, non_jax_result.weight, atol=1e-10), (
        "Weights do not match with ignore_psf=True"
    )

    assert jax_result.wcs.dudx == non_jax_result.jacobian.dudcol, (
        "wcs dudx does not match with ignore_psf=True"
    )
    assert jax_result.wcs.dudy == non_jax_result.jacobian.dudrow, (
        "wcs dudy does not match with ignore_psf=True"
    )
    assert jax_result.wcs.dvdx == non_jax_result.jacobian.dvdcol, (
        "wcs dvdx does not match with ignore_psf=True"
    )
    assert jax_result.wcs.dvdy == non_jax_result.jacobian.dvdrow, (
        "wcs dvdy does not match with ignore_psf=True"
    )
    assert jax_result.wcs.origin.x == non_jax_result.jacobian.col0 + 1, (
        "wcs origin.x does not match with ignore_psf=True"
    )
    assert jax_result.wcs.origin.y == non_jax_result.jacobian.row0 + 1, (
        "wcs origin.y does not match with ignore_psf=True"
    )

    assert not jax_result.has_psf(), (
        "JAX result should not have PSF when ignore_psf=True"
    )
    assert not non_jax_result.has_psf(), (
        "Non-JAX result should not have PSF when ignore_psf=True"
    )


def test_jax_add_dfmd_obs_vs_add_dfmd_obs_skip_mfrac():
    """Test that both functions handle skip_mfrac_for_second correctly."""
    obs1 = _create_test_dfmd_obs(has_mfrac=True, has_psf=True, seed=16)
    obs2 = _create_test_dfmd_obs(has_mfrac=True, has_psf=True, seed=12)
    ngmix_obs1 = dfmd_obs_to_ngmix_obs(obs1)
    ngmix_obs2 = dfmd_obs_to_ngmix_obs(obs2)

    jax_result = jax_add_dfmd_obs(
        obs1, obs2, ignore_psf=True, skip_mfrac_for_second=True
    )
    non_jax_result = add_ngmix_obs(
        ngmix_obs1, ngmix_obs2, ignore_psf=True, skip_mfrac_for_second=True
    )

    assert jnp.allclose(jax_result.mfrac, non_jax_result.mfrac, atol=1e-10), (
        "mfrac do not match with skip_mfrac_for_second=True"
    )
    assert jnp.allclose(jax_result.mfrac, obs1.mfrac, atol=1e-10), (
        "mfrac should equal obs1.mfrac when skip_mfrac_for_second=True"
    )

    jax_result = jax_add_dfmd_obs(
        obs1, obs2, ignore_psf=True, skip_mfrac_for_second=False
    )
    non_jax_result = add_ngmix_obs(
        ngmix_obs1, ngmix_obs2, ignore_psf=True, skip_mfrac_for_second=False
    )

    assert jnp.allclose(jax_result.mfrac, non_jax_result.mfrac, atol=1e-10), (
        "mfrac do not match with skip_mfrac_for_second=False"
    )
    expected_mfrac = (obs1.mfrac + obs2.mfrac) / 2
    assert jnp.allclose(jax_result.mfrac, expected_mfrac, atol=1e-10), (
        "mfrac should be average when skip_mfrac_for_second=False"
    )
