import numpy as np

from deep_field_metadetect.jaxify.jax_metacal import (
    get_jax_galsim_object_from_dfmd_obs_nopix,
    jax_get_gauss_reconv_psf_galsim,
    jax_get_max_gauss_reconv_psf_galsim,
    jax_metacal_op_g1g2,
    jax_metacal_op_shears,
)
from deep_field_metadetect.jaxify.jax_utils import compute_dk, compute_kim_size
from deep_field_metadetect.jaxify.observation import (
    dfmd_obs_to_ngmix_obs,
    ngmix_obs_to_dfmd_obs,
)
from deep_field_metadetect.metacal import (
    get_galsim_object_from_ngmix_obs_nopix,
    get_gauss_reconv_psf_galsim,
    get_max_gauss_reconv_psf_galsim,
    metacal_op_g1g2,
    metacal_op_shears,
)
from deep_field_metadetect.utils import make_simple_sim


def _create_simple_obs_pair():
    """Create a simple observation pair for testing."""
    nxy = 53
    nxy_psf = 53
    scale = 0.2

    obs_w_ngmix, obs_d_ngmix, obs_dn_ngmix = make_simple_sim(
        seed=12345,
        g1=0.02,
        g2=0.0,
        s2n=1e8,
        dim=nxy,
        dim_psf=nxy_psf,
        scale=scale,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        return_dfmd_obs=False,
    )

    # Convert to JAX observations
    obs_w_jax = ngmix_obs_to_dfmd_obs(obs_w_ngmix)
    obs_d_jax = ngmix_obs_to_dfmd_obs(obs_d_ngmix)
    obs_dn_jax = ngmix_obs_to_dfmd_obs(obs_dn_ngmix)

    return {
        "ngmix": (obs_w_ngmix, obs_d_ngmix, obs_dn_ngmix),
        "jax": (obs_w_jax, obs_d_jax, obs_dn_jax),
        "params": {"nxy": nxy, "nxy_psf": nxy_psf, "scale": scale},
    }


def test_gauss_reconv_psf_consistency():
    """Test Gaussian reconvolution PSF."""
    simple_obs_pair = _create_simple_obs_pair()
    obs_w_ngmix, obs_d_ngmix, _ = simple_obs_pair["ngmix"]
    obs_w_jax, obs_d_jax, _ = simple_obs_pair["jax"]
    nxy_psf = simple_obs_pair["params"]["nxy_psf"]
    scale = simple_obs_pair["params"]["scale"]

    # Test single PSF
    psf_ngmix = get_galsim_object_from_ngmix_obs_nopix(obs_w_ngmix.psf, kind="image")
    psf_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w_jax.psf, kind="image")

    dk = compute_dk(pixel_scale=scale, image_size=nxy_psf)

    kim_size = 173
    reconv_psf_jax = jax_get_gauss_reconv_psf_galsim(psf_jax, dk=dk, kim_size=kim_size)
    reconv_psf_ngmix = get_gauss_reconv_psf_galsim(psf_ngmix, dk=dk, kim_size=kim_size)

    # Test PSF properties - relax tolerance for small numerical differences
    assert np.allclose(
        reconv_psf_ngmix.fwhm, reconv_psf_jax.fwhm, rtol=1e-10, atol=1e-12
    ), f"FWHM mismatch: {reconv_psf_ngmix.fwhm} vs {reconv_psf_jax.fwhm}"

    assert np.allclose(
        reconv_psf_ngmix.flux, reconv_psf_jax.flux, rtol=1e-10, atol=1e-12
    ), f"Flux mismatch: {reconv_psf_ngmix.flux} vs {reconv_psf_jax.flux}"


def test_max_gauss_reconv_psf_consistency():
    """Test max Gaussian reconvolution PSF.
    kim_size and dk are not get for Galsim in this case."""
    simple_obs_pair = _create_simple_obs_pair()
    obs_w_ngmix, obs_d_ngmix, _ = simple_obs_pair["ngmix"]
    obs_w_jax, obs_d_jax, _ = simple_obs_pair["jax"]
    nxy_psf = simple_obs_pair["params"]["nxy_psf"]
    scale = simple_obs_pair["params"]["scale"]
    dk = compute_dk(pixel_scale=scale, image_size=nxy_psf)
    kim_size = compute_kim_size(image_size=nxy_psf)

    # Get PSFs
    psf_w_ngmix = get_galsim_object_from_ngmix_obs_nopix(obs_w_ngmix.psf, kind="image")
    psf_d_ngmix = get_galsim_object_from_ngmix_obs_nopix(obs_d_ngmix.psf, kind="image")
    psf_w_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w_jax.psf, kind="image")
    psf_d_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_d_jax.psf, kind="image")

    # Compare maximum reconvolution PSFs
    max_reconv_psf_ngmix = get_max_gauss_reconv_psf_galsim(psf_w_ngmix, psf_d_ngmix)
    max_reconv_psf_jax = jax_get_max_gauss_reconv_psf_galsim(
        psf_w_jax,
        psf_d_jax,
        dk=dk,
        kim_size=kim_size,
    )

    # Test PSF properties
    # Not we have note fixed the dk and kim_size size for non-jax, so rtol is high
    assert np.allclose(
        max_reconv_psf_ngmix.fwhm, max_reconv_psf_jax.fwhm, rtol=0.02, atol=1e-6
    ), f"Max FWHM: {max_reconv_psf_ngmix.fwhm} vs {max_reconv_psf_jax.fwhm}"

    # Now test if the values are closer if dk and kim is fixed in non-jax
    max_reconv_psf_ngmix = get_max_gauss_reconv_psf_galsim(
        psf_w_ngmix,
        psf_d_ngmix,
        dk=dk,
        kim_size=kim_size,
    )
    assert np.allclose(
        max_reconv_psf_ngmix.fwhm, max_reconv_psf_jax.fwhm, rtol=1e-10, atol=1e-12
    ), f"Max FWHM: {max_reconv_psf_ngmix.fwhm} vs {max_reconv_psf_jax.fwhm}"


def test_metacal_single_shear_consistency():
    """Test single shear operations."""
    simple_obs_pair = _create_simple_obs_pair()
    obs_w_ngmix, _, _ = simple_obs_pair["ngmix"]
    obs_w_jax, _, _ = simple_obs_pair["jax"]
    nxy_psf = simple_obs_pair["params"]["nxy_psf"]
    scale = simple_obs_pair["params"]["scale"]

    # Test single shear transformation
    g1, g2 = 0.01, 0.0

    # Get reconvolution PSFs for both versions
    psf_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w_jax.psf, kind="image")
    psf_ngmix = get_galsim_object_from_ngmix_obs_nopix(obs_w_ngmix.psf, kind="image")

    dk = compute_dk(pixel_scale=scale, image_size=nxy_psf)
    kim_size = compute_kim_size(image_size=nxy_psf)
    reconv_psf_jax = jax_get_gauss_reconv_psf_galsim(psf_jax, dk, kim_size=kim_size)
    reconv_psf_ngmix = get_gauss_reconv_psf_galsim(psf_ngmix, dk=dk, kim_size=kim_size)

    # Run metacal operations
    mcal_obs_ngmix = metacal_op_g1g2(obs_w_ngmix, reconv_psf_ngmix, g1, g2)
    mcal_obs_jax = jax_metacal_op_g1g2(obs_w_jax, reconv_psf_jax, g1, g2, nxy_psf)

    # Convert JAX result to ngmix for comparison
    mcal_obs_jax_ngmix = dfmd_obs_to_ngmix_obs(mcal_obs_jax)

    # Compare image statistics
    # I don't expect them to be be exactly the same even with same fft_size
    # because psf_inv has some forcestepk and forcemaxk which are not the same
    assert np.allclose(
        mcal_obs_ngmix.image,
        mcal_obs_jax_ngmix.image,
        rtol=1e-8,
        atol=1e-8,
    ), "Image mean mismatch"


def test_metacal_shears_intermediate_values():
    """Test intermediate values in metacal shears operations."""
    simple_obs_pair = _create_simple_obs_pair()
    obs_w_ngmix, _, _ = simple_obs_pair["ngmix"]
    obs_w_jax, _, _ = simple_obs_pair["jax"]
    nxy_psf = simple_obs_pair["params"]["nxy_psf"]
    scale = simple_obs_pair["params"]["scale"]
    reconv_psf_dk = compute_dk(pixel_scale=scale, image_size=nxy_psf)
    reconv_psf_kim_size = compute_kim_size(image_size=nxy_psf)

    test_shears = ("noshear", "1p", "1m")

    # Run metacal operations
    mcal_res_ngmix = metacal_op_shears(
        obs_w_ngmix,
        shears=test_shears,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )
    mcal_res_jax = jax_metacal_op_shears(
        obs_w_jax,
        nxy_psf=nxy_psf,
        shears=test_shears,
        reconv_psf_dk=reconv_psf_dk,
        reconv_psf_kim_size=reconv_psf_kim_size,
    )

    # Convert JAX results to ngmix for comparison
    mcal_res_jax_ngmix = {}
    for shear in test_shears:
        mcal_res_jax_ngmix[shear] = dfmd_obs_to_ngmix_obs(mcal_res_jax[shear])

    # Compare results for each shear
    for shear in test_shears:
        obs_ngmix = mcal_res_ngmix[shear]
        obs_jax_ngmix = mcal_res_jax_ngmix[shear]

        assert np.allclose(obs_ngmix.image, obs_jax_ngmix.image, rtol=1e-8, atol=1e-8)
        assert np.allclose(obs_ngmix.noise, obs_jax_ngmix.noise, rtol=1e-10, atol=1e-12)

        assert np.allclose(
            obs_ngmix.weight, obs_jax_ngmix.weight, rtol=1e-10, atol=1e-12
        )
