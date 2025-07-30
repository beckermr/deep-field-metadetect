import numpy as np
import pytest

from deep_field_metadetect.jaxify.jax_metacal import (
    get_jax_galsim_object_from_dfmd_obs_nopix,
    jax_get_gauss_reconv_psf_galsim,
    jax_get_max_gauss_reconv_psf_galsim,
    jax_metacal_op_g1g2,
    jax_metacal_op_shears,
)
from deep_field_metadetect.jaxify.jax_utils import compute_stepk
from deep_field_metadetect.jaxify.observation import (
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


class TestJaxNgmixIntermediates:
    """Test is the versions produce the same intermediate values."""

    @pytest.fixture(scope="class")
    def simple_obs_pair(self):
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

    def test_gauss_reconv_psf_consistency(self, simple_obs_pair):
        """Test Gaussian reconvolution PSF."""
        obs_w_ngmix, obs_d_ngmix, _ = simple_obs_pair["ngmix"]
        obs_w_jax, obs_d_jax, _ = simple_obs_pair["jax"]
        nxy_psf = simple_obs_pair["params"]["nxy_psf"]
        scale = simple_obs_pair["params"]["scale"]

        # Test single PSF
        psf_ngmix = get_galsim_object_from_ngmix_obs_nopix(
            obs_w_ngmix.psf, kind="image"
        )
        psf_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w_jax.psf, kind="image")

        dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)

        kim_size = 173
        reconv_psf_jax = jax_get_gauss_reconv_psf_galsim(
            psf_jax, dk=dk, kim_size=kim_size
        )
        reconv_psf_ngmix = get_gauss_reconv_psf_galsim(
            psf_ngmix, dk=dk, kim_size=kim_size
        )

        # Test PSF properties - relax tolerance for small numerical differences
        assert np.allclose(
            reconv_psf_ngmix.fwhm, reconv_psf_jax.fwhm, rtol=1e-6, atol=1e-10
        ), f"FWHM mismatch: {reconv_psf_ngmix.fwhm} vs {reconv_psf_jax.fwhm}"

        assert np.allclose(
            reconv_psf_ngmix.flux, reconv_psf_jax.flux, rtol=1e-10, atol=1e-12
        ), f"Flux mismatch: {reconv_psf_ngmix.flux} vs {reconv_psf_jax.flux}"

    def test_max_gauss_reconv_psf_consistency(self, simple_obs_pair):
        """Test max Gaussian reconvolution PSF.
        kim_size and dk are not get for Galsim in this case."""
        obs_w_ngmix, obs_d_ngmix, _ = simple_obs_pair["ngmix"]
        obs_w_jax, obs_d_jax, _ = simple_obs_pair["jax"]
        nxy_psf = simple_obs_pair["params"]["nxy_psf"]
        scale = simple_obs_pair["params"]["scale"]

        # Get PSFs
        psf_w_ngmix = get_galsim_object_from_ngmix_obs_nopix(
            obs_w_ngmix.psf, kind="image"
        )
        psf_d_ngmix = get_galsim_object_from_ngmix_obs_nopix(
            obs_d_ngmix.psf, kind="image"
        )
        psf_w_jax = get_jax_galsim_object_from_dfmd_obs_nopix(
            obs_w_jax.psf, kind="image"
        )
        psf_d_jax = get_jax_galsim_object_from_dfmd_obs_nopix(
            obs_d_jax.psf, kind="image"
        )

        # Compare maximum reconvolution PSFs
        max_reconv_psf_ngmix = get_max_gauss_reconv_psf_galsim(psf_w_ngmix, psf_d_ngmix)
        max_reconv_psf_jax = jax_get_max_gauss_reconv_psf_galsim(
            psf_w_jax, psf_d_jax, nxy_psf, scale
        )

        # Test PSF properties
        assert np.allclose(
            max_reconv_psf_ngmix.fwhm, max_reconv_psf_jax.fwhm, rtol=0.02, atol=1e-6
        ), f"Max FWHM err: {max_reconv_psf_ngmix.fwhm} vs {max_reconv_psf_jax.fwhm}"

    def test_metacal_single_shear_consistency(self, simple_obs_pair):
        """Test single shear operations."""
        obs_w_ngmix, _, _ = simple_obs_pair["ngmix"]
        obs_w_jax, _, _ = simple_obs_pair["jax"]
        nxy_psf = simple_obs_pair["params"]["nxy_psf"]
        scale = simple_obs_pair["params"]["scale"]

        # Test single shear transformation
        g1, g2 = 0.01, 0.0

        # Get reconvolution PSFs for both versions
        dk = compute_stepk(pixel_scale=scale, image_size=nxy_psf)
        psf_jax = get_jax_galsim_object_from_dfmd_obs_nopix(obs_w_jax.psf, kind="image")
        psf_ngmix = get_galsim_object_from_ngmix_obs_nopix(
            obs_w_ngmix.psf, kind="image"
        )

        kim_size = 173
        reconv_psf_jax = jax_get_gauss_reconv_psf_galsim(psf_jax, dk, kim_size=kim_size)
        reconv_psf_ngmix = get_gauss_reconv_psf_galsim(
            psf_ngmix, dk=dk, kim_size=kim_size
        )

        # Run metacal operations
        mcal_obs_ngmix = metacal_op_g1g2(obs_w_ngmix, reconv_psf_ngmix, g1, g2)
        mcal_obs_jax = jax_metacal_op_g1g2(obs_w_jax, reconv_psf_jax, g1, g2, nxy_psf)

        # Convert JAX result to ngmix for comparison
        from deep_field_metadetect.jaxify.observation import dfmd_obs_to_ngmix_obs

        mcal_obs_jax_ngmix = dfmd_obs_to_ngmix_obs(mcal_obs_jax)

        # Compare image statistics
        assert np.allclose(
            np.mean(mcal_obs_ngmix.image),
            np.mean(mcal_obs_jax_ngmix.image),
            rtol=1e-5,
            atol=1e-9,
        ), "Image mean mismatch"

        assert np.allclose(
            np.std(mcal_obs_ngmix.image),
            np.std(mcal_obs_jax_ngmix.image),
            rtol=1e-5,
            atol=1e-9,
        ), "Image std mismatch"

    def test_metacal_shears_intermediate_values(self, simple_obs_pair):
        """Test intermediate values in metacal shears operations."""
        obs_w_ngmix, _, _ = simple_obs_pair["ngmix"]
        obs_w_jax, _, _ = simple_obs_pair["jax"]
        scale = simple_obs_pair["params"]["scale"]

        test_shears = ("noshear", "1p", "1m")

        # Run metacal operations
        mcal_res_ngmix = metacal_op_shears(obs_w_ngmix, shears=test_shears)
        mcal_res_jax = jax_metacal_op_shears(obs_w_jax, shears=test_shears, scale=scale)

        # Convert JAX results to ngmix for comparison
        from deep_field_metadetect.jaxify.observation import dfmd_obs_to_ngmix_obs

        mcal_res_jax_ngmix = {}
        for shear in test_shears:
            mcal_res_jax_ngmix[shear] = dfmd_obs_to_ngmix_obs(mcal_res_jax[shear])

        # Compare results for each shear
        for shear in test_shears:
            obs_ngmix = mcal_res_ngmix[shear]
            obs_jax_ngmix = mcal_res_jax_ngmix[shear]

            # Compare image statistics
            img_mean_diff = abs(np.mean(obs_ngmix.image) - np.mean(obs_jax_ngmix.image))
            img_std_ratio = np.std(obs_ngmix.image) / np.std(obs_jax_ngmix.image)

            assert img_mean_diff < 1e-4, (
                f"Shear {shear}: Image mean difference too large: {img_mean_diff}"
            )
            assert 0.99 < img_std_ratio < 1.01, (
                f"Shear {shear}: Image std ratio out of range: {img_std_ratio}"
            )

            # Compare weight statistics
            weight_ratio = np.mean(obs_ngmix.weight) / np.mean(obs_jax_ngmix.weight)
            assert 0.99 < weight_ratio < 1.01, (
                f"Shear {shear}: Weight ratio out of range: {weight_ratio}"
            )
