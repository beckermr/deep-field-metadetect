"""Test vmapping jax_multi_band_deep_field_metadetect_jitted."""

import jax
import jax.numpy as np

from deep_field_metadetect.jaxify import jax_dfmd_defaults
from deep_field_metadetect.jaxify.jax_metadetect import (
    jax_multi_band_deep_field_metadetect_jitted,
)
from deep_field_metadetect.jaxify.jax_sims import (
    generate_jax_galsim_multiband_sim_observations_jitted,
)
from deep_field_metadetect.metacal import DEFAULT_SHEARS


def test_vmap_metadetect():
    """Test vmapping jax_multi_band_deep_field_metadetect_jitted."""
    n_fields = 2
    bands = ("g", "r", "i")
    n_bands = len(bands)
    dim = 53
    nxy = dim
    nxy_psf = dim
    n_objs = 5

    # Generate independent observations for each field
    keys = jax.random.split(jax.random.PRNGKey(42), n_fields)

    # Generate observations for each field
    obs_list = []
    for i, key in enumerate(keys):
        mb_obs_wide, mb_obs_deep, mb_obs_deep_noise = (
            generate_jax_galsim_multiband_sim_observations_jitted(
                key,
                bands=bands,
                max_n_objs=n_objs,
                dim=dim,
                dim_psf=dim,
            )
        )
        obs_list.append((mb_obs_wide, mb_obs_deep, mb_obs_deep_noise))

    # Stack observations into batched structure
    mb_obs_wide_batched = jax.tree_util.tree_map(
        lambda *xs: np.stack(xs, axis=0), *[obs[0] for obs in obs_list]
    )
    mb_obs_deep_batched = jax.tree_util.tree_map(
        lambda *xs: np.stack(xs, axis=0), *[obs[1] for obs in obs_list]
    )
    mb_obs_deep_noise_batched = jax.tree_util.tree_map(
        lambda *xs: np.stack(xs, axis=0), *[obs[2] for obs in obs_list]
    )

    # Verify batch structure
    sample_obs = mb_obs_wide_batched[0][0]
    assert sample_obs.image.shape == (n_fields, dim, dim), "Incorrect batch shape!"

    # Create vmapped metadetect function
    vmapped_metadetect = jax.vmap(
        jax_multi_band_deep_field_metadetect_jitted,
        in_axes=(0, 0, 0, None, None, None, None),
    )

    result_batched = vmapped_metadetect(
        mb_obs_wide_batched,
        mb_obs_deep_batched,
        mb_obs_deep_noise_batched,
        nxy,
        nxy_psf,
        n_bands,
        None,  # detband_indices (use all bands)
    )

    dfmdet_res = result_batched["dfmdet_res"]

    # Calculate expected dimensions
    max_objects = jax_dfmd_defaults.MAX_OBJECTS
    n_shears = len(DEFAULT_SHEARS)
    expected_detections = max_objects * n_shears * n_bands

    assert dfmdet_res["wmom_s2n"].shape[0] == n_fields, "Incorrect batch dimension"
    assert dfmdet_res["wmom_s2n"].shape[1] == expected_detections, (
        f"Incorrect detection dimension: expected {expected_detections} "
        f"(max_objects={max_objects} * n_shears={n_shears} * n_bands={n_bands})"
    )
