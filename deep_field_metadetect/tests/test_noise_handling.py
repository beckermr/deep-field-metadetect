import numpy as np
from esutil.pbar import PBar

from deep_field_metadetect.metacal import (
    metacal_op_g1g2,
    metacal_wide_and_deep_psf_matched,
)
from deep_field_metadetect.pixel_cov import meas_pixel_cov
from deep_field_metadetect.utils import make_simple_sim


def _simple_noise_sim(seed):
    obs_wide, obs_deep, obs_deep_noise = make_simple_sim(
        seed=seed,
        g1=0.0,
        g2=0.0,
        s2n=10,
        deep_noise_fac=1.0 / np.sqrt(10),
        deep_psf_fac=1.0,
        obj_flux_factor=0,
    )

    mcal_res = metacal_wide_and_deep_psf_matched(
        obs_wide,
        obs_deep,
        obs_deep_noise,
        shears=["noshear"],
        return_noshear_deep=True,
    )

    mwide_mcal = metacal_op_g1g2(obs_wide, mcal_res["noshear"].psf.galsim_obj, 0, 0)

    return {
        "noshear": mcal_res["noshear"],
        "noshear_deep": mcal_res["noshear_deep"],
        "mcal_wide": mwide_mcal,
    }


def test_noise_handling():
    rng = np.random.RandomState(seed=10)
    seeds = rng.randint(0, 2**30, size=1000) + 1
    print(flush=True)
    covs = {}
    for seed in PBar(seeds, desc="running noise sims"):
        res = _simple_noise_sim(seed)
        for k, v in res.items():
            if k not in covs:
                covs[k] = []
            covs[k].append(
                meas_pixel_cov(v.image.copy(), np.ones_like(v.image).astype(bool))
            )

    for k in covs:
        covs[k] = np.mean(covs[k], axis=0)

    for k, v in covs.items():
        print("%s:\n" % k, v, flush=True)

    assert covs["mcal_wide"][1, 1] > covs["noshear"][1, 1] * 1.5
    assert np.allclose(covs["noshear"], covs["noshear_deep"], rtol=0, atol=7e-4)
