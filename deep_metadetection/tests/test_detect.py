import numpy as np
import pytest

import ngmix

from deep_metadetection.detect import make_detection_coadd
from deep_metadetection.utils import make_simple_sim


@pytest.mark.parametrize('has_bmask', [True, False])
@pytest.mark.parametrize('detbands', [
    None,
    [True, True, True],
    [True, False, True],
    [False, True, False],
])
def test_make_detection_coadd(detbands, has_bmask):
    seed = 10
    rng = np.random.RandomState(seed)

    n_bands = 3
    fluxes = rng.uniform(0.2, 1, size=n_bands)
    n_obs = rng.randint(1, 3, size=3)

    tot_mbobs = ngmix.MultiBandObsList()
    for band in range(n_bands):
        obslist = ngmix.ObsList()
        for i in range(n_obs[band]):
            obs, *_ = make_simple_sim(
                seed=seed,
                obj_flux_factor=fluxes[band],
                s2n=100,
                n_objs=10,
                dim=100,
                buff=20,
            )
            if has_bmask:
                obs.bmask = rng.choice([0, 2**0, 2**5], size=obs.image.shape, p=[0.8, 0.1, 0.1]).astype(np.int32)
                assert np.any(obs.bmask != 0)
            else:
                obs.bmask = None

            obs.weight = obs.weight * rng.choice([0, 1], size=obs.image.shape, p=[0.1, 0.9])
            assert np.any(obs.weight == 0)

            obslist.append(obs)

        tot_mbobs.append(obslist)

    detobs = make_detection_coadd(tot_mbobs, detbands=detbands)

    if detbands is None:
        detbands = [True] * n_bands

    wgts = []
    all_obs = []
    for band in range(n_bands):
        if not detbands[band]:
            continue

        for obs in tot_mbobs[band]:
            wgts.append(np.median(obs.weight[obs.weight > 0]))
            all_obs.append(obs)
    wgts = np.array(wgts)
    wgts /= wgts.sum()

    assert np.all(np.isfinite(detobs.image))
    assert np.all(np.isfinite(detobs.weight))
    assert np.all(np.isfinite(detobs.bmask))

    expected_im = np.sum([obs.image * wgt for obs, wgt in zip(all_obs, wgts)], axis=0)
    assert np.allclose(detobs.image, expected_im)

    all_vars = [obs.weight.copy() for obs in all_obs]
    for i in range(len(all_vars)):
        msk = all_vars[i] > 0
        all_vars[i][msk] = 1.0 / all_vars[i][msk]
        all_vars[i][~msk] = np.inf
    expected_var = np.sum([wgt**2 * var for var, wgt in zip(all_vars, wgts)], axis=0) / np.sum(wgts)**2
    assert np.allclose(detobs.weight, 1.0 / expected_var)
    assert np.any(detobs.weight == 0)

    if has_bmask:
        expected_bmask = np.zeros_like(detobs.image, dtype=np.int32)
        for obs in all_obs:
            expected_bmask |= obs.bmask
        assert np.array_equal(detobs.bmask, expected_bmask)
        assert np.any(detobs.bmask != 0)
    else:
        assert np.all(detobs.bmask == 0)

    if False:
        import proplot as pplt
        fig, axs = pplt.subplots(nrows=1, ncols=n_bands+1, figsize=(3 * (n_bands + 1), 3))
        for i in range(n_bands):
            axs[i].imshow(np.arcsinh(tot_mbobs[i][0].image * np.sqrt(tot_mbobs[i][0].weight)))
            axs[i].format(title=f'band {i}', grid=False)
        i = n_bands
        axs[i].imshow(np.arcsinh(detobs.image * np.sqrt(detobs.weight)))
        axs[i].format(title='coadd', grid=False)
        fig.show()
        import pdb; pdb.set_trace()
