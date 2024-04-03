import ngmix
import numpy as np
import pytest

from deep_metadetection.detect import (
    BMASK_EDGE,
    generate_mbobs_for_detections,
    make_detection_coadd,
    run_detection_sep,
)
from deep_metadetection.utils import make_simple_sim


@pytest.mark.parametrize("has_bmask", [True, False])
@pytest.mark.parametrize(
    "detbands",
    [
        None,
        [True, True, True],
        [True, False, True],
        [False, True, False],
    ],
)
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
                obs.bmask = rng.choice(
                    [0, 2**0, 2**5], size=obs.image.shape, p=[0.8, 0.1, 0.1]
                ).astype(np.int32)
                assert np.any(obs.bmask != 0)
            else:
                obs.bmask = None

            obs.weight = obs.weight * rng.choice(
                [0, 1], size=obs.image.shape, p=[0.1, 0.9]
            )
            assert np.any(obs.weight == 0)

            obslist.append(obs)

        tot_mbobs.append(obslist)

    detobs = make_detection_coadd(tot_mbobs, detbands=detbands)

    if False:
        import proplot as pplt

        fig, axs = pplt.subplots(
            nrows=1, ncols=n_bands + 1, figsize=(3 * (n_bands + 1), 3)
        )
        for i in range(n_bands):
            axs[i].imshow(
                np.arcsinh(tot_mbobs[i][0].image * np.sqrt(tot_mbobs[i][0].weight))
            )
            axs[i].format(title=f"band {i}", grid=False)
        i = n_bands
        axs[i].imshow(np.arcsinh(detobs.image * np.sqrt(detobs.weight)))
        axs[i].format(title="coadd", grid=False)
        fig.show()
        import pdb

        pdb.set_trace()

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
    expected_var = (
        np.sum([wgt**2 * var for var, wgt in zip(all_vars, wgts)], axis=0)
        / np.sum(wgts) ** 2
    )
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


def test_run_detection_sep():
    obs, *_ = make_simple_sim(
        seed=10,
        obj_flux_factor=1,
        s2n=100,
        n_objs=10,
        dim=100,
        buff=20,
    )

    detdata = run_detection_sep(obs)
    cat = detdata["catalog"]
    assert cat.shape[0] > 0

    for i, col in enumerate(["y", "x"]):
        assert np.all(cat[col] >= 0), f"{col} {cat[col]} too small"
        assert np.all(cat[col] <= obs.image.shape[i]), f"{col} {cat[col]} too big"

    seg = detdata["segmap"]
    assert np.all(seg >= 0)
    assert np.max(seg) == cat.shape[0]


def test_run_detection_sep_bmask():
    obs, *_ = make_simple_sim(
        seed=10,
        obj_flux_factor=1,
        s2n=100,
        n_objs=10,
        dim=100,
        buff=20,
    )

    bmask = np.zeros_like(obs.image, dtype=np.int32)
    bmask[:, 60:] = 2**1
    bmask[:40, :] = 2**12
    obs.bmask = bmask
    detdata = run_detection_sep(obs, nodet_flags=2**1)
    cat = detdata["catalog"]

    if False:
        import proplot as pplt

        fig, axs = pplt.subplots(nrows=1, ncols=1, figsize=(3, 3))
        axs.imshow(np.arcsinh(obs.image * np.sqrt(obs.weight)))
        axs.plot(cat["x"], cat["y"], "b.")
        fig.show()
        import pdb

        pdb.set_trace()

    assert cat.shape[0] > 0

    for i, col in enumerate(["y", "x"]):
        assert np.all(cat[col] >= 0), f"{col} {cat[col]} too small"
        assert np.all(cat[col] <= obs.image.shape[i]), f"{col} {cat[col]} too big"

    assert np.all(cat["xpeak"] < 65)
    assert np.any(cat["ypeak"] < 40)

    seg = detdata["segmap"]
    assert np.all(seg >= 0)
    assert np.max(seg) == cat.shape[0]


@pytest.mark.parametrize("has_bmask", [True, False])
def test_generate_mbobs_for_detections(has_bmask):
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
                n_objs=100,
                dim=100,
                buff=0,
            )
            if has_bmask:
                obs.bmask = rng.choice(
                    [0, 2**0, 2**5], size=obs.image.shape, p=[0.8, 0.1, 0.1]
                ).astype(np.int32)
                assert np.any(obs.bmask != 0)
            else:
                obs.set_bmask(None)

            obs.weight = obs.weight * rng.choice(
                [0, 1], size=obs.image.shape, p=[0.1, 0.9]
            )
            assert np.any(obs.weight == 0)

            obs.mfrac = rng.uniform(0.0, 0.1, size=obs.image.shape)

            obslist.append(obs)

        tot_mbobs.append(obslist)

    xs = []
    ys = []
    for xv in [13, 50, 83]:
        for yv in [11, 51, 81]:
            xs.append(xv)
            ys.append(yv)
    xs = np.concatenate(
        [
            np.array(xs),
            rng.uniform(0, 100, size=10),
        ]
    )
    ys = np.concatenate(
        [
            np.array(ys),
            rng.uniform(0, 100, size=10),
        ]
    )

    bs = 32
    bs_2 = bs // 2
    bshape = (bs, bs)
    for obs_data, _mbobs in generate_mbobs_for_detections(
        tot_mbobs, xs, ys, box_size=bs
    ):
        i = obs_data["id"]
        x = obs_data["x"]
        y = obs_data["y"]
        ix = int(x)
        iy = int(y)
        assert xs[i] == x
        assert ys[i] == y
        assert len(_mbobs) == n_bands
        for band, obsl in enumerate(_mbobs):
            assert len(obsl) == n_obs[band]
            for obsind, obs in enumerate(obsl):
                assert obs.image.shape == bshape
                assert obs.bmask.shape == bshape
                assert obs.weight.shape == bshape
                assert obs.mfrac.shape == bshape

                if x <= bs_2 or x >= 100 - bs_2 or y <= bs_2 or y >= 100 - bs_2:
                    assert np.any(obs.bmask & BMASK_EDGE != 0)
                    assert np.any(obs.mfrac == 1.0)
                else:
                    assert np.all(obs.bmask & BMASK_EDGE == 0)
                    assert np.all(obs.mfrac < 0.1)

                    start_x = ix - bs_2 + 1
                    start_y = iy - bs_2 + 1
                    assert np.array_equal(
                        obs.image,
                        tot_mbobs[band][obsind].image[
                            start_y : start_y + bs, start_x : start_x + bs
                        ],
                    )
                    if has_bmask:
                        assert np.array_equal(
                            obs.bmask,
                            tot_mbobs[band][obsind].bmask[
                                start_y : start_y + bs, start_x : start_x + bs
                            ],
                        )
                    else:
                        assert np.all(obs.bmask == 0)
                    assert np.array_equal(
                        obs.weight,
                        tot_mbobs[band][obsind].weight[
                            start_y : start_y + bs, start_x : start_x + bs
                        ],
                    )
                    assert np.array_equal(
                        obs.mfrac,
                        tot_mbobs[band][obsind].mfrac[
                            start_y : start_y + bs, start_x : start_x + bs
                        ],
                    )