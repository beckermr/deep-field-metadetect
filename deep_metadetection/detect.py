import ngmix
import numpy as np
import sxdes

BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    "image": 0.0,
    "weight": 0.0,
    "seg": 0,
    "bmask": BMASK_EDGE,
    "noise": 0.0,
    "mfrac": 0.0,
}


def make_detection_coadd(mbobs, detbands=None):
    detim = np.zeros_like(mbobs[0][0].image)
    detvar = np.zeros_like(mbobs[0][0].image)
    mask = np.zeros(detim.shape, dtype=np.int32)

    if detbands is None:
        detbands = [True] * len(mbobs)

    weights = []
    for i, obslist in enumerate(mbobs):
        if not detbands[i]:
            continue
        for obs in obslist:
            weights.append(np.median(obs.weight[obs.weight > 0]))

    weights = np.array(weights)
    wsum = weights.sum()
    weights /= wsum

    loc = 0
    for i, obslist in enumerate(mbobs):
        if not detbands[i]:
            continue
        for obs in obslist:
            detim += obs.image * weights[loc]
            var = obs.weight.copy()
            msk = var > 0
            var[msk] = 1.0 / var[msk]
            var[~msk] = np.inf
            detvar += weights[loc] ** 2 * var
            if obs.has_bmask():
                mask |= obs.bmask

            loc += 1

    msk = np.isfinite(detvar)
    wgt = np.zeros_like(detvar)
    wgt[msk] = 1.0 / detvar[msk]

    return ngmix.Observation(
        image=detim,
        weight=wgt,
        bmask=mask.astype(np.int32),
    )


def run_detection_sep(
    detobs,
    sep_config=None,
    detect_thresh=None,
    nodet_flags=0,
):
    if sep_config is None:
        sep_config = sxdes.SX_CONFIG
    if detect_thresh is None:
        detect_thresh = sxdes.DETECT_THRESH

    objs, seg = sxdes.run_sep(
        image=detobs.image.copy(),
        noise=1.0 / np.sqrt(np.median(detobs.weight[detobs.weight > 0])),
        mask=detobs.bmask & nodet_flags != 0 if detobs.has_bmask() else None,
        config=sep_config,
        thresh=detect_thresh,
    )

    return {
        "catalog": objs,
        "segmap": seg,
    }


def _get_subboxes(start, end, box_size, max_size):
    assert end - start == box_size
    orig_box = [start, end]
    sub_box = [0, box_size]

    if start < 0:
        sub_box[0] = -start
        orig_box[0] = 0

    if end > max_size:
        sub_box[1] = box_size - (end - max_size)
        orig_box[1] = max_size

    return orig_box, sub_box


def _get_subobs(obs, x, y, start_x, start_y, end_x, end_y, box_size):
    max_y, max_x = obs.image.shape
    orig_x_box, sub_x_box = _get_subboxes(start_x, end_x, box_size, max_x)
    orig_y_box, sub_y_box = _get_subboxes(start_y, end_y, box_size, max_y)

    kwargs = {}
    kwargs["jacobian"] = ngmix.Jacobian(
        row=y - start_y,
        col=x - start_x,
        dvdrow=obs.jacobian.dvdrow,
        dvdcol=obs.jacobian.dvdcol,
        dudrow=obs.jacobian.dudrow,
        dudcol=obs.jacobian.dudcol,
    )
    if obs.has_psf():
        kwargs["psf"] = obs.psf

    for key in ["image", "bmask", "noise", "mfrac", "weight"]:
        subim = None
        if key == "image":
            subim = np.zeros((box_size, box_size), dtype=obs.image.dtype)
            subim += DEFAULT_IMAGE_VALUES[key]
            subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = obs.image[
                orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]
            ]
        elif key == "bmask":
            if obs.has_bmask():
                subim = np.zeros((box_size, box_size), dtype=obs.bmask.dtype)
            else:
                subim = np.zeros((box_size, box_size), dtype=np.int32)

            subim += DEFAULT_IMAGE_VALUES[key]

            if obs.has_bmask():
                subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = (
                    obs.bmask[
                        orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]
                    ]
                )
            else:
                subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = 0
        elif key == "noise" and obs.has_noise():
            subim = np.zeros((box_size, box_size), dtype=obs.noise.dtype)
            subim += DEFAULT_IMAGE_VALUES[key]
            subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = obs.noise[
                orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]
            ]
        elif key == "weight":
            subim = np.zeros((box_size, box_size), dtype=obs.noise.dtype)
            subim += DEFAULT_IMAGE_VALUES[key]
            subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = (
                obs.weight[orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]]
            )
        elif key == "mfrac" and obs.has_mfrac():
            subim = np.zeros((box_size, box_size), dtype=obs.mfrac.dtype)
            subim += DEFAULT_IMAGE_VALUES[key]
            subim[sub_y_box[0] : sub_y_box[1], sub_x_box[0] : sub_x_box[1]] = obs.mfrac[
                orig_y_box[0] : orig_y_box[1], orig_x_box[0] : orig_x_box[1]
            ]

        if subim is not None:
            kwargs[key] = subim

    msk = kwargs["bmask"] & BMASK_EDGE != 0
    kwargs["mfrac"][msk] = 1.0

    return ngmix.Observation(**kwargs)


def generate_mbobs_for_detections(
    mbobs,
    xs,
    ys,
    box_size=48,
    ids=None,
):
    half_box_size = box_size // 2

    for i, (x, y) in enumerate(zip(xs, ys)):
        ix = int(x)
        iy = int(y)

        start_x = ix - half_box_size + 1
        start_y = iy - half_box_size + 1
        end_x = ix + half_box_size + 1  # plus one for slices
        end_y = iy + half_box_size + 1

        _mbobs = ngmix.MultiBandObsList()
        for obslist in mbobs:
            _obslist = ngmix.ObsList()
            _mbobs.append(_obslist)
            for obs in obslist:
                _obslist.append(
                    _get_subobs(obs, x, y, start_x, start_y, end_x, end_y, box_size)
                )

        yield (
            {
                "id": ids[i] if ids is not None else i,
                "x": x,
                "y": y,
            },
            _mbobs,
        )
