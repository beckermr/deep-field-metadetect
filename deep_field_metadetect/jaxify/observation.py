from typing import NamedTuple, Optional

import jax
import jax_galsim
import ngmix
import numpy as np
from ngmix.observation import Observation


@jax.tree_util.register_pytree_node_class
class DFMdetObservation(NamedTuple):
    image: jax.Array
    weight: Optional[jax.Array]
    bmask: Optional[jax.Array]
    ormask: Optional[jax.Array]
    noise: Optional[jax.Array]
    wcs: Optional[jax_galsim.wcs.AffineTransform]
    psf: Optional["DFMdetObservation"]
    mfrac: Optional[jax.Array]
    meta: Optional[dict]
    store_pixels: bool
    ignore_zero_weight: bool

    def tree_flatten(self):
        children = (
            self.image,
            self.weight,
            self.bmask,
            self.ormask,
            self.noise,
            self.wcs,
            self.psf,
            self.mfrac,
        )

        aux_data = (self.meta, self.store_pixels, self.ignore_zero_weight)

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct the object from flattened data
        return cls(*children, *aux_data)

    def has_bmask(self) -> bool:
        if self.bmask is None:
            return False
        return True

    def has_mfrac(self) -> bool:
        if self.bmask is None:
            return False
        return True

    def has_noise(self) -> bool:
        if self.noise is None:
            return False
        return True

    def has_ormask(self) -> bool:
        if self.ormask is None:
            return False
        return True

    def has_psf(self) -> bool:
        if self.psf is None:
            return False
        return True


def ngmix_obs_to_dfmd_obs(obs: ngmix.observation.Observation) -> DFMdetObservation:
    jacobian = obs.get_jacobian()

    psf = None
    if obs.has_psf():
        psf = ngmix_obs_to_dfmd_obs(obs.get_psf())

    return DFMdetObservation(
        image=obs.image,
        weight=obs.weight,
        bmask=obs.bmask if obs.has_bmask() else None,
        ormask=obs.ormask if obs.has_ormask() else None,
        noise=obs.noise if obs.has_noise() else None,
        wcs=jax_galsim.wcs.AffineTransform(
            dudx=jacobian.dudcol,
            dudy=jacobian.dudrow,
            dvdx=jacobian.dvdcol,
            dvdy=jacobian.dvdrow,
            origin=jax_galsim.PositionD(
                y=jacobian.row0 + 1,
                x=jacobian.col0 + 1,
            ),
        ),
        psf=psf,
        meta=obs.meta,
        mfrac=obs.mfrac if obs.has_mfrac() else None,
        store_pixels=getattr(obs, "store_pixels", True),
        ignore_zero_weight=getattr(obs, "ignore_zero_weight", True),
    )


def dfmd_obs_to_ngmix_obs(dfmd_obs) -> Observation:
    psf = None
    if dfmd_obs.psf is not None:
        psf = dfmd_obs_to_ngmix_obs(dfmd_obs.psf)
    return Observation(
        image=np.array(dfmd_obs.image),
        weight=np.array(dfmd_obs.weight),
        bmask=dfmd_obs.bmask,
        ormask=dfmd_obs.ormask,
        noise=dfmd_obs.noise if dfmd_obs.noise is None else np.array(dfmd_obs.noise),
        jacobian=ngmix.jacobian.Jacobian(
            row=dfmd_obs.wcs.origin.y - 1,
            col=dfmd_obs.wcs.origin.x - 1,
            dudcol=dfmd_obs.wcs.dudx,
            dudrow=dfmd_obs.wcs.dudy,
            dvdcol=dfmd_obs.wcs.dvdx,
            dvdrow=dfmd_obs.wcs.dvdy,
        ),
        psf=psf,
        mfrac=dfmd_obs.mfrac if dfmd_obs.mfrac is None else np.array(dfmd_obs.mfrac),
        meta=dfmd_obs.meta,
        store_pixels=np.array(dfmd_obs.store_pixels, dtype=np.bool_),
        ignore_zero_weight=np.array(dfmd_obs.ignore_zero_weight, dtype=np.bool_),
    )
