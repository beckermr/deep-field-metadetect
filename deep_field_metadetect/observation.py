from typing import NamedTuple, Optional
import numpy as np

import ngmix
from ngmix.jacobian import Jacobian

import jax
import jax_galsim

from ngmix.observation import Observation

@jax.tree_util.register_pytree_node_class
class NTObservation(NamedTuple):
    image: jax.Array
    weight: Optional[jax.Array]
    bmask: Optional[jax.Array]
    ormask: Optional[jax.Array]
    noise: Optional[jax.Array]
    jacobian: Optional[jax.Array]
    psf: Optional["NTObservation"]
    mfrac: Optional[jax.Array]
    jac_row0: Optional[float]
    jac_col0: Optional[float]
    jac_det: Optional[float]
    jac_scale: Optional[float]
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
            self.jacobian, 
            self.psf, 
            self.mfrac, 
            self.jac_row0, 
            self.jac_col0, 
            self.jac_det, 
            self.jac_scale,
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



def ngmix_Obs_to_NT(obs: ngmix.observation.Observation) -> NTObservation:
    jacobian = obs.get_jacobian()

    psf=None
    if obs.has_psf():
        psf = ngmix_Obs_to_NT(obs.get_psf())

    return NTObservation(
        image=jax.numpy.array(obs.image), 
        weight=jax.numpy.array(obs.weight),
        bmask=jax.numpy.array(obs.bmask) if obs.has_bmask() else None,
        ormask=jax.numpy.array(obs.ormask) if obs.has_ormask() else None,
        noise=jax.numpy.array(obs.noise) if obs.has_noise() else None,
        jacobian=jax_galsim.BaseWCS().from_galsim(jacobian.get_galsim_wcs()),
        psf=psf,
        meta=obs.meta,  # Directly copy metadata
        mfrac=jax.numpy.array(obs.mfrac) if obs.has_mfrac() else None,
        store_pixels=getattr(obs, "store_pixels", True), 
        ignore_zero_weight=getattr(obs, "ignore_zero_weight", True), 
        jac_row0=jacobian.row0,
        jac_col0=jacobian.col0,
        jac_det=jacobian.det,
        jac_scale=jacobian.scale,
    )
  
def NT_to_ngmix_obs(nt_obs) -> Observation:
    psf= None
    if nt_obs.psf is not None:
        psf= NT_to_ngmix_obs(nt_obs.psf)
    return Observation(
        image=np.array(nt_obs.image), 
        weight=np.array(nt_obs.weight), 
        bmask=nt_obs.bmask, 
        ormask=nt_obs.ormask,
        noise=nt_obs.noise if nt_obs.noise is None else np.array(nt_obs.noise), 
        jacobian=Jacobian(
            row=nt_obs.jac_row0,
            col=nt_obs.jac_col0,
            dudrow=nt_obs.jacobian.dudx,
            dudcol=nt_obs.jacobian.dudy,
            dvdrow=nt_obs.jacobian.dvdx,
            dvdcol=nt_obs.jacobian.dvdy,
            det=nt_obs.jac_det,
            scale=nt_obs.jac_scale,
        ), 
        psf=psf, 
        mfrac=nt_obs.mfrac if nt_obs.mfrac is None else np.array(nt_obs.mfrac), 
        meta=nt_obs.meta, 
        store_pixels=np.array(nt_obs.store_pixels, dtype=np.bool_),
        ignore_zero_weight=np.array(nt_obs.ignore_zero_weight, dtype=np.bool_),
    )

    