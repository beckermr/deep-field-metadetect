from typing import NamedTuple

import jax
import jax.numpy as jnp
import ngmix


@jax.tree_util.register_pytree_node_class
class GaussMomObs(NamedTuple):
    u: jax.Array
    v: jax.Array
    image: jax.Array
    area: float
    pixelwise_wgt: jax.Array

    def tree_flatten(self):
        return (
            self.u,
            self.v,
            self.image,
            self.area,
            self.pixelwise_wgt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class GaussMomData(NamedTuple):
    obs: list[GaussMomObs]
    npix: int
    sums: jnp.ndarray
    sums_cov: jnp.ndarray
    pars: jnp.ndarray
    wsum: float = 0
    flags: int = 0
    flux: float = 0
    sums_norm: jnp.ndarray = jnp.nan
    flux_flags: int = 0
    T_flags: int = 0
    flux_err: float = jnp.nan
    T: float = jnp.nan
    T_err: float = jnp.nan
    s2n: float = jnp.nan
    e1: float = jnp.nan
    e2: float = jnp.nan
    e: jnp.ndarray = jnp.array([jnp.nan, jnp.nan])
    e_err: jnp.ndarray = jnp.array([jnp.nan, jnp.nan])
    e_cov: jnp.ndarray = jnp.diag(jnp.array([jnp.nan, jnp.nan]))
    sums_err: jnp.ndarray = jnp.array([jnp.nan] * 6)

    Mv: float = jnp.nan
    Mu: float = jnp.nan
    M1: float = jnp.nan
    M2: float = jnp.nan
    MT: float = jnp.nan
    MF: float = jnp.nan

    # these are same as above but with the alternative notation
    M00: float = jnp.nan
    M10: float = jnp.nan
    M01: float = jnp.nan
    M11: float = jnp.nan
    M20: float = jnp.nan
    M02: float = jnp.nan

    # Now the errors
    Mv_err: float = jnp.nan
    Mu_err: float = jnp.nan
    M1_err: float = jnp.nan
    M2_err: float = jnp.nan
    MT_err: float = jnp.nan
    MF_err: float = jnp.nan

    # these are same as above but with the alternative notation
    M00_err: float = jnp.nan
    M10_err: float = jnp.nan
    M01_err: float = jnp.nan
    M11_err: float = jnp.nan
    M20_err: float = jnp.nan
    M02_err: float = jnp.nan

    def tree_flatten(self):
        return (
            self.obs,
            self.npix,
            self.sums,
            self.sums_cov,
            self.pars,
            self.wsum,
            self.flags,
            self.flux,
            self.sums_norm,
            self.flux_flags,
            self.T_flags,
            self.flux_err,
            self.T,
            self.T_err,
            self.s2n,
            self.e1,
            self.e2,
            self.e,
            self.e_err,
            self.e_cov,
            self.sums_err,
            self.Mv,
            self.Mu,
            self.M1,
            self.M2,
            self.MT,
            self.MF,
            self.Mv_err,
            self.Mu_err,
            self.M1_err,
            self.M2_err,
            self.MT_err,
            self.MF_err,
            self.M00,
            self.M10,
            self.M01,
            self.M11,
            self.M20,
            self.M02,
            self.M00_err,
            self.M10_err,
            self.M01_err,
            self.M11_err,
            self.M20_err,
            self.M02_err,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def obs_to_gaussmom_obs(obs: ngmix.Observation) -> GaussMomObs:
    x, y = jnp.meshgrid(
        jnp.arange(obs.image.shape[1], dtype=float),
        jnp.arange(obs.image.shape[0], dtype=float),
    )
    dx = x - obs.jacobian.col0
    dy = y - obs.jacobian.row0
    u = obs.jacobian.dudcol * dx + obs.jacobian.dudrow * dy
    v = obs.jacobian.dvdcol * dx + obs.jacobian.dvdrow * dy

    return GaussMomObs(
        u,
        v,
        obs.image,
        obs.jacobian.dudcol * obs.jacobian.dvdrow
        - obs.jacobian.dudrow * obs.jacobian.dvdcol,
        obs.weight,
    )
