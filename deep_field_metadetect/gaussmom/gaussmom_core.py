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
    wgt: float
    pixelwise_wgt: jax.Array

    def tree_flatten(self):
        return (
            self.u,
            self.v,
            self.image,
            self.area,
            self.wgt,
            self.pixelwise_wgt,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class GaussMomData(NamedTuple):
    obs: list[GaussMomObs]
    sums: jnp.ndarray
    sums_cov: jnp.ndarray
    pars: jnp.ndarray
    wsum: float = 0
    flags: int = 0
    flagstr: str = ""
    flux: float = 0
    sums_norm: jnp.ndarray = jnp.nan
    flux_flags: int = 0
    flux_flagstr: str = ""
    T_flags: int = 0
    T_flagstr: str = ""
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

    def tree_flatten(self):
        return (
            self.obs,
            self.sums,
            self.sums_cov,
            self.pars,
            self.wsum,
            self.flags,
            self.flagstr,
            self.flux,
            self.sums_norm,
            self.flux_flags,
            self.flux_flagstr,
            self.T_flags,
            self.T_flagstr,
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
        jnp.median(obs.weight),
        obs.weight,
    )


# def obs_to_ngmix_obs(obs: GaussMomObs) -> ngmix.Observation:
#     return ngmix.Observation(
#         np.array(obs.image),
#         weight=np.array(obs.weight),
#         bmask=np.zeros_like(obs.image, dtype=int),
#         jacobian=ngmix.Jacobian(
#             x=obs.cen_x,
#             y=obs.cen_y,
#             dudx=obs.dudx,
#             dudy=obs.dudy,
#             dvdx=obs.dvdx,
#             dvdy=obs.dvdy,
#         ),
#         psf=None if obs.psf is None else obs_to_ngmix_obs(obs.psf),
#     )


@jax.jit
def _eval_gauss2d(pars, u, v, area):
    cen_v, cen_u, irr, irc, icc = pars[0:5]

    det = irr * icc - irc * irc
    idet = 1.0 / det
    drr = irr * idet
    drc = irc * idet
    dcc = icc * idet
    norm = 1.0 / (2 * jnp.pi * jnp.sqrt(det))

    # v->row, u->col in gauss
    vdiff = v - cen_v
    udiff = u - cen_u
    chi2 = dcc * vdiff * vdiff + drr * udiff * udiff - 2.0 * drc * vdiff * udiff

    return norm * jnp.exp(-0.5 * chi2) * area, norm
