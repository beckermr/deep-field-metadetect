import jax
import jax.numpy as jnp
import jax_galsim
import ngmix
import numpy as np
from ngmix.observation import Observation


@jax.tree_util.register_pytree_node_class
class DFMdetPSF:
    def __init__(
        self,
        image,
        weight=None,
        wcs=None,
        meta=None,
        store_pixels=True,
        ignore_zero_weight=True,
    ):
        if meta is None:
            meta = {}

        if wcs is None:
            wcs = jax_galsim.wcs.AffineTransform(
                dudx=1.0,
                dudy=0.0,
                dvdx=0.0,
                dvdy=1.0,
                origin=jax_galsim.PositionD(
                    y=(image.shape[0] + 1) / 2,
                    x=(image.shape[1] + 1) / 2,
                ),
            )

        self.image = image
        if weight is None:
            weight = jnp.ones_like(self.image, dtype=jnp.float32)
        self.weight = weight
        self.wcs = wcs
        self.meta = meta
        self.store_pixels = store_pixels
        self.ignore_zero_weight = ignore_zero_weight

    def tree_flatten(self):
        children = (self.image, self.weight)
        aux_data = (self.wcs, self.meta, self.store_pixels, self.ignore_zero_weight)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            image=children[0],
            weight=children[1],
            wcs=aux_data[0],
            meta=aux_data[1],
            store_pixels=aux_data[2],
            ignore_zero_weight=aux_data[3],
        )

    def has_bmask(self):
        return False

    def has_mfrac(self):
        return False

    def has_noise(self):
        return False

    def has_ormask(self):
        return False

    def has_psf(self):
        return False

    @jax.jit
    def replace(self, **kwargs):
        """Create a new instance similar to NamedTuple._replace"""
        new_kwargs = {
            "image": self.image,
            "wcs": self.wcs,
            "meta": self.meta,
            "store_pixels": self.store_pixels,
            "ignore_zero_weight": self.ignore_zero_weight,
        }
        new_kwargs.update(kwargs)
        return DFMdetPSF(**new_kwargs)


@jax.tree_util.register_pytree_node_class
class DFMdetObservation:
    def __init__(
        self,
        image,
        weight=None,
        bmask=None,
        ormask=None,
        noise=None,
        wcs=None,
        psf=None,
        mfrac=None,
        meta=None,
        store_pixels=True,
        ignore_zero_weight=True,
    ):
        image = image
        if weight is None:
            weight = jnp.ones_like(image, dtype=jnp.float32)
        if bmask is None:
            bmask = jnp.zeros_like(image, dtype=jnp.int32)
        if ormask is None:
            ormask = jnp.zeros_like(image, dtype=jnp.int32)
        if noise is None:
            noise = jnp.zeros_like(image, dtype=jnp.float32)
        if mfrac is None:
            mfrac = jnp.zeros_like(image, dtype=jnp.float32)
        if meta is None:
            meta = {}

        if psf is None:
            psf = DFMdetPSF(image=jnp.zeros_like(image, dtype=jnp.float32))

        if wcs is None:
            wcs = jax_galsim.wcs.AffineTransform(
                dudx=1.0,
                dudy=0.0,
                dvdx=0.0,
                dvdy=1.0,
                origin=jax_galsim.PositionD(
                    y=(image.shape[0] + 1) / 2,
                    x=(image.shape[1] + 1) / 2,
                ),
            )
        self.image = image
        self.weight = weight
        self.bmask = bmask
        self.ormask = ormask
        self.noise = noise
        self.wcs = wcs
        self.psf = psf
        self.mfrac = mfrac
        self.meta = meta
        self.store_pixels = store_pixels
        self.ignore_zero_weight = ignore_zero_weight

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
        return cls(
            image=children[0],
            weight=children[1],
            bmask=children[2],
            ormask=children[3],
            noise=children[4],
            wcs=children[5],
            psf=children[6],
            mfrac=children[7],
            meta=aux_data[0],
            store_pixels=aux_data[1],
            ignore_zero_weight=aux_data[2],
        )

    def has_bmask(self):
        return True

    def has_mfrac(self):
        return True

    def has_noise(self):
        return True

    def has_ormask(self):
        return True

    def has_psf(self):
        return jnp.any(self.psf.image != 0)

    @jax.jit
    def replace(self, **kwargs):
        """Create a new instance similar to NamedTuple._replace"""
        new_kwargs = {
            "image": self.image,
            "weight": self.weight,
            "bmask": self.bmask,
            "ormask": self.ormask,
            "noise": self.noise,
            "wcs": self.wcs,
            "psf": self.psf,
            "mfrac": self.mfrac,
            "meta": self.meta,
            "store_pixels": self.store_pixels,
            "ignore_zero_weight": self.ignore_zero_weight,
        }
        new_kwargs.update(kwargs)
        return DFMdetObservation(**new_kwargs)


def ngmix_obs_to_dfmd_obs(obs: ngmix.observation.Observation) -> DFMdetObservation:
    """Convert an ngmix Observation to a DFMdetObservation.
    Note that unlike the non-jax version, PSF is no longer an instance of
    observation and default values of bmask, ormask, mfrac are arrays of zeros.

    Parameters
    ----------
    obs: ngmix.observation.Observation
        The ngmix observation object to convert.

    Returns
    -------
    DFMdetObservation
        The converted DFMdetObservation with JAX arrays.
    """
    jacobian = obs.get_jacobian()

    psf = None
    if obs.has_psf():
        psf_obs = obs.get_psf()
        psf_jacobian = psf_obs.get_jacobian()
        psf = DFMdetPSF(
            image=psf_obs.image,
            wcs=jax_galsim.wcs.AffineTransform(
                dudx=psf_jacobian.dudcol,
                dudy=psf_jacobian.dudrow,
                dvdx=psf_jacobian.dvdcol,
                dvdy=psf_jacobian.dvdrow,
                origin=jax_galsim.PositionD(
                    y=psf_jacobian.row0 + 1,
                    x=psf_jacobian.col0 + 1,
                ),
            ),
            meta=psf_obs.meta,
            store_pixels=getattr(psf_obs, "store_pixels", True),
            ignore_zero_weight=getattr(psf_obs, "ignore_zero_weight", True),
        )

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


def dfmd_psf_to_ngmix_obs(dfmd_psf: DFMdetPSF) -> Observation:
    """Convert a DFMdetPSF to an ngmix Observation.

    Parameters
    ----------
    dfmd_psf: DFMdetPSF
        The Deep Field Metadetect PSF object to convert.

    Returns
    -------
    ngmix.observation.Observation
        The converted ngmix observation representing the PSF.
    """
    psf = Observation(
        image=np.array(dfmd_psf.image),
        jacobian=ngmix.jacobian.Jacobian(
            row=dfmd_psf.wcs.origin.y - 1,
            col=dfmd_psf.wcs.origin.x - 1,
            dudcol=dfmd_psf.wcs.dudx,
            dudrow=dfmd_psf.wcs.dudy,
            dvdcol=dfmd_psf.wcs.dvdx,
            dvdrow=dfmd_psf.wcs.dvdy,
        ),
        meta=dfmd_psf.meta,
        store_pixels=np.array(dfmd_psf.store_pixels, dtype=np.bool_),
        ignore_zero_weight=np.array(dfmd_psf.ignore_zero_weight, dtype=np.bool_),
    )
    return psf


def dfmd_obs_to_ngmix_obs(dfmd_obs: DFMdetObservation) -> Observation:
    """Convert a DFMdetObservation to an ngmix Observation.

    This function transforms a JAX-compatible DFMdetObservation object into
    a standard ngmix Observation object, converting all JAX arrays to numpy
    arrays and transforming the JAX-galsim WCS to an ngmix Jacobian.
    Note: This function never passes None values for the following:
    bmask, ormask, mfrac, instead sets default arrays of zeros.

    Parameters
    ----------
    dfmd_obs: DFMdetObservation
        The Deep Field Metadetect observation object to convert.

    Returns
    -------
    ngmix.observation.Observation
        The converted ngmix observation with numpy arrays and ngmix Jacobian.
    """
    psf = None
    if dfmd_obs.has_psf():
        psf = dfmd_psf_to_ngmix_obs(dfmd_obs.psf)

    bmask = np.array(dfmd_obs.bmask)
    ormask = np.array(dfmd_obs.ormask)
    noise = np.array(dfmd_obs.noise) if dfmd_obs.has_noise() else None
    mfrac = np.array(dfmd_obs.mfrac)

    return Observation(
        image=np.array(dfmd_obs.image),
        weight=np.array(dfmd_obs.weight),
        bmask=bmask,
        ormask=ormask,
        noise=noise,
        jacobian=ngmix.jacobian.Jacobian(
            row=dfmd_obs.wcs.origin.y - 1,
            col=dfmd_obs.wcs.origin.x - 1,
            dudcol=dfmd_obs.wcs.dudx,
            dudrow=dfmd_obs.wcs.dudy,
            dvdcol=dfmd_obs.wcs.dvdx,
            dvdrow=dfmd_obs.wcs.dvdy,
        ),
        psf=psf,
        mfrac=mfrac,
        meta=dfmd_obs.meta,
        store_pixels=np.array(dfmd_obs.store_pixels, dtype=np.bool_),
        ignore_zero_weight=np.array(dfmd_obs.ignore_zero_weight, dtype=np.bool_),
    )
