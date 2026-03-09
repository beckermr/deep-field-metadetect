from functools import partial

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
        if weight is None:
            weight = jnp.ones_like(image)
        if bmask is None:
            bmask = jnp.zeros_like(image, dtype=jnp.int32)
        if ormask is None:
            ormask = jnp.zeros_like(image, dtype=jnp.int32)
        if noise is None:
            noise = jnp.zeros_like(image)
        if mfrac is None:
            mfrac = jnp.zeros_like(image)
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
            image=psf_obs.image.astype(float),
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


@jax.tree_util.register_pytree_node_class
class DFMdetObsList:
    """JAX-compatible observation list for Deep Field Metadetect.

    Similar to ngmix.ObsList but designed for JAX compatibility.
    This class is immutable and JIT-compatible with fixed size.

    Parameters
    ----------
    obs_list : tuple or list of DFMdetObservation, optional
        The observations to store. Must be provided at construction.
    meta : dict, optional
        Metadata dictionary.
    """

    def __init__(self, obs_list=None, meta=None):
        if obs_list is None:
            self._obs_list = ()
        else:
            # Store as immutable tuple for JIT compatibility
            self._obs_list = tuple(obs_list)

        self.meta = meta if meta is not None else {}

    def set(self, index, obs):
        """Create a new DFMdetObsList with an observation replaced at index.

        This is a functional operation that returns a new instance with the same size.

        Parameters
        ----------
        index : int
            Index of the observation to replace.
        obs : DFMdetObservation
            The new observation.

        Returns
        -------
        DFMdetObsList
            New instance with the observation replaced.
        """
        if not isinstance(obs, DFMdetObservation):
            raise TypeError("Can only set DFMdetObservation objects")
        new_list = list(self._obs_list)
        new_list[index] = obs
        return DFMdetObsList(tuple(new_list), self.meta)

    def __len__(self):
        return len(self._obs_list)

    def __getitem__(self, index):
        return self._obs_list[index]

    def __setitem__(self, index, obs):
        raise NotImplementedError(
            "DFMdetObsList is immutable. Use .set(index, obs) instead."
        )

    def __iter__(self):
        return iter(self._obs_list)

    def tree_flatten(self):
        return self._obs_list, self.meta

    @classmethod
    def tree_unflatten(cls, meta, obs_list):
        return cls(obs_list, meta)


@jax.tree_util.register_pytree_node_class
class DFMdetMultiBandObsList:
    """JAX-compatible multi-band observation list for Deep Field Metadetect.

    Similar to ngmix.MultiBandObsList but designed for JAX compatibility.
    This class is immutable and JIT-compatible with fixed size.

    Parameters
    ----------
    mb_obs_list : tuple or list of DFMdetObsList, optional
        The observation lists for each band. Must be provided at construction.
    meta : dict, optional
        Metadata dictionary.
    """

    def __init__(self, mb_obs_list=None, meta=None):
        if mb_obs_list is None:
            self._mb_obs_list = ()
        else:
            # Store as immutable tuple for JIT compatibility
            self._mb_obs_list = tuple(mb_obs_list)
        self.meta = meta if meta is not None else {}

    def set(self, index, obs_list):
        """Create a new DFMdetMultiBandObsList with an obs list replaced at index.

        This is a functional operation that returns a new instance with the same size.

        Parameters
        ----------
        index : int
            Index of the band to replace.
        obs_list : DFMdetObsList
            The new observation list for this band.

        Returns
        -------
        DFMdetMultiBandObsList
            New instance with the observation list replaced.
        """
        if not isinstance(obs_list, DFMdetObsList):
            raise TypeError("Can only set DFMdetObsList objects")
        new_list = list(self._mb_obs_list)
        new_list[index] = obs_list
        return DFMdetMultiBandObsList(tuple(new_list), self.meta)

    def __len__(self):
        return len(self._mb_obs_list)

    def __getitem__(self, index):
        return self._mb_obs_list[index]

    def __setitem__(self, index, obs_list):
        raise NotImplementedError(
            "DFMdetMultiBandObsList is immutable. Use .set(index, obs_list) instead."
        )

    def __iter__(self):
        return iter(self._mb_obs_list)

    def tree_flatten(self):
        return self._mb_obs_list, self.meta

    @classmethod
    def tree_unflatten(cls, meta, mb_obs_list):
        return cls(mb_obs_list, meta)


# Constants for padding
BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    "image": 0.0,
    "weight": 0.0,
    "bmask": BMASK_EDGE,
    "noise": 0.0,
    "mfrac": 0.0,
}


@partial(jax.jit, static_argnames=["pad_width"])
def pad_observation(obs: DFMdetObservation, pad_width: int) -> DFMdetObservation:
    """Pad observation arrays with appropriate default values.

    This function pads all image arrays in the observation with a fixed width
    on all sides, using appropriate default values for regions outside the
    original image bounds. The WCS is updated to account for the padding offset.

    Parameters
    ----------
    obs : DFMdetObservation
        The observation to pad.
    pad_width : int
        Number of pixels to pad on all sides.

    Returns
    -------
    DFMdetObservation
        New observation with padded arrays and adjusted WCS.
    """
    # Pad each array with appropriate default values
    padded_image = jnp.pad(
        obs.image,
        pad_width=pad_width,
        mode="constant",
        constant_values=DEFAULT_IMAGE_VALUES["image"],
    )

    padded_weight = jnp.pad(
        obs.weight,
        pad_width=pad_width,
        mode="constant",
        constant_values=DEFAULT_IMAGE_VALUES["weight"],
    )

    padded_bmask = jnp.pad(
        obs.bmask,
        pad_width=pad_width,
        mode="constant",
        constant_values=DEFAULT_IMAGE_VALUES["bmask"],
    )

    padded_ormask = jnp.pad(
        obs.ormask, pad_width=pad_width, mode="constant", constant_values=0
    )

    padded_noise = jnp.pad(
        obs.noise,
        pad_width=pad_width,
        mode="constant",
        constant_values=DEFAULT_IMAGE_VALUES["noise"],
    )

    padded_mfrac = jnp.pad(
        obs.mfrac,
        pad_width=pad_width,
        mode="constant",
        constant_values=DEFAULT_IMAGE_VALUES["mfrac"],
    )

    # Update WCS origin to account for padding
    # The origin shifts by pad_width pixels in both x and y
    new_wcs = jax_galsim.wcs.AffineTransform(
        dudx=obs.wcs.dudx,
        dudy=obs.wcs.dudy,
        dvdx=obs.wcs.dvdx,
        dvdy=obs.wcs.dvdy,
        origin=jax_galsim.PositionD(
            x=obs.wcs.origin.x + pad_width,
            y=obs.wcs.origin.y + pad_width,
        ),
    )

    return DFMdetObservation(
        image=padded_image,
        weight=padded_weight,
        bmask=padded_bmask,
        ormask=padded_ormask,
        noise=padded_noise,
        mfrac=padded_mfrac,
        wcs=new_wcs,
        psf=obs.psf,  # PSF doesn't need padding
        meta=obs.meta,
        store_pixels=obs.store_pixels,
        ignore_zero_weight=obs.ignore_zero_weight,
    )


def jax_get_mb_obs(obs_in, pad_width=0):
    """Convert input to a DFMdetMultiBandObsList.

    JAX equivalent of ngmix.observation.get_mb_obs.

    Parameters
    ----------
    obs_in : DFMdetObservation, DFMdetObsList, or DFMdetMultiBandObsList
        Input data to convert to a DFMdetMultiBandObsList.
    pad_width : int, optional
        If > 0, pad all observations by this amount on all sides.
        This is useful for sub-observation extraction to avoid boundary issues.
        Default is 0 (no padding).

    Returns
    -------
    mbobs : DFMdetMultiBandObsList
        A DFMdetMultiBandObsList containing the input data, optionally padded.
    """
    if isinstance(obs_in, DFMdetObservation):
        obs_list = DFMdetObsList([obs_in])
        obs = DFMdetMultiBandObsList([obs_list])
    elif isinstance(obs_in, DFMdetObsList):
        obs = DFMdetMultiBandObsList([obs_in])
    elif isinstance(obs_in, DFMdetMultiBandObsList):
        obs = obs_in
    else:
        raise ValueError(
            "obs should be DFMdetObservation, DFMdetObsList, or DFMdetMultiBandObsList"
        )

    # Pad all observations if requested
    if pad_width > 0:
        obs = DFMdetMultiBandObsList(
            [
                DFMdetObsList([pad_observation(o, pad_width) for o in obslist])
                for obslist in obs
            ]
        )

    return obs
