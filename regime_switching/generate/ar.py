"""Generators for autoregressive processes."""

import warnings
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd  # noqa
import xarray as xr

from regime_switching.generate.base import CanRandomInstance, SeriesGenerator
from regime_switching.utils.rng import AnyRandomState


class AutoregressiveGenerator(SeriesGenerator):
    """Autoregressive process generator.
    
    Attributes
    ----------
    params : xr.Dataset
        ''
    random_state : np.random.Generator
        Random generator.
    """

    # TODO: Generalization?


def try_get_dim(x, axis=0) -> Union[int, None]:
    """"""
    try:
        x = np.array(x)
        return x.shape[axis]
    except IndexError:
        return None


def try_many_get_dim(xs: List[Tuple[Any, int]]) -> Union[int, None]:
    for x, axis in xs:
        res = try_get_dim(x, axis=axis)
        if res is not None:
            return res
    return None


def require_cov_matrix(x: np.ndarray) -> np.ndarray:
    """Checks that the matrix is a valid covariance matrix."""

    x = np.asfarray(x)
    if (x.ndim != 2) or (x.shape[0] != x.shape[1]):
        raise ValueError(f"Covariance must be 2D square, got shape: {x.shape}")
    if not np.allclose(x, x.T):
        raise ValueError("Covariance matrix should be symmetric.")
    try:
        np.linalg.cholesky(x)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix should be positive definite.")
    return x


class VARXGenerator(AutoregressiveGenerator, CanRandomInstance):
    """Vector Autoregression w/ Exogenous Vars generator.
    
    Attributes
    ----------
    params : xr.Dataset
        ''
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        endog=None,  # (ny)
        exog=None,  # (nx)
        target=None,  # == endog (ny)
        lag_endog=None,  # (ky)
        lag_exog=None,  # (kx)
        coef_ar=None,  # (ny, ny, ky)
        coef_exog=None,  # (ny, nx, kx)
        coef_covariance=None,  # (ny, ny)
        coef_const=None,  # (ny)
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            endog=endog,
            exog=exog,
            target=target,
            lag_endog=lag_endog,
            lag_exog=lag_exog,
            coef_ar=coef_ar,
            coef_exog=coef_exog,
            coef_covariance=coef_covariance,
            coef_const=coef_const,
        )

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on parameters."""

        # TODO: Densify parameters (e.g. lags from 0/1 to maxlag)
        # Covariance matrix must be
        require_cov_matrix(params["coef_covariance"].values)

        # TODO: AR coefs must be stationary

        return params

    @classmethod
    def create_params(
        cls,
        endog=None,  # (ny)
        exog=None,  # (nx)
        target=None,  # == endog (ny)
        lag_endog=None,  # (ky)
        lag_exog=None,  # (kx)
        coef_ar=None,  # (ny, ny, ky)
        coef_exog=None,  # (ny, nx, kx)
        coef_covariance=None,  # (ny, ny)
        coef_const=None,  # (ny)
    ) -> xr.Dataset:
        """Creates parameters from passed values."""

        #####
        # ny
        if endog is None:
            endog = target  # `target` is an alias of `endog`
        elif target is not None:
            warnings.warn("Passing `endog` and `target` will ignore `target`.")

        if endog is None:
            # Figure out ny
            ny = try_many_get_dim(
                [
                    (coef_const, 0),
                    (coef_covariance, 0),
                    (coef_ar, 0),
                    (coef_exog, 0),
                ]
            )
            if ny is None:
                raise ValueError("Could not calculate `ny` (endog size).")
            endog = ny

        if isinstance(endog, int):
            endog = np.arange(endog)
        else:
            endog = np.array(endog).squeeze()
            if endog.ndim < 1:
                endog = endog[np.newaxis]
            elif endog.ndim > 1:
                raise ValueError(f"Too many dims for endog: {endog.shape}")
        ny = len(endog)

        # Set the alias
        target = endog

        #####
        # nx
        if exog is None:
            # Figure out nx another way
            nx = try_get_dim(coef_exog, axis=1)
            if nx is None:
                raise ValueError("Could not calculate `nx` (exog size).")
            exog = nx

        if isinstance(exog, int):
            exog = np.arange(exog)
        else:
            exog = np.array(exog).squeeze()
            if exog.ndim < 1:
                exog = exog[np.newaxis]
            elif exog.ndim > 1:
                raise ValueError(f"Too many dims for exog: {exog.shape}")
        nx = len(exog)

        #####
        # ky, i.e. lag_endog
        if lag_endog is None:
            # Figure out ky
            ky = try_get_dim(coef_ar, axis=2)
            if ky is None:
                raise ValueError("Could not calculate `ky` (endog lags).")
            lag_endog = ky

        if isinstance(lag_endog, int):
            lag_endog = np.arange(1, lag_endog + 1)  # no lag 0 for endog
        else:
            # It's possibly given in a sparse fashion
            lag_endog = np.array(lag_endog, dtype=int).squeeze()
            if lag_endog.ndim < 1:
                lag_endog = lag_endog[np.newaxis]
            elif lag_endog.ndim > 1:
                raise ValueError(
                    f"Too many dims for lag_endog: {lag_endog.shape}"
                )
            if np.any(lag_endog) <= 0:
                raise ValueError(f"Lags must be positive, got: {lag_endog}")
        ky = len(lag_endog)

        #####
        # kx, i.e. lag_exog (including 0)
        if lag_exog is None:
            # Figure out kx
            kx = try_get_dim(coef_exog, axis=2)
            if kx is None:
                raise ValueError("Could not calculate `kx` (exog lags).")
            lag_exog = kx - 1  # HACK: We add one more later on...

        if isinstance(lag_exog, int):
            lag_exog = np.arange(0, lag_exog + 1)  # yes lag 0 for exog
        else:
            # It's possibly given in a sparse fashion
            lag_exog = np.array(lag_exog, dtype=int).squeeze()
            if lag_exog.ndim < 1:
                lag_exog = lag_exog[np.newaxis]
            elif lag_exog.ndim > 1:
                raise ValueError(
                    f"Too many dims for lag_exog: {lag_exog.shape}"
                )
            if np.any(lag_exog) < 0:
                raise ValueError(f"Lags must be nonnegative, got: {lag_exog}")
        kx = len(lag_exog)

        #####
        # Set coefficients (no checks, just shapes)

        # coef_ar (ny, ny, ky)
        _sh = (ny, ny, ky)
        coef_ar = np.asfarray(coef_ar).squeeze()
        err_ar = ValueError(
            "Could not figure out shape for `coef_ar`.\n"
            f"Squeezed is {coef_ar.shape}, expected {_sh}."
        )
        if coef_ar.ndim == 0:
            # all dims must be 1
            if (ny != 1) or (ky != 1):
                raise err_ar
            coef_ar = coef_ar[np.newaxis]
        elif coef_ar.ndim == 1:
            # we squeezed, so ny must be 1
            if (ny != 1) or (ky != coef_ar.shape[0]):
                raise err_ar
        elif coef_ar.ndim == 2:
            # we squeezed, so ky must be 1
            if (ky != 1) or (coef_ar.shape != (ny, ny)):
                raise err_ar
        elif coef_ar.ndim > 3:
            raise err_ar
        # coef_ar = np.reshape(coef_ar, _sh)
        coef_ar.shape = _sh

        # coef_exog (ny, nx, kx)
        _sh = (ny, nx, kx)
        coef_exog = np.asfarray(coef_exog).squeeze()
        err_exog = ValueError(
            "Could not figure out shape for `coef_exog`.\n"
            f"Squeezed is {coef_exog.shape}, expected {_sh}."
        )
        if coef_exog.ndim == 0:
            if (ny != 1) or (nx != 1) or (kx != 1):
                raise err_exog
            coef_exog = coef_exog[np.newaxis]
        elif coef_exog.ndim == 1:
            # Does exactly 1 shape dim equal 1?
            if np.sum(np.array(_sh) == 1) != 1:
                raise err_exog
        elif coef_exog.ndim == 2:
            # Do exactly 2 shape dims equal 1?
            if np.sum(np.array(_sh) == 1) != 2:
                raise err_exog
        elif coef_exog.ndim > 3:
            raise err_exog
        coef_exog.shape = _sh

        # coef_covariance (ny, ny)
        _sh = (ny, ny)
        coef_covariance = np.asfarray(coef_covariance).squeeze()
        err_covariance = ValueError(
            "Could not figure out shape for `coef_covariance`.\n"
            f"Squeezed is {coef_covariance.shape}, expected {_sh}."
        )
        if coef_covariance.ndim == 0:
            if ny != 1:
                raise err_covariance
            coef_covariance = coef_covariance[np.newaxis, np.newaxis]
        elif coef_covariance.ndim == 1:
            if coef_covariance.shape[0] != ny:
                raise err_covariance
            # If it works, then it's a diagonal matrix
            coef_covariance = np.diag(coef_covariance)
        elif coef_covariance.ndim > 2:
            # we squeezed, so this is too big
            raise err_covariance
        coef_covariance.shape = _sh  # = np.reshape(coef_covariance, _sh)

        # coef_const (ny)
        _sh = (ny,)
        coef_const = np.asfarray(coef_const).squeeze()
        err_const = ValueError(
            "Could not figure out shape for `coef_const`.\n"
            f"Squeezed is {coef_const.shape}, expected {_sh}."
        )
        if coef_const.ndim == 0:
            if ny > 1:
                # Warn, but still allow (e.g. broadcast const = 0)
                warnings.warn(f"Broadcasting `coef_const` to shape {_sh}.")
            coef_const = np.full(shape=_sh, fill_value=coef_const, dtype=float)
        elif coef_const.ndim > 1:
            raise err_const
        # coef_const = np.reshape(coef_const, _sh)
        coef_const.shape = _sh

        #####
        # Combine everything
        res = xr.Dataset(
            data_vars={
                "coef_ar": (["target", "endog", "lag_endog"], coef_ar),
                "coef_exog": (["target", "exog", "lag_exog"], coef_exog),
                "coef_covariance": (["target", "endog"], coef_covariance),
                "coef_const": (["target"], coef_const)
                # nothing
            },
            coords={
                "endog": endog,
                "target": target,
                "exog": exog,
                "lag_endog": lag_endog,
                "lag_exog": lag_exog,
            },
        )
        return res

    @classmethod
    def get_random_instance(cls, states=2) -> "VARXGenerator":
        """Create a random instance of VARXGenerator."""

        raise NotImplementedError("TODO: Implement.")

    def generate(
        self, index: Union[int, pd.Index], time_dim: str = "time",
    ) -> xr.Dataset:
        raise NotImplementedError("TODO: Implement.")


if __name__ == "__main__":
    vg = VARXGenerator(
        # endog="abc",  # could also be ["abc"]
        coef_const=[99.0],
        coef_covariance=[[0.1]],
        # lag_endog=[1, 2],
        coef_ar=np.array([[[0.2, 0.3]]]),
        # exog=[1],
        # lag_exog=[1],
        coef_exog=np.array([[[0.5]]]),
    )
    p = vg.params
    print(p)
