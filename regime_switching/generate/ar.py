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
        self, params: xr.Dataset = None, random_state: AnyRandomState = None,
    ):
        super().__init__(
            params=params, random_state=random_state,
        )

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on parameters."""

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

        # coef_ar = xr.DataArray(
        #     coef_ar,
        #     # coords={'tar'},
        #     dims=['target', 'lag_endog', 'endog']
        # )

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


if __name__ == "__main__":
    p = VARXGenerator.create_params(
        # endog="abc",  # could also be ["abc"]
        coef_const=[99.0],
        coef_covariance=[[0.1]],
        # lag_endog=[1, 2],
        coef_ar=np.array([[[0.2, 0.3]]]),
        # exog=[1],
        # lag_exog=[1],
        coef_exog=np.array([[[0.5]]]),
    )
    print(p)
