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
        Dataset with coords:
            'endog' (ny)
            'target' (ny) equal to 'endog'
            'exog' (nx)
            'lag_endog' (ky)
            'lag_exog' (kx)
        and variables:
            'coef_ar' (ny, ny, ky)
            'coef_exog' (ny, nx, kx)
            'coef_covariance' (ny, ny)
            'coef_const' (ny)
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        endog=None,  # (ny)
        exog=[],  # (nx)
        lag_endog=None,  # (ky)
        lag_exog=[],  # (kx)
        coef_ar=None,  # (ny, ny, ky)
        coef_exog=[],  # (ny, nx, kx)
        coef_covariance=None,  # (ny, ny)
        coef_const=None,  # (ny)
        target=None,  # == endog (ny)
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
    def check_params(
        cls, params: xr.Dataset, action_nonstationary: str = "error"
    ) -> xr.Dataset:
        """Checks assumptions on parameters and densifies coefficients.
        
        This function:
            - checks the lag structure (endog and exog lags)
            - densifies the ar/exog params to be from 0/1 to max lag
            - checks the covariance matrix (symmetric, positive definite)
            - checks stationarity

        Parameters
        ----------
        params : xr.Dataset
            Input parameters in xarray form.
        action_nonstationary : str
            One of {"error", "ignore", "always", "default", "module", or "once"}
            Default is "error".
        """

        # Check lag structure
        lag_endog = params["lag_endog"].values
        if np.any(lag_endog) <= 0:
            raise ValueError(f"Lags must be positive, got: {lag_endog}")

        lag_exog = params["lag_exog"].values
        if np.any(lag_exog) < 0:
            raise ValueError(f"Lags must be nonnegative, got: {lag_exog}")

        # Densify parameters (e.g. lags from 0/1 to maxlag)
        max_lag_endog = lag_endog.max()
        max_lag_exog = lag_exog.max() if len(lag_exog > 0) else -1
        # -1 means range will be [0, 0) i.e. empty

        params = (
            params.reindex(lag_endog=pd.RangeIndex(1, max_lag_endog + 1))
            .reindex(lag_exog=pd.RangeIndex(0, max_lag_exog + 1))
            .fillna({"coef_ar": 0, "coef_exog": 0})
        )

        # Covariance matrix must be symmetric, positive definite
        require_cov_matrix(params["coef_covariance"].values)

        # AR coefs must be stationary

        def check_stationary(coef_ar: xr.DataArray) -> Tuple[bool, np.ndarray]:
            """Takes densified AR coeffs and checks stationarity.
            
            Not sure where this code is from - probably based on `statsmodels`.
            """

            ny = len(coef_ar["endog"])
            if np.any(coef_ar["target"].values != coef_ar["endog"].values):
                raise ValueError("'target' and 'endog' must be the same!")
            ky = len(coef_ar["lag_endog"])

            c_stack = coef_ar.stack(z=["lag_endog", "endog"]).transpose(
                "z", "target"
            )
            A = np.c_[c_stack.values, np.eye(N=ny * ky, M=ny * (ky - 1))]
            eigv = np.linalg.eigvals(A)
            is_stationary = np.all(np.abs(eigv) < 1)
            return is_stationary, eigv

        is_stationary, eigv = check_stationary(params["coef_ar"])
        with warnings.catch_warnings():
            warnings.simplefilter(action_nonstationary, category=UserWarning)
            if not is_stationary:
                warnings.warn(
                    "Autoregressive coefficients are nonstationary."
                    f" Norms of eigenvalues should be <1: {list(eigv)}."
                )

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
            if exog.size == 0:
                pass
            elif exog.ndim < 1:
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
            warnings.warn(
                f"Assuming {ky} lags for `lag_endog`. "
                "Please specify in the future."
            )
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
        ky = len(lag_endog)

        #####
        # kx, i.e. lag_exog (including 0)
        if lag_exog is None:
            # Figure out kx
            kx = try_get_dim(coef_exog, axis=2)
            if kx is None:
                raise ValueError("Could not calculate `kx` (exog lags).")
            lag_exog = kx - 1  # HACK: We add one more later on...
            warnings.warn(
                f"Assuming {kx} lags for `lag_exog` (including 0th lag). "
                "Please specify in the future."
            )

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
        if coef_exog.size == 0:
            # sometimes size will be 0
            if (nx != 0) and (kx != 0):
                raise err_exog
        elif coef_exog.ndim == 0:
            if (ny != 1) or (nx != 1) or (kx != 1):
                raise err_exog
            coef_exog = coef_exog[np.newaxis]
        elif coef_exog.ndim == 1:
            # Does exactly 1 shape dim not equal 1?
            if np.sum(np.array(_sh) != 1) != 1:
                raise err_exog
        elif coef_exog.ndim == 2:
            # Do exactly 2 shape dims equal 1?
            if np.sum(np.array(_sh) != 1) != 2:
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
    def get_random_instance(
        cls,
        endog: Union[int, List] = 1,
        exog: Union[int, List] = 0,
        lag_endog: Union[int, List] = 2,
        lag_exog: Union[int, List] = tuple(),
        max_tries: int = 100,
    ) -> "VARXGenerator":
        """Create a random instance of VARXGenerator."""

        rng = np.random.default_rng()

        if isinstance(endog, int):
            endog = np.arange(endog)
        else:
            endog = np.array(endog)

        if isinstance(exog, int):
            exog = np.arange(exog)
        else:
            exog = np.array(exog)

        if isinstance(lag_endog, int):
            lag_endog = np.arange(1, lag_endog + 1, dtype=int)
        else:
            lag_endog = np.array(lag_endog, dtype=int)

        if isinstance(lag_exog, int):
            lag_exog = np.arange(0, lag_exog + 1, dtype=int)
        else:
            lag_exog = np.array(lag_exog, dtype=int)

        # Try generating instances; can fail due to contraints
        n_errors = 0
        res = None
        while (res is None) and (n_errors < max_tries):
            try:
                ny = len(endog)
                nx = len(exog)
                ky = len(lag_endog)
                kx = len(lag_exog)

                # NOTE: We add exponential decay to autoregressive weights
                # (including exog lags) to more likely have a reasonable,
                # stationary process

                # AR coeffs: (ny, ny, ky)
                lo_ar = -0.5
                hi_ar = 1
                decay_factor_ar = 0.1
                coef_ar = (
                    rng.uniform(lo_ar, hi_ar, size=(ny, ny, ky))
                    * np.exp(-decay_factor_ar * np.arange(ky))[
                        np.newaxis, np.newaxis, :
                    ]
                )

                # Exog coeffs: (ny, nx, kx)
                lo_exog = -1
                hi_exog = 1
                decay_factor_exog = 0.3
                coef_exog = (
                    rng.uniform(lo_exog, hi_exog, size=(ny, nx, kx))
                    * np.exp(-decay_factor_exog * np.arange(kx))[
                        np.newaxis, np.newaxis, :
                    ]
                )

                # Covariance: (ny, ny)
                rank_L = ny
                L = rng.uniform(-0.5, 1, size=(ny, rank_L))
                coef_covariance = L @ L.T  # (ny, ny)

                # Constant
                lo_const = -1
                hi_const = 2
                coef_const = rng.uniform(lo_const, hi_const, size=(ny,))

                # Create class
                res = cls(
                    endog=endog,
                    exog=exog,
                    lag_endog=lag_endog,
                    lag_exog=lag_exog,
                    coef_ar=coef_ar,
                    coef_exog=coef_exog,
                    coef_covariance=coef_covariance,
                    coef_const=coef_const,
                )
            except Exception:
                # TODO: Specify exact exceptions.
                n_errors += 1
        if res is None:
            raise ValueError(
                "Could not generate a random instance"
                f" within {max_tries} tries."
            )

        return res

    def generate(
        self,
        index: Union[int, pd.Index],
        exog=None,
        time_dim: str = "time",
        target_name: str = "output",
    ) -> xr.Dataset:
        """Generates an ND series."""

        res = super().generate(index, time_dim=time_dim)

        T = len(res[time_dim])
        ny = self.n_endog
        nx = self.n_exog
        # ky = len(self.lag_endog)
        # kx = len(self.lag_exog)

        # Result will go here
        res[target_name] = xr.DataArray(
            np.zeros(shape=(T, ny), dtype=float),
            coords={time_dim: res[time_dim], "target": self.target},
            dims=[time_dim, "target"],
        )

        # Add constant
        res[target_name] += self.coef_const  # broadcasts

        # Check exog, turn into DataArray, find impact
        if self.is_pure_ar:
            if exog is not None:
                warnings.warn("Ignoring the passed `exog`, this is pure VAR.")
            exog = None
        else:
            if isinstance(exog, xr.DataArray):
                # Check sizes and dims
                if "exog" not in exog.coords:
                    raise ValueError("Rename the variable dimension 'exog'.")
                if time_dim not in exog.coords:
                    raise ValueError(f"Expected '{time_dim}' in coords.")
                if not np.all((exog[time_dim]) == res[time_dim]):
                    raise ValueError("`exog` time dim differs from index.")
                if not np.all(exog["exog"] == self.exog):
                    raise ValueError("`exog` variables differ from params.")
            else:
                # TODO: Rearrange dimensions if required...
                exog = np.asfarray(exog).squeeze()
                err_bad_exog = ValueError(
                    "Bad shape for `exog`."
                    f" Expected {(T, nx)}, got {exog.shape}."
                )
                if exog.ndim == 0:
                    if (nx == 1) and (T == 1):
                        exog = exog[np.newaxis, np.newaxis]
                    else:
                        raise ValueError("Broadcasting `exog` not supported.")
                elif exog.ndim == 1:
                    if nx == 1:
                        if T == 1:
                            raise err_bad_exog
                        else:
                            exog = exog[:, np.newaxis]
                    elif T == 1:
                        exog = exog[np.newaxis, :]
                    else:
                        raise err_bad_exog
                elif exog.ndim == 2:
                    if exog.shape == (T, nx):
                        pass
                    elif exog.shape == (nx, T):
                        exog = exog.T
                    else:
                        raise err_bad_exog
                else:
                    raise err_bad_exog

                exog = xr.DataArray(
                    exog,
                    coords={time_dim: res[time_dim], "exog": self.exog},
                    dims=[time_dim, "exog"],
                )

            # Start with empty slate
            exog_effect = xr.DataArray(
                np.zeros(shape=(T, ny), dtype=float),
                coords={time_dim: res[time_dim], "target": self.target},
                dims=[time_dim, "target"],
            )
            # Create a copy to work with
            e = exog.copy()
            e[time_dim] = pd.RangeIndex(T)
            for i in range(T):
                # Align the AR coeffs with time
                q = self.coef_exog.copy()
                q["lag_exog"] = i - self.lag_exog
                q = q.rename({"lag_exog": time_dim})
                # Calculate the effect and align with time
                effect_i = (
                    (e * q)
                    .sum(dim=["time", "exog"])
                    .expand_dims({"time": [i]})
                    .reindex(time=e["time"])
                    .fillna(0)
                )
                exog_effect = exog_effect + effect_i
            # Add values
            res[target_name] += exog_effect

        # Create innovation (error) process
        innovations = self.random_state.multivariate_normal(
            mean=np.zeros(ny), cov=self.coef_covariance.values, size=T,
        )

        res[target_name] += innovations

        # Add rolling AR process

        # Get an easier-to-work-with DataArray
        b = res[target_name].rename({"target": "endog"})
        b[time_dim] = pd.RangeIndex(T)
        for i in range(T):
            # Align the AR coeffs with time
            q = self.coef_ar.copy()
            q["lag_endog"] = i - self.lag_endog
            q = q.rename({"lag_endog": time_dim})
            # Calculate the effect and align with time
            effect_i = (
                ((b * q).sum(dim=["time", "endog"]).rename({"target": "endog"}))
                .expand_dims({"time": [i]})
                .reindex(time=b["time"])
                .fillna(0)
            )
            b = b + effect_i
        # Rename back
        b = b.rename({"endog": "target"})
        b[time_dim] = res[time_dim]
        # Set back values
        res[target_name] = b

        return res

    @property
    def n_endog(self) -> int:
        return len(self.params["endog"])

    @property
    def n_exog(self) -> int:
        return len(self.params["exog"])

    @property
    def is_pure_ar(self) -> bool:
        """Pure AR if there are no exogenous variables."""
        return self.n_exog == 0

    @property
    def endog(self) -> xr.DataArray:
        return self.params["endog"].copy()

    @property
    def exog(self) -> xr.DataArray:
        return self.params["exog"].copy()

    @property
    def target(self) -> xr.DataArray:
        return self.params["target"].copy()

    @property
    def lag_endog(self) -> xr.DataArray:
        return self.params["lag_endog"].copy()

    @property
    def lag_exog(self) -> xr.DataArray:
        return self.params["lag_exog"].copy()

    @property
    def coef_ar(self) -> xr.DataArray:
        return self.params["coef_ar"].copy()

    @property
    def coef_exog(self) -> xr.DataArray:
        return self.params["coef_exog"].copy()

    @property
    def coef_covariance(self) -> xr.DataArray:
        return self.params["coef_covariance"].copy()

    @property
    def coef_const(self) -> xr.DataArray:
        return self.params["coef_const"].copy()


if __name__ == "__main__":
    # vg = VARXGenerator(
    #     endog=["abc", "def"],  # could also be "abc"
    #     coef_const=[99.0, 98],
    #     coef_covariance=[[0.1, 0.1], [0.1, 0.3]],
    #     lag_endog=[2],
    #     coef_ar=np.array([[[0.2, 0.3], [0.21, -0.31]]]),
    #     exog=[],
    #     lag_exog=[],
    #     coef_exog=[],  # np.array([[[0.5]]]),
    # )

    # vg = VARXGenerator(
    #     endog=["a", "b"],  # could also be "abc"
    #     coef_const=3,
    #     coef_covariance=[[0.1, 0.1], [0.1, 0.3]],
    #     lag_endog=[2],
    #     coef_ar=np.array([[[0.2, 0.3], [0.21, -0.31]]]),
    # )

    # Get a random instance
    vg = VARXGenerator.get_random_instance(
        endog=["a", "b", "c"],
        lag_endog=[1, 12],
        exog=["x"],
        lag_exog=[0, 2],  # [0, 1, 2],
        max_tries=100,
    )

    p = vg.params
    print(p)

    # generate series
    T = 50
    time_dim = "time"
    target_name = "output"

    r1 = np.random.default_rng()
    fake_exog = r1.normal(0, 1, size=T)  # also could be (T, 1)

    res = vg.generate(
        T, time_dim=time_dim, target_name=target_name, exog=fake_exog,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        2, 1, figsize=(20, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    res[target_name].plot.line(hue="target", ax=ax[0])
    ax[1].set_ylabel("exog")
    ax[1].plot(fake_exog, color="black")
    plt.show()
