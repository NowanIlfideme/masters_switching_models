import warnings
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from regime_switching.generate.base import CanRandomInstance, SeriesGenerator
from regime_switching.utils.rng import AnyRandomState


class ChainGenerator(SeriesGenerator):
    """Base object for chain generators."""

    # __init__ defined

    # generate() is an abstractmethod

    @property
    def states(self) -> xr.DataArray:
        return self.params["state"]


class IndependentChainGenerator(ChainGenerator, CanRandomInstance):
    """Independent sampling chain.
    
    Attributes
    ----------
    params : xr.Dataset
        'prob' and 'state'
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        probs=None,
        states=None,
    ):
        super().__init__(
            params=params, random_state=random_state, probs=probs, states=states
        )

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on 'probs' and corrects, if needed."""

        p = params["prob"]
        if (p < 0).any().item():
            raise ValueError(f"Probability cannot be negative: {p}")
        if not np.allclose(p.sum(), 1):
            warnings.warn("Probabilities don't sum to one, rescaling.")
            params["prob"] = p = p / p.sum()

        return params

    @classmethod
    def create_params(cls, probs=None, states=None) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""

        if probs is None:
            if states is None:
                raise ValueError(
                    "At least one of `params`, `probs`"
                    " or `states` must be given."
                )
            if isinstance(states, int):
                states = range(states)
            states = np.array(states)
            # Assume `states` is array-like from now on
            prob_per = 1.0 / len(states)
            probs = xr.DataArray(
                prob_per, coords={"state": states}, dims=["state"]
            )
        elif isinstance(probs, xr.DataArray):
            if "state" not in probs.coords:
                raise ValueError(
                    f"Expected 'state' coord, found: {probs.coords}"
                )
            # Ignore `states` as we already have the "state" coord.
        else:
            # probs is array-like
            probs = np.array(probs)
            if states is None:
                states = range(len(probs))
            elif isinstance(states, int):
                states = range(states)
            states = np.array(states)
            # Set
            probs = xr.DataArray(
                probs, coords={"state": states}, dims=["state"]
            )
        params = xr.Dataset({"prob": probs})
        return params

    @property
    def probs(self) -> xr.DataArray:
        return self.params["prob"]

    def generate(
        self,
        index: Union[int, pd.Index],
        time_dim: str = "time",
        target_name: str = "target",
    ) -> xr.Dataset:
        """Generates 1D series."""

        res = super().generate(index, time_dim=time_dim)
        _target = self.random_state.choice(
            self.states, size=len(res[time_dim]), p=self.probs
        )
        res[target_name] = xr.DataArray(_target, dims=[time_dim])

        return res

    @classmethod
    def get_random_instance(cls, states=2) -> "IndependentChainGenerator":
        """Create a random instance of ICG."""

        if isinstance(states, int):
            states = np.arange(states)
        else:
            states = np.array(states)
        rng = np.random.default_rng()
        probs = rng.uniform(0, 1, size=len(states))
        probs = probs / probs.sum()

        return cls(probs=probs, states=states)


ICG = IndependentChainGenerator
