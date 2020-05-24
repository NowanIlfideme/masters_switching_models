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


class MarkovChainGenerator(ChainGenerator, CanRandomInstance):
    """Markov chain.
    
    Attributes
    ----------
    params : xr.Dataset
        'initial_probs', 'transition_matrix', 'state' == 'state_out'
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        initial_probs=None,
        transition_matrix=None,
        states=None,
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            initial_probs=initial_probs,
            transition_matrix=transition_matrix,
            states=states,
        )

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on 'initial_prob' and 'transition_matrix`."""

        # Coordinates
        if "state" not in params.coords or "state_out" not in params.coords:
            raise ValueError("Require coordinates 'state' and 'state_out'.")
        if not np.all(params["state"].values == params["state_out"].values):
            raise ValueError("Coords 'state' and 'state_out' are different!")

        # Initial prob
        p = params["initial_prob"]
        if (p < 0).any().item():
            raise ValueError(f"Probability cannot be negative: {p}")
        if not np.allclose(p.sum(), 1):
            warnings.warn("Probabilities don't sum to one, rescaling.")
            params["initial_prob"] = p = p / p.sum()

        # Transition matrix
        A = params["transition_matrix"]
        if (A < 0).any().item():
            raise ValueError(f"Probability cannot be negative: {A}")
        if not np.allclose(A.sum(dim="state_out"), 1):  # TODO: Check dim?
            warnings.warn("Probabilities don't sum to one, rescaling.")
            params["transition_matrix"] = A = A / A.sum(dim="state_out")

        return params

    @classmethod
    def create_params(
        cls, initial_probs=None, transition_matrix=None, states=None
    ) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""

        # Figure out number of states, and the states themselves
        if isinstance(states, int):
            n_states = states
            states = np.arange(states)
        elif states is None:
            # Figure out dimensionality from the others
            try:
                n_states = len(transition_matrix)
            except TypeError:
                try:
                    n_states = len(initial_probs)
                except TypeError:
                    raise ValueError("Can't determine number of states.")
            states = np.arange(n_states)
        else:
            states = np.array(states)
            n_states = len(states)

        # Figure out transition matrix and initial probs
        if transition_matrix is None:
            transition_matrix = np.full(
                shape=(n_states, n_states), fill_value=1.0 / n_states
            )
        if initial_probs is None:
            initial_probs = np.full(shape=(n_states), fill_value=1.0 / n_states)

        # Bring it together
        initial_probs = xr.DataArray(
            initial_probs, coords={"state": states}, dims=["state"]
        )
        transition_matrix = xr.DataArray(
            transition_matrix,
            coords={"state": states, "state_out": states},
            dims=["state", "state_out"],
        )
        params = xr.Dataset(
            {
                "initial_prob": initial_probs,
                "transition_matrix": transition_matrix,
            }
        )
        return params

    def generate(
        self,
        index: Union[int, pd.Index],
        time_dim: str = "time",
        target_name: str = "target",
    ) -> xr.Dataset:
        """Generates 1D series."""

        # res = super().generate(index, time_dim=time_dim)
        # _target = self.random_state.choice(
        #     self.states, size=len(res[time_dim]), p=self.probs
        # )
        # res[target_name] = xr.DataArray(_target, dims=[time_dim])

        # return res

        raise NotImplementedError("TODO: Implement.")

    @classmethod
    def get_random_instance(cls, states=2) -> "MarkovChainGenerator":
        """Create a random instance of ICG."""

        rng = np.random.default_rng()

        if isinstance(states, int):
            states = np.arange(states)
        else:
            states = np.array(states)
        N = len(states)

        p = rng.uniform(0, 1, size=N)
        p = p / np.sum(p)

        A = rng.uniform(0, 1, size=(N, N))
        A = A / np.sum(A, axis=1)[:, np.newaxis]  # sum by 'state_out'

        return cls(initial_probs=p, transition_matrix=A, states=states)


ICG = IndependentChainGenerator
MCG = MarkovChainGenerator
