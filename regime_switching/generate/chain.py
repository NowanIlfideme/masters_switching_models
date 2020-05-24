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
        return self.params["states"]


class IndependentChainGenerator(ChainGenerator, CanRandomInstance):
    """Independent sampling chain.
    
    Attributes
    ----------
    params : xr.Dataset
        'prob' and 'states'
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        prob=None,
        states=None,
    ):
        super().__init__(
            params=params, random_state=random_state, prob=prob, states=states
        )

    @property
    def prob(self) -> xr.DataArray:
        """States probability vector."""
        return self.params["prob"].copy()

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on 'prob' and corrects, if needed."""

        p = params["prob"]
        if (p < 0).any().item():
            raise ValueError(f"Probability cannot be negative: {p}")
        if not np.allclose(p.sum(), 1):
            warnings.warn("Probabilities don't sum to one, rescaling.")
            params["prob"] = p = p / p.sum()

        return params

    @classmethod
    def create_params(cls, prob=None, states=None) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""

        if prob is None:
            if states is None:
                raise ValueError(
                    "At least one of `params`, `prob`"
                    " or `states` must be given."
                )
            if isinstance(states, int):
                states = range(states)
            states = np.array(states)
            # Assume `states` is array-like from now on
            prob_per = 1.0 / len(states)
            prob = xr.DataArray(
                prob_per, coords={"states": states}, dims=["states"]
            )
        elif isinstance(prob, xr.DataArray):
            if "states" not in prob.coords:
                raise ValueError(
                    f"Expected 'states' coord, found: {prob.coords}"
                )
            # Ignore `states` as we already have the "states" coord.
        else:
            # prob is array-like
            prob = np.array(prob)
            if states is None:
                states = range(len(prob))
            elif isinstance(states, int):
                states = range(states)
            states = np.array(states)
            # Set
            prob = xr.DataArray(
                prob, coords={"states": states}, dims=["states"]
            )
        params = xr.Dataset({"prob": prob})
        return params

    def generate(
        self,
        index: Union[int, pd.Index],
        time_dim: str = "time",
        target_name: str = "target",
    ) -> xr.Dataset:
        """Generates 1D series."""

        res = super().generate(index, time_dim=time_dim)
        _target = self.random_state.choice(
            self.states, size=len(res[time_dim]), p=self.prob
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
        prob = rng.uniform(0, 1, size=len(states))
        prob = prob / prob.sum()

        return cls(prob=prob, states=states)


class MarkovChainGenerator(ChainGenerator, CanRandomInstance):
    """Markov chain.
    
    Attributes
    ----------
    params : xr.Dataset
        'initial_prob', 'transition_matrix', 'states' == 'states_out'
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        initial_prob=None,
        transition_matrix=None,
        states=None,
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            initial_prob=initial_prob,
            transition_matrix=transition_matrix,
            states=states,
        )

    @property
    def initial_prob(self) -> xr.DataArray:
        """Initial state probability vector."""
        return self.params["initial_prob"].copy()

    @property
    def transition_matrix(self) -> xr.DataArray:
        """Markov state transition matrix."""
        return self.params["transition_matrix"].copy()

    @property
    def stationary_distribution(self) -> xr.DataArray:
        """Gets stationary distribution of the Markov chain.
        
        Based on this SO answer: https://stackoverflow.com/a/58334399
        """
        A = self.transition_matrix.values
        evals, evecs = np.linalg.eig(A.T)
        evec1 = evecs[:, np.isclose(evals, 1)][:, 0]
        stationary = (evec1 / evec1.sum()).real
        res = xr.DataArray(
            stationary, coords={"states": self.states}, dims=["states"]
        )
        return res

    @classmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks assumptions on 'initial_prob' and 'transition_matrix`."""

        # Coordinates
        if "states" not in params.coords or "states_out" not in params.coords:
            raise ValueError("Require coordinates 'states' and 'states_out'.")
        if not np.all(params["states"].values == params["states_out"].values):
            raise ValueError("Coords 'states' and 'states_out' are different!")

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
        if not np.allclose(A.sum(dim="states_out"), 1):  # TODO: Check dim?
            warnings.warn("Probabilities don't sum to one, rescaling.")
            params["transition_matrix"] = A = A / A.sum(dim="states_out")

        return params

    @classmethod
    def create_params(
        cls, initial_prob=None, transition_matrix=None, states=None
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
                    n_states = len(initial_prob)
                except TypeError:
                    raise ValueError("Can't determine number of states.")
            states = np.arange(n_states)
        else:
            states = np.array(states)
            n_states = len(states)

        # Figure out transition matrix and initial prob
        if transition_matrix is None:
            transition_matrix = np.full(
                shape=(n_states, n_states), fill_value=1.0 / n_states
            )
        if initial_prob is None:
            initial_prob = np.full(shape=(n_states), fill_value=1.0 / n_states)

        # Bring it together
        initial_prob = xr.DataArray(
            initial_prob, coords={"states": states}, dims=["states"]
        )
        transition_matrix = xr.DataArray(
            transition_matrix,
            coords={"states": states, "states_out": states},
            dims=["states", "states_out"],
        )
        params = xr.Dataset(
            {
                "initial_prob": initial_prob,
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

        res = super().generate(index, time_dim=time_dim)
        T = len(res[time_dim])

        # TODO: Possible to vectorize instead of loop?

        rng = self.random_state

        _target = np.empty(shape=T, dtype=self.states.dtype)
        curr = _target[0] = rng.choice(self.states, p=self.initial_prob)
        for i in range(1, T):
            tp = self.transition_matrix.sel(states=curr)
            curr = _target[i] = rng.choice(self.states, p=tp)

        res[target_name] = xr.DataArray(_target, dims=[time_dim])

        return res

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
        A = A / np.sum(A, axis=1)[:, np.newaxis]  # sum by 'states_out'

        return cls(initial_prob=p, transition_matrix=A, states=states)


ICG = IndependentChainGenerator
MCG = MarkovChainGenerator
