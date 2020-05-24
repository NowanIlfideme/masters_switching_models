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
    def state(self) -> xr.DataArray:
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
        prob=None,
        state=None,
    ):
        super().__init__(
            params=params, random_state=random_state, prob=prob, state=state
        )

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
    def create_params(cls, prob=None, state=None) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""

        if prob is None:
            if state is None:
                raise ValueError(
                    "At least one of `params`, `prob`"
                    " or `state` must be given."
                )
            if isinstance(state, int):
                state = range(state)
            state = np.array(state)
            # Assume `state` is array-like from now on
            prob_per = 1.0 / len(state)
            prob = xr.DataArray(
                prob_per, coords={"state": state}, dims=["state"]
            )
        elif isinstance(prob, xr.DataArray):
            if "state" not in prob.coords:
                raise ValueError(
                    f"Expected 'state' coord, found: {prob.coords}"
                )
            # Ignore `state` as we already have the "state" coord.
        else:
            # prob is array-like
            prob = np.array(prob)
            if state is None:
                state = range(len(prob))
            elif isinstance(state, int):
                state = range(state)
            state = np.array(state)
            # Set
            prob = xr.DataArray(prob, coords={"state": state}, dims=["state"])
        params = xr.Dataset({"prob": prob})
        return params

    @property
    def prob(self) -> xr.DataArray:
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
            self.state, size=len(res[time_dim]), p=self.prob
        )
        res[target_name] = xr.DataArray(_target, dims=[time_dim])

        return res

    @classmethod
    def get_random_instance(cls, state=2) -> "IndependentChainGenerator":
        """Create a random instance of ICG."""

        if isinstance(state, int):
            state = np.arange(state)
        else:
            state = np.array(state)
        rng = np.random.default_rng()
        prob = rng.uniform(0, 1, size=len(state))
        prob = prob / prob.sum()

        return cls(prob=prob, state=state)


class MarkovChainGenerator(ChainGenerator, CanRandomInstance):
    """Markov chain.
    
    Attributes
    ----------
    params : xr.Dataset
        'initial_prob', 'transition_matrix', 'state' == 'state_out'
    random_state : np.random.Generator
        Random generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        initial_prob=None,
        transition_matrix=None,
        state=None,
    ):
        super().__init__(
            params=params,
            random_state=random_state,
            initial_prob=initial_prob,
            transition_matrix=transition_matrix,
            state=state,
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
        cls, initial_prob=None, transition_matrix=None, state=None
    ) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""

        # Figure out number of state, and the state themselves
        if isinstance(state, int):
            n_states = state
            state = np.arange(state)
        elif state is None:
            # Figure out dimensionality from the others
            try:
                n_states = len(transition_matrix)
            except TypeError:
                try:
                    n_states = len(initial_prob)
                except TypeError:
                    raise ValueError("Can't determine number of state.")
            state = np.arange(n_states)
        else:
            state = np.array(state)
            n_states = len(state)

        # Figure out transition matrix and initial prob
        if transition_matrix is None:
            transition_matrix = np.full(
                shape=(n_states, n_states), fill_value=1.0 / n_states
            )
        if initial_prob is None:
            initial_prob = np.full(shape=(n_states), fill_value=1.0 / n_states)

        # Bring it together
        initial_prob = xr.DataArray(
            initial_prob, coords={"state": state}, dims=["state"]
        )
        transition_matrix = xr.DataArray(
            transition_matrix,
            coords={"state": state, "state_out": state},
            dims=["state", "state_out"],
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

        # res = super().generate(index, time_dim=time_dim)
        # _target = self.random_state.choice(
        #     self.state, size=len(res[time_dim]), p=self.prob
        # )
        # res[target_name] = xr.DataArray(_target, dims=[time_dim])

        # return res

        raise NotImplementedError("TODO: Implement.")

    @classmethod
    def get_random_instance(cls, state=2) -> "MarkovChainGenerator":
        """Create a random instance of ICG."""

        rng = np.random.default_rng()

        if isinstance(state, int):
            state = np.arange(state)
        else:
            state = np.array(state)
        N = len(state)

        p = rng.uniform(0, 1, size=N)
        p = p / np.sum(p)

        A = rng.uniform(0, 1, size=(N, N))
        A = A / np.sum(A, axis=1)[:, np.newaxis]  # sum by 'state_out'

        return cls(initial_prob=p, transition_matrix=A, state=state)


ICG = IndependentChainGenerator
MCG = MarkovChainGenerator
