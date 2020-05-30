"""Base class for data series generation."""

import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import pandas as pd
import xarray as xr

from regime_switching.utils.rng import AnyRandomState, fix_rng


class SeriesGenerator(ABC):
    """Base object for data series generators.
    
    Attributes
    ----------
    params : xr.Dataset
        Parameters for the generator.
    random_state : np.random.Generator
        Random state (can be set by seed) used by the generator.
    """

    def __init__(
        self,
        params: xr.Dataset = None,
        random_state: AnyRandomState = None,
        check_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        if isinstance(params, xr.Dataset):
            params = params.copy()
        else:
            params = self.create_params(**kwargs)
        if check_kwargs is None:
            check_kwargs = {}
        self.params = self.check_params(params, **check_kwargs)
        self.random_state = fix_rng(random_state)

    def __repr__(self) -> str:
        """Incomplete representation to hide params"""
        return f"{type(self).__name__} with params:\n" + textwrap.indent(
            repr(self.params), prefix="  "
        )

    @classmethod
    @abstractmethod
    def check_params(cls, params: xr.Dataset) -> xr.Dataset:
        """Checks and/or corrects parameters."""
        return params

    @classmethod
    @abstractmethod
    def create_params(cls, **kwargs) -> xr.Dataset:
        """Creates dataset of parameters from keyword arguments."""
        return xr.Dataset()

    @abstractmethod
    def generate(
        self, index: Union[int, pd.Index], time_dim: str = "time"
    ) -> xr.Dataset:
        """Generates a dataset.
        
        The base method generates an empty dataset with the proper index.
        """
        if not isinstance(index, pd.Index):
            if isinstance(index, int):
                index = pd.RangeIndex(index)
            else:
                index = pd.Index(index)
                # raise TypeError(f"Currently don't support {type(index)} index.")

        res = xr.Dataset(coords={time_dim: index})
        return res


class CanRandomInstance(ABC):
    """Mixin class that specifies a generator can randomize its parameters."""

    @classmethod
    @abstractmethod
    def get_random_instance(cls) -> "CanRandomInstance":
        raise NotImplementedError("This functionality isn't implemented here.")
