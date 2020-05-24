"""Base class for data series generation."""

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import xarray as xr

from regime_switching.utils.rng import RandomState, fix_rng


class SeriesGenerator(ABC):
    """Base object for data series generators.
    
    Attributes
    ----------
    random_state : RandomState
        Random state (can be set by seed) used by the generator.
    """

    def __init__(self, random_state: Union[int, RandomState, None] = None):
        self.random_state = fix_rng(random_state)

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
