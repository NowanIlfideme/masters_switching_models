"""Time series data generators (e.g. for regime-switching models)."""

import numbers 
import numpy as np 
import pandas as pd 
from copy import deepcopy

from numpy.random import RandomState 


class SeriesGenerator(object):
    """Base series generator object."""
    
    def __init__(self, random_state=None):
        # Set internal RNG 
        self.random_state = self._fix_rng(random_state)

    def generate(self, index, random_state=None): 
        """Override this: create a Series or DataFrame with `index`."""
        index = self._fix_index(index) 
        rng = self._fix_rng(random_state, self.random_state)
        index, rng
        raise NotImplementedError("Method not overwritten.")

    def __call__(self, index, random_state=None):
        """Alias for `generate`."""
        return self.generate(index, random_state=random_state)

    def copy(self):
        """Returns deep copy of self."""
        return deepcopy(self) 

    # Helper methods

    @staticmethod
    def _fix_index(index): 
        """Returns a pd.Index; if index is an integer, makes a RangeIndex."""
        if np.isscalar(index) and isinstance(index, numbers.Integral):
            return pd.RangeIndex(index) 
        return pd.Index(index)

    @staticmethod
    def _fix_rng(random_state, alternate=None): 
        """Helper function for copying a random state."""
        if random_state is None:
            random_state = alternate
        
        if isinstance(random_state, RandomState):
            random_state = deepcopy(random_state) 
        else:
            random_state = RandomState(random_state) 
        return random_state
        
