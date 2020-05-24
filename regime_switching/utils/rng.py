from typing import Union
from numpy.random import RandomState
from copy import deepcopy


def fix_rng(random_state: Union[int, RandomState, None] = None) -> RandomState:
    """Helper function for copying a random state."""

    if isinstance(random_state, RandomState):
        random_state = deepcopy(random_state)
    else:
        random_state = RandomState(random_state)
    return random_state
